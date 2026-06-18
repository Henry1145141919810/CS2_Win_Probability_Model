"""Training & evaluation harness for the round win-probability models.

Implements the project's evaluation protocol:
  - 5-fold GroupKFold by match_id (never split within a match -> no leakage)
  - feature sets A..E (A=economy baseline; B=+map control; C/D/E need firepower/tactical)
  - models: logistic regression, random forest, XGBoost
  - metrics: AUC-ROC (discrimination) + log-loss & Brier (calibration)
  - time-window AUC (the "control signal emergence point" headline)
  - DeLong test comparing each feature set vs Model A (same model), out-of-fold

Usage:
    python src/models/train_pipeline.py
    python src/models/train_pipeline.py --models logreg,xgb --sets A,B
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import (MAPCONTROL_COLS, MAPCONTROL_LOS_COLS,  # noqa: E402
                                 TERRITORY_COLS, TERRITORY_ZONE_COLS)
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from features.assemble import INTERACTION_COLS  # noqa: E402

TACTICAL = TACTICAL_COLS + BOMB_COLS
IMPROVED = TERRITORY_COLS + TERRITORY_ZONE_COLS + INTERACTION_COLS  # better spatial encoding
DATA = ROOT / "data" / "training_dataset.parquet"
# Three control models kept side-by-side for the comparison narrative:
#   Voronoi (proximity) | grey (instantaneous LOS/FOV/smoke) | territory (grey + memory/decay)
FEATURE_SETS = {
    "A": ECONOMY_COLS,                                        # economy only (baseline)
    "B": ECONOMY_COLS + MAPCONTROL_COLS,                      # + Voronoi
    "G": ECONOMY_COLS + MAPCONTROL_LOS_COLS,                  # + grey (instantaneous)
    "Terr": ECONOMY_COLS + TERRITORY_COLS,                    # + territory (memory+decay)
    "E": ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL,           # Voronoi + tactical (best so far)
    "DG": ECONOMY_COLS + TACTICAL + MAPCONTROL_LOS_COLS,      # tactical + grey (no Voronoi)
    "DT": ECONOMY_COLS + TACTICAL + TERRITORY_COLS,           # tactical + territory (no Voronoi)
    "ET": ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL + TERRITORY_COLS,  # E + territory
    "ET+": ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL + IMPROVED,  # + per-zone terr + interactions
}
WINDOWS = [5, 10, 15, 20, 25]


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)
    if name == "xgb":
        from xgboost import XGBClassifier
        # tuned via tune_xgb.py (GroupKFold): shallow+regularized trees fix the depth-6
        # overfitting (+0.007 AUC). depth 3 / min_child 10 / lambda 10.
        return XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.03,
                             min_child_weight=10, reg_lambda=10.0,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             tree_method="hist", eval_metric="logloss", random_state=0)
    raise ValueError(name)


def ece(y, p, bins: int = 10) -> float:
    """Expected Calibration Error: avg |predicted% - actual win-rate| over prob bins.
    The single-number 'are the probabilities honest?' score (lower = better; <0.02 good)."""
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def oof_predict(df: pl.DataFrame, cols: list[str], model_name: str):
    """Return out-of-fold predicted probabilities aligned to df rows."""
    X = np.nan_to_num(df[cols].to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_model(model_name).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y


# ---- DeLong's test for two correlated ROC AUCs (fast implementation) ----
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_pvalue(y_true, p1, p2):
    order = np.argsort(-y_true.astype(float), kind="mergesort")
    yt = y_true[order]; m = int(yt.sum()); n = len(yt) - m
    preds = np.vstack([p1[order], p2[order]])
    pos = preds[:, :m]; neg = preds[:, m:]
    k = 2
    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(preds[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    L = np.array([1, -1])
    var = L @ cov @ L
    if var <= 0:
        return aucs[0], aucs[1], np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], p


def block_bootstrap(df, oof_a, oof_b, B=500, seed=42):
    """Match-level block bootstrap CIs from fixed out-of-fold predictions.

    Resamples whole matches with replacement (the independent unit; rounds within a
    match are correlated) and recomputes AUC for the baseline (A) and a feature set,
    plus their difference. Returns percentile 95% CIs. This bootstraps the metric on
    fixed OOF preds (cheap); the heavier full-retrain bootstrap is the cloud variant.
    """
    rng = np.random.default_rng(seed)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    uniq = np.unique(groups)
    idx_by_match = {m: np.where(groups == m)[0] for m in uniq}
    a_s, b_s, d_s = [], [], []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_match[m] for m in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        aa = roc_auc_score(yy, oof_a[idx]); bb = roc_auc_score(yy, oof_b[idx])
        a_s.append(aa); b_s.append(bb); d_s.append(bb - aa)
    q = lambda s: (np.percentile(s, 2.5), np.percentile(s, 97.5))
    return {"auc_a": (np.mean(a_s), *q(a_s)), "auc_b": (np.mean(b_s), *q(b_s)),
            "diff": (np.mean(d_s), *q(d_s))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb")
    ap.add_argument("--sets", default="A,B")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="match-level block-bootstrap B iterations for AUC + (set-A) diff CIs")
    args = ap.parse_args()
    models = args.models.split(",")
    sets = args.sets.split(",")

    df = pl.read_parquet(DATA)
    print(f"data: {df.height} snapshots, {df['match_id'].n_unique()} matches, "
          f"ct_won={df['ct_won'].mean():.3f}\n")

    # --- overall metrics + DeLong vs set A ---
    oof_store = {}
    print(f"{'model':8s} {'set':>3} {'AUC':>7} {'logloss':>8} {'brier':>7} {'ECE':>7}")
    for mdl in models:
        for s in sets:
            oof, y = oof_predict(df, FEATURE_SETS[s], mdl)
            oof_store[(mdl, s)] = oof
            print(f"{mdl:8s} {s:>3} {roc_auc_score(y, oof):>7.4f} "
                  f"{log_loss(y, oof):>8.4f} {brier_score_loss(y, oof):>7.4f} "
                  f"{ece(y, oof):>7.4f}")
    print()
    y = df["ct_won"].to_numpy()
    for mdl in models:
        if (mdl, "A") in oof_store:
            for s in sets:
                if s == "A":
                    continue
                a, b, p = delong_pvalue(y, oof_store[(mdl, "A")], oof_store[(mdl, s)])
                print(f"DeLong {mdl}: set {s} (AUC {b:.4f}) vs A (AUC {a:.4f})  "
                      f"delta={b-a:+.4f}  p={p:.4g}")

    # --- block-bootstrap CIs (plan: B=500 for AUC + E-vs-A difference) ---
    if args.bootstrap:
        print(f"\nMatch-level block bootstrap (B={args.bootstrap}):")
        for mdl in models:
            for s in sets:
                if s == "A" or (mdl, "A") not in oof_store or (mdl, s) not in oof_store:
                    continue
                bs = block_bootstrap(df, oof_store[(mdl, "A")], oof_store[(mdl, s)],
                                     B=args.bootstrap)
                ma, la, ua = bs["auc_a"]; mb, lb, ub = bs["auc_b"]; md, ld, ud = bs["diff"]
                print(f"  {mdl} set {s}: AUC {mb:.4f} (95% CI {lb:.4f}-{ub:.4f}) | "
                      f"A {ma:.4f} ({la:.4f}-{ua:.4f}) | "
                      f"diff {md:+.4f} (95% CI {ld:+.4f} to {ud:+.4f})"
                      f"{'  [excludes 0]' if ld > 0 or ud < 0 else ''}")

    # --- time-window AUC (headline) ---
    print(f"\nTime-window AUC (the control-signal-emergence analysis):")
    print(f"{'t(s)':>4} " + " ".join(f"{m}_{s:>6}" for m in models for s in sets))
    for w in WINDOWS:
        sub = df.filter((pl.col("time_elapsed_sec") >= w - 0.5) &
                        (pl.col("time_elapsed_sec") < w + 0.5))
        cells = []
        for mdl in models:
            for s in sets:
                if len(np.unique(sub["ct_won"].to_numpy())) < 2:
                    cells.append("   nan"); continue
                oof, yy = oof_predict(sub, FEATURE_SETS[s], mdl)
                cells.append(f"{roc_auc_score(yy, oof):6.4f}")
        print(f"{w:>4} " + " ".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
