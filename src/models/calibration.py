"""Calibration ('are the probabilities honest?') across models, WITH confidence intervals.

AUC only checks ranking. This checks whether "the model said 70%" -> CT won ~70%.
  - ECE table (Brier, log-loss, ECE) for the key (model, feature-set) pairs.
  - Reliability diagram for the two headline models with PER-BIN bootstrap error bars
    + a bootstrap 95% CI on ECE -> tells which deviations from the diagonal are REAL
    miscalibration vs sampling noise (the CI question).
Match-level bootstrap (resample the 220 matches) so the CIs respect within-match correlation.

Usage: python src/models/calibration.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS, BOMB_LIVE_COLS, BOMB_DEFUSE_COLS  # noqa: E402
from features.firepower import FIREPOWER_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures" / "calibration.png"
TAC = TACTICAL_COLS + BOMB_COLS
_EF = ECONOMY_COLS + MAPCONTROL_COLS + TAC + FIREPOWER_COLS                 # all 4 pillars
_EFB2 = (ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS              # everything
         + FIREPOWER_COLS + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS)
SETS = {  # label -> (columns, model-kind)
    "logreg A": (ECONOMY_COLS, "logreg"),
    "logreg E": (ECONOMY_COLS + MAPCONTROL_COLS + TAC, "logreg"),
    "logreg EF": (_EF, "logreg"),
    "logreg EFB2": (_EFB2, "logreg"),
    "xgb A": (ECONOMY_COLS, "xgb"),
    "xgb E": (ECONOMY_COLS + MAPCONTROL_COLS + TAC, "xgb"),
    "xgb EF": (_EF, "xgb"),
    "xgb EFB2": (_EFB2, "xgb"),
}
HEADLINE = ["logreg EFB2", "xgb EFB2"]
BINS = 10
B = 200


def make_model(kind):
    if kind == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.03,
                         min_child_weight=10, reg_lambda=10.0, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss", random_state=0)


def oof(df, cols, y, groups, kind):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def bin_stats(y, p):
    edges = np.linspace(0, 1, BINS + 1)
    conf, acc, e = [], [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            conf.append(np.nan); acc.append(np.nan); continue
        conf.append(p[m].mean()); acc.append(y[m].mean())
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return np.array(conf), np.array(acc), e


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    midx = {m: np.where(groups == m)[0] for m in np.unique(groups)}
    matches = list(midx)
    rng = np.random.default_rng(42)

    print(f"{'model/set':12s} {'Brier':>7} {'logloss':>8} {'ECE':>7} {'ECE 95% CI':>18}")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect (honest)")
    for label, (cols, kind) in SETS.items():
        p = oof(df, cols, y, groups, kind)
        conf, acc, e = bin_stats(y, p)
        # match-level bootstrap -> ECE CI (+ per-bin acc CI for headline)
        boot_ece, boot_acc = [], []
        for _ in range(B):
            idx = np.concatenate([midx[matches[i]] for i in
                                  rng.integers(0, len(matches), len(matches))])
            c, a, ee = bin_stats(y[idx], p[idx])
            boot_ece.append(ee); boot_acc.append(a)
        lo, hi = np.percentile(boot_ece, [2.5, 97.5])
        print(f"{label:12s} {brier_score_loss(y,p):>7.4f} {log_loss(y,p):>8.4f} "
              f"{e:>7.4f}  ({lo:.4f},{hi:.4f})")
        if label in HEADLINE:
            ba = np.array(boot_acc)
            alo = np.nanpercentile(ba, 2.5, axis=0)
            ahi = np.nanpercentile(ba, 97.5, axis=0)
            ax.errorbar(conf, acc, yerr=[acc - alo, ahi - acc], marker="o", capsize=3,
                        lw=1.8, label=f"{label} (ECE={e:.3f}, CI {lo:.3f}-{hi:.3f})")
    ax.set_xlabel("predicted P(CT win)"); ax.set_ylabel("observed CT win-rate")
    ax.set_title("Reliability diagram with 95% CIs — are the probabilities honest?\n"
                 "(on the diagonal = honest; error bar crossing the line = no real miscalibration)")
    ax.legend(loc="upper left", fontsize=9); ax.set_aspect("equal")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
