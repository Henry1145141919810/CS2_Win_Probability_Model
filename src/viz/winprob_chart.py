"""Per-second round win-probability chart (BLAST.tv-style headline figure) WITH a 95% CI band.

Trains on ALL OTHER matches, then plots P(CT win) second-by-second for one round of a
held-out match, annotated with kills and the bomb plant. The CI band is the EPISTEMIC
uncertainty on the curve ("how much would the prediction move had we trained on a different
sample of matches"):
  - logistic: ANALYTIC delta-method interval (asymptotic coefficient covariance) — exact, instant.
  - tree models: match-level block BOOTSTRAP retrain band (resample matches, refit, re-predict).
This CI band is part of the project's standard uncertainty protocol (docs/methodology.md).

Usage:
  python src/viz/winprob_chart.py --match faze-vs-g2-m1-inferno              # logreg, analytic CI
  python src/viz/winprob_chart.py --match X --round 7 --model xgb --bootstrap 50
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import make_model  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
KILLS_DIR = ROOT / "data" / "parquet" / "kills"
ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
OUT = ROOT / "outputs" / "figures"
TICKRATE = 64
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS + TERRITORY_COLS


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logreg_analytic(tr, te):
    """Fit logistic pipeline; return point preds + 95% band via delta method.
    Band = sigmoid(logit ± 1.96·SE), SE from asymptotic covariance (X'WX + ridge)^-1."""
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    Xtr = tr.select(COLS).to_numpy().astype(float)
    ytr = tr["ct_won"].to_numpy()
    imp = SimpleImputer(strategy="median").fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    Ztr = sc.transform(imp.transform(Xtr))
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    # design with intercept column
    A = np.hstack([np.ones((Ztr.shape[0], 1)), Ztr])
    p = clf.predict_proba(Ztr)[:, 1]
    W = p * (1 - p)
    H = A.T @ (W[:, None] * A)
    ridge = np.eye(A.shape[1]) / clf.C      # L2 penalty curvature
    ridge[0, 0] = 0.0                        # intercept unpenalized
    cov = np.linalg.pinv(H + ridge)
    Xte = sc.transform(imp.transform(te.select(COLS).to_numpy().astype(float)))
    Ate = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    logit = Ate @ np.concatenate([clf.intercept_, clf.coef_[0]])
    se = np.sqrt(np.einsum("ij,jk,ik->i", Ate, cov, Ate))
    return _sigmoid(logit), _sigmoid(logit - 1.96 * se), _sigmoid(logit + 1.96 * se)


def _bootstrap_band(tr, te, kind, B, seed=0):
    """Match-level block-bootstrap retrain band for any model."""
    rng = np.random.default_rng(seed)
    Xtr = np.nan_to_num(tr.select(COLS).to_numpy().astype(float))
    ytr = tr["ct_won"].to_numpy()
    groups = tr["match_id"].to_numpy()
    uniq = np.unique(groups)
    idx_by = {m: np.where(groups == m)[0] for m in uniq}
    Xte = np.nan_to_num(te.select(COLS).to_numpy().astype(float))
    point = make_model(kind).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    preds = np.zeros((B, len(Xte)))
    for b in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[m] for m in pick])
        preds[b] = make_model(kind).fit(Xtr[idx], ytr[idx]).predict_proba(Xte)[:, 1]
        if (b + 1) % 10 == 0:
            print(f"  bootstrap {b+1}/{B}")
    lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)
    return point, lo, hi


def _pick_round(pred):
    swings = (pred.group_by("round_num")
              .agg((pl.col("p_ct").max() - pl.col("p_ct").min()).alias("swing")))
    return int(swings.sort("swing", descending=True)["round_num"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--model", default="logreg")
    ap.add_argument("--bootstrap", type=int, default=50,
                    help="B for the bootstrap band (ignored for logreg analytic)")
    ap.add_argument("--ci", choices=["auto", "analytic", "bootstrap", "none"], default="auto")
    args = ap.parse_args()

    df = pl.read_parquet(DATA)
    tr = df.filter(pl.col("match_id") != args.match)
    te_all = df.filter(pl.col("match_id") == args.match)
    if te_all.height == 0:
        print(f"no snapshots for match {args.match}"); return

    ci = args.ci
    if ci == "auto":
        ci = "analytic" if args.model == "logreg" else "bootstrap"
    print(f"model={args.model}  ci={ci}")

    if ci == "analytic":
        if args.model != "logreg":
            print("analytic CI only for logreg; use --ci bootstrap"); return
        point, lo, hi = _logreg_analytic(tr, te_all)
    elif ci == "bootstrap":
        point, lo, hi = _bootstrap_band(tr, te_all, args.model, args.bootstrap)
    else:
        Xtr = np.nan_to_num(tr.select(COLS).to_numpy().astype(float))
        Xte = np.nan_to_num(te_all.select(COLS).to_numpy().astype(float))
        point = make_model(args.model).fit(Xtr, tr["ct_won"].to_numpy()).predict_proba(Xte)[:, 1]
        lo = hi = point
    pred = te_all.with_columns(p_ct=pl.Series(point), p_lo=pl.Series(lo), p_hi=pl.Series(hi))

    rn = args.round or _pick_round(pred)
    r = pred.filter(pl.col("round_num") == rn).sort("tick")
    if r.height == 0:
        print(f"no snapshots for round {rn}"); return

    rounds = pl.read_parquet(ROUNDS_DIR / f"{args.match}.parquet")
    rr = rounds.filter(pl.col("round_num") == rn).row(0, named=True)
    fe = rr["freeze_end"]
    t = (r["tick"].to_numpy() - fe) / TICKRATE
    p = r["p_ct"].to_numpy()
    plo, phi = r["p_lo"].to_numpy(), r["p_hi"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.fill_between(t, plo, phi, color="#888", alpha=0.30, label="95% CI band")
    ax.fill_between(t, 0.5, p, where=(p >= 0.5), color="#2b72d4", alpha=0.18)
    ax.fill_between(t, 0.5, p, where=(p < 0.5), color="#f08c1e", alpha=0.18)
    ax.plot(t, p, color="black", lw=2, label=f"P(CT win) [{args.model}]")

    kills = pl.read_parquet(KILLS_DIR / f"{args.match}.parquet").filter(pl.col("round_num") == rn)
    for k in kills.iter_rows(named=True):
        kt = (k["tick"] - fe) / TICKRATE
        vs = str(k.get("victim_side", "")).lower()
        col = "#f08c1e" if vs.startswith("t") else "#2b72d4"
        ax.axvline(kt, color=col, alpha=0.30, lw=1)
    bp = rr.get("bomb_plant")
    if bp is not None:
        ax.axvline((bp - fe) / TICKRATE, color="red", lw=1.6, ls=":")
        ax.text((bp - fe) / TICKRATE, 0.02, " bomb plant", color="red", fontsize=9)

    mean_w = float(np.mean(phi - plo))
    ax.set_xlabel("seconds since freeze-end"); ax.set_ylabel("P(CT win round)")
    ax.set_ylim(0, 1); ax.set_xlim(0, t.max()); ax.legend(loc="best", fontsize=9)
    ax.set_title(f"{args.match}  round {rn}  (winner: {rr['winner'].upper()})  — "
                 f"{args.model} live win prob + 95% CI ({ci})\n"
                 f"mean band width {mean_w:.2f}; vertical lines = kills "
                 f"(blue: T died, orange: CT died), red dotted = bomb plant", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"winprob_{args.match}_r{rn}_{args.model}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}  (round {rn}, {r.height} snaps, swing={p.max()-p.min():.2f}, "
          f"mean CI width={mean_w:.3f})")


if __name__ == "__main__":
    main()
