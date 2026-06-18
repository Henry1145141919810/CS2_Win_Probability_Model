"""Model-agnostic permutation importance (AUC drop) — the collinearity-robust importance.

Raw logistic coefficients are unstable under collinearity (e.g. the Voronoi control_deficit
"+0.27" artifact from a duplicate column). Permutation importance sidesteps that: fit the
model, then for each feature, shuffle that column in the held-out fold and measure how much
OOF AUC drops. A feature the model genuinely relies on causes a big drop; a redundant one
(its info is carried by a correlated feature) causes little drop — which is the honest read.

GroupKFold by match (no leakage); importance averaged over folds. Works for ANY model, so it
is the standard interpretation step for new architectures (see docs/methodology.md protocol).

Outputs: console table + outputs/permutation_importance_{set}.csv + a grouped bar chart
(top features, colored by pillar) for the requested models.

Usage:
  python src/models/permutation_importance.py --models logreg,xgb,lgbm,catboost,rf --set E
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
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import (MAPCONTROL_COLS, TERRITORY_COLS,  # noqa: E402
                                 MAPCONTROL_LOS_COLS)
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import make_model  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs"
TAC = TACTICAL_COLS + BOMB_COLS
SETS = {
    "A": ECONOMY_COLS,
    "E": ECONOMY_COLS + MAPCONTROL_COLS + TAC,
    "ET": ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS,
}
PILLAR = ({c: "economy" for c in ECONOMY_COLS} | {c: "voronoi" for c in MAPCONTROL_COLS}
          | {c: "territory" for c in TERRITORY_COLS} | {c: "grey" for c in MAPCONTROL_LOS_COLS}
          | {c: "tactical" for c in TAC})
PCOLOR = {"economy": "#4477aa", "voronoi": "#ee6677", "territory": "#228833",
          "grey": "#aa3377", "tactical": "#ccbb44"}


def perm_importance(df, cols, y, groups, kind, n_repeats=1, seed=0):
    """Mean OOF-AUC drop when each column is permuted within the held-out fold."""
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    rng = np.random.default_rng(seed)
    drops = np.zeros((len(cols),))
    fold_w = 0
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        base = roc_auc_score(y[te], m.predict_proba(X[te])[:, 1])
        Xte = X[te].copy()
        for j in range(len(cols)):
            acc = 0.0
            col = Xte[:, j].copy()
            for _ in range(n_repeats):
                Xte[:, j] = rng.permutation(col)
                acc += base - roc_auc_score(y[te], m.predict_proba(Xte)[:, 1])
            Xte[:, j] = col
            drops[j] += (acc / n_repeats) * len(te)
        fold_w += len(te)
    return drops / fold_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb,lgbm,catboost,rf")
    ap.add_argument("--set", default="E", choices=list(SETS))
    args = ap.parse_args()
    cols = SETS[args.set]
    models = args.models.split(",")

    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    print(f"data {len(y)} snaps; set {args.set} ({len(cols)} feats); models {models}\n")

    imp = {}
    for mdl in models:
        imp[mdl] = perm_importance(df, cols, y, groups, mdl)
        order = np.argsort(-imp[mdl])
        print(f"=== {mdl} — top 12 by AUC-drop ===")
        for j in order[:12]:
            print(f"  {cols[j]:26s} {imp[mdl][j]:+.4f}  {PILLAR.get(cols[j],'?')}")
        print()

    # CSV (all models, all features)
    OUT.mkdir(parents=True, exist_ok=True)
    tbl = {"feature": cols, "pillar": [PILLAR.get(c, "?") for c in cols]}
    for mdl in models:
        tbl[f"{mdl}_auc_drop"] = imp[mdl].tolist()
    csv = OUT / f"permutation_importance_{args.set}.csv"
    pl.DataFrame(tbl).sort(f"{models[0]}_auc_drop", descending=True).write_csv(csv)
    print(f"wrote {csv}")

    # figure: top-15 by mean drop across models, grouped bars
    mean_drop = np.mean([imp[m] for m in models], axis=0)
    top = np.argsort(-mean_drop)[:15][::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    n = len(models); h = 0.8 / n
    for i, mdl in enumerate(models):
        ax.barh(np.arange(len(top)) + i * h, imp[mdl][top], height=h, label=mdl)
    ax.set_yticks(np.arange(len(top)) + 0.4 - h / 2)
    ax.set_yticklabels([f"{cols[j]} [{PILLAR.get(cols[j],'?')[:4]}]" for j in top], fontsize=8)
    ax.set_xlabel("OOF AUC drop when permuted (bigger = more important)")
    ax.set_title(f"Permutation importance — set {args.set} (collinearity-robust)\n"
                 "top 15 features by mean drop across models")
    ax.legend(fontsize=8)
    fig.tight_layout()
    figp = OUT / "figures" / f"permutation_importance_{args.set}.png"
    figp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figp, dpi=120); plt.close(fig)
    print(f"wrote {figp}")


if __name__ == "__main__":
    main()
