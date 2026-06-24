"""SHAP analysis — interaction-aware feature attribution, for ANY model and feature set.

Permutation importance gives a collinearity-robust ranking; SHAP adds DIRECTION and the
per-snapshot distribution of each feature's contribution (and captures interactions linear
coefficients can't). TreeSHAP (exact) for xgb/lgbm/catboost/rf; LinearExplainer for logreg.
Default set = ET so the TERRITORY features are included and can be ranked. Explains a random
sample of snapshots (full 476k is unnecessary for a stable summary).

Per model: console mean|SHAP| top-15 + where each pillar's best feature ranks + a beeswarm
and bar plot. Part of the standard interpretation protocol (docs/methodology.md).

NOTE: RandomForest is excluded by default — TreeSHAP on 300 deep unpruned RF trees is
pathologically slow (>50 min even on a few-k sample). Use permutation_importance.py for RF
(it covers all 5 models cheaply); SHAP here is for the GBMs + logreg (LinearExplainer).

Usage:
  python src/models/shap_analysis.py                              # logreg+GBMs, set ET
  python src/models/shap_analysis.py --models xgb --set E --sample 20000
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
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS, MAPCONTROL_LOS_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import make_model  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures"
TAC = TACTICAL_COLS + BOMB_COLS
SETS = {
    "E": ECONOMY_COLS + MAPCONTROL_COLS + TAC,
    "ET": ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS,
}
PILLAR = ({c: "economy" for c in ECONOMY_COLS} | {c: "voronoi" for c in MAPCONTROL_COLS}
          | {c: "territory" for c in TERRITORY_COLS} | {c: "grey" for c in MAPCONTROL_LOS_COLS}
          | {c: "tactical" for c in TAC})
# slower TreeSHAP for RF -> cap its sample
RF_CAP = 6000


def shap_for(kind, cols, X, y, sample, rng):
    """Return (mean_abs[len(cols)], shap_values[n,len(cols)], Xs[n,len(cols)])."""
    import shap
    n = min(sample, len(y))
    if kind == "rf":
        n = min(n, RF_CAP)
    idx = rng.choice(len(y), size=n, replace=False)
    if kind == "logreg":
        m = make_model("logreg").fit(X, y)                      # pipeline: scaler+logreg
        sc, clf = m.steps[0][1], m.steps[1][1]
        Xt = sc.transform(X)                                    # explain in the model's own space
        bg = Xt[rng.choice(len(y), size=2000, replace=False)]
        expl = shap.LinearExplainer(clf, bg)
        Xs = Xt[idx]
        sv = expl.shap_values(Xs)
    else:
        m = make_model(kind).fit(X, y)
        Xs = X[idx]
        sv = shap.TreeExplainer(m).shap_values(Xs)
        if isinstance(sv, list):       # some versions return per-class list
            sv = sv[1]
    sv = np.asarray(sv)
    return np.abs(sv).mean(axis=0), sv, Xs, n


def main():
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb,lgbm,catboost")  # rf excluded: TreeSHAP too slow
    ap.add_argument("--set", default="ET", choices=list(SETS))
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()
    cols = SETS[args.set]
    models = args.models.split(",")

    df = pl.read_parquet(DATA)
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    print(f"data {len(y)} snaps; set {args.set} ({len(cols)} feats); models {models}\n")
    OUT.mkdir(parents=True, exist_ok=True)

    table = {"feature": cols, "pillar": [PILLAR.get(c, "?") for c in cols]}
    for kind in models:
        rng = np.random.default_rng(0)
        mean_abs, sv, Xs, n = shap_for(kind, cols, X, y, args.sample, rng)
        table[f"{kind}_mean_abs_shap"] = mean_abs.tolist()
        order = np.argsort(-mean_abs)
        rank = {cols[order[r]]: r + 1 for r in range(len(order))}
        print(f"=== {kind}  (TreeSHAP/Linear on {n} snaps) — top 12 ===")
        for j in order[:12]:
            print(f"  {cols[j]:26s} {mean_abs[j]:>9.4f}  {PILLAR.get(cols[j],'?')}")
        # where does each spatial pillar's BEST feature rank?
        for pil in ("voronoi", "territory"):
            pc = [c for c in cols if PILLAR.get(c) == pil]
            if pc:
                best = max(pc, key=lambda c: mean_abs[cols.index(c)])
                print(f"   -> best {pil:9s}: {best} rank #{rank[best]} "
                      f"(mean|SHAP| {mean_abs[cols.index(best)]:.4f})")
        print()
        # plots (skip beeswarm for logreg standardized space readability -> bar only)
        shap.summary_plot(sv, Xs, feature_names=cols, plot_type="bar", show=False, max_display=18)
        plt.tight_layout(); plt.savefig(OUT / f"shap_bar_{args.set}_{kind}.png", dpi=120); plt.close()
        if kind != "logreg":
            shap.summary_plot(sv, Xs, feature_names=cols, show=False, max_display=18)
            plt.tight_layout(); plt.savefig(OUT / f"shap_beeswarm_{args.set}_{kind}.png", dpi=120); plt.close()

    csv = ROOT / "outputs" / f"shap_importance_{args.set}.csv"
    pl.DataFrame(table).sort(f"{models[-1]}_mean_abs_shap", descending=True).write_csv(csv)
    print(f"wrote {csv} and per-model plots in {OUT}")


if __name__ == "__main__":
    main()
