"""SHAP analysis for the tree model — interaction-aware feature attribution.

Permutation importance gives a collinearity-robust ranking; SHAP adds the DIRECTION and the
per-snapshot distribution of each feature's contribution (and captures interactions the
linear coefficients can't). Tree SHAP is exact and fast. We fit XGBoost on set E and explain
a random sample of snapshots (full 476k is unnecessary for a stable summary).

Outputs: console mean|SHAP| ranking + beeswarm and bar summary plots.
Part of the standard interpretation protocol for tree models (docs/methodology.md).

Usage: python src/models/shap_analysis.py [--sample 20000]
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
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import make_model  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures"
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS


def main():
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=20000)
    args = ap.parse_args()

    df = pl.read_parquet(DATA)
    X = np.nan_to_num(df.select(COLS).to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    print(f"fitting xgb on {len(y)} snapshots, set E ({len(COLS)} feats) ...")
    m = make_model("xgb").fit(X, y)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(args.sample, len(y)), replace=False)
    Xs = X[idx]
    print(f"computing TreeSHAP on {len(idx)} sampled snapshots ...")
    expl = shap.TreeExplainer(m)
    sv = expl.shap_values(Xs)

    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(-mean_abs)
    print(f"\n{'feature':26s} {'mean|SHAP|':>10}  (top 15)")
    for j in order[:15]:
        print(f"  {COLS[j]:26s} {mean_abs[j]:>10.4f}")

    OUT.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(sv, Xs, feature_names=COLS, show=False, max_display=18)
    plt.tight_layout(); plt.savefig(OUT / "shap_beeswarm_E.png", dpi=120); plt.close()
    shap.summary_plot(sv, Xs, feature_names=COLS, plot_type="bar", show=False, max_display=18)
    plt.tight_layout(); plt.savefig(OUT / "shap_bar_E.png", dpi=120); plt.close()
    print(f"\nwrote {OUT/'shap_beeswarm_E.png'} and shap_bar_E.png")


if __name__ == "__main__":
    main()
