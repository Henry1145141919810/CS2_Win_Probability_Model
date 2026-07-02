"""SHAP dependence / interaction plots — HOW the key features act, not just how much.

For the headline features we plot SHAP value vs feature value (colored by the strongest
interacting feature SHAP auto-detects). Shows e.g. whether defuse_time_margin flips the win
prob sharply around 0 (the defuse-feasibility boundary) and how Voronoi control acts.

Usage: python src/models/shap_dependence.py [--set EB2 --sample 25000]
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
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures"
FEATURES = ["defuse_time_margin", "defuse_margin_kit", "defuse_contest_margin",
            "control_deficit", "min_ct_dist_to_bomb", "ct_bomb_local_deficit",
            "firepower_rating_diff", "ct_clutch_score", "ct_firepower_rating"]


def main():
    import shap
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="EB2")
    ap.add_argument("--sample", type=int, default=25000)
    args = ap.parse_args()
    cols = FEATURE_SETS[args.set]
    df = pl.read_parquet(DATA)
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    print(f"fitting xgb on set {args.set} ({len(cols)} feats) ...")
    m = make_model("xgb").fit(X, y)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(args.sample, len(y)), replace=False)
    Xs = X[idx]
    print(f"TreeSHAP on {len(idx)} snaps ...")
    sv = shap.TreeExplainer(m).shap_values(Xs)
    OUT.mkdir(parents=True, exist_ok=True)
    for feat in FEATURES:
        if feat not in cols:
            print(f"  skip {feat} (not in set)"); continue
        j = cols.index(feat)
        shap.dependence_plot(j, sv, Xs, feature_names=cols, interaction_index="auto",
                             show=False)
        plt.tight_layout()
        p = OUT / f"shap_dependence_{feat}.png"
        plt.savefig(p, dpi=120); plt.close()
        # quick numeric read: SHAP at low vs high feature values
        v = Xs[:, j]; lo = v < np.percentile(v, 25); hi = v > np.percentile(v, 75)
        print(f"  {feat:24s} mean SHAP low-quartile {sv[lo,j].mean():+.3f}  "
              f"high-quartile {sv[hi,j].mean():+.3f}  -> {p.name}")


if __name__ == "__main__":
    main()
