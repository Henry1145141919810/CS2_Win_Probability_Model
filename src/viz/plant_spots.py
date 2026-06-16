"""Cluster bomb-plant coordinates into named plant spots and render on the Inferno radar.

Produces outputs/figures/plant_spots.png (plants colored by site, KMeans cluster centroids
numbered) and prints each cluster's centroid + count, so the common plant spots can be
labeled with standard callouts (A: pit/default/graveyard...; B: coffin/fountain/box...).
"""
from __future__ import annotations
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "plant_spots.png"
RADAR = Path.home() / ".awpy" / "maps" / "de_inferno.png"
# de_inferno radar transform (awpy map_data)
POS_X, POS_Y, SCALE = -2087, 3870, 4.9
K = {"BombsiteA": 5, "BombsiteB": 5}


def to_radar(x, y):
    return (np.asarray(x) - POS_X) / SCALE, (POS_Y - np.asarray(y)) / SCALE


def main():
    plants = []
    for f in glob.glob(str(ROOT / "data" / "parquet" / "bomb" / "*.parquet")):
        b = pl.read_parquet(f).filter(pl.col("event") == "plant")
        if len(b):
            plants.append(b.select(["X", "Y", "bombsite"]))
    df = pl.concat(plants).filter(pl.col("bombsite") != "")

    fig, ax = plt.subplots(figsize=(10, 10))
    if RADAR.exists():
        ax.imshow(plt.imread(str(RADAR)), zorder=0)

    print("Plant-spot clusters (world centroid -> radar):")
    for site, color in [("BombsiteA", "#2b72d4"), ("BombsiteB", "#f08c1e")]:
        s = df.filter(pl.col("bombsite") == site)
        XY = s.select(["X", "Y"]).to_numpy().astype(float)
        rx, ry = to_radar(XY[:, 0], XY[:, 1])
        ax.scatter(rx, ry, s=4, c=color, alpha=0.25, zorder=1)
        km = KMeans(n_clusters=K[site], n_init=10, random_state=0).fit(XY)
        print(f"\n{site} ({len(s)} plants):")
        for i, c in enumerate(km.cluster_centers_):
            n = int((km.labels_ == i).sum())
            cx, cy = to_radar(c[0], c[1])
            ax.scatter(cx, cy, s=220, marker="o", facecolors="none",
                       edgecolors="black", linewidths=2, zorder=2)
            tag = f"{'A' if site=='BombsiteA' else 'B'}{i+1}"
            ax.text(cx, cy, tag, ha="center", va="center", fontsize=11,
                    fontweight="bold", zorder=3)
            print(f"  {tag}: world=({c[0]:.0f},{c[1]:.0f})  n={n} ({100*n/len(s):.0f}%)")
    ax.set_title("Inferno bomb-plant clusters (A=blue, B=orange) — label these with callouts")
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=110); plt.close(fig)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
