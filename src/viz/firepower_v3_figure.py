"""F9 - Firepower encodings vs EB2 (in-sample, paired CIs).

Forest plot of the AUC gain over EB2 (no firepower) for every firepower encoding x model:
v2 (raw sum) and the three team-ranking-weighted v3 variants (log2 / inv / linear). Every paired
match-level bootstrap interval crosses zero: no encoding significantly improves on the skill-free
model in-sample.

Reads outputs/firepower_v3_full.csv.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "paper" / "F9_firepower_encodings.png"
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})
ENC = [("EB2+v2", "v2 (raw sum)", BLUE),
       ("EB2+v3-log2", "v3 weight 1/log2(rank+1)", AQUA),
       ("EB2+v3-inv", "v3 weight 1/rank", YELLOW),
       ("EB2+v3-linear", "v3 weight linear", RED)]
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
NICE = {"logreg": "Logistic", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost", "rf": "Random forest"}


def main():
    d = pl.read_csv(ROOT / "outputs" / "firepower_v3_full.csv")
    fig, ax = plt.subplots(figsize=(9, 7))
    ylabels, y = [], 0
    yticks = []
    for enc, elab, col in ENC:
        for m in MODELS:
            r = d.filter((pl.col("encoding") == enc) & (pl.col("model") == m))
            if not r.height:
                continue
            v = float(r["dAUC_vs_EB2"][0]); lo = float(r["vsEB2_lo"][0]); hi = float(r["vsEB2_hi"][0])
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=5,
                        ecolor=col, elinewidth=1.4, capsize=2.5, mec=SURF, mew=0.8)
            ylabels.append(f"{NICE[m]}"); yticks.append(y)
            y += 1
        # group separator label
        ax.text(-0.010, y - len(MODELS) / 2 - 0.5, elab, fontsize=8.5, color=col,
                ha="right", va="center", fontweight="bold", rotation=90)
        y += 1
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("AUC gain over EB2 (no firepower), 5-fold OOF, paired 95% bootstrap CI")
    ax.set_title("No firepower encoding beats the skill-free model in-sample", fontsize=11, loc="left")
    ax.set_xlim(-0.008, 0.008)
    ax.invert_yaxis()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
