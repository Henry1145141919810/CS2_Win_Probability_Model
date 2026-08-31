"""F9 - Firepower encodings: what cross-validation rewards, and what survives out-of-time.

Two panels of the same quantity, contested-AUC gain over the skill-free EB2 base, for the
four encodings of Sect. 4.4 across the five classical models. Left is five-fold
cross-validation, right is the 2026 out-of-time holdout. Plotted together because the
finding is the reversal between them, not the level in either: cross-validation places every
encoding at or above the base and prefers the situationally-gated one, while out-of-time that
same encoding falls furthest and none rises clearly above the base.

Reads outputs/firepower_ladder.csv (src/models/eval_firepower_ladder.py).
Writes outputs/figures/paper/F9_firepower_encodings.png.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "firepower_ladder.csv"
OUT = ROOT / "outputs" / "figures" / "paper" / "F9_firepower_encodings.png"

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False,
})

BASE = "EB2 (no FP)"
RUNGS = [("V1  sum", "(a) summed rating", BLUE),
         ("V4  mean", "(b) mean rating", AQUA),
         ("V4.2 mean x rank", "(c) mean $\\times$ team rank", YELLOW),
         ("V4.3 mean + gates", "(d) mean + gates", RED)]
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
MARKS = ["o", "s", "^", "D", "v"]
NICE = {"logreg": "Logistic", "xgb": "XGBoost", "lgbm": "LightGBM",
        "catboost": "CatBoost", "rf": "Random forest"}


def panel(ax, d, col, title):
    base = {m: d.filter((pl.col("rung") == BASE) & (pl.col("model") == m))[col][0]
            for m in MODELS}
    yticks, ylabels = [], []
    for i, (rung, label, colr) in enumerate(RUNGS):
        y0 = -i * (len(MODELS) + 1.6)
        deltas = []
        for j, m in enumerate(MODELS):
            r = d.filter((pl.col("rung") == rung) & (pl.col("model") == m))
            if not r.height:
                continue
            dlt = r[col][0] - base[m]
            deltas.append(dlt)
            y = y0 - j
            ax.plot([0, dlt], [y, y], color=colr, lw=1.4, alpha=0.55, zorder=2)
            ax.plot(dlt, y, MARKS[j], color=colr, ms=6, mec="white", mew=0.8, zorder=3)
        if deltas:  # rung mean, as a heavier tick
            ax.plot(np.mean(deltas), y0 - (len(MODELS) - 1) / 2, "|", color=colr,
                    ms=26, mew=2.4, zorder=4)
        yticks.append(y0 - (len(MODELS) - 1) / 2)
        ylabels.append(label)
    ax.axvline(0, color=INK2, lw=1.2, zorder=1)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_xlabel("contested-AUC gain over EB2 (no firepower)")
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(axis="y", visible=False)


def main():
    d = pl.read_csv(SRC)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panel(axes[0], d, "cv_cauc", "Cross-validation (5-fold, out-of-fold)")
    panel(axes[1], d, "oot_cauc", "Out-of-time (2026 holdout)")
    # Independent x-limits: shared limits render the cross-validation panel unreadable,
    # because the out-of-time damage is roughly five times the size of any in-fold gain.
    # The axis labels carry that scale difference; the caption states it.
    handles = [plt.Line2D([], [], marker=MARKS[j], ls="", color=MUTED, ms=6,
                          mec="white", mew=0.8, label=NICE[m]) for j, m in enumerate(MODELS)]
    handles.append(plt.Line2D([], [], marker="|", ls="", color=MUTED, ms=14, mew=2.2,
                              label="encoding mean"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
