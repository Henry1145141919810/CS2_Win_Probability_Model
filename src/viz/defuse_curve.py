"""F11 - Defuse-progress curve honesty.

Mean predicted P(CT win) as a defuse runs, by progress bucket, on the 2026 test set:
  - actual empirical CT win rate (ground truth)
  - EB2 (no defuse feature): flat, does not move while the defuse completes
  - EB2D (+ defuse progress): tracks the empirical curve

Reads outputs/defuse_benchmark_curve.csv (model, bucket, n, actual, EB2, EB2D).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "paper" / "F11_defuse_curve.png"
BLUE, AQUA, RED = "#2a78d6", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    d = pl.read_csv(ROOT / "outputs" / "defuse_benchmark_curve.csv").filter(pl.col("model") == "logreg")
    x = np.arange(d.height)
    labels = [b.replace("0.", ".").replace("-", "–") for b in d["bucket"].to_list()]
    labels = [f"{int(float(b.split('-')[0])*100)}–{min(100,int(float(b.split('-')[1])*100))}%"
              for b in d["bucket"].to_list()]
    actual = d["actual"].to_numpy(); eb2 = d["EB2"].to_numpy(); eb2d = d["EB2D"].to_numpy()

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(x, actual, "o-", color=INK, lw=2.4, ms=8, mec=SURF, mew=1.4, zorder=5, label="actual CT win rate")
    ax.plot(x, eb2, "s--", color=RED, lw=2, ms=7, mec=SURF, mew=1.2, zorder=4, label="EB2 (no defuse feature)")
    ax.plot(x, eb2d, "^-", color=AQUA, lw=2, ms=8, mec=SURF, mew=1.2, zorder=4, label="EB2D (+ defuse progress)")

    # annotate the gap at the last bucket
    ax.annotate("EB2 stays flat as\nthe defuse completes", xy=(x[-1], eb2[-1]),
                xytext=(x[-1]-1.6, eb2[-1]-0.06), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("defuse progress")
    ax.set_ylabel("P(CT win)")
    ax.set_title("Predicted vs actual CT win rate during a defuse (2026 test)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
