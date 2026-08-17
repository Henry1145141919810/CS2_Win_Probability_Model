"""F10 - Pathwise extreme-path calibration (trajectory honesty).

Left  : winner's-trough write-off frequency vs the calibrated-martingale benchmark, at thresholds
        0.05/0.10/0.20, for the production model (lgbm EB2) and the headline model (logreg EFB2).
        Observed sits below the continuous benchmark at every threshold.
Right : PIT curve of the loser's-peak statistic. Under an honest forecast the empirical CDF hugs the
        diagonal; upper-tail inflation (over-reaction) would push it BELOW the diagonal. The CS2 curve
        sits ON or ABOVE the diagonal everywhere: zero upper-tail inflation, mild discrete conservatism.

Reads outputs/pathwise_benchmark.csv, outputs/pathwise_perround_*.parquet.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "paper" / "F10_pathwise_calibration.png"
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    d = pl.read_csv(ROOT / "outputs" / "pathwise_benchmark.csv").filter(pl.col("threshold") > 0)
    thr = [0.05, 0.10, 0.20]
    x = np.arange(len(thr)); w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: observed vs benchmark ----
    specs = [("lgbm", "EB2", BLUE, "lgbm EB2 (production)"),
             ("logreg", "EFB2", AQUA, "logreg EFB2 (headline)")]
    for j, (m, fs, col, lab) in enumerate(specs):
        sub = d.filter((pl.col("model") == m) & (pl.col("fs") == fs)).sort("threshold")
        obs = sub["observed"].to_numpy() * 100
        lo = sub["obs_lo"].to_numpy() * 100
        hi = sub["obs_hi"].to_numpy() * 100
        ax1.bar(x + (j - 0.5) * w, obs, w, color=col, edgecolor=SURF, linewidth=1.0, label=lab)
        ax1.errorbar(x + (j - 0.5) * w, obs, yerr=[obs - lo, hi - obs], fmt="none",
                     ecolor=INK2, elinewidth=1, capsize=2.5)
    ben = d.filter((pl.col("model") == "lgbm") & (pl.col("fs") == "EB2")).sort("threshold")["benchmark"].to_numpy() * 100
    ax1.scatter(x, ben, marker="_", s=600, color=RED, linewidths=2.2, zorder=6,
                label="martingale benchmark (continuous)")
    for xi, b in zip(x, ben):
        ax1.text(xi, b + 0.6, f"{b:.1f}%", ha="center", va="bottom", fontsize=8, color=RED)
    ax1.set_xticks(x); ax1.set_xticklabels([f"$\\leq${int(t*100)}%" for t in thr], fontsize=9)
    ax1.set_xlabel("eventual winner written off to ...")
    ax1.set_ylabel("share of rounds (%)")
    ax1.set_title("Write-offs are rarer than the honest benchmark (never more common)",
                  fontsize=10, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    # ---- right: PIT curve (loser's-peak) ----
    ax2.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.2, label="honest (uniform)")
    # over-reaction = upper-tail inflation = empirical CDF BELOW the diagonal (the lower-right triangle)
    ax2.fill_between([0, 1], [0, 1], [0, 0], color=RED, alpha=0.07)
    ax2.text(0.70, 0.34, "over-reaction zone\n(empirical below diagonal)\n— empty here",
             fontsize=7.8, color=RED, ha="center")
    for m, fs, col, lab in specs:
        f = ROOT / "outputs" / f"pathwise_perround_{m}_{fs}.parquet"
        U = np.sort(pl.read_parquet(f)["U"].to_numpy())
        ecdf = np.arange(1, len(U) + 1) / len(U)
        ax2.plot(U, ecdf, color=col, lw=2, label=f"{m} {fs}")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal")
    ax2.set_xlabel("PIT value $U$ = benchmark CDF of loser's peak")
    ax2.set_ylabel("empirical CDF")
    ax2.set_title("PIT hugs / rises above the diagonal: no over-reaction", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
