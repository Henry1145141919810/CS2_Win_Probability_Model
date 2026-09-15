"""Figure for Study 3 (no-economy ablation): how far the non-economy pillars get alone.

Panel A: OOF AUC by feature set (logistic and XGBoost in-time, logistic out-of-time), with match-
         bootstrap 95% CIs on the in-time logistic value.
Panel B: OOF AUC by seconds-into-round bucket for the four single-block sets (logistic), with the
         economy baseline and the production set as neutral reference lines.

Reads outputs/noecon_ablation.csv, outputs/noecon_holdout.csv (optional), outputs/noecon_timeprofile.csv.
Writes outputs/figures/paper/F12_noecon_ablation.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
FIG = OUT / "figures" / "paper" / "F12_noecon_ablation.png"

# validated default palette (dataviz skill, light mode): slot1 blue, slot2 orange, slot3 aqua, slot4 yellow
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

ORDER = ["EB2", "NoMoney", "A", "NoEcon", "SpatialT", "NoEcon-", "SpatialT-", "Spatial",
         "Bomb", "Money", "Combat", "Score", "Clock"]
LABEL = {"A": "A  economy block (17)", "EB2": "EB2  production set (72)",
         "Clock": "clock only (1)", "Score": "score + round (4)", "Money": "money only (7)",
         "Combat": "combat state only (6)", "Spatial": "Voronoi control only (9)",
         "SpatialT-": "Voronoi + tactical, no counts (29)", "SpatialT": "Voronoi + tactical (37)",
         "Bomb": "bomb geometry + defuse race (18)", "NoEcon-": "all non-economy, no counts (41)",
         "NoEcon": "all non-economy (55)", "NoMoney": "everything except money (65)"}


def main():
    res = pl.read_csv(OUT / "noecon_ablation.csv")
    tp = pl.read_csv(OUT / "noecon_timeprofile.csv")
    hold = pl.read_csv(OUT / "noecon_holdout.csv") if (OUT / "noecon_holdout.csv").exists() else None
    present = res.filter(pl.col("model") == "logreg").sort("AUC", descending=True)["set"].to_list()
    sets = [s for s in present if s in ORDER]          # sorted by in-time logistic AUC
    n = len(sets)
    # label column counts from the CSV (the hard-coded counts drifted from the sets)
    ncols = {row["set"]: int(row["ncols"]) for row in res.filter(pl.col("model") == "logreg").to_dicts()}
    for k in list(LABEL):
        if k in ncols:
            LABEL[k] = LABEL[k].rsplit(" (", 1)[0] + f" ({ncols[k]})"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                         "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- Panel A: dot plot ----------------------------------------------------------------
    ypos = np.arange(n)[::-1]
    for y in ypos:
        ax.axhline(y, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.text(0.503, n - 1.5, "chance", color=INK2, fontsize=8, ha="left", va="center")

    def col(model, s, c):
        r = res.filter((pl.col("model") == model) & (pl.col("set") == s))
        return float(r[c][0]) if r.height else np.nan
    for i, s in enumerate(sets):
        y = ypos[i]
        a = col("logreg", s, "AUC"); lo = col("logreg", s, "AUC_lo"); hi = col("logreg", s, "AUC_hi")
        if np.isfinite(lo):
            ax.plot([lo, hi], [y, y], color=C1, lw=1.2, zorder=2)
        ax.plot(a, y, "o", ms=8, color=C1, zorder=3)
        x = col("xgb", s, "AUC")
        if np.isfinite(x):
            ax.plot(x, y - 0.22, "o", ms=8, color=C2, zorder=3)
        if hold is not None:
            h = hold.filter((pl.col("model") == "logreg") & (pl.col("set") == s))
            if h.height:
                ax.plot(float(h["AUC"][0]), y, "o", ms=8, mfc=SURF, mec=C3, mew=2, zorder=4)
        ax.text(1.005, y, f"{a:.3f}", va="center", ha="left", fontsize=8, color=INK2)
    ax.set_yticks(ypos); ax.set_yticklabels([LABEL[s] for s in sets], fontsize=8.5)
    ax.set_xlim(0.48, 1.0); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("AUC (5-fold GroupKFold, out-of-fold)")
    ax.set_title("A. Discrimination of each block on its own", loc="left", fontsize=10, color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h1 = plt.Line2D([], [], color=C1, marker="o", ms=8, lw=0, label="logistic, in-time (95% CI)")
    h2 = plt.Line2D([], [], color=C2, marker="o", ms=8, lw=0, label="XGBoost, in-time")
    h3 = plt.Line2D([], [], mfc=SURF, mec=C3, mew=2, marker="o", ms=8, lw=0, label="logistic, out-of-time 2026")
    ax.legend(handles=[h1, h2, h3] if hold is not None else [h1, h2], loc="lower right", frameon=False, fontsize=8)

    # ---- Panel B: time profile ---------------------------------------------------------------
    t = tp.filter(pl.col("model") == "logreg")
    def series(s):
        d = t.filter(pl.col("set") == s).sort("t_lo")
        x = [(lo + min(hi, 120)) / 2 for lo, hi in zip(d["t_lo"], d["t_hi"])]
        return np.array(x), d["AUC"].to_numpy()
    bx.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=0)
    for s, ls, dy in [("A", (0, (1, 2)), 0.0), ("EB2", "-", 0.0)]:
        if s in sets:
            x, y = series(s)
            bx.plot(x, y, color=INK2, lw=1.5, ls=ls, zorder=1)
    # reference labels placed at the 40-60 s bucket, above the lines, to avoid the crowded right edge
    xa, ya = series("A"); xe, ye = series("EB2")
    bx.annotate("A (economy)", (xa[3], ya[3]), xytext=(xa[3] - 6, ya[3] + 0.055), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    bx.annotate("EB2 (production)", (xe[5], ye[5]), xytext=(xe[5] - 24, ye[5] + 0.05), color=INK2, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ends = []
    for s, c in [("Money", C1), ("Combat", C2), ("Spatial", C3), ("Bomb", C4)]:
        if s not in sets:
            continue
        x, y = series(s)
        bx.plot(x, y, color=c, lw=2, marker="o", ms=6, zorder=3, label=LABEL[s].split(" (")[0])
        ends.append((y[-1], LABEL[s].split(" (")[0], x[-1]))
    # direct labels at line ends, nudged apart if closer than 0.03
    ends.sort()
    placed = []
    for yv, lab, xv in ends:
        yy = yv
        if placed and yy - placed[-1] < 0.03:
            yy = placed[-1] + 0.03
        placed.append(yy)
        bx.text(xv + 1.5, yy, lab, color=INK, fontsize=8, va="center")
    bx.set_xlabel("seconds into the round (bucket midpoint; last bucket = 90 s+)")
    bx.set_ylabel("AUC on snapshots in the bucket")
    bx.set_ylim(0.45, 1.0); bx.set_xlim(0, 125)
    bx.set_title("B. When each block becomes informative", loc="left", fontsize=10, color=INK)
    bx.grid(axis="y", color=GRID, lw=0.8); bx.set_axisbelow(True)
    for sp in ["top", "right"]:
        bx.spines[sp].set_visible(False)
    bx.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle("No-economy ablation: what the non-economy pillars predict alone (220 matches, 476,595 snapshots)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=200)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
