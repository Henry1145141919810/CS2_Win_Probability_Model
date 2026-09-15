"""Figures for the round-to-map study.

Panel A: calibration by score margin, realised vs M0 (i.i.d. state model) vs M1/M2 (economy chain).
Panel B: one full map, per-second map win probability (M4 chaining) with the round-start values and
         the per-second round probability underneath.
Panel C: learning curve (OOF log-loss vs number of maps) for M0/M1/M2/M3, if available.

Reads outputs/map_wp_calibration_margin.csv, outputs/map_wp_persecond.parquet, outputs/map_wp_roundstart.parquet,
outputs/map_wp_learning_curve.csv. Writes outputs/figures/paper/F14_map_wp.png
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
FIG = OUT / "figures" / "paper" / "F14_map_wp.png"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
EXAMPLE = "cloud9-vs-vitality-m1-inferno"


def main():
    cal = pl.read_csv(OUT / "map_wp_calibration_margin.csv")
    ps = pl.read_parquet(OUT / "map_wp_persecond.parquet")
    rs = pl.read_parquet(OUT / "map_wp_roundstart.parquet")
    lc = pl.read_csv(OUT / "map_wp_learning_curve.csv") if (OUT / "map_wp_learning_curve.csv").exists() else None

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                         "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1], width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, 0]); bx = fig.add_subplot(gs[1, :]); cx = fig.add_subplot(gs[0, 1])

    # ---- A: calibration by margin --------------------------------------------------------------
    sd = cal["sd"].to_numpy()
    ax.plot(sd, cal["realised"].to_numpy(), "o-", color=INK, lw=2, ms=6, label="realised (X wins the map)")
    ax.plot(sd, cal["M0"].to_numpy(), "s--", color=C2, lw=1.5, ms=5, label="M0: i.i.d. rounds, side rate only")
    ax.plot(sd, cal["M1"].to_numpy(), "D-", color=C1, lw=1.5, ms=5, label="M1: economy chain")
    if "M2c" in cal.columns:
        ax.plot(sd, cal["M2c"].to_numpy(), "^-", color=C3, lw=1.2, ms=5, label="M2c: + strength, recalibrated")
    ax.set_xlabel("score margin at round start (X minus Y)"); ax.set_ylabel("P(X wins the map)")
    ax.set_title("A. Calibration by score margin (all round starts, out-of-fold)", loc="left", fontsize=10, color=INK)
    ax.set_ylim(0, 1); ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ---- C: learning curve -----------------------------------------------------------------------
    if lc is not None:
        for name, c, mk in [("M0", C2, "s"), ("M1", C1, "D"), ("M2", C3, "^"), ("M3", C4, "o")]:
            if name in lc.columns:
                cx.plot(lc["n_maps"].to_numpy(), lc[name].to_numpy(), marker=mk, color=c, lw=1.5, ms=6, label=name)
        cx.set_xlabel("number of training maps"); cx.set_ylabel("out-of-fold log-loss (round starts)")
        cx.set_title("C. Learning curve: variance of each rung", loc="left", fontsize=10, color=INK)
        cx.grid(color=GRID, lw=0.8); cx.set_axisbelow(True); cx.legend(frameon=False, fontsize=8)
        for sp in ["top", "right"]:
            cx.spines[sp].set_visible(False)
    else:
        cx.axis("off")

    # ---- B: one full map, per-second ------------------------------------------------------------
    m = ps.filter(pl.col("match_id") == EXAMPLE).sort(["round_num", "tick"])
    r = rs.filter(pl.col("match_id") == EXAMPLE).sort("round_num")
    # continuous map clock: cumulative seconds across rounds
    offsets = {}; t0 = 0.0
    for rn, g in m.group_by("round_num", maintain_order=True):
        offsets[int(rn[0] if isinstance(rn, tuple) else rn)] = t0
        t0 += float(g["time_elapsed_sec"].max()) + 20.0     # + freeze time between rounds
    m = m.with_columns(pl.col("round_num").map_elements(lambda k: offsets[int(k)], return_dtype=pl.Float64).alias("off"))
    m = m.with_columns(tmap=(pl.col("off") + pl.col("time_elapsed_sec")) / 60.0)
    bx.axhline(0.5, color=GRID, lw=1)
    bx.plot(m["tmap"].to_numpy(), m["p_x"].to_numpy(), color=INK2, lw=0.9, alpha=0.8, label="round win probability, X (per second)")
    bx.plot(m["tmap"].to_numpy(), m["p_map"].to_numpy(), color=C1, lw=2, label="map win probability, X (M4 chaining)")
    xs = [offsets[int(k)] / 60.0 for k in r["round_num"]]
    bx.plot(xs, r["M2"].to_numpy(), "o", ms=5, mfc=SURF, mec=C1, mew=1.5, label="round-start value (M2)")
    for k, x in zip(r["round_num"], xs):
        if int(k) in (1, 13) or int(k) % 4 == 0:
            bx.text(x, 0.02, f"r{int(k)}", fontsize=7, color=INK2, ha="left")
    won = int(r["x_won"][0]); xteam = "CT-first team (X)"
    bx.set_title(f"B. {EXAMPLE}: per-second map win probability for the {xteam}; X {'won' if won else 'lost'} the map "
                 f"{int(r['a'].max() + (1 if won else 0))}-{int(r['b'].max() + (0 if won else 1))}", loc="left", fontsize=10, color=INK)
    bx.set_xlabel("map clock (minutes, live time plus 20 s freeze per round)"); bx.set_ylabel("probability")
    bx.set_ylim(0, 1); bx.grid(axis="y", color=GRID, lw=0.8); bx.set_axisbelow(True)
    bx.legend(frameon=False, fontsize=8, loc="upper left", ncol=3)
    for sp in ["top", "right"]:
        bx.spines[sp].set_visible(False)

    fig.suptitle("From round to map: the economy-chain state model and the per-second map win probability",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
