"""Visualize de_inferno map control over the radar: ALL THREE control models, side by side.

Row 1 = Voronoi          (each nav area -> nearest living player's team; the KEPT feature).
Row 2 = grey (LOS+FOV+smoke) — instantaneous 4-state CT/T/contested/grey: an area is
              controlled only if a living player is in range AND has line-of-sight AND is
              facing it AND no smoke blocks it. Overlaid with FACING LINES (yaw) + SMOKES.
Row 3 = territory (memory+decay) — the grey model but cleared space STAYS a team's for
              `decay`=15s without re-checking (replayed from freeze-end to each tick).

Shows the PROCESS: Voronoi is greedy/total, grey is realistic-but-flickery (~80% grey),
territory is the stabilized version that recovers Voronoi-level predictiveness.

Reads the parsed parquet (memory-light; ticks already carry yaw), so no demo re-parse.

Usage:
  python src/viz/mapcontrol_viz.py --match faze-vs-g2-m1-inferno --round 5
  python src/viz/mapcontrol_viz.py --match faze-vs-g2-m1-inferno --round 5 --secs 5,15,25
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path
from awpy.plot.utils import game_to_pixel  # noqa: E402
from features.mapcontrol import (load_nav, voronoi_owner, contest_owner,  # noqa: E402
                                 voronoi_control, contest_control, TerritoryControl)

ROOT = Path(__file__).resolve().parents[2]
PARQ = ROOT / "data" / "parquet"
OUT = ROOT / "outputs" / "figures"
RADAR = Path.home() / ".awpy" / "maps" / "de_inferno.png"
TICKRATE = 64
SMOKE_DUR = 18 * 64
COLORS = {"CT": (0.16, 0.44, 0.82), "T": (0.92, 0.52, 0.12),
          "contested": (0.55, 0.27, 0.63), "grey": (0.78, 0.78, 0.78)}

_nav = load_nav()
_tris_px = [np.array([game_to_pixel("de_inferno", (c.x, c.y, c.z))[:2] for c in a.corners])
            for a in _nav.areas.values()]
_nav_sizes = np.array([a.size for a in _nav.areas.values()], float)


def _pix(x, y):
    p = game_to_pixel("de_inferno", (x, y, 0.0))
    return p[0], p[1]


def _draw_players(ax, px, py, side, yaws=None, facing=False):
    for i, (x, y, s) in enumerate(zip(px, py, side)):
        ppx, ppy = _pix(x, y)
        c = COLORS["CT"] if str(s).lower().startswith("c") else COLORS["T"]
        if facing and yaws is not None and not np.isnan(yaws[i]):
            ex, ey = _pix(x + 320 * np.cos(np.radians(yaws[i])),
                          y + 320 * np.sin(np.radians(yaws[i])))
            ax.plot([ppx, ex], [ppy, ey], color=c, lw=1.6, zorder=4, alpha=0.9)
        ax.scatter(ppx, ppy, s=70, c=[c], edgecolors="white", linewidths=1.3, zorder=5)


def _draw_smokes(ax, smokes):
    r_px = abs(_pix(144, 0)[0] - _pix(0, 0)[0])
    for sx, sy in smokes:
        cx, cy = _pix(sx, sy)
        ax.add_patch(Circle((cx, cy), r_px, color="white", alpha=0.55, zorder=3,
                            ec="grey"))


def _surface(ax, owner):
    ax.imshow(plt.imread(RADAR))
    ax.add_collection(PolyCollection(_tris_px, facecolors=[COLORS[o] for o in owner],
                                     alpha=0.5, edgecolors="none"))
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--secs", default="5,15,25")
    args = ap.parse_args()
    secs = [int(s) for s in args.secs.split(",")]

    t = pl.read_parquet(PARQ / "ticks" / f"{args.match}.parquet")
    r = pl.read_parquet(PARQ / "rounds" / f"{args.match}.parquet")
    sm = pl.read_parquet(PARQ / "smokes" / f"{args.match}.parquet")
    rr = r.filter(pl.col("round_num") == args.round).row(0, named=True)
    fe = rr["freeze_end"]
    rt = t.filter((pl.col("round_num") == args.round) & (pl.col("tick") >= fe))
    avail = sorted(rt["tick"].unique().to_list())
    target_ticks = [min(avail, key=lambda x: abs(x - (fe + sec * TICKRATE))) for sec in secs]

    def _smokes_at(tk):
        return [(row["X"], row["Y"]) for row in
                sm.filter((pl.col("round_num") == args.round) & (pl.col("start_tick") <= tk)).iter_rows(named=True)
                if (row["end_tick"] or row["start_tick"] + SMOKE_DUR) > tk]

    # Row 3 (territory) is STATEFUL: replay the round freeze-end -> each target tick,
    # snapshotting the memory owner-map whenever we hit a requested second.
    terr = TerritoryControl()
    terr_owner_at = {}
    want = set(target_ticks)
    for tk in avail:
        s = rt.filter((pl.col("tick") == tk) & (pl.col("health") > 0))
        if s.height >= 1:  # update memory whenever >=1 player is alive (carry over otherwise)
            yw = s["yaw"].to_list() if "yaw" in s.columns else None
            terr.update(s["X"].to_list(), s["Y"].to_list(), s["side"].to_list(),
                        yaws=yw, smokes=_smokes_at(tk), tick=tk)
        if tk in want:  # always snapshot a requested second (territory persists from memory)
            terr_owner_at[tk] = terr.owner(tk)

    fig, axes = plt.subplots(3, len(secs), figsize=(5.2 * len(secs), 15.6))
    for col, (sec, tk) in enumerate(zip(secs, target_ticks)):
        s = rt.filter((pl.col("tick") == tk) & (pl.col("health") > 0))
        px, py, side = s["X"].to_list(), s["Y"].to_list(), s["side"].to_list()
        yaws = s["yaw"].to_list() if "yaw" in s.columns else None
        smk = _smokes_at(tk)

        # row 1: Voronoi
        a0 = axes[0, col] if len(secs) > 1 else axes[0]
        _surface(a0, voronoi_owner(px, py, side))
        _draw_players(a0, px, py, side)
        vc = voronoi_control(px, py, side)["ct_voronoi_control_pct"]
        a0.set_title(f"t+{sec}s — Voronoi\nCT {vc:.0%}", fontsize=10)

        # row 2: grey model + facing + smokes
        a1 = axes[1, col] if len(secs) > 1 else axes[1]
        _surface(a1, contest_owner(px, py, side, yaws=yaws, smokes=smk))
        _draw_smokes(a1, smk)
        _draw_players(a1, px, py, side, yaws=np.asarray(yaws, float) if yaws else None, facing=True)
        cc = contest_control(px, py, side, yaws=yaws, smokes=smk)
        a1.set_title(f"t+{sec}s — grey (LOS+FOV+smoke)\nCT {cc['ct_los_control']:.0%} "
                     f"grey {cc['grey_pct']:.0%}", fontsize=10)

        # row 3: territory (memory + decay)
        a2 = axes[2, col] if len(secs) > 1 else axes[2]
        _surface(a2, terr_owner_at[tk])
        _draw_smokes(a2, smk)
        _draw_players(a2, px, py, side, yaws=np.asarray(yaws, float) if yaws else None, facing=True)
        owner = terr_owner_at[tk]
        ct_terr = float(_nav_sizes[owner == "CT"].sum() / _nav_sizes.sum())
        grey_terr = float(_nav_sizes[owner == "grey"].sum() / _nav_sizes.sum())
        a2.set_title(f"t+{sec}s — territory (memory+decay 15s)\nCT {ct_terr:.0%} "
                     f"grey {grey_terr:.0%}", fontsize=10)

    fig.suptitle(f"{args.match}  round {args.round} (winner {rr['winner'].upper()}) — "
                 f"three control models: Voronoi / grey / territory   "
                 f"[blue=CT orange=T purple=contested grey=neutral; lines=facing, white=smoke]",
                 fontsize=11)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"mapcontrol_compare_{args.match}_r{args.round}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
