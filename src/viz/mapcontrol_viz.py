"""Visualize the Pillar-2 Voronoi map-control surface over the de_inferno radar.

Produces paper figures:
  - snapshots():  control surface at chosen seconds into a round (e.g. t=5/15/25s)
  - animate():    a per-second GIF of the control surface evolving through a round

The control surface = each nav-mesh triangle shaded by the team of the nearest ALIVE
player (the same Voronoi assignment used for the features), overlaid with player dots.
This makes the abstract "territorial control %" visually concrete for readers.

Usage:
  python src/viz/mapcontrol_viz.py demos/extracted/faze-vs-g2-m1-inferno.dem --round 5
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # src/ on path
from awpy.plot.utils import game_to_pixel  # noqa: E402
from features.mapcontrol import load_nav, voronoi_owner, voronoi_control  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures"
RADAR = Path.home() / ".awpy" / "maps" / "de_inferno.png"
TICKRATE = 64
CT_COLOR = (0.16, 0.44, 0.82)   # blue
T_COLOR = (0.92, 0.52, 0.12)    # orange

# Precompute nav-area polygon vertices in radar-pixel coords (once; order matches
# nav_grid / voronoi_owner). Areas have variable corner counts, so keep a list.
_nav = load_nav()
_tris_px = [
    np.array([game_to_pixel("de_inferno", (c.x, c.y, c.z))[:2] for c in a.corners])
    for a in _nav.areas.values()
]


def _alive(ticks, round_num, tick):
    import polars as pl
    return ticks.filter(
        (pl.col("round_num") == round_num) & (pl.col("tick") == tick) & (pl.col("health") > 0)
    )


def _render(ax, alive, title):
    """Draw radar + control surface + players onto ax."""
    px = alive["X"].to_list(); py = alive["Y"].to_list(); side = alive["side"].to_list()
    owner = voronoi_owner(px, py, side)
    colors = [CT_COLOR if o == "CT" else T_COLOR for o in owner]

    ax.imshow(plt.imread(RADAR))
    ax.add_collection(PolyCollection(_tris_px, facecolors=colors, alpha=0.45, edgecolors="none"))

    # player dots in pixel space
    for x, y, s in zip(px, py, side):
        ppx, ppy, _ = game_to_pixel("de_inferno", (x, y, 0))
        c = CT_COLOR if str(s).lower().startswith("c") else T_COLOR
        ax.scatter(ppx, ppy, s=70, c=[c], edgecolors="white", linewidths=1.3, zorder=5)

    ctrl = voronoi_control(px, py, side)["ct_voronoi_control_pct"]
    ax.set_title(f"{title}\nCT control = {ctrl:.0%}", fontsize=11)
    ax.axis("off")
    return ctrl


def snapshots(ticks, rounds, round_num, secs=(5, 15, 25), tag=""):
    import polars as pl
    rr = rounds.filter(pl.col("round_num") == round_num).row(0, named=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(secs), figsize=(6 * len(secs), 6))
    if len(secs) == 1:
        axes = [axes]
    for ax, sec in zip(axes, secs):
        tick = rr["freeze_end"] + sec * TICKRATE
        alive = _alive(ticks, round_num, tick)
        if len(alive) == 0:
            ax.set_title(f"t+{sec}s (no data)"); ax.axis("off"); continue
        _render(ax, alive, f"t+{sec}s into round {round_num}")
    out = OUT / f"mapcontrol_snapshots_{tag}r{round_num}.png"
    fig.suptitle(f"de_inferno Voronoi map control — round {round_num} "
                 f"(winner: {rr['winner'].upper()})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")
    return out


def animate(ticks, rounds, round_num, tag="", fps=4):
    """Per-second GIF through the round (freeze_end .. end)."""
    import polars as pl
    from PIL import Image
    rr = rounds.filter(pl.col("round_num") == round_num).row(0, named=True)
    dur = int((rr["end"] - rr["freeze_end"]) / TICKRATE)
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for sec in range(0, dur + 1):
        tick = rr["freeze_end"] + sec * TICKRATE
        alive = _alive(ticks, round_num, tick)
        if len(alive) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        _render(ax, alive, f"round {round_num}  t+{sec}s")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                       fig.canvas.buffer_rgba().tobytes()).convert("RGB"))
        plt.close(fig)
    out = OUT / f"mapcontrol_anim_{tag}r{round_num}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"saved {out}  ({len(frames)} frames)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("demo", type=Path)
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    from awpy import Demo
    try:
        from data import awpy_patch; awpy_patch.apply()
    except Exception:
        pass
    dem = Demo(args.demo, tickrate=TICKRATE)
    dem.parse(player_props=["team_name", "X", "Y", "Z", "health", "last_place_name"])
    tag = args.demo.stem.split("-m")[0][:20] + "_"
    snapshots(dem.ticks, dem.rounds, args.round, tag=tag)
    if args.gif:
        animate(dem.ticks, dem.rounds, args.round, tag=tag)


if __name__ == "__main__":
    main()
