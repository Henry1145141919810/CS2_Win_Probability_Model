"""Build a data-driven nav-area -> zone map for de_inferno.

awpy nav areas carry no callout label, but every player tick reports `place` (the
in-game callout, e.g. 'Banana', 'BombsiteA'). We assign each nav-mesh area the callout
of the nearest labelled player position (aggregated over many ticks), then group the 24
callouts into the 5 named macro-zones from the project plan.

Output: configs/inferno_zone_map.parquet  (area_idx, place, zone)

Run once after demos are parsed:
    python src/features/build_zone_map.py
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.mapcontrol import nav_grid  # noqa: E402

TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
OUT = ROOT / "configs" / "inferno_zone_map.parquet"

# The 5 named macro-zones (plan) -> the unambiguous callouts that define them.
# Approach callouts (Apartments, Arch, Pit, ...) are intentionally left as "other"
# so the named-zone features are clean; they still count in OVERALL control.
ZONE_CALLOUTS = {
    "a_site": {"BombsiteA"},
    "b_site": {"BombsiteB"},
    "banana": {"Banana"},
    "mid": {"Middle", "LowerMid", "SecondMid", "TopofMid"},
    "ct_spawn": {"CTSpawn"},
}
_CALLOUT_TO_ZONE = {c: z for z, cs in ZONE_CALLOUTS.items() for c in cs}


def main():
    files = sorted(glob.glob(str(TICKS_DIR / "*.parquet")))
    if not files:
        print(f"No ticks parquet in {TICKS_DIR}")
        return
    # collect (X, Y, place) samples from all demos
    parts = []
    for f in files:
        df = pl.read_parquet(f, columns=["X", "Y", "place"]).drop_nulls()
        parts.append(df)
    pts = pl.concat(parts)
    # subsample for speed (positions repeat heavily)
    if pts.height > 300_000:
        pts = pts.sample(300_000, seed=0)
    xy = pts.select(["X", "Y"]).to_numpy()
    place = pts["place"].to_list()
    print(f"samples: {len(place)} from {len(files)} demos; callouts: {sorted(set(place))}")

    tree = cKDTree(xy)
    cents, _, _ = nav_grid()
    _, idx = tree.query(cents, k=1)
    area_place = [place[i] for i in idx]
    area_zone = [_CALLOUT_TO_ZONE.get(p, "other") for p in area_place]

    out = pl.DataFrame({
        "area_idx": list(range(len(cents))),
        "place": area_place,
        "zone": area_zone,
    })
    out.write_parquet(OUT)
    print(f"\nWROTE {OUT}")
    print("zone area counts:", out["zone"].value_counts().sort("zone").to_dicts())


if __name__ == "__main__":
    main()
