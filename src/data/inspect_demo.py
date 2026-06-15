"""First-demo diagnostic: parse ONE .dem and report the exact awpy v2 schema.

Run this on the first downloaded demo before writing the feature pipelines. It resolves
every assumption the feature code depends on:
  - exact column names per channel (ticks/rounds/kills/bomb/grenades/damages)
  - the tick column name + tickrate (so batch_parse downsampling is correct)
  - the team_name encoding ("CT" / "TERRORIST" / "T" ?) -> needed by mapcontrol
  - round/round_num column name + number of rounds
  - freeze-period flag + how to find freeze-end (round start for sampling)
  - the inventory representation (for utility/AWP counts)
  - last_place_name callout values (for data-driven Pillar-2 zone calibration)

Usage:
    python src/data/inspect_demo.py demos/extracted/<file>.dem
    python src/data/inspect_demo.py            # auto-picks first .dem in demos/extracted
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRACTED = ROOT / "demos" / "extracted"

PLAYER_PROPS = [
    "team_name", "X", "Y", "Z", "health", "armor_value",
    "inventory", "current_equip_value", "has_defuser", "has_helmet",
    "last_place_name", "flash_duration",
]
WORLD_PROPS = [
    "game_time", "is_bomb_planted", "which_bomb_zone",
    "is_freeze_period", "is_warmup_period",
]


def hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def describe(df, name, n=3):
    if df is None:
        print(f"[{name}] -> None")
        return
    try:
        nrows = len(df)
    except Exception:
        print(f"[{name}] -> not a frame: {type(df)}")
        return
    print(f"[{name}] rows={nrows} cols={len(df.columns)}")
    print("  columns:", list(df.columns))
    if nrows:
        print(df.head(n))


def col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    return None


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if path is None:
        dems = sorted(EXTRACTED.glob("*.dem"))
        if not dems:
            print(f"No .dem in {EXTRACTED}. Pass a path explicitly.")
            return
        path = dems[0]
    print(f"Inspecting: {path}")

    from awpy import Demo

    dem = Demo(path)
    print("tickrate (constructor default may differ from header):", dem.tickrate)
    dem.parse(player_props=PLAYER_PROPS, other_props=WORLD_PROPS)

    hr("HEADER")
    try:
        print(dem.header)
    except Exception as e:  # noqa: BLE001
        print("header err:", e)

    for ch in ["rounds", "ticks", "kills", "bomb", "grenades", "damages"]:
        hr(ch.upper())
        describe(getattr(dem, ch, None), ch)

    # Targeted sanity facts the feature code depends on
    hr("KEY FACTS FOR FEATURE CODE")
    ticks = getattr(dem, "ticks", None)
    if ticks is not None and len(ticks):
        tcol = col(ticks, "tick", "tick_id", "game_tick")
        rcol = col(ticks, "round_num", "round", "round_number")
        team = col(ticks, "team_name", "team")
        print("tick column :", tcol)
        print("round column:", rcol)
        if tcol:
            print("  tick range:", ticks[tcol].min(), "..", ticks[tcol].max())
        if rcol:
            print("  n rounds (ticks):", ticks[rcol].n_unique())
        if team:
            print("team_name values:", ticks[team].unique().to_list()[:10])
        if "last_place_name" in ticks.columns:
            vals = ticks["last_place_name"].unique().to_list()
            print(f"last_place_name ({len(vals)} callouts):", sorted(v for v in vals if v)[:40])
        if "is_freeze_period" in ticks.columns:
            print("is_freeze_period values:", ticks["is_freeze_period"].unique().to_list())
        if "inventory" in ticks.columns:
            print("inventory sample:", ticks["inventory"].head(2).to_list())
    rounds = getattr(dem, "rounds", None)
    if rounds is not None and len(rounds):
        print("rounds columns:", list(rounds.columns))
        wcol = col(rounds, "winner", "winner_side", "round_winner")
        if wcol:
            print("winner values:", rounds[wcol].unique().to_list())


if __name__ == "__main__":
    main()
