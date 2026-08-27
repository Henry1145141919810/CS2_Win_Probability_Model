"""Fault-tolerant batch demo parser (Week-1 pipeline).

Parses each extracted .dem with awpy v2 into 5 Parquet channels, one demo at a time,
with aggressive memory management (parse -> save -> free -> gc -> next). Never loads
all demos at once: one demo's ticks table is ~hundreds of MB.

Channels written (per demo, named <demo_stem>.parquet):
    data/parquet/ticks/    (downsampled to ~1 sample/sec by default)
    data/parquet/kills/
    data/parquet/rounds/
    data/parquet/bomb/
    data/parquet/grenades/

Design notes:
- Idempotent: skips a demo if all its channel files already exist (unless --overwrite).
- Robust: per-demo try/except; failures appended to failed_demos.txt and logged.
- Schema-defensive: detects the tick-column name for downsampling rather than assuming.
- Only a focused set of player props is parsed (positions, hp, armor, inventory, econ,
  defuser, team, callout) to keep memory and file size down.

Usage:
    python src/data/batch_parse.py                 # parse all in demos/extracted
    python src/data/batch_parse.py --limit 1       # parse just one (test)
    python src/data/batch_parse.py --stride 64     # sample every 64th tick
"""
from __future__ import annotations
import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DEFAULT = ROOT / "demos" / "extracted"
OUT_DEFAULT = ROOT / "data" / "parquet"
FAIL_LOG = ROOT / "failed_demos.txt"

# smokes/infernos (tiny: position + start/end tick per nade) enable smoke-aware LOS.
# grenades (per-tick trajectories, ~446MB, unused) is dropped to keep parsed data lean.
CHANNELS = ["ticks", "kills", "rounds", "bomb", "smokes", "infernos"]
# Channels we DERIVE rather than read off awpy. `defuse` is one row per defuse ATTEMPT,
# computed from the full-resolution tick stream before downsampling (see _defuse_attempts).
DERIVED_CHANNELS = ["defuse"]
DEFUSE_PROP = "is_defusing"

# Focused player props (subset of awpy DEFAULT_PLAYER_PROPS) — enough for all 4 pillars.
# team_clan_name is needed to track each TEAM through the halftime side-swap (labels are
# per-side 'ct'/'t', but first-to-13 and map completeness are per-team).
PLAYER_PROPS = [
    "team_name", "team_clan_name", "X", "Y", "Z", "health", "armor_value",
    "inventory", "current_equip_value", "has_defuser", "has_helmet",
    "last_place_name", "flash_duration",
    # is_defusing: per-tick "this player is holding the defuse right now"
    # (engine field CCSPlayerPawn.m_bIsDefusing). The bomb event stream carries only
    # bomb_defused -- awpy's parse_bomb() hardcodes 5 events and bomb_begindefuse comes
    # back EMPTY even when requested explicitly -- so this prop is the ONLY source for
    # when a defuse STARTS, and the only way to see attempts that FAILED.
    "is_defusing",
    # facing + movement (added for FOV/grey control + influence model)
    "yaw", "pitch", "velocity_X", "velocity_Y", "velocity_Z",
]
WORLD_PROPS = [
    "game_time", "is_bomb_planted", "which_bomb_zone",
    "is_freeze_period", "is_warmup_period",
]

TICK_COL_CANDIDATES = ["tick", "tick_id", "game_tick"]


MIN_FREE_GB = 7.0  # a single demo can peak ~6.7GB during awpy parse


def _wait_for_memory(min_free: float = MIN_FREE_GB, max_wait: int = 120):
    """Pause before parsing a demo if free RAM is too low (avoids OOM on big demos).
    Waits up to max_wait seconds for headroom, then proceeds with a warning."""
    try:
        import psutil
    except ImportError:
        return
    waited = 0
    while waited < max_wait:
        free = psutil.virtual_memory().available / 1e9
        if free >= min_free:
            return
        print(f"    [memory] only {free:.1f}GB free (<{min_free}GB) — waiting "
              f"for headroom; close WeChat/Weixin to speed this up...", flush=True)
        time.sleep(15)
        waited += 15
    print("    [memory] proceeding despite low RAM (Windows will page to disk).", flush=True)


def _table(dem, name):
    """Return the awpy dataframe for a channel, or None if unavailable."""
    try:
        return getattr(dem, name)
    except Exception:  # noqa: BLE001
        return None


def _tick_column(df):
    for c in TICK_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _downsample_ticks(ticks, stride: int):
    """Keep ~1 row per `stride` ticks (per the once-per-second sampling plan).

    Uses the detected tick column: keep rows where tick % stride == first-tick offset.
    Falls back to returning the full table (with a warning) if no tick column found.
    """
    import polars as pl  # local import so the module imports without polars present

    col = _tick_column(ticks)
    if col is None:
        print("    [warn] no tick column found; saving full ticks table", file=sys.stderr)
        return ticks
    t0 = ticks[col].min()
    return ticks.filter(((pl.col(col) - t0) % stride) == 0)


def _defuse_attempts(ticks, bomb_df):
    """One row per defuse ATTEMPT, derived from the FULL-RESOLUTION tick stream.

    MUST run before _downsample_ticks. The project's time convention is uniform --
    every time feature is `(snapshot_tick - reference_tick) / 64` against a reference
    that lives in a NON-downsampled table (rounds.freeze_end in economy.py,
    bomb.plant tick in bomb.py). At 1 Hz the exact defuse start is gone forever, so
    this channel IS that reference table for the defuse timer: it keeps the tick-exact
    start so `defuse_elapsed_sec` has the same 1/64 s precision as `time_elapsed_sec`
    and `defuse_time_margin`.

    An attempt = a maximal run of consecutive ticks where one player has is_defusing
    set. Verified on a real demo: runs are perfectly contiguous (no flicker) and a
    kit defuse is exactly 320 ticks = 5.000 s.

    Columns: round_num, steamid, start_tick, end_tick, n_ticks, had_kit, completed.
    `completed` = a bomb 'defuse' event lands on the attempt's end (it fires at
    end_tick + 1); everything else is an attempt the Ts interrupted.
    """
    import polars as pl

    if ticks is None or len(ticks) == 0 or DEFUSE_PROP not in ticks.columns:
        return None
    cols = [c for c in ("tick", "steamid", "round_num", "has_defuser") if c in ticks.columns]
    t = (ticks.filter(pl.col(DEFUSE_PROP).cast(pl.Boolean, strict=False))
              .select(cols).sort(["steamid", "tick"]))
    if t.is_empty():
        return None
    # run-length encode: a new attempt starts when the player changes or the ticks break
    t = t.with_columns([
        (pl.col("steamid") != pl.col("steamid").shift(1)).fill_null(value=True).alias("_newp"),
        ((pl.col("tick") - pl.col("tick").shift(1)) > 1).fill_null(value=True).alias("_gap"),
    ])
    t = t.with_columns((pl.col("_newp") | pl.col("_gap")).cast(pl.Int32).cum_sum().alias("_run"))
    agg = [pl.col("steamid").first(), pl.col("tick").min().alias("start_tick"),
           pl.col("tick").max().alias("end_tick"), pl.len().alias("n_ticks")]
    if "round_num" in cols:
        agg.append(pl.col("round_num").first())
    if "has_defuser" in cols:
        agg.append(pl.col("has_defuser").first().alias("had_kit"))
    g = t.group_by("_run").agg(agg).sort("start_tick").drop("_run")

    # completed? the bomb 'defuse' event fires one tick after the last defusing tick
    done = set()
    if bomb_df is not None and len(bomb_df) and "event" in bomb_df.columns:
        done = set(bomb_df.filter(pl.col("event") == "defuse")["tick"].to_list())
    return g.with_columns(
        pl.col("end_tick")
          .map_elements(lambda e: int(any(abs(x - e) <= 2 for x in done)),
                        return_dtype=pl.Int32)
          .alias("completed"))


def parse_one(dem_path: Path, out: Path, stride: int, overwrite: bool) -> str:
    from awpy import Demo
    try:
        from . import awpy_patch  # when imported as a package
    except ImportError:
        import awpy_patch  # when run as a script (src/data on sys.path)
    awpy_patch.apply()  # fixes int-encoded winner column on some demos

    stem = dem_path.stem
    targets = {ch: out / ch / f"{stem}.parquet" for ch in CHANNELS + DERIVED_CHANNELS}
    # Skip-check on CHANNELS only: a match where nobody ever touched the bomb writes no
    # `defuse` file at all (1 of the 32 2026 demos), and including it here would re-parse
    # those demos on every run.
    if not overwrite and all(targets[ch].exists() for ch in CHANNELS):
        return "skip"

    # CS2 GOTV is 64-tick; awpy's Demo() default of 128 is wrong (header carries no
    # tickrate) and would corrupt any time-from-tick math. Verified empirically =64.
    dem = Demo(dem_path, tickrate=64)
    dem.parse(player_props=PLAYER_PROPS, other_props=WORLD_PROPS)

    # DERIVED first: needs the tick stream at full 64 Hz, before downsampling below.
    written = 0
    att = _defuse_attempts(_table(dem, "ticks"), _table(dem, "bomb"))
    if att is not None and len(att):
        targets["defuse"].parent.mkdir(parents=True, exist_ok=True)
        att.write_parquet(targets["defuse"])
        written += 1
        n_done = int(att["completed"].sum())
        print(f"    defuse: {len(att)} attempts ({n_done} completed, "
              f"{len(att) - n_done} interrupted)")
    else:
        print(f"    [warn] {stem}: no defuse attempts found", file=sys.stderr)

    for ch in CHANNELS:
        df = _table(dem, ch)
        if df is None or len(df) == 0:
            print(f"    [warn] {stem}: channel '{ch}' empty/missing", file=sys.stderr)
            continue
        if ch == "ticks":
            df = _downsample_ticks(df, stride)
        targets[ch].parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(targets[ch])
        written += 1

    del dem
    gc.collect()
    return "ok" if written else "empty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stride", type=int, default=64,
                    help="sample every Nth tick for the ticks channel (default 64)")
    ap.add_argument("--limit", type=int, default=0, help="parse at most N demos (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    dems = sorted(args.raw_dir.glob("*.dem"))
    if args.limit:
        dems = dems[: args.limit]
    if not dems:
        print(f"No .dem files in {args.raw_dir}")
        return

    print(f"Found {len(dems)} demo(s) in {args.raw_dir}")
    counts = {"ok": 0, "skip": 0, "empty": 0, "fail": 0}
    for i, dem_path in enumerate(dems, 1):
        _wait_for_memory()  # one demo can peak ~6.7GB; don't start if RAM is too low
        t0 = time.time()
        print(f"[{i}/{len(dems)}] {dem_path.name} ...", flush=True)
        try:
            status = parse_one(dem_path, args.out, args.stride, args.overwrite)
            counts[status] = counts.get(status, 0) + 1
            print(f"    {status} ({time.time()-t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            counts["fail"] += 1
            with FAIL_LOG.open("a", encoding="utf-8") as f:
                f.write(f"{dem_path.name}\t{type(e).__name__}: {e}\n")
            print(f"    [FAIL] {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
        gc.collect()

    print(f"\nDone: {counts}")
    if counts["fail"]:
        print(f"Failures logged to {FAIL_LOG}")


if __name__ == "__main__":
    main()
