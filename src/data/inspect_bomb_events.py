"""Bomb-event vocabulary probe — does the parsed data support a DEFUSE-PROGRESS feature?

`features/bomb.py` currently only consumes plant / defuse / detonate / drop / pickup, i.e.
the moment a defuse FINISHES. A "how many seconds into the defuse" feature needs the moment
it STARTS (and ideally when it is ABORTED). This script reports what the parsed bomb channel
actually contains, so we know whether the feature is buildable from the existing bundle or
needs a re-parse.

It answers, over all (or --limit N) demos in data/parquet/bomb/:
  1. the exact column names of the bomb table
  2. the full `event` vocabulary + counts
  3. for every completed defuse: is there a start event before it, and how many ticks earlier
     (expect ~320 ticks = 5s with kit, ~640 = 10s without)
  4. how many defuse ATTEMPTS never completed (aborted / defuser killed) — these are the rows
     that make the feature non-leaky; if there are none, the feature is a label in disguise.

Usage:
    python src/data/inspect_bomb_events.py
    python src/data/inspect_bomb_events.py --limit 20 --bomb-dir data/parquet/bomb
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BOMB_DIR = ROOT / "data" / "parquet" / "bomb"
TICKRATE = 64

# every spelling awpy / demoparser2 might use for the two events we care about
START_ALIASES = {"begin_defuse", "begindefuse", "bomb_begindefuse", "defuse_start",
                 "start_defuse", "defusing"}
ABORT_ALIASES = {"abort_defuse", "abortdefuse", "bomb_abortdefuse", "defuse_abort"}
DONE_ALIASES = {"defuse", "defused", "bomb_defused"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bomb-dir", type=Path, default=BOMB_DIR)
    args = ap.parse_args()

    files = sorted(args.bomb_dir.glob("*.parquet"))
    if not files:
        print(f"no parquet in {args.bomb_dir} — run this where the data bundle lives")
        return
    if args.limit:
        files = files[: args.limit]
    print(f"scanning {len(files)} demos in {args.bomb_dir}\n")

    events = Counter()
    cols_seen = None
    gaps = []            # (start_event, ticks_between_start_and_defuse, had_kit_unknown)
    n_defuse = n_defuse_with_start = 0
    n_starts = n_starts_completed = n_aborts = 0
    rounds_total = 0

    for f in files:
        df = pl.read_parquet(f)
        if cols_seen is None:
            cols_seen = list(df.columns)
            print("bomb table columns:", cols_seen, "\n")
        if "event" not in df.columns:
            print(f"!! {f.name}: no 'event' column")
            continue
        events.update(df["event"].to_list())
        rcol = "round_num" if "round_num" in df.columns else None
        if rcol is None:
            continue
        for rn, g in df.group_by(rcol):
            rounds_total += 1
            ev = sorted(((r["tick"], r["event"]) for r in g.iter_rows(named=True)),
                        key=lambda e: e[0])
            starts = [t for t, e in ev if e in START_ALIASES]
            aborts = [t for t, e in ev if e in ABORT_ALIASES]
            dones = [t for t, e in ev if e in DONE_ALIASES]
            n_starts += len(starts)
            n_aborts += len(aborts)
            n_defuse += len(dones)
            for dt in dones:
                prev = [t for t in starts if t <= dt]
                if prev:
                    n_defuse_with_start += 1
                    n_starts_completed += 1
                    gaps.append(dt - max(prev))

    print("=" * 68)
    print("EVENT VOCABULARY (all demos)")
    print("=" * 68)
    for e, c in events.most_common():
        mark = ""
        if e in START_ALIASES:
            mark = "   <-- DEFUSE START (this is what the feature needs)"
        elif e in ABORT_ALIASES:
            mark = "   <-- DEFUSE ABORT"
        print(f"  {str(e):<24} {c:>8}{mark}")

    print("\n" + "=" * 68)
    print("DEFUSE-PROGRESS FEASIBILITY")
    print("=" * 68)
    print(f"  rounds scanned           : {rounds_total}")
    print(f"  completed defuses        : {n_defuse}")
    print(f"  defuse-start events      : {n_starts}")
    print(f"  defuse-abort events      : {n_aborts}")
    print(f"  defuses with a start ev  : {n_defuse_with_start}")
    failed = n_starts - n_starts_completed
    print(f"  STARTED but not completed: {failed}   <-- the non-leaky rows")
    if gaps:
        g = pl.Series(gaps)
        secs = [x / TICKRATE for x in gaps]
        print(f"\n  start->defuse ticks: min={g.min()} median={g.median()} max={g.max()}")
        print(f"  start->defuse secs : min={min(secs):.2f} median={sorted(secs)[len(secs)//2]:.2f} "
              f"max={max(secs):.2f}   (expect ~5s with kit, ~10s without)")
    if n_starts == 0:
        print("\n  VERDICT: no defuse-START event in the parsed bundle.")
        print("  -> the feature CANNOT be built from data/parquet/bomb as-is.")
        print("  -> options: (a) re-parse bomb events keeping bomb_begindefuse/bomb_abortdefuse,")
        print("              (b) re-parse ticks with the per-player `is_defusing` prop,")
        print("              (c) do NOT back-derive the start from a completed defuse — that")
        print("                  only ever sees defuses that SUCCEEDED = label leakage.")
    else:
        print("\n  VERDICT: defuse-start events present — feature is buildable from the bundle.")


if __name__ == "__main__":
    main()
