"""Synthetic unit tests for the defuse-progress features (features/defuse.py).

No demos required — builds tiny polars frames by hand. Run wherever polars is installed:

    python tests/test_defuse_progress.py

Covers the four cases that decide whether the feature is honest:
  1. a normal kit defuse: elapsed/frac climb second by second
  2. an ABORTED attempt: progress exists mid-attempt, then goes back to 0 (this is the case
     that keeps the feature from being a relabelling of `ct_won`)
  3. a defuser KILLED with no abort event in the demo: must NOT stay stuck at 100%
  4. the per-tick `is_defusing` fallback when the bomb channel has no start event
"""
from __future__ import annotations
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.defuse import DefuseTracker, defuse_progress_features  # noqa: E402

TR = 64
PLANT_TICK = 1000
PLANT = {"tick": PLANT_TICK, "x": 0.0, "y": 0.0, "site": 0, "area": 0}
DEFUSER = 111


def snap(tick, health=100, kit=1, defusing=None):
    """One-tick frame: the defusing CT + one T, in the columns the feature reads."""
    row = {"tick": [tick, tick], "steamid": [DEFUSER, 222], "side": ["ct", "t"],
           "health": [health, 100], "has_defuser": [kit, 0], "X": [0.0, 500.0],
           "Y": [0.0, 500.0]}
    if defusing is not None:
        row["is_defusing"] = defusing
    return pl.DataFrame(row)


def bomb(events):
    return pl.DataFrame({"tick": [t for t, _ in events],
                         "event": [e for _, e in events],
                         "steamid": [DEFUSER] * len(events),
                         "X": [0.0] * len(events), "Y": [0.0] * len(events)})


def check(name, got, want):
    for k, v in want.items():
        assert abs(got[k] - v) < 1e-6, f"{name}: {k} = {got[k]!r}, expected {v!r}"
    print(f"  ok  {name}")


def test_kit_defuse_climbs():
    start = PLANT_TICK + 10 * TR
    tr = DefuseTracker(bomb([(PLANT_TICK, "plant"), (start, "begin_defuse"),
                             (start + 5 * TR, "defuse")]))
    assert tr.source == "events", tr.source
    check("before the attempt", defuse_progress_features(snap(start - TR), tr, PLANT, start - TR),
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0, "defuse_attempts_so_far": 0})
    check("1s in (kit)", defuse_progress_features(snap(start + TR), tr, PLANT, start + TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 1.0, "defuse_progress_frac": 0.2,
           "defuse_beats_fuse": 1, "defuse_attempts_so_far": 1})
    check("4s in (kit)", defuse_progress_features(snap(start + 4 * TR), tr, PLANT, start + 4 * TR),
          {"defuse_elapsed_sec": 4.0, "defuse_progress_frac": 0.8})
    # no kit -> same wall-clock second is half the progress
    check("4s in (no kit)",
          defuse_progress_features(snap(start + 4 * TR, kit=0), tr, PLANT, start + 4 * TR),
          {"defuse_elapsed_sec": 4.0, "defuse_progress_frac": 0.4})


def test_aborted_attempt_returns_to_zero():
    start = PLANT_TICK + 10 * TR
    abort = start + 3 * TR
    tr = DefuseTracker(bomb([(PLANT_TICK, "plant"), (start, "begin_defuse"),
                             (abort, "abort_defuse"), (PLANT_TICK + 40 * TR, "detonate")]))
    check("2s into the aborted try",
          defuse_progress_features(snap(start + 2 * TR), tr, PLANT, start + 2 * TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 2.0})
    got = defuse_progress_features(snap(abort + TR), tr, PLANT, abort + TR)
    check("after the abort", got,
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0, "defuse_progress_frac": 0.0,
           "defuse_attempts_so_far": 1})  # the failed try is still remembered


def test_killed_defuser_does_not_stick():
    """No abort event in the demo: the attempt must not read as a finished defuse forever."""
    start = PLANT_TICK + 10 * TR
    tr = DefuseTracker(bomb([(PLANT_TICK, "plant"), (start, "begin_defuse")]))
    check("2s in, still alive", defuse_progress_features(snap(start + 2 * TR), tr, PLANT,
                                                         start + 2 * TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 2.0})
    check("defuser dead -> closed",
          defuse_progress_features(snap(start + 3 * TR, health=0), tr, PLANT, start + 3 * TR),
          {"defuse_in_progress": 0, "defuse_progress_frac": 0.0})
    check("outlived the longest defuse -> closed",
          defuse_progress_features(snap(start + 12 * TR), tr, PLANT, start + 12 * TR),
          {"defuse_in_progress": 0, "defuse_progress_frac": 0.0})


def test_is_defusing_fallback():
    """Bomb channel has no start event -> fall back to the per-tick is_defusing prop."""
    start = PLANT_TICK + 10 * TR
    ticks = pl.concat([snap(start + i * TR, defusing=[i >= 1 and i <= 5, False])
                       for i in range(8)])
    tr = DefuseTracker(bomb([(PLANT_TICK, "plant")]), ticks)
    assert tr.source == "ticks", f"expected ticks fallback, got {tr.source}"
    at = start + 3 * TR
    check("3rd defusing sample", defuse_progress_features(snap(at), tr, PLANT, at),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 2.0})  # start recovered at i=1
    last = start + 5 * TR
    check("last defusing sample still counts",
          defuse_progress_features(snap(last), tr, PLANT, last),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 4.0})
    after = start + 7 * TR
    check("after it stops", defuse_progress_features(snap(after), tr, PLANT, after),
          {"defuse_in_progress": 0})


if __name__ == "__main__":
    for fn in (test_kit_defuse_climbs, test_aborted_attempt_returns_to_zero,
               test_killed_defuser_does_not_stick, test_is_defusing_fallback):
        print(fn.__name__)
        fn()
    print("\nall defuse-progress tests passed")
