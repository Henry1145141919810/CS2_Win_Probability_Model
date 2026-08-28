"""Unit tests for the defuse-progress features (features/defuse.py).

No demos needed — builds a `defuse` channel frame by hand, the same shape batch_parse
derives from the full-resolution tick stream. Run:

    .venv/bin/python tests/test_defuse_progress.py

Covers what actually decides whether the feature is correct and honest:
  1. sub-second precision — the reason the channel exists rather than a 1 Hz tick column
  2. kit-awareness — the same wall-clock second is half the progress without a kit
  3. an INTERRUPTED attempt — progress exists mid-attempt, then returns to 0 (these are the
     rows that keep the feature from being a relabelling of `ct_won`)
  4. two attempts in one round — each is timed from its own start
"""
from __future__ import annotations
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features.defuse import (DefuseTracker, defuse_progress_features,  # noqa: E402
                             attempts_by_round)

TR = 64
PLANT_TICK = 1000
PLANT = {"tick": PLANT_TICK, "x": 0.0, "y": 0.0, "site": 0, "area": 0}


def channel(rows):
    """A `defuse` channel frame: (round_num, start_tick, end_tick, had_kit, completed)."""
    return pl.DataFrame(
        {"round_num": [r[0] for r in rows], "steamid": [111] * len(rows),
         "start_tick": [r[1] for r in rows], "end_tick": [r[2] for r in rows],
         "had_kit": [r[3] for r in rows], "completed": [r[4] for r in rows]})


def track(rows, rn=1):
    return DefuseTracker(attempts_by_round(channel(rows)).get(rn))


def check(name, got, want):
    for k, v in want.items():
        assert abs(got[k] - v) < 1e-9, f"{name}: {k} = {got[k]!r}, expected {v!r}"
    print(f"  ok  {name}")


def test_subsecond_precision():
    """The whole point of the channel: a start that is NOT on a whole second."""
    start = PLANT_TICK + 10 * TR + 17          # 17 ticks past the second = 0.265625 s
    tr = track([(1, start, start + 5 * TR, True, 1)])
    at = PLANT_TICK + 12 * TR                  # a 1 Hz snapshot
    exp = (at - start) / TR                    # 1.734375 s — not an integer
    check("fractional elapsed", defuse_progress_features(tr, PLANT, at),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": exp,
           "defuse_progress_frac": exp / 5.0})
    assert abs(exp - round(exp)) > 0.1, "test is pointless if elapsed lands on an integer"


def test_kit_vs_nokit():
    start = PLANT_TICK + 10 * TR
    at = start + 4 * TR
    kit = track([(1, start, start + 5 * TR, True, 1)])
    nok = track([(1, start, start + 10 * TR, False, 1)])
    check("4s in, with kit", defuse_progress_features(kit, PLANT, at),
          {"defuse_elapsed_sec": 4.0, "defuse_progress_frac": 0.8, "defuse_beats_fuse": 1})
    check("4s in, no kit", defuse_progress_features(nok, PLANT, at),
          {"defuse_elapsed_sec": 4.0, "defuse_progress_frac": 0.4})


def test_interrupted_attempt():
    """A no-kit defuse killed at 4.9 s of the 10 s it needed (the real case we measured)."""
    start = PLANT_TICK + 10 * TR
    end = start + 314                          # 4.90625 s
    tr = track([(1, start, end, False, 0)])
    check("mid-attempt", defuse_progress_features(tr, PLANT, start + 3 * TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 3.0, "defuse_progress_frac": 0.3})
    check("at the last defusing tick", defuse_progress_features(tr, PLANT, end),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 314 / TR})
    check("after it was interrupted", defuse_progress_features(tr, PLANT, end + TR),
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0, "defuse_progress_frac": 0.0})


def test_two_attempts_one_round():
    a1, a2 = PLANT_TICK + 5 * TR, PLANT_TICK + 20 * TR
    tr = track([(1, a1, a1 + 2 * TR, False, 0), (1, a2, a2 + 5 * TR, True, 1)])
    check("before anything", defuse_progress_features(tr, PLANT, PLANT_TICK),
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0})
    check("during the 1st try", defuse_progress_features(tr, PLANT, a1 + TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 1.0, "defuse_progress_frac": 0.1})
    check("during the 2nd try", defuse_progress_features(tr, PLANT, a2 + TR),
          {"defuse_in_progress": 1, "defuse_elapsed_sec": 1.0, "defuse_progress_frac": 0.2})


def test_beats_fuse_flips_when_the_bomb_wins():
    """The 15-in-631 case that makes this column worth keeping: still defusing, fuse runs out."""
    late = PLANT_TICK + 33 * TR                  # 7 s of fuse left
    tr = track([(1, late, late + 10 * TR, False, 0)])   # no kit -> needs 10 s
    check("no kit, 7s of fuse left", defuse_progress_features(tr, PLANT, late),
          {"defuse_in_progress": 1, "defuse_beats_fuse": 0})
    early = PLANT_TICK + 10 * TR
    tr2 = track([(1, early, early + 10 * TR, False, 1)])
    check("same defuse, 30s of fuse left", defuse_progress_features(tr2, PLANT, early),
          {"defuse_in_progress": 1, "defuse_beats_fuse": 1})


def test_pre_plant_is_neutral():
    start = PLANT_TICK + 10 * TR
    tr = track([(1, start, start + 5 * TR, True, 1)])
    check("before the plant", defuse_progress_features(tr, PLANT, PLANT_TICK - TR),
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0})
    check("no plant at all", defuse_progress_features(tr, None, start + TR),
          {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0})


if __name__ == "__main__":
    for fn in (test_subsecond_precision, test_kit_vs_nokit, test_interrupted_attempt,
               test_two_attempts_one_round, test_beats_fuse_flips_when_the_bomb_wins,
               test_pre_plant_is_neutral):
        print(fn.__name__)
        fn()
    print("\nall defuse-progress tests passed")
