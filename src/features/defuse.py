"""Defuse PROGRESS ("how many seconds into the defuse are we?") — Pillar 4b.

Reads the `defuse` channel that batch_parse derives from the FULL-RESOLUTION tick stream
(one row per defuse ATTEMPT: round_num, steamid, start_tick, end_tick, had_kit, completed).

Why a channel and not a tick column: the project's time convention is uniform — every time
feature is `(snapshot_tick - reference_tick) / 64` against a reference kept in a table that
is NOT downsampled (`rounds.freeze_end` in economy.py, the plant tick in bomb.py). Ticks are
saved at 1 Hz, so an is_defusing column alone would only ever yield integer seconds with a
+/-1 s error on the start. The defuse channel is that non-downsampled reference table, so
`defuse_elapsed_sec` lands at the same 1/64 s precision as `time_elapsed_sec` and
`defuse_time_margin`.

Per snapshot (post-plant, while an attempt is live; 0 everywhere else):
  - defuse_in_progress     : is someone on the bomb right now
  - defuse_elapsed_sec     : seconds into the CURRENT attempt          <- the feature
  - defuse_progress_frac   : elapsed / required, kit-aware, in [0,1]
  - defuse_beats_fuse      : will the remaining defuse time fit inside the remaining fuse

`defuse_beats_fuse` looks redundant next to `defuse_in_progress` -- it is 1 on 616 of 631
defusing rows -- but the 15 rows where it is 0 are the ones that decide the round: the
defuse is running and the bomb explodes first. Measured, it is the single most valuable of
these columns: on the pilot it takes log-loss over defusing rows from 0.1727 to 0.0605.

It is also not learnable from what the model has. It compares the remaining defuse time
against the remaining FUSE time, and "seconds since the plant" is in no feature set --
`time_elapsed_sec` counts from freeze-end and `defuse_time_margin` folds the fuse into a
distance term. Supplying the raw quantity does not help either: giving logreg a
`fuse_time_left` column instead moved log-loss only 0.1727 -> 0.1719, because a linear
model cannot express a threshold comparison at all. Hand-building it is the standard remedy
for a GLM and is what the project already does for `ctrl_x_eveneco` and friends.

A fifth column, `defuse_attempts_so_far`, was dropped: 0 on 98.6% of rows, and adding it
made log-loss WORSE (0.1727 -> 0.1905).

All four are 0 outside an attempt, which is the truthful value (no progress), so the
`nan_to_num` in train_pipeline cannot invent an "about to finish" state.

LEAKAGE NOTE. A defuse that runs to completion IS the CT win, so these columns are kept in
their own feature sets (EB3/EFB3) and never fold into the map-control headline numbers. They
are honest only because the data also contains attempts that FAILED — measured on a real
demo: 5 attempts, 4 completed, 1 interrupted (a no-kit defuse killed at 4.91 s of the 10 s
it needed). `completed=0` rows are exactly the non-leaky ones.
"""
from __future__ import annotations

import polars as pl

TICKRATE = 64
BOMB_TIMER_SEC = 40.0    # CS2 C4 fuse
DEFUSE_KIT_SEC = 5.0     # defuse time with kit   (verified: exactly 320 ticks)
DEFUSE_NOKIT_SEC = 10.0  # without kit

DEFUSE_COLS_IN = ["round_num", "steamid", "start_tick", "end_tick", "had_kit", "completed"]


def attempts_by_round(defuse_df: pl.DataFrame | None) -> dict[int, list[dict]]:
    """round_num -> attempts, from a demo's `defuse` channel. {} if the channel is absent.

    No fallback to the 1 Hz `is_defusing` column on purpose: it would silently degrade the
    feature to integer seconds, and a silent precision drop is worse than a visible zero.
    """
    if defuse_df is None or defuse_df.height == 0:
        return {}
    out: dict[int, list[dict]] = {}
    for r in defuse_df.sort("start_tick").iter_rows(named=True):
        out.setdefault(int(r["round_num"]), []).append({
            "start": int(r["start_tick"]),
            "end": int(r["end_tick"]),
            "steamid": r.get("steamid"),
            "kit": bool(r.get("had_kit")),
            "completed": bool(r.get("completed")),
        })
    return out


class DefuseTracker:
    """The defuse attempts of ONE round, queried by tick."""

    def __init__(self, attempts: list[dict] | None = None):
        self.attempts = sorted(attempts or [], key=lambda a: a["start"])

    def active_at(self, tick: int) -> dict | None:
        for a in self.attempts:
            if a["start"] <= tick <= a["end"]:
                return a
        return None


def defuse_progress_features(tracker: DefuseTracker | None, plant: dict | None,
                             tick: int) -> dict:
    """How far into an in-flight defuse this snapshot is (0 everywhere else)."""
    base = {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0,
            "defuse_progress_frac": 0.0, "defuse_beats_fuse": 0}
    if tracker is None or plant is None or tick < plant["tick"]:
        return base
    att = tracker.active_at(tick)
    if att is None:
        return base

    # same shape as economy.py / bomb.py: (snapshot tick - reference tick) / 64
    elapsed = (tick - att["start"]) / TICKRATE
    required = DEFUSE_KIT_SEC if att["kit"] else DEFUSE_NOKIT_SEC
    fuse_left = BOMB_TIMER_SEC - (tick - plant["tick"]) / TICKRATE
    return {
        "defuse_in_progress": 1,
        "defuse_elapsed_sec": float(min(elapsed, required)),
        "defuse_progress_frac": float(min(1.0, elapsed / required)),
        "defuse_beats_fuse": int(max(0.0, required - elapsed) <= fuse_left),
    }


BOMB_PROGRESS_COLS = [
    "defuse_in_progress", "defuse_elapsed_sec", "defuse_progress_frac",
    "defuse_beats_fuse",
]
