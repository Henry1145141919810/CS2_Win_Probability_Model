"""Defuse PROGRESS ("how many seconds into the defuse are we?") — Pillar 4b, v3.

Split out of `features/bomb.py` on purpose: unlike the defuse-RACE features this needs no
nav mesh, no LOS matrix and no awpy — only the bomb event stream (or the per-tick
`is_defusing` prop) and polars. That keeps it importable (and unit-testable) anywhere.
`features/bomb.py` re-exports everything here, so existing imports keep working.
"""
from __future__ import annotations

import polars as pl

TICKRATE = 64
BOMB_TIMER_SEC = 40.0    # CS2 C4 fuse
DEFUSE_KIT_SEC = 5.0     # defuse time with kit
DEFUSE_NOKIT_SEC = 10.0  # without kit

# ---------------------------------------------------------------------------
# Defuse PROGRESS ("how many seconds into the defuse are we?") — Pillar 4b, v3
# ---------------------------------------------------------------------------
# The v1/v2 defuse features are all COUNTERFACTUAL: they ask "could a CT get there and
# defuse in time?" from geometry. They say nothing about a defuse that is ACTUALLY
# happening. Once a CT is on the bomb, the round state is qualitatively different: the
# fuse and the defuse bar are both running, and every second of accumulated progress is
# a second the Ts must now win the duel within.
#
# We reconstruct each defuse ATTEMPT of a round as an interval and read progress off it:
#   - defuse_in_progress    : is someone defusing at this snapshot
#   - defuse_elapsed_sec    : seconds into the CURRENT attempt (0 if none)   <- the feature
#   - defuse_progress_frac  : elapsed / required, kit-aware (5s w/ kit, 10s without), in [0,1]
#   - defuse_beats_fuse     : 1 if the remaining defuse time fits inside the remaining fuse
#   - defuse_attempts_so_far: attempts STARTED up to this tick (>1 means an earlier one failed)
#
# All five default to 0 outside a defuse, which is the truthful value (no progress), so the
# nan_to_num(...) in train_pipeline does not create a phantom "about to finish" state.
#
# LEAKAGE WARNING — read before adding these to a headline feature set. A defuse that runs
# to completion IS the CT win; at elapsed=4.5s with a kit the label is nearly determined.
# These columns are therefore kept in their own set (see train_pipeline "EB3"/"EFB3") so the
# map-control results are never quoted from a model that contains them. The columns are only
# honest if the data also contains attempts that FAILED (started, then aborted or the defuser
# was killed) — otherwise "a defuse started" is a relabelling of `ct_won`. Run
# `src/data/inspect_bomb_events.py` to check that failed attempts exist in the bundle.

DEFUSE_START_EVENTS = {"begin_defuse", "begindefuse", "bomb_begindefuse",
                       "defuse_start", "start_defuse"}
DEFUSE_ABORT_EVENTS = {"abort_defuse", "abortdefuse", "bomb_abortdefuse", "defuse_abort"}
DEFUSE_DONE_EVENTS = {"defuse", "defused", "bomb_defused"}
DEFUSE_END_EVENTS = DEFUSE_ABORT_EVENTS | DEFUSE_DONE_EVENTS | {"detonate", "explode"}


class DefuseTracker:
    """Every defuse attempt of one round as (start_tick, end_tick, steamid, completed).

    Two sources, in order of preference:
      1. bomb events — a begin_defuse/abort_defuse pair (exact, tick-accurate);
      2. the per-tick `is_defusing` player prop, if the ticks table carries it — consecutive
         True samples per player form an attempt. NOTE: ticks are downsampled to ~1 Hz, so a
         start recovered this way is accurate only to +/-1s and a defuse shorter than the
         sampling interval can be missed entirely.

    If neither source exists the tracker is empty and every feature stays 0 — we deliberately
    do NOT back-derive a start from a completed defuse (that would only ever see successful
    defuses, i.e. the label).
    """

    def __init__(self, bomb_df_round=None, ticks_round=None):
        self.attempts: list[dict] = []
        self.source = "none"
        if bomb_df_round is not None and bomb_df_round.height:
            self.attempts = self._from_events(bomb_df_round)
            if self.attempts:
                self.source = "events"
        if not self.attempts and ticks_round is not None:
            self.attempts = self._from_ticks(ticks_round)
            if self.attempts:
                self.source = "ticks"
        self.attempts.sort(key=lambda a: a["start"])

    @staticmethod
    def _from_events(bomb_df_round) -> list[dict]:
        ev = sorted(({"tick": r["tick"], "event": r["event"],
                      "steamid": r.get("steamid")} for r in bomb_df_round.iter_rows(named=True)),
                    key=lambda e: e["tick"])
        out, open_att = [], None
        for e in ev:
            name = str(e["event"]).lower()
            if name in DEFUSE_START_EVENTS:
                if open_att is not None:            # a new start closes the previous attempt
                    open_att["end"] = e["tick"]
                    out.append(open_att)
                open_att = {"start": e["tick"], "end": None,
                            "steamid": e["steamid"], "completed": False}
            elif name in DEFUSE_END_EVENTS and open_att is not None:
                open_att["end"] = e["tick"]
                open_att["completed"] = name in DEFUSE_DONE_EVENTS
                out.append(open_att)
                open_att = None
        if open_att is not None:                    # never closed (defuser killed / round end)
            out.append(open_att)
        return out

    @staticmethod
    def _from_ticks(ticks_round) -> list[dict]:
        cols = ticks_round.columns
        col = next((c for c in ("is_defusing", "is_defuse", "defusing") if c in cols), None)
        if col is None:
            return []
        df = ticks_round.filter(pl.col(col).cast(pl.Boolean, strict=False)).select(
            ["tick", "steamid"]).sort(["steamid", "tick"])
        out, cur = [], None
        prev_sid = prev_tick = None
        for r in df.iter_rows(named=True):
            sid, tk = r["steamid"], r["tick"]
            new_run = (cur is None or sid != prev_sid
                       or tk - prev_tick > 2 * TICKRATE)  # >2s gap = a separate attempt
            if new_run:
                if cur is not None:
                    out.append(cur)
                cur = {"start": tk, "end": None, "steamid": sid, "completed": False}
            cur["end"] = tk + 1   # +1 tick so the last defusing SAMPLE counts as in-progress
            prev_sid, prev_tick = sid, tk
        if cur is not None:
            out.append(cur)
        return out

    def active_at(self, tick: int) -> dict | None:
        for a in self.attempts:
            if a["start"] <= tick and (a["end"] is None or tick < a["end"]):
                return a
        return None

    def n_started_by(self, tick: int) -> int:
        return sum(1 for a in self.attempts if a["start"] <= tick)


def defuse_progress_features(snap, tracker: "DefuseTracker", plant: dict | None,
                             tick: int) -> dict:
    """How far into an in-flight defuse this snapshot is (0s everywhere else)."""
    base = {"defuse_in_progress": 0, "defuse_elapsed_sec": 0.0,
            "defuse_progress_frac": 0.0, "defuse_beats_fuse": 0,
            "defuse_attempts_so_far": 0}
    if plant is None or tick < plant["tick"] or tracker is None:
        return base
    base["defuse_attempts_so_far"] = tracker.n_started_by(tick)
    att = tracker.active_at(tick)
    if att is None:
        return base
    elapsed = max(0.0, (tick - att["start"]) / TICKRATE)
    # An attempt with no closing event (defuser killed and the demo emits no abort) would
    # otherwise stay "active, 100% done" for the rest of the round — a pure phantom. Close it
    # if the defuser is dead, or if it has outlived the longest possible defuse.
    if att["end"] is None and elapsed > DEFUSE_NOKIT_SEC:
        return base
    # kit-awareness: the defuser's own kit sets the bar length (5s vs 10s)
    kit = 0
    if att["steamid"] is not None and "steamid" in snap.columns:
        who = snap.filter(pl.col("steamid") == att["steamid"])
        if who.height:
            if who["health"][0] <= 0:
                return base
            if "has_defuser" in who.columns:
                kit = int(bool(who["has_defuser"][0]))
    required = DEFUSE_KIT_SEC if kit else DEFUSE_NOKIT_SEC
    remaining = max(0.0, required - elapsed)
    fuse_left = BOMB_TIMER_SEC - (tick - plant["tick"]) / TICKRATE
    return {
        "defuse_in_progress": 1,
        "defuse_elapsed_sec": float(min(elapsed, required)),
        "defuse_progress_frac": float(min(1.0, elapsed / required)),
        "defuse_beats_fuse": int(remaining <= fuse_left),
        "defuse_attempts_so_far": base["defuse_attempts_so_far"],
    }


BOMB_PROGRESS_COLS = [
    "defuse_in_progress", "defuse_elapsed_sec", "defuse_progress_frac",
    "defuse_beats_fuse", "defuse_attempts_so_far",
]
