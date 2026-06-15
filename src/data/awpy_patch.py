"""Runtime patch for an awpy v2 bug that fails ~1/3 of pro demos.

Bug: `awpy.parsers.rounds.create_round_df` does `pl.col("winner").str.replace(...)`,
but on some demos awpy parses the `round_end` event's `winner` column as an INTEGER
team code (2 = TERRORIST, 3 = CT in CS2) instead of a string, raising
`InvalidOperationError: expected String type, got: i32` and aborting the whole parse.

This patch wraps `create_round_df`: if `round_end.winner` is not a string, it maps the
integer team codes to the strings awpy expects ("TERRORIST"/"CT") before delegating to
the original function. Idempotent; call `apply()` once before parsing.

Pinned to awpy 2.0.2. If awpy is upgraded and fixes this upstream, this becomes a no-op
(string winners are passed through untouched).
"""
from __future__ import annotations

import awpy.parsers.rounds as _rounds
import polars as pl

# CS2 team numbers as they appear in demo events.
_TEAM_CODE = {2: "TERRORIST", 3: "CT"}

_orig_create_round_df = _rounds.create_round_df
_applied = False


def _patched_create_round_df(events: dict[str, pl.DataFrame]) -> pl.DataFrame:
    re_ = events.get("round_end")
    if re_ is not None and "winner" in re_.columns and re_["winner"].dtype != pl.Utf8:
        events = dict(events)  # shallow copy; don't mutate caller's dict
        events["round_end"] = re_.with_columns(
            pl.col("winner")
            .cast(pl.Int64, strict=False)
            .replace_strict(_TEAM_CODE, default=None, return_dtype=pl.Utf8)
            .alias("winner")
        )
    return _orig_create_round_df(events)


def apply() -> None:
    """Install the patch (idempotent)."""
    global _applied
    if _applied:
        return
    _rounds.create_round_df = _patched_create_round_df
    _applied = True
