"""Firepower v4.1 — rating mean only, no ranking weight (PROXY experiment).

Derived from columns already in an assembled table, so assemble.py need not be re-run:

    ct_rating_v41 = ct_rating_sum / ct_players_alive

Everything except Rating is dropped, to isolate the rating signal from the 20-column
v2 block.

CAVEAT -- the divisor. This proxy divides by `players_alive`, whereas the integrated
version in firepower.py divides by `n_with_stats` (alive players actually present in the
HLTV table). They are NOT the same feature. Players missing from the table contribute 0
to the sum, so dividing by players_alive reports a low "skill mean" for a side that
simply has poor HLTV coverage. Measured on the 2026 set: 8.4% of snapshots have alive
players and rating_sum = 0, and the implied per-player rating has median 0.836 under this
divisor versus 1.086 under n_with_stats. Read v4.1/v4.2/v4.3 results as "sum / headcount",
not as "mean skill", and compare them to FIREPOWER_MEAN_COLS only with that in mind.
"""
from __future__ import annotations

import polars as pl

FIREPOWER_V4_1_COLS = ["ct_rating_v41", "t_rating_v41"]


def add_v41(df: pl.DataFrame) -> pl.DataFrame:
    """Attach the v4.1 columns to an assembled table."""
    out = df
    for side in ("ct", "t"):
        alive = pl.col(f"{side}_players_alive")
        out = out.with_columns(
            pl.when(alive > 0)
            .then(pl.col(f"{side}_rating_sum") / alive)
            .otherwise(None)
            .alias(f"{side}_rating_v41"))
    return out
