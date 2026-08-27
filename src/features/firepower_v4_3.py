"""Firepower v4.3 — every v2 sum as a mean, situational gates preserved (PROXY).

The 14 summed v2 columns are divided by players_alive; the 6 columns that are already
per-player values (kast_mean, clutch_score, awp_sniping_skill) pass through untouched.

Gates are preserved by construction: entry/trading are 0 for a lone survivor because
nothing was accumulated, and opening is NaN outside 5v5 for the same reason.

Same divisor caveat as firepower_v4_1 -- this is sum/headcount, not mean skill. See that
module's docstring.
"""
from __future__ import annotations

import polars as pl

# v2 column -> v4.3 column, for the ones that get divided
_DIVIDE = {
    "rating_sum": "rating_v43",
    "adr_sum": "adr_v43",
    "hltv_firepower_sum": "firepower_v43",
    "entry_sum": "entry_v43",
    "trading_sum": "trading_v43",
    "opening_sum": "opening_v43",
    "weighted_utility": "utility_v43",
}
# already per-player: renamed only, never divided
_PASS = {
    "kast_mean": "kast_v43",
    "clutch_score": "clutch_v43",
    "awp_sniping_skill": "awp_v43",
}

FIREPOWER_V4_3_COLS = (
    [f"{s}_{v}" for s in ("ct", "t") for v in _DIVIDE.values()]
    + [f"{s}_{v}" for s in ("ct", "t") for v in _PASS.values()])


def add_v43(df: pl.DataFrame) -> pl.DataFrame:
    out = df
    for side in ("ct", "t"):
        alive = pl.col(f"{side}_players_alive")
        exprs = [
            pl.when(alive > 0).then(pl.col(f"{side}_{src}") / alive).otherwise(None)
              .alias(f"{side}_{dst}")
            for src, dst in _DIVIDE.items()
        ]
        exprs += [pl.col(f"{side}_{src}").alias(f"{side}_{dst}") for src, dst in _PASS.items()]
        out = out.with_columns(exprs)
    return out
