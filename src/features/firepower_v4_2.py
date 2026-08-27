"""Firepower v4.2 — rating mean x team-ranking weight (PROXY experiment).

    ct_rating_v42 = team_weight x (ct_rating_sum / ct_players_alive)
                  = team_weight x ct_rating_v41

with team_weight = 1 / log2(hltv_rank + 1), the same log2 formula as v3. Every alive
player on a side belongs to the same team, so the weight is one constant per side per
half, and this is exactly "v4.1 scaled by team strength": a rank-1 team's mean of 1.05
stays 1.05, a rank-30 team's 1.05 becomes ~0.21.

Halftime is handled the way v3 handles it: rounds 1-12 use the first-half CT/T
assignment, rounds 13+ swap. Weights come from eval_firepower_v3.build_match_weights,
which resolves each match's first-half CT team from round-1 ticks via
player_team_year.csv + team_rankings.csv (NOT configs/hltv_rankings.csv, which does not
exist -- the v4 notes name it wrongly).

Same divisor caveat as firepower_v4_1: this is sum/headcount, not mean skill.
"""
from __future__ import annotations

import polars as pl

from features.firepower_v4_1 import add_v41

FIREPOWER_V4_2_COLS = ["ct_rating_v42", "t_rating_v42"]
HALFTIME_ROUND = 13   # rounds >= this use the swapped side assignment


def add_v42(df: pl.DataFrame, match_weights: dict, formula: str = "log2") -> pl.DataFrame:
    """Attach v4.2 columns.

    match_weights: {match_id: {formula: {'h1_ct': float, 'h1_t': float}}}, as returned by
    models.eval_firepower_v3.build_match_weights.
    """
    if "ct_rating_v41" not in df.columns:
        df = add_v41(df)

    wt = pl.DataFrame({
        "match_id": list(match_weights.keys()),
        "_h1_ct": [match_weights[m][formula]["h1_ct"] for m in match_weights],
        "_h1_t": [match_weights[m][formula]["h1_t"] for m in match_weights],
    })
    out = df.join(wt, on="match_id", how="left")

    first_half = pl.col("round_num") < HALFTIME_ROUND
    ct_w = pl.when(first_half).then(pl.col("_h1_ct")).otherwise(pl.col("_h1_t"))
    t_w = pl.when(first_half).then(pl.col("_h1_t")).otherwise(pl.col("_h1_ct"))
    return out.with_columns([
        (pl.col("ct_rating_v41") * ct_w).alias("ct_rating_v42"),
        (pl.col("t_rating_v41") * t_w).alias("t_rating_v42"),
    ]).drop(["_h1_ct", "_h1_t"])
