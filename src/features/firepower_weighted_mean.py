"""Firepower rung (d) — mean-normalised stats scaled by the team's HLTV rank weight.

The ladder in section 7.5 fixes one defect per rung:

  (a) summed rating          the original encoding; r = 0.987 with the player-count advantage
  (b) mean rating            removes that confound
  (c) mean + situational     adds the conditional detail (entry/trading/opening/clutch gates)
  (d) mean + situational + team weight     <- this module

Rung (d) multiplies each mean by `1 / log2(hltv_rank + 1)` for the side that owns it, so a
rank-1 team's mean of 1.05 stays 1.05 while a rank-30 team's 1.05 reads as ~0.21. Every
alive player on a side belongs to the same team, so the weight is one constant per side per
half and the scaling is exact rather than approximate.

Halftime is handled the way firepower_v3 handles it: rounds 1-12 use the first-half CT/T
assignment, rounds 13+ swap. Weights come from eval_firepower_v3.build_match_weights, which
resolves each match's first-half CT team from its round-1 ticks via player_team_year.csv and
team_rankings.csv.

Only the 14 genuinely averaged columns are scaled. kast_mean, clutch_score and
awp_sniping_skill are per-player values that were never summed, so weighting them would
change what they mean rather than rescale it; they pass through.
"""
from __future__ import annotations

import polars as pl

from features.firepower import FIREPOWER_MEAN_COLS

HALFTIME_ROUND = 13
# per-player values that are not averages -- passed through unweighted
_PASS_THROUGH = {"ct_kast_mean", "t_kast_mean", "ct_clutch_score", "t_clutch_score",
                 "ct_awp_sniping_skill", "t_awp_sniping_skill"}

FIREPOWER_WMEAN_COLS = [c.replace("_mean", "_wmean") if c not in _PASS_THROUGH else f"{c}_w"
                        for c in FIREPOWER_MEAN_COLS]


def add_weighted_mean(df: pl.DataFrame, match_weights: dict,
                      formula: str = "log2") -> pl.DataFrame:
    """Attach rung (d): every mean column scaled by its side's team-rank weight.

    match_weights: {match_id: {formula: {'h1_ct': float, 'h1_t': float}}}, from
    models.eval_firepower_v3.build_match_weights.
    """
    wt = pl.DataFrame({
        "match_id": list(match_weights.keys()),
        "_h1_ct": [match_weights[m][formula]["h1_ct"] for m in match_weights],
        "_h1_t": [match_weights[m][formula]["h1_t"] for m in match_weights],
    })
    out = df.join(wt, on="match_id", how="left")
    first = pl.col("round_num") < HALFTIME_ROUND
    ct_w = pl.when(first).then(pl.col("_h1_ct")).otherwise(pl.col("_h1_t"))
    t_w = pl.when(first).then(pl.col("_h1_t")).otherwise(pl.col("_h1_ct"))

    exprs = []
    for c in FIREPOWER_MEAN_COLS:
        w = ct_w if c.startswith("ct_") else t_w
        if c in _PASS_THROUGH:
            exprs.append(pl.col(c).alias(f"{c}_w"))          # unweighted pass-through
        else:
            exprs.append((pl.col(c) * w).alias(c.replace("_mean", "_wmean")))
    return out.with_columns(exprs).drop(["_h1_ct", "_h1_t"])
