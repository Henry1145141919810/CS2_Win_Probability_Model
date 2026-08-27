"""Firepower v1 — the original encoding, recomputed for the version comparison.

v1 (commit b27c7ec) predates side-aware stats and situational gating: it reads
configs/player_stats_raw.csv (one un-split rating/adr/kast/clutching per player-year) and
emits nine columns. It was replaced by v2 before the training table was ever assembled with
it, so those columns exist in no saved table -- they have to be recomputed to put v1 in the
same benchmark as v2/v3/v4.

Semantics preserved exactly from the original `_side_firepower`:
  - rating/adr are SUMS over alive players that have a stats row;
  - kast is the MEAN over those same players;
  - clutch is that player's `clutching` only when the side has exactly ONE alive player
    (counted over all alive players, not only those with stats), else NaN;
  - a side with no alive player carrying stats gets 0.0 sums and NaN kast/clutch.

Columns are suffixed `_v1` because v1's `ct_clutch_score` collides with v2's column of the
same name and different meaning.

This is a vectorised reimplementation (group-by rather than per-snapshot Python), verified
row-for-row against the original function.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STATS_RAW = ROOT / "configs" / "player_stats_raw.csv"

FIREPOWER_V1_COLS = [
    "ct_fp_rating_v1", "t_fp_rating_v1", "fp_rating_diff_v1",
    "ct_fp_adr_v1", "t_fp_adr_v1",
    "ct_fp_kast_v1", "t_fp_kast_v1",
    "ct_clutch_v1", "t_clutch_v1",
]


@lru_cache(maxsize=1)
def _stats() -> pl.DataFrame:
    return (pl.read_csv(STATS_RAW)
            .select(["steamid", "year", "rating", "adr", "kast", "clutching"])
            .unique(subset=["steamid", "year"]))


def v1_for_demo(ticks: pl.DataFrame, year: int) -> pl.DataFrame:
    """(tick, 9 v1 columns) for one demo's tick table."""
    alive = (ticks.filter(pl.col("health") > 0)
                  .select(["tick", "steamid", "side"])
                  .with_columns(pl.col("steamid").cast(pl.Int64)))
    st = _stats().filter(pl.col("year") == year).drop("year")
    j = alive.join(st, on="steamid", how="left")

    g = j.group_by(["tick", "side"]).agg([
        pl.len().alias("n_alive"),
        pl.col("rating").drop_nulls().len().alias("n_stats"),
        pl.col("rating").sum().alias("rating_sum"),
        pl.col("adr").sum().alias("adr_sum"),
        pl.col("kast").mean().alias("kast_mean"),
        pl.col("clutching").drop_nulls().first().alias("clutch_first"),
    ])
    g = g.with_columns([
        pl.when(pl.col("n_stats") > 0).then(pl.col("rating_sum")).otherwise(0.0).alias("rating_sum"),
        pl.when(pl.col("n_stats") > 0).then(pl.col("adr_sum")).otherwise(0.0).alias("adr_sum"),
        pl.when(pl.col("n_stats") > 0).then(pl.col("kast_mean")).otherwise(None).alias("kast_mean"),
        pl.when((pl.col("n_alive") == 1) & (pl.col("n_stats") > 0))
          .then(pl.col("clutch_first")).otherwise(None).alias("clutch"),
    ])

    # Base on every tick in the demo, not on the group-by result: a tick where a side has
    # nobody alive produces no group row, and one where NEITHER side does would vanish
    # entirely. The original returns 0.0 sums and NaN kast/clutch for an empty side.
    out = ticks.select("tick").unique()
    for side, pfx in (("ct", "ct"), ("t", "t")):
        s = (g.filter(pl.col("side") == side)
              .select(["tick",
                       pl.col("rating_sum").alias(f"{pfx}_fp_rating_v1"),
                       pl.col("adr_sum").alias(f"{pfx}_fp_adr_v1"),
                       pl.col("kast_mean").alias(f"{pfx}_fp_kast_v1"),
                       pl.col("clutch").alias(f"{pfx}_clutch_v1")]))
        out = out.join(s, on="tick", how="left")
    # a side wiped out has no group row at all -> sums are 0.0, kast/clutch stay null
    out = out.with_columns([
        pl.col("ct_fp_rating_v1").fill_null(0.0), pl.col("t_fp_rating_v1").fill_null(0.0),
        pl.col("ct_fp_adr_v1").fill_null(0.0), pl.col("t_fp_adr_v1").fill_null(0.0),
    ])
    return out.with_columns(
        (pl.col("ct_fp_rating_v1") - pl.col("t_fp_rating_v1")).alias("fp_rating_diff_v1")
    ).sort("tick")
