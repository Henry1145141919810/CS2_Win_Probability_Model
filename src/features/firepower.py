"""Pillar 3 — Firepower: team-level skill weights from HLTV player stats.

Joins each alive player (by steamid) to their HLTV Rating 3.0 / ADR / KAST for the
YEAR the current match was played — player skill drifts year to year, so a 2024
match must use 2024 stats, not whatever HLTV shows today. Source data:
configs/player_stats_raw.csv (scraped per (steamid, year)) and
configs/demo_year_map.csv (match_id -> year).

Team aggregates are SUMMED over alive players, same logic as economy.py's equipment
total: more skilled players alive = more aggregate threat, not just higher average.
KAST is a per-player rate (not additive), so it's averaged instead.

Clutch: when a side is down to exactly one alive player (1vN, including 1v1), that
lone player's HLTV Clutching score (0-100) is exposed; NaN otherwise (same "neutral
default" convention as bomb.py's pre-plant fields). No separate is_clutch flag is
needed — a real Clutching score is never 0 (observed range ~21-92), so after the
pipeline's nan_to_num, "score > 0" already means "this side is in a clutch".
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "configs" / "player_stats_raw.csv"
YEAR_MAP_PATH = ROOT / "configs" / "demo_year_map.csv"
DEFAULT_YEAR = 2024  # the one demo with no resolvable date (off-list qualifier)


@lru_cache(maxsize=1)
def _stats_lookup() -> dict[tuple[int, int], dict]:
    df = pl.read_csv(STATS_PATH)
    return {(r["steamid"], r["year"]): r for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _year_by_demo() -> dict[str, int]:
    df = pl.read_csv(YEAR_MAP_PATH)
    return {r["demo_id"]: r["year"] for r in df.iter_rows(named=True)}


def year_for_match(match_id: str) -> int:
    return _year_by_demo().get(match_id) or DEFAULT_YEAR


def _side_firepower(alive: pl.DataFrame, year: int):
    """(rating_sum, adr_sum, kast_mean, clutch_score_if_lone_survivor)."""
    lookup = _stats_lookup()
    stats = [lookup.get((sid, year)) for sid in alive["steamid"].to_list()]
    stats = [s for s in stats if s is not None]
    if not stats:
        return 0.0, 0.0, float("nan"), float("nan")
    rating_sum = sum(s["rating"] for s in stats)
    adr_sum = sum(s["adr"] for s in stats)
    kast_mean = sum(s["kast"] for s in stats) / len(stats)
    clutch = stats[0]["clutching"] if alive.height == 1 else float("nan")
    return rating_sum, adr_sum, kast_mean, clutch


def firepower_features(snap: pl.DataFrame, match_id: str) -> dict:
    """Firepower features for one snapshot.

    snap: player rows at this tick (cols: side, health, steamid). match_id: this
    demo's id, used to resolve which year's HLTV stats apply.
    """
    year = year_for_match(match_id)
    ct = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    t = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))

    ct_rating, ct_adr, ct_kast, ct_clutch = _side_firepower(ct, year)
    t_rating, t_adr, t_kast, t_clutch = _side_firepower(t, year)

    return {
        "ct_firepower_rating": ct_rating,
        "t_firepower_rating": t_rating,
        "firepower_rating_diff": ct_rating - t_rating,
        "ct_firepower_adr": ct_adr,
        "t_firepower_adr": t_adr,
        "ct_firepower_kast": ct_kast,
        "t_firepower_kast": t_kast,
        "ct_clutch_score": ct_clutch,
        "t_clutch_score": t_clutch,
    }


FIREPOWER_COLS = [
    "ct_firepower_rating", "t_firepower_rating", "firepower_rating_diff",
    "ct_firepower_adr", "t_firepower_adr",
    "ct_firepower_kast", "t_firepower_kast",
    "ct_clutch_score", "t_clutch_score",
]
