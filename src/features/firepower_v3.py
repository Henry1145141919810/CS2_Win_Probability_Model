"""Firepower v3: team-ranking-weighted skill features (negative result).

All v2 stats re-computed with a per-player team-ranking weight:
    weight = 1 / log2(hltv_rank + 1)   [rank 1 -> 1.0, rank 30 -> 0.20]

Team rank resolved via:
  configs/player_team_year.csv  -- (steamid, year) -> team_canonical
  configs/team_rankings.csv     -- (team_canonical, year) -> weight

Players with no team info get DEFAULT_WEIGHT (rank-35 equivalent ~0.19).
AWP sniping skill is not included (it is individual, not a team sum).

Result: all three weighting formulas score below EFB2 (v2 raw sum).
See docs/notes_firepower_v3.md for full analysis.
"""
from __future__ import annotations
import math
from functools import lru_cache
from pathlib import Path

import polars as pl

from features.firepower import (  # shared infrastructure
    _stats_lookup, year_for_match, _grenade_value,
)

ROOT = Path(__file__).resolve().parents[2]
PLAYER_TEAM_PATH = ROOT / "configs" / "player_team_year.csv"
TEAM_RANK_PATH = ROOT / "configs" / "team_rankings.csv"
DEFAULT_WEIGHT = round(1.0 / math.log2(36), 4)  # rank-35 equivalent ~0.1934


@lru_cache(maxsize=1)
def _player_team_lookup() -> dict[tuple[int, int], str]:
    """(steamid, year) -> team_canonical"""
    df = pl.read_csv(PLAYER_TEAM_PATH)
    return {(r["steamid"], r["year"]): r["team_canonical"]
            for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _team_weight_lookup() -> dict[tuple[str, int], float]:
    """(team_canonical, year) -> 1/log2(rank+1) weight"""
    df = pl.read_csv(TEAM_RANK_PATH)
    return {(r["team_canonical"], r["year"]): float(r["weight"])
            for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _team_rank_lookup() -> dict[tuple[str, int], int]:
    """(team_canonical, year) -> hltv_rank (for alternative formulas)"""
    df = pl.read_csv(TEAM_RANK_PATH)
    return {(r["team_canonical"], r["year"]): int(r["hltv_rank"])
            for r in df.iter_rows(named=True)}


def _resolve_weight(sid: int, year: int, formula: str = "log2") -> float:
    """Return rank weight for a player given a weighting formula.

    formula: 'log2' (1/log2(rank+1)), 'inv' (1/rank), 'linear' ((31-rank)/30)
    """
    team = _player_team_lookup().get((sid, year))
    if team is None:
        rank = 35
    else:
        rank = _team_rank_lookup().get((team, year), 35)
    rank = max(1, rank)
    if formula == "log2":
        return 1.0 / math.log2(rank + 1)
    if formula == "inv":
        return 1.0 / rank
    # linear
    return max(0.0, (31 - rank) / 30.0)


def firepower_features_v3(
    snap: pl.DataFrame, match_id: str, formula: str = "log2"
) -> dict:
    """Compute v3 rank-weighted firepower features for one snapshot.

    snap: player rows at this tick (cols: side, health, steamid, inventory).
    match_id: this demo's id, used to resolve which year's stats apply.
    formula: weighting scheme -- 'log2', 'inv', or 'linear'.
    Returns 18 columns with _v3 suffix.
    """
    year = year_for_match(match_id)
    lookup = _stats_lookup()

    ct_alive = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    t_alive = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))
    is_opening = (ct_alive.height == 5) and (t_alive.height == 5)

    out: dict = {}
    nan = float("nan")

    for side_str, alive in (("ct", ct_alive), ("t", t_alive)):
        n = alive.height
        sids = alive["steamid"].to_list()
        invs = (alive["inventory"].to_list()
                if "inventory" in alive.columns else [None] * n)
        pfx = side_str

        rating_v3 = adr_v3 = kast_sum = kast_n = fp_v3 = 0.0
        entry_v3 = trading_v3 = opening_v3 = util_v3 = 0.0
        clutch_v3 = nan

        for i, sid in enumerate(sids):
            stats = lookup.get((int(sid), year))
            if stats is None:
                continue

            teammates_alive = n - 1
            rw = _resolve_weight(int(sid), year, formula)

            rating_val = stats.get(f"rating_{side_str}") or 0.0
            adr_val = stats.get("adr") or 0.0
            fp_val = stats.get(f"firepower_{side_str}") or 0.0
            kv = stats.get("kast")

            rating_v3 += rating_val * rw
            adr_v3 += adr_val * rw
            if kv is not None:
                kast_sum += kv * rw
                kast_n += 1
            fp_v3 += fp_val * rw

            if teammates_alive >= 1:
                entry_v3 += (stats.get(f"entrying_{side_str}") or 0.0) * rw
                trading_v3 += (stats.get(f"trading_{side_str}") or 0.0) * rw

            if is_opening:
                opening_v3 += (stats.get(f"opening_{side_str}") or 0.0) * rw

            if teammates_alive == 0:
                cv = stats.get("clutching")
                clutch_v3 = float(cv) * rw if cv is not None else nan

            util_v3 += (stats.get("utility") or 0) * _grenade_value(invs[i]) * rw

        out[f"{pfx}_rating_v3"] = rating_v3
        out[f"{pfx}_adr_v3"] = adr_v3
        out[f"{pfx}_kast_v3"] = kast_sum / kast_n if kast_n else nan
        out[f"{pfx}_hltv_fp_v3"] = fp_v3
        out[f"{pfx}_entry_v3"] = entry_v3
        out[f"{pfx}_trading_v3"] = trading_v3
        out[f"{pfx}_opening_v3"] = opening_v3 if is_opening else nan
        out[f"{pfx}_clutch_v3"] = clutch_v3
        out[f"{pfx}_util_v3"] = util_v3

    return out


FIREPOWER_COLS_V3 = [
    "ct_rating_v3", "t_rating_v3",
    "ct_adr_v3", "t_adr_v3",
    "ct_kast_v3", "t_kast_v3",
    "ct_hltv_fp_v3", "t_hltv_fp_v3",
    "ct_entry_v3", "t_entry_v3",
    "ct_trading_v3", "t_trading_v3",
    "ct_opening_v3", "t_opening_v3",
    "ct_clutch_v3", "t_clutch_v3",
    "ct_util_v3", "t_util_v3",
]
