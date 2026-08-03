"""Pillar 3 — Firepower v2 + v3: per-player, side-aware skill features.

v2 (unchanged):
- Loads player_stats_sided.csv, splits Rating / Firepower / Entrying /
  Trading / Opening by CT-side vs T-side.
- Conditional gates: clutch (lone survivor), entry/trading (has teammates),
  opening (5v5 only), sniping (AWP holder), utility (grenade value × skill).

v3 (additive):
- All v2 stats are re-computed with a per-player team-ranking weight:
    weight = 1 / log2(hltv_rank + 1)   [rank 1 → 1.0, rank 30 → 0.20]
- Team rank is resolved via player_team_year.csv (steamid,year → team)
  and team_rankings.csv (team,year → weight).
- Players with no team info get the DEFAULT_WEIGHT (rank 35 equivalent).
- v3 columns have the suffix _v3; v2 columns are untouched.

Source data:
  configs/player_stats_sided.csv  — (steamid, year) → stats
  configs/player_team_year.csv    — (steamid, year) → team_canonical
  configs/team_rankings.csv       — (team_canonical, year) → weight
  configs/demo_year_map.csv       — demo_id → year
"""
from __future__ import annotations
import math
import os
from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "configs" / "player_stats_sided.csv"
YEAR_MAP_PATH = ROOT / "configs" / "demo_year_map.csv"
PLAYER_TEAM_PATH = ROOT / "configs" / "player_team_year.csv"
TEAM_RANK_PATH = ROOT / "configs" / "team_rankings.csv"
DEFAULT_YEAR = 2024   # the one demo with unresolvable date (off-list qualifier)
SNIPING_THRESHOLD = 70  # >70 = full-time AWP specialist
# weight for players whose team is not found in team_rankings (rank~35)
DEFAULT_WEIGHT = round(1.0 / math.log2(36), 4)  # ≈ 0.1934

GRENADE_PRICES: dict[str, int] = {
    "Smoke Grenade": 300,
    "Flashbang": 200,
    "Molotov": 400,
    "Incendiary Grenade": 600,
    "High Explosive Grenade": 300,
}


@lru_cache(maxsize=1)
def _stats_lookup() -> dict[tuple[int, int], dict]:
    df = pl.read_csv(STATS_PATH)
    return {(r["steamid"], r["year"]): r for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _year_by_demo() -> dict[str, int]:
    df = pl.read_csv(YEAR_MAP_PATH)
    return {r["demo_id"]: r["year"] for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _player_team_lookup() -> dict[tuple[int, int], str]:
    """(steamid, year) → team_canonical"""
    df = pl.read_csv(PLAYER_TEAM_PATH)
    return {(r["steamid"], r["year"]): r["team_canonical"]
            for r in df.iter_rows(named=True)}


@lru_cache(maxsize=1)
def _team_weight_lookup() -> dict[tuple[str, int], float]:
    """(team_canonical, year) → rank weight (1/log2(rank+1))"""
    df = pl.read_csv(TEAM_RANK_PATH)
    return {(r["team_canonical"], r["year"]): float(r["weight"])
            for r in df.iter_rows(named=True)}


def _year_lag() -> int:
    """Opt-in skill-prior year lag, read from the env var FIREPOWER_YEAR_LAG.

    0 (default) = same-year stats, exactly as before — training builds are
    unaffected unless the caller deliberately sets the var.

    Set FIREPOWER_YEAR_LAG=1 when assembling the 2026 out-of-time holdout to
    build the LEAK-FREE LAGGED-PRIOR variant: each 2026 match then looks up
    the previous season's (2025) stats, i.e. only information a live system
    would actually have at match time. See docs/TODO_leu_2026_scrape.md §2.
    """
    try:
        return int(os.environ.get("FIREPOWER_YEAR_LAG", "0"))
    except ValueError:
        return 0


def year_for_match(match_id: str) -> int:
    base = _year_by_demo().get(match_id) or DEFAULT_YEAR
    return base - _year_lag()


def _grenade_value(inventory) -> int:
    """Dollar value of grenades in a player's inventory list."""
    if not inventory:
        return 0
    return sum(GRENADE_PRICES.get(item, 0) for item in inventory)


def firepower_features(snap: pl.DataFrame, match_id: str) -> dict:
    """Compute all firepower features for one snapshot.

    snap: player rows at this tick (cols: side, health, steamid, inventory).
    match_id: this demo's id, used to resolve which year's stats apply.
    """
    year = year_for_match(match_id)
    lookup = _stats_lookup()
    player_team_lut = _player_team_lookup()
    team_weight_lut = _team_weight_lookup()

    ct_alive = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    t_alive = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))

    # Opening phase gate: both sides still at full starting headcount
    is_opening = (ct_alive.height == 5) and (t_alive.height == 5)

    out: dict = {}
    nan = float("nan")

    for side_str, alive in (("ct", ct_alive), ("t", t_alive)):
        n = alive.height
        sids = alive["steamid"].to_list()
        invs = alive["inventory"].to_list() if "inventory" in alive.columns else [None] * n
        pfx = side_str

        # Accumulators — v2
        rating_sum = adr_sum = kast_sum = kast_n = fp_sum = 0.0
        entry_sum = trading_sum = opening_sum = weighted_util = 0.0
        clutch_score = nan
        awp_skill = nan
        # Accumulators — v3 (rank-weighted)
        rating_v3 = adr_v3 = kast_v3_sum = kast_v3_n = fp_v3 = 0.0
        entry_v3 = trading_v3 = opening_v3 = util_v3 = 0.0
        clutch_v3 = nan

        for i, sid in enumerate(sids):
            stats = lookup.get((int(sid), year))
            if stats is None:
                continue

            teammates_alive = n - 1  # other alive teammates on this side

            # ── resolve v3 rank weight for this player ────────────────────
            team = player_team_lut.get((int(sid), year))
            rw = team_weight_lut.get((team, year), DEFAULT_WEIGHT) \
                if team else DEFAULT_WEIGHT

            # ── always-active, side-specific metrics ─────────────────────
            rating_val = stats.get(f"rating_{side_str}") or 0.0
            adr_val = stats.get("adr") or 0.0
            fp_val = stats.get(f"firepower_{side_str}") or 0.0
            rating_sum += rating_val
            adr_sum += adr_val
            kv = stats.get("kast")
            if kv is not None:
                kast_sum += kv
                kast_n += 1
            fp_sum += fp_val
            # v3
            rating_v3 += rating_val * rw
            adr_v3 += adr_val * rw
            if kv is not None:
                kast_v3_sum += kv * rw
                kast_v3_n += 1
            fp_v3 += fp_val * rw

            # ── conditional: only when this player has living teammates ───
            if teammates_alive >= 1:
                ev = stats.get(f"entrying_{side_str}") or 0.0
                tv = stats.get(f"trading_{side_str}") or 0.0
                entry_sum += ev
                trading_sum += tv
                entry_v3 += ev * rw
                trading_v3 += tv * rw

            # ── conditional: opening phase (no kills yet this round) ──────
            if is_opening:
                ov = stats.get(f"opening_{side_str}") or 0.0
                opening_sum += ov
                opening_v3 += ov * rw

            # ── conditional: lone survivor gets Clutching score ───────────
            if teammates_alive == 0:
                cv = stats.get("clutching")
                clutch_score = float(cv) if cv is not None else nan
                clutch_v3 = float(cv) * rw if cv is not None else nan

            # ── Sniping: role flag — who's holding the AWP? ──────────────
            inv = invs[i]
            if inv and "AWP" in inv:
                sv = stats.get("sniping") or 0
                awp_skill = float(sv)

            # ── Utility skill × current grenade dollar value ──────────────
            util_skill = stats.get("utility") or 0
            ug = util_skill * _grenade_value(inv)
            weighted_util += ug
            util_v3 += ug * rw

        out[f"{pfx}_rating_sum"] = rating_sum
        out[f"{pfx}_adr_sum"] = adr_sum
        out[f"{pfx}_kast_mean"] = kast_sum / kast_n if kast_n else nan
        out[f"{pfx}_hltv_firepower_sum"] = fp_sum
        out[f"{pfx}_entry_sum"] = entry_sum
        out[f"{pfx}_trading_sum"] = trading_sum
        out[f"{pfx}_opening_sum"] = opening_sum if is_opening else nan
        out[f"{pfx}_clutch_score"] = clutch_score
        out[f"{pfx}_awp_sniping_skill"] = awp_skill
        out[f"{pfx}_weighted_utility"] = weighted_util
        # v3 rank-weighted outputs
        out[f"{pfx}_rating_v3"] = rating_v3
        out[f"{pfx}_adr_v3"] = adr_v3
        out[f"{pfx}_kast_v3"] = kast_v3_sum / kast_v3_n if kast_v3_n else nan
        out[f"{pfx}_hltv_fp_v3"] = fp_v3
        out[f"{pfx}_entry_v3"] = entry_v3
        out[f"{pfx}_trading_v3"] = trading_v3
        out[f"{pfx}_opening_v3"] = opening_v3 if is_opening else nan
        out[f"{pfx}_clutch_v3"] = clutch_v3
        out[f"{pfx}_util_v3"] = util_v3

    return out


FIREPOWER_COLS = [
    "ct_rating_sum", "t_rating_sum",
    "ct_adr_sum", "t_adr_sum",
    "ct_kast_mean", "t_kast_mean",
    "ct_hltv_firepower_sum", "t_hltv_firepower_sum",
    "ct_entry_sum", "t_entry_sum",
    "ct_trading_sum", "t_trading_sum",
    "ct_opening_sum", "t_opening_sum",
    "ct_clutch_score", "t_clutch_score",
    "ct_awp_sniping_skill", "t_awp_sniping_skill",
    "ct_weighted_utility", "t_weighted_utility",
]

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
