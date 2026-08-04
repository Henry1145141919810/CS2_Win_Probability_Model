"""Pillar 3 — Firepower v2: per-player, side-aware skill features.

Loads player_stats_sided.csv, splits Rating / Firepower / Entrying /
Trading / Opening by CT-side vs T-side.
Conditional gates: clutch (lone survivor), entry/trading (has teammates),
opening (5v5 only), sniping (AWP holder), utility (grenade value × skill).

See src/features/firepower_v3.py for the team-ranking-weighted variant
(negative result, kept for reproducibility — docs/notes_firepower_v3.md).

Source data:
  configs/player_stats_sided.csv  — (steamid, year) -> stats
  configs/demo_year_map.csv       — demo_id -> year
"""
from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = ROOT / "configs" / "player_stats_sided.csv"
YEAR_MAP_PATH = ROOT / "configs" / "demo_year_map.csv"
DEFAULT_YEAR = 2024   # the one demo with unresolvable date (off-list qualifier)
SNIPING_THRESHOLD = 70  # >70 = full-time AWP specialist

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


def _year_lag() -> int:
    """Opt-in skill-prior year lag via FIREPOWER_YEAR_LAG env var.

    0 (default) = same-year stats. Set to 1 for the leak-free lagged-prior
    holdout: each 2026 match then looks up 2025 stats.
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
    """Compute v2 firepower features for one snapshot.

    snap: player rows at this tick (cols: side, health, steamid, inventory).
    match_id: this demo's id, used to resolve which year's stats apply.
    """
    year = year_for_match(match_id)
    lookup = _stats_lookup()

    ct_alive = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    t_alive = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))

    # Opening phase gate: both sides still at full starting headcount
    is_opening = (ct_alive.height == 5) and (t_alive.height == 5)

    out: dict = {}
    nan = float("nan")

    for side_str, alive in (("ct", ct_alive), ("t", t_alive)):
        n = alive.height
        sids = alive["steamid"].to_list()
        invs = (alive["inventory"].to_list()
                if "inventory" in alive.columns else [None] * n)
        pfx = side_str

        rating_sum = adr_sum = kast_sum = kast_n = fp_sum = 0.0
        entry_sum = trading_sum = opening_sum = weighted_util = 0.0
        clutch_score = nan
        awp_skill = nan

        for i, sid in enumerate(sids):
            stats = lookup.get((int(sid), year))
            if stats is None:
                continue

            teammates_alive = n - 1

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

            if teammates_alive >= 1:
                entry_sum += stats.get(f"entrying_{side_str}") or 0.0
                trading_sum += stats.get(f"trading_{side_str}") or 0.0

            if is_opening:
                opening_sum += stats.get(f"opening_{side_str}") or 0.0

            if teammates_alive == 0:
                cv = stats.get("clutching")
                clutch_score = float(cv) if cv is not None else nan

            inv = invs[i]
            if inv and "AWP" in inv:
                awp_skill = float(stats.get("sniping") or 0)

            weighted_util += (stats.get("utility") or 0) * _grenade_value(inv)

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
