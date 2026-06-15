"""Pillar 4 — Tactical readiness: callout-based positional structure + utility.

All features are computed per snapshot from the per-tick `place` (callout) and
`inventory` columns we already parse — no extra data needed.

Positional structure (from `place`):
  - {side}_positional_entropy : Shannon entropy (bits) of ALIVE players over callouts.
        low = stacked on one spot, high = spread across the map.
  - {side}_{zone}_players      : how many alive players are committed to each named zone
        (a_site, b_site, banana, mid, ct_spawn).
  - {side}_awp_alive / _awp_zone : is an AWP alive, and which zone the AWPer holds.
  - bomb_carrier_zone          : zone of the T player holding C4 (pre-plant), else -1.

Utility (from `inventory`):
  - {side}_smokes/_flashes/_fire/_he/_util_total ; utility_advantage (CT - T).
    'fire' = Molotov + Incendiary (same tool, opposite sides).
"""
from __future__ import annotations
import math
from collections import Counter

import polars as pl

NAMED_ZONES = ["a_site", "b_site", "banana", "mid", "ct_spawn"]
CALLOUT_TO_ZONE = {
    "BombsiteA": "a_site", "BombsiteB": "b_site", "Banana": "banana",
    "Middle": "mid", "LowerMid": "mid", "SecondMid": "mid", "TopofMid": "mid",
    "CTSpawn": "ct_spawn",
}
GRENADES = {
    "Smoke Grenade": "smokes", "Flashbang": "flashes",
    "Molotov": "fire", "Incendiary Grenade": "fire",
    "High Explosive Grenade": "he",
}


def _entropy(places) -> float:
    c = Counter(p for p in places if p)
    n = sum(c.values())
    if n == 0:
        return 0.0
    return float(-sum((v / n) * math.log2(v / n) for v in c.values()))


def _zone_code(place) -> int:
    z = CALLOUT_TO_ZONE.get(place)
    return NAMED_ZONES.index(z) if z in NAMED_ZONES else -1


def positional_features(snap: pl.DataFrame) -> dict:
    out: dict[str, float] = {}
    for side in ("ct", "t"):
        s = snap.filter((pl.col("side") == side) & (pl.col("health") > 0))
        places = s["place"].to_list()
        invs = s["inventory"].to_list()

        out[f"{side}_positional_entropy"] = _entropy(places)
        zc = {z: 0 for z in NAMED_ZONES}
        for p in places:
            z = CALLOUT_TO_ZONE.get(p)
            if z:
                zc[z] += 1
        for z in NAMED_ZONES:
            out[f"{side}_{z}_players"] = zc[z]

        awp_alive, awp_zone = 0, -1
        for inv, p in zip(invs, places):
            if inv and "AWP" in inv:
                awp_alive, awp_zone = 1, _zone_code(p)
                break
        out[f"{side}_awp_alive"] = awp_alive
        out[f"{side}_awp_zone"] = awp_zone

    # bomb carrier (T holding C4) zone, pre-plant
    bz = -1
    t_alive = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))
    for inv, p in zip(t_alive["inventory"].to_list(), t_alive["place"].to_list()):
        if inv and "C4 Explosive" in inv:
            bz = _zone_code(p)
            break
    out["bomb_carrier_zone"] = bz
    return out


def utility_features(snap: pl.DataFrame) -> dict:
    out: dict[str, float] = {}
    for side in ("ct", "t"):
        s = snap.filter((pl.col("side") == side) & (pl.col("health") > 0))
        cnt = {"smokes": 0, "flashes": 0, "fire": 0, "he": 0}
        for inv in s["inventory"].to_list():
            if inv:
                for it in inv:
                    k = GRENADES.get(it)
                    if k:
                        cnt[k] += 1
        for k, v in cnt.items():
            out[f"{side}_{k}"] = v
        out[f"{side}_util_total"] = sum(cnt.values())
    out["utility_advantage"] = out["ct_util_total"] - out["t_util_total"]
    return out


def tactical_features(snap: pl.DataFrame) -> dict:
    return {**positional_features(snap), **utility_features(snap)}


TACTICAL_COLS = (
    [f"{s}_positional_entropy" for s in ("ct", "t")]
    + [f"{s}_{z}_players" for s in ("ct", "t") for z in NAMED_ZONES]
    + [f"{s}_awp_alive" for s in ("ct", "t")]
    + [f"{s}_awp_zone" for s in ("ct", "t")]
    + ["bomb_carrier_zone"]
    + [f"{s}_{u}" for s in ("ct", "t") for u in ("smokes", "flashes", "fire", "he", "util_total")]
    + ["utility_advantage"]
)
