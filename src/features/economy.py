"""Pillar 1 — Economy / combat state (the literature baseline = Model A).

Replicates the established economy+combat feature set (Xenopoulos/ESTA) that prior CS
win-probability work uses, computed per per-second snapshot from the parsed ticks.

A "snapshot" is the set of (up to) 10 player rows at one sampled tick within a round.
Dead players have health==0; team aggregates use ALIVE players only (equipment/HP/armor
of a dead player no longer contributes to that team's round-winning capability).
"""
from __future__ import annotations
import polars as pl

ECO, FORCE = 2000, 3800  # equipment-value thresholds for economy class (eco/force/full)


def _econ_class(v: float) -> int:
    return 0 if v < ECO else (1 if v < FORCE else 2)  # 0=eco,1=force,2=full


def economy_features(snap: pl.DataFrame, round_row: dict, tick: int,
                     ct_score: int, t_score: int) -> dict:
    """Economy/combat features for one snapshot.

    snap: player rows at this tick (cols: side, health, armor, current_equip_value,
          has_defuser). round_row: the round's metadata (freeze_end, bomb_plant).
    ct_score/t_score: cumulative side-win counts BEFORE this round.
    """
    ct = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    t = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))

    ct_equip = float(ct["current_equip_value"].sum())
    t_equip = float(t["current_equip_value"].sum())
    bomb_plant = round_row.get("bomb_plant")
    bomb_planted = int(bomb_plant is not None and tick >= bomb_plant)

    return {
        "time_elapsed_sec": (tick - round_row["freeze_end"]) / 64.0,
        "ct_players_alive": ct.height,
        "t_players_alive": t.height,
        "ct_health_total": int(ct["health"].sum()),
        "t_health_total": int(t["health"].sum()),
        "ct_armor_total": int(ct["armor"].sum()),
        "t_armor_total": int(t["armor"].sum()),
        "ct_equipment_value": ct_equip,
        "t_equipment_value": t_equip,
        "ct_economy_class": _econ_class(ct_equip),
        "t_economy_class": _econ_class(t_equip),
        "ct_defuse_kits": int(ct["has_defuser"].sum()),
        "bomb_planted": bomb_planted,
        "ct_score": ct_score,
        "t_score": t_score,
        "score_diff": ct_score - t_score,
        "round_num": round_row["round_num"],
    }


# Column groups so the training pipeline can select feature sets (Model A vs B/C/D/E).
ECONOMY_COLS = [
    "time_elapsed_sec", "ct_players_alive", "t_players_alive",
    "ct_health_total", "t_health_total", "ct_armor_total", "t_armor_total",
    "ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
    "ct_defuse_kits", "bomb_planted", "ct_score", "t_score", "score_diff", "round_num",
]
