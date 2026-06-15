"""Bomb / rotation features (Pillar 4, post-plant) — the defuse-race geometry.

After the bomb is planted the round becomes a race: can the CTs reach and defuse before
it detonates? "Bomb planted = True" alone misses this; these features encode WHERE the
bomb is and HOW FAR the nearest CT is from it, including true nav-mesh path distance
(Dijkstra), not just straight-line.

Per snapshot (post-plant only; pre-plant -> neutral defaults):
  - bomb_site            : 0 = A, 1 = B, -1 = not planted
  - bomb_plant_x/y       : plant coordinates (NaN pre-plant)
  - min_ct_dist_to_bomb  : nearest alive CT straight-line distance to the bomb
  - min_ct_path_to_bomb  : nearest alive CT NAV-PATH (Dijkstra) distance to the bomb
  - n_ct_near_bomb       : alive CTs within 500u (straight-line) of the bomb
"""
from __future__ import annotations
import math
from functools import lru_cache

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from features.mapcontrol import load_nav, nav_grid

SITE_CODE = {"BombsiteA": 0, "BombsiteB": 1}
_NEAR = 500.0

# area-id lookup structures (built once)
_nav = load_nav()
_area_ids = list(_nav.areas.keys())
_cents, _, _ = nav_grid()
_tree = cKDTree(_cents[:, :2])
_id_centroid = {aid: np.array([a.centroid.x, a.centroid.y]) for aid, a in _nav.areas.items()}


def _area_of(x: float, y: float) -> int:
    return _area_ids[_tree.query([x, y])[1]]


@lru_cache(maxsize=200_000)
def _path_dist(a_id: int, b_id: int) -> float:
    """Nav-mesh shortest-path distance (sum of centroid hops). Euclid fallback."""
    try:
        path = _nav.find_path(a_id, b_id, weight="dist")
    except Exception:
        path = None
    if not path or len(path) < 2:
        return float(np.linalg.norm(_id_centroid[a_id] - _id_centroid[b_id]))
    d = 0.0
    for p, q in zip(path[:-1], path[1:]):
        d += math.dist((p.centroid.x, p.centroid.y), (q.centroid.x, q.centroid.y))
    return d


def plant_info(bomb_df: pl.DataFrame) -> dict[int, dict]:
    """round_num -> {tick, x, y, site} for the plant event of each planted round."""
    out = {}
    for r in bomb_df.filter(pl.col("event") == "plant").iter_rows(named=True):
        out[r["round_num"]] = {"tick": r["tick"], "x": r["X"], "y": r["Y"],
                               "site": SITE_CODE.get(r["bombsite"], -1),
                               "area": _area_of(r["X"], r["Y"])}
    return out


def bomb_features(snap: pl.DataFrame, plant: dict | None, tick: int) -> dict:
    nan = float("nan")
    if plant is None or tick < plant["tick"]:
        return {"bomb_site": -1, "bomb_plant_x": nan, "bomb_plant_y": nan,
                "min_ct_dist_to_bomb": nan, "min_ct_path_to_bomb": nan,
                "n_ct_near_bomb": 0}
    bx, by = plant["x"], plant["y"]
    ct = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    xs, ys = ct["X"].to_list(), ct["Y"].to_list()
    if not xs:
        return {"bomb_site": plant["site"], "bomb_plant_x": bx, "bomb_plant_y": by,
                "min_ct_dist_to_bomb": nan, "min_ct_path_to_bomb": nan, "n_ct_near_bomb": 0}
    dists = [math.dist((x, y), (bx, by)) for x, y in zip(xs, ys)]
    min_i = int(np.argmin(dists))
    path = _path_dist(_area_of(xs[min_i], ys[min_i]), plant["area"])
    return {
        "bomb_site": plant["site"],
        "bomb_plant_x": float(bx),
        "bomb_plant_y": float(by),
        "min_ct_dist_to_bomb": float(min(dists)),
        "min_ct_path_to_bomb": float(path),
        "n_ct_near_bomb": int(sum(d < _NEAR for d in dists)),
    }


BOMB_COLS = [
    "bomb_site", "bomb_plant_x", "bomb_plant_y",
    "min_ct_dist_to_bomb", "min_ct_path_to_bomb", "n_ct_near_bomb",
]
