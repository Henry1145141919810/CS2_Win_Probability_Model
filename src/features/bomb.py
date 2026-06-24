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

from features.mapcontrol import load_nav, nav_grid, _norm_side

SITE_CODE = {"BombsiteA": 0, "BombsiteB": 1}
_NEAR = 500.0
BOMB_TIMER_SEC = 40.0    # CS2 C4 fuse
DEFUSE_KIT_SEC = 5.0     # defuse time with kit
DEFUSE_NOKIT_SEC = 10.0  # without kit
CT_SPEED = 250.0         # ~run speed (u/s) for a rough defuse-race time
BOMB_LOCAL_RADIUS = 600.0  # "around the bomb" neighbourhood

# area-id lookup structures (built once)
_nav = load_nav()
_area_ids = list(_nav.areas.keys())
_cents, _sizes, _ = nav_grid()
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


# ---------------------------------------------------------------------------
# Bomb-STATE tracking + map control AROUND the bomb (carried / dropped / planted)
# ---------------------------------------------------------------------------
# Motivation: the existing features only fire post-PLANT and only use the fixed plant
# location. But control of the bomb's neighbourhood matters in other high-leverage states:
#   - retake (planted): who controls the area immediately around the planted bomb decides
#     whether a defuse is even attemptable, more precisely than whole-site control.
#   - dropped (T carrier killed -> loose C4): a scramble — CTs "sitting on" the bomb deny
#     the pickup; whoever controls that spot effectively holds the round.
#   - carried: the bomb is wherever the T carrier is (which site is being hit).
# So we track the bomb's live (state, x, y) from the bomb event stream and compute Voronoi
# control in a radius around it, plus a defuse-race feasibility margin for the planted case.

STATE_CODE = {"carried": 0, "dropped": 1, "planted": 2, "over": 3}


class BombTracker:
    """Reconstruct the bomb's (state, x, y) at any tick from a round's bomb events.

    Events (each with X,Y): pickup / drop / plant / defuse / detonate. The bomb's position is
    that of the latest event, except while 'carried' (after a pickup) when it rides the carrier
    — we resolve that from the live tick snapshot by steamid.
    """

    def __init__(self, bomb_df_round):
        self.ev = sorted(
            ({"tick": r["tick"], "event": r["event"], "x": r["X"], "y": r["Y"],
              "steamid": r["steamid"]} for r in bomb_df_round.iter_rows(named=True)),
            key=lambda e: e["tick"])

    def _last(self, tick):
        last = None
        for e in self.ev:
            if e["tick"] <= tick:
                last = e
            else:
                break
        return last

    def state_at(self, tick, snap):
        """Return (state_str, x, y). snap = tick rows (needs steamid, X, Y, side, health)."""
        e = self._last(tick)
        if e is None:                       # before any event: bomb is with the T side
            t = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))
            if t.height:
                return "carried", float(t["X"].mean()), float(t["Y"].mean())
            return "carried", float("nan"), float("nan")
        ev = e["event"]
        if ev == "plant":
            return "planted", float(e["x"]), float(e["y"])
        if ev in ("defuse", "detonate"):
            return "over", float(e["x"]), float(e["y"])
        if ev == "drop":
            return "dropped", float(e["x"]), float(e["y"])
        # pickup -> carried: find the carrier in the snapshot, else fall back to event pos
        carrier = snap.filter(pl.col("steamid") == e["steamid"])
        if carrier.height and carrier["health"][0] > 0:
            return "carried", float(carrier["X"][0]), float(carrier["Y"][0])
        return "carried", float(e["x"]), float(e["y"])


def _local_control(bx, by, px, py, teams):
    """Area-weighted CT/T Voronoi control among nav areas within BOMB_LOCAL_RADIUS of (bx,by)."""
    if not np.isfinite(bx) or not np.isfinite(by) or len(px) == 0:
        return float("nan"), float("nan")
    near = (_cents[:, 0] - bx) ** 2 + (_cents[:, 1] - by) ** 2 <= BOMB_LOCAL_RADIUS ** 2
    idx = np.where(near)[0]
    if idx.size == 0:                       # bomb far from any centroid -> nearest area
        idx = np.array([int(np.argmin((_cents[:, 0] - bx) ** 2 + (_cents[:, 1] - by) ** 2))])
    teams = _norm_side(teams)
    ptree = cKDTree(np.column_stack([np.asarray(px, float), np.asarray(py, float)]))
    owner = teams[ptree.query(_cents[idx, :2])[1]]
    w = _sizes[idx]
    tot = w.sum()
    ct = float(w[owner == "CT"].sum() / tot)
    t = float(w[owner == "T"].sum() / tot)
    return ct, ct - t


def bomb_live_features(snap, tracker: "BombTracker", plant: dict | None, tick: int) -> dict:
    """Bomb-state + map-control-around-the-bomb + defuse-race features (all round states)."""
    state, bx, by = tracker.state_at(tick, snap)
    alive = snap.filter(pl.col("health") > 0)
    ct = alive.filter(pl.col("side") == "ct")
    t = alive.filter(pl.col("side") == "t")
    loc_ct, loc_def = _local_control(bx, by, alive["X"].to_list(), alive["Y"].to_list(),
                                     alive["side"].to_list())

    def _min_dist(team_df):
        if not team_df.height or not np.isfinite(bx):
            return float("nan")
        return float(np.min(np.hypot(np.asarray(team_df["X"]) - bx,
                                     np.asarray(team_df["Y"]) - by)))
    d_ct, d_t = _min_dist(ct), _min_dist(t)
    closer = float(d_ct < d_t) if (np.isfinite(d_ct) and np.isfinite(d_t)) else float("nan")

    # defuse-race margin (planted only): time left on fuse minus (run-to-bomb + defuse) time
    margin = float("nan")
    if state == "planted" and plant is not None:
        time_left = BOMB_TIMER_SEC - (tick - plant["tick"]) / 64.0
        if ct.height:
            xs, ys = ct["X"].to_list(), ct["Y"].to_list()
            di = int(np.argmin([math.dist((x, y), (bx, by)) for x, y in zip(xs, ys)]))
            path = _path_dist(_area_of(xs[di], ys[di]), plant["area"])
            defuse = DEFUSE_KIT_SEC if int(ct["has_defuser"].sum()) > 0 else DEFUSE_NOKIT_SEC
            margin = time_left - (path / CT_SPEED + defuse)

    return {
        "bomb_state": STATE_CODE.get(state, 0),
        "bomb_dropped": int(state == "dropped"),
        "ct_bomb_local_control": loc_ct,
        "ct_bomb_local_deficit": loc_def,
        "min_ct_dist_to_bomb_live": d_ct,
        "min_t_dist_to_bomb_live": d_t,
        "ct_closer_to_bomb": closer,
        "defuse_time_margin": margin,
    }


BOMB_LIVE_COLS = [
    "bomb_state", "bomb_dropped",
    "ct_bomb_local_control", "ct_bomb_local_deficit",
    "min_ct_dist_to_bomb_live", "min_t_dist_to_bomb_live", "ct_closer_to_bomb",
    "defuse_time_margin",
]


def defuse_race_features(snap, plant: dict | None, tick: int) -> dict:
    """Refined defuse-race geometry (post-plant only; the v1 winner was defuse_time_margin).

    Per-CT, kit-aware: each alive CT's finish time = nav-path/run-speed + (5s w/ kit else 10s).
    The defusing CT is the one with the smallest finish time. Adds:
      - defuse_margin_kit     : fuse time left - best CT finish time  (per-CT kit-aware; >0 feasible)
      - n_ct_can_defuse       : # CTs that could arrive+defuse before detonation
      - best_defuser_has_kit  : does the fastest-to-finish CT carry a kit
      - defuse_contest_margin : nearest-T arrival time - best CT finish time  (>0 = CT finishes
                                before any T can rotate back to interrupt = a 'clean' defuse)
    """
    nan = float("nan")
    base = {"defuse_margin_kit": nan, "n_ct_can_defuse": 0,
            "best_defuser_has_kit": 0, "defuse_contest_margin": nan}
    if plant is None or tick < plant["tick"]:
        return base
    time_left = BOMB_TIMER_SEC - (tick - plant["tick"]) / 64.0
    barea = plant["area"]
    ct = snap.filter((pl.col("side") == "ct") & (pl.col("health") > 0))
    if not ct.height:
        return base
    xs, ys = ct["X"].to_list(), ct["Y"].to_list()
    kits = ct["has_defuser"].to_list()
    finishes = []
    for x, y, k in zip(xs, ys, kits):
        arrive = _path_dist(_area_of(x, y), barea) / CT_SPEED
        finishes.append((arrive + (DEFUSE_KIT_SEC if k else DEFUSE_NOKIT_SEC), bool(k)))
    best_finish, best_kit = min(finishes, key=lambda f: f[0])
    n_can = int(sum(f <= time_left for f, _ in finishes))
    # nearest T arrival (path) to contest the defuse
    t = snap.filter((pl.col("side") == "t") & (pl.col("health") > 0))
    if t.height:
        t_arrive = min(_path_dist(_area_of(x, y), barea) / CT_SPEED
                       for x, y in zip(t["X"].to_list(), t["Y"].to_list()))
        contest = t_arrive - best_finish
    else:
        contest = float(time_left)   # no T alive -> uncontested by definition
    return {"defuse_margin_kit": float(time_left - best_finish),
            "n_ct_can_defuse": n_can,
            "best_defuser_has_kit": int(best_kit),
            "defuse_contest_margin": float(contest)}


BOMB_DEFUSE_COLS = [
    "defuse_margin_kit", "n_ct_can_defuse", "best_defuser_has_kit", "defuse_contest_margin",
]
