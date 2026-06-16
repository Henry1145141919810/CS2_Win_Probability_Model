"""Pillar 2 — Map control (Voronoi territorial ownership) for de_inferno.

Core idea (from the project plan): at each sampled tick, take only ALIVE players,
build a KD-tree from their (x, y) positions, and assign every nav-mesh area to the
nearest living player's team. Aggregate to a continuous territorial-control surface.

Control is **area-weighted** by default (a 34k-unit area counts more than a 1-unit
sliver) which is more faithful to "territory owned" than counting areas equally.

Named-zone control (A site, banana, mid, ...) is supported via `zone_control`, but
the Inferno zone boxes are PROVISIONAL and must be calibrated against a radar overlay
on a real parsed round before use in the paper (see Week-2 plan). The overall-control
function below needs no zone definitions and is the validated core.

Time-series derivatives (trend slope, volatility) operate on a per-tick control series.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

NAVS_DIR = Path.home() / ".awpy" / "navs"
ZONE_MAP_PATH = Path(__file__).resolve().parents[2] / "configs" / "inferno_zone_map.parquet"
NAMED_ZONES = ["a_site", "b_site", "banana", "mid", "ct_spawn"]


@lru_cache(maxsize=4)
def load_nav(map_name: str = "de_inferno"):
    from awpy.nav import Nav

    return Nav.from_json(str(NAVS_DIR / f"{map_name}.json"))


@lru_cache(maxsize=4)
def nav_grid(map_name: str = "de_inferno"):
    """Return (centroids_xy[N,2], sizes[N], centroids_z[N]) for all nav areas."""
    nav = load_nav(map_name)
    xy, sizes, z = [], [], []
    for a in nav.areas.values():
        xy.append((a.centroid.x, a.centroid.y))
        sizes.append(a.size)
        z.append(a.centroid.z)
    return np.asarray(xy, float), np.asarray(sizes, float), np.asarray(z, float)


def _norm_side(teams):
    """Normalize awpy side labels to 'CT'/'T'. Accepts 'ct'/'CT'/'CounterTerrorist'
    and 't'/'T'/'TERRORIST' (case-insensitive)."""
    out = []
    for s in teams:
        sl = str(s).lower()
        out.append("CT" if sl.startswith("c") else "T")
    return np.asarray(out)


def voronoi_owner(px, py, teams, map_name: str = "de_inferno"):
    """Assign each nav area to the nearest alive player's team.

    Args:
        px, py: 1-D arrays of ALIVE player x / y world coords.
        teams:  1-D array of side labels aligned with px/py. awpy uses lowercase
                'ct'/'t'; normalized internally to 'CT'/'T'.
    Returns:
        owner_team: array[N_areas] of the controlling team ('CT'/'T') per nav area
                    (or None per element if no players supplied).
    """
    xy, _, _ = nav_grid(map_name)
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    teams = _norm_side(teams)
    if px.size == 0:
        return np.full(len(xy), None, dtype=object)
    tree = cKDTree(np.column_stack([px, py]))
    _, idx = tree.query(xy, k=1)
    return teams[idx]


def voronoi_control(px, py, teams, map_name: str = "de_inferno", weight: str = "area"):
    """Overall territorial control split.

    weight: 'area' (size-weighted, default) or 'count' (areas counted equally).
    Returns dict with ct/t control fractions and CT control deficit (ct - 0.5).
    """
    xy, sizes, _ = nav_grid(map_name)
    owner = voronoi_owner(px, py, teams, map_name)
    if owner[0] is None:
        return {"ct_voronoi_control_pct": np.nan, "t_voronoi_control_pct": np.nan,
                "control_deficit": np.nan}
    w = sizes if weight == "area" else np.ones_like(sizes)
    total = w.sum()
    ct = w[owner == "CT"].sum() / total
    t = w[owner == "T"].sum() / total
    return {
        "ct_voronoi_control_pct": float(ct),
        "t_voronoi_control_pct": float(t),
        "control_deficit": float(ct - 0.5),
    }


@lru_cache(maxsize=2)
def _zone_labels(map_name: str = "de_inferno"):
    """Per-nav-area macro-zone labels (aligned to nav_grid order). Built by
    build_zone_map.py; returns all-'other' if the map file is missing."""
    if not ZONE_MAP_PATH.exists():
        n = len(nav_grid(map_name)[1])
        return np.array(["other"] * n)
    import polars as pl
    return pl.read_parquet(ZONE_MAP_PATH)["zone"].to_numpy()


def zone_control(px, py, teams, map_name: str = "de_inferno"):
    """CT control fraction within each named macro-zone (area-weighted)."""
    _, sizes, _ = nav_grid(map_name)
    owner = voronoi_owner(px, py, teams, map_name)
    zones = _zone_labels(map_name)
    if owner[0] is None:
        return {f"ct_{z}_control": np.nan for z in NAMED_ZONES}
    out = {}
    for z in NAMED_ZONES:
        m = zones == z
        denom = sizes[m].sum()
        out[f"ct_{z}_control"] = float(sizes[m & (owner == "CT")].sum() / denom) if denom else np.nan
    return out


def control_features(px, py, teams, map_name: str = "de_inferno") -> dict:
    """All instantaneous map-control features: overall + deficit + per-zone."""
    return {**voronoi_control(px, py, teams, map_name),
            **zone_control(px, py, teams, map_name)}


MAPCONTROL_COLS = [
    "ct_voronoi_control_pct", "control_deficit",
    "ct_a_site_control", "ct_b_site_control", "ct_banana_control",
    "ct_mid_control", "ct_ct_spawn_control",
    "control_trend", "control_volatility",
]


@lru_cache(maxsize=4)
def _area_tree(map_name: str = "de_inferno"):
    xy, _, _ = nav_grid(map_name)
    return cKDTree(xy)


def contest_control(px, py, teams, map_name: str = "de_inferno",
                    max_range: float = 1800.0):
    """4-state map control: a team 'controls' an area only if a living player can
    actually contest it = within `max_range` AND has line-of-sight (cached LOS matrix).

    Returns area-weighted fractions: ct / t / contested (both) / grey (neither).
    Fixes the pure-Voronoi flaw where far, wall-blocked players still 'own' territory
    (e.g. banana going CT just because the nearest player is a distant CT).
    """
    from features import visibility
    xy, sizes, _ = nav_grid(map_name)
    vis = visibility.load_if_cached()  # None if not built (memory-heavy) -> distance-only
    tree = _area_tree(map_name)
    teams = _norm_side(teams)
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    if px.size == 0:
        return {"ct_los_control": np.nan, "t_los_control": np.nan,
                "contested_pct": np.nan, "grey_pct": np.nan, "ct_los_deficit": np.nan}
    player_areas = tree.query(np.column_stack([px, py]))[1]
    n = len(xy)
    ct_can = np.zeros(n, bool)
    t_can = np.zeros(n, bool)
    for k in range(px.size):
        d = np.hypot(xy[:, 0] - px[k], xy[:, 1] - py[k])
        reach = d <= max_range
        if vis is not None:           # gate by line-of-sight when available
            reach &= vis[player_areas[k]]
        if teams[k] == "CT":
            ct_can |= reach
        else:
            t_can |= reach
    w = sizes
    tot = w.sum()
    ct = float(w[ct_can & ~t_can].sum() / tot)
    t = float(w[t_can & ~ct_can].sum() / tot)
    return {
        "ct_los_control": ct,
        "t_los_control": t,
        "contested_pct": float(w[ct_can & t_can].sum() / tot),
        "grey_pct": float(w[~ct_can & ~t_can].sum() / tot),
        "ct_los_deficit": ct - t,
    }


MAPCONTROL_LOS_COLS = [
    "ct_los_control", "t_los_control", "contested_pct", "grey_pct", "ct_los_deficit",
]


def control_trend(series, window: int = 10) -> float:
    """Linear slope of CT control over the last `window` samples (per-sample units).
    Positive => CT gaining territory. NaN if insufficient data."""
    s = np.asarray(series, float)
    s = s[~np.isnan(s)][-window:]
    if s.size < 2:
        return np.nan
    x = np.arange(s.size)
    return float(np.polyfit(x, s, 1)[0])


def control_volatility(series, window: int = 10) -> float:
    """Rolling std of CT control over the last `window` samples. NaN if <2 points."""
    s = np.asarray(series, float)
    s = s[~np.isnan(s)][-window:]
    return float(np.std(s)) if s.size >= 2 else np.nan
