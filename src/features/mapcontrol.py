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


def voronoi_owner(px, py, teams, map_name: str = "de_inferno"):
    """Assign each nav area to the nearest alive player's team.

    Args:
        px, py: 1-D arrays of ALIVE player x / y world coords.
        teams:  1-D array of team labels ('CT'/'T') aligned with px/py.
    Returns:
        owner_team: array[N_areas] of the controlling team per nav area
                    (or None per element if no players supplied).
    """
    xy, _, _ = nav_grid(map_name)
    px = np.asarray(px, float)
    py = np.asarray(py, float)
    teams = np.asarray(teams)
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


# --- PROVISIONAL Inferno named zones (axis-aligned x/y boxes). CALIBRATE before use. ---
# Boxes are (x_min, x_max, y_min, y_max) in world units. These are rough guesses from
# the nav coordinate bounds and MUST be validated on a radar overlay (Week 2).
INFERNO_ZONES_PROVISIONAL: dict[str, tuple[float, float, float, float]] = {
    # placeholder extents — intentionally empty until calibrated against a real round.
}


def zone_control(px, py, teams, zones: dict, map_name: str = "de_inferno",
                 weight: str = "area"):
    """Per-zone CT control fraction. `zones` maps name -> (xmin,xmax,ymin,ymax).

    A nav area belongs to a zone if its centroid falls inside the box. Returns
    {f'ct_{zone}_control': frac} for each zone (NaN if the zone contains no areas).
    """
    xy, sizes, _ = nav_grid(map_name)
    owner = voronoi_owner(px, py, teams, map_name)
    out: dict[str, float] = {}
    if owner[0] is None:
        return {f"ct_{z}_control": np.nan for z in zones}
    w = sizes if weight == "area" else np.ones_like(sizes)
    for name, (xmin, xmax, ymin, ymax) in zones.items():
        in_zone = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & \
                  (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        denom = w[in_zone].sum()
        if denom == 0:
            out[f"ct_{name}_control"] = np.nan
            continue
        ct = w[in_zone & (owner == "CT")].sum()
        out[f"ct_{name}_control"] = float(ct / denom)
    return out


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
