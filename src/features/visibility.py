"""Precompute & cache the de_inferno nav-area visibility (line-of-sight) matrix.

True LOS via awpy's BVH ray-caster is ~674 rays/s single-threaded; the full
3060x3060 matrix (~4.68M symmetric pairs) is ~2 h serially, so we parallelize across
CPU cores (~10 min on a 16-core box) and cache the boolean matrix to disk. Per-snapshot
map control then does O(1) lookups instead of ray-casting.

Eye height: area centroids are floor-level; we raise them by EYE units so sightlines
approximate standing players (not the ground).

Usage:
    python src/features/visibility.py            # build + cache
    python src/features/visibility.py --workers 12
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "visibility_de_inferno.npy"
TRI = Path.home() / ".awpy" / "tris" / "de_inferno.tri"
EYE = 64.0

_P = None   # worker-local centroid array
_vc = None  # worker-local VisibilityChecker


def _centroids() -> np.ndarray:
    from features.mapcontrol import load_nav
    nav = load_nav()
    return np.array([[a.centroid.x, a.centroid.y, a.centroid.z + EYE]
                     for a in nav.areas.values()], dtype=float)


def _init(P):
    global _P, _vc
    import awpy.visibility as V
    _P = P
    _vc = V.VisibilityChecker(path=TRI)


def _rows(rng):
    i0, i1 = rng
    n = len(_P)
    out = np.zeros((i1 - i0, n), dtype=bool)
    for i in range(i0, i1):
        pi = tuple(_P[i])
        for j in range(i + 1, n):
            if _vc.is_visible(pi, tuple(_P[j])):
                out[i - i0, j] = True
    return i0, out


def build(workers: int = 1) -> np.ndarray:
    import multiprocessing as mp
    import psutil
    avail = psutil.virtual_memory().available / 1e9
    safe = max(1, int((avail - 1.5) / GB_PER_BVH))  # leave 1.5GB headroom
    workers = workers or 1
    if workers > safe:
        print(f"[memory guard] only {avail:.1f}GB free -> capping workers "
              f"{workers}->{safe} ({GB_PER_BVH}GB per BVH). Close apps for more.")
        workers = safe
    if avail < GB_PER_BVH + 1.0:
        raise MemoryError(
            f"Only {avail:.1f}GB RAM free; one visibility BVH needs ~{GB_PER_BVH}GB. "
            "Close some apps and retry (this build is one-time; result is cached).")
    P = _centroids()
    n = len(P)
    step = max(1, n // 120)  # ~120 chunks -> frequent checkpoints, fine progress
    chunks = [(i, min(i + step, n)) for i in range(0, n, step)]
    idx_of = {c[0]: i for i, c in enumerate(chunks)}

    # --- resumable checkpoint (a ~2h serial build must survive interruption) ---
    pm, pd = CACHE.with_suffix(".partial.npy"), CACHE.with_suffix(".done.npy")
    mat = np.zeros((n, n), dtype=bool)
    done = np.zeros(len(chunks), dtype=bool)
    if pm.exists() and pd.exists():
        mat, done = np.load(pm), np.load(pd)
        print(f"resuming: {int(done.sum())}/{len(chunks)} chunks already done")
    todo = [c for i, c in enumerate(chunks) if not done[i]]
    print(f"building {n}x{n} visibility, {len(todo)}/{len(chunks)} chunks, "
          f"{workers} worker(s)...")
    t0 = time.time()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with mp.Pool(workers, initializer=_init, initargs=(P,)) as pool:
        cnt = 0
        for i0, block in pool.imap_unordered(_rows, todo):
            mat[i0:i0 + block.shape[0]] = block
            done[idx_of[i0]] = True
            cnt += 1
            if cnt % 5 == 0:
                np.save(pm, mat)
                np.save(pd, done)
                el = time.time() - t0
                eta = el / cnt * (len(todo) - cnt)
                print(f"  {int(done.sum())}/{len(chunks)} chunks "
                      f"({el:.0f}s, eta {eta/60:.0f}min)", flush=True)
    mat |= mat.T  # symmetrize
    np.fill_diagonal(mat, True)
    np.save(CACHE, mat)
    pm.unlink(missing_ok=True)
    pd.unlink(missing_ok=True)
    print(f"done in {time.time()-t0:.0f}s -> {CACHE} (visible frac {mat.mean():.3f})")
    return mat


_MAT = None
GB_PER_BVH = 4.4  # measured: awpy VisibilityChecker BVH for de_inferno


def load_if_cached():
    """Return the cached matrix, or None if it hasn't been built. NEVER builds
    (building needs ~4.4GB per worker — see build())."""
    global _MAT
    if _MAT is None and CACHE.exists():
        _MAT = np.load(CACHE)
    return _MAT


def load() -> np.ndarray:
    """Memoized load; builds if missing (memory-heavy — prefer load_if_cached)."""
    global _MAT
    if _MAT is None:
        _MAT = np.load(CACHE) if CACHE.exists() else build()
    return _MAT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    build(args.workers)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
