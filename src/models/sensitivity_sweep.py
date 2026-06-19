"""Sensitivity sweep — are the results artifacts of arbitrary feature parameters?

Re-assembles a SUBSET of demos under different settings of four design constants and checks
how much the (logreg, full-set) OOF AUC moves. A small, stable spread ⇒ the result is robust
to the parameter choice, not an artifact.

Parameters swept (each varied alone, others at default):
  - territory decay_sec        (memory horizon; default 15)
  - Voronoi weighting          (area vs count; default area)
  - bomb-local radius          (BOMB_LOCAL_RADIUS; default 600)
  - CT run speed for defuse     (CT_SPEED; default 250)

Usage: python src/models/sensitivity_sweep.py [--n 35]
"""
from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import features.mapcontrol as mc  # noqa: E402
import features.bomb as bombmod  # noqa: E402
from features import assemble as asm  # noqa: E402
from models.train_pipeline import FEATURE_SETS  # noqa: E402

ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
SET = "EBT2"


def _set_decay(v):
    d = list(mc.TerritoryControl.__init__.__defaults__); d[1] = float(v)
    mc.TerritoryControl.__init__.__defaults__ = tuple(d)


def _set_vweight(v):
    d = list(mc.voronoi_control.__defaults__); d[1] = v
    mc.voronoi_control.__defaults__ = tuple(d)


def assemble_subset(demos):
    parts = [asm.assemble_demo(m) for m in demos]
    return pl.concat([p for p in parts if p is not None], how="vertical")


def auc_of(df):
    cols = FEATURE_SETS[SET]
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    y = df["ct_won"].to_numpy(); g = df["match_id"].to_numpy()
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35)
    args = ap.parse_args()
    excl = set()
    ef = ROOT / "configs" / "excluded_offlist.txt"
    if ef.exists():
        excl = {ln.strip() for ln in ef.read_text(encoding="utf-8").splitlines() if ln.strip()}
    demos = [Path(f).stem for f in sorted(glob.glob(str(ROUNDS_DIR / "*.parquet")))
             if Path(f).stem not in excl][: args.n]
    print(f"sensitivity on {len(demos)} demos, set {SET} (logreg OOF AUC)\n")

    base = auc_of(assemble_subset(demos))
    print(f"BASELINE (decay15, area, radius600, speed250): AUC {base:.4f}\n")

    grid = [
        ("decay_sec", _set_decay, [10, 15, 20], 15),
        ("voronoi_weight", _set_vweight, ["area", "count"], "area"),
        ("bomb_local_radius", lambda v: setattr(bombmod, "BOMB_LOCAL_RADIUS", float(v)), [400, 600, 800], 600),
        ("ct_speed", lambda v: setattr(bombmod, "CT_SPEED", float(v)), [215, 250, 285], 250),
    ]
    for name, setter, values, default in grid:
        print(f"### {name}")
        for v in values:
            setter(v)
            a = auc_of(assemble_subset(demos))
            print(f"  {name}={v!s:>6}: AUC {a:.4f}  (Δ vs baseline {a-base:+.4f})")
        setter(default)  # restore
        print()
    print("Interpretation: |Δ| small across a parameter ⇒ result robust to that choice.")


if __name__ == "__main__":
    main()
