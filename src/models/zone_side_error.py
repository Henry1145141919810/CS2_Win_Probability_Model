"""Where is the model weakest? Error broken down by side, bombsite, phase, and map zone.

Uses OOF predictions (logreg, set EB2) and reports per group: n, log-loss, AUC, and the
calibration gap (mean predicted P(CT win) - actual CT win-rate; +ve = CT-biased). Groups:
  - winner side (CT / T)            -> directional bias check
  - bombsite of the plant (A / B / none)
  - round phase (early / mid / endgame)
  - T-commitment zone (argmax of T players across a_site/b_site/banana/mid/ct_spawn)
    -> "which part of Inferno is hardest to call"

Usage: python src/models/zone_side_error.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
ZONES = ["a_site", "b_site", "banana", "mid", "ct_spawn"]


def _grp(name, y, p, mask):
    if mask.sum() < 200:
        return None
    yy, pp = y[mask], p[mask]
    auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
    ll = log_loss(yy, pp, labels=[0, 1])
    return (f"  {name:16s} n={mask.sum():>7}  logloss {ll:.4f}  "
            f"AUC {auc:.4f}  calib-gap {pp.mean()-yy.mean():+.3f}")


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    p, _ = oof_predict(df, FEATURE_SETS["EB2"], "logreg")
    tsec = df["time_elapsed_sec"].to_numpy()
    planted = df["bomb_state"].to_numpy() == 2
    print(f"model logreg EB2; {len(y)} snaps; overall logloss {log_loss(y,p):.4f} "
          f"AUC {roc_auc_score(y,p):.4f}\n")

    print("by WINNER side:")
    for s, m in [("CT won", y == 1), ("T won", y == 0)]:
        print(_grp(s, y, p, m))
    print("\nby BOMBSITE:")
    site = df["bomb_site"].to_numpy()
    for s, m in [("A", site == 0), ("B", site == 1), ("not planted", site == -1)]:
        r = _grp(s, y, p, m);  print(r) if r else None
    print("\nby PHASE:")
    for s, m in [("early(0-10)", (tsec <= 10) & ~planted), ("mid(10-30)", (tsec > 10) & (tsec <= 30) & ~planted),
                 ("endgame", planted | (tsec > 30))]:
        r = _grp(s, y, p, m);  print(r) if r else None

    print("\nby T-COMMITMENT zone (where most Ts are):")
    tz = np.column_stack([df[f"t_{z}_players"].to_numpy() for z in ZONES])
    dom = tz.argmax(axis=1)
    has = tz.max(axis=1) > 0
    for i, z in enumerate(ZONES):
        r = _grp(z, y, p, (dom == i) & has);  print(r) if r else None


if __name__ == "__main__":
    main()
