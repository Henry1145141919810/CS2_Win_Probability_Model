"""Phase-aware (time-windowed) calibration — less overconfident early, still sharp late.

Tests whether calibrating the probabilities SEPARATELY by round phase helps the
overconfidence-vs-endgame tension. Compares, on cross-fitted OOF predictions (no leak):
  - raw model
  - global isotonic recalibration
  - PHASE isotonic (a separate isotonic map per phase: early / mid / endgame)
Reports log-loss + ECE split by phase. Also fits a single global TEMPERATURE: if T*≈1 the
model is already calibrated and uniform softening offers nothing (only phase-aware can help).

Usage: python src/models/phase_calibration.py [--model logreg]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
TAC = TACTICAL_COLS + BOMB_COLS
SETS = {"logreg": ECONOMY_COLS + MAPCONTROL_COLS + TAC,
        "xgb": ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS}


def phase_of(tsec, planted):
    ph = np.where(planted | (tsec > 30), "endgame",
                  np.where(tsec > 10, "mid", "early"))
    return ph


def crossfit_iso(p, y, groups, key=None):
    """Cross-fitted isotonic: fit on train folds, apply to test fold (no leak).
    If key given (per-snapshot phase label), fit a SEPARATE isotonic within each key value."""
    out = np.zeros_like(p)
    for tr, te in GroupKFold(5).split(p, y, groups):
        if key is None:
            iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = iso.predict(p[te])
        else:
            for k in np.unique(key):
                mtr = tr[key[tr] == k]; mte = te[key[te] == k]
                if len(mtr) > 50 and len(mte):
                    iso = IsotonicRegression(out_of_bounds="clip").fit(p[mtr], y[mtr])
                    out[mte] = iso.predict(p[mte])
                elif len(mte):
                    out[mte] = p[mte]
    return np.clip(out, 1e-6, 1 - 1e-6)


def report(name, y, p, ph):
    row = f"{name:16s} {log_loss(y,p):>8.4f} {ece(y,p):>7.4f} |"
    for k in ["early", "mid", "endgame"]:
        m = ph == k
        row += f" {k}:{log_loss(y[m],p[m]):.4f}"
    print(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logreg", choices=list(SETS))
    args = ap.parse_args()
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy(); groups = df["match_id"].to_numpy()
    ph = phase_of(df["time_elapsed_sec"].to_numpy(), df["bomb_planted"].to_numpy() == 1)

    p_raw, _ = oof_predict(df, SETS[args.model], args.model)
    p_glob = crossfit_iso(p_raw, y, groups)
    p_phase = crossfit_iso(p_raw, y, groups, key=ph)

    # global temperature on logits (T*>1 => was overconfident; ~1 => already calibrated)
    z = np.log(np.clip(p_raw, 1e-6, 1 - 1e-6) / (1 - np.clip(p_raw, 1e-6, 1 - 1e-6)))
    T = minimize_scalar(lambda t: log_loss(y, 1 / (1 + np.exp(-z / t))),
                        bounds=(0.5, 3.0), method="bounded").x

    print(f"model={args.model}; optimal global temperature T*={T:.3f} "
          f"({'≈1 → already calibrated' if abs(T-1) < 0.05 else 'softening helps' if T>1 else 'sharpening helps'})\n")
    print(f"{'calibration':16s} {'logloss':>8} {'ECE':>7} | per-phase log-loss")
    print("-" * 64)
    report("raw", y, p_raw, ph)
    report("global isotonic", y, p_glob, ph)
    report("PHASE isotonic", y, p_phase, ph)


if __name__ == "__main__":
    main()
