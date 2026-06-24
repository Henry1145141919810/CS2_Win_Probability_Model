"""Training-loss comparison for the win-prob model: log loss vs Brier vs focal.

Trains XGBoost (same tuned hyperparameters) with three OBJECTIVES and compares OOF metrics,
with log-loss + ECE split by round phase (early / mid / endgame) — the phases where the
overconfidence-vs-sharp-endgame tension lives.

  - logloss (baseline)  : binary cross-entropy  (proper scoring rule)
  - brier               : squared error on probability  (proper scoring rule, gentler tails)
  - focal (gamma=2)     : down-weights easy examples  (NOT a proper rule -> expect miscalibration)

Shows empirically why proper rules (logloss/brier) stay calibrated and focal does not.

Usage: python src/models/loss_experiment.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import ece  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS
HP = dict(n_estimators=600, max_depth=3, learning_rate=0.03, min_child_weight=10,
          reg_lambda=10.0, subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
          tree_method="hist", random_state=0)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def obj_brier(y, raw):
    """Squared error on p=sigmoid(raw). Gauss-Newton (positive) hessian."""
    p = _sig(raw)
    g = 2.0 * (p - y) * p * (1 - p)
    h = np.maximum(2.0 * (p * (1 - p)) ** 2, 1e-6)
    return g, h


def obj_focal(y, raw, gamma=2.0):
    """Binary focal loss gradient/hessian (down-weights well-classified)."""
    p = _sig(raw)
    pt = np.where(y == 1, p, 1 - p)
    # d(FL)/d(raw); standard focal derivation, hessian floored positive
    g = (p - y) * (gamma * (1 - pt) ** (gamma - 1) * (-np.log(np.clip(pt, 1e-6, 1))) * pt
                   + (1 - pt) ** gamma)
    h = np.maximum(np.abs(g) * (1 - p) * p * 4 + 1e-3, 1e-3)
    return g, h


def oof(df, y, groups, objective):
    from xgboost import XGBClassifier
    X = np.nan_to_num(df.select(COLS).to_numpy().astype(float))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        if objective == "logloss":
            m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **HP)
            m.fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        else:
            fn = obj_brier if objective == "brier" else obj_focal
            m = XGBClassifier(objective=fn, **HP)
            m.fit(X[tr], y[tr])
            p[te] = _sig(m.predict(X[te], output_margin=True))
    return p


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    tsec = df["time_elapsed_sec"].to_numpy()
    planted = (df["bomb_planted"].to_numpy() == 1)
    phases = {
        "early(0-10s)": (tsec <= 10) & ~planted,
        "mid(10-30s)": (tsec > 10) & (tsec <= 30) & ~planted,
        "endgame(plant/30+)": planted | (tsec > 30),
    }

    print(f"data {len(y)} snaps\n")
    hdr = f"{'objective':10s} {'AUC':>7} {'logloss':>8} {'ECE':>7} | " + \
          " ".join(f"{k.split('(')[0]:>8}" for k in phases)
    print(hdr); print("-" * len(hdr))
    for objective in ["logloss", "brier", "focal"]:
        p = oof(df, y, groups, objective)
        line = f"{objective:10s} {roc_auc_score(y,p):>7.4f} {log_loss(y,p):>8.4f} {ece(y,p):>7.4f} | "
        line += " ".join(f"{log_loss(y[m],p[m]):>8.4f}" for m in phases.values())
        print(line)
    print("\n(phase columns = LOG-LOSS within that phase; lower=better. "
          "watch early overconfidence vs endgame sharpness.)")
    print("ECE column = overall calibration; focal should be visibly worse (improper rule).")


if __name__ == "__main__":
    main()
