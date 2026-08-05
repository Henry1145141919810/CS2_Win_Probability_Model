"""Appendix B: full in-time metric battery for the 5 classical models x main feature sets.

Computes 5-fold GroupKFold OOF and the full battery (AUC, log-loss, Brier, ECE, BSS, contested-AUC,
calibration slope/intercept) for every (model, set) in {logreg,xgb,lgbm,catboost,rf} x {A,E,EB2,EFB2}.
Writes outputs/appendix_metrics_intime.csv. Deep/ensemble in-time come from extended_metrics.csv;
out-of-time comes from holdout_2026_sameyr2026.csv.

Usage: python src/models/appendix_metrics.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict, ece, bss  # noqa: E402
from models.extended_metrics import cal_slope_intercept  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "appendix_metrics_intime.csv"
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
SETS = ["A", "E", "EB2", "EFB2"]


def main():
    df = pl.read_parquet(TRAIN)
    y = df["ct_won"].to_numpy().astype(float)
    cont = ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()
    rows = []
    for st in SETS:
        for m in MODELS:
            p, _ = oof_predict(df, FEATURE_SETS[st], m)
            slope, inter = cal_slope_intercept(y, p)
            rows.append({"set": st, "model": m,
                         "AUC": round(roc_auc_score(y, p), 4),
                         "logloss": round(log_loss(y, p, labels=[0, 1]), 4),
                         "brier": round(brier_score_loss(y, p), 4),
                         "ECE": round(ece(y, p), 4),
                         "BSS": round(bss(y, p), 4),
                         "cAUC": round(roc_auc_score(y[cont], p[cont]), 4),
                         "cal_slope": round(slope, 3),
                         "cal_intercept": round(inter, 3)})
            print(f"{st:5s} {m:9s} AUC {rows[-1]['AUC']:.4f} ll {rows[-1]['logloss']:.4f} "
                  f"brier {rows[-1]['brier']:.4f} ECE {rows[-1]['ECE']:.3f} BSS {rows[-1]['BSS']:.3f} "
                  f"cAUC {rows[-1]['cAUC']:.3f}")
    pl.DataFrame(rows).write_csv(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
