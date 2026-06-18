"""Soft-voting ensemble of the classical model matrix — does combining help?

Averages the OOF predicted probabilities of the five classical models on set E and reports
the standard metric battery vs the best single model. (A soft vote is the honest, leak-free
combine since each model's OOF preds are already cross-validated.) Also reports a simple
logit-average variant. Part of the model matrix; runs the standard metrics.

Usage: python src/models/ensemble.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece, bss  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]


def row(name, y, p):
    return (f"{name:18s} {roc_auc_score(y,p):>7.4f} {log_loss(y,p):>8.4f} "
            f"{brier_score_loss(y,p):>7.4f} {ece(y,p):>7.4f} {bss(y,p):>6.3f}")


def main():
    df = pl.read_parquet(DATA)
    preds = {}
    y = None
    for m in MODELS:
        preds[m], y = oof_predict(df, COLS, m)

    print(f"set E, {len(y)} snapshots\n")
    print(f"{'model':18s} {'AUC':>7} {'logloss':>8} {'brier':>7} {'ECE':>7} {'BSS':>6}")
    for m in MODELS:
        print(row(m, y, preds[m]))

    P = np.vstack([preds[m] for m in MODELS])
    mean_p = P.mean(axis=0)                                   # soft vote (prob average)
    eps = 1e-7
    logit_mean = 1 / (1 + np.exp(-np.log((P + eps) / (1 - P + eps)).mean(axis=0)))  # logit avg
    print("  " + "-" * 55)
    print(row("ENSEMBLE (mean)", y, mean_p))
    print(row("ENSEMBLE (logit)", y, logit_mean))

    best = max(MODELS, key=lambda m: roc_auc_score(y, preds[m]))
    print(f"\nbest single = {best} (AUC {roc_auc_score(y, preds[best]):.4f}); "
          f"ensemble-mean AUC {roc_auc_score(y, mean_p):.4f} "
          f"(delta {roc_auc_score(y, mean_p)-roc_auc_score(y, preds[best]):+.4f})")


if __name__ == "__main__":
    main()
