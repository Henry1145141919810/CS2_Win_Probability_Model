"""Stacked generalization — a meta-learner over the 5 classical models vs soft-vote.

Each base model's OOF predictions (already cross-validated by match) become features for a
logistic meta-learner, trained with a SECOND GroupKFold (cross-fitting) so the stack is also
leak-free. Compares stack vs the best single model and vs the soft-vote ensemble.

Usage: python src/models/stacking.py [--set EB2]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict, ece, bss  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="EB2")
    args = ap.parse_args()
    cols = FEATURE_SETS[args.set]
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()

    base = {}
    print(f"set {args.set}; base models {MODELS}\n{'model':18s} {'AUC':>7} {'logloss':>8} {'ECE':>7} {'BSS':>6}")
    for m in MODELS:
        base[m], _ = oof_predict(df, cols, m)
        print(f"{m:18s} {roc_auc_score(y,base[m]):>7.4f} {log_loss(y,base[m]):>8.4f} "
              f"{ece(y,base[m]):>7.4f} {bss(y,base[m]):>6.3f}")

    Z = np.column_stack([base[m] for m in MODELS])           # meta-features = base OOF preds
    soft = Z.mean(axis=1)
    stack = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, groups):         # cross-fit the meta-learner
        meta = LogisticRegression(max_iter=2000).fit(Z[tr], y[tr])
        stack[te] = meta.predict_proba(Z[te])[:, 1]
    # average meta coefficients (full-data fit) for interpretability
    coef = LogisticRegression(max_iter=2000).fit(Z, y).coef_[0]

    print("  " + "-" * 50)
    print(f"{'SOFT-VOTE':18s} {roc_auc_score(y,soft):>7.4f} {log_loss(y,soft):>8.4f} "
          f"{ece(y,soft):>7.4f} {bss(y,soft):>6.3f}")
    print(f"{'STACK (logistic)':18s} {roc_auc_score(y,stack):>7.4f} {log_loss(y,stack):>8.4f} "
          f"{ece(y,stack):>7.4f} {bss(y,stack):>6.3f}")
    best = max(MODELS, key=lambda m: roc_auc_score(y, base[m]))
    print(f"\nbest single = {best} ({roc_auc_score(y,base[best]):.4f}); "
          f"stack {roc_auc_score(y,stack):.4f} (delta {roc_auc_score(y,stack)-roc_auc_score(y,base[best]):+.4f})")
    print("meta weights:", {m: round(float(c), 2) for m, c in zip(MODELS, coef)})


if __name__ == "__main__":
    main()
