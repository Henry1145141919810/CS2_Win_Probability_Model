"""Tune XGBoost (GroupKFold AUC) to test 'is it overfitting / untuned, or is the signal linear?'

The default XGBoost (depth 6) underperforms logistic — likely overfitting match-level noise
(only 220 groups). This grid favors shallower, more-regularized trees. If a tuned config
reaches logistic's ~0.849, we've confirmed overfitting/undertuning; if not, the signal is
genuinely near-linear. Run on the best feature set (ET).

Usage: python src/models/tune_xgb.py
"""
from __future__ import annotations
import sys
import itertools
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS + TERRITORY_COLS

GRID = {
    "max_depth": [3, 4, 5, 6],
    "min_child_weight": [1, 10, 50],
    "reg_lambda": [1.0, 10.0],
}
FIXED = dict(n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             n_jobs=-1, eval_metric="logloss", random_state=0, tree_method="hist")


def cv_auc(X, y, groups, params):
    from xgboost import XGBClassifier
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = XGBClassifier(**{**FIXED, **params})
        m.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


def main():
    df = pl.read_parquet(DATA)
    X = np.nan_to_num(df.select(COLS).to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"tuning XGBoost on ET ({len(COLS)} features), {len(combos)} configs, "
          f"5-fold GroupKFold...\n(reference: default depth-6 xgb ET=0.8412; logistic=0.8493)")
    results = []
    for i, p in enumerate(combos, 1):
        auc = cv_auc(X, y, groups, p)
        results.append((auc, p))
        print(f"  [{i:>2}/{len(combos)}] depth={p['max_depth']} "
              f"min_child={p['min_child_weight']} lambda={p['reg_lambda']:>4} -> AUC {auc:.4f}",
              flush=True)
    results.sort(reverse=True)
    print("\nTOP 5:")
    for auc, p in results[:5]:
        print(f"  AUC {auc:.4f}  {p}")
    best_auc, best_p = results[0]
    print(f"\nBEST: AUC {best_auc:.4f} with {{**FIXED, **{best_p}}}")
    print(f"vs default depth-6 ET 0.8412  |  logistic 0.8493")


if __name__ == "__main__":
    main()
