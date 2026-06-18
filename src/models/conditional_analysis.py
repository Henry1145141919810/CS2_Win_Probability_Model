"""Conditional analysis: WHERE does map control actually help?

The aggregate AUC lift of the spatial pillars is small (~+0.009) because it's averaged
over many "easy" snapshots (lopsided rounds) economy already nails. This script measures
the lift of E (econ+Voronoi+tactical) and ET (best) over A (economy) *within* subsets where
control should matter: equal alive-counts, even economies, early round, pre-plant, and the
intersection ("contested"). Uses out-of-fold predictions from the same 5-fold GroupKFold.

Usage: python src/models/conditional_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
TAC = TACTICAL_COLS + BOMB_COLS
SETS = {
    "A": ECONOMY_COLS,
    "E": ECONOMY_COLS + MAPCONTROL_COLS + TAC,
    "ET": ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS,
}


def xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                         colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss", random_state=0)


def logreg():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0))


def oof(df, cols, y, groups, model_fn):
    X = df.select(cols).to_numpy().astype(float)
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = model_fn()
        m.fit(np.nan_to_num(X[tr]), y[tr])
        pred[te] = m.predict_proba(np.nan_to_num(X[te]))[:, 1]
    return pred


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()

    alive_eq = (df["ct_players_alive"] == df["t_players_alive"]).to_numpy()
    even_econ = ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500).to_numpy()
    early = (df["time_elapsed_sec"] <= 15).to_numpy()
    preplant = (df["bomb_planted"] == 0).to_numpy()
    full_alive = ((df["ct_players_alive"] == 5) & (df["t_players_alive"] == 5)).to_numpy()
    contested = alive_eq & even_econ
    subsets = {
        "ALL": np.ones(len(y), bool),
        "equal alive-count": alive_eq,
        "5v5 (no kills yet)": full_alive,
        "even economy (|Δ|<=1500)": even_econ,
        "early round (<=15s)": early,
        "pre-plant": preplant,
        "CONTESTED (equal alive & even econ)": contested,
    }

    for mname, mfn in [("XGBoost", xgb), ("Logistic", logreg)]:
        print(f"\n===== {mname} — AUC lift of map-control where it matters =====")
        preds = {s: oof(df, cols, y, groups, mfn) for s, cols in SETS.items()}
        print(f"{'subset':36s} {'N':>9} {'base%':>6} {'AUC A':>7} {'AUC E':>7} "
              f"{'E-A':>8} {'AUC ET':>7} {'ET-A':>8}")
        for name, mask in subsets.items():
            n = int(mask.sum())
            yy = y[mask]
            if n < 50 or yy.min() == yy.max():
                continue
            aA = roc_auc_score(yy, preds["A"][mask])
            aE = roc_auc_score(yy, preds["E"][mask])
            aET = roc_auc_score(yy, preds["ET"][mask])
            print(f"{name:36s} {n:>9,} {yy.mean():>6.2f} {aA:>7.4f} {aE:>7.4f} "
                  f"{aE-aA:>+8.4f} {aET:>7.4f} {aET-aA:>+8.4f}")


if __name__ == "__main__":
    main()
