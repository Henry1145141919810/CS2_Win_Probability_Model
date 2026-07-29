"""Diagnostic: WHEN, if ever, does firepower add value out-of-time?

Trains EB2 (no firepower) and EFB2 (with firepower) on the full training set, predicts on the
same-year 2026 holdout (the BEST case for firepower: full coverage + same-era knowledge), and slices
firepower's marginal AUC (EFB2 - EB2) by:
  (a) seconds into round      -> tests the prior-vs-posterior hypothesis (help early, fade late?)
  (b) total players alive      -> 10 (5v5) down to 2
  (c) clutch situations        -> exactly one side has 1 player alive (skill "should" dominate)
  (d) contested even duels     -> equal alive & even econ (the hard subset)

If firepower helps early / at 5v5 / in clutches and not elsewhere -> skill is real but subsumed by
state. If it helps nowhere (even in clutches) -> encoding failure or a deeper null.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
TEST = ROOT / "data" / "test_dataset_2026_sameyr.parquet"   # best case for firepower
MODELS = ["logreg", "lgbm"]


def auc(y, p):
    return roc_auc_score(y, p) if (y.min() != y.max() and len(y) > 50) else np.nan


def fit_predict(tr, te, cols, model):
    Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(float))
    Xte = np.nan_to_num(te.select(cols).to_numpy().astype(float))
    ytr = tr["ct_won"].to_numpy().astype(float)
    m = make_model(model).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def main():
    tr = pl.read_parquet(TRAIN)
    te = pl.read_parquet(TEST)
    yte = te["ct_won"].to_numpy().astype(float)
    t = te["time_elapsed_sec"].to_numpy()
    ca = te["ct_players_alive"].to_numpy()
    ta = te["t_players_alive"].to_numpy()
    ce = te["ct_equipment_value"].to_numpy()
    tee = te["t_equipment_value"].to_numpy()
    total_alive = ca + ta
    is_clutch = (np.minimum(ca, ta) == 1)                       # one side down to its last player
    contested = (ca == ta) & (np.abs(ce - tee) <= 1500)

    preds = {}
    for model in MODELS:
        preds[(model, "EB2")] = fit_predict(tr, te, FEATURE_SETS["EB2"], model)
        preds[(model, "EFB2")] = fit_predict(tr, te, FEATURE_SETS["EFB2"], model)

    def slice_report(name, mask_fn, buckets):
        print(f"\n===== {name} =====")
        print(f"{'bucket':>16s} {'n':>7s} " + "  ".join(f"{m:>16s}" for m in MODELS))
        for label, mask in buckets:
            n = int(mask.sum())
            cells = []
            for model in MODELS:
                if n < 100 or yte[mask].min() == yte[mask].max():
                    cells.append(f"{'--':>16s}")
                    continue
                a_eb = auc(yte[mask], preds[(model, "EB2")][mask])
                a_ef = auc(yte[mask], preds[(model, "EFB2")][mask])
                d = a_ef - a_eb
                cells.append(f"{a_eb:.3f}->{a_ef:.3f}({d:+.3f})")
            print(f"{label:>16s} {n:>7d} " + "  ".join(cells))

    # (a) seconds into round
    edges = [(0, 5), (5, 15), (15, 30), (30, 45), (45, 300)]
    slice_report("(a) firepower marginal AUC by SECONDS INTO ROUND",
                 None, [(f"{lo}-{hi}s", (t >= lo) & (t < hi)) for lo, hi in edges])

    # (b) total players alive
    slice_report("(b) by TOTAL PLAYERS ALIVE (10 = 5v5)",
                 None, [(f"{k} alive", total_alive == k) for k in [10, 8, 6, 4, 2]])

    # (c) clutch vs not
    slice_report("(c) CLUTCH (a side at 1 alive) vs not",
                 None, [("clutch (1vX)", is_clutch), ("not clutch", ~is_clutch)])

    # (d) contested even duels
    slice_report("(d) CONTESTED (equal alive & even econ) vs rest",
                 None, [("contested", contested), ("rest", ~contested)])

    # overall for reference
    slice_report("overall (all snapshots)", None, [("all", np.ones_like(yte, bool))])


if __name__ == "__main__":
    main()
