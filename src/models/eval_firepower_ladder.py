"""The firepower ladder: one defect fixed per rung, all on one table and one baseline.

Presented this way rather than by version history, because the history is not the logic.
The count confound was found in V1 but only repaired in V4; the two exploratory directions
(V2's situational gates, V3's team-rank weight) were built on the unrepaired summed
encoding in between. Repairing first, then exploring, is the order that lets each rung be
attributed:

  EB2     no firepower at all                      the reference
  V1      rating SUMMED over alive players         r = 0.987 with the player-count advantage
  V4      rating AVERAGED                          removes that confound
  V4.2    average x team-rank weight                + team strength
  V4.3    every stat averaged, situational gates    + conditional detail

V4/V4.2/V4.3 here divide by `n_with_stats` (alive players actually in the HLTV table), not
by the headcount as the original proxy scripts did. Where coverage is partial the two differ
substantially -- median implied per-player rating 0.836 vs 1.086 -- so these numbers are not
comparable to the earlier proxy run, and supersede it.

Usage:
    python src/models/eval_firepower_ladder.py
    python src/models/eval_firepower_ladder.py --models logreg,xgb      # quick look
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.firepower import FIREPOWER_MEAN_COLS, year_for_match  # noqa: E402
from features.firepower_weighted_mean import (add_weighted_mean,  # noqa: E402
                                              FIREPOWER_WMEAN_COLS)
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402
import models.eval_firepower_v3 as v3mod  # noqa: E402

V1_COLS = ["ct_rating_sum", "t_rating_sum"]
V4_COLS = ["ct_rating_mean", "t_rating_mean"]
V42_COLS = ["ct_rating_wmean", "t_rating_wmean"]


def contested(df: pl.DataFrame) -> np.ndarray:
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def add_weights(df: pl.DataFrame, ticks_dir: Path) -> pl.DataFrame:
    v3mod.TICKS_DIR = ticks_dir
    mids = df["match_id"].unique().to_list()
    w = v3mod.build_match_weights(mids, {m: year_for_match(m) for m in mids})
    return add_weighted_mean(df, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "training_dataset_clean220.parquet")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test_dataset_2026_defuse.parquet")
    ap.add_argument("--train-ticks", type=Path, default=ROOT / "data" / "parquet" / "ticks")
    ap.add_argument("--test-ticks", type=Path,
                    default=ROOT / "data" / "holdout2026" / "parquet_defuse" / "ticks")
    ap.add_argument("--models", default="logreg,xgb,lgbm,catboost,rf")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "firepower_ladder.csv")
    args = ap.parse_args()

    tr, te = pl.read_parquet(args.train), pl.read_parquet(args.test)
    print(f"train {tr.height:,} / {tr['match_id'].n_unique()} matches | "
          f"test {te.height:,} / {te['match_id'].n_unique()} matches", flush=True)
    print("deriving team weights ...", flush=True)
    tr = add_weights(tr, args.train_ticks)
    te = add_weights(te, args.test_ticks)

    base = FEATURE_SETS["EB2"]
    rungs = {
        "EB2 (no FP)": base,
        "V1  sum":     base + V1_COLS,
        "V4  mean":    base + V4_COLS,
        "V4.2 mean x rank": base + V42_COLS,
        "V4.3 mean + gates": base + FIREPOWER_MEAN_COLS,
    }
    ytr, yte = tr["ct_won"].to_numpy(), te["ct_won"].to_numpy()
    ctr, cte = contested(tr), contested(te)
    groups = tr["match_id"].to_numpy()

    rows = []
    print(f"\n{'rung':<20}{'model':<10}{'#f':>4}{'CV AUC':>9}{'CV cAUC':>9}"
          f"{'OOT AUC':>9}{'OOT cAUC':>10}{'':>3}", flush=True)
    for name, cols in rungs.items():
        cols = [c for c in cols if c in tr.columns and c in te.columns]
        Xtr = np.nan_to_num(tr[cols].to_numpy().astype(float))
        Xte = np.nan_to_num(te[cols].to_numpy().astype(float))
        for mn in args.models.split(","):
            t0 = time.time()
            oof = np.zeros(len(ytr))
            for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
                m = make_model(mn); m.fit(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
            m = make_model(mn); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
            r = {"rung": name, "model": mn, "n_features": len(cols),
                 "cv_auc": roc_auc_score(ytr, oof), "cv_cauc": roc_auc_score(ytr[ctr], oof[ctr]),
                 "cv_logloss": log_loss(ytr, oof), "cv_brier": brier_score_loss(ytr, oof),
                 "oot_auc": roc_auc_score(yte, p), "oot_cauc": roc_auc_score(yte[cte], p[cte]),
                 "oot_logloss": log_loss(yte, p), "oot_brier": brier_score_loss(yte, p)}
            rows.append(r)
            print(f"{name:<20}{mn:<10}{len(cols):>4}{r['cv_auc']:>9.4f}{r['cv_cauc']:>9.4f}"
                  f"{r['oot_auc']:>9.4f}{r['oot_cauc']:>10.4f}  ({time.time()-t0:.0f}s)", flush=True)
            pl.DataFrame(rows).write_csv(args.out)      # checkpoint after every cell
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
