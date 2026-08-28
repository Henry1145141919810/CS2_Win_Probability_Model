"""Integrated mean-firepower benchmark: EB2 / EFB2 / EB2_FPmean / EFB3.

Reads the `*_mean` columns produced by firepower.py during assemble, so the divisor is
`n_with_stats` -- alive players actually present in the HLTV table -- rather than the
headcount. Requires tables assembled AFTER the v4 change to firepower.py.

Reports CV (5-fold GroupKFold OOF on the training set) and OOT (full train -> 2026),
each as AUC and contested-AUC (equal alive counts and |dequip| <= $1500).

Leak-free variant -- the 2026 rows currently look up 2026 HLTV stats, which nobody has
before the matches are played:
    FIREPOWER_YEAR_LAG=1 python src/models/eval_firepower_mean.py --tag lag

Usage:
    python src/models/eval_firepower_mean.py --models logreg,xgb
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.firepower import FIREPOWER_MEAN_COLS  # noqa: E402
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402

SETS = ["EB2", "EFB2", "EB2_FPmean", "EFB3"]


def contested(df: pl.DataFrame) -> np.ndarray:
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "training_dataset.parquet")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test_dataset_2026.parquet")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tr, te = pl.read_parquet(args.train), pl.read_parquet(args.test)
    missing = [c for c in FIREPOWER_MEAN_COLS if c not in tr.columns]
    if missing:
        sys.exit(f"training table lacks the mean columns ({missing[:3]}...). "
                 "Re-run assemble.py with the v4 firepower.py first.")
    ytr, yte = tr["ct_won"].to_numpy(), te["ct_won"].to_numpy()
    ctr, cte = contested(tr), contested(te)
    groups = tr["match_id"].to_numpy()

    rows = []
    print(f"{'set':<12}{'model':<10}{'CV AUC':>9}{'CV cAUC':>9}"
          f"{'OOT AUC':>9}{'OOT cAUC':>10}{'OOT LL':>9}{'OOT Brier':>10}")
    for name in args.sets.split(","):
        cols = [c for c in FEATURE_SETS[name] if c in tr.columns and c in te.columns]
        Xtr = np.nan_to_num(tr[cols].to_numpy().astype(float))
        Xte = np.nan_to_num(te[cols].to_numpy().astype(float))
        for mname in args.models.split(","):
            oof = np.zeros(len(ytr))
            for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
                m = make_model(mname); m.fit(Xtr[a], ytr[a])
                oof[b] = m.predict_proba(Xtr[b])[:, 1]
            m = make_model(mname); m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            r = {"set": name, "model": mname, "n_features": len(cols),
                 "cv_auc": roc_auc_score(ytr, oof),
                 "cv_cauc": roc_auc_score(ytr[ctr], oof[ctr]),
                 "oot_auc": roc_auc_score(yte, p),
                 "oot_cauc": roc_auc_score(yte[cte], p[cte]),
                 "oot_logloss": log_loss(yte, p),
                 "oot_brier": brier_score_loss(yte, p)}
            rows.append(r)
            print(f"{name:<12}{mname:<10}{r['cv_auc']:>9.4f}{r['cv_cauc']:>9.4f}"
                  f"{r['oot_auc']:>9.4f}{r['oot_cauc']:>10.4f}"
                  f"{r['oot_logloss']:>9.4f}{r['oot_brier']:>10.4f}")
    suffix = f"_{args.tag}" if args.tag else ""
    out = ROOT / "outputs" / f"firepower_mean_benchmark{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
