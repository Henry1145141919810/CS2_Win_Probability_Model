"""Firepower v4 PROXY benchmark — mean-normalised variants, derived without re-assembling.

Builds v4.1 / v4.2 / v4.3 from columns already in the assembled tables and compares them
against EB2 (no firepower) and EFB2 (the v2 summed encoding), on 5-fold GroupKFold CV over
the training set and on the 2026 out-of-time holdout.

READ THIS BEFORE COMPARING TO eval_firepower_mean.py. The proxies divide by
`players_alive`; the integrated columns in firepower.py divide by `n_with_stats` (alive
players actually present in the HLTV table). Where coverage is incomplete these differ
substantially -- on the 2026 set 8.4% of snapshots have alive players and rating_sum = 0,
and the implied per-player rating has median 0.836 under players_alive vs 1.086 under
n_with_stats. The proxies therefore measure "sum / headcount", which is part skill and
part HLTV coverage; they are a cheap screen, not a substitute for the integrated run.

Usage:
    python src/models/eval_firepower_v4.py
    python src/models/eval_firepower_v4.py --models logreg,xgb
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.firepower_v4_1 import add_v41, FIREPOWER_V4_1_COLS  # noqa: E402
from features.firepower_v4_2 import add_v42, FIREPOWER_V4_2_COLS  # noqa: E402
from features.firepower_v4_3 import add_v43, FIREPOWER_V4_3_COLS  # noqa: E402
from features.firepower import year_for_match  # noqa: E402
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
TEST = ROOT / "data" / "test_dataset_2026.parquet"
OUT = ROOT / "outputs" / "firepower_v4_benchmark.csv"


def contested(df: pl.DataFrame) -> np.ndarray:
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def prepare(df: pl.DataFrame) -> pl.DataFrame:
    df = add_v41(df)
    df = add_v43(df)
    try:
        from models.eval_firepower_v3 import build_match_weights
        mids = df["match_id"].unique().to_list()
        w = build_match_weights(mids, {m: year_for_match(m) for m in mids})
        if w:
            df = add_v42(df, w)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] v4.2 weights unavailable ({type(e).__name__}: {e}); skipping v4.2",
              file=sys.stderr)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg")
    ap.add_argument("--train", type=Path, default=TRAIN)
    ap.add_argument("--test", type=Path, default=TEST)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    tr, te = prepare(pl.read_parquet(args.train)), prepare(pl.read_parquet(args.test))
    base = FEATURE_SETS["EB2"]
    variants = {
        "EB2": base,
        "EFB2": FEATURE_SETS["EFB2"],
        "EFB4.1": base + FIREPOWER_V4_1_COLS,
        "EFB4.2": base + FIREPOWER_V4_2_COLS,
        "EFB4.3": base + FIREPOWER_V4_3_COLS,
    }
    ytr, yte = tr["ct_won"].to_numpy(), te["ct_won"].to_numpy()
    ctr, cte = contested(tr), contested(te)
    groups = tr["match_id"].to_numpy()

    rows = []
    print(f"{'set':<10}{'model':<10}{'CV AUC':>9}{'CV cAUC':>9}{'OOT AUC':>9}{'OOT cAUC':>10}")
    for name, cols in variants.items():
        cols = [c for c in cols if c in tr.columns and c in te.columns]
        if not cols:
            print(f"{name:<10} skipped (columns absent)")
            continue
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
                 "oot_cauc": roc_auc_score(yte[cte], p[cte])}
            rows.append(r)
            print(f"{name:<10}{mname:<10}{r['cv_auc']:>9.4f}{r['cv_cauc']:>9.4f}"
                  f"{r['oot_auc']:>9.4f}{r['oot_cauc']:>10.4f}")
    if rows:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(rows).write_csv(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
