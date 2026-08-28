"""The whole Firepower exploration in one table: v1 vs v2 vs v3 vs v4, on clean data.

Each version was benchmarked against a different baseline at a different time, on tables of
different vintages -- and the v4 run turned out to have trained on the out-of-time holdout
(see docs/notes_firepower_v4.md, Finding 5). This puts all four encodings on ONE training
table, ONE test set and ONE baseline, so the progression can actually be read.

  v1  sums of un-split rating/adr, kast mean, lone-survivor clutch   (9 cols, recomputed)
  v2  side-aware stats + situational gates, still sums               (20 cols, in table)
  v3  v2 scaled by team rank weight 1/log2(rank+1)                   (18 cols, derived)
  v4  v2 stats divided by n_with_stats                               (20 cols, in table)

v1 has to be recomputed (it was replaced before any surviving table was assembled) and v3
is derived as v2 x team_weight, which is exact: every alive player on a side shares one
team, so a summed stat scales linearly with that side's weight.

Usage:
    python src/models/eval_firepower_versions.py --models logreg,xgb
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
from features.firepower import FIREPOWER_COLS, FIREPOWER_MEAN_COLS, year_for_match  # noqa: E402
from features.firepower_v1 import v1_for_demo, FIREPOWER_V1_COLS  # noqa: E402
from features.firepower_v3 import FIREPOWER_COLS_V3  # noqa: E402
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402
import models.eval_firepower_v3 as v3mod  # noqa: E402


def add_v1(df: pl.DataFrame, ticks_dir: Path) -> pl.DataFrame:
    """Recompute the v1 columns for every (match_id, tick) in df."""
    parts = []
    for m in df["match_id"].unique().to_list():
        f = ticks_dir / f"{m}.parquet"
        if not f.exists():
            continue
        t = pl.read_parquet(f, columns=["tick", "steamid", "side", "health"])
        parts.append(v1_for_demo(t, year_for_match(m)).with_columns(pl.lit(m).alias("match_id")))
    if not parts:
        raise SystemExit(f"no tick files under {ticks_dir}")
    return df.join(pl.concat(parts), on=["match_id", "tick"], how="left")


def add_v3(df: pl.DataFrame, ticks_dir: Path, formula: str = "log2") -> pl.DataFrame:
    """v3 = v2 x the side's team-rank weight, with the halftime swap."""
    v3mod.TICKS_DIR = ticks_dir           # build_match_weights reads round-1 ticks from here
    mids = df["match_id"].unique().to_list()
    w = v3mod.build_match_weights(mids, {m: year_for_match(m) for m in mids})
    wt = pl.DataFrame({"match_id": list(w),
                       "_h1_ct": [w[m][formula]["h1_ct"] for m in w],
                       "_h1_t": [w[m][formula]["h1_t"] for m in w]})
    out = df.join(wt, on="match_id", how="left")
    first = pl.col("round_num") < 13
    ct_w = pl.when(first).then(pl.col("_h1_ct")).otherwise(pl.col("_h1_t"))
    t_w = pl.when(first).then(pl.col("_h1_t")).otherwise(pl.col("_h1_ct"))
    exprs = []
    for v2c, v3c in v3mod.V2_TO_V3.items():
        exprs.append((pl.col(v2c) * (ct_w if v2c.startswith("ct_") else t_w)).alias(v3c))
    return out.with_columns(exprs).drop(["_h1_ct", "_h1_t"])


def contested(df: pl.DataFrame) -> np.ndarray:
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "training_dataset_clean220.parquet")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test_dataset_2026_v4.parquet")
    ap.add_argument("--train-ticks", type=Path, default=ROOT / "data" / "parquet" / "ticks")
    ap.add_argument("--test-ticks", type=Path,
                    default=ROOT / "data" / "holdout2026" / "parquet_defuse" / "ticks")
    ap.add_argument("--models", default="logreg,xgb")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "firepower_versions.csv")
    args = ap.parse_args()

    tr, te = pl.read_parquet(args.train), pl.read_parquet(args.test)
    print(f"train {tr.height:,} / {tr['match_id'].n_unique()} matches | "
          f"test {te.height:,} / {te['match_id'].n_unique()} matches")
    print()

    for name, d, tk in (("train", "tr", args.train_ticks), ("test", "te", args.test_ticks)):
        print(f"deriving v1/v3 for {name} ...", flush=True)
    tr = add_v3(add_v1(tr, args.train_ticks), args.train_ticks)
    te = add_v3(add_v1(te, args.test_ticks), args.test_ticks)

    base = FEATURE_SETS["EB2"]
    variants = {
        "EB2 (no FP)": base,
        "+v1 (raw sums)": base + FIREPOWER_V1_COLS,
        "+v2 (sided sums)": base + FIREPOWER_COLS,
        "+v3 (rank-weighted)": base + FIREPOWER_COLS_V3,
        "+v4 (means)": base + FIREPOWER_MEAN_COLS,
    }
    ytr, yte = tr["ct_won"].to_numpy(), te["ct_won"].to_numpy()
    ctr, cte = contested(tr), contested(te)
    groups = tr["match_id"].to_numpy()

    rows = []
    print(f"\n{'variant':<22}{'model':<9}{'#f':>4}{'CV AUC':>9}{'CV cAUC':>9}"
          f"{'OOT AUC':>9}{'OOT cAUC':>10}")
    for name, cols in variants.items():
        cols = [c for c in cols if c in tr.columns and c in te.columns]
        Xtr = np.nan_to_num(tr[cols].to_numpy().astype(float))
        Xte = np.nan_to_num(te[cols].to_numpy().astype(float))
        for mn in args.models.split(","):
            oof = np.zeros(len(ytr))
            for a, b in GroupKFold(n_splits=5).split(Xtr, ytr, groups):
                m = make_model(mn); m.fit(Xtr[a], ytr[a]); oof[b] = m.predict_proba(Xtr[b])[:, 1]
            m = make_model(mn); m.fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
            r = {"variant": name, "model": mn, "n_features": len(cols),
                 "cv_auc": roc_auc_score(ytr, oof), "cv_cauc": roc_auc_score(ytr[ctr], oof[ctr]),
                 "oot_auc": roc_auc_score(yte, p), "oot_cauc": roc_auc_score(yte[cte], p[cte])}
            rows.append(r)
            print(f"{name:<22}{mn:<9}{len(cols):>4}{r['cv_auc']:>9.4f}{r['cv_cauc']:>9.4f}"
                  f"{r['oot_auc']:>9.4f}{r['oot_cauc']:>10.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
