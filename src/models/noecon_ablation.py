"""Study 3 - the no-economy ablation: how far do the non-economy pillars get ALONE?

Every published feature set includes ECONOMY_COLS, and the residual analysis (Study 1 of
plan_beyond_economy.md) measures what the other pillars add *after* economy. This script asks the
complementary question: what does each block predict *on its own*, with economy absent.

The definitional trap (see docs/plan_t0_match_noecon.md): ECONOMY_COLS is money + combat state +
clock + score, and headcount leaks into every block that aggregates over alive players (per-zone
player counts, utility totals, n_ct_near_bomb ...). So the ladder below separates the economy block
into MONEY / COMBAT / SCORE, builds "minus count-like columns" variants, and reports every set on the
EQUAL-ALIVE subset (where headcount carries no information) next to the pooled number.

Outputs
  outputs/noecon_ablation.csv      in-time OOF metrics + paired match-bootstrap CIs vs A and vs EB2
  outputs/noecon_timeprofile.csv   OOF AUC by seconds-into-round bucket (no refit)
  outputs/noecon_holdout.csv       out-of-time 2026 (fit on all training, scored once)
  outputs/oof_noecon_<model>.parquet   OOF predictions for every set (for figures)

Usage: python src/models/noecon_ablation.py [--models logreg,xgb] [--bootstrap 500] [--no-holdout]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import (FEATURE_SETS, ECONOMY_COLS, MAPCONTROL_COLS,  # noqa: E402
                                   TACTICAL, oof_predict, make_model, ece, bss)
from features.bomb import BOMB_COLS, BOMB_LIVE_COLS, BOMB_DEFUSE_COLS  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
HOLD = ROOT / "data" / "test_dataset_2026.parquet"     # primary holdout; all sets here are firepower-free
OUT = ROOT / "outputs"

# --- decompose the economy block -------------------------------------------------------------
MONEY = ["ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
         "ct_armor_total", "t_armor_total", "ct_defuse_kits"]
COMBAT = ["ct_players_alive", "t_players_alive", "ct_health_total", "t_health_total",
          "bomb_planted", "time_elapsed_sec"]
SCORE = ["ct_score", "t_score", "score_diff", "round_num"]
assert sorted(MONEY + COMBAT + SCORE) == sorted(ECONOMY_COLS), "economy decomposition drifted"

# columns that are literally counts of alive players (headcount in disguise)
COUNT_LIKE = [c for c in TACTICAL if c.endswith("_players")] + \
             ["ct_awp_alive", "t_awp_alive", "n_ct_near_bomb", "n_ct_can_defuse"]

EB2 = FEATURE_SETS["EB2"]
NONECON = [c for c in EB2 if c not in ECONOMY_COLS]
BOMB_ALL = BOMB_COLS + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS

SETS = {
    # references
    "A":         ECONOMY_COLS,
    "EB2":       EB2,
    # the economy block, split
    "Clock":     ["time_elapsed_sec"],
    "Score":     SCORE,
    "Money":     MONEY,
    "Combat":    COMBAT,
    # non-economy blocks alone
    "Spatial":   MAPCONTROL_COLS,
    "SpatialT-": MAPCONTROL_COLS + [c for c in TACTICAL if c not in COUNT_LIKE],
    "SpatialT":  MAPCONTROL_COLS + TACTICAL,
    "Bomb":      BOMB_ALL,
    # everything except ...
    "NoEcon-":   [c for c in NONECON if c not in COUNT_LIKE],
    "NoEcon":    NONECON,
    "NoMoney":   [c for c in EB2 if c not in MONEY],
}
ORDER = list(SETS)
TIME_BUCKETS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 90), (90, 1e9)]


def masks(df: pl.DataFrame) -> dict[str, np.ndarray]:
    alive_eq = (df["ct_players_alive"] == df["t_players_alive"]).to_numpy()
    even = ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500).to_numpy()
    return {"all": np.ones(df.height, bool), "equal_alive": alive_eq,
            "contested": alive_eq & even, "post_plant": (df["bomb_planted"] == 1).to_numpy()}


def metric_row(y, p, m):
    def auc(mask):
        return roc_auc_score(y[mask], p[mask]) if mask.sum() > 50 and len(np.unique(y[mask])) == 2 else np.nan
    return dict(AUC=auc(m["all"]), logloss=log_loss(y, p, labels=[0, 1]), brier=brier_score_loss(y, p),
                ECE=ece(y, p), BSS=bss(y, p), cAUC=auc(m["contested"]),
                AUC_equal_alive=auc(m["equal_alive"]), AUC_post_plant=auc(m["post_plant"]))


def paired_boot(df, y, preds: dict, refs=("A", "EB2"), B=500, seed=42):
    """One match resample per iteration, every set scored on it -> paired deltas."""
    rng = np.random.default_rng(seed)
    groups = df["match_id"].to_numpy()
    uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    names = list(preds)
    auc_s = {n: [] for n in names}; ll_s = {n: [] for n in names}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        for n in names:
            pp = preds[n][idx]
            auc_s[n].append(roc_auc_score(yy, pp)); ll_s[n].append(log_loss(yy, pp, labels=[0, 1]))
    out = {}
    q = lambda s: (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))
    for n in names:
        a = np.array(auc_s[n]); l = np.array(ll_s[n])
        row = {"AUC_lo": q(a)[0], "AUC_hi": q(a)[1]}
        for r in refs:
            if r in preds and r != n:
                da = a - np.array(auc_s[r]); dl = l - np.array(ll_s[r])
                row[f"dAUC_vs_{r}"] = float(da.mean()); row[f"dAUC_vs_{r}_lo"], row[f"dAUC_vs_{r}_hi"] = q(da)
                row[f"dLL_vs_{r}"] = float(dl.mean()); row[f"dLL_vs_{r}_lo"], row[f"dLL_vs_{r}_hi"] = q(dl)
        out[n] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--sets", default=",".join(ORDER))
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args()
    models = args.models.split(","); sets = args.sets.split(",")
    OUT.mkdir(exist_ok=True)

    df = pl.read_parquet(TRAIN)
    y = df["ct_won"].to_numpy().astype(int)
    m = masks(df)
    tsec = df["time_elapsed_sec"].to_numpy()
    print(f"train: {df.height} snaps / {df['match_id'].n_unique()} matches / base {y.mean():.4f}")
    print(f"sets: " + ", ".join(f"{s}({len(SETS[s])})" for s in sets))
    for s in sets:
        missing = [c for c in SETS[s] if c not in df.columns]
        assert not missing, f"{s}: missing {missing}"

    rows, trows = [], []
    for mdl in models:
        preds = {}
        for s in sets:
            t0 = time.time()
            oof, _ = oof_predict(df, SETS[s], mdl)
            preds[s] = oof
            r = dict(model=mdl, set=s, ncols=len(SETS[s]), **metric_row(y, oof, m))
            rows.append(r)
            print(f"  {mdl:7s} {s:10s} n={len(SETS[s]):3d} AUC {r['AUC']:.4f} ll {r['logloss']:.4f} "
                  f"ECE {r['ECE']:.4f} cAUC {r['cAUC']:.4f} eqAlive {r['AUC_equal_alive']:.4f} "
                  f"postplant {r['AUC_post_plant']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            for lo, hi in TIME_BUCKETS:
                mk = (tsec >= lo) & (tsec < hi)
                trows.append(dict(model=mdl, set=s, t_lo=lo, t_hi=min(hi, 999), n=int(mk.sum()),
                                  AUC=roc_auc_score(y[mk], oof[mk]) if mk.sum() > 50 else np.nan))
        # paired bootstrap on the fixed OOF predictions
        if args.bootstrap:
            print(f"  paired match-bootstrap B={args.bootstrap} ...", flush=True)
            bs = paired_boot(df, y, preds, B=args.bootstrap)
            for r in rows:
                if r["model"] == mdl:
                    r.update(bs[r["set"]])
        # dump OOF preds
        pl.DataFrame({"match_id": df["match_id"], "round_num": df["round_num"], "tick": df["tick"],
                      "ct_won": y, **{f"p_{s}": preds[s] for s in sets}}
                     ).write_parquet(OUT / f"oof_noecon_{mdl}.parquet")

    res = pl.DataFrame(rows)
    # share of EB2's above-chance AUC recovered by each set
    for mdl in models:
        ref = res.filter((pl.col("model") == mdl) & (pl.col("set") == "EB2"))["AUC"]
        if ref.len():
            res = res.with_columns(
                pl.when(pl.col("model") == mdl)
                  .then((pl.col("AUC") - 0.5) / (ref[0] - 0.5))
                  .otherwise(pl.col("share_of_EB2") if "share_of_EB2" in res.columns else None)
                  .alias("share_of_EB2"))
    res.write_csv(OUT / "noecon_ablation.csv")
    pl.DataFrame(trows).write_csv(OUT / "noecon_timeprofile.csv")
    print(f"wrote {OUT/'noecon_ablation.csv'}, {OUT/'noecon_timeprofile.csv'}")

    if args.no_holdout:
        return
    # ---- out-of-time (2026), fit on all training, scored once --------------------------------
    te = pl.read_parquet(HOLD)
    yt = te["ct_won"].to_numpy().astype(int); mt = masks(te)
    print(f"\nholdout: {te.height} snaps / {te['match_id'].n_unique()} matches / base {yt.mean():.4f}")
    hrows = []
    for mdl in models:
        hp = {}
        for s in sets:
            cols = SETS[s]
            X = np.nan_to_num(df[cols].to_numpy().astype(float)); Xt = np.nan_to_num(te[cols].to_numpy().astype(float))
            p = make_model(mdl).fit(X, y).predict_proba(Xt)[:, 1]
            hp[s] = p
            r = dict(model=mdl, set=s, ncols=len(cols), **metric_row(yt, p, mt))
            hrows.append(r)
            print(f"  {mdl:7s} {s:10s} OOT AUC {r['AUC']:.4f} ll {r['logloss']:.4f} ECE {r['ECE']:.4f} "
                  f"cAUC {r['cAUC']:.4f} eqAlive {r['AUC_equal_alive']:.4f}", flush=True)
        if args.bootstrap:
            bs = paired_boot(te, yt, hp, B=args.bootstrap)
            for r in hrows:
                if r["model"] == mdl:
                    r.update(bs[r["set"]])
    pl.DataFrame(hrows).write_csv(OUT / "noecon_holdout.csv")
    print(f"wrote {OUT/'noecon_holdout.csv'}")


if __name__ == "__main__":
    main()
