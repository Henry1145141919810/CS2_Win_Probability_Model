"""FULL benchmark for the defuse-progress feature.

Compares EB2 vs EB2D and EFB2 vs EFB2D across all classical models:
  - in-time 5-fold OOF: AUC, log-loss, Brier, ECE, contested-AUC + match-level B=500 CIs
  - out-of-time on the 2026 defuse table: same battery
  - paired dAUC (EB2D - EB2) with CI, the honest marginal test
  - the point of the feature: per-second curve honesty on the DEFUSING rows
      * mean predicted P(CT win) by defuse_progress_frac bucket, vs the empirical rate
      * log-loss / Brier over defusing rows
      * calibration on INTERRUPTED attempts (defuse_in_progress==1 & completed==0 proxy: p on those)

Usage:
  python src/models/defuse_full_benchmark.py \
      --train data/training_dataset_defuse.parquet \
      --test  data/test_dataset_2026_defuse.parquet
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, make_model, oof_predict, ece, bss  # noqa: E402

MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
PAIRS = [("EB2", "EB2D"), ("EFB2", "EFB2D")]
B = 500


def contested_mask(df):
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def boot_auc_ci(y, p, groups, B=B, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); by = {m: np.where(groups == m)[0] for m in uniq}
    vals = []
    for _ in range(B):
        idx = np.concatenate([by[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        yy = y[idx]
        if yy.min() != yy.max():
            vals.append(roc_auc_score(yy, p[idx]))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


def boot_diff_ci(y, pa, pb, groups, B=B, seed=0):
    """paired dAUC = AUC(pb) - AUC(pa) with CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups); by = {m: np.where(groups == m)[0] for m in uniq}
    d = []
    for _ in range(B):
        idx = np.concatenate([by[m] for m in rng.choice(uniq, len(uniq), replace=True)])
        yy = y[idx]
        if yy.min() != yy.max():
            d.append(roc_auc_score(yy, pb[idx]) - roc_auc_score(yy, pa[idx]))
    return (np.mean(d), np.percentile(d, 2.5), np.percentile(d, 97.5)) if d else (np.nan, np.nan, np.nan)


def metrics(y, p, cont):
    return dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), ECE=ece(y, p), BSS=bss(y, p),
                cAUC=roc_auc_score(y[cont], p[cont]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=ROOT / "data" / "training_dataset_defuse.parquet")
    ap.add_argument("--test", type=Path, default=ROOT / "data" / "test_dataset_2026_defuse.parquet")
    args = ap.parse_args()

    tr = pl.read_parquet(args.train)
    te = pl.read_parquet(args.test)
    ytr = tr["ct_won"].to_numpy().astype(float)
    yte = te["ct_won"].to_numpy().astype(float)
    gtr = tr["match_id"].to_numpy(); gte = te["match_id"].to_numpy()
    ctr = contested_mask(tr); cte = contested_mask(te)
    print(f"TRAIN {tr.height} snaps / {tr['match_id'].n_unique()} matches | "
          f"defusing rows {int(tr['defuse_in_progress'].sum())} ({tr['defuse_in_progress'].mean()*100:.2f}%)")
    print(f"TEST  {te.height} snaps / {te['match_id'].n_unique()} matches | "
          f"defusing rows {int(te['defuse_in_progress'].sum())}\n")

    rows = []
    oof_store = {}; test_store = {}
    for st in ["EB2", "EB2D", "EFB2", "EFB2D"]:
        cols = FEATURE_SETS[st]
        Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(float))
        Xte = np.nan_to_num(te.select(cols).to_numpy().astype(float))
        for m in MODELS:
            oof, _ = oof_predict(tr, cols, m)
            oof_store[(st, m)] = oof
            im = metrics(ytr, oof, ctr); ilo, ihi = boot_auc_ci(ytr, oof, gtr)
            mdl = make_model(m).fit(Xtr, ytr); pte = mdl.predict_proba(Xte)[:, 1]
            test_store[(st, m)] = pte
            om = metrics(yte, pte, cte); olo, ohi = boot_auc_ci(yte, pte, gte)
            rows.append(dict(set=st, model=m,
                             in_AUC=im["AUC"], in_AUC_lo=ilo, in_AUC_hi=ihi, in_cAUC=im["cAUC"],
                             in_logloss=im["logloss"], in_brier=im["brier"], in_ECE=im["ECE"], in_BSS=im["BSS"],
                             out_AUC=om["AUC"], out_AUC_lo=olo, out_AUC_hi=ohi, out_cAUC=om["cAUC"],
                             out_logloss=om["logloss"], out_brier=om["brier"], out_ECE=om["ECE"], out_BSS=om["BSS"]))
            print(f"{st:6s} {m:9s} | IN AUC {im['AUC']:.4f}({ilo:.4f},{ihi:.4f}) cAUC {im['cAUC']:.3f} "
                  f"| OUT AUC {om['AUC']:.4f}({olo:.4f},{ohi:.4f}) ll {om['logloss']:.4f}")

    # paired dAUC (EB2D - EB2, EFB2D - EFB2), in-time and out-of-time
    print("\n=== paired dAUC: defuse-progress marginal effect ===")
    dfrows = []
    for base, ext in PAIRS:
        for m in MODELS:
            di = boot_diff_ci(ytr, oof_store[(base, m)], oof_store[(ext, m)], gtr)
            do = boot_diff_ci(yte, test_store[(base, m)], test_store[(ext, m)], gte)
            dfrows.append(dict(pair=f"{ext}-{base}", model=m,
                               in_dAUC=di[0], in_lo=di[1], in_hi=di[2],
                               out_dAUC=do[0], out_lo=do[1], out_hi=do[2]))
            sig_i = "SIG" if di[1] > 0 or di[2] < 0 else "ns"
            sig_o = "SIG" if do[1] > 0 or do[2] < 0 else "ns"
            print(f"{ext}-{base} {m:9s} | IN {di[0]:+.4f}({di[1]:+.4f},{di[2]:+.4f}) {sig_i} "
                  f"| OUT {do[0]:+.4f}({do[1]:+.4f},{do[2]:+.4f}) {sig_o}")

    # ---- the real point: curve honesty on DEFUSING rows (2026 test, logreg) ----
    print("\n=== curve honesty on defusing rows (2026 test) ===")
    dmask = (te["defuse_in_progress"] == 1).to_numpy()
    frac = te["defuse_progress_frac"].to_numpy()
    curve_rows = []
    for m in ["logreg", "xgb"]:
        pE = test_store[("EB2", m)]; pED = test_store[("EB2D", m)]
        # metrics on defusing rows
        yd = yte[dmask]
        llE = log_loss(yd, pE[dmask], labels=[0, 1]); llED = log_loss(yd, pED[dmask], labels=[0, 1])
        brE = brier_score_loss(yd, pE[dmask]); brED = brier_score_loss(yd, pED[dmask])
        print(f"  [{m}] defusing rows n={int(dmask.sum())}: "
              f"log-loss {llE:.4f}->{llED:.4f} ({(llED-llE)/llE*100:+.0f}%) | "
              f"brier {brE:.4f}->{brED:.4f} ({(brED-brE)/brE*100:+.0f}%)")
        for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]:
            b = dmask & (frac >= lo) & (frac < hi)
            if b.sum() < 5:
                continue
            curve_rows.append(dict(model=m, bucket=f"{lo:.1f}-{hi:.1f}", n=int(b.sum()),
                                   actual=float(yte[b].mean()), EB2=float(pE[b].mean()), EB2D=float(pED[b].mean())))
            if m == "logreg":
                print(f"    progress {lo:.1f}-{hi:.1f}: n={int(b.sum())} "
                      f"actual {yte[b].mean():.3f} | EB2 {pE[b].mean():.3f} | EB2D {pED[b].mean():.3f}")

    out = ROOT / "outputs"
    pl.DataFrame(rows).write_csv(out / "defuse_benchmark_matrix.csv")
    pl.DataFrame(dfrows).write_csv(out / "defuse_benchmark_paired.csv")
    pl.DataFrame(curve_rows).write_csv(out / "defuse_benchmark_curve.csv")
    print(f"\nwrote {out}/defuse_benchmark_{{matrix,paired,curve}}.csv")


if __name__ == "__main__":
    main()
