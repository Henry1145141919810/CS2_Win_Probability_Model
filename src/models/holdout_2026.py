"""2026 OUT-OF-TIME HOLDOUT — the final validity gate (touch-once).

Trains each model on the FULL 220-demo training set (2024-2025) and evaluates ONCE on 27 fresh
2026 Inferno matches (55,271 snapshots) that were never seen in any fold. This tests real
generalisation (out-of-time), not just cross-validation.

Reports, for every (model, feature set):
  - IN-TIME  : 5-fold GroupKFold OOF on the training set (the number we've been quoting)
  - OUT-OF-TIME: trained on all training data, predicted on 2026
  - full metric battery (AUC/log-loss/Brier/ECE/BSS/contested-AUC) + calibration slope/intercept
  - B=500 match-level bootstrap CIs over the 27 test matches
  - the soft-vote ensemble on the holdout
Plus a DISTRIBUTION-SHIFT check: training base rate 0.445 vs 2026 base rate 0.512.

Usage: python src/models/holdout_2026.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, make_model, oof_predict, ece, bss  # noqa: E402
from models.extended_metrics import brier_decomp, cal_slope_intercept, adaptive_ece  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
TEST = ROOT / "data" / "test_dataset_2026.parquet"
OUTCSV = ROOT / "outputs" / "holdout_2026.csv"
OUTFIG = ROOT / "outputs" / "figures" / "holdout_2026.png"
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
SETS = ["A", "E", "EB2", "EFB2"]
B = 500


def contested_mask(df):
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def boot_ci(y, p, groups, cont, B=B, seed=0):
    """Match-level block bootstrap over the TEST matches -> 95% CI for AUC / ECE / cAUC."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    auc, e, ca = [], [], []
    for _ in range(B):
        s = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys, ps = y[s], p[s]
        if ys.min() == ys.max():
            continue
        auc.append(roc_auc_score(ys, ps)); e.append(ece(ys, ps))
        cs = cont[s]
        if cs.sum() > 50 and ys[cs].min() != ys[cs].max():
            ca.append(roc_auc_score(ys[cs], ps[cs]))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
    return q(auc), q(e), q(ca)


def metrics(y, p, cont):
    rel, res, _ = brier_decomp(y, p)
    slope, inter = cal_slope_intercept(y, p)
    return dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), ECE=ece(y, p), adaptECE=adaptive_ece(y, p),
                BSS=bss(y, p), cAUC=roc_auc_score(y[cont], p[cont]),
                REL=rel, RES=res, cal_slope=slope, cal_intercept=inter)


def main():
    tr = pl.read_parquet(TRAIN)
    te = pl.read_parquet(TEST)
    ytr = tr["ct_won"].to_numpy().astype(float)
    yte = te["ct_won"].to_numpy().astype(float)
    gte = te["match_id"].to_numpy()
    cte = contested_mask(te)
    print(f"TRAIN: {tr.height} snaps / {tr['match_id'].n_unique()} matches / base rate {ytr.mean():.3f}")
    print(f"TEST 2026: {te.height} snaps / {te['match_id'].n_unique()} matches / base rate {yte.mean():.3f}")
    print(f"** DISTRIBUTION SHIFT: base rate {ytr.mean():.3f} -> {yte.mean():.3f} "
          f"(2026 Inferno is more CT-sided) **\n")

    rows = []
    test_preds = {}
    for st in SETS:
        cols = FEATURE_SETS[st]
        Xtr = np.nan_to_num(tr.select(cols).to_numpy().astype(float))
        Xte = np.nan_to_num(te.select(cols).to_numpy().astype(float))
        for m in MODELS:
            oof, _ = oof_predict(tr, cols, m)                      # IN-TIME (5-fold OOF)
            in_auc = roc_auc_score(ytr, oof); in_ece = ece(ytr, oof)
            in_cauc = roc_auc_score(ytr[contested_mask(tr)], oof[contested_mask(tr)])
            mdl = make_model(m).fit(Xtr, ytr)                      # OUT-OF-TIME (fit all, predict 2026)
            p = mdl.predict_proba(Xte)[:, 1]
            test_preds[(m, st)] = p
            om = metrics(yte, p, cte)
            (alo, ahi), (elo, ehi), (clo, chi) = boot_ci(yte, p, gte, cte)
            rows.append({"model": m, "set": st, "in_AUC": in_auc, "out_AUC": om["AUC"],
                         "dAUC": om["AUC"] - in_auc, "out_AUC_lo": alo, "out_AUC_hi": ahi,
                         "in_ECE": in_ece, "out_ECE": om["ECE"], "out_ECE_lo": elo, "out_ECE_hi": ehi,
                         "in_cAUC": in_cauc, "out_cAUC": om["cAUC"], "out_cAUC_lo": clo, "out_cAUC_hi": chi,
                         "out_logloss": om["logloss"], "out_brier": om["brier"], "out_BSS": om["BSS"],
                         "cal_slope": om["cal_slope"], "cal_intercept": om["cal_intercept"]})
            print(f"{m:9s} {st:5s} | IN AUC {in_auc:.4f} -> OUT {om['AUC']:.4f} "
                  f"({om['AUC']-in_auc:+.4f}) CI({alo:.4f},{ahi:.4f}) | "
                  f"ECE {in_ece:.3f}->{om['ECE']:.3f} | cAUC {in_cauc:.3f}->{om['cAUC']:.3f} | "
                  f"slope {om['cal_slope']:.2f} int {om['cal_intercept']:+.2f}")

    # soft-vote ensemble on the holdout (5 classical models, EFB2)
    ens = np.mean([test_preds[(m, "EFB2")] for m in MODELS], axis=0)
    om = metrics(yte, ens, cte)
    (alo, ahi), (elo, ehi), (clo, chi) = boot_ci(yte, ens, gte, cte)
    rows.append({"model": "SOFT-VOTE(5)", "set": "EFB2", "in_AUC": np.nan, "out_AUC": om["AUC"],
                 "dAUC": np.nan, "out_AUC_lo": alo, "out_AUC_hi": ahi, "in_ECE": np.nan,
                 "out_ECE": om["ECE"], "out_ECE_lo": elo, "out_ECE_hi": ehi, "in_cAUC": np.nan,
                 "out_cAUC": om["cAUC"], "out_cAUC_lo": clo, "out_cAUC_hi": chi,
                 "out_logloss": om["logloss"], "out_brier": om["brier"], "out_BSS": om["BSS"],
                 "cal_slope": om["cal_slope"], "cal_intercept": om["cal_intercept"]})
    print(f"\nSOFT-VOTE(5) EFB2 | OUT AUC {om['AUC']:.4f} CI({alo:.4f},{ahi:.4f}) "
          f"ECE {om['ECE']:.3f} cAUC {om['cAUC']:.3f} slope {om['cal_slope']:.2f}")

    tbl = pl.DataFrame(rows)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True); tbl.write_csv(OUTCSV)
    print(f"\nwrote {OUTCSV}")

    # ---- figure: in-time vs out-of-time AUC + reliability on the holdout ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    d = tbl.filter(pl.col("model") != "SOFT-VOTE(5)")
    for st, mk in zip(SETS, ["o", "s", "^", "D"]):
        s = d.filter(pl.col("set") == st)
        ax1.scatter(s["in_AUC"], s["out_AUC"], marker=mk, s=70, label=f"set {st}")
    lo = min(d["in_AUC"].min(), d["out_AUC"].min()) - 0.005
    hi = max(d["in_AUC"].max(), d["out_AUC"].max()) + 0.005
    ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x (no drop)")
    ax1.set_xlabel("IN-TIME AUC (5-fold OOF, 2024-25)"); ax1.set_ylabel("OUT-OF-TIME AUC (2026)")
    ax1.set_title("Does the model hold up out-of-time?\n(points below the line = degradation)")
    ax1.legend(fontsize=8)

    p_best = test_preds[("logreg", "EFB2")]
    edges = np.linspace(0, 1, 11); conf, acc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p_best >= a) & (p_best < b if b < 1 else p_best <= b)
        if m.sum():
            conf.append(p_best[m].mean()); acc.append(yte[m].mean())
    ax2.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax2.plot(conf, acc, "o-", color="#cc3311", lw=2, label="logreg EFB2 on 2026")
    ax2.axhline(yte.mean(), color="#2a8", ls=":", lw=1, label=f"2026 base rate {yte.mean():.3f}")
    ax2.axhline(ytr.mean(), color="#888", ls=":", lw=1, label=f"train base rate {ytr.mean():.3f}")
    ax2.set_xlabel("predicted P(CT win)"); ax2.set_ylabel("observed CT win-rate")
    ax2.set_title("Calibration on the 2026 holdout\n(base-rate shift 0.445 -> 0.512)")
    ax2.legend(fontsize=8); ax2.set_aspect("equal")
    fig.tight_layout(); OUTFIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTFIG, dpi=120); plt.close(fig)
    print(f"wrote {OUTFIG}")


if __name__ == "__main__":
    main()
