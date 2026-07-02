"""Extended metric battery — computed from OUT-OF-FOLD predictions only (no model retraining
beyond the cheap classical OOF, all local). Adds, on top of AUC/logloss/Brier/ECE/BSS/cAUC:

  1. Brier decomposition (URR): Brier = Reliability - Resolution + Uncertainty
     - Reliability  = calibration error (lower=better)
     - Resolution   = how sharply the model separates outcomes (higher=better) -> "skill/sharpness"
     - Uncertainty  = irreducible base-rate variance (dataset constant)
  2. Sharpness = std(predictions): a calibrated model that always predicts the base rate is
     useless; sharpness rewards confident-when-justified.
  3. Calibration slope & intercept (bin-FREE, the TRIPOD standard): fit outcome ~ a + b*logit(p);
     b=1 & a=0 = perfect; b<1 = overconfident, b>1 = underconfident.
  4. Bin-free calibration (Balance-score spirit, Choi et al. 2023 arXiv:2309.06248): adaptive
     equal-mass ECE + KS calibration error (no equal-width binning artifact).
  5. Comeback / tail honesty: low/high-probability reliability + per-round minimum win-prob of
     the eventual winner (how often, and how calibrated-ly, the model "writes off" the winner).

Usage: python src/models/extended_metrics.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict, ece, bss  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "extended_metrics.csv"
EPS = 1e-7


# ----------------------------- metric functions -----------------------------
def brier_decomp(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); n = len(y); base = y.mean()
    rel = res = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        nk = m.sum(); pk = p[m].mean(); ok = y[m].mean()
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - base) ** 2
    return rel / n, res / n, float(base * (1 - base))   # reliability, resolution, uncertainty


def sharpness(p):
    return float(np.std(p))


def cal_slope_intercept(y, p):
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS)))  # logit
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=2000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])                 # slope, intercept


def adaptive_ece(y, p, bins=10):
    order = np.argsort(p); ys = y[order]; ps = p[order]
    e = 0.0; n = len(y)
    for chunk in np.array_split(np.arange(n), bins):
        if len(chunk):
            e += len(chunk) / n * abs(ps[chunk].mean() - ys[chunk].mean())
    return float(e)


def ks_cal_error(y, p):
    order = np.argsort(p)
    return float(np.max(np.abs(np.cumsum(y[order] - p[order]) / len(y))))


def base_metrics(y, p, contested):
    return dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), ECE=ece(y, p), BSS=bss(y, p),
                cAUC=roc_auc_score(y[contested], p[contested]))


def extended(y, p, contested):
    rel, res, unc = brier_decomp(y, p)
    slope, intercept = cal_slope_intercept(y, p)
    m = base_metrics(y, p, contested)
    m.update(REL=rel, RES=res, UNC=unc, sharp=sharpness(p),
             cal_slope=slope, cal_intercept=intercept,
             adaptECE=adaptive_ece(y, p), KScal=ks_cal_error(y, p))
    return m


# ----------------------------- comeback / tail -----------------------------
def tail_report(y, p):
    print("\n--- TAIL CALIBRATION (are extreme calls honest?) ---")
    print(f"  {'pred bin':12s} {'n':>7} {'mean pred':>10} {'actual CT win':>14}")
    for lo, hi in [(0, .05), (.05, .1), (.1, .2), (.8, .9), (.9, .95), (.95, 1.0001)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  [{lo:.2f},{hi:.2f}) {m.sum():>7} {p[m].mean():>10.3f} {y[m].mean():>14.3f}")


def comeback_report(df, p):
    print("\n--- COMEBACK (eventual winner's lowest win-prob per round) ---")
    d = df.select(["match_id", "round_num", "ct_won"]).with_columns(p=pl.Series(p))
    d = d.with_columns(wp_winner=pl.when(pl.col("ct_won") == 1).then(pl.col("p")).otherwise(1 - pl.col("p")))
    per = d.group_by(["match_id", "round_num"]).agg(minwp=pl.col("wp_winner").min())
    mw = per["minwp"].to_numpy()
    for thr in (0.2, 0.1, 0.05):
        print(f"  eventual winner dipped <= {thr:.0%} at some point in {100*(mw<=thr).mean():5.1f}% of rounds")
    print(f"  median winner-min-WP {np.median(mw):.3f}; mean {mw.mean():.3f}")
    # comeback calibration: among snapshots where a side is <=10%, how often does it come back?
    low = p <= 0.10
    print(f"  snapshots with CT<=10%: n={low.sum()}, actual CT-win {df['ct_won'].to_numpy()[low].mean():.3f} "
          f"(calibrated ~{p[low].mean():.3f})")


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy().astype(float)
    contested = ((df["ct_players_alive"] == df["t_players_alive"])
                 & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()

    rows = []
    grid = [("logreg", "A"), ("logreg", "E"), ("logreg", "EFB2"),
            ("xgb", "A"), ("xgb", "EFB2"), ("lgbm", "EFB2"),
            ("catboost", "EFB2"), ("rf", "EFB2")]
    preds = {}
    for mdl, st in grid:
        p, _ = oof_predict(df, FEATURE_SETS[st], mdl)
        preds[(mdl, st)] = p
        rows.append({"model": f"{mdl}/{st}", **extended(y, p, contested)})

    # deep OOF (saved on Betty, downloaded) — align to df by (match_id, tick)
    key = df.select(["match_id", "tick"]).with_columns(_i=pl.int_range(0, df.height))
    for name, path in [("transformer", ROOT / "outputs" / "oof_transformer.parquet"),
                       ("tcn", ROOT / "outputs" / "oof_tcn.parquet")]:
        if path.exists():
            d = pl.read_parquet(path)
            pcol = [c for c in d.columns if c.startswith("p_")][0]
            j = key.join(d.select(["match_id", "tick", pcol]), on=["match_id", "tick"], how="inner")
            idx = j["_i"].to_numpy(); pv = j[pcol].to_numpy()
            preds[(name, "deep")] = (idx, pv)
            rows.append({"model": name, **extended(y[idx], pv, contested[idx])})

    # soft-vote ensemble (xgb EFB2 + logreg EFB2 + deep), on the intersection
    if ("transformer", "deep") in preds and ("tcn", "deep") in preds:
        it, pt = preds[("transformer", "deep")]; ic, pc = preds[("tcn", "deep")]
        common = np.intersect1d(it, ic)
        mp = {i: v for i, v in zip(it, pt)}; mc = {i: v for i, v in zip(ic, pc)}
        ptr = np.array([mp[i] for i in common]); ptc = np.array([mc[i] for i in common])
        ens = (preds[("logreg", "EFB2")][common] + preds[("xgb", "EFB2")][common] + ptr + ptc) / 4
        rows.append({"model": "SOFT-VOTE(4)", **extended(y[common], ens, contested[common])})

    tbl = pl.DataFrame(rows)
    cols = ["model", "AUC", "logloss", "brier", "REL", "RES", "UNC", "sharp",
            "cal_slope", "cal_intercept", "ECE", "adaptECE", "KScal", "BSS", "cAUC"]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4, tbl_width_chars=200):
        print(tbl.select(cols))
    OUT.parent.mkdir(parents=True, exist_ok=True); tbl.write_csv(OUT)
    print(f"\nwrote {OUT}")

    # comeback/tail on the headline model
    print("\n" + "=" * 60 + "\nHEADLINE MODEL: logreg / EFB2")
    tail_report(y, preds[("logreg", "EFB2")])
    comeback_report(df, preds[("logreg", "EFB2")])


if __name__ == "__main__":
    main()
