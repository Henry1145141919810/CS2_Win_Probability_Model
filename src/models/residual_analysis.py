"""Study 1 - Residual analysis: the beyond-economy signal, isolated.

Two analyses:

(1) Stacked-logit (offset) model. Fit economy alone (OOF), form the economy logit z = logit(p_A),
    and compare a model on [z] alone (recovers economy) against a model on [z, X_spatial]. The spatial
    /tactical/bomb features can only act through the residual economy leaves behind, so the improvement
    is the beyond-economy contribution. Reported across models and metrics, in-time (OOF) and
    out-of-time (2026 holdout), with paired match-level bootstrap CIs on the difference.

(2) Frisch-Waugh-Lovell orthogonalization (diagnostic). Partial economy out of both the outcome and
    each spatial feature (OOF), then rank features by the partial correlation of the feature residual
    with the outcome residual. This separates features that carry information orthogonal to economy
    from features that are largely economy re-encoded.

Usage: python src/models/residual_analysis.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import (  # noqa: E402
    FEATURE_SETS, ECONOMY_COLS, MAPCONTROL_COLS, TACTICAL, BOMB_LIVE_COLS, BOMB_DEFUSE_COLS,
    oof_predict, make_model, ece, bss)

# Non-economy, non-firepower block = EB2 minus economy (Voronoi + tactical + bomb-live + defuse-race)
X_SPATIAL = MAPCONTROL_COLS + TACTICAL + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS
MODELS = ["logreg", "xgb", "lgbm", "catboost"]
TRAIN = ROOT / "data" / "training_dataset.parquet"
HOLD = ROOT / "data" / "test_dataset_2026_sameyr.parquet"
OUTCSV = ROOT / "outputs" / "residual_analysis.csv"
OUTFWL = ROOT / "outputs" / "residual_beyond_economy_importance.csv"
B = 500


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def contested_mask(df):
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def metrics(y, p, cont):
    return dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                brier=brier_score_loss(y, p), ECE=ece(y, p), BSS=bss(y, p),
                cAUC=roc_auc_score(y[cont], p[cont]) if cont.sum() > 50 else np.nan)


def paired_boot(df, y, pa, pb, cont, B=B, seed=0):
    """Paired match-level bootstrap: 95% CI of (metric_b - metric_a) for AUC, logloss, cAUC."""
    rng = np.random.default_rng(seed)
    g = df["match_id"].to_numpy()
    by = {}
    for i, m in enumerate(g):
        by.setdefault(m, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    dA, dL, dC = [], [], []
    for _ in range(B):
        s = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys = y[s]
        if ys.min() == ys.max():
            continue
        dA.append(roc_auc_score(ys, pb[s]) - roc_auc_score(ys, pa[s]))
        dL.append(log_loss(ys, pb[s], labels=[0, 1]) - log_loss(ys, pa[s], labels=[0, 1]))
        cs = cont[s]
        if cs.sum() > 50 and ys[cs].min() != ys[cs].max():
            dC.append(roc_auc_score(ys[cs], pb[s][cs]) - roc_auc_score(ys[cs], pa[s][cs]))
    q = lambda v: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan,)*3
    return {"dAUC": q(dA), "dlogloss": q(dL), "dcAUC": q(dC)}


def oof_from_matrix(X, y, groups, model_name):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        oof[te] = make_model(model_name).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def main():
    tr = pl.read_parquet(TRAIN)
    te = pl.read_parquet(HOLD)
    y = tr["ct_won"].to_numpy().astype(float)
    yte = te["ct_won"].to_numpy().astype(float)
    g = tr["match_id"].to_numpy()
    cont_tr, cont_te = contested_mask(tr), contested_mask(te)
    Xsp_tr = np.nan_to_num(tr.select(X_SPATIAL).to_numpy().astype(float))
    Xsp_te = np.nan_to_num(te.select(X_SPATIAL).to_numpy().astype(float))
    print(f"train {tr.height} snaps / {tr['match_id'].n_unique()} matches; "
          f"X_spatial = {len(X_SPATIAL)} cols (Voronoi+tactical+bomb, NO economy, NO firepower)\n")

    rows = []
    for mdl in MODELS:
        # --- economy alone (OOF) -> economy logit z ---
        pa_oof, _ = oof_predict(tr, ECONOMY_COLS, mdl)
        z_tr = logit(pa_oof)
        # base = model on [z] alone (recovers economy); +spatial = model on [z, X_spatial]
        base_oof = oof_from_matrix(z_tr.reshape(-1, 1), y, g, mdl)
        sp_oof = oof_from_matrix(np.column_stack([z_tr, Xsp_tr]), y, g, mdl)
        mb, ms = metrics(y, base_oof, cont_tr), metrics(y, sp_oof, cont_tr)
        cis = paired_boot(tr, y, base_oof, sp_oof, cont_tr)

        # --- out-of-time: fit on ALL training, predict 2026 holdout ---
        # economy fit -> holdout logit
        Xe_tr = np.nan_to_num(tr.select(ECONOMY_COLS).to_numpy().astype(float))
        Xe_te = np.nan_to_num(te.select(ECONOMY_COLS).to_numpy().astype(float))
        econ = make_model(mdl).fit(Xe_tr, y)
        z_te = logit(econ.predict_proba(Xe_te)[:, 1])
        z_tr_full = logit(econ.predict_proba(Xe_tr)[:, 1])
        base_te = make_model(mdl).fit(z_tr_full.reshape(-1, 1), y).predict_proba(z_te.reshape(-1, 1))[:, 1]
        sp_te = make_model(mdl).fit(np.column_stack([z_tr_full, Xsp_tr]), y)\
            .predict_proba(np.column_stack([z_te, Xsp_te]))[:, 1]
        mbo, mso = metrics(yte, base_te, cont_te), metrics(yte, sp_te, cont_te)
        cis_o = paired_boot(te, yte, base_te, sp_te, cont_te)

        for scope, mba, msa, ci in [("in_time", mb, ms, cis), ("out_time", mbo, mso, cis_o)]:
            rows.append({"model": mdl, "scope": scope,
                         "base_AUC": mba["AUC"], "sp_AUC": msa["AUC"],
                         "dAUC": ci["dAUC"][0], "dAUC_lo": ci["dAUC"][1], "dAUC_hi": ci["dAUC"][2],
                         "base_logloss": mba["logloss"], "sp_logloss": msa["logloss"],
                         "dlogloss": ci["dlogloss"][0], "dll_lo": ci["dlogloss"][1], "dll_hi": ci["dlogloss"][2],
                         "base_brier": mba["brier"], "sp_brier": msa["brier"],
                         "base_BSS": mba["BSS"], "sp_BSS": msa["BSS"],
                         "base_cAUC": mba["cAUC"], "sp_cAUC": msa["cAUC"],
                         "dcAUC": ci["dcAUC"][0], "dcAUC_lo": ci["dcAUC"][1], "dcAUC_hi": ci["dcAUC"][2]})
        sig = "SIG" if (cis["dAUC"][1] > 0) else "ns "
        print(f"{mdl:9s} IN  base AUC {mb['AUC']:.4f} -> +spatial {ms['AUC']:.4f} "
              f"| dAUC {cis['dAUC'][0]:+.4f} ({cis['dAUC'][1]:+.4f},{cis['dAUC'][2]:+.4f}) {sig} "
              f"| dcAUC {cis['dcAUC'][0]:+.4f} ({cis['dcAUC'][1]:+.4f},{cis['dcAUC'][2]:+.4f})")
        print(f"{'':9s} OUT base AUC {mbo['AUC']:.4f} -> +spatial {mso['AUC']:.4f} "
              f"| dAUC {cis_o['dAUC'][0]:+.4f} ({cis_o['dAUC'][1]:+.4f},{cis_o['dAUC'][2]:+.4f})")

    pl.DataFrame(rows).write_csv(OUTCSV)
    print(f"\nwrote {OUTCSV}")

    # --- (2) FWL orthogonalization: partial economy out, rank spatial features by partial corr ---
    print("\n=== FWL beyond-economy importance (partial corr of feature_res with outcome_res) ===")
    # outcome residual y_res = y - p_A (use logreg economy OOF)
    pa_oof, _ = oof_predict(tr, ECONOMY_COLS, "logreg")
    y_res = y - pa_oof
    Xe = np.nan_to_num(tr.select(ECONOMY_COLS).to_numpy().astype(float))
    fwl = []
    for j, col in enumerate(X_SPATIAL):
        s = Xsp_tr[:, j]
        # OOF E[s | economy]
        s_hat = np.zeros(len(s))
        for trn, tev in GroupKFold(5).split(Xe, y, g):
            s_hat[tev] = LinearRegression().fit(Xe[trn], s[trn]).predict(Xe[tev])
        s_res = s - s_hat
        if s_res.std() < 1e-9:
            pcorr = 0.0
        else:
            pcorr = float(np.corrcoef(s_res, y_res)[0, 1])
        # fraction of variance explained by economy (how "economy-like" the feature is)
        r2_econ = 1 - (s_res.var() / s.var()) if s.var() > 1e-12 else np.nan
        fwl.append({"feature": col, "partial_corr_with_outcome": pcorr,
                    "abs_partial_corr": abs(pcorr), "economy_r2": r2_econ})
    fwl_df = pl.DataFrame(fwl).sort("abs_partial_corr", descending=True)
    fwl_df.write_csv(OUTFWL)
    with pl.Config(tbl_rows=20, fmt_str_lengths=40):
        print(fwl_df.head(15))
    print(f"\nwrote {OUTFWL}")


if __name__ == "__main__":
    main()
