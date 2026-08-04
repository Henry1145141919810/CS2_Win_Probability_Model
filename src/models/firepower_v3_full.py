"""Full firepower-v3 benchmark: all models x all firepower encodings x full metric battery + CIs.

Clean comparison that isolates the FIREPOWER ENCODING on a common EB2 base:
    EB2            = no firepower (reference)
    EB2 + v2       = raw summed HLTV stats (firepower v2)
    EB2 + v3-log2  = team-ranking-weighted, weight = 1/log2(rank+1)
    EB2 + v3-inv   = weight = 1/rank
    EB2 + v3-linear= weight = (31-rank)/30

For each (encoding, model): 5-fold GroupKFold OOF, full metrics (AUC, log-loss, Brier, ECE, BSS,
contested-AUC), and a PAIRED match-level bootstrap 95% CI of the AUC difference vs EB2 (does firepower
help?) and vs EB2+v2 (does v3 beat v2?).

Reuses Leu's proxy weight application (v3_col = v2_col x team_weight, exact because all alive players on
a side share one team weight). Usage: python src/models/firepower_v3_full.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, make_model, ece, bss  # noqa: E402
from features.firepower import FIREPOWER_COLS  # v2, 20 cols  # noqa: E402
from features.firepower_v3 import FIREPOWER_COLS_V3  # v3, 18 cols  # noqa: E402
from models.eval_firepower_v3 import build_match_weights, apply_v3_weights  # noqa: E402

TRAIN = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "firepower_v3_full.csv"
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
EB2 = FEATURE_SETS["EB2"]
B = 500


def contested_mask(df):
    return ((df["ct_players_alive"] == df["t_players_alive"])
            & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()


def oof(df, cols, mdl):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    y = df["ct_won"].to_numpy(); g = df["match_id"].to_numpy()
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = make_model(mdl).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def paired_ci(df, y, pa, pb, B=B, seed=0):
    """95% CI of AUC(pb) - AUC(pa), paired match-level bootstrap."""
    rng = np.random.default_rng(seed)
    g = df["match_id"].to_numpy(); by = {}
    for i, m in enumerate(g):
        by.setdefault(m, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    d = []
    for _ in range(B):
        s = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys = y[s]
        if ys.min() != ys.max():
            d.append(roc_auc_score(ys, pb[s]) - roc_auc_score(ys, pa[s]))
    return (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (np.nan,)*3


def main():
    df = pl.read_parquet(TRAIN)
    y = df["ct_won"].to_numpy().astype(float)
    cont = contested_mask(df)
    print(f"{df.height} snaps / {df['match_id'].n_unique()} matches\n")

    # v3 columns via Leu's proxy (needs match team weights from round-1 ticks)
    from features.firepower import _year_by_demo, DEFAULT_YEAR
    ybm = {m: (_year_by_demo().get(m) or DEFAULT_YEAR) for m in df["match_id"].unique().to_list()}
    mw = build_match_weights(df["match_id"].to_list(), ybm)
    dfw = {f: apply_v3_weights(df, mw, f) for f in ("log2", "inv", "linear")}

    encodings = {
        "EB2 (no fp)": (df, EB2),
        "EB2+v2": (df, EB2 + FIREPOWER_COLS),
        "EB2+v3-log2": (dfw["log2"], EB2 + FIREPOWER_COLS_V3),
        "EB2+v3-inv": (dfw["inv"], EB2 + FIREPOWER_COLS_V3),
        "EB2+v3-linear": (dfw["linear"], EB2 + FIREPOWER_COLS_V3),
    }

    # cache EB2 and v2 OOF per model for paired CIs
    base_oof = {m: oof(df, EB2, m) for m in MODELS}
    v2_oof = {m: oof(df, EB2 + FIREPOWER_COLS, m) for m in MODELS}

    rows = []
    for enc, (d, cols) in encodings.items():
        for mdl in MODELS:
            if enc == "EB2 (no fp)":
                p = base_oof[mdl]
            elif enc == "EB2+v2":
                p = v2_oof[mdl]
            else:
                p = oof(d, cols, mdl)
            m = dict(AUC=roc_auc_score(y, p), logloss=log_loss(y, p, labels=[0, 1]),
                     brier=brier_score_loss(y, p), ECE=ece(y, p), BSS=bss(y, p),
                     cAUC=roc_auc_score(y[cont], p[cont]))
            dvE = paired_ci(df, y, base_oof[mdl], p)      # vs EB2
            dv2 = paired_ci(df, y, v2_oof[mdl], p)        # vs EB2+v2
            rows.append({"encoding": enc, "model": mdl, **{k: round(v, 4) for k, v in m.items()},
                         "dAUC_vs_EB2": dvE[0], "vsEB2_lo": dvE[1], "vsEB2_hi": dvE[2],
                         "dAUC_vs_v2": dv2[0], "vsv2_lo": dv2[1], "vsv2_hi": dv2[2]})
            sig = "SIG+" if dvE[1] > 0 else ("SIG-" if dvE[2] < 0 else "ns")
            print(f"{enc:15s} {mdl:9s} AUC {m['AUC']:.4f} cAUC {m['cAUC']:.3f} "
                  f"| vs EB2 {dvE[0]:+.4f} ({dvE[1]:+.4f},{dvE[2]:+.4f}) {sig} "
                  f"| vs v2 {dv2[0]:+.4f} ({dv2[1]:+.4f},{dv2[2]:+.4f})")
    pl.DataFrame(rows).write_csv(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
