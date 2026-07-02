"""Ensemble (architecture #9) — XGBoost + deep models, on out-of-fold predictions.

Completes the 9-architecture matrix. Combines the CLASSICAL models' OOF (computed here from
training_dataset.parquet) with the DEEP models' OOF (saved on Betty via `--save-oof` and copied
back), aligned by (match_id, tick). Reports the full metric suite + B=500 match-bootstrap CIs for:
each base model, the SOFT-VOTE (mean), and a cross-fitted LOGISTIC STACK meta-learner.

Run LOCALLY (needs the classical env). Example:
  # after copying the deep OOF parquets into outputs/:
  python src/models/ensemble_oof.py --classical xgb,logreg --oof outputs/oof_transformer.parquet,outputs/oof_tcn.parquet
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict, ece, bss  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
BOOT = ["AUC", "logloss", "brier", "ECE", "BSS", "cAUC"]


def boot(y, p, groups, contested, B=500, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(g, []).append(i)
    keys = list(by); arrs = [np.asarray(by[k]) for k in keys]
    acc = {m: [] for m in BOOT}
    for _ in range(B):
        s = np.concatenate([arrs[j] for j in rng.integers(0, len(keys), len(keys))])
        ys, ps = y[s], p[s]
        if ys.min() == ys.max():
            continue
        acc["AUC"].append(roc_auc_score(ys, ps)); acc["logloss"].append(log_loss(ys, ps, labels=[0, 1]))
        acc["brier"].append(brier_score_loss(ys, ps)); acc["ECE"].append(ece(ys, ps)); acc["BSS"].append(bss(ys, ps))
        cs = ys[contested[s]]
        if contested[s].sum() > 50 and cs.min() != cs.max():
            acc["cAUC"].append(roc_auc_score(ys[contested[s]], ps[contested[s]]))
    return {m: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for m, v in acc.items() if v}


def report(name, y, p, groups, contested, B):
    ci = boot(y, p, groups, contested, B)
    line = (f"{name:18s} AUC {roc_auc_score(y,p):.4f}  logloss {log_loss(y,p):.4f}  "
            f"brier {brier_score_loss(y,p):.4f}  ECE {ece(y,p):.4f}  BSS {bss(y,p):.3f}  "
            f"cAUC {roc_auc_score(y[contested],p[contested]):.4f}")
    print(line)
    print(f"    AUC 95% CI ({ci['AUC'][1]:.4f}, {ci['AUC'][2]:.4f})  "
          f"cAUC CI ({ci['cAUC'][1]:.4f}, {ci['cAUC'][2]:.4f})  ECE CI ({ci['ECE'][1]:.4f}, {ci['ECE'][2]:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", default="xgb", help="comma list of classical models for the ensemble")
    ap.add_argument("--oof", default="outputs/oof_transformer.parquet",
                    help="comma list of deep OOF parquets (match_id,tick,y,p_*) copied from Betty")
    ap.add_argument("--set", default="EFB2")
    ap.add_argument("--bootstrap", type=int, default=500)
    args = ap.parse_args()

    df = pl.read_parquet(DATA)
    contested_full = ((df["ct_players_alive"] == df["t_players_alive"])
                      & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()
    base = df.select(["match_id", "tick", "ct_won"]).with_columns(
        contested=pl.Series(contested_full))
    pred_cols = []
    for m in args.classical.split(","):
        p, _ = oof_predict(df, FEATURE_SETS[args.set], m)
        base = base.with_columns(pl.Series(f"p_{m}", p)); pred_cols.append(f"p_{m}")
        print(f"classical {m} ({args.set}) OOF: AUC {roc_auc_score(df['ct_won'].to_numpy(), p):.4f}")

    for path in [s for s in args.oof.split(",") if s]:
        d = pl.read_parquet(path)
        pcol = [c for c in d.columns if c.startswith("p_")][0]
        base = base.join(d.select(["match_id", "tick", pcol]), on=["match_id", "tick"], how="inner")
        pred_cols.append(pcol)
        print(f"joined deep OOF {Path(path).name} -> {pcol} ({base.height} rows after join)")

    y = base["ct_won"].to_numpy().astype(float)
    g = base["match_id"].to_numpy()
    cont = base["contested"].to_numpy()
    P = np.column_stack([base[c].to_numpy() for c in pred_cols])
    print(f"\nensemble members: {pred_cols}; {len(y)} aligned snapshots\n")

    print("=== base members ===")
    for j, c in enumerate(pred_cols):
        report(c, y, P[:, j], g, cont, args.bootstrap)
    print("\n=== ensembles ===")
    report("SOFT-VOTE", y, P.mean(1), g, cont, args.bootstrap)
    stack = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(P, y, g):
        stack[te] = LogisticRegression(max_iter=2000).fit(P[tr], y[tr]).predict_proba(P[te])[:, 1]
    report("LOGISTIC-STACK", y, stack, g, cont, args.bootstrap)
    print("\nclassical best logreg EFB2 AUC 0.8515 | TCN 0.8488 | GAT 0.8465 (all OOF)")


if __name__ == "__main__":
    main()
