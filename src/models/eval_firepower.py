"""Evaluate Pillar 3 (firepower) — overall lift + where it matters (clutch 1vN).

Mirrors eval_bomb_features.py. Firepower is a pre-round skill prior; the hypothesis (Leu's
doc) is that it helps most (a) in CONTESTED rounds where economy fails, and (b) in CLUTCH
(1vN) situations where the lone survivor's individual skill dominates. We check both, across
all 5 models, with bootstrap CIs, plus permutation importance of the firepower columns.

Usage: python src/models/eval_firepower.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from features.firepower import FIREPOWER_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece, bss, block_bootstrap  # noqa: E402
from models.permutation_importance import perm_importance  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
TAC = TACTICAL_COLS + BOMB_COLS
A = ECONOMY_COLS
F = ECONOMY_COLS + FIREPOWER_COLS                              # economy + firepower (isolate)
E = ECONOMY_COLS + MAPCONTROL_COLS + TAC                       # 3-pillar workhorse
EF = ECONOMY_COLS + MAPCONTROL_COLS + TAC + FIREPOWER_COLS     # all 4 pillars
MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]


def m(y, p):
    return f"AUC {roc_auc_score(y,p):.4f} logloss {log_loss(y,p):.4f} ECE {ece(y,p):.4f} BSS {bss(y,p):.3f}"


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    ca = df["ct_players_alive"].to_numpy(); ta = df["t_players_alive"].to_numpy()
    clutch = ((ca == 1) | (ta == 1)) & (ca >= 1) & (ta >= 1)   # someone in a 1vN
    contested = ((ca == ta) & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500).to_numpy())
    print(f"data {len(y)} snaps; clutch(1vN) {clutch.mean():.1%}; contested {contested.mean():.1%}")
    print(f"models {MODELS}; sets A / F(+firepower) / E(3-pillar) / EF(4-pillar)\n")

    for mdl in MODELS:
        pA, _ = oof_predict(df, A, mdl)
        pF, _ = oof_predict(df, F, mdl)
        pE, _ = oof_predict(df, E, mdl)
        pEF, _ = oof_predict(df, EF, mdl)
        print(f"### {mdl}")
        print(f"  A   : {m(y,pA)} | cAUC {roc_auc_score(y[contested],pA[contested]):.4f}")
        print(f"  F   : {m(y,pF)} | cAUC {roc_auc_score(y[contested],pF[contested]):.4f}")
        print(f"  E   : {m(y,pE)} | cAUC {roc_auc_score(y[contested],pE[contested]):.4f}")
        print(f"  EF  : {m(y,pEF)} | cAUC {roc_auc_score(y[contested],pEF[contested]):.4f}")
        bF = block_bootstrap(df, pA, pF, B=300)["diff"]
        bEF = block_bootstrap(df, pE, pEF, B=300)["diff"]
        print(f"  F - A   AUC {bF[0]:+.4f} (95% CI {bF[1]:+.4f},{bF[2]:+.4f}){'  [sig]' if bF[1]*bF[2]>0 else ''}")
        print(f"  EF - E  AUC {bEF[0]:+.4f} (95% CI {bEF[1]:+.4f},{bEF[2]:+.4f}){'  [sig]' if bEF[1]*bEF[2]>0 else ''}")
        # where firepower helps: clutch & contested subsets (F vs A)
        for name, msk in [("clutch(1vN)", clutch), ("contested", contested)]:
            if msk.sum() > 200:
                print(f"    {name:12s} n={msk.sum():>6}: AUC A {roc_auc_score(y[msk],pA[msk]):.4f} "
                      f"-> F {roc_auc_score(y[msk],pF[msk]):.4f} "
                      f"-> EF {roc_auc_score(y[msk],pEF[msk]):.4f}")
        print()

    print("### permutation importance of firepower features (xgb, set EF)")
    imp = perm_importance(df, EF, y, groups, "xgb")
    order = np.argsort(-imp); rank = {EF[order[r]]: r + 1 for r in range(len(order))}
    for c in FIREPOWER_COLS:
        print(f"  {c:24s} AUC-drop {imp[EF.index(c)]:+.4f}  rank #{rank[c]}/{len(EF)}")


if __name__ == "__main__":
    main()
