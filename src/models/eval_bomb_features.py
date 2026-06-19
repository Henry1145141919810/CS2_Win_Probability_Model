"""Evaluate the new bomb-state / bomb-local-control features (v2 dataset).

Compares E (existing) vs E + bomb-live features, with the standard battery, AND specifically
on the subsets where these features are designed to help:
  - post-plant / retake (bomb_state == planted)
  - dropped bomb (loose C4 scramble)
  - endgame (planted or 30s+)  <- does it lower the endgame log loss?
Plus permutation importance of the new features (do they actually get used?).

Usage: python src/models/eval_bomb_features.py
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
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS, BOMB_LIVE_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece, bss, block_bootstrap  # noqa: E402
from models.permutation_importance import perm_importance  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"  # canonical (now includes bomb-live cols)
TAC = TACTICAL_COLS + BOMB_COLS
E = ECONOMY_COLS + MAPCONTROL_COLS + TAC
EB = E + BOMB_LIVE_COLS                       # + bomb-live
ETB = E + TERRITORY_COLS + BOMB_LIVE_COLS     # full + bomb-live


def metrics(y, p):
    return f"AUC {roc_auc_score(y,p):.4f}  logloss {log_loss(y,p):.4f}  ECE {ece(y,p):.4f}  BSS {bss(y,p):.3f}"


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    planted = (df["bomb_state"].to_numpy() == 2)
    dropped = (df["bomb_dropped"].to_numpy() == 1)
    endgame = planted | (df["time_elapsed_sec"].to_numpy() > 30)
    contested = ((df["ct_players_alive"] == df["t_players_alive"])
                 & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()
    MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
    print(f"v2 data: {len(y)} snaps, {df['match_id'].n_unique()} matches; "
          f"planted {planted.mean():.1%}, dropped {dropped.mean():.1%}")
    print(f"benchmarking models: {MODELS}; sets: E vs E+bomb-live\n")

    for mdl in MODELS:
        pE, _ = oof_predict(df, E, mdl)
        pEB, _ = oof_predict(df, EB, mdl)
        print(f"### {mdl}")
        print(f"  E      : {metrics(y, pE)}  | contested-AUC {roc_auc_score(y[contested], pE[contested]):.4f}")
        print(f"  E+bomb : {metrics(y, pEB)}  | contested-AUC {roc_auc_score(y[contested], pEB[contested]):.4f}")
        bs = block_bootstrap(df, pE, pEB, B=300)["diff"]
        print(f"  EB - E overall AUC diff = {bs[0]:+.4f} (95% CI {bs[1]:+.4f}, {bs[2]:+.4f})"
              f"{'  [significant]' if bs[1]*bs[2] > 0 else ''}")
        # subset log-loss (where the features should help)
        for name, m in [("post-plant/retake", planted), ("dropped-bomb", dropped),
                        ("endgame(plant/30+)", endgame)]:
            if m.sum() > 200:
                print(f"    {name:20s} n={m.sum():>6}: logloss E {log_loss(y[m],pE[m]):.4f} "
                      f"-> E+bomb {log_loss(y[m],pEB[m]):.4f}  "
                      f"(AUC {roc_auc_score(y[m],pE[m]):.4f} -> {roc_auc_score(y[m],pEB[m]):.4f})")
        print()

    # permutation importance of the NEW features within E+bomb (xgb): do they get used?
    print("### permutation importance of new bomb features (xgb, E+bomb)")
    imp = perm_importance(df, EB, y, groups, "xgb")
    order = np.argsort(-imp)
    rank = {EB[order[r]]: r + 1 for r in range(len(order))}
    for c in BOMB_LIVE_COLS:
        print(f"  {c:26s} AUC-drop {imp[EB.index(c)]:+.4f}  rank #{rank[c]}/{len(EB)}")


if __name__ == "__main__":
    main()
