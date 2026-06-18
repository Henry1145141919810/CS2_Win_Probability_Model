"""Worst-prediction audit — WHERE is the model confidently wrong, and is there a pattern?

Takes OOF predictions for the headline model and finds the most confidently-wrong snapshots
(highest log-loss: predicted >0.8 for the side that lost, or <0.2 for the side that won).
Then categorizes them by situation (post-plant, man-advantage, eco mismatch, contested) to
see whether errors are systematic (a fixable blind spot) or just the irreducible noise of
even rounds. Part of the standard error-analysis protocol (docs/methodology.md).

Usage: python src/models/worst_prediction_audit.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import oof_predict  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "worst_predictions.csv"
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS


def main():
    df = pl.read_parquet(DATA)
    oof, y = oof_predict(df, COLS, "logreg")
    eps = 1e-7
    ll = -(y * np.log(oof + eps) + (1 - y) * np.log(1 - oof + eps))  # per-snapshot log-loss
    d = df.with_columns(p_ct=pl.Series(oof), logloss=pl.Series(ll))

    # situation flags
    d = d.with_columns(
        post_plant=pl.col("bomb_planted") == 1,
        man_adv=(pl.col("ct_players_alive") - pl.col("t_players_alive")).abs() >= 2,
        eco_mismatch=(pl.col("ct_equipment_value") - pl.col("t_equipment_value")).abs() > 3000,
        contested=((pl.col("ct_players_alive") == pl.col("t_players_alive"))
                   & ((pl.col("ct_equipment_value") - pl.col("t_equipment_value")).abs() <= 1500)),
        confidently_wrong=((pl.col("p_ct") > 0.8) & (pl.col("ct_won") == 0))
        | ((pl.col("p_ct") < 0.2) & (pl.col("ct_won") == 1)),
    )

    print(f"snapshots {d.height}; mean log-loss {ll.mean():.4f}")
    cw = d.filter(pl.col("confidently_wrong"))
    print(f"\nconfidently-wrong snapshots: {cw.height} ({100*cw.height/d.height:.2f}%)")

    print("\n=== mean log-loss by situation (higher = harder for the model) ===")
    for flag in ["contested", "post_plant", "man_adv", "eco_mismatch"]:
        sub = d.filter(pl.col(flag))
        rest = d.filter(~pl.col(flag))
        cwr = 100 * sub.filter(pl.col("confidently_wrong")).height / max(sub.height, 1)
        print(f"  {flag:14s} n={sub.height:>7}  logloss {sub['logloss'].mean():.4f}  "
              f"(rest {rest['logloss'].mean():.4f})  conf-wrong {cwr:.2f}%")

    # share of confidently-wrong errors that fall in each situation
    print("\n=== of the confidently-wrong snapshots, what situation are they? ===")
    for flag in ["contested", "post_plant", "man_adv", "eco_mismatch"]:
        share = 100 * cw.filter(pl.col(flag)).height / max(cw.height, 1)
        print(f"  {flag:14s} {share:5.1f}%")

    # save the 500 worst with context
    worst = (d.sort("logloss", descending=True).head(500)
             .select(["match_id", "round_num", "tick", "time_elapsed_sec", "ct_won", "p_ct",
                      "logloss", "ct_players_alive", "t_players_alive", "ct_equipment_value",
                      "t_equipment_value", "bomb_planted", "ct_voronoi_control_pct"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    worst.write_csv(OUT)
    print(f"\nwrote 500 worst to {OUT}")


if __name__ == "__main__":
    main()
