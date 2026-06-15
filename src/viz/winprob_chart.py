"""Per-second round win-probability chart (BLAST.tv-style headline figure).

Trains the full-feature model (set E) on ALL OTHER matches, then plots P(CT win) second
-by-second for one round of a held-out match, annotated with kills and the bomb plant.
Auto-selects the most dramatic round (largest probability swing) in the chosen match.

Usage:
    python src/viz/winprob_chart.py --match faze-vs-g2-m1-inferno
    python src/viz/winprob_chart.py --match cloud9-vs-vitality-m1-inferno --round 7
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
KILLS_DIR = ROOT / "data" / "parquet" / "kills"
ROUNDS_DIR = ROOT / "data" / "parquet" / "rounds"
OUT = ROOT / "outputs" / "figures"
TICKRATE = 64
COLS = ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS


def _fit_predict(df, match_id):
    from xgboost import XGBClassifier
    tr = df.filter(pl.col("match_id") != match_id)
    te = df.filter(pl.col("match_id") == match_id)
    Xtr = np.nan_to_num(tr[COLS].to_numpy().astype(float)); ytr = tr["ct_won"].to_numpy()
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
                      colsample_bytree=0.9, n_jobs=-1, eval_metric="logloss", random_state=0)
    m.fit(Xtr, ytr)
    Xte = np.nan_to_num(te[COLS].to_numpy().astype(float))
    return te.with_columns(pl.Series("p_ct", m.predict_proba(Xte)[:, 1]))


def _pick_round(pred):
    swings = (pred.group_by("round_num")
              .agg((pl.col("p_ct").max() - pl.col("p_ct").min()).alias("swing")))
    return int(swings.sort("swing", descending=True)["round_num"][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True)
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(DATA)
    pred = _fit_predict(df, args.match)
    rn = args.round or _pick_round(pred)
    r = pred.filter(pl.col("round_num") == rn).sort("tick")
    if r.height == 0:
        print(f"no snapshots for round {rn}"); return

    rounds = pl.read_parquet(ROUNDS_DIR / f"{args.match}.parquet")
    rr = rounds.filter(pl.col("round_num") == rn).row(0, named=True)
    fe = rr["freeze_end"]
    t = (r["tick"].to_numpy() - fe) / TICKRATE
    p = r["p_ct"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.fill_between(t, 0.5, p, where=(p >= 0.5), color="#2b72d4", alpha=0.25)
    ax.fill_between(t, 0.5, p, where=(p < 0.5), color="#f08c1e", alpha=0.25)
    ax.plot(t, p, color="black", lw=2)

    # kill + plant annotations
    kills = pl.read_parquet(KILLS_DIR / f"{args.match}.parquet").filter(pl.col("round_num") == rn)
    for k in kills.iter_rows(named=True):
        kt = (k["tick"] - fe) / TICKRATE
        vs = str(k.get("victim_side", "")).lower()
        col = "#f08c1e" if vs.startswith("t") else "#2b72d4"  # a T died -> good for CT
        ax.axvline(kt, color=col, alpha=0.35, lw=1)
    bp = rr.get("bomb_plant")
    if bp is not None:
        ax.axvline((bp - fe) / TICKRATE, color="red", lw=1.6, ls=":")
        ax.text((bp - fe) / TICKRATE, 0.02, " bomb plant", color="red", fontsize=9)

    ax.set_xlabel("seconds since freeze-end"); ax.set_ylabel("P(CT win round)")
    ax.set_ylim(0, 1); ax.set_xlim(0, t.max())
    ax.set_title(f"{args.match}  round {rn}  (winner: {rr['winner'].upper()})  "
                 f"— Model E live win probability\nvertical lines = kills (blue: T died, "
                 f"orange: CT died), red dotted = bomb plant", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"winprob_{args.match}_r{rn}.png"
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"saved {out}  (round {rn}, {r.height} snapshots, swing={p.max()-p.min():.2f})")


if __name__ == "__main__":
    main()
