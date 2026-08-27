"""PILOT: does the win-probability curve move correctly while a defuse is running?

NOT a paper experiment. Trains on the 2026 set only, split in half BY MATCH, because the
training set has not been re-parsed with is_defusing yet. Logistic regression: cheap, low
variance, and the point here is direction, not a headline number.

Two things are deliberately NOT claimed:
  - overall AUC. The feature fires on ~1% of rows; it cannot move a global metric, and
    reading "no AUC gain" as "no value" would be the wrong conclusion.
  - anything transferable. 13-14 matches cannot estimate 70 features well, and half of the
    2026 out-of-time holdout is being used for TRAINING here, which is why this writes to
    its own output and must never be quoted next to the touch-once holdout result.

What it DOES report, which is the actual question (goal b):
  - log-loss / Brier restricted to the rows where a defuse is in progress
  - mean predicted P(CT win) bucketed by defuse progress -- does the curve climb?
  - the same, split by whether the attempt was COMPLETED or INTERRUPTED. A model that has
    merely memorised "defusing => CT wins" looks identical on both; a model that has learned
    the timer separates them.

Usage:
    python src/models/pilot_defuse_lr.py
    python src/models/pilot_defuse_lr.py --sets A,AD,ATF,ATFD
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.assemble_defuse_pilot import PILOT_SETS  # noqa: E402

DATA = ROOT / "data" / "holdout2026" / "pilot_defuse.parquet"


def fit_predict(df: pl.DataFrame, cols: list[str], folds: list[np.ndarray]) -> np.ndarray:
    """2-fold cross-fitting by match: every row is predicted by a model that never saw it."""
    X = np.nan_to_num(df[cols].to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    oof = np.zeros(len(y))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="A,AD,ATF,ATFD")
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pl.read_parquet(args.data)
    matches = sorted(df["match_id"].unique().to_list())
    rng = np.random.default_rng(args.seed)
    rng.shuffle(matches)
    half = len(matches) // 2
    groups = [set(matches[:half]), set(matches[half:])]
    mid = df["match_id"].to_numpy()
    folds = [np.where(np.isin(mid, list(g)))[0] for g in groups]
    print(f"{df.height} rows / {len(matches)} matches -> halves of "
          f"{len(groups[0])} and {len(groups[1])} matches "
          f"({len(folds[0])} / {len(folds[1])} rows)")

    y = df["ct_won"].to_numpy()
    live = df["defuse_in_progress"].to_numpy() == 1
    done = (df["defuse_attempts_so_far"].to_numpy() > 0)
    print(f"defusing rows: {live.sum()} ({live.mean() * 100:.2f}%)\n")

    preds = {}
    print(f"{'set':<6}{'#feat':>6}{'AUC':>8}{'logloss':>9}{'Brier':>8}   |"
          f"{'  defusing rows: logloss':>24}{'Brier':>8}{'mean p':>8}")
    for name in args.sets.split(","):
        cols = [c for c in PILOT_SETS[name] if c in df.columns]
        p = fit_predict(df, cols, folds)
        preds[name] = p
        row = (f"{name:<6}{len(cols):>6}{roc_auc_score(y, p):>8.4f}"
               f"{log_loss(y, p):>9.4f}{brier_score_loss(y, p):>8.4f}   |")
        row += (f"{log_loss(y[live], p[live], labels=[0, 1]):>24.4f}"
                f"{brier_score_loss(y[live], p[live]):>8.4f}{p[live].mean():>8.3f}")
        print(row)

    # the actual question: does predicted WP climb with defuse progress?
    frac = df["defuse_progress_frac"].to_numpy()
    edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\nmean predicted P(CT win) by defuse progress (defusing rows only)")
    print(f"{'progress':<12}{'n':>5}{'actual':>9}" + "".join(f"{s:>9}" for s in preds))
    for lo, hi in edges:
        m = live & (frac >= lo) & (frac < hi)
        if not m.sum():
            continue
        line = f"{f'{lo:.0%}-{hi if hi <= 1 else 1:.0%}':<12}{m.sum():>5}{y[m].mean():>9.3f}"
        line += "".join(f"{preds[s][m].mean():>9.3f}" for s in preds)
        print(line)

    # completed vs interrupted: a model that only memorised "defusing => win" cannot separate
    comp = df["defuse_in_progress"].to_numpy() * 0
    print(f"\nsame rows split by how the attempt ENDED")
    print(f"{'outcome':<12}{'n':>5}{'actual':>9}" + "".join(f"{s:>9}" for s in preds))
    for label, m in (("completed", live & (y == 1)), ("interrupted", live & (y == 0))):
        if not m.sum():
            continue
        line = f"{label:<12}{m.sum():>5}{y[m].mean():>9.3f}"
        line += "".join(f"{preds[s][m].mean():>9.3f}" for s in preds)
        print(line)
    print("\n(round outcome is a proxy for attempt outcome here: a completed defuse IS a CT win)")


if __name__ == "__main__":
    main()
