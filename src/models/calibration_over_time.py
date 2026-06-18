"""Calibration over time — are the probabilities honest at EACH second of the round?

Pooled ECE can hide phase-specific miscalibration (e.g. honest early, overconfident late).
This bins snapshots by time-into-round and reports ECE + Brier per window, for the headline
models, plus an ECE-vs-time plot. Part of the standard calibration protocol
(docs/methodology.md) — run for every new model.

Usage: python src/models/calibration_over_time.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUTFIG = ROOT / "outputs" / "figures" / "calibration_over_time.png"
TAC = TACTICAL_COLS + BOMB_COLS
MODELS = {  # label -> (model_kind, cols)
    "logreg E": ("logreg", ECONOMY_COLS + MAPCONTROL_COLS + TAC),
    "xgb ET": ("xgb", ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS),
}
EDGES = [0, 5, 10, 15, 20, 25, 30, 1e9]
LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"]


def main():
    df = pl.read_parquet(DATA)
    tsec = df["time_elapsed_sec"].to_numpy()
    fig, ax = plt.subplots(figsize=(9, 5))
    print(f"{'model':10s} {'window(s)':>9} {'n':>7} {'ECE':>7} {'Brier':>7}")
    for label, (kind, cols) in MODELS.items():
        oof, y = oof_predict(df, cols, kind)
        eces, centers = [], []
        for k in range(len(EDGES) - 1):
            m = (tsec >= EDGES[k]) & (tsec < EDGES[k + 1])
            if m.sum() < 200:
                eces.append(np.nan); centers.append(k); continue
            e = ece(y[m], oof[m]); b = brier_score_loss(y[m], oof[m])
            eces.append(e); centers.append(k)
            print(f"{label:10s} {LABELS[k]:>9} {m.sum():>7} {e:>7.4f} {b:>7.4f}")
        ax.plot(range(len(LABELS)), eces, marker="o", lw=2, label=label)
        print()
    ax.axhline(0.02, color="grey", ls="--", lw=0.8, label="ECE=0.02 (well-calibrated)")
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_xlabel("seconds into round"); ax.set_ylabel("ECE (lower = more honest)")
    ax.set_title("Calibration over time — is the model honest at each phase of the round?")
    ax.legend(fontsize=9)
    OUTFIG.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=120); plt.close(fig)
    print(f"saved {OUTFIG}")


if __name__ == "__main__":
    main()
