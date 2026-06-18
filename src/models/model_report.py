"""Unified STANDARD EVALUATION BATTERY — run this for every new model/method.

The project's consistency contract (docs/methodology.md "Standard evaluation protocol",
and the standing instruction): no model is "done" until it has the SAME interpretation +
uncertainty + calibration evidence as the others. This one command produces all of it for
any (model, feature-set) and writes a single markdown report so results stay comparable:

  1. Metrics      — AUC / log-loss / Brier (primary) + ECE / BSS / contested-AUC (complementary)
  2. Uncertainty  — match-level block-bootstrap 95% CI on AUC (+ pointer to the win-prob CI band)
  3. Calibration  — ECE + reliability curve PNG
  4. Interpretation — model-agnostic permutation importance (top features)

Usage:
  python src/models/model_report.py --models logreg,xgb --sets E,ET
  python src/models/model_report.py --models lgbm --sets E --bootstrap 300
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
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS, MAPCONTROL_LOS_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402
from models.train_pipeline import oof_predict, ece, bss, block_bootstrap  # noqa: E402
from models.permutation_importance import perm_importance, PILLAR  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUTDIR = ROOT / "outputs"
TAC = TACTICAL_COLS + BOMB_COLS
SETS = {
    "A": ECONOMY_COLS,
    "B": ECONOMY_COLS + MAPCONTROL_COLS,
    "G": ECONOMY_COLS + MAPCONTROL_LOS_COLS,
    "Terr": ECONOMY_COLS + TERRITORY_COLS,
    "E": ECONOMY_COLS + MAPCONTROL_COLS + TAC,
    "ET": ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS,
}


def _reliability(y, p, path, label):
    edges = np.linspace(0, 1, 11)
    conf, acc = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum():
            conf.append(p[m].mean()); acc.append(y[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(conf, acc, "o-", lw=2)
    ax.set_xlabel("predicted P(CT win)"); ax.set_ylabel("observed win-rate")
    ax.set_title(f"Reliability — {label}"); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="logreg,xgb")
    ap.add_argument("--sets", default="E,ET")
    ap.add_argument("--bootstrap", type=int, default=300)
    args = ap.parse_args()
    models = args.models.split(",")
    sets = args.sets.split(",")

    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    contested = ((df["ct_players_alive"] == df["t_players_alive"])
                 & ((df["ct_equipment_value"] - df["t_equipment_value"]).abs() <= 1500)).to_numpy()
    (OUTDIR / "figures").mkdir(parents=True, exist_ok=True)
    oof_a = {m: oof_predict(df, SETS["A"], m)[0] for m in models}  # baseline for the diff CI

    lines = ["# Standard model report", "", f"Data: {len(y)} snapshots, "
             f"{df['match_id'].n_unique()} matches, base P(CT win)={y.mean():.3f}. "
             f"Bootstrap B={args.bootstrap}.", ""]
    for mdl in models:
        for st in sets:
            cols = SETS[st]
            oof, _ = oof_predict(df, cols, mdl)
            auc = roc_auc_score(y, oof)
            cauc = roc_auc_score(y[contested], oof[contested])
            bs = block_bootstrap(df, oof_a[mdl], oof, B=args.bootstrap)
            d = bs["diff"]
            relp = OUTDIR / "figures" / f"reliability_{mdl}_{st}.png"
            _reliability(y, oof, relp, f"{mdl} {st}")
            imp = perm_importance(df, cols, y, df["match_id"].to_numpy(), mdl)
            top = np.argsort(-imp)[:8]
            print(f"{mdl} {st}: AUC {auc:.4f} cAUC {cauc:.4f} ECE {ece(y,oof):.4f} "
                  f"BSS {bss(y,oof):.3f} | E-A diff {d[0]:+.4f} ({d[1]:+.4f},{d[2]:+.4f})")
            lines += [
                f"## {mdl} — set {st}", "",
                f"- **Metrics:** AUC {auc:.4f} · log-loss {log_loss(y,oof):.4f} · "
                f"Brier {brier_score_loss(y,oof):.4f} · ECE {ece(y,oof):.4f} · "
                f"BSS {bss(y,oof):.3f} · contested-AUC {cauc:.4f}",
                f"- **Uncertainty:** AUC lift vs A = {d[0]:+.4f} (95% CI {d[1]:+.4f}, {d[2]:+.4f})"
                f"{'  [significant]' if d[1]*d[2] > 0 else ''}; "
                f"per-round win-prob CI band via `winprob_chart.py --model {mdl}`.",
                f"- **Calibration:** reliability curve → `figures/{relp.name}` (ECE {ece(y,oof):.4f}).",
                "- **Interpretation (permutation importance, top 8):** "
                + ", ".join(f"{cols[j]} ({PILLAR.get(cols[j],'?')[:4]} {imp[j]:+.3f})" for j in top),
                "",
            ]
    rep = OUTDIR / "model_report.md"
    rep.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {rep}")


if __name__ == "__main__":
    main()
