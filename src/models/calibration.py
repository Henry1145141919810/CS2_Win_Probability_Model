"""Calibration test: are the predicted probabilities HONEST? (reliability diagram + ECE)

AUC only checks ranking. This checks whether "the model said 70%" actually means CT won
~70% of the time. Produces:
  - outputs/figures/calibration.png : reliability diagram (predicted % vs observed win-rate)
  - ECE (expected calibration error) + Brier + log-loss, for logistic E and xgb ET.
Uses out-of-fold predictions from the same 5-fold GroupKFold.

Usage: python src/models/calibration.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures" / "calibration.png"
TAC = TACTICAL_COLS + BOMB_COLS
MODELS = {
    "logistic E": (ECONOMY_COLS + MAPCONTROL_COLS + TAC, "logreg"),
    "xgb ET": (ECONOMY_COLS + MAPCONTROL_COLS + TAC + TERRITORY_COLS, "xgb"),
}


def make_model(kind):
    if kind == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000))
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.03,
                         min_child_weight=10, reg_lambda=10.0, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, tree_method="hist",
                         eval_metric="logloss", random_state=0)


def oof(df, cols, y, groups, kind):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_model(kind)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, acc, w = p[m].mean(), y[m].mean(), m.mean()
        e += w * abs(conf - acc)
        rows.append((conf, acc, int(m.sum())))
    return e, rows


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, (cols, kind) in MODELS.items():
        p = oof(df, cols, y, groups, kind)
        e, rows = ece(y, p, bins=10)
        conf = [r[0] for r in rows]
        acc = [r[1] for r in rows]
        ax.plot(conf, acc, "o-", label=f"{name}  (ECE={e:.3f}, Brier={brier_score_loss(y,p):.3f}, "
                                       f"logloss={log_loss(y,p):.3f})")
        print(f"{name:12s} ECE={e:.4f}  Brier={brier_score_loss(y,p):.4f}  "
              f"logloss={log_loss(y,p):.4f}")
        print("   pred% -> actual% (n):  " +
              "  ".join(f"{c:.2f}->{a:.2f}({n})" for c, a, n in rows))
    ax.set_xlabel("predicted P(CT win)")
    ax.set_ylabel("observed CT win-rate")
    ax.set_title("Reliability diagram — are the probabilities honest?\n"
                 "(on the diagonal = honest; ECE = avg gap)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=120); plt.close(fig)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
