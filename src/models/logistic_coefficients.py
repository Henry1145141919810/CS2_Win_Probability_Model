"""Explicit feature list + standardized logistic-regression coefficients.

Logistic regression is the headline interpretable model (it ~ties tuned XGBoost, so the
signal is near-linear). Because every feature is z-scored before the fit (StandardScaler),
the coefficients are directly comparable in magnitude: a coefficient of +0.5 means "a
one-standard-deviation increase in this feature multiplies the CT-win odds by exp(0.5)."
This is the table that goes in the paper to show WHAT the model learned and that the signs
are sensible (more CT equipment -> CT favored; bomb planted -> CT disfavored; etc.).

Outputs:
  - console table (sorted by |coef|) per feature set
  - outputs/logistic_coefficients.csv  (feature, coef, odds_ratio, pillar) for set ET
  - outputs/figures/logistic_coefficients.png  (horizontal bar, color by pillar)

Usage: python src/models/logistic_coefficients.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS, BOMB_LIVE_COLS, BOMB_DEFUSE_COLS  # noqa: E402
from features.firepower import FIREPOWER_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUTCSV = ROOT / "outputs" / "logistic_coefficients.csv"
OUTFIG = ROOT / "outputs" / "figures" / "logistic_coefficients.png"

# map every feature to its pillar (for coloring / grouping)
PILLAR = {}
for c in ECONOMY_COLS:
    PILLAR[c] = "economy"
for c in MAPCONTROL_COLS:
    PILLAR[c] = "voronoi"
for c in TERRITORY_COLS:
    PILLAR[c] = "territory"
for c in TACTICAL_COLS + BOMB_COLS + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS:
    PILLAR[c] = "tactical"
for c in FIREPOWER_COLS:
    PILLAR[c] = "firepower"
PILLAR_COLOR = {"economy": "#4477aa", "voronoi": "#ee6677", "territory": "#228833",
                "tactical": "#ccbb44", "firepower": "#aa3377"}

SETS = {
    "A (economy)": ECONOMY_COLS,
    "F (econ+firepower)": ECONOMY_COLS + FIREPOWER_COLS,
    "EF (4 pillars)": ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS + FIREPOWER_COLS,
    "EFB2 (all pillars+bomb)": (ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL_COLS + BOMB_COLS
                                + TERRITORY_COLS + FIREPOWER_COLS + BOMB_LIVE_COLS + BOMB_DEFUSE_COLS),
}


def fit_coefs(df, cols, y):
    X = df.select(cols).to_numpy().astype(float)
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    Xs = sc.fit_transform(imp.fit_transform(X))
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Xs, y)
    return lr.coef_[0], lr.intercept_[0]


def main():
    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    print(f"data: {len(y)} snapshots, base rate P(CT win)={y.mean():.3f}\n")

    et_rows = None
    for label, cols in SETS.items():
        coef, b0 = fit_coefs(df, cols, y)
        order = np.argsort(-np.abs(coef))
        print(f"=== {label} | intercept={b0:+.3f} (logit at feature means) ===")
        print(f"  {'feature':28s} {'coef(z)':>9} {'odds_x':>7}  pillar")
        for i in order:
            print(f"  {cols[i]:28s} {coef[i]:>+9.3f} {np.exp(coef[i]):>7.3f}  "
                  f"{PILLAR.get(cols[i],'?')}")
        print()
        if label.startswith("EFB2"):
            et_rows = [{"feature": cols[i], "coef_standardized": float(coef[i]),
                        "odds_ratio_per_sd": float(np.exp(coef[i])),
                        "pillar": PILLAR.get(cols[i], "?")} for i in order]

    # CSV + figure for set ET (the full interpretable model)
    OUTCSV.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(et_rows).write_csv(OUTCSV)
    print(f"wrote {OUTCSV}")

    # figure: top-25 by |coef|, horizontal bars colored by pillar
    top = et_rows[:25][::-1]
    names = [r["feature"] for r in top]
    vals = [r["coef_standardized"] for r in top]
    cols_ = [PILLAR_COLOR[r["pillar"]] for r in top]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(range(len(top)), vals, color=cols_)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("standardized logistic coefficient  (+ favors CT win, per 1 SD)")
    ax.set_title("Logistic regression — top 25 features (set ET)\n"
                 "z-scored inputs, so bar length = effect size; color = pillar")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PILLAR_COLOR.values()]
    ax.legend(handles, PILLAR_COLOR.keys(), loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(OUTFIG, dpi=120); plt.close(fig)
    print(f"wrote {OUTFIG}")


if __name__ == "__main__":
    main()
