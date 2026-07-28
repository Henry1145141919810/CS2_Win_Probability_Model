"""F5 — the firepower recovery figure (paper v2).

Compares the 2026 out-of-time holdout under two firepower constructions:
  - same-year (original, BROKEN): 2026 demos looked up stats that did not exist -> ~30% zero-filled
  - lagged-2025 (FIXED): 2026 demos look up the previous season's stats (leak-free, deployment-real)

Two panels:
  (left)  EFB2 out-of-time AUC per model: broken vs fixed, with the EB2 (no-firepower) band.
  (right) calibration intercept per model: broken (over-confident, negative) vs fixed (~base-rate).

Reads outputs/holdout_2026.csv (broken) and outputs/holdout_2026_lag2025.csv (fixed).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
BROKEN = ROOT / "outputs" / "holdout_2026.csv"
FIXED = ROOT / "outputs" / "holdout_2026_lag2025.csv"
OUT = ROOT / "outputs" / "figures" / "paper" / "F5_firepower_recovery.png"

BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "font.family": "sans-serif", "font.size": 9, "axes.grid": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
NICE = {"logreg": "Logistic", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost", "rf": "Random forest"}


def col(df, model, st, c):
    r = df.filter((pl.col("model") == model) & (pl.col("set") == st))
    return float(r[c][0]) if r.height else np.nan


def main():
    b = pl.read_csv(BROKEN)
    f = pl.read_csv(FIXED)
    x = np.arange(len(MODELS))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ---- left: EFB2 out-of-time AUC, broken vs fixed ----
    ob = [col(b, m, "EFB2", "out_AUC") for m in MODELS]
    of = [col(f, m, "EFB2", "out_AUC") for m in MODELS]
    inb = [col(b, m, "EFB2", "in_AUC") for m in MODELS]
    ax1.bar(x - w/2, ob, w, color=RED, edgecolor=SURF, linewidth=1.2, label="same-year (broken)")
    ax1.bar(x + w/2, of, w, color=AQUA, edgecolor=SURF, linewidth=1.2, label="lagged-2025 (fixed)")
    # in-sample marker per model
    ax1.scatter(x, inb, marker="_", s=420, color=INK, linewidths=1.8, zorder=5, label="in-time (5-fold OOF)")
    # EB2 no-firepower reference band (mean out-of-time across models)
    eb2 = np.nanmean([col(f, m, "EB2", "out_AUC") for m in MODELS])
    ax1.axhline(eb2, color=BLUE, ls="--", lw=1.3, zorder=1)
    ax1.text(len(MODELS) - 0.5, eb2 + 0.0012, f"EB2 (no firepower) {eb2:.3f} — still ahead",
             ha="right", va="bottom", fontsize=8, color=BLUE)
    for xi, (vb, vf) in enumerate(zip(ob, of)):
        ax1.text(xi - w/2, vb - 0.004, f"{vb:.3f}", ha="center", va="top", fontsize=7.2, color="white")
        ax1.text(xi + w/2, vf + 0.001, f"{vf:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK2)
    ax1.set_xticks(x); ax1.set_xticklabels([NICE[m] for m in MODELS], fontsize=8, rotation=15)
    ax1.set_ylabel("2026 out-of-time AUC (set EFB2)")
    lo = min(np.nanmin(ob), np.nanmin(of)) - 0.008
    ax1.set_ylim(lo, max(np.nanmax(inb), eb2) + 0.006)
    ax1.set_title("The lagged fix repairs most of the collapse — but firepower still trails\n"
                  "red = broken same-year run; aqua = fixed, yet stays below the no-firepower line",
                  fontsize=10, loc="left")
    ax1.legend(loc="lower right", frameon=False, fontsize=8)

    # ---- right: calibration intercept, broken vs fixed ----
    ib = [col(b, m, "EFB2", "cal_intercept") for m in MODELS]
    iff = [col(f, m, "EFB2", "cal_intercept") for m in MODELS]
    ax2.bar(x - w/2, ib, w, color=RED, edgecolor=SURF, linewidth=1.2, label="same-year (broken)")
    ax2.bar(x + w/2, iff, w, color=AQUA, edgecolor=SURF, linewidth=1.2, label="lagged-2025 (fixed)")
    ax2.axhline(0, color=AXIS, lw=1)
    for xi, (vb, vf) in enumerate(zip(ib, iff)):
        ax2.text(xi - w/2, vb + (0.008 if vb >= 0 else -0.008), f"{vb:+.2f}",
                 ha="center", va="bottom" if vb >= 0 else "top", fontsize=7.2, color=INK2)
        ax2.text(xi + w/2, vf + (0.008 if vf >= 0 else -0.008), f"{vf:+.2f}",
                 ha="center", va="bottom" if vf >= 0 else "top", fontsize=7.2, color=INK2)
    ax2.set_xticks(x); ax2.set_xticklabels([NICE[m] for m in MODELS], fontsize=8, rotation=15)
    ax2.set_ylabel("calibration intercept on 2026 holdout")
    ax2.set_title("Calibration, however, fully recovers\n"
                  "broken run is over-confident in the CTs (negative intercept); fixed sits near 0",
                  fontsize=10, loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
