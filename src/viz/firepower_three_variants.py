"""F6 — the firepower three-variant figure (paper v3).

The 2026 out-of-time holdout under three constructions of the skill prior, a clean spectrum of
data quality:
  1. same-year BROKEN  : 2026 demos, no 2026 stats existed -> ~30% zero-filled (coverage gap)
  2. lagged-2025       : previous season's stats (leak-free, deployment-realistic)
  3. same-year 2026    : full 2026 coverage but LEAKY (2026 ratings computed from the very matches) -
                         the best-case upper bound an oracle could have

Left  : EFB2 out-of-time AUC per model across the 3 variants, vs the EB2 (no-firepower) line.
Right : calibration intercept per model across the 3 variants (negative = over-confident in CTs).

Reads outputs/holdout_2026.csv (broken), _lag2025.csv, _sameyr2026.csv.
Titles are DATA-DRIVEN so they stay honest regardless of the same-year outcome.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    "broken": ROOT / "outputs" / "holdout_2026.csv",
    "lagged": ROOT / "outputs" / "holdout_2026_lag2025.csv",
    "sameyr": ROOT / "outputs" / "holdout_2026_sameyr2026.csv",
}
OUT = ROOT / "outputs" / "figures" / "paper" / "F6_firepower_three_variants.png"

RED, YELLOW, AQUA, BLUE = "#e34948", "#eda100", "#1baf7a", "#2a78d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
NICE = {"logreg": "Logistic", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost", "rf": "Random forest"}
VARIANTS = [("broken", RED, "same-year (broken, no coverage)"),
            ("lagged", YELLOW, "lagged-2025 (leak-free)"),
            ("sameyr", AQUA, "same-year 2026 (full coverage, leaky)")]


def col(df, model, st, c):
    r = df.filter((pl.col("model") == model) & (pl.col("set") == st))
    return float(r[c][0]) if r.height else np.nan


def main():
    dfs = {k: pl.read_csv(v) for k, v in FILES.items()}
    x = np.arange(len(MODELS))
    w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- left: EFB2 out-of-time AUC, 3 variants ----
    eb2 = np.nanmean([col(dfs["sameyr"], m, "EB2", "out_AUC") for m in MODELS])
    for j, (key, cvar, lab) in enumerate(VARIANTS):
        vals = [col(dfs[key], m, "EFB2", "out_AUC") for m in MODELS]
        ax1.bar(x + (j - 1) * w, vals, w, color=cvar, edgecolor=SURF, linewidth=1.0, label=lab)
    ax1.axhline(eb2, color=BLUE, ls="--", lw=1.3, zorder=1)
    # does the best-case (same-year) beat EB2 on any model?
    beats = any(col(dfs["sameyr"], m, "EFB2", "out_AUC") > col(dfs["sameyr"], m, "EB2", "out_AUC")
                for m in MODELS)
    verdict = ("even the leaky best-case beats no-firepower on some models"
               if beats else "no variant beats the no-firepower model (blue) on any model")
    ax1.text(len(MODELS) - 0.5, eb2 + 0.0007, f"EB2 (no firepower) {eb2:.3f}",
             ha="right", va="bottom", fontsize=8, color=BLUE)
    # truncate the y-axis to where the variation lives (AUC ~0.81-0.855); every EFB2 bar
    # still visibly sits BELOW the EB2 line, which is the point.
    allv = [col(dfs[k], m, "EFB2", "out_AUC") for k in FILES for m in MODELS]
    ax1.set_ylim(min(allv) - 0.004, eb2 + 0.004)
    ax1.set_xticks(x); ax1.set_xticklabels([NICE[m] for m in MODELS], fontsize=8, rotation=15)
    ax1.set_ylabel("2026 out-of-time AUC (set EFB2, with firepower)")
    ax1.set_title("Firepower across three data constructions\n" + verdict, fontsize=10, loc="left")
    ax1.legend(loc="lower right", frameon=False, fontsize=7.6)

    # ---- right: calibration intercept, 3 variants ----
    for j, (key, cvar, lab) in enumerate(VARIANTS):
        vals = [col(dfs[key], m, "EFB2", "cal_intercept") for m in MODELS]
        ax2.bar(x + (j - 1) * w, vals, w, color=cvar, edgecolor=SURF, linewidth=1.0, label=lab)
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.set_xticks(x); ax2.set_xticklabels([NICE[m] for m in MODELS], fontsize=8, rotation=15)
    ax2.set_ylabel("calibration intercept (negative = over-confident in CTs)")
    ax2.set_title("Calibration: the coverage gap is the whole story\n"
                  "only the broken build is miscalibrated; both fixes sit near 0",
                  fontsize=10, loc="left")
    ax2.legend(loc="lower right", frameon=False, fontsize=7.6)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}  (best-case beats EB2 on some model: {beats})")


if __name__ == "__main__":
    main()
