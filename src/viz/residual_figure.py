"""F7 - Residual analysis (Study 1): beyond-economy signal.

Left  : beyond-economy AUC gain (spatial block added on top of the economy logit) per model, in-time
        and out-of-time, with paired match-level bootstrap CIs. All intervals exclude zero.
Right : FWL beyond-economy importance. Top features by |partial correlation| of the economy-residualised
        feature with the economy-residualised outcome, shaded by how much economy already explains the
        feature (economy_r2; darker = more orthogonal to economy = genuinely new information).

Reads outputs/residual_analysis.csv, outputs/residual_beyond_economy_importance.csv.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "paper" / "F7_residual.png"
BLUE, AQUA, INK, INK2, MUTED = "#2a78d6", "#1baf7a", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# blue sequential ramp for economy-orthogonality
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})
NICE = {"logreg": "Logistic", "xgb": "XGBoost", "lgbm": "LightGBM", "catboost": "CatBoost"}


def ramp(v):  # v in [0,1] -> ramp colour
    return RAMP[int(np.clip(v, 0, 1) * (len(RAMP) - 1))]


def main():
    d = pl.read_csv(ROOT / "outputs" / "residual_analysis.csv")
    fwl = pl.read_csv(ROOT / "outputs" / "residual_beyond_economy_importance.csv")
    models = ["logreg", "xgb", "lgbm", "catboost"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: beyond-economy dAUC per model, in vs out ----
    x = np.arange(len(models)); w = 0.38
    for j, (scope, col, lab) in enumerate([("in_time", BLUE, "in-time (OOF)"),
                                           ("out_time", AQUA, "out-of-time (2026)")]):
        vals, los, his = [], [], []
        for m in models:
            r = d.filter((pl.col("model") == m) & (pl.col("scope") == scope))
            vals.append(float(r["dAUC"][0])); los.append(float(r["dAUC_lo"][0])); his.append(float(r["dAUC_hi"][0]))
        vals = np.array(vals)
        ax1.bar(x + (j - 0.5) * w, vals, w, color=col, edgecolor=SURF, linewidth=1.0, label=lab)
        ax1.errorbar(x + (j - 0.5) * w, vals, yerr=[vals - los, np.array(his) - vals],
                     fmt="none", ecolor=INK2, elinewidth=1, capsize=3)
    ax1.axhline(0, color=AXIS, lw=1)
    ax1.set_xticks(x); ax1.set_xticklabels([NICE[m] for m in models], fontsize=8.5)
    ax1.set_ylabel("beyond-economy ΔAUC (spatial added on economy logit)")
    ax1.set_title("Signal added beyond economy, controlling for economy", fontsize=10.5, loc="left")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")

    # ---- right: FWL importance, top features, shaded by orthogonality ----
    top = fwl.head(12).reverse()
    feats = top["feature"].to_list()
    pc = top["abs_partial_corr"].to_numpy()
    r2 = top["economy_r2"].to_numpy()
    cols = [ramp(1 - v) for v in r2]      # darker = lower economy_r2 = more orthogonal
    yy = np.arange(len(feats))
    ax2.barh(yy, pc, color=cols, edgecolor=SURF, linewidth=0.8)
    ax2.set_yticks(yy); ax2.set_yticklabels(feats, fontsize=7.4)
    ax2.set_xlabel("|partial correlation| with outcome (economy removed)")
    ax2.set_title("Beyond-economy feature importance\n"
                  "(darker = more orthogonal to economy)", fontsize=10.5, loc="left")
    ax2.grid(axis="y", visible=False)
    # colourbar-style legend
    import matplotlib.cm as cm
    from matplotlib.colors import ListedColormap, Normalize
    sm = cm.ScalarMappable(cmap=ListedColormap(RAMP), norm=Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax2, fraction=0.035, pad=0.02)
    cb.set_label("economy-orthogonality (1 - economy_r²)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
