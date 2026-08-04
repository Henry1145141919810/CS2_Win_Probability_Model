"""F8 - The contested-round ceiling (Study 2).

Left  : information-saturation curve. Contested-AUC across representations (economy -> spatial -> bomb
        -> firepower -> TCN -> Transformer). Flat near 0.585: no representation breaks the ceiling.
Right : by alive-state. Model contested-AUC and the model-free matching oracle-AUC (empirical ceiling)
        for 1v1..5v5 even, plus a man-advantage control. 5v5-even (71% of the contested set) sits at a
        coin flip; the control reaches ~0.83, validating the estimator.

Reads outputs/contested_saturation.csv, contested_by_alivestate.csv, bayes_error_matching.csv.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "figures" / "paper" / "F8_contested_ceiling.png"
BLUE, AQUA, YELLOW, RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "font.family": "sans-serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def main():
    sat = pl.read_csv(ROOT / "outputs" / "contested_saturation.csv")
    al = pl.read_csv(ROOT / "outputs" / "contested_by_alivestate.csv")
    be = pl.read_csv(ROOT / "outputs" / "bayes_error_matching.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- left: saturation curve ----
    order = ["A", "E", "EB2", "EFB2", "TCN", "Transformer"]
    labs = {"A": "economy", "E": "+spatial", "EB2": "+bomb", "EFB2": "+firepower",
            "TCN": "TCN", "Transformer": "Transf."}
    xs, ys, los, his, cols = [], [], [], [], []
    for i, r in enumerate(order):
        row = sat.filter(pl.col("repr") == r)
        if not row.height:
            continue
        xs.append(i); ys.append(float(row["cAUC"][0]))
        los.append(float(row["lo"][0])); his.append(float(row["hi"][0]))
        cols.append(BLUE if row["family"][0] == "classical" else AQUA)
    xs = np.array(xs)
    ax1.errorbar(xs, ys, yerr=[np.array(ys) - los, np.array(his) - ys], fmt="none",
                 ecolor=AXIS, elinewidth=1.2, capsize=3, zorder=1)
    ax1.scatter(xs, ys, c=cols, s=70, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax1.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax1.text(0.05, 0.505, "coin flip", fontsize=8, color=MUTED)
    for x, yv in zip(xs, ys):
        ax1.text(x, yv + 0.004, f"{yv:.3f}", ha="center", va="bottom", fontsize=7.6, color=INK2)
    ax1.set_xticks(xs); ax1.set_xticklabels([labs[order[i]] for i in xs], fontsize=8, rotation=20)
    ax1.set_ylim(0.49, 0.63)
    ax1.set_ylabel("contested-AUC (equal alive & even economy)")
    ax1.set_title("Information saturation on contested rounds", fontsize=10.5, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="classical", ms=8),
         plt.Line2D([], [], marker="o", ls="", color=AQUA, label="deep", ms=8)]
    ax1.legend(handles=h, frameon=False, fontsize=8.5, loc="upper right")

    # ---- right: model vs matching-oracle ceiling by alive-state ----
    states = ["1v1", "2v2", "3v3", "4v4", "5v5"]
    be_map = {"1v1 even": "1v1", "2v2 even": "2v2", "3v3 even": "3v3", "4v4 even": "4v4",
              "5v5 even": "5v5", "CONTROL 1-man-adv": "control"}
    cats = states + ["control"]
    x = np.arange(len(cats)); w = 0.38
    model_auc = {r["state"]: (r["cAUC"], r["lo"], r["hi"]) for r in al.iter_rows(named=True)}
    orac = {}
    for r in be.iter_rows(named=True):
        s = be_map.get(r["category"])
        if s:
            orac[s] = (r["oracle_AUC"], r["oracle_lo"], r["oracle_hi"])
    # control has no model_auc row (uneven) -> only oracle
    mv = [model_auc.get(s, (np.nan, np.nan, np.nan)) for s in cats]
    ov = [orac.get(s, (np.nan, np.nan, np.nan)) for s in cats]
    m_pt = np.array([v[0] for v in mv]); o_pt = np.array([v[0] for v in ov])
    ax2.bar(x - w/2, m_pt, w, color=BLUE, edgecolor=SURF, linewidth=1.0, label="model (xgb EB2)")
    ax2.bar(x + w/2, o_pt, w, color=AQUA, edgecolor=SURF, linewidth=1.0,
            label="matching oracle (empirical ceiling)")
    ax2.errorbar(x - w/2, m_pt, yerr=[m_pt - [v[1] for v in mv], [v[2] for v in mv] - m_pt],
                 fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax2.errorbar(x + w/2, o_pt, yerr=[o_pt - [v[1] for v in ov], [v[2] for v in ov] - o_pt],
                 fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax2.axhline(0.5, color=MUTED, ls="--", lw=1)
    ax2.text(len(cats) - 0.5, 0.505, "coin flip", fontsize=8, color=MUTED, ha="right")
    ax2.set_xticks(x); ax2.set_xticklabels([c if c != "control" else "man-adv\n(control)" for c in cats],
                                           fontsize=8)
    ax2.set_ylim(0.45, 0.9)
    ax2.set_ylabel("AUC")
    ax2.set_title("Model AUC vs matching-oracle ceiling, by alive-state", fontsize=10.5, loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
