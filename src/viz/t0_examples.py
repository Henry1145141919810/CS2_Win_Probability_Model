"""Does the win-probability curve favour the stronger team at t=0?  Examples + the aggregate answer.

Panels 1-4: live per-second P(CT win) from the production model (logistic EB2, out-of-fold) for four
hand-picked rounds, with kills (dot colour = side that lost a player), the bomb plant, and the round-
start priors at t=0 (the model's own first snapshot vs the stacked prior with team/player priors).
  1  full buy vs eco, favourite wins          2  full buy vs eco, the eco side (stronger team) wins
  3  pistol round, rank 1 vs rank 35          4  both full buy, even money, rank 4 vs rank 35
Panels 5-6: realised win rate of the HIGHER-RANKED team at equal buys (pistol rounds; both-full-buy
even-money rounds) by HLTV rank gap, against what the t=0 models say.

Inputs: outputs/t0_round_table.parquet, outputs/t0_pilot_predictions.parquet (pilot_t0_round_start.py,
supplement source), outputs/oof_noecon_logreg.parquet (per-second OOF), the supplement's parsed
kills/bomb/rounds channels. Output: outputs/figures/paper/F13_t0_examples.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs"
SUPP = ROOT / "data" / "cs2_inferno_raw_supplement_v1" / "training_2024_2025" / "parsed"
FIG = OUT / "figures" / "paper" / "F13_t0_examples.png"

CT, T = "#2a78d6", "#eb6834"                  # validated palette slots 1-2
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
AQUA, YELLOW = "#1baf7a", "#eda100"

EXAMPLES = [
    ("mouz-vs-faze-m1-inferno", 20, "1. Full buy vs eco: favourite wins"),
    ("cloud9-vs-vitality-m1-inferno", 18, "2. Full buy vs eco: the eco side (Vitality) wins"),
    ("ecstatic-vs-g2-m1-inferno", 1, "3. Pistol round, equal money: rank 35 (CT) vs rank 1 (T)"),
    ("faze-vs-ninjas-in-pyjamas-m2-inferno", 6, "4. Both full buy, equal money: rank 35 (CT) vs rank 4 (T)"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    rt = pl.read_parquet(OUT / "t0_round_table.parquet")
    pr = pl.read_parquet(OUT / "t0_pilot_predictions.parquet").drop("ct_won")
    rt = rt.join(pr, on=["match_id", "round_num"])
    oof = pl.read_parquet(OUT / "oof_noecon_logreg.parquet", columns=["match_id", "round_num", "tick", "p_A", "p_EB2"])
    tsec = pl.read_parquet(ROOT / "data" / "training_dataset.parquet", columns=["match_id", "round_num", "tick", "time_elapsed_sec"])
    oof = oof.join(tsec, on=["match_id", "round_num", "tick"]).sort(["match_id", "round_num", "tick"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                         "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
    axes = axes.ravel()

    # ---------------------------------------------------------------- example curves
    for ax, (mid, rn, title) in zip(axes[:4], EXAMPLES):
        row = rt.filter((pl.col("match_id") == mid) & (pl.col("round_num") == rn)).to_dicts()[0]
        path = oof.filter((pl.col("match_id") == mid) & (pl.col("round_num") == rn))
        x = path["time_elapsed_sec"].to_numpy(); y = path["p_EB2"].to_numpy(); ya = path["p_A"].to_numpy()
        fe = int(row["freeze_end"])
        kills = (pl.read_parquet(SUPP / "kills" / f"{mid}.parquet", columns=["round_num", "tick", "victim_side", "is_freeze_period"])
                   .filter((pl.col("round_num") == rn) & (pl.col("is_freeze_period") == False)).sort("tick"))
        plant = (pl.read_parquet(SUPP / "bomb" / f"{mid}.parquet", columns=["round_num", "tick", "event"])
                   .filter((pl.col("round_num") == rn) & (pl.col("event") == "plant")))
        ax.axhline(0.5, color=GRID, lw=1, zorder=0)
        ax.plot(x, ya, color=INK2, lw=1.2, ls=(0, (1, 2)), zorder=2, label="economy only (A)")
        ax.plot(x, y, color=INK, lw=2, zorder=3, label="production model (EB2)")
        for kt, vs in zip(kills["tick"].to_numpy(), kills["victim_side"].to_list()):
            ts = (kt - fe) / 64.0
            yy = np.interp(ts, x, y)
            ax.plot(ts, yy, "o", ms=7, mfc=CT if vs == "ct" else T, mec=SURF, mew=1.2, zorder=4)
        if plant.height:
            tp = (int(plant["tick"][0]) - fe) / 64.0
            ax.axvline(tp, color=INK2, lw=1, ls=(0, (3, 3)), zorder=1)
            ax.text(tp + 1, 0.95, "bomb planted", color=INK2, fontsize=8, va="top")
        # t=0 priors
        ax.plot(0, row["p0_snapEB2"], "s", ms=8, mfc=SURF, mec=INK, mew=1.5, zorder=5)
        ax.plot(0, row["p0_stack_sameyr"], "D", ms=8, mfc=SURF, mec=AQUA, mew=1.8, zorder=5)
        won = "CT" if row["ct_won"] == 1 else "T"
        # title above the details block; dollar signs escaped so matplotlib does not parse mathtext
        sub = (f"CT {row['ct_team_clan']} (rank {int(row['ct_rank_same'])}, rating {row['ct_rating_same_mean']:.2f}, "
               f"\${int(row['ct_equipment_value']):,})   vs   T {row['t_team_clan']} (rank {int(row['t_rank_same'])}, "
               f"rating {row['t_rating_same_mean']:.2f}, \${int(row['t_equipment_value']):,})\n"
               f"{mid}, round {rn}: {won} won ({row['reason']}).  t=0: model {row['p0_snapEB2']:.2f}, "
               f"stacked prior {row['p0_stack_sameyr']:.2f}")
        ax.text(0, 1.10, title, transform=ax.transAxes, fontsize=10, color=INK, va="bottom")
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7.3, color=INK2, va="bottom")
        ax.set_ylim(0, 1); ax.set_xlim(-2, max(x.max() + 3, 40))
        ax.set_ylabel("P(CT wins the round)"); ax.set_xlabel("seconds since freeze end")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], color=INK, lw=2, label="production model (EB2), per second"),
         plt.Line2D([], [], color=INK2, lw=1.2, ls=(0, (1, 2)), label="economy only (A)"),
         plt.Line2D([], [], marker="o", ms=7, mfc=CT, mec=SURF, lw=0, label="a CT died"),
         plt.Line2D([], [], marker="o", ms=7, mfc=T, mec=SURF, lw=0, label="a T died"),
         plt.Line2D([], [], marker="s", ms=8, mfc=SURF, mec=INK, mew=1.5, lw=0, label="t=0: model's own first value"),
         plt.Line2D([], [], marker="D", ms=8, mfc=SURF, mec=AQUA, mew=1.8, lw=0, label="t=0: stacked prior + team/player priors")]
    axes[0].legend(handles=h, loc="lower left", frameon=False, fontsize=7.5, ncol=2)

    # ---------------------------------------------------------------- aggregate panels
    rt = rt.with_columns(rank_gap=(pl.col("t_rank_same") - pl.col("ct_rank_same")),
                         equip_diff=(pl.col("ct_equipment_value") - pl.col("t_equipment_value")))
    conds = [("5. Pistol rounds (both sides \$800 to start)", pl.col("is_pistol_round") == 1),
             ("6. Both full buy, |money difference| <= \$1,500", (pl.col("ct_equipment_value") >= 20000)
              & (pl.col("t_equipment_value") >= 20000) & (pl.col("equip_diff").abs() <= 1500))]
    buckets = [("0", 0, 0), ("1-5", 1, 5), ("6-15", 6, 15), ("16+", 16, 99)]
    for ax, (title, cond) in zip(axes[4:], conds):
        s = rt.filter(cond).with_columns(fav_ct=(pl.col("rank_gap") > 0), gap=pl.col("rank_gap").abs())
        s = s.with_columns(
            fav_won=pl.when(pl.col("fav_ct")).then(pl.col("ct_won")).otherwise(1 - pl.col("ct_won")),
            m_snap=pl.when(pl.col("fav_ct")).then(pl.col("p0_snapEB2")).otherwise(1 - pl.col("p0_snapEB2")),
            m_same=pl.when(pl.col("fav_ct")).then(pl.col("p0_stack_sameyr")).otherwise(1 - pl.col("p0_stack_sameyr")),
            m_lag=pl.when(pl.col("fav_ct")).then(pl.col("p0_stack_lag")).otherwise(1 - pl.col("p0_stack_lag")))
        xs = np.arange(len(buckets))
        ax.axhline(0.5, color=INK2, lw=1, ls=(0, (3, 3)), zorder=0)
        for i, (lab, lo, hi) in enumerate(buckets):
            b = s.filter((pl.col("gap") >= lo) & (pl.col("gap") <= hi))
            n = b.height; k = int(b["fav_won"].sum()) if n else 0
            rate = k / n if n else np.nan; lo_ci, hi_ci = wilson(k, n)
            ax.bar(i, rate, width=0.55, color=CT, alpha=0.25, edgecolor=CT, zorder=2)
            ax.plot([i, i], [lo_ci, hi_ci], color=CT, lw=1.5, zorder=3)
            ax.text(i, 0.31, f"n={n}", ha="center", fontsize=8, color=INK2)
            for col, mk, c in [("m_snap", "s", INK), ("m_same", "D", AQUA), ("m_lag", "^", YELLOW)]:
                ax.plot(i, float(b[col].mean()) if n else np.nan, mk, ms=8, mfc=SURF, mec=c, mew=1.8, zorder=4)
        ax.set_xticks(xs); ax.set_xticklabels([f"gap {b[0]}" for b in buckets])
        ax.set_ylim(0.3, 0.75); ax.set_ylabel("P(higher-ranked team wins the round)")
        ax.set_xlabel("HLTV rank gap between the two teams (same-year snapshot)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    h2 = [plt.Rectangle((0, 0), 1, 1, color=CT, alpha=0.25, ec=CT, label="realised win rate of the higher-ranked team (95% CI)"),
          plt.Line2D([], [], marker="s", ms=8, mfc=SURF, mec=INK, mew=1.5, lw=0, label="production model, first snapshot"),
          plt.Line2D([], [], marker="D", ms=8, mfc=SURF, mec=AQUA, mew=1.8, lw=0, label="stacked prior + same-year priors (leaky)"),
          plt.Line2D([], [], marker="^", ms=8, mfc=SURF, mec=YELLOW, mew=1.8, lw=0, label="stacked prior + lagged priors (2025 matches only)")]
    axes[4].legend(handles=h2, loc="upper left", frameon=False, fontsize=7.5)

    fig.suptitle("Round-start priors and the live curve: four example rounds, and the higher-ranked team's win rate at equal buys (out-of-fold, 2024-25)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=4.0)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=170)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
