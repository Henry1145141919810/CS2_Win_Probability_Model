"""Find & visualize CLASSIC rounds where MAP CONTROL shifts the win probability.

We fit two out-of-fold logistic models on the real data:
  - A = economy/combat only (the baseline)
  - E = economy + map control + tactical
For every snapshot, delta = P_E(CT win) - P_A(CT win) is the part of the probability that
COMES FROM map control. A round is a good teaching example when, at some moment, control
moves the probability substantially AND toward the side that actually won, while economy
alone was near a coin flip (so the shift is genuinely spatial, not eco recomputed).

We rank rounds by that criterion, pick 3 DISTINCT matches (mixing CT- and T-favoring
shifts), and for each save a win-probability timeline (P_A vs P_E over the round, peak
moment marked, control features annotated). It then prints the exact mapcontrol_viz.py
command for the peak second so the spatial picture can be rendered alongside.

Usage: python src/viz/control_shift_examples.py [--n 3]
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
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from features.economy import ECONOMY_COLS  # noqa: E402
from features.mapcontrol import MAPCONTROL_COLS, TERRITORY_COLS  # noqa: E402
from features.positional import TACTICAL_COLS  # noqa: E402
from features.bomb import BOMB_COLS  # noqa: E402

DATA = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "figures"
# Isolate the MAP-CONTROL contribution: A = economy only, E = economy + map control ONLY
# (Voronoi + territory). Deliberately EXCLUDE tactical/bomb so that delta = P_E - P_A is
# attributable to spatial control, not to the bomb timer or utility (which would muddy the
# "map control shifted the win prob" story, e.g. post-plant 1v0 rounds).
A_COLS = ECONOMY_COLS
E_COLS = ECONOMY_COLS + MAPCONTROL_COLS + TERRITORY_COLS
ANNOT = ["ct_voronoi_control_pct", "control_deficit", "ct_terr_deficit",
         "ct_banana_control", "ct_mid_control"]


def oof(df, cols, y, groups):
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(max_iter=2000))
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    df = pl.read_parquet(DATA)
    y = df["ct_won"].to_numpy()
    groups = df["match_id"].to_numpy()
    print(f"data: {len(y)} snapshots; fitting OOF logreg A and E ...")
    pA = oof(df, A_COLS, y, groups)
    pE = oof(df, E_COLS, y, groups)

    truth_dir = 2 * y - 1                     # +1 if CT won, -1 if T won
    delta = pE - pA                           # win-prob attributable to control
    helpful = delta * truth_dir               # >0: control moved prob toward the winner

    d = df.select(["match_id", "round_num", "tick", "time_elapsed_sec", "ct_won",
                   "ct_players_alive", "t_players_alive",
                   "ct_equipment_value", "t_equipment_value", *ANNOT]).with_columns(
        pA=pl.Series(pA), pE=pl.Series(pE), delta=pl.Series(delta),
        helpful=pl.Series(helpful))
    # genuinely contested snapshot: both teams alive and within 1 player of each other,
    # so a probability shift reflects POSITIONING, not a man-advantage endgame.
    d = d.with_columns(
        contested=((pl.col("ct_players_alive") >= 1) & (pl.col("t_players_alive") >= 1)
                   & ((pl.col("ct_players_alive") - pl.col("t_players_alive")).abs() <= 1)),
    )
    d = d.with_columns(helpful_c=pl.when(pl.col("contested")).then(pl.col("helpful"))
                       .otherwise(-1e9))

    # score each round: sustained helpful shift where economy was uncertain (|pA-.5| small)
    # and control confident+correct (helpful large), measured on CONTESTED snapshots. >=8 snaps.
    rnd = (d.group_by(["match_id", "round_num"]).agg(
        n=pl.len(),
        n_contested=pl.col("contested").sum(),
        ct_won=pl.col("ct_won").first(),
        peak_helpful=pl.col("helpful_c").max(),
        mean_helpful=pl.col("helpful").mean(),
        eco_uncertainty=(0.5 - (pl.col("pA") - 0.5).abs()).mean(),  # high = pA near .5
    ).filter((pl.col("n") >= 8) & (pl.col("n_contested") >= 5)))
    rnd = rnd.with_columns(
        score=pl.col("peak_helpful") * 0.6 + pl.col("mean_helpful") * 0.4
        + pl.col("eco_uncertainty") * 0.3).sort("score", descending=True)

    # pick top distinct matches, alternate CT-won / T-won for variety
    picks, used_matches, want_ct = [], set(), True
    for _ in range(2):  # two passes to allow fallback
        for row in rnd.iter_rows(named=True):
            if len(picks) >= args.n:
                break
            if row["match_id"] in used_matches:
                continue
            if want_ct is not None and bool(row["ct_won"]) != want_ct:
                continue
            picks.append(row); used_matches.add(row["match_id"]); want_ct = not want_ct
        want_ct = None  # second pass: take any
        if len(picks) >= args.n:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\nselected {len(picks)} example rounds:\n")
    for i, p in enumerate(picks, 1):
        mid, rn = p["match_id"], p["round_num"]
        sub = d.filter((pl.col("match_id") == mid) & (pl.col("round_num") == rn)).sort("tick")
        ts = sub["time_elapsed_sec"].to_numpy()
        won = "CT" if p["ct_won"] else "T"
        j = int(np.argmax(sub["helpful_c"].to_numpy()))  # peak among contested snapshots
        peak_sec = float(ts[j])
        peak = sub.row(j, named=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(ts, sub["pA"], color="#888", lw=2, label="P(CT win) — economy only (A)")
        ax.plot(ts, sub["pE"], color="#cc3311", lw=2.4, label="P(CT win) — + map control (E)")
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.axhline(1.0 if p["ct_won"] else 0.0, color="#2a8", ls="--", lw=1.2,
                   label=f"actual outcome ({won} won)")
        ax.axvline(peak_sec, color="#333", ls="-", lw=0.8, alpha=0.6)
        ax.scatter([peak_sec], [peak["pE"]], s=60, color="#cc3311", zorder=5)
        gap = peak["pE"] - peak["pA"]
        ax.annotate(f"control shifts P by {gap:+.0%}\n"
                    f"Voronoi CT {peak['ct_voronoi_control_pct']:.0%}, "
                    f"terr deficit {peak['ct_terr_deficit']:+.2f}",
                    xy=(peak_sec, peak["pE"]), xytext=(peak_sec + 1, peak["pE"] + (0.12 if gap > 0 else -0.18)),
                    fontsize=9, arrowprops=dict(arrowstyle="->", color="#333"))
        ax.set_xlabel("time into round (s)"); ax.set_ylabel("P(CT win)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"Example {i}: {mid}  round {rn}  ({won} won)\n"
                     f"economy alone ~coin-flip; map control correctly favors {won}")
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        out = OUT / f"control_shift_example_{i}_{mid}_r{rn}.png"
        fig.savefig(out, dpi=120); plt.close(fig)

        print(f"[{i}] {mid} round {rn} | winner {won} | peak @ t+{peak_sec:.0f}s "
              f"| pA={peak['pA']:.2f} -> pE={peak['pE']:.2f} (shift {gap:+.2f})")
        print(f"     alive CT{peak['ct_players_alive']}-T{peak['t_players_alive']}  "
              f"equip CT{peak['ct_equipment_value']:.0f}-T{peak['t_equipment_value']:.0f}  "
              f"Voronoi {peak['ct_voronoi_control_pct']:.0%}")
        print(f"     timeline saved {out.name}")
        secs = sorted({max(1, int(peak_sec) - 8), int(peak_sec), int(peak_sec) + 8})
        print(f"     MAP: python src/viz/mapcontrol_viz.py --match {mid} --round {rn} "
              f"--secs {','.join(map(str, secs))}\n")


if __name__ == "__main__":
    main()
