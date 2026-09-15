"""PILOT for Study 2 - match-level win probability from the score state (in-time only).

Establishes the floor every later rung must beat, and checks the format facts the DP relies on:
  * side schedule by round index (halftime swap at 13, overtime side pattern) - read from the data
  * end-of-match rule (first to 13; overtime periods of 6, first to 4 in a period)
  * A0: score-state DP with a constant side-specific per-round probability
  * A1-lite: same DP but the CURRENT round's probability comes from a round-level logistic on
    (format + buy state), fitted out-of-fold; future rounds use the constant
Evaluated at every round start (4,866 rows) against the match winner, with match-level bootstrap.

Usage: python src/models/pilot_match_state.py
"""
from __future__ import annotations
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "exports" / "cs2_inferno_round_level_v1"
OUT = ROOT / "outputs"
B = 500


def side_schedule(r: pl.DataFrame):
    """Empirical: is the team that was CT in round 1 also CT in round k?  Returns dict k -> frac."""
    first_ct = r.filter(pl.col("round_num") == 1).select(["match_id", pl.col("ct_team_clan").alias("x")])
    j = r.join(first_ct, on="match_id").with_columns(x_is_ct=(pl.col("ct_team_clan") == pl.col("x")).cast(pl.Int8))
    return j.group_by("round_num").agg(frac=pl.col("x_is_ct").mean(), n=pl.len()).sort("round_num")


def x_is_ct(k: int) -> bool:
    """Side of team X (CT in round 1) at round k. Regulation: CT for 1..12, T for 13..24.
    Overtime (MR3): periods of 6; teams swap at the start of OT relative to the end of regulation and
    again at each OT half. Verified against the data by side_schedule()."""
    if k <= 12:
        return True
    if k <= 24:
        return False
    j = (k - 25) % 6          # position within the OT period
    period = (k - 25) // 6    # 0-based OT period
    # Verified on the data (side_schedule): X ended regulation on T and STAYS on T for rounds 25-27,
    # swaps at the OT half (28-30 CT), and each new period starts on the side the previous one ended
    # on: 31-33 CT, 34-36 T, 37-39 T, 40-42 CT ...
    first_half_ct = (period % 2 == 1)
    return first_half_ct if j < 3 else (not first_half_ct)


def terminal(a: int, b: int, k: int):
    """Match over before round k is played? Returns +1 (X won), -1 (Y won) or 0."""
    if k <= 25:
        if a >= 13: return 1
        if b >= 13: return -1
        return 0
    # in overtime: at start of period n (n>=1) both have 12 + 3(n-1); winner reaches 13 + 3(n-1)... i.e. 4 in period
    period = (k - 25) // 6 + 1
    base = 12 + 3 * (period - 1)
    if a - base >= 4: return 1
    if b - base >= 4: return -1
    # a period can also end early: handled because k advances only when a round is played
    return 0


def match_wp(a: int, b: int, k: int, p_ct: float, p_now: float | None = None, max_k: int = 60) -> float:
    """P(team X wins the map | X has a, Y has b, next round is k). p_ct = P(CT side wins a round).
    p_now overrides the probability for the round about to be played (current round only)."""
    @lru_cache(maxsize=None)
    def V(a, b, k):
        t = terminal(a, b, k)
        if t: return 1.0 if t > 0 else 0.0
        if k > max_k: return 0.5
        # overtime period boundary: if a period ended 3-3, we continue (handled by terminal + k)
        p = p_ct if x_is_ct(k) else 1 - p_ct
        return p * V(a + 1, b, k + 1) + (1 - p) * V(a, b + 1, k + 1)
    t = terminal(a, b, k)
    if t: return 1.0 if t > 0 else 0.0
    p = (p_now if p_now is not None else (p_ct if x_is_ct(k) else 1 - p_ct))
    return p * V(a + 1, b, k + 1) + (1 - p) * V(a, b + 1, k + 1)


def main():
    r = pl.read_csv(EXP / "rounds.csv", infer_schema_length=10000).sort(["match_id", "round_num"])
    m = pl.read_csv(EXP / "matches.csv", infer_schema_length=10000)
    print(f"rounds {r.height}, matches {r['match_id'].n_unique()}")

    # --- format checks -------------------------------------------------------------------------
    sched = side_schedule(r)
    print("\nside schedule (fraction of matches where the round-1 CT team is CT at round k):")
    print(sched.filter(pl.col("round_num").is_in([1, 12, 13, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37])))
    # does the score at each round start reproduce first-to-13 + OT rule?
    ot = r.filter(pl.col("round_num") == 25).select(["match_id", "ct_score", "t_score"])
    print(f"\novertime matches: {ot.height}; all 12-12 at round 25: {((ot['ct_score']==12)&(ot['t_score']==12)).all()}")
    # final score check vs matches.csv
    last = r.group_by("match_id").agg(pl.col("round_num").max().alias("n"), pl.col("ct_score").last(), pl.col("t_score").last(),
                                      pl.col("round_winner_team_clan").last(), pl.col("ct_won").last())
    print("rounds per match:", last["n"].min(), "..", last["n"].max())

    # --- team X = the team on CT in the first half (mode; 10 matches lack a round 1) -------------
    # NOTE: the export's ct_score/t_score columns are NOT usable after halftime (they are cumulative
    # SIDE wins, see pilot findings), so team scores are rebuilt from the round winners.
    first_ct = (r.filter(pl.col("round_num") <= 12).group_by("match_id")
                 .agg(pl.col("ct_team_clan").mode().first().alias("x_clan")))
    r = r.join(first_ct, on="match_id")
    r = r.with_columns(x_is_ct_now=(pl.col("ct_team_clan") == pl.col("x_clan")),
                       xw=(pl.col("round_winner_team_clan") == pl.col("x_clan")).cast(pl.Int64))
    r = r.with_columns(a=pl.col("xw").cum_sum().over("match_id") - pl.col("xw"),
                       b=(1 - pl.col("xw")).cum_sum().over("match_id") - (1 - pl.col("xw")))
    # corrected side-perspective scores for the round-level logistic
    r = r.with_columns(ct_score_fix=pl.when(pl.col("x_is_ct_now")).then(pl.col("a")).otherwise(pl.col("b")),
                       t_score_fix=pl.when(pl.col("x_is_ct_now")).then(pl.col("b")).otherwise(pl.col("a")))
    r = r.with_columns(score_diff_fix=pl.col("ct_score_fix") - pl.col("t_score_fix"))
    # label: X won the map. 11 matches lose their clinching round to validation trimming, so the
    # curated match winner (matches.csv, from the demo list) is authoritative; cumulative wins are
    # the fallback when the clan name cannot be matched to a curated team name.
    fin = r.group_by("match_id").agg(x_final=pl.col("xw").sum(), y_final=(1 - pl.col("xw")).sum())
    r = r.join(fin, on="match_id").join(m.select(["match_id", "team_a", "team_b", "match_winner"]), on="match_id")

    def _norm(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())

    def _x_won(s):
        cx, ca, cb, cw = _norm(s["x_clan"]), _norm(s["team_a"]), _norm(s["team_b"]), _norm(s["match_winner"])
        hit_a = cx == ca or cx in ca or ca in cx
        hit_b = cx == cb or cx in cb or cb in cx
        if hit_a != hit_b:
            picked = ca if hit_a else cb
            return int(picked == cw or picked in cw or cw in picked)
        return int(s["x_final"] > s["y_final"])
    r = r.with_columns(pl.struct(["x_clan", "team_a", "team_b", "match_winner", "x_final", "y_final"])
                         .map_elements(_x_won, return_dtype=pl.Int64).cast(pl.Int8).alias("x_won"))
    agree = (r["x_won"] == (r["x_final"] > r["y_final"]).cast(pl.Int8)).mean()
    print(f"label check: curated winner agrees with cumulative-wins winner on {agree:.1%} of round rows")
    # sanity: schedule function vs data
    r = r.with_columns(pred_side=pl.col("round_num").map_elements(lambda k: x_is_ct(int(k)), return_dtype=pl.Boolean))
    mism = (r["pred_side"] != r["x_is_ct_now"]).sum()
    print(f"side-schedule mismatches (function vs data): {mism} / {r.height}")
    y = r["x_won"].to_numpy().astype(int); g = r["match_id"].to_numpy()
    print(f"X (round-1 CT team) wins the map: {y.mean():.3f} (match level: {r.group_by('match_id').agg(pl.col('x_won').first())['x_won'].mean():.3f})")

    # --- A0: constant p_ct (out-of-fold estimate of the CT round-win rate) --------------------
    a = r["a"].to_numpy(); b = r["b"].to_numpy(); k = r["round_num"].to_numpy()
    p_ct_all = float((r["ct_won"]).mean())
    print(f"\nCT round-win rate (train): {p_ct_all:.4f}")
    A0 = np.array([match_wp(int(a[i]), int(b[i]), int(k[i]), p_ct_all) for i in range(len(y))])
    A0_half = np.array([match_wp(int(a[i]), int(b[i]), int(k[i]), 0.5) for i in range(len(y))])  # side-blind

    # --- A1-lite: current round p from a round-level logistic (format + buy), OOF -------------
    cols = ["round_num", "half", "is_pistol_round", "ct_score_fix", "t_score_fix", "score_diff_fix",
            "ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
            "ct_armor_total", "t_armor_total", "ct_defuse_kits", "ct_util_total", "t_util_total",
            "utility_advantage", "ct_awp_alive", "t_awp_alive"]
    X = np.nan_to_num(r[cols].to_numpy().astype(float)); yr = r["ct_won"].to_numpy().astype(int)
    p_round = np.zeros(len(yr))
    for tr, te in GroupKFold(5).split(X, yr, g):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(X[tr], yr[tr])
        p_round[te] = mdl.predict_proba(X[te])[:, 1]
    print(f"round-level t=0 logistic (format+buy): AUC {roc_auc_score(yr, p_round):.4f} ll {log_loss(yr, p_round):.4f}")
    xnow = r["x_is_ct_now"].to_numpy()
    p_now_x = np.where(xnow, p_round, 1 - p_round)
    A1 = np.array([match_wp(int(a[i]), int(b[i]), int(k[i]), p_ct_all, p_now=float(p_now_x[i])) for i in range(len(y))])

    # --- evaluate at round-start level, match bootstrap ---------------------------------------
    preds = {"A0 side-blind p=0.5": A0_half, "A0 constant side rate": A0, "A1-lite current-round t=0 econ": A1}
    rng = np.random.default_rng(0); uniq = np.unique(g); idx_by = {u: np.where(g == u)[0] for u in uniq}
    boots = {n: {"auc": [], "ll": [], "brier": []} for n in preds}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True); idx = np.concatenate([idx_by[u] for u in pick])
        for n, p in preds.items():
            boots[n]["auc"].append(roc_auc_score(y[idx], p[idx])); boots[n]["ll"].append(log_loss(y[idx], p[idx], labels=[0, 1]))
            boots[n]["brier"].append(brier_score_loss(y[idx], p[idx]))
    rows = []
    print(f"\n{'model':34s} {'AUC':>7} {'logloss':>8} {'brier':>7}  (95% CI on log-loss)")
    for n, p in preds.items():
        ll = np.array(boots[n]["ll"])
        rows.append(dict(model=n, AUC=roc_auc_score(y, p), logloss=log_loss(y, p), brier=brier_score_loss(y, p),
                         ll_lo=np.percentile(ll, 2.5), ll_hi=np.percentile(ll, 97.5)))
        print(f"{n:34s} {rows[-1]['AUC']:7.4f} {rows[-1]['logloss']:8.4f} {rows[-1]['brier']:7.4f}  ({rows[-1]['ll_lo']:.4f}, {rows[-1]['ll_hi']:.4f})")
    d = np.array(boots["A1-lite current-round t=0 econ"]["ll"]) - np.array(boots["A0 constant side rate"]["ll"])
    print(f"paired dLL A1-lite vs A0: {d.mean():+.4f} ({np.percentile(d,2.5):+.4f}, {np.percentile(d,97.5):+.4f})")
    # by phase
    rr = r.with_columns(A0=pl.Series(A0), A1=pl.Series(A1), y=pl.Series(y))
    for lo, hi, name in [(1, 1, "round 1"), (2, 12, "half 1"), (13, 13, "round 13"), (14, 24, "half 2"), (25, 99, "overtime")]:
        s = rr.filter((pl.col("round_num") >= lo) & (pl.col("round_num") <= hi))
        yy = s["y"].to_numpy()
        print(f"  {name:9s} n={s.height:4d}  A0 ll {log_loss(yy, s['A0'].to_numpy(), labels=[0,1]):.4f}  A1 ll {log_loss(yy, s['A1'].to_numpy(), labels=[0,1]):.4f}"
              + (f"  A0 AUC {roc_auc_score(yy, s['A0'].to_numpy()):.3f}" if len(np.unique(yy)) == 2 and s['A0'].std() > 0 else ""))
    # calibration by score-diff bucket (A0)
    rr = rr.with_columns(sd=(pl.col("a") - pl.col("b")).clip(-6, 6))
    cal = rr.group_by("sd").agg(n=pl.len(), A0=pl.col("A0").mean(), A1=pl.col("A1").mean(), y=pl.col("y").mean()).sort("sd")
    print("\ncalibration by score difference (X - Y), all round starts:"); print(cal)
    # --- mean-reversion test: does the trailing team win the NEXT round more often than the base rate?
    # Under independent rounds P(X wins round | a-b) is flat in (a-b). CS2's loss bonus predicts a
    # downward slope (trailing team buys better next round).
    rr = rr.with_columns(x_won_round=pl.col("xw"), x_econ=pl.when(pl.col("x_is_ct_now")).then(pl.col("ct_economy_class")).otherwise(pl.col("t_economy_class")),
                         y_econ=pl.when(pl.col("x_is_ct_now")).then(pl.col("t_economy_class")).otherwise(pl.col("ct_economy_class")))
    mr = (rr.filter(~pl.col("round_num").is_in([1, 13]))     # pistol rounds have no economy carry-over
            .group_by("sd").agg(n=pl.len(), p_x_wins_round=pl.col("x_won_round").mean(),
                                x_econ=pl.col("x_econ").mean(), y_econ=pl.col("y_econ").mean()).sort("sd"))
    print("\nmean-reversion test (non-pistol rounds): P(X wins the NEXT round | X-Y score diff), and buy tiers:")
    print(mr)
    mr.write_csv(OUT / "match_pilot_meanreversion.csv")
    # same, conditioning also on the last round's outcome (momentum vs reversion at the round scale)
    rr = rr.with_columns(x_won_prev=pl.col("x_won_round").shift(1).over("match_id"))
    mm = (rr.filter(~pl.col("round_num").is_in([1, 13]) & pl.col("x_won_prev").is_not_null())
            .group_by("x_won_prev").agg(n=pl.len(), p_x_wins_round=pl.col("x_won_round").mean()).sort("x_won_prev"))
    print("P(X wins round | X won previous round):"); print(mm)
    # is the momentum just economy? condition on both teams' buy tier
    mm2 = (rr.filter(~pl.col("round_num").is_in([1, 13]) & pl.col("x_won_prev").is_not_null())
             .with_columns(both_full=((pl.col("x_econ") == 2) & (pl.col("y_econ") == 2)))
             .group_by(["both_full", "x_won_prev"]).agg(n=pl.len(), p_x_wins_round=pl.col("x_won_round").mean())
             .sort(["both_full", "x_won_prev"]))
    print("P(X wins round | X won previous round, both teams full-buy?):"); print(mm2)
    mm2.write_csv(OUT / "match_pilot_momentum_econ.csv")
    pl.DataFrame(rows).write_csv(OUT / "match_pilot_dp.csv"); cal.write_csv(OUT / "match_pilot_calibration.csv")
    print("wrote outputs/match_pilot_dp.csv, outputs/match_pilot_calibration.csv")


if __name__ == "__main__":
    main()
