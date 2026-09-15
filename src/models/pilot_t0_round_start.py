"""PILOT for Study 1 - the round-start (t=0) win probability, on the round-level export.

In-time only (5-fold GroupKFold by match on 4,866 rounds). The 2026 holdout is NOT touched here;
the pilot exists to make the plan concrete (which rungs carry signal, how the t=0 prior relates to
the per-second model, and whether the martingale identities hold), not to produce paper numbers.

Parts
  A. Nested feature ladder at t=0 (logistic, round level) with paired match-bootstrap CIs.
     R1 format/side/score -> R2 +buy state -> R3 +recent history -> R4 +team prior -> R5 +player prior
     Priors come in two constructions: same-year (leaky, upper bound) and lagged (previous season;
     unavailable for 2024 matches because no 2023 tables exist - reported as coverage).
  B. Comparison against the per-second EB2 model's FIRST-SNAPSHOT value used as p0 today
     (pathwise_calibration.py). Is a dedicated round-level prior better than the first snapshot?
  C. Martingale identities for the per-second curve anchored at p0:
       (i)  E[first-snapshot p | p0 bin] vs p0                 (no jump at t=0+)
       (ii) E[(Y-p0)^2] vs E[p0(1-p0)]                          (terminal variance = implied variance)
       (iii) E[sum_t (p_t - p_{t-1})^2] vs E[(Y-p0)^2]           (realised quadratic variation)
  D. Opening-kill increment: WP before/after the first kill of the round, by victim side and p0 bin;
     calibration of the post-kill value against the realised outcome.

Usage: python src/models/pilot_t0_round_start.py
"""
from __future__ import annotations
import glob
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.train_pipeline import FEATURE_SETS, oof_predict, ece  # noqa: E402

EXP = ROOT / "exports" / "cs2_inferno_round_level_v1"
OUT = ROOT / "outputs"
B = 500


# ----------------------------------------------------------------------------------------------
# Build the round-level t=0 table
# ----------------------------------------------------------------------------------------------
SUPP_REF = ROOT / "data" / "cs2_inferno_raw_supplement_v1" / "reference"


def build_table(source: str = "supplement") -> pl.DataFrame:
    if source == "supplement":
        # straight from the raw supplement (per-second first snapshot + rounds + ticks channels)
        from models.t0_table import build as _build_raw
        r, rp, mt = _build_raw()
        rp = rp.rename({"current_equip_value": "equip_value"})
        rk = pl.read_csv(SUPP_REF / "team_rankings.csv")
        ps = pl.read_csv(SUPP_REF / "player_stats_sided.csv", infer_schema_length=10000)
    else:
        r = pl.read_csv(EXP / "rounds.csv", infer_schema_length=10000)
        rp = pl.read_csv(EXP / "round_players.csv", infer_schema_length=10000)
        mt = pl.read_csv(EXP / "matches.csv", infer_schema_length=10000)
        rk = pl.read_csv(EXP / "team_rankings.csv")
        ps = pl.read_csv(EXP / "player_season_stats.csv", infer_schema_length=10000)

    # --- R1 format ---
    # The export's ct_score/t_score are cumulative SIDE wins (never re-keyed at the halftime swap),
    # so rebuild true team scores from the round winners and express them for the current sides.
    r = r.sort(["match_id", "round_num"])
    x = (r.filter(pl.col("round_num") <= 12).group_by("match_id")
          .agg(pl.col("ct_team_clan").mode().first().alias("x_clan")))
    r = r.join(x, on="match_id").with_columns(
        xw=(pl.col("round_winner_team_clan") == pl.col("x_clan")).cast(pl.Int64),
        x_is_ct=(pl.col("ct_team_clan") == pl.col("x_clan")))
    r = r.with_columns(a=pl.col("xw").cum_sum().over("match_id") - pl.col("xw"),
                       b=(1 - pl.col("xw")).cum_sum().over("match_id") - (1 - pl.col("xw")))
    r = r.with_columns(ct_score=pl.when(pl.col("x_is_ct")).then(pl.col("a")).otherwise(pl.col("b")),
                       t_score=pl.when(pl.col("x_is_ct")).then(pl.col("b")).otherwise(pl.col("a")))
    r = r.with_columns(score_diff=pl.col("ct_score") - pl.col("t_score")).drop(["a", "b", "xw", "x_is_ct", "x_clan"])
    r = r.with_columns(
        overtime=(pl.col("round_num") > 24).cast(pl.Int8),
        ct_to_win=(13 - pl.col("ct_score")).clip(0, 13),
        t_to_win=(13 - pl.col("t_score")).clip(0, 13),
        second_half=(pl.col("half") == 2).cast(pl.Int8),
    )

    # --- R2 buy state: per-player loadout spread from round_players ---
    agg = (rp.group_by(["match_id", "round_num", "side"]).agg(
        equip_min=pl.col("equip_value").min(), equip_std=pl.col("equip_value").std().fill_null(0.0),
        helmets=pl.col("has_helmet").cast(pl.Int8).sum(),
        n_full=(pl.col("equip_value") >= 3800).cast(pl.Int8).sum()))
    for side in ["ct", "t"]:
        a = agg.filter(pl.col("side") == side).drop("side").rename(
            {c: f"{side}_{c}" for c in ["equip_min", "equip_std", "helmets", "n_full"]})
        r = r.join(a, on=["match_id", "round_num"], how="left")

    # --- R3 recent history, from the TEAM's perspective (sides swap at halftime) ---
    # per (match, round): winner team clan, each team's equipment that round, loser-streak state
    r = r.sort(["match_id", "round_num"])
    hist_rows = []
    for mid, g in r.group_by("match_id", maintain_order=True):
        streak, last_won, last_equip, last_reason_bomb = {}, {}, {}, {}
        for row in g.iter_rows(named=True):
            ct, t = row["ct_team_clan"], row["t_team_clan"]
            out = {"match_id": mid[0] if isinstance(mid, tuple) else mid, "round_num": row["round_num"]}
            for side, team in [("ct", ct), ("t", t)]:
                out[f"{side}_loss_streak"] = streak.get(team, 0)
                out[f"{side}_won_prev"] = last_won.get(team, np.nan)
                out[f"{side}_prev_equip"] = last_equip.get(team, np.nan)
            # after the round resolves, update state for both teams
            w = row["round_winner_team_clan"]
            for side, team in [("ct", ct), ("t", t)]:
                won = int(w == team)
                streak[team] = 0 if won else streak.get(team, 0) + 1
                last_won[team] = won
                last_equip[team] = row[f"{side}_equipment_value"]
            hist_rows.append(out)
    h = pl.DataFrame(hist_rows)
    r = r.join(h, on=["match_id", "round_num"], how="left")
    # pistol-round resets: streak/won_prev are undefined at rounds 1 and 13 -> encode as 0 + flag
    r = r.with_columns([pl.col(c).fill_null(0.0).fill_nan(0.0) for c in
                        ["ct_won_prev", "t_won_prev", "ct_prev_equip", "t_prev_equip"]])
    r = r.with_columns((pl.col("ct_won_prev") - pl.col("t_won_prev")).alias("momentum_diff"),
                       (pl.col("ct_loss_streak") - pl.col("t_loss_streak")).alias("streak_diff"))

    # --- R4 team prior: HLTV rank, same-year (leaky) and lagged (previous season) ---
    rk = rk.with_columns(pl.col("hltv_rank").cast(pl.Float64))
    rk_same = rk.select(["team_canonical", "year", "hltv_rank"])
    rk_lag = rk.select(["team_canonical", (pl.col("year") + 1).alias("year"),
                        pl.col("hltv_rank").alias("hltv_rank_lag")])
    for side in ["ct", "t"]:
        r = (r.join(rk_same.rename({"team_canonical": f"{side}_team_canonical",
                                    "hltv_rank": f"{side}_rank_same"}),
                    on=[f"{side}_team_canonical", "year"], how="left")
              .join(rk_lag.rename({"team_canonical": f"{side}_team_canonical",
                                   "hltv_rank_lag": f"{side}_rank_lag"}),
                    on=[f"{side}_team_canonical", "year"], how="left"))
    UNRANKED = 35.0
    r = r.with_columns([
        pl.col("ct_rank_same").fill_null(UNRANKED), pl.col("t_rank_same").fill_null(UNRANKED),
        pl.col("ct_rank_lag").is_null().cast(pl.Int8).alias("ct_rank_lag_missing"),
        pl.col("t_rank_lag").is_null().cast(pl.Int8).alias("t_rank_lag_missing"),
    ]).with_columns([
        pl.col("ct_rank_lag").fill_null(UNRANKED), pl.col("t_rank_lag").fill_null(UNRANKED),
    ]).with_columns([
        (np.log2(pl.col("ct_rank_same") + 1) - np.log2(pl.col("t_rank_same") + 1)).alias("lrank_diff_same"),
        (np.log2(pl.col("ct_rank_lag") + 1) - np.log2(pl.col("t_rank_lag") + 1)).alias("lrank_diff_lag"),
    ])

    # Inferno form (leave-future-out): expanding map win rate of each team from EARLIER matches
    mt = mt.with_columns(pl.col("series_date").str.strptime(pl.Date, "%Y/%m/%d", strict=False).alias("date"))
    mt = mt.sort("date", nulls_last=True)
    form_rows = []
    wins, played = {}, {}
    for row in mt.iter_rows(named=True):
        for team in [row["team_a"], row["team_b"]]:
            n = played.get(team, 0); w = wins.get(team, 0)
            form_rows.append({"match_id": row["match_id"], "team": team,
                              "form": (w + 1.0) / (n + 2.0), "n_prev": n})  # Laplace-shrunk
        for team in [row["team_a"], row["team_b"]]:
            played[team] = played.get(team, 0) + 1
            wins[team] = wins.get(team, 0) + int(row["match_winner"] == team)
    form = pl.DataFrame(form_rows)
    # map ct/t team to team_a/team_b names via the crosswalk in rounds (clan vs curated name)
    # rounds carry canonical names; matches carry curated names. Build clan->curated via matches+rounds.
    m_names = mt.select(["match_id", "team_a", "team_b"])
    r = r.join(m_names, on="match_id", how="left")
    # canonical names in rounds vs curated in matches: match by normalised token overlap
    def _norm(s): return "".join(ch for ch in str(s).lower() if ch.isalnum())
    def pick(canon, a, b):
        ca, cb = _norm(a), _norm(b); cc = _norm(canon)
        if cc == ca or cc in ca or ca in cc: return a
        if cc == cb or cc in cb or cb in cc: return b
        return None
    r = r.with_columns([
        pl.struct(["ct_team_canonical", "team_a", "team_b"]).map_elements(
            lambda s: pick(s["ct_team_canonical"], s["team_a"], s["team_b"]), return_dtype=pl.Utf8).alias("ct_team_curated"),
        pl.struct(["t_team_canonical", "team_a", "team_b"]).map_elements(
            lambda s: pick(s["t_team_canonical"], s["team_a"], s["team_b"]), return_dtype=pl.Utf8).alias("t_team_curated"),
    ])
    for side in ["ct", "t"]:
        r = r.join(form.rename({"team": f"{side}_team_curated", "form": f"{side}_form", "n_prev": f"{side}_n_prev"}),
                   on=["match_id", f"{side}_team_curated"], how="left")
    r = r.with_columns([pl.col("ct_form").fill_null(0.5), pl.col("t_form").fill_null(0.5),
                        pl.col("ct_n_prev").fill_null(0), pl.col("t_n_prev").fill_null(0)])
    r = r.with_columns((pl.col("ct_form") - pl.col("t_form")).alias("form_diff"))

    # --- R5 player prior: side-specific mean rating of the five starters, same-year and lagged ---
    ps = ps.with_columns(pl.col("steamid").cast(pl.Int64), pl.col("year").cast(pl.Int64))
    rp = rp.with_columns(pl.col("steamid").cast(pl.Int64), pl.col("year").cast(pl.Int64))
    same = ps.select(["steamid", "year", "rating_ct", "rating_t"])
    lag = ps.select(["steamid", (pl.col("year") + 1).alias("year"),
                     pl.col("rating_ct").alias("rating_ct_lag"), pl.col("rating_t").alias("rating_t_lag")])
    rpj = rp.join(same, on=["steamid", "year"], how="left").join(lag, on=["steamid", "year"], how="left")
    rpj = rpj.with_columns(
        pl.when(pl.col("side") == "ct").then(pl.col("rating_ct")).otherwise(pl.col("rating_t")).alias("rating_same"),
        pl.when(pl.col("side") == "ct").then(pl.col("rating_ct_lag")).otherwise(pl.col("rating_t_lag")).alias("rating_lag"))
    pagg = rpj.group_by(["match_id", "round_num", "side"]).agg(
        rating_same_mean=pl.col("rating_same").mean(), n_same=pl.col("rating_same").is_not_null().sum(),
        rating_lag_mean=pl.col("rating_lag").mean(), n_lag=pl.col("rating_lag").is_not_null().sum())
    for side in ["ct", "t"]:
        a = pagg.filter(pl.col("side") == side).drop("side").rename(
            {c: f"{side}_{c}" for c in ["rating_same_mean", "n_same", "rating_lag_mean", "n_lag"]})
        r = r.join(a, on=["match_id", "round_num"], how="left")
    r = r.with_columns([pl.col(c).fill_null(1.0) for c in ["ct_rating_same_mean", "t_rating_same_mean",
                                                            "ct_rating_lag_mean", "t_rating_lag_mean"]])
    r = r.with_columns((pl.col("ct_rating_same_mean") - pl.col("t_rating_same_mean")).alias("rating_diff_same"),
                       (pl.col("ct_rating_lag_mean") - pl.col("t_rating_lag_mean")).alias("rating_diff_lag"))
    return r


R1 = ["round_num", "second_half", "is_pistol_round", "overtime", "ct_score", "t_score", "score_diff",
      "ct_to_win", "t_to_win"]
R2 = ["ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
      "ct_armor_total", "t_armor_total", "ct_defuse_kits", "ct_smokes", "ct_flashes", "ct_fire", "ct_he",
      "t_smokes", "t_flashes", "t_fire", "t_he", "ct_util_total", "t_util_total", "utility_advantage",
      "ct_awp_alive", "t_awp_alive", "ct_equip_min", "t_equip_min", "ct_equip_std", "t_equip_std",
      "ct_helmets", "t_helmets", "ct_n_full", "t_n_full"]
R3 = ["ct_loss_streak", "t_loss_streak", "ct_won_prev", "t_won_prev", "ct_prev_equip", "t_prev_equip",
      "momentum_diff", "streak_diff"]
R4_SAME = ["ct_rank_same", "t_rank_same", "lrank_diff_same", "ct_form", "t_form", "form_diff"]
R4_LAG = ["ct_rank_lag", "t_rank_lag", "lrank_diff_lag", "ct_rank_lag_missing", "t_rank_lag_missing",
          "ct_form", "t_form", "form_diff"]
R5_SAME = ["ct_rating_same_mean", "t_rating_same_mean", "rating_diff_same"]
R5_LAG = ["ct_rating_lag_mean", "t_rating_lag_mean", "rating_diff_lag", "ct_n_lag", "t_n_lag"]

LADDER = {
    "R1 format+score": R1,
    "R2 +buy": R1 + R2,
    "R3 +history": R1 + R2 + R3,
    "R4 +team prior (same-yr, leaky)": R1 + R2 + R3 + R4_SAME,
    "R5 +player prior (same-yr, leaky)": R1 + R2 + R3 + R4_SAME + R5_SAME,
    "R4L +team prior (lagged)": R1 + R2 + R3 + R4_LAG,
    "R5L +player prior (lagged)": R1 + R2 + R3 + R4_LAG + R5_LAG,
    "priors only (same-yr)": R4_SAME + R5_SAME + ["second_half", "is_pistol_round"],
    "priors only (lagged)": R4_LAG + R5_LAG + ["second_half", "is_pistol_round"],
}


def oof_round(df: pl.DataFrame, cols, C=1.0):
    X = np.nan_to_num(df[cols].to_numpy().astype(float)); y = df["ct_won"].to_numpy().astype(int)
    g = df["match_id"].to_numpy(); oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof, y


def boot_paired(groups, y, preds: dict, B=B, seed=0):
    rng = np.random.default_rng(seed); uniq = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in uniq}
    res = {n: {"auc": [], "ll": [], "brier": []} for n in preds}
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick]); yy = y[idx]
        if len(np.unique(yy)) < 2: continue
        for n, p in preds.items():
            res[n]["auc"].append(roc_auc_score(yy, p[idx])); res[n]["ll"].append(log_loss(yy, p[idx], labels=[0, 1]))
            res[n]["brier"].append(brier_score_loss(yy, p[idx]))
    return {n: {k: np.array(v) for k, v in d.items()} for n, d in res.items()}


def q(a): return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["supplement", "export"], default="supplement")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    print(f"round-start table source: {args.source}")
    r = build_table(args.source)
    r = r.sort(["match_id", "round_num"])
    y = r["ct_won"].to_numpy().astype(int); g = r["match_id"].to_numpy()
    print(f"round table: {r.shape}, matches {r['match_id'].n_unique()}, base {y.mean():.4f}")
    print(f"lagged coverage: rank {1-r['ct_rank_lag_missing'].mean():.2%} of CT rows; player lag n>=1: "
          f"{(r['ct_n_lag']>0).mean():.2%}; by year: "
          + ", ".join(f"{yr}: {(r.filter(pl.col('year')==yr)['ct_n_lag']>0).mean():.0%}" for yr in sorted(r['year'].unique())))

    # ---- A. ladder --------------------------------------------------------------------------
    preds = {}
    rows = []
    for name, cols in LADDER.items():
        cols = [c for c in cols if c in r.columns]
        p, _ = oof_round(r, cols)
        preds[name] = p
        rows.append(dict(rung=name, ncols=len(cols), AUC=roc_auc_score(y, p), logloss=log_loss(y, p),
                         brier=brier_score_loss(y, p), ECE=ece(y, p)))
        print(f"  {name:36s} n={len(cols):2d} AUC {rows[-1]['AUC']:.4f} ll {rows[-1]['logloss']:.4f} "
              f"brier {rows[-1]['brier']:.4f} ECE {rows[-1]['ECE']:.4f}")

    # ---- B. the per-second model's first snapshot as p0 ---------------------------------------
    tr = pl.read_parquet(ROOT / "data" / "training_dataset.parquet").sort(["match_id", "round_num", "tick"])
    ps_oof = {}
    for fs in ["A", "EB2"]:
        p, _ = oof_predict(tr, FEATURE_SETS[fs], "logreg")
        ps_oof[fs] = p
    snap = tr.select(["match_id", "round_num", "tick", "ct_won", "time_elapsed_sec"]).with_columns(
        p_A=pl.Series(ps_oof["A"]), p_EB2=pl.Series(ps_oof["EB2"]))
    first = snap.group_by(["match_id", "round_num"], maintain_order=True).first()
    fj = r.select(["match_id", "round_num"]).join(first, on=["match_id", "round_num"], how="left")
    for fs in ["A", "EB2"]:
        p = fj[f"p_{fs}"].to_numpy()
        name = f"first-snapshot {fs} (per-second model)"
        preds[name] = p
        rows.append(dict(rung=name, ncols=len(FEATURE_SETS[fs]), AUC=roc_auc_score(y, p), logloss=log_loss(y, p),
                         brier=brier_score_loss(y, p), ECE=ece(y, p)))
        print(f"  {name:36s}        AUC {rows[-1]['AUC']:.4f} ll {rows[-1]['logloss']:.4f} "
              f"brier {rows[-1]['brier']:.4f} ECE {rows[-1]['ECE']:.4f}")

    # ---- B2. diagnostics: why is the first snapshot better than the round-level fit? -------------
    #  (i) the exact set-A columns fitted at round level (same information, 4,866 rows vs 476k)
    #  (ii) R2 with stronger regularisation
    #  (iii) STACKED: first-snapshot A logit as an offset + history + priors (the design candidate)
    fj_full = r.select(["match_id", "round_num"]).join(
        snap.select(["match_id", "round_num", "tick", "p_A", "p_EB2"]).group_by(["match_id", "round_num"], maintain_order=True).first(),
        on=["match_id", "round_num"], how="left")
    zA = np.log(np.clip(fj_full["p_A"].to_numpy(), 1e-6, 1 - 1e-6) / np.clip(1 - fj_full["p_A"].to_numpy(), 1e-6, 1))
    r = r.with_columns(zA=pl.Series(zA))
    # rounds.csv's `bomb_planted` is a ROUND OUTCOME (was the bomb planted this round), not the
    # round-start flag the per-second table carries under the same name -> excluded (it leaks).
    a_cols = [c for c in FEATURE_SETS["A"] if c in r.columns and c != "bomb_planted"]
    extra = {
        "A-cols at round level (17)": (a_cols, 1.0),
        "R2 +buy, C=0.1": (R1 + R2, 0.1),
        "R2 +buy, C=0.01": (R1 + R2, 0.01),
        "STACK zA + history": (["zA"] + R3, 1.0),
        "STACK zA + history + priors(same-yr)": (["zA"] + R3 + R4_SAME + R5_SAME, 1.0),
        "STACK zA + history + priors(lagged)": (["zA"] + R3 + R4_LAG + R5_LAG, 1.0),
    }
    for name, (cols, C) in extra.items():
        cols = [c for c in cols if c in r.columns]
        p, _ = oof_round(r, cols, C=C)
        preds[name] = p
        rows.append(dict(rung=name, ncols=len(cols), AUC=roc_auc_score(y, p), logloss=log_loss(y, p),
                         brier=brier_score_loss(y, p), ECE=ece(y, p)))
        print(f"  {name:36s} n={len(cols):2d} AUC {rows[-1]['AUC']:.4f} ll {rows[-1]['logloss']:.4f} "
              f"brier {rows[-1]['brier']:.4f} ECE {rows[-1]['ECE']:.4f}")

    bs = boot_paired(g, y, preds)
    ref = "R1 format+score"; ref2 = "R2 +buy"
    for row in rows:
        n = row["rung"]; a = bs[n]["auc"]
        row["AUC_lo"], row["AUC_hi"] = q(a)
        for rname, tag in [(ref, "R1"), (ref2, "R2"), ("first-snapshot EB2 (per-second model)", "snapEB2")]:
            if n != rname:
                d = a - bs[rname]["auc"]; dl = bs[n]["ll"] - bs[rname]["ll"]
                row[f"dAUC_vs_{tag}"] = float(d.mean()); row[f"dAUC_vs_{tag}_lo"], row[f"dAUC_vs_{tag}_hi"] = q(d)
                row[f"dLL_vs_{tag}"] = float(dl.mean()); row[f"dLL_vs_{tag}_lo"], row[f"dLL_vs_{tag}_hi"] = q(dl)
    pl.DataFrame(rows).write_csv(OUT / f"t0_pilot_ladder_{args.source}.csv")
    # per-round OOF priors for the example figures (src/viz/t0_examples.py)
    keep = {"R2 +buy": "p0_buy", "R3 +history": "p0_hist",
            "R5 +player prior (same-yr, leaky)": "p0_priors_sameyr", "R5L +player prior (lagged)": "p0_priors_lag",
            "first-snapshot A (per-second model)": "p0_snapA", "first-snapshot EB2 (per-second model)": "p0_snapEB2",
            "STACK zA + history": "p0_stack_hist", "STACK zA + history + priors(same-yr)": "p0_stack_sameyr",
            "STACK zA + history + priors(lagged)": "p0_stack_lag"}
    pl.DataFrame({"match_id": r["match_id"], "round_num": r["round_num"], "ct_won": y,
                  **{v: preds[k] for k, v in keep.items() if k in preds}}
                 ).write_parquet(OUT / "t0_pilot_predictions.parquet")
    print("\n paired deltas (AUC) vs R2 +buy:")
    for row in rows:
        if "dAUC_vs_R2" in row:
            print(f"  {row['rung']:36s} {row['dAUC_vs_R2']:+.4f} ({row['dAUC_vs_R2_lo']:+.4f}, {row['dAUC_vs_R2_hi']:+.4f})"
                  f"   dLL {row['dLL_vs_R2']:+.4f} ({row['dLL_vs_R2_lo']:+.4f}, {row['dLL_vs_R2_hi']:+.4f})")

    # ---- C. martingale identities, per-second EB2 anchored at the t=0 model ------------------
    p0 = preds["R3 +history"]                      # the demo-derivable t=0 model (no external priors)
    r0 = r.select(["match_id", "round_num"]).with_columns(p0=pl.Series(p0))
    s = snap.join(r0, on=["match_id", "round_num"], how="inner").sort(["match_id", "round_num", "tick"])
    per = s.group_by(["match_id", "round_num"], maintain_order=True).agg(
        p0=pl.col("p0").first(), y=pl.col("ct_won").first(), p_first=pl.col("p_EB2").first(),
        p_last=pl.col("p_EB2").last(),
        qv=(pl.col("p_EB2").diff().fill_null(0.0) ** 2).sum(),
        jump0=(pl.col("p_EB2").first() - pl.col("p0").first()),
        n=pl.len())
    # Full identity for a discrete-time martingale that starts at p0 and ends at Y:
    #   (Y - p0)^2 = jump0^2 + sum_t dp_t^2 + (Y - p_last)^2 + cross terms,  E[cross] = 0.
    # The final jump (last snapshot -> outcome) must be included; omitting it understates the path.
    per = per.with_columns(term=(pl.col("y") - pl.col("p0")) ** 2, implied=pl.col("p0") * (1 - pl.col("p0")),
                           final=(pl.col("y") - pl.col("p_last")) ** 2)
    per = per.with_columns(qv_from_p0=(pl.col("jump0") ** 2 + pl.col("qv") + pl.col("final")))
    print("\n martingale identities (per-second EB2, anchored at t=0 model R3):")
    print(f"  mean p0 {per['p0'].mean():.4f} | mean first-snapshot {per['p_first'].mean():.4f} | mean jump at 0+ {per['jump0'].mean():+.4f} (|jump| {per['jump0'].abs().mean():.4f})")
    print(f"  E[(Y-p0)^2] {per['term'].mean():.4f} vs E[p0(1-p0)] {per['implied'].mean():.4f}  ratio {per['term'].mean()/per['implied'].mean():.3f}")
    print(f"  E[jump0^2 + sum dp^2 + (Y-p_last)^2] {per['qv_from_p0'].mean():.4f} vs E[(Y-p0)^2] {per['term'].mean():.4f}  "
          f"ratio {per['qv_from_p0'].mean()/per['term'].mean():.3f}   (path QV alone {per['qv'].mean():.4f}, final jump {per['final'].mean():.4f})")
    per = per.with_columns(bin=(pl.col("p0") * 10).floor().clip(0, 9))
    tab = per.group_by("bin").agg(n=pl.len(), p0=pl.col("p0").mean(), y=pl.col("y").mean(), p_first=pl.col("p_first").mean(),
                                  term=pl.col("term").mean(), implied=pl.col("implied").mean(), qv=pl.col("qv_from_p0").mean()).sort("bin")
    print(tab)
    tab.write_csv(OUT / "t0_pilot_martingale.csv")

    # ---- D. opening-kill increment ----------------------------------------------------------
    KC = ["tick", "round_num", "attacker_side", "victim_side", "is_warmup_period", "is_freeze_period"]
    ks = []
    for f in glob.glob(str(ROOT / "data" / "parquet" / "kills" / "*.parquet")):
        d = pl.read_parquet(f, columns=KC).with_columns(pl.lit(os.path.basename(f)[:-8]).alias("match_id"))
        ks.append(d)
    k = pl.concat(ks, how="diagonal").filter((pl.col("is_warmup_period") == False) & (pl.col("is_freeze_period") == False))
    fk = k.sort(["match_id", "round_num", "tick"]).group_by(["match_id", "round_num"], maintain_order=True).first()
    fk = fk.select(["match_id", "round_num", pl.col("tick").alias("kill_tick"), "victim_side"])
    sj = s.join(fk, on=["match_id", "round_num"], how="inner")
    before = (sj.filter(pl.col("tick") <= pl.col("kill_tick")).group_by(["match_id", "round_num"]).agg(
        p_before=pl.col("p_EB2").last(), p0=pl.col("p0").first(), y=pl.col("ct_won").first(), victim=pl.col("victim_side").first()))
    after = (sj.filter(pl.col("tick") > pl.col("kill_tick")).group_by(["match_id", "round_num"]).agg(p_after=pl.col("p_EB2").first()))
    ev = before.join(after, on=["match_id", "round_num"], how="inner").with_columns(jump=pl.col("p_after") - pl.col("p_before"))
    print(f"\n opening kill: {ev.height} rounds with before/after snapshots")
    for vs in ["ct", "t"]:
        e = ev.filter(pl.col("victim") == vs)
        print(f"  victim {vs.upper():2s}: n={e.height:4d} p_before {e['p_before'].mean():.3f} -> p_after {e['p_after'].mean():.3f} "
              f"(jump {e['jump'].mean():+.3f}) | realised CT win {e['y'].mean():.3f} | post-kill ECE {ece(e['y'].to_numpy(), e['p_after'].to_numpy()):.4f}")
    ev = ev.with_columns(bin=(pl.col("p0") * 5).floor().clip(0, 4))
    t2 = ev.group_by(["victim", "bin"]).agg(n=pl.len(), p0=pl.col("p0").mean(), p_before=pl.col("p_before").mean(),
                                            p_after=pl.col("p_after").mean(), y=pl.col("y").mean(), jump=pl.col("jump").mean()).sort(["victim", "bin"])
    print(t2)
    t2.write_csv(OUT / "t0_pilot_openingkill.csv")
    r.write_parquet(OUT / "t0_round_table.parquet")
    print("wrote outputs/t0_pilot_*.csv, outputs/t0_round_table.parquet")


if __name__ == "__main__":
    main()
