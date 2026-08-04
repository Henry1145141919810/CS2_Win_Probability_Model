"""Full model benchmark for Firepower v3 (team-ranking-weighted).

Proxy evaluation — mathematically exact because all 5 alive CT (or T) players
share the same team, so summed stats scale linearly with team weight:
    ct_col_v3 = ct_col_v2 x ct_team_weight

Procedure:
  1. For each match, read round-1 tick data to identify the first-half CT team.
  2. Look up that team's rank weight via player_team_year.csv + team_rankings.csv.
  3. Apply: rounds 1-12 use first-half weights; rounds 13+ swap CT/T.
  4. Run 5-fold GroupKFold on EB2, EFB2 (v2), and three EFB3 variants (v3).

Models: logreg, xgb, lgbm, catboost, rf
Output: outputs/firepower_v3_benchmark.csv

Usage: python src/models/eval_firepower_v3.py
"""
from __future__ import annotations
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from features.firepower_v3 import (  # noqa: E402
    FIREPOWER_COLS_V3,
    _player_team_lookup,
    _team_rank_lookup,
)
from models.train_pipeline import FEATURE_SETS, make_model  # noqa: E402

TICKS_DIR = ROOT / "data" / "parquet" / "ticks"
TRAIN = ROOT / "data" / "training_dataset.parquet"
OUT = ROOT / "outputs" / "firepower_v3_benchmark.csv"

MODELS = ["logreg", "xgb", "lgbm", "catboost", "rf"]
DEFAULT_RANK = 35

# V2 -> V3 column name mapping (18 pairs; AWP sniping has no v3 equivalent)
V2_TO_V3 = {
    "ct_rating_sum": "ct_rating_v3",
    "t_rating_sum": "t_rating_v3",
    "ct_adr_sum": "ct_adr_v3",
    "t_adr_sum": "t_adr_v3",
    "ct_kast_mean": "ct_kast_v3",
    "t_kast_mean": "t_kast_v3",
    "ct_hltv_firepower_sum": "ct_hltv_fp_v3",
    "t_hltv_firepower_sum": "t_hltv_fp_v3",
    "ct_entry_sum": "ct_entry_v3",
    "t_entry_sum": "t_entry_v3",
    "ct_trading_sum": "ct_trading_v3",
    "t_trading_sum": "t_trading_v3",
    "ct_opening_sum": "ct_opening_v3",
    "t_opening_sum": "t_opening_v3",
    "ct_clutch_score": "ct_clutch_v3",
    "t_clutch_score": "t_clutch_v3",
    "ct_weighted_utility": "ct_util_v3",
    "t_weighted_utility": "t_util_v3",
}


def _weight_from_rank(rank: int, formula: str) -> float:
    rank = max(1, rank)
    if formula == "log2":
        return 1.0 / math.log2(rank + 1)
    if formula == "inv":
        return 1.0 / rank
    return max(0.0, (31 - rank) / 30.0)


def build_match_weights(
    match_ids: list[str], year_by_match: dict[str, int]
) -> dict:
    """For each match determine first-half CT and T rank weights.

    Returns {match_id: {formula: {'h1_ct': float, 'h1_t': float}}}
    """
    player_team_lut = _player_team_lookup()
    rank_lut = _team_rank_lookup()

    def _team_rank(sids: list, year: int) -> int:
        teams = [player_team_lut.get((int(s), year)) for s in sids]
        teams = [tm for tm in teams if tm is not None]
        if not teams:
            return DEFAULT_RANK
        team = Counter(teams).most_common(1)[0][0]
        return rank_lut.get((team, year), DEFAULT_RANK)

    result: dict = {}
    unique = list(dict.fromkeys(match_ids))
    print(f"Building team-weight lookup for {len(unique)} matches...")

    for i, mid in enumerate(unique, 1):
        tick_path = TICKS_DIR / f"{mid}.parquet"
        if not tick_path.exists():
            continue
        year = year_by_match.get(mid, 2024)
        try:
            tdf = pl.read_parquet(
                tick_path,
                columns=["round_num", "side", "health", "steamid"],
            ).filter(
                (pl.col("round_num") == 1) & (pl.col("health") > 0)
            )
        except Exception:
            continue
        if tdf.is_empty():
            continue

        ct_sids = tdf.filter(pl.col("side") == "ct")["steamid"].to_list()
        t_sids = tdf.filter(pl.col("side") == "t")["steamid"].to_list()
        ct_rank = _team_rank(ct_sids, year)
        t_rank = _team_rank(t_sids, year)

        result[mid] = {
            formula: {
                "h1_ct": _weight_from_rank(ct_rank, formula),
                "h1_t": _weight_from_rank(t_rank, formula),
            }
            for formula in ("log2", "inv", "linear")
        }
        if i % 50 == 0:
            print(f"  {i}/{len(unique)}")

    print(f"  done -- {len(result)}/{len(unique)} matches resolved")
    return result


def apply_v3_weights(
    df: pl.DataFrame, match_weights: dict, formula: str
) -> pl.DataFrame:
    """Add v3 columns: each v2 firepower col multiplied by the team weight.

    Rounds 1-12 use first-half weights; rounds 13+ swap CT/T sides.
    """
    mid_arr = df["match_id"].to_list()
    rnum_arr = df["round_num"].to_numpy()
    n = len(df)
    default = _weight_from_rank(DEFAULT_RANK, formula)

    ct_w = np.full(n, default)
    t_w = np.full(n, default)

    for j in range(n):
        info = match_weights.get(mid_arr[j], {}).get(formula)
        if info is None:
            continue
        if rnum_arr[j] <= 12:
            ct_w[j] = info["h1_ct"]
            t_w[j] = info["h1_t"]
        else:
            ct_w[j] = info["h1_t"]
            t_w[j] = info["h1_ct"]

    new_cols = {}
    for v2_col, v3_col in V2_TO_V3.items():
        if v2_col not in df.columns:
            continue
        v2 = df[v2_col].to_numpy()
        weight = ct_w if v2_col.startswith("ct_") else t_w
        new_cols[v3_col] = (v2 * weight).tolist()

    return df.with_columns([
        pl.Series(name=k, values=v) for k, v in new_cols.items()
    ])


def run_cv(df: pl.DataFrame, cols: list[str], label: str) -> dict:
    """5-fold GroupKFold AUC for each model."""
    X = np.nan_to_num(df.select(cols).to_numpy().astype(float))
    y = df["ct_won"].to_numpy()
    g = df["match_id"].to_numpy()
    row: dict = {"feature_set": label}
    for mdl in MODELS:
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = make_model(mdl).fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        row[mdl] = round(float(roc_auc_score(y, oof)), 4)
        print(f"  {label:20s} {mdl:9s} AUC {row[mdl]:.4f}")
    return row


def main():
    df = pl.read_parquet(TRAIN)
    n_matches = df["match_id"].n_unique()
    print(f"Loaded {df.height} snapshots, {n_matches} matches\n")

    from features.firepower import _year_by_demo, DEFAULT_YEAR  # noqa: E402
    year_by_match = {
        m: (_year_by_demo().get(m) or DEFAULT_YEAR)
        for m in df["match_id"].unique().to_list()
    }
    match_weights = build_match_weights(
        df["match_id"].to_list(), year_by_match
    )

    rows = []

    print("\n--- EB2 (no firepower) ---")
    rows.append(run_cv(df, FEATURE_SETS["EB2"], "EB2"))

    print("\n--- EFB2 (v2 raw sum) ---")
    rows.append(run_cv(df, FEATURE_SETS["EFB2"], "EFB2"))

    eb2_base = FEATURE_SETS["EB2"]
    efb3_cols = eb2_base + FIREPOWER_COLS_V3

    for formula, label in [
        ("log2", "EFB3_log2"),
        ("inv", "EFB3_inv"),
        ("linear", "EFB3_linear"),
    ]:
        print(f"\n--- {label} ---")
        df_v3 = apply_v3_weights(df, match_weights, formula)
        rows.append(run_cv(df_v3, efb3_cols, label))

    out_df = pl.DataFrame(rows)
    out_df.write_csv(OUT)
    print(f"\nWrote {OUT}")
    print(out_df)


if __name__ == "__main__":
    main()
