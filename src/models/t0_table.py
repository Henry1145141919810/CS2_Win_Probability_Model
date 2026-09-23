"""Round-start (t=0) tables built DIRECTLY from the raw supplement, bypassing the Wyner export.

Sources (all under data/cs2_inferno_raw_supplement_v1/):
  training_2024_2025/modelling/training_dataset_per_second.parquet   first snapshot of every round
  training_2024_2025/parsed/rounds/*.parquet                          winner, freeze_end, reason
  training_2024_2025/parsed/ticks/*.parquet                           the ten players at the first snapshot
  reference/demo_list_final.csv, demo_year_map.csv                    event, date, teams, map winner, season
  reference/player_team_year.csv, team_rankings.csv                   clan -> org -> HLTV rank
  reference/player_stats_sided.csv                                    per-player side-split season stats

Returns three frames shaped like the export's rounds.csv / round_players.csv / matches.csv, so the
pilot ladder can run on either source and the two can be diffed. Scores are TEAM scores rebuilt from
round winners (the pipeline's ct_score/t_score are cumulative side wins and are ignored).
"""
from __future__ import annotations
import re
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SUPP = ROOT / "data" / "cs2_inferno_raw_supplement_v1"
TRAIN = SUPP / "training_2024_2025"
REF = SUPP / "reference"

BASE_STATE = [
    "time_elapsed_sec", "ct_players_alive", "t_players_alive",
    "ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
    "ct_health_total", "t_health_total", "ct_armor_total", "t_armor_total", "ct_defuse_kits",
    "ct_smokes", "ct_flashes", "ct_fire", "ct_he", "t_smokes", "t_flashes", "t_fire", "t_he",
    "ct_util_total", "t_util_total", "utility_advantage", "ct_awp_alive", "t_awp_alive",
]
TICK_COLS = ["round_num", "tick", "steamid", "side", "team_clan_name", "has_helmet", "has_defuser",
             "current_equip_value"]


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    for suf in ("esports", "esport", "gaming", "team", "clan", "globant"):
        s = s.replace(suf, "")
    return s


def build(era_dir: Path = TRAIN, per_second: Path | None = None, demo_list: Path | None = None):
    """era_dir: training_2024_2025 (default) or holdout_2026; per_second: the era's modelling table;
    demo_list: curated match list (demo_list_final.csv for training, demo_list_2026_test.csv for 2026)."""
    per_second = per_second or (era_dir / "modelling" / "training_dataset_per_second.parquet")
    demo_list = demo_list or (REF / "demo_list_final.csv")
    tr = pl.read_parquet(per_second, columns=["match_id", "round_num", "tick", "ct_won"] + BASE_STATE)
    first = (tr.sort(["match_id", "round_num", "tick"])
               .group_by(["match_id", "round_num"], maintain_order=True).first()
               .rename({"tick": "base_state_tick"}))
    matches = sorted(first["match_id"].unique().to_list())

    # ---- rounds channel + ten players at the base tick --------------------------------------
    rr, pr = [], []
    for mid in matches:
        rd = pl.read_parquet(era_dir / "parsed" / "rounds" / f"{mid}.parquet",
                             columns=["round_num", "winner", "reason", "freeze_end", "end"])
        rr.append(rd.with_columns(pl.col("reason").cast(pl.Utf8), pl.lit(mid).alias("match_id")))
        tk = pl.read_parquet(era_dir / "parsed" / "ticks" / f"{mid}.parquet", columns=TICK_COLS)
        idx = first.filter(pl.col("match_id") == mid).select(["round_num", "base_state_tick"])
        pr.append(tk.join(idx, on="round_num", how="inner")
                    .filter(pl.col("tick") == pl.col("base_state_tick"))
                    .filter(pl.col("side").is_not_null() & pl.col("team_clan_name").is_not_null())
                    .with_columns(pl.lit(mid).alias("match_id")))
    rounds_ch = pl.concat(rr, how="diagonal_relaxed")
    players = pl.concat(pr, how="diagonal_relaxed")

    # ---- match metadata: season year, curated teams and winner --------------------------------
    dl = pl.read_csv(demo_list, infer_schema_length=0).rename({"demo_id": "match_id"})
    ym = pl.read_csv(REF / "demo_year_map.csv", infer_schema_length=0).select(["demo_id", "year"]).rename({"demo_id": "match_id"})
    meta = (dl.select(["match_id", "event", "series_date", "team_a", "team_b",
                       pl.col("winner_team").alias("match_winner")])
              .join(ym, on="match_id", how="left")
              .with_columns(pl.when(pl.col("year").is_null() | (pl.col("year") == ""))
                              .then(pl.col("series_date").str.extract(r"(20\d\d)", 1))
                              .otherwise(pl.col("year")).alias("year"))
              .with_columns(pl.col("year").cast(pl.Int64, strict=False))
              .filter(pl.col("match_id").is_in(matches)))

    # ---- sides per round, canonical org per side (majority of the roster's player_team_year) ---
    side_team = (players.group_by(["match_id", "round_num", "side"])
                        .agg(pl.col("team_clan_name").mode().first().alias("clan")))
    pty = pl.read_csv(REF / "player_team_year.csv").with_columns(pl.col("steamid").cast(pl.Int64), pl.col("year").cast(pl.Int64))
    # a match with no recorded date (one off-list demo): infer the season whose rosters match best
    for mid in meta.filter(pl.col("year").is_null())["match_id"].to_list():
        ids = players.filter(pl.col("match_id") == mid)["steamid"].cast(pl.Int64).unique()
        best = (pty.filter(pl.col("steamid").is_in(ids)).group_by("year").len().sort("len", descending=True))
        yr = int(best["year"][0]) if best.height else 2024
        print(f"[info] inferred year={yr} for {mid} ({int(best['len'][0]) if best.height else 0}/{ids.len()} players matched)")
        meta = meta.with_columns(pl.when(pl.col("match_id") == mid).then(pl.lit(yr)).otherwise(pl.col("year")).alias("year"))
    pj = (players.join(meta.select(["match_id", "year"]), on="match_id", how="left")
                 .with_columns(pl.col("steamid").cast(pl.Int64))
                 .join(pty, on=["steamid", "year"], how="left"))
    # the one match without a year: infer from the roster's best-matching season
    canon = (pj.filter(pl.col("team_canonical").is_not_null())
               .group_by(["match_id", "round_num", "side"])
               .agg(pl.col("team_canonical").mode().first().alias("canonical")))
    side_team = side_team.join(canon, on=["match_id", "round_num", "side"], how="left")
    ct = side_team.filter(pl.col("side") == "ct").drop("side").rename({"clan": "ct_team_clan", "canonical": "ct_team_canonical"})
    tt = side_team.filter(pl.col("side") == "t").drop("side").rename({"clan": "t_team_clan", "canonical": "t_team_canonical"})

    rounds = (first.join(rounds_ch, on=["match_id", "round_num"], how="left")
                   .join(ct, on=["match_id", "round_num"], how="left")
                   .join(tt, on=["match_id", "round_num"], how="left")
                   .join(meta, on="match_id", how="left"))
    rounds = rounds.with_columns(
        half=pl.when(pl.col("round_num") <= 12).then(1).otherwise(2),
        is_pistol_round=pl.col("round_num").is_in([1, 13]).cast(pl.Int8),
        round_winner_team_clan=pl.when(pl.col("ct_won") == 1).then(pl.col("ct_team_clan")).otherwise(pl.col("t_team_clan")),
    ).sort(["match_id", "round_num"])
    # label sanity: per-second ct_won must equal the rounds channel winner
    mism = (rounds.filter(pl.col("winner").is_not_null())
                  .filter((pl.col("ct_won") == 1) != (pl.col("winner") == "ct")).height)
    assert mism == 0, f"{mism} rounds where ct_won disagrees with the rounds channel"

    # ---- per-player loadout aggregates (the R2 spread features) --------------------------------
    players = players.with_columns(pl.col("steamid").cast(pl.Int64)).join(meta.select(["match_id", "year"]), on="match_id", how="left")
    return rounds, players, meta


if __name__ == "__main__":
    r, p, m = build()
    print("rounds", r.shape, "players", p.shape, "matches", m.shape)
    print("rounds/match:", r.group_by("match_id").len()["len"].min(), "..", r.group_by("match_id").len()["len"].max())
    print("canonical resolved:", r["ct_team_canonical"].is_not_null().mean())
