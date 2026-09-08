"""Build the round-level export for Prof. Wyner (CS2 Inferno Round-Level Dataset).

One row per round across the full TRAINING set (220 Inferno matches, 4,866 rounds), with the
match identity, the tournament, which team played which side, the round and match winners, the
round-start ("base") state, and the per-player roster on the server -- plus the reference tables
(player season stats, team rankings) needed to join skill and team strength onto any round.

Sources (all already in the repo; nothing is re-parsed):
  data/training_dataset.parquet          per-second modelling table -> round-start base state + label
  data/parquet_defuse/rounds/*.parquet   round winner / end reason / bomb / tick boundaries
  data/parquet_defuse/ticks/*.parquet    per-player side, team_clan_name, steamid, equipment
  configs/demo_list_final.csv            event, stage, date, teams, match score + winner, HLTV url
  configs/demo_year_map.csv              demo -> season year
  configs/player_stats_sided.csv         player x year sided HLTV stats
  configs/team_rankings.csv              team x year HLTV world rank
  configs/player_team_year.csv           steamid -> team_canonical per year (name crosswalk)

Run:  python src/data/build_wyner_export.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import polars as pl

# In-game clan names carry sponsors/abbreviations; the rankings table uses org names. Normalising
# both (lowercase, strip non-alphanumerics and sponsor/suffix noise) matches 55 of 57 outright;
# these two need an explicit alias.
TEAM_ALIASES = {
    "nip": "Ninjas in Pyjamas",
    # IMPERIAL sits outside the ranked top-35, so it is absent from team_rankings.csv; this is the
    # canonical spelling used by player_team_year.csv, so the name still joins the roster tables.
    "imperialsportsbet": "IMPERIAL",
}


# awpy leaves a minority of round-end reasons as raw engine enum codes. Each code's
# winner/bomb-planted signature matches exactly one decoded label, so they are mapped back.
REASON_CODES = {
    "1": "bomb_exploded", "7": "bomb_defused", "8": "t_killed",
    "9": "ct_killed", "12": "time_ran_out",
}


def _norm_team(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    for suf in ("esports", "esport", "gaming", "team", "clan", "globant"):
        s = s.replace(suf, "")
    return s

ROOT = Path(__file__).resolve().parents[2]
TREE = ROOT / "data" / "parquet_defuse"
OUT = ROOT / "exports" / "cs2_inferno_round_level_v1"

# round-start ("base state") columns lifted from the modelling table's first snapshot of each round
BASE_STATE = [
    "time_elapsed_sec", "ct_score", "t_score", "score_diff",
    "ct_players_alive", "t_players_alive",
    "ct_equipment_value", "t_equipment_value", "ct_economy_class", "t_economy_class",
    "ct_health_total", "t_health_total", "ct_armor_total", "t_armor_total",
    "ct_defuse_kits",
    "ct_smokes", "ct_flashes", "ct_fire", "ct_he",
    "t_smokes", "t_flashes", "t_fire", "t_he",
    "ct_util_total", "t_util_total", "utility_advantage",
    "ct_awp_alive", "t_awp_alive",
]

PLAYER_TICK_COLS = [
    "round_num", "tick", "steamid", "name", "side", "team_clan_name",
    "health", "armor", "has_helmet", "has_defuser", "current_equip_value", "place",
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- training table -> base state
    tr = pl.read_parquet(ROOT / "data" / "training_dataset.parquet")
    keep = ["match_id", "round_num", "tick", "ct_won"] + [c for c in BASE_STATE if c in tr.columns]
    missing = [c for c in BASE_STATE if c not in tr.columns]
    if missing:
        print(f"[warn] base-state columns absent from the modelling table: {missing}")
    first = (tr.select(keep)
               .sort(["match_id", "round_num", "tick"])
               .group_by(["match_id", "round_num"], maintain_order=True)
               .first()
               .rename({"tick": "base_state_tick"}))
    matches = sorted(first["match_id"].unique().to_list())
    print(f"training set: {len(matches)} matches, {first.height} rounds")

    # ---------------------------------------------------------------------- match metadata
    dl = pl.read_csv(ROOT / "configs" / "demo_list_final.csv", infer_schema_length=0)
    ym = pl.read_csv(ROOT / "configs" / "demo_year_map.csv", infer_schema_length=0)
    ym = ym.select(["demo_id", "year"]).rename({"demo_id": "match_id"})
    meta = (dl.rename({"demo_id": "match_id"})
              .select(["match_id", "event", "stage", "series_date", "team_a", "team_b",
                       "team_score", "winner_team", "hltv_match_url"])
              .join(ym, on="match_id", how="left")
              .filter(pl.col("match_id").is_in(matches)))
    # fill the one missing year from the series date
    meta = meta.with_columns(
        pl.when(pl.col("year").is_null() | (pl.col("year") == ""))
          .then(pl.col("series_date").str.extract(r"(20\d\d)", 1))
          .otherwise(pl.col("year")).alias("year"))
    meta = meta.with_columns(pl.col("year").cast(pl.Int64, strict=False))
    still = meta.filter(pl.col("year").is_null())
    if still.height:
        print(f"[warn] {still.height} match(es) still without a year: "
              f"{still['match_id'].to_list()}")

    # -------------------------------------------------- per-match: rounds channel + player rosters
    tick_index = {int(r["round_num"]): int(r["base_state_tick"]) for r in []}  # placeholder
    round_rows, player_rows, bomb_rows = [], [], []
    for i, mid in enumerate(matches, 1):
        if i % 40 == 0:
            print(f"  ...{i}/{len(matches)}")
        rd = pl.read_parquet(TREE / "rounds" / f"{mid}.parquet")
        # NB: the rounds channel's own bomb_site is unreliable (it labels every plant "bombsite_b"),
        # so the plant is taken from the bomb channel instead, which carries the real site.
        want = {"round_num", "winner", "reason", "start", "freeze_end", "end", "official_end"}
        rd = rd.select([c for c in rd.columns if c in want])
        rd = rd.with_columns(pl.lit(mid).alias("match_id"))
        round_rows.append(rd)

        bm = pl.read_parquet(TREE / "bomb" / f"{mid}.parquet",
                             columns=["round_num", "tick", "event", "X", "Y", "bombsite"])
        bm = (bm.filter(pl.col("event") == "plant")
                .group_by("round_num").first()
                .select([pl.col("round_num"),
                         pl.col("tick").alias("bomb_plant_tick"),
                         pl.col("X").alias("bomb_plant_x"),
                         pl.col("Y").alias("bomb_plant_y"),
                         pl.col("bombsite").alias("_site_raw")])
                .with_columns(pl.lit(mid).alias("match_id")))
        bomb_rows.append(bm)

        # player roster at each round's base-state tick
        want_ticks = [c for c in PLAYER_TICK_COLS]
        tk = pl.read_parquet(TREE / "ticks" / f"{mid}.parquet",
                             columns=[c for c in want_ticks])
        idx = (first.filter(pl.col("match_id") == mid)
                    .select(["round_num", "base_state_tick"]))
        # side/team are null for observer + coach entities that appear in some demos; drop them
        # so every round carries exactly the ten players actually on the server.
        pr = (tk.join(idx, on="round_num", how="inner")
                .filter(pl.col("tick") == pl.col("base_state_tick"))
                .filter(pl.col("side").is_not_null() & pl.col("team_clan_name").is_not_null())
                .with_columns(pl.lit(mid).alias("match_id")))
        player_rows.append(pr)

    rounds_ch = pl.concat(round_rows, how="diagonal_relaxed")
    players = pl.concat(player_rows, how="diagonal_relaxed")
    plants = pl.concat(bomb_rows, how="diagonal_relaxed")
    print(f"rounds channel rows: {rounds_ch.height} | player-round rows: {players.height}")

    # ------------------------------------------------------- which team played which side, per round
    side_team = (players.group_by(["match_id", "round_num", "side"])
                        .agg(pl.col("team_clan_name").mode().first().alias("team_clan")))
    ct_side = (side_team.filter(pl.col("side") == "ct")
                        .select(["match_id", "round_num", "team_clan"])
                        .rename({"team_clan": "ct_team_clan"}))
    t_side = (side_team.filter(pl.col("side") == "t")
                       .select(["match_id", "round_num", "team_clan"])
                       .rename({"team_clan": "t_team_clan"}))

    # resolve the canonical (rankable) team name by majority vote of each side's players
    pty = pl.read_csv(ROOT / "configs" / "player_team_year.csv")
    canon = {(int(r["steamid"]), int(r["year"])): r["team_canonical"]
             for r in pty.iter_rows(named=True)}
    yr_by_match = {r["match_id"]: r["year"] for r in meta.iter_rows(named=True)}

    # A few off-list demos carry no event date, so no season year. Infer it from the rosters:
    # pick the year whose player -> team mapping explains the most of this match's ten players.
    years_avail = sorted({y for (_s, y) in canon})
    for mid, y in list(yr_by_match.items()):
        if y is not None:
            continue
        sids = (players.filter(pl.col("match_id") == mid)["steamid"].unique().to_list())
        best, best_n = None, -1
        for cand in years_avail:
            n = sum(1 for s in sids if (int(s), cand) in canon)
            if n > best_n:
                best, best_n = cand, n
        yr_by_match[mid] = best
        print(f"[info] inferred year={best} for {mid} ({best_n}/{len(sids)} players matched)")
    meta = meta.with_columns(
        pl.col("match_id").replace_strict(yr_by_match, default=None).alias("year"))

    # Map the in-game clan name to the rankable canonical name BY NAME, not by player majority:
    # players transfer between orgs, so a majority vote over player->team rows misattributes a
    # side whenever its roster is recorded under a different org for that season.
    rank_names = pl.read_csv(ROOT / "configs" / "team_rankings.csv")["team_canonical"].unique().to_list()
    norm_to_canon: dict[str, str] = {}
    for c in sorted(rank_names):
        norm_to_canon.setdefault(_norm_team(c), c)
    clans = sorted(set(side_team["team_clan"].to_list()))
    clan_to_canon: dict[str, str | None] = {}
    for cl in clans:
        n = _norm_team(cl)
        clan_to_canon[cl] = norm_to_canon.get(n) or TEAM_ALIASES.get(n)
    unresolved = [c for c, v in clan_to_canon.items() if v is None]
    if unresolved:
        print(f"[warn] clan names with no canonical/ranked match: {unresolved}")
    else:
        print(f"clan -> canonical: all {len(clans)} clan names resolved")

    canon_df = side_team.with_columns(
        pl.col("team_clan").replace_strict(clan_to_canon, default=None).alias("team_canonical"))
    ct_can = (canon_df.filter(pl.col("side") == "ct")
                      .select(["match_id", "round_num", "team_canonical"])
                      .rename({"team_canonical": "ct_team_canonical"}))
    t_can = (canon_df.filter(pl.col("side") == "t")
                     .select(["match_id", "round_num", "team_canonical"])
                     .rename({"team_canonical": "t_team_canonical"}))

    # --------------------------------------------------------------------------- rounds.csv
    rounds = (first
              .join(rounds_ch, on=["match_id", "round_num"], how="left")
              .join(ct_side, on=["match_id", "round_num"], how="left")
              .join(t_side, on=["match_id", "round_num"], how="left")
              .join(ct_can, on=["match_id", "round_num"], how="left")
              .join(t_can, on=["match_id", "round_num"], how="left")
              .join(meta.select(["match_id", "event", "stage", "series_date", "year",
                                 "winner_team"]), on="match_id", how="left"))
    rounds = rounds.join(plants, on=["match_id", "round_num"], how="left")
    rounds = rounds.with_columns([
        pl.col("winner").alias("round_winner_side"),
        pl.when(pl.col("winner") == "ct").then(pl.col("ct_team_clan"))
          .otherwise(pl.col("t_team_clan")).alias("round_winner_team_clan"),
        pl.col("_site_raw").is_not_null().cast(pl.Int8).alias("bomb_planted"),
        pl.col("_site_raw").replace({"BombsiteA": "A", "BombsiteB": "B"})
          .fill_null("none").alias("bomb_site"),
        pl.col("reason").replace(REASON_CODES).alias("reason"),
        pl.when(pl.col("round_num") <= 12).then(1).otherwise(2).alias("half"),
        pl.col("round_num").is_in([1, 13]).cast(pl.Int8).alias("is_pistol_round"),
        pl.col("winner_team").alias("match_winner"),
    ]).drop(["winner", "winner_team", "_site_raw"])

    order = (["match_id", "event", "stage", "series_date", "year", "round_num", "half",
              "is_pistol_round", "ct_team_clan", "t_team_clan",
              "ct_team_canonical", "t_team_canonical",
              "round_winner_side", "round_winner_team_clan", "ct_won", "match_winner",
              "reason", "bomb_planted", "bomb_site",
              "bomb_plant_tick", "bomb_plant_x", "bomb_plant_y",
              "start", "freeze_end", "end", "official_end", "base_state_tick"]
             + [c for c in BASE_STATE if c in rounds.columns])
    rounds = rounds.select([c for c in order if c in rounds.columns])
    rounds = rounds.sort(["match_id", "round_num"])
    rounds.write_csv(OUT / "rounds.csv")

    # -------------------------------------------------------------------- round_players.csv
    rp = (players
          .join(meta.select(["match_id", "year"]), on="match_id", how="left")
          .join(canon_df.select(["match_id", "round_num", "side", "team_canonical"]),
                on=["match_id", "round_num", "side"], how="left")
          .rename({"name": "player_name", "team_clan_name": "team_clan",
                   "current_equip_value": "equip_value"})
          .select([c for c in ["match_id", "round_num", "year", "steamid", "player_name",
                               "side", "team_clan", "team_canonical", "health", "armor",
                               "has_helmet", "has_defuser", "equip_value", "place",
                               "base_state_tick"] if c in players.columns
                   or c in ("year", "team_canonical", "player_name", "team_clan", "equip_value")])
          .sort(["match_id", "round_num", "side", "player_name"]))
    rp.write_csv(OUT / "round_players.csv")

    # --------------------------------------------------------------------------- matches.csv
    nr = rounds.group_by("match_id").len().rename({"len": "n_rounds"})
    mt = (meta.join(nr, on="match_id", how="left")
              .select(["match_id", "event", "stage", "series_date", "year",
                       "team_a", "team_b", "team_score", "winner_team", "n_rounds",
                       "hltv_match_url"])
              .rename({"team_score": "inferno_score", "winner_team": "match_winner"})
              .sort(["year", "event", "match_id"]))
    mt.write_csv(OUT / "matches.csv")

    # ------------------------------------------------------- reference tables + team crosswalk
    shutil.copy(ROOT / "configs" / "player_stats_sided.csv", OUT / "player_season_stats.csv")
    shutil.copy(ROOT / "configs" / "team_rankings.csv", OUT / "team_rankings.csv")
    shutil.copy(ROOT / "configs" / "player_team_year.csv", OUT / "player_team_year.csv")

    cross = (rounds.select([pl.col("ct_team_clan").alias("team_clan"),
                            pl.col("ct_team_canonical").alias("team_canonical")])
             .vstack(rounds.select([pl.col("t_team_clan").alias("team_clan"),
                                    pl.col("t_team_canonical").alias("team_canonical")]))
             .drop_nulls().unique().sort("team_clan"))
    cross.write_csv(OUT / "team_name_crosswalk.csv")

    print(f"\nwrote -> {OUT}")
    for f in sorted(OUT.glob("*.csv")):
        n = pl.read_csv(f, infer_schema_length=0).height
        print(f"  {f.name:28s} {n:>7,} rows  ({f.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
