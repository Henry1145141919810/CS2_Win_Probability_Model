# CS2 Inferno Round-Level Dataset — v1

**One row per round** for the full training set of the CS2 round win-probability project:
**220 professional `de_inferno` matches, 4,866 rounds, 48,660 player-rounds**, spanning
**18 tournaments** across **2024 (92 matches)** and **2025 (128 matches)**.

Everything here is derived from data already in the project — no re-parsing, no new scraping.
Built by [`src/data/build_wyner_export.py`](../../src/data/build_wyner_export.py) (re-runnable).

Prepared for Prof. Abraham Wyner · UPenn WSABI · generated 2026-09-08.

---

## 1. What this is (and what it is not)

The modelling dataset for this project is **per-second** (476,595 snapshots × 135 features). That
resolution is the right unit for a live win-probability curve but the wrong unit for most
round-level questions. This export **collapses each round to a single row** describing the round's
identity, its participants, its starting conditions, and its outcome — and ships the reference
tables needed to attach **player skill** and **team strength** to any round.

If you want the full per-second data instead, see the companion supplement package
(`../cs2_inferno_raw_supplement_v1/`).

---

## 2. Files

| File | Rows | Grain | What it is |
|---|---:|---|---|
| `matches.csv` | 220 | one row per match | Match identity: tournament, stage, date, the two teams, final Inferno score, match winner, HLTV link |
| `rounds.csv` | 4,866 | one row per (match, round) | **The main table.** Who played which side, who won the round, why it ended, and the full round-start state |
| `round_players.csv` | 48,660 | one row per (match, round, player) | Exactly the ten players on the server that round, their side, team, and starting loadout |
| `player_season_stats.csv` | 476 | one row per (player, year) | HLTV season stats per player, **split by side** (CT/T) |
| `team_rankings.csv` | 141 | one row per (team, year) | HLTV world rank per team per year |
| `team_name_crosswalk.csv` | 57 | one row per in-game clan name | Maps the in-game clan tag to the canonical org name used by the ranking/roster tables |
| `player_team_year.csv` | 507 | one row per (player, year) | Which org a player belonged to in a given year |

### How the tables join

```
matches.csv ──match_id──┐
                        ├──> rounds.csv ──(match_id, round_num)──> round_players.csv
                        │                                              │
                        │                                              ├─(steamid, year)──> player_season_stats.csv
                        │                                              └─(team_canonical, year)──> team_rankings.csv
                        └─ rounds.ct_team_canonical / t_team_canonical ─(team_canonical, year)──> team_rankings.csv
```

**Verified join coverage:** `(steamid, year)` → `player_season_stats` = **377/377 (100%)**;
`(team_canonical, year)` → `team_rankings` = **70/71 (98.6%)** (the one gap is documented in §6).

---

## 3. `matches.csv` — one row per match

| Column | Type | Definition |
|---|---|---|
| `match_id` | string | **Primary key.** The demo identifier, e.g. `g2-vs-ninjas-in-pyjamas-m2-inferno`. Used as the join key in every other table. |
| `event` | string | Tournament name, e.g. `BLAST Premier Fall Final 2024`. 18 distinct events. |
| `stage` | string | Stage within the tournament (`Main/Playoffs`, `Groups`, `Elimination Stage`, …). |
| `series_date` | string | Date of the series (`YYYY/M/D`). Blank for a small number of off-list demos. |
| `year` | int | **Season year** (2024 or 2025). This is the key used to look up player and team stats. |
| `team_a`, `team_b` | string | The two teams in the series, as named in the curated match list. |
| `inferno_score` | string | Final round score **on this Inferno map**, e.g. `11-13`. |
| `match_winner` | string | **Team that won this map.** |
| `n_rounds` | int | Rounds present in this export for the match (13–42; overtime included). |
| `hltv_match_url` | string | Link to the HLTV match page, where available. |

> **Note on `team_a`/`team_b`:** these come from the curated match list and are *not* side-specific.
> For "who played CT in round N", use `rounds.ct_team_clan` — sides swap at halftime.

---

## 4. `rounds.csv` — the main table (one row per round)

### 4.1 Identity

| Column | Type | Definition |
|---|---|---|
| `match_id` | string | Joins to `matches.csv`. |
| `event`, `stage`, `series_date`, `year` | | Denormalised from `matches.csv` for convenience. |
| `round_num` | int | Round number within the match, starting at 1. |
| `half` | int | `1` for rounds 1–12, `2` for rounds 13+ (includes overtime). |
| `is_pistol_round` | 0/1 | `1` for rounds 1 and 13 (the two pistol rounds). |

### 4.2 Who played which side

| Column | Type | Definition |
|---|---|---|
| `ct_team_clan` | string | **In-game clan tag of the team playing CT this round** (e.g. `NIP`). Sides swap at halftime, so this changes between half 1 and half 2. |
| `t_team_clan` | string | In-game clan tag of the team playing T this round. |
| `ct_team_canonical` | string | Canonical org name of the CT team (e.g. `Ninjas in Pyjamas`) — **this is the key that joins `team_rankings.csv`**. |
| `t_team_canonical` | string | Canonical org name of the T team. |

> Verified: in all 220 matches the CT team differs between the first and second half, i.e. the
> halftime swap is correctly represented.

### 4.3 Outcome (the labels)

| Column | Type | Definition |
|---|---|---|
| `ct_won` | 0/1 | **The round label.** `1` if the CT side won this round. This is exactly the target the model predicts. Round-level mean = **0.4871** (Inferno is slightly T-sided). |
| `round_winner_side` | `ct`/`t` | Which **side** won. Perfectly consistent with `ct_won` (0 mismatches in 4,866 rounds). |
| `round_winner_team_clan` | string | Which **team** won the round (the clan name on the winning side). |
| `match_winner` | string | Which team won the **match** overall (constant within a match). |
| `reason` | string | How the round ended. One of: `t_killed` (all T killed → CT win), `ct_killed` (all CT killed → T win), `bomb_exploded`, `bomb_defused`, `time_ran_out`. |

> `reason` uses CS naming: `t_killed` means *the Ts were killed*, so the **CT** side won.

### 4.4 Bomb

| Column | Type | Definition |
|---|---|---|
| `bomb_planted` | 0/1 | Whether the bomb was planted this round (2,917 of 4,866 rounds). |
| `bomb_site` | `A`/`B`/`none` | Which site the bomb was planted at. A = 1,428, B = 1,489, none = 1,949. |
| `bomb_plant_tick` | int | Game tick of the plant (null if never planted). |
| `bomb_plant_x`, `bomb_plant_y` | float | Map coordinates of the plant. |

### 4.5 Round-start ("base") state

All of the following are measured at the **first sampled snapshot of the round**, i.e. immediately
after freeze-time ends and the round goes live — buys are complete, nobody has moved or died.
(`time_elapsed_sec` is typically 0.25–0.5 s; it is < 1 s for 99.94% of rounds.)

| Column | Type | Definition |
|---|---|---|
| `base_state_tick` | int | The game tick this state was read at. Also the tick used for `round_players.csv`. |
| `time_elapsed_sec` | float | Seconds since freeze-end at that snapshot (near 0 by construction). |
| `ct_score`, `t_score` | int | Rounds won by each side **before** this round. |
| `score_diff` | int | `ct_score − t_score`. |
| `ct_players_alive`, `t_players_alive` | int | Players alive — always 5/5 at round start (verified). |
| `ct_equipment_value`, `t_equipment_value` | float | **Summed dollar value of equipment held by the side** (range 1,000–35,100). This is the round's buy. |
| `ct_economy_class`, `t_economy_class` | int | Buy tier from equipment value: **`0` = eco (< $2,000), `1` = force ($2,000–$3,799), `2` = full buy (≥ $3,800)**. |
| `ct_health_total`, `t_health_total` | float | Summed health (500 at round start). |
| `ct_armor_total`, `t_armor_total` | float | Summed armour. |
| `ct_defuse_kits` | int | Number of defuse kits held by the CT side. |
| `ct_smokes`, `ct_flashes`, `ct_fire`, `ct_he` | int | Grenade counts held by CT, by type (`fire` = molotov/incendiary). |
| `t_smokes`, `t_flashes`, `t_fire`, `t_he` | int | Same for T. |
| `ct_util_total`, `t_util_total` | int | Total grenades held per side. |
| `utility_advantage` | int | `ct_util_total − t_util_total`. |
| `ct_awp_alive`, `t_awp_alive` | 0/1 | Whether a living player on that side holds an AWP. |

### 4.6 Tick boundaries

| Column | Definition |
|---|---|
| `start`, `freeze_end`, `end`, `official_end` | Game ticks marking the round's phases. The round is live from `freeze_end` to `end`. |

---

## 5. `round_players.csv` — who was on the server

Exactly **ten rows per round** (verified: 4,866 × 10 = 48,660). State is read at the same
`base_state_tick` as the round's base state, so this is each player's **starting loadout**.

| Column | Type | Definition |
|---|---|---|
| `match_id`, `round_num` | | Join to `rounds.csv`. |
| `year` | int | Season year — **use with `steamid` to join `player_season_stats.csv`**. |
| `steamid` | int64 | **Stable player identifier.** The reliable key (names change; SteamIDs do not). 260 distinct players. |
| `player_name` | string | In-game name at the time of the match. |
| `side` | `ct`/`t` | Which side this player was on **this round**. |
| `team_clan` | string | In-game clan tag of the player's team. |
| `team_canonical` | string | Canonical org name — joins `team_rankings.csv` with `year`. |
| `health` | float | Health at round start (100). |
| `armor` | float | Armour at round start. |
| `has_helmet` | bool | Whether the player bought a helmet. |
| `has_defuser` | bool | Whether the player carries a defuse kit (CT only). |
| `equip_value` | float | **Dollar value of this player's equipment** at round start. Sums to the side's `*_equipment_value`. |
| `place` | string | Named map area. At round start everyone is still in spawn, so this is effectively constant (`CTSpawn`/`TSpawn`) — included for completeness only. |
| `base_state_tick` | int | The tick this was read at. |

---

## 6. Reference tables

### `player_season_stats.csv` — player skill, per season, **split by side**

One row per **(player, year)**; 476 rows, 274 distinct players, years 2024/2025/2026. These are
HLTV's whole-season aggregates for that player, so they are a **prior** on skill, not a per-match
measurement.

| Column | Definition |
|---|---|
| `steamid` | Join key with `year`. |
| `hltv_name` | HLTV handle. |
| `year` | Season (calendar year). |
| `rating_ct`, `rating_t` | **HLTV Rating 3.0**, computed separately for the player's CT and T sides. |
| `firepower_ct`, `firepower_t` | HLTV **Firepower** rating — the raw fragging/damage component of Rating 3.0 — per side. |
| `entrying_ct`, `entrying_t` | Entry-fragging rating per side. |
| `trading_ct`, `trading_t` | Trading rating per side. |
| `opening_ct`, `opening_t` | Opening-duel rating per side. |
| `adr` | Average damage per round (not side-split). |
| `kast` | % of rounds with a Kill, Assist, Survival or Trade (not side-split). |
| `sniping` | AWP/sniping rating. |
| `utility` | Utility-usage rating. |
| `clutching` | Clutch rating. |
| `found` | Whether the player was located in the HLTV table (all `yes` in this file). |

> **Granularity warning:** these are **calendar-year** aggregates. There is no sub-year / per-event
> "season" split. A player has one CT rating and one T rating for all of 2024, one for 2025, etc.

### `team_rankings.csv` — team strength, per season

| Column | Definition |
|---|---|
| `team_canonical` | Org name. Join with `year`. |
| `year` | Season (2024/2025/2026). |
| `hltv_rank` | **HLTV world ranking** for that team that year (1 = best; range 1–35). |
| `weight` | A convenience decay weight, `1 / log2(hltv_rank + 1)`, used in one of our modelling experiments. Ignore it if you want the raw rank. |

> This is **HLTV world rank**, *not* Valve Regional Standings (VRS) and not an Elo. 47 teams per year.
> It is a single yearly snapshot — within-year form changes are not captured.

### `team_name_crosswalk.csv` — clan tag → org name

In-game clan tags carry sponsors and abbreviations (`NIP`, `9z Globant`, `Imperial Sportsbet`) while
the ranking tables use org names (`Ninjas in Pyjamas`, `9z`, `IMPERIAL`). This table is the mapping,
covering all 57 clan names that appear. It was built by name normalisation (55/57 matched outright,
2 by explicit alias) — deliberately **not** by majority-voting player rosters, which misattributes
teams whenever players transfer between orgs.

---

## 7. Known caveats (please read)

1. **Team rank has one gap.** `IMPERIAL` in 2024 (13 rounds, 0.27%) sits outside the ranked top-35
   and therefore has no `hltv_rank`. All other 70 (team, year) combinations join cleanly.
2. **Player stats are yearly, not per-event.** See the granularity warning in §6. Also note these
   are *whole-season* aggregates, so for a match played early in a year the stats partly reflect
   games played *after* that match — a mild look-ahead if used naively as a pre-match prior. Our own
   modelling offers a lagged (previous-season) variant to avoid this.
3. **Nine rounds have minor source inconsistencies** (0.18% total): 1 round is labelled
   `bomb_exploded`/`bomb_defused` with no plant event recorded, and 8 rounds are labelled
   `time_ran_out` despite a plant being recorded. The winner labels (`ct_won`) are unaffected.
4. **`bomb_site` was repaired.** The parser's own round-level site field mislabels every plant as
   site B; the values here are taken from the bomb-event stream instead, which carries the true site.
5. **One match has an inferred year.** `flyquest-vs-virtus-pro-m1-inferno-p2` is an off-list demo
   with no recorded date; its season was inferred as **2024** because all 10 of its players map to
   their 2024 rosters (FlyQuest 5 / Virtus.pro 5) and only 7/10 map to 2025.
6. **`round_players.place` is uninformative** by construction (everyone is in spawn at round start).
7. **This is `de_inferno` only**, and this round-level table covers only the *training* era
   (2024–2025). The out-of-time **2026 hold-out set is in the supplement package** under
   `cs2_inferno_raw_supplement_v1/holdout_2026/` — please treat it as touch-once (it must not
   inform model or feature selection, or the out-of-time results lose their meaning).

---

## 8. Provenance

| Output | Built from |
|---|---|
| Round-start state, `ct_won` | `data/training_dataset.parquet` (the per-second modelling table), first snapshot of each round |
| Round winner, end reason, tick boundaries | parsed `rounds` channel |
| Sides, team clan tags, player rosters, loadouts | parsed `ticks` channel |
| Bomb plant, site, coordinates | parsed `bomb` channel |
| Tournament, stage, date, teams, match winner | `configs/demo_list_final.csv` |
| Season year | `configs/demo_year_map.csv` (one match inferred from rosters) |
| Player season stats | `configs/player_stats_sided.csv` |
| Team rank | `configs/team_rankings.csv` |

Demos are HLTV GOTV recordings, parsed with **awpy 2.0.2**. Player statistics are from HLTV;
team rankings are HLTV world rankings.

## 9. Validation performed

- `ct_won` vs `round_winner_side`: **0 mismatches** in 4,866 rounds.
- Exactly **10 players in every round** (observer/coach entities excluded).
- **5 v 5 alive** at the start of every round.
- Halftime side swap present in **all 220 matches**.
- **Zero nulls** in every identity, side, outcome and economy column.
- Join coverage: player stats **100%**, team rankings **98.6%** (§7.1).
