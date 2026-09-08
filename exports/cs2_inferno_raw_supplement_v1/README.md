# CS2 Inferno Raw Supplement — v1

**Full-resolution companion** to the [CS2 Inferno Round-Level Dataset](../cs2_inferno_round_level_v1/README.md).

Everything behind the round-level export at native resolution, split into the two eras this
project keeps strictly apart:

| Era | Matches | What it is |
|---|---:|---|
| `training_2024_2025/` | 220 | The `de_inferno` matches every model is fitted on |
| `holdout_2026/` | 27 | The **out-of-time test set** — held out, evaluated once |

**~339 MB.** Built by [`src/data/build_wyner_supplement.py`](../../src/data/build_wyner_supplement.py)
(re-runnable). A machine-readable inventory — every file, size, channel schema and the full match
list per era — is in **`MANIFEST.json`**.

Prepared for Prof. Abraham Wyner · UPenn WSABI · generated 2026-09-08.

> ⚠️ **Please treat `holdout_2026/` as touch-once.** Every out-of-time number in the paper depends
> on that set never having informed model or feature selection. It is included here for inspection
> and for your own analysis — just not for tuning.

---

## 1. File tree

```
cs2_inferno_raw_supplement_v1/
├── README.md                 <- this file
├── MANIFEST.json             <- machine-readable inventory + full match lists
│
├── training_2024_2025/                              220 matches, 2024-2025
│   ├── modelling/
│   │   ├── training_dataset_per_second.parquet      476,595 x 135   (84 MB)  <- the table models train on
│   │   └── trajectory_dataset_per_player.parquet    474,577 x 117   (68 MB)  <- per-player trajectories (GAT input)
│   └── parsed/                                      one parquet per match
│       ├── ticks/     220 files  110.4 MB           per-player state, ~1 Hz
│       ├── rounds/    220 files    0.8 MB           round boundaries + outcome
│       ├── kills/     220 files   11.0 MB           every kill event
│       ├── bomb/      220 files    1.2 MB           plant / defuse / explode / pickup / drop
│       ├── smokes/    220 files    4.3 MB           smoke grenades
│       ├── infernos/  220 files    4.2 MB           molotov / incendiary fires
│       └── defuse/    214 files    0.5 MB           defuse attempts (see §4.2)
│
├── holdout_2026/                                    27 matches, 2026 (OUT-OF-TIME)
│   ├── modelling/
│   │   ├── test_dataset_2026_per_second.parquet               55,271 x 115   <- PRIMARY holdout
│   │   ├── test_dataset_2026_with_defuse.parquet              55,271 x 135
│   │   ├── test_dataset_2026_firepower_lagged2025.parquet     55,271 x 115
│   │   └── test_dataset_2026_firepower_sameyear.parquet       55,271 x 115
│   └── parsed/        same 7 channels, 27 matches (~15 MB)
│
└── reference/         17 shared lookup tables (player stats, team ranks, match lists, zone map)
```

**Join key everywhere:** `match_id` (the parquet filename) + `round_num`, and `tick` for
time-resolved channels. `match_id` is identical to the round-level export's `match_id`.

---

## 2. When to use this instead of the round-level export

| Question | Use |
|---|---|
| "Who won this round, with what economy, which teams, which players?" | round-level export |
| "How did win probability / map control evolve **within** a round?" | `modelling/*_per_second.parquet` |
| "Where exactly was every player at tick *t*?" | `parsed/ticks/` |
| "Who killed whom, with what weapon, from where?" | `parsed/kills/` |
| "Where and when was every smoke / molotov thrown?" | `parsed/smokes/`, `parsed/infernos/` |
| "Does the model transfer to a later season?" | `holdout_2026/` |

The round-level export is a **summary**; this is the **source**.

---

## 3. The modelling tables

### 3.1 `training_dataset_per_second.parquet` — 476,595 × 135

One row per **game-state snapshot per second**: for each round the state is sampled roughly once
per second from freeze-end until the round ends (first sample lands <1 s after freeze-end for
99.94% of rounds). This is exactly the table the classical models, TCN and Transformer train on.

**Label:** `ct_won` (1 if the CT side won that round), repeated on every snapshot of the round.
Snapshot-weighted mean = 0.4454; round-weighted mean = 0.4871. (Rounds the CTs lose tend to run
longer, so the two differ — use the round-level export for one row per round.)

#### Column groups (all 135)

**Identifiers & label (4)** — `match_id`, `round_num`, `tick`, `ct_won`

**Economy & combat (10)** — `time_elapsed_sec`, `ct_/t_equipment_value`, `ct_/t_economy_class`
(0 = eco < $2,000, 1 = force $2,000–3,799, 2 = full ≥ $3,800), `ct_/t_armor_total`,
`ct_/t_score`, `score_diff`

**Map control — Voronoi / line-of-sight / territory (27)** — `ct_/t_voronoi_control_pct`,
`control_deficit`, per-zone control (`ct_a_site_control`, `ct_b_site_control`, `ct_banana_control`,
`ct_mid_control`, `ct_ct_spawn_control`), `control_trend`, `control_volatility`,
`ct_/t_los_control`, `contested_pct`, `grey_pct`, `ct_los_deficit`, `ct_/t_terr_control`,
`terr_contested_pct`, `terr_grey_pct`, `ct_terr_deficit`, per-zone territory deficits,
`ct_bomb_local_control`, `ct_closer_to_bomb`

> Three different control models are included: **Voronoi** (nearest-player area ownership),
> **LOS/grey** (line-of-sight + FOV + smoke, instantaneous), and **territory** (grey model with a
> 15-second memory). They answer different questions; see the paper for the ablation.

**Tactical & bomb geometry (50)** — alive counts, `ct_/t_health_total`, `ct_defuse_kits`,
positional entropy, per-zone player counts, AWP presence/zone, grenade counts by type,
`utility_advantage`, bomb state (`bomb_planted`, `bomb_site`, `bomb_plant_x/y`, `bomb_state`,
`bomb_dropped`, `bomb_carrier_zone`), and the **defuse-race** geometry (`defuse_time_margin`,
`defuse_margin_kit`, `n_ct_can_defuse`, `best_defuser_has_kit`, `defuse_contest_margin`,
`min_ct_dist_to_bomb`, `min_ct_path_to_bomb`, …)

> The defuse-race features are **counterfactual**: they ask whether a CT *could* reach the bomb and
> finish a defuse in time, from geometry — not whether one is happening.

**Defuse-progress — live defuse (4)** — `defuse_in_progress`, `defuse_elapsed_sec`,
`defuse_progress_frac` (kit-aware fraction of required time completed, 0–1), `defuse_beats_fuse`

> These measure an **actually running** defuse, unlike the counterfactual block above. They fire on
> only ~1% of snapshots.

**Firepower — skill prior (36)** — summed and mean-normalised HLTV stats over the *alive* players of
each side, recomputed every snapshot: `ct_/t_rating_sum`, `adr_sum`, `kast_mean`,
`hltv_firepower_sum`, `entry_sum`, `trading_sum`, `opening_sum`, `clutch_score`,
`awp_sniping_skill`, `weighted_utility`, plus the mean versions (`rating_mean`, `adr_mean`,
`fp_mean`, `entry_mean`, `trading_mean`, `opening_mean`, `utility_mean`) and `n_with_stats`.

> **`*_sum` vs `*_mean`:** the sums correlate ~0.99 with how many players are alive, so they partly
> restate the man-advantage. The `*_mean` versions divide by `n_with_stats` (alive players found in
> the HLTV table, **not** the headcount) to remove that confound. `*_rating_mean` is mean HLTV
> Rating 3.0; `*_fp_mean` is mean HLTV *Firepower*, the fragging sub-component of that rating.

**Interactions (4)** — `ctrl_x_eveneco`, `terr_x_eveneco`, `ctrl_x_equalalive`, `terr_x_equalalive`

### 3.2 `trajectory_dataset_per_player.parquet` — 474,577 × 117

A **different shape** of the same matches: per-player trajectories rather than one aggregated
team-state row. This is the input to the **graph attention network (GAT)**, which treats the ten
players as nodes rather than collapsing them into side-level aggregates. Included so the GAT
branch of the project is reproducible.

### 3.3 The four `holdout_2026/` tables — read this before using them

All four cover the **same 55,271 snapshots / 27 matches**; they differ **only** in how the
firepower (player-skill) columns were built. This matters because one of the paper's findings is
that the skill prior has an *inference-time data dependency*.

| File | Cols | Firepower construction | Use |
|---|---:|---|---|
| `test_dataset_2026_per_second.parquet` | 115 | 2026 stats **as available at the time** — coverage 91.6%, i.e. new players missing | **The primary holdout.** The paper's headline out-of-time numbers use this. |
| `test_dataset_2026_with_defuse.parquet` | 135 | fully populated (99.99%) | Adds the 4 defuse-progress columns. Used for the live-defuse calibration analysis. |
| `test_dataset_2026_firepower_lagged2025.parquet` | 115 | previous season (2025) stats | **Leak-free** variant: skill known strictly before the match. |
| `test_dataset_2026_firepower_sameyear.parquet` | 115 | full 2026 season stats | **Leaky best case**: uses stats partly generated after the match. An upper bound. |

> **Distribution shift, by design:** 2026 Inferno is more CT-sided than 2024–25
> (`ct_won` 0.5121 vs 0.4454). That shift is part of what makes this a real out-of-time test.

---

## 4. `parsed/` — the demo channels

One parquet per match, parsed from HLTV GOTV demos with **awpy 2.0.2**. All carry `round_num`;
time-resolved channels carry `tick`. Schemas are identical across both eras.

### 4.1 `ticks/` — per-player state (~1 Hz)

The richest channel: `tick`, `round_num`, `steamid`, `name`, `side` (`ct`/`t`), `team_clan_name`,
`X`, `Y`, `Z`, `velocity_X/Y/Z`, `pitch`, `yaw`, `place` (named map area), `health`, `armor`,
`has_helmet`, `has_defuser`, `is_defusing`, `flash_duration`, `current_equip_value`,
`inventory` (list of weapon/item names).

> `side` and `team_clan_name` are **null** for observer/coach entities present in some demos —
> filter them out to get exactly the ten players on the server.

### 4.2 `rounds/` — one row per round

`round_num`, `start`, `freeze_end`, `end`, `official_end` (ticks), `winner` (`ct`/`t`), `reason`,
`bomb_plant`, `bomb_site`.

> **Two known defects in this channel** (both repaired in the round-level export): `bomb_site`
> labels *every* plant `bombsite_b` — use the `bomb/` channel instead; and `reason` leaves ~3.5% of
> rounds as raw engine enum codes (`1`=bomb_exploded, `7`=bomb_defused, `8`=t_killed,
> `9`=ct_killed, `12`=time_ran_out).

### 4.3 `kills/` — every kill

~90 columns: full state of **attacker**, **victim** and **assister** at the moment of the kill
(position, health, armour, equipment, inventory, place, view angles), plus `weapon`, `distance`,
`headshot`, `hitgroup`, `penetrated`, `noscope`, `thrusmoke`, `attackerblind`, `attackerinair`,
`dmg_health`, `dmg_armor`, `is_bomb_planted`, `is_freeze_period`, `game_time`, `tick`, `round_num`.

### 4.4 `bomb/` — bomb events

`tick`, `round_num`, `event` (`plant`, `defuse`, `detonate`, `pickup`, `drop`), `X`, `Y`, `Z`,
`steamid`, `name`, `bombsite` (`BombsiteA`/`BombsiteB`, non-null on plants).
**Authoritative source for plant site.**

### 4.5 `smokes/` and `infernos/` — grenade areas of effect

`entity_id`, `start_tick`, `end_tick`, the resulting `X`, `Y`, `Z`, and the full state of the
thrower at release (`thrower_*`). `infernos` covers molotovs and incendiaries.

### 4.6 `defuse/` — defuse attempts

`round_num`, `steamid`, `start_tick`, `end_tick`, `n_ticks`, `had_kit`, `completed`. One row per
attempt, **including interrupted attempts** (~40% of all attempts) — which is what makes this a
live-state signal rather than a restatement of the outcome.

---

## 5. Caveats

1. **`defuse/` is short a few files** (training 214/220, holdout 26/27). Those matches contain
   **zero** defuse attempts — verified against the modelling tables (0 defusing snapshots each).
   Treat a missing file as "no attempts", not missing data.
2. **`ticks` is ~1 Hz, not every server tick.** Sub-second detail (spray patterns, exact peek
   timing) is not recoverable from `ticks/`. Kill and bomb events *are* exact — those channels
   record the true event tick.
3. **`rounds/` defects** — see §4.2. Prefer the round-level export, or `bomb/`, for plant site and
   round-end reason.
4. **Firepower stats are yearly aggregates.** Player skill columns come from whole-season HLTV
   statistics, so for a match early in a season they partly reflect games played *after* it. The
   `firepower_lagged2025` holdout table is the leak-free variant.
5. **Team ranks are yearly HLTV world ranks** (1–35), not VRS and not Elo. One team-year
   (`IMPERIAL` 2024) falls outside the ranked top-35 and has no rank.
6. **Coordinates are game units** in Valve's `de_inferno` frame.
   `reference/inferno_zone_map.parquet` maps nav-mesh `area_idx` → `place` → coarse `zone`.
7. **`holdout_2026/` is touch-once** — see the note at the top.

---

## 6. `reference/` — shared lookup tables

All 17 project config files, copied verbatim. The ones that matter:

| File | Grain | Contents |
|---|---|---|
| `player_stats_sided.csv` | (player, year) | HLTV season stats **split by CT/T side** — ratings, firepower, entry, trading, opening, plus adr/kast/sniping/utility/clutching. 476 rows, 2024–2026. |
| `player_stats_raw.csv` | (player, year) | Non-sided version (rating, adr, kast, clutching). |
| `team_rankings.csv` | (team, year) | HLTV **world rank** (1–35) + a `1/log2(rank+1)` weight. 47 teams × 3 years. |
| `player_team_year.csv` | (player, year) | Which org a player belonged to that year. |
| `player_match_year.csv` | (player, match) | Which players appear in which demo. |
| `demo_list_final.csv` | match | Event, stage, date, teams, map score, **match winner**, HLTV URL. |
| `demo_year_map.csv` | match | Demo → season year. |
| `inferno_matches_liquipedia.csv` | match | The Liquipedia-sourced candidate list the corpus was drawn from. |
| `target_events.csv` | event | The 21 tournaments targeted, with tier and dates. |
| `inferno_zone_map.parquet` | nav area | `area_idx` → `place` → `zone`. |

Join recipes: `ticks.steamid` + match year → `player_stats_sided` on `(steamid, year)`;
`team_clan_name` → `team_name_crosswalk.csv` (in the round-level export) → `team_rankings` on
`(team_canonical, year)`.

---

## 7. Provenance

HLTV GOTV `.dem` files for 247 professional `de_inferno` maps (220 training across 18 tier-1 events
in 2024–2025; 27 out-of-time in 2026), parsed with **awpy 2.0.2**; features engineered by this
project's pipeline (`src/features/`) and assembled by `src/data/`. Player statistics and team
rankings are from HLTV. Map zone definitions come from the awpy-bundled `de_inferno` nav mesh.
