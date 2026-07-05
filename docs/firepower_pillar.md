# Firepower (Pillar 3) — data, design, and results

Firepower is the fourth feature pillar planned from day one (see `README.md`'s pillar table)
but never implemented in the first two weeks — economy, map control, and tactical readiness
all draw on data already inside the parsed demos, but **player skill does not**: GOTV demos
record what happened in the game, not how good a player has historically been. Building this
pillar meant a new, independent data-acquisition step before any feature code could be written.

This doc records that process end to end: where the data came from, the data-quality issues
found along the way, the feature design, and the evaluation results against the project's
standard Experimental Matrix (Models A–E).

---

## 1. Hypothesis

A team's chance of winning a round should depend not just on economy/equipment and position,
but on **how skilled the players currently alive are**. Two teams with identical economy and
map control are not equally likely to win if one side's alive players are significantly
better individual performers. HLTV already publishes exactly this kind of per-player,
per-period skill data (Rating, ADR, KAST, and a set of 0–100 "playstyle" composite scores),
so the pillar is a **join problem**, not a new-measurement problem: attach the right HLTV
numbers to the right alive player at the right point in time.

---

## 2. Data acquisition

### 2.1 Building the target list — who do we even need?

The 220 demos that make it into `training_dataset.parquet` (validated via `match_id` overlap
with `configs/demo_list_final.csv` — confirmed identical sets, 0 difference) contain a fixed,
enumerable cast: every `steamid` that ever appears in their `ticks` tables. Restricting to
exactly those 220 demos (not the 266 raw parsed demos sitting in `data/parquet/ticks/`, which
include ones later dropped by validation) gives **260 unique players** — `configs/player_roster.csv`.

A data-quality bug surfaced while building this list: `team_clan_name` in the ticks data is
occasionally **mis-attributed to the opposing team** for a handful of ticks within a demo —
almost certainly related to the halftime side-swap (`validate_parquet.py`'s own docstring flags
this as a known hazard). Naive `unique(steamid, name, team_clan_name)` produced obviously wrong
rows (e.g. ZywOo, a long-time Team Vitality player, showing up paired with "3DMAX" in one demo).
Fix: group by `steamid` and take the **mode** (most frequent value) of `team_clan_name` and
`name` instead of a raw unique — the same defensive pattern `validate_parquet.py` already uses
for round-winner reconstruction (`ct_clan_by_round`'s `.mode()` call), which is why the round
labels themselves were never at risk. `name_variants` (every distinct spelling ever seen for a
`steamid`) is kept alongside the mode name, because in-game names have cosmetic noise (leading
spaces, `$`/`S` swaps, case) that must be tried when matching against HLTV's displayed names.

### 2.2 Why "year" matters, and how it's resolved

A player's skill drifts over a season — using a player's *current* (2026) stats to describe
their play in a *2024* match is a temporal mismatch, the same category of error the project
already guards against with the 2026 out-of-time holdout. So every HLTV lookup is keyed on
**`(steamid, year)`**, not just `steamid`.

`year` is a property of the **match**, not the player (the same player can appear in both
2024 and 2025 demos). `configs/demo_year_map.csv` resolves it per `demo_id`, preferring the
parsed `series_date` (format `YYYY/M/D`, populated for 205/220 demos) and falling back to a
regex for a `2024`/`2025` token in the `event` string (covers all but one demo — an
off-list/unlisted qualifier match with no date metadata in either field; defaulted to 2024 and
flagged for manual confirmation). `configs/player_match_year.csv` is the long-format
`(steamid, demo_id, year)` table used to scope exactly which `(player, year)` pairs need
scraping — **388 unique pairs** in total, well under naively scraping `260 × 2`.

One sanity check worth recording: two players (`kyxsan`, `TeSeS`) initially looked like a bug
(only a 2025 entry, no 2024) — traced to all 10 of their matches in-sample being **Team
Falcons** matches, and Falcons' roster of that caliber only appears in the 2025 portion of the
220-demo sample. Confirmed via `demo_id` listing, not a scraping gap.

### 2.3 Scraping (HLTV, via an LLM browser agent)

HLTV per-player history was pulled via a browser-automation agent (Claude for Chrome), reading
each player's stats page filtered to the relevant year's date range. Two practical lessons from
this part of the process:

- **Batch size matters more than total volume.** A first attempt sending all 240 remaining
  players in one prompt hit the agent's spend limit after 742 steps with **zero output saved**
  — it was set up to compile one final answer at the end rather than emit partial results, so
  the entire run was lost. Fixed by (a) capping batches at 40 players and (b) explicitly
  instructing the agent to flush a CSV snapshot every ~8 players, so a mid-run cutoff never
  loses completed work again.
- **Identity, not display name, is the join key.** HLTV's displayed name can differ trivially
  from the demo's in-game name (whitespace, leetspeak substitutions); every batch result was
  joined back to `steamid` via the `name_variants` list, and every batch round-tripped through
  this join with **zero unmatched rows** across all 260 players.

Final coverage: `configs/player_stats_raw.csv` — 377 rows, 260 unique `steamid`s, **zero**
missing values across `rating`, `adr`, `kast`, `clutching`. One row (`horvy`, 2024) carries an
explicit small-sample caveat — only 3 maps in that year, Rating 0.57 (far below the ~0.85–1.3
typical pro range) — recorded as-is rather than discarded, but worth excluding or down-weighting
if it ever shows up as an influential outlier in a fitted model.

---

## 3. Feature design (`src/features/firepower.py`)

Mirrors `economy.py`'s shape: a `firepower_features(snap, match_id) -> dict` function plus a
`FIREPOWER_COLS` list, called once per snapshot from `assemble.py`.

```
ct_alive, t_alive = snap filtered to side==ct / side==t, health > 0
for each side:
    look up each alive player's (steamid, year) row in player_stats_raw.csv
    ct_firepower_rating = SUM of alive players' Rating      (mirrors economy.py's equipment-value
    ct_firepower_adr    = SUM of alive players' ADR          sum: more skilled players alive =
                                                               more aggregate threat, not just a
                                                               higher average)
    ct_firepower_kast   = MEAN of alive players' KAST         (a per-player rate; summing it across
                                                               players has no clean interpretation)
firepower_rating_diff = ct_firepower_rating - t_firepower_rating
```

**Clutch feature.** When a side is down to exactly **one** alive player (1vN, including 1v1 —
HLTV's own Clutching composite blends all 1vX categories, so 1v1 is included rather than carved
out as a separate "duel" category), that lone player's HLTV Clutching score (0–100) is exposed
as `ct_clutch_score` / `t_clutch_score`; **NaN** otherwise — the same "neutral default" idiom
`bomb.py` uses for pre-plant fields. In a true 1v1 **both** sides' scores are populated
simultaneously (no extra branch needed — each side's check is independent), letting the model
read off both players' clutch skill rather than a precomputed, sign-ambiguous "advantage" term.

No separate `is_clutch` flag was added: a real Clutching score is never 0 in the scraped data
(observed range ≈ 21–92), and `train_pipeline.py` already runs every feature column through
`np.nan_to_num` before fitting (NaN → 0), so "`clutch_score > 0`" is already a perfect proxy for
"this side is in a clutch" — an extra column would be redundant.

`year_for_match(match_id)` resolves the match's year once per demo (not per tick) from
`demo_year_map.csv`, with the single unresolved demo defaulting to 2024 (see §2.2).

---

## 4. Pipeline integration

`assemble.py` changes were limited to three lines: import `firepower_features`, call it once
per snapshot (`fp = firepower_features(snap, match_id)`), and merge it into the row dict
alongside the other pillars' outputs. `train_pipeline.py` gained one import (`FIREPOWER_COLS`)
and new `FEATURE_SETS` entries. A `--limit 3` smoke test (5,002 snapshots) was run before the
full 220-demo regeneration to catch bugs cheaply; it confirmed clutch activation rates (337
CT-clutch / 293 T-clutch snapshots out of 5,002) and correct NaN gating before committing to the
~20-minute full rebuild. The full rebuild reproduced the exact same row/column/label-balance
shape as before (476,595 rows, `ct_won` base rate 0.445) plus the new firepower columns — no
regressions in the existing 83 columns.

---

## 5. Results — Experimental Matrix (Models A–E)

5-fold GroupKFold by match, B=500 match-level block bootstrap, DeLong vs the economy baseline.
Model letters follow the project's Experimental Matrix spec exactly (A = economy baseline,
B/C/D = economy + exactly one pillar, E = all four pillars combined):

| Model | Feature set | AUC (logreg) | Δ vs A (95% CI) | AUC (xgb) | Δ vs A (95% CI) |
|---|---|---|---|---|---|
| **A** | Economy only | 0.8465 | — | 0.8443 | — |
| **B** | Economy + Map control | 0.8485 | +0.0020 (+0.0011, +0.0029) ✅ | 0.8470 | +0.0027 (+0.0019, +0.0036) ✅ |
| **C** | Economy + Firepower | 0.8484 | +0.0019 (+0.0004, +0.0033) ✅ | 0.8458 | +0.0015 (−0.0010, +0.0038) ❌ |
| **D** | Economy + Tactical | 0.8487 | +0.0021 (+0.0009, +0.0036) ✅ | 0.8477 | +0.0034 (+0.0023, +0.0045) ✅ |
| **E** | All four pillars | **0.8500** | **+0.0035** (+0.0016, +0.0054) ✅ | **0.8488** | **+0.0045** (+0.0021, +0.0067) ✅ |

✅ = bootstrap CI excludes 0 (statistically significant). All DeLong p ≈ 0 for the logreg rows
(DeLong understates variance vs. the match-level bootstrap, hence checking both).

**Contested-AUC** (58,368 snapshots with equal players alive and even economy — the subset
where the economy baseline collapses toward a coin flip): firepower alone (Model C) reaches
**0.5934 / 0.5930** (logreg/xgb) — *higher* than map control alone (B: 0.5936/0.5886) and well
above tactical alone (D: 0.5873/0.5879). In the rounds where economy already fails, individual
player skill is at least as informative as positioning — arguably the more publishable framing
of this pillar's contribution than the modest aggregate AUC lift.

---

## 6. Honest takeaways

- **C (firepower alone) is the least robust of the three single-pillar additions** — its
  XGBoost bootstrap CI is the only one of the four model comparisons that includes 0. Consistent
  with the project's standing rule of never trusting a single architecture's verdict on a small
  lift.
- **Model E (all four) is the new best result on both architectures**, ahead of the
  three-pillar `E` set (map control + tactical, no firepower) used before this pillar existed —
  the four pillars' signals are at least partially complementary, not redundant.
- **Firepower's real strength shows up conditionally** (contested rounds), not in the
  unconditional aggregate AUC — same shape of finding as map control's "matters most where
  economy fails" result, just for a different pillar.
- **Open item:** Pillar 3 currently uses season-level Rating/ADR/KAST, not the within-match
  trajectory (a player who's on a hot streak *this map*). The Clutching feature is the only
  "in-the-moment, situation-gated" signal implemented so far; extending that idea (e.g. recent
  multi-kill rate, opening-duel win rate keyed by the same 1vN/entry-situation gating logic)
  is the natural next step if more lift is wanted from this pillar.

---

## 7. Reference: data files this pillar depends on

| File | Grain | Role |
|---|---|---|
| `configs/player_roster.csv` | 1 row / steamid (260) | identity + name variants + team (mode) + years active |
| `configs/demo_year_map.csv` | 1 row / demo (220) | `demo_id -> year`, used by `firepower.py` at match granularity |
| `configs/player_match_year.csv` | 1 row / (steamid, demo) (2,202) | long-format scrape-scoping table, not read at train time |
| `configs/player_stats_raw.csv` | 1 row / (steamid, year) (377) | v1 lookup table: rating/adr/kast/clutching |
| `configs/player_stats_sided.csv` | 1 row / (steamid, year) (377) | **v2** lookup: side-split Rating/Firepower/Entry/Trading/Opening + blended ADR/KAST/Sniping/Utility |

---

## 8. v2 (July 2026) — side-aware + situational gating (commit fc4f719)
The sections above document **v1** (kept for the process). **v2** is a redesign of `firepower.py`
(`FIREPOWER_COLS` 9 → **20**):
- **Side-aware:** each alive player is scored for the side they are *currently* playing (CT-side vs
  T-side Rating/Firepower/Entrying/Trading/Opening from `player_stats_sided.csv`); ADR/KAST/Sniping/
  Utility stay blended (HLTV has no per-side split).
- **Per-player conditional gates:** lone survivor → Clutching (suppress Entry/Trading); teammates alive
  → Entry/Trading (suppress Clutch); Opening only when both sides are 5-alive (no kills yet).
- **Sniping** = the AWP holder's role score (NaN if no AWP held). **Utility** = HLTV utility skill ×
  current grenade dollar value carried, summed per side.

**Benchmark (same 220 demos / 5-fold OOF / B=500; full numbers in `docs/results_checkpoint.md` and
`docs/methodology.md` "Pillar 3 v1→v2"):**
- ⬆ **Helps the linear/headline model** — **logreg contested-AUC 0.593 → 0.603** and **logreg EFB2
  0.8515 → 0.8519** (both the study's best).
- ⬇ **Hurts the tree models** — EF < E on xgb/lgbm/catboost (catboost −0.0026, significant): the 20
  sparse, NaN-gated features overfit GBMs.
- ⚠️ **Count confound persists** — v2 kept **sums**, so `ct_rating_sum − t_rating_sum` is still
  **0.987**-correlated with the player-count advantage. Permutation importance: sum features still lead
  (`ct_rating_sum` #9, `t_rating_sum` #12), with `t_trading_sum` #10 and `t_awp_sniping_skill` #19 as
  the strongest new side-aware signals.

**Open — firepower v3:** use **average** rating per alive player (per-capita) to decouple skill from
count, and **prune** the sparse gated features (keep the ones importance likes). Use v2 for the
logistic/headline model; v1 or a pruned v2 for the GBMs.
