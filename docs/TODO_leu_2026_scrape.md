# TASK (Leu): 2026 HLTV stats scrape — unblock the firepower holdout re-run

**Owner:** Leu · **Created:** 2026-07-03 by Henry/Claude · **Blocks:** the 2026 out-of-time re-evaluation
**Everything else is already done.** This scrape is the only remaining human step.

---

## 1. Why (read this first — it's the whole point)

We ran the **2026 out-of-time holdout** (touch-once): trained on all 220 demos (2024-25), evaluated
once on **27 fresh 2026 Inferno matches**. Result:

- ✅ **The core model generalises.** Without firepower, out-of-time ≈ in-time, some even better:
  `lgbm EB2 0.8493 → 0.8501`, `xgb E 0.8476 → 0.8476`. Contested-AUC *improves* (0.590 → 0.64-0.65).
- ❌ **With firepower it COLLAPSES.** `logreg EFB2 0.8519 (best in-sample) → 0.8236` (−0.028);
  ECE 0.016 → 0.071; calibration intercept −0.36. **The best in-sample model became the worst out-of-time.**

**Diagnosed cause = a data-coverage gap, not a signal failure:**
- `player_stats_sided.csv` has **only 2024 & 2025 rows — zero for 2026**.
- The 2026 demos weren't in `demo_year_map.csv`, so `year_for_match()` silently fell back to
  `DEFAULT_YEAR=2024` and looked up **2024 stats for 2026 matches**.
- **~30% of 2026 players had no usable stats** → their firepower contributed **0**. At 5v5, mean
  `ct_rating_sum` is **5.28 in training but only 3.66 on 2026**; 11.8% of 2026 5v5 snapshots have
  `rating_sum < 3` (vs **0.0%** in training).
- The model reads that corrupted firepower as *"few/weak players alive"* → systematic mispredictions.

**Already fixed by Henry (you don't need to touch these):**
- `configs/demo_year_map.csv` — the 32 2026 demos are now mapped with `year=2026` (220 → 252 rows).
- `src/features/assemble.py` — gained `--parquet-root` / `--no-exclude` so the 2026 test set can be
  **re-assembled** from the isolated `data/holdout2026/parquet` tree (the corrupted firepower is
  *baked into* the current test parquet, so a rebuild is mandatory after your scrape).
- `configs/player_roster_2026.csv` — **your scrape list**, generated for you (see below).

---

## 2. ⚠️ IMPORTANT: same-year stats are LEAKY — so please scrape TWO things

The pillar currently uses **same-year** stats (a 2024 match → 2024 stats). But a player's 2024 HLTV
Rating is *computed from* their 2024 matches — **including the very matches we train/test on**. That's
a mild but real **leakage** (~1-2% per match), and for the *holdout* it would be worse: 2026 stats
computed over Jan–Jun 2026 include the exact test matches. A reviewer will catch this.

So please scrape **both** variants so we can run the clean experiment:

| variant | what it is | why |
|---|---|---|
| **A. same-year (2026)** | 2026 stats for the 2026 matches | consistent with how the model was **trained**; the direct fix. Leaky — we'll disclose it. |
| **B. lagged prior (2025)** | 2025 stats used as the prior for 2026 matches | **leak-free** and a realistic deployment scenario ("at match time you only know last season"). This is the scientifically clean test. |

**Good news: variant B is cheap** — **65 of the 82** players already have 2025 rows. You only need to
scrape 2025 for the **17 players who are missing it** (list in §3, Task B).

---

## 3. THE TASK

### Task A (required) — scrape **2026** stats for all **82** players
- **Input:** `configs/player_roster_2026.csv`
  → columns: `steamid, name, team, name_variants, n_demos, has_any_year_stats`
  → 82 players across the 32 2026 demos; `name_variants` is `|`-separated (use it to match HLTV's
    displayed name — in-game names have cosmetic noise).
- **HLTV date filter:** **2026-01-01 → 2026-06-30** (the matches span **2026/1/30 – 2026/6/7**:
  IEM Kraków 2026, IEM Cologne Major 2026, IEM Atlanta 2026, BLAST Rivals 2026 Season 1).
- **Output:** append **82 rows with `year=2026`** to `configs/player_stats_sided.csv`.

### Task B (cheap, high value) — scrape **2025** stats for the **17** players missing them
**65 of the 82 already have 2025 rows.** These **17** do not (14 are brand-new; 3 have 2024 only):

`hypex, arT, zweih, HUASOPEEK, max, Rainwaker, Gizmy, Bymas, luchov, cobrazera, meyern, afro,
soulfly, ryu, s1zzi, piriajr, v$m`

- **Input:** `configs/player_roster_2026_missing2025.csv` (steamid, name, team, name_variants,
  n_demos, `has_2024`) — generated for you.
- **HLTV date filter:** **2025-01-01 → 2025-12-31**
- **Output:** append **17 rows with `year=2025`** to `configs/player_stats_sided.csv`.
- (If a rookie genuinely has no 2025 data, set `found=no` and leave stats null — do **not** invent values.)

---

## 4. Output schema — `configs/player_stats_sided.csv` (match EXACTLY, 19 columns)

```
hltv_name, year, rating_ct, rating_t, adr, kast,
firepower_ct, firepower_t, entrying_ct, entrying_t,
trading_ct, trading_t, opening_ct, opening_t,
sniping, utility, found, steamid, clutching
```

**Real example row (from your existing file):**
```
frozen,2025,1.19,1.15,81.0,75.0,79,72,45,34,61,73,55,40,0,63,yes,76561198068422762,75
```

**Where each field comes from on HLTV:**
| field | source | notes |
|---|---|---|
| `rating_ct` / `rating_t` | player stats page, **CT-side / T-side filter** | float, ~0.8–1.4 |
| `firepower_ct/_t`, `entrying_ct/_t`, `trading_ct/_t`, `opening_ct/_t` | **per-side** playstyle scores | int 0–100 |
| `adr`, `kast` | **blended** (both sides) | HLTV has no per-side split |
| `sniping`, `utility`, `clutching` | **blended** playstyle scores | int 0–100 |
| `steamid` | from `player_roster_2026.csv` | **THE JOIN KEY** — must be exact |
| `hltv_name` | HLTV's displayed name | for humans; not the join key |
| `found` | `yes` / `no` | `no` if the player has no data for that period |

---

## 5. Practical scraping guidance (your own v1 lessons — they cost you a run last time)

- **Batch ≤ 40 players** per agent run.
- **Flush a CSV snapshot every ~8 players** — a mid-run cutoff must never lose completed work.
  (Your first v1 attempt hit the spend limit after 742 steps with **zero output saved**.)
- **Join back on `steamid`, never on the display name.** Use `name_variants` to find the right HLTV
  player, then write the `steamid` from the roster file.
- Don't discard small-sample players — record them as-is and set `found=yes`, but note it. (v1
  recorded `horvy` with only 3 maps / Rating 0.57 — kept, but flagged.)

---

## 6. Validate before you commit (please run these)

```python
import polars as pl
s = pl.read_csv('configs/player_stats_sided.csv')
r = pl.read_csv('configs/player_roster_2026.csv')
need = set(r['steamid'].to_list())

got26 = set(s.filter(pl.col('year') == 2026)['steamid'].to_list())
print('2026 coverage:', len(need & got26), '/', len(need))          # want 82/82
print('missing 2026:', need - got26)

got25 = set(s.filter(pl.col('year') == 2025)['steamid'].to_list())
print('2025 coverage (for lagged variant):', len(need & got25), '/', len(need))  # want 82/82

# sanity: ratings in a plausible pro range
x = s.filter(pl.col('year') == 2026)
print('rating_ct range:', x['rating_ct'].min(), '-', x['rating_ct'].max())        # expect ~0.7-1.5
print('nulls:', {c: x[c].null_count() for c in x.columns if x[c].null_count()})
```
✅ Pass criteria: **82/82 for 2026**, **82/82 for 2025**, ratings in ~0.7–1.5, no unexpected nulls.

---

## 7. Commit + push

```bash
git pull                       # get Henry's year-map + roster + assemble changes first
git add configs/player_stats_sided.csv
git commit -m "Add 2026 HLTV stats (82 players) + 2025 stats for 14 new players (lagged-prior variant)"
git push origin main
```
Then **tell Henry** — no further action needed from you.

---

## 8. What Henry/Claude does next (so you know where it goes)

1. **Rebuild** the 2026 test set with correct firepower (the corrupted one is baked into the parquet):
   ```bash
   python src/features/assemble.py --parquet-root data/holdout2026/parquet --no-exclude \
       --out data/test_dataset_2026.parquet
   ```
2. **Re-run the holdout** as a **separate, disclosed** second evaluation (the first run stands as the
   honest touch-once result for the old pipeline), for **both** variants:
   - same-year (2026 stats) — consistent with training, leaky
   - lagged prior (2025 stats) — leak-free, the clean test
3. **Answer the real question:** *does firepower transfer out-of-time when the data is right?*
   Either outcome is publishable — a recovered pillar, or a rigorous negative result about skill priors.

---

## 9. Optional bonus — firepower v3 (not blocking)

Two known issues from the v2 benchmark, if you want to bundle a fix:
1. **Count confound (still present in v2):** `ct_rating_sum − t_rating_sum` is **0.987**-correlated with
   the player-count advantage, because Rating is **summed** over alive players → it's mostly a
   count proxy, not skill. **Fix: use the *average* rating per alive player** (per-capita).
2. **Sparse gated features hurt the GBMs:** v2's `opening_*` (only at 5v5), `clutch_*` (only 1vN),
   `entry/trading_*` are mostly NaN → trees overfit them (EF < E; catboost −0.0026, significant).
   **Fix: prune them**, keep what permutation importance likes (`rating`→avg, `adr`, `t_trading`,
   AWP `sniping`).

**But the scrape (§3) is the priority — it unblocks everything.**
