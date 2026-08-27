# Firepower v4: Mean-Normalised Variants

## Motivation

v1/v2/v3 all compute firepower statistics as **sums over alive players**.
Because professional HLTV ratings cluster tightly around 1.0, a sum over
*n* alive players ≈ *n*.  The correlation between `ct_rating_sum` and
`ct_players_alive` is **0.987** — the "skill" feature is almost entirely
re-encoding headcount.

v2 attempted to fix this with situational gates (clutch, entry, opening),
but kept sums.  v3 added team-ranking weights, but also kept sums.
Neither divides by the number of alive players, so the count confound
survives in both.

The most obvious fix — dividing by `n_alive` to produce a genuine
per-player average — was never tested in v1, v2, or v3.  v4 tests it.

---

## Three Variants

### v4.1 — Rating mean only, no ranking weight

**Formula:**
```
ct_rating_v41 = ct_rating_sum / ct_players_alive
t_rating_v41  = t_rating_sum  / t_players_alive
```

Only HLTV Rating is used (CT-side and T-side separately).  All other
statistics (ADR, entry, trading, …) are dropped to isolate the rating
signal.  No team-ranking weight.

**Files:** `src/features/firepower_v4_1.py`
**Cols (2):** `ct_rating_v41`, `t_rating_v41`

---

### v4.2 — Rating weighted mean, ranking weight preserved

**Formula:**
```
ct_rating_v42 = team_weight × (ct_rating_sum / ct_players_alive)
             = team_weight × ct_rating_v41
```

where `team_weight = 1 / log2(hltv_rank + 1)` (same as v3, log2 formula).

Since all alive CT players share the same team, the weight is constant
per-side per-match.  This is equivalent to v4.1 scaled by team strength:
a top-1 team's mean 1.05 appears as 1.05; a rank-30 team's mean 1.05
appears as ~0.21.

Halftime side-swap handled identically to v3 (rounds 1–12 use first-half
assignments; rounds 13+ swap).

**Files:** `src/features/firepower_v4_2.py`
**Cols (2):** `ct_rating_v42`, `t_rating_v42`

---

### v4.3 — All v2 stats as means, situational gates preserved

All 14 summed columns from v2 are divided by `n_alive`.  The 6
pass-through columns (kast_mean, clutch_score, awp_sniping_skill) are
already per-player values and are kept as-is.

| v2 column | v4.3 column | note |
|---|---|---|
| ct/t_rating_sum | ct/t_rating_v43 | ÷ n_alive |
| ct/t_adr_sum | ct/t_adr_v43 | ÷ n_alive |
| ct/t_hltv_firepower_sum | ct/t_firepower_v43 | ÷ n_alive |
| ct/t_entry_sum | ct/t_entry_v43 | ÷ n_alive; 0 when lone survivor (gate preserved) |
| ct/t_trading_sum | ct/t_trading_v43 | ÷ n_alive |
| ct/t_opening_sum | ct/t_opening_v43 | ÷ n_alive; NaN when not 5v5 (gate preserved) |
| ct/t_weighted_utility | ct/t_utility_v43 | ÷ n_alive |
| ct/t_kast_mean | ct/t_kast_v43 | pass-through (already mean) |
| ct/t_clutch_score | ct/t_clutch_v43 | pass-through (lone survivor, individual) |
| ct/t_awp_sniping_skill | ct/t_awp_v43 | pass-through (individual) |

**Files:** `src/features/firepower_v4_3.py`
**Cols (20):** see `FIREPOWER_V4_3_COLS`

---

## Implementation Notes

All v4 features are **proxy-derived** from columns already present in
`training_dataset.parquet` (`ct_rating_sum`, `ct_players_alive`, etc.).
`assemble.py` does not need to be re-run.

For v4.2 team weights, `eval_firepower_v3.build_match_weights()` is
reused directly.

Evaluation script: `src/models/eval_firepower_v4.py`
Output: `outputs/firepower_v4_benchmark.csv`

---

## Results (Logistic Regression only)

Evaluation: 5-fold GroupKFold OOF on training set (CV) and full
train → 2026 holdout (OOT).  Contested = equal alive players AND
|Δequipment| ≤ $1500.

| Version | CV AUC | CV cAUC | OOT AUC | OOT cAUC |
|---|---|---|---|---|
| EB2 (no FP, baseline) | 0.8508 | 0.5963 | **0.8474** | **0.6553** |
| EFB2 (v2 raw sum) | 0.8519 | 0.6033 | 0.8236 | 0.6166 |
| EFB4.1 (rating mean) | 0.8508 | 0.5955 | 0.8458 | 0.5948 |
| EFB4.2 (wtd mean) | 0.8510 | 0.5940 | **0.8478** | **0.6554** |
| EFB4.3 (all means) | **0.8520** | **0.6051** | 0.8107 | 0.6085 |

---

## Conclusions

**1. Mean normalisation does not rescue the Firepower pillar.**
v4.1 is statistically indistinguishable from EB2 (no firepower) both
in-sample and out-of-time.  Dividing by player count removes the count
confound but also removes the only signal the feature had: the model
already sees `ct_players_alive` directly.

**2. v4.2 (weighted mean) is the best-behaving FP variant out-of-time.**
OOT AUC 0.8478 and OOT cAUC 0.6554 are essentially tied with the
EB2 baseline (no FP at all).  Unlike v2/v3, v4.2 does not degrade
sample-out performance.  However, it provides no statistically
significant *improvement* either.

**3. v4.3 (all means) overfits, same pattern as v2/v3.**
CV is the best of all variants (0.8520 / 0.6051), but OOT AUC collapses
to 0.8107 — a larger drop than even v2 (0.8236).  More features with
gate-dependent sparsity continue to overfit despite mean normalisation.

**4. The binding constraint is information, not encoding.**
All three variants confirm the paper's conclusion: the observable match
state (economy, map control, bomb geometry) already absorbs whatever
team-quality signal an external skill prior can offer.  The problem is
not how the rating is encoded — it is that the rating adds nothing once
the demo-derived features are present.

**5. v4.2 as a practical recommendation.**
If a FP feature must be included, v4.2 (ranking-weighted mean rating,
2 features) is the preferred encoding: minimal feature count, no
out-of-time degradation, and avoids the count confound that made v1 a
false positive.

---

# Verification on the shared checkout — 2026-08-27

Everything above was produced on a machine with the rebuilt 129-column datasets. This
section records what could and could not be reproduced on the shared checkout, plus three
findings that change how the results should be read. Code for all of it is on the
`exp/defuse-time` branch.

## Finding 1 — CORRECTED: the stale-year table is the June bundle, not the one used above

*(Superseded by the drive check on 2026-08-27. Kept because the diagnosis method is the
same one that found the leak below.)*

The 2026 table distributed in `cs2_test_bundle_2026.zip` (July, 115 columns) resolves its
skill prior to **2024**, not 2026 — proven by recomputing `firepower_features()` at three
lags and matching the shipped values exactly at 2024 (max diff 0.0000, 100%), while 2025
and 2026 miss by up to 1.2. Cause: `year_for_match()` falls back to `DEFAULT_YEAR = 2024`
for demos missing from `configs/demo_year_map.csv`.

**But the table the §五 benchmark actually used is the rebuilt one (2026-08-25, 129
columns), and that one resolves to 2026 correctly** (100% match at lag 0, zero rows with
alive players and `rating_sum = 0`). So the stale year was already fixed before §五 ran,
and it does not explain the OOT results. §六.1's statement that the OOT benchmark uses
2026 stats is correct for that table. What does explain them is Finding 5.

### Original text (applies only to the July bundle)

`data/test_dataset_2026.parquet` was assembled with the skill prior resolved to **2024**,
not 2026. Proven by recomputing `firepower_features()` at three lags against the shipped
table, over 4 matches:

| lookup year | max abs diff vs shipped table | exact match |
|---|---|---|
| 2026 (lag 0) | 1.2000 | 0.0% |
| 2025 (lag 1) | 1.2000 | 0.0% |
| **2024 (lag 2)** | **0.0000** | **100.0%** |

Cause: `year_for_match()` falls back to `DEFAULT_YEAR = 2024` for any demo missing from
`configs/demo_year_map.csv`. That file *now* labels all 27 matches as 2026, so it was
extended after the table was built. (`FIREPOWER_YEAR_LAG=2` would be indistinguishable, but
a stale year map is the likelier cause.)

**Consequences.**

- §六.1 above states the OOT benchmark evaluates 2026 matches with 2026 stats. It does not
  — it uses 2024 stats.
- It explains the 8.4% of OOT rows that have alive players and `rating_sum = 0`: players who
  debuted after 2024 have no 2024 row. Recomputing with the current configs gives **zero**
  such rows.
- The magnitude is large, not marginal: mean `ct_rating_sum` is 2.986 in the shipped table
  vs 4.405 recomputed (+47.5%), and only 0.2% of the non-zero rows agree.

**This confounds the OOT firepower conclusion.** If the training table used same-year stats
(the default) while the test table used 2024, then the firepower features are *a different
quantity* in train and test, and the observed pattern — the more FP columns a set has, the
worse it does out-of-time (EFB2 0.8236, v4.3 0.8107, both below the FP-free EB2 0.8474) — is
exactly what a train/test feature mismatch produces. The mismatch has to be ruled out before
"the rating adds nothing" (conclusion 4) can stand. Caveat on the caveat: the training
table on this checkout predates the firepower pillar and has no FP columns at all, so what
*it* used could not be checked here.

**Fix:** rebuild the 2026 table with the current `demo_year_map.csv`, then re-run both
benchmarks. For the leak-free variant, apply `FIREPOWER_YEAR_LAG=1` to *both* tables so the
lag policy is consistent, rather than leaving it stale on one side only.

## Finding 2 — the proxy variants and the integrated columns are different features

The proxies (v4.1/4.2/4.3) divide by `players_alive`; the integrated `*_mean` columns divide
by `n_with_stats`, as §三 above correctly argues they should. Where HLTV coverage is partial
these disagree substantially. Measured over 400 real snapshots:

| divisor | median implied per-player rating | range |
|---|---|---|
| `players_alive` (proxy) | 0.836 | [0.000, 1.350] |
| `n_with_stats` (integrated) | 1.086 | [1.030, 1.125] |

Only the second is a skill mean; the first is part skill, part coverage. So the results
table above and a run of `eval_firepower_mean.py` are not measuring the same thing, and
v4.1's null result cannot be transferred to `FIREPOWER_MEAN_COLS` without re-testing. Both
proxy modules now carry this caveat in their docstrings.

## Finding 3 — the mean features are near-constant within a match

Within one match the CT-side rating mean has σ = 0.0169 (T-side 0.0097): the same five
players, so the mean barely moves as they die. v4.1 is therefore close to a **per-match
constant**, and evaluation is 5-fold GroupKFold **by match**. A match-level constant can only
help by generalising across matches, which is the hardest thing for it to do. This is a
plausible mechanism for v4.1 ≈ baseline that is about the *evaluation design*, not about
whether skill carries information.

## Finding 4 — the count confound is weaker than r = 0.987 on 2026

On `data/test_dataset_2026.parquet`, `corr(ct_rating_sum, ct_players_alive)` = **0.629**
(0.711 excluding the zero-coverage rows), not 0.987. Some of that gap is Finding 1's stale
lookup. The 0.987 figure is presumably from the training set and could not be checked here —
the training table on this checkout has no firepower columns. Worth re-measuring on a
correctly-assembled pair before the number is quoted in the paper.

## Naming collision — resolved

`EFB3` was claimed twice: by the mean-FP set above, and by the defuse-progress work on
`exp/defuse-time`. Duplicate keys in a dict literal overwrite **silently**. The defuse sets
were renamed `EB2D` / `EFB2D` (they are unpushed and unreferenced, so they were the cheaper
side to move); `EB2_FPmean` and `EFB3` keep the names used above.

## What is implemented here

| file | state |
|---|---|
| `src/features/firepower.py` | `FIREPOWER_MEAN_COLS` (20) + `FIREPOWER_COVERAGE_COLS`, divisor `n_with_stats`; verified exact on 400 real snapshots |
| `src/features/firepower_v4_1.py` / `_v4_2.py` / `_v4_3.py` | proxy variants, with the divisor caveat |
| `src/models/eval_firepower_v4.py` | proxy benchmark |
| `src/models/eval_firepower_mean.py` | integrated benchmark, `--tag` for the lagged variant |
| `src/models/train_pipeline.py` | `EB2_FPmean`, `EFB3`; 21 sets, no duplicate keys |

**Not runnable on this checkout.** `data/training_dataset.parquet` here is a 69-column build
that predates the firepower pillar entirely, so neither benchmark can produce CV numbers;
and `assemble.py` cannot regenerate it because `~/.awpy/navs/de_inferno.json` is missing and
awpy's data mirror is dead. Both benchmarks need a machine with the 129-column tables.

`configs/hltv_rankings.csv` in the file index above does not exist; the rankings actually
come from `configs/team_rankings.csv` via `firepower_v3._team_rank_lookup()`.


## Finding 5 — the §五 benchmark trained on the out-of-time holdout

`training_dataset.parquet` on the other machine holds **247 matches / 531,866 rows**, not
220 / 476,595. The 27 extra matches are the entire 2026 holdout, present row-for-row
(tick, ct_won, ct_players_alive, ct_equipment_value, ct_rating_mean all identical).

**How.** The 2026 demos were parsed into `data/parquet/` — the default `--out` of
`batch_parse.py` — where 32 files with "2026" in the name still sit. `assemble.py` globs
that directory, `configs/excluded_offlist.txt` has two entries and neither is one of them,
and the 27 that pass round validation went into the table. Nothing on that path errors.

**Confirmed by reproduction.** Re-running the benchmark on the shipped table reproduces
§五 exactly (EB2/logreg OOT 0.8513 vs 0.8513; EB2/XGB 0.8678 vs 0.8676; EB2_FPmean/XGB
0.8784 vs 0.8792). Re-running with the 27 matches removed reproduces the *paper's* record
instead (EB2/logreg OOT 0.8474 — the published value to four decimals).

**Cost, measured:**

| set / model | leaked OOT AUC | clean | Δ | leaked OOT cAUC | clean | Δ |
|---|---|---|---|---|---|---|
| EB2 / logreg | 0.8513 | 0.8474 | −0.0039 | 0.6699 | 0.6551 | −0.0148 |
| EB2 / xgb | 0.8676 | 0.8498 | −0.0178 | 0.7371 | 0.6278 | **−0.1093** |
| EFB2 / xgb | 0.8780 | 0.8411 | −0.0369 | 0.7601 | 0.5844 | **−0.1757** |
| EB2_FPmean / xgb | 0.8792 | 0.8370 | −0.0422 | 0.7611 | 0.5965 | **−0.1646** |

Trees are hit far harder than the linear model (they can memorise; logreg mostly cannot),
and the more features a set has the more it gains from the leak. The one visible symptom
was **OOT scoring above CV** (XGB 0.8676 vs 0.8516) — impossible for a genuine holdout, and
gone once the leak is removed (0.8498 vs 0.8493).

**All eight rows of §五 are void.** Guard added in `assemble.py` (`_guard_test_demos`); it
compares on the bare demo name, because `demo_list_2026_test.csv` stores names without the
`<series_id>__` prefix that parsed files carry, so a plain intersection matches nothing.

## Clean re-run — `outputs/firepower_mean_benchmark_clean220.csv`

Training: `data/training_dataset_clean220.parquet` (220 matches, identical match set to the
paper's). Test: the 2026-08-25 table (2026 stats, verified above).

| set | model | CV AUC | CV cAUC | OOT AUC | OOT cAUC |
|---|---|---|---|---|---|
| EB2 | logreg | 0.8508 | 0.5963 | **0.8474** | **0.6551** |
| EB2 | xgb | 0.8493 | 0.5862 | **0.8498** | 0.6278 |
| EFB2 | logreg | 0.8519 | 0.6035 | 0.8450 | 0.5851 |
| EFB2 | xgb | 0.8483 | 0.5875 | 0.8411 | 0.5844 |
| EB2_FPmean | logreg | 0.8520 | 0.6050 | 0.8458 | 0.5916 |
| EB2_FPmean | xgb | 0.8487 | 0.5887 | 0.8370 | 0.5965 |
| EFB3 | logreg | 0.8520 | 0.6026 | 0.8457 | 0.5874 |
| EFB3 | xgb | 0.8482 | 0.5886 | 0.8382 | 0.5974 |

**Firepower does not help in any configuration — it costs.** Relative to EB2:

| | logreg AUC | logreg cAUC | xgb AUC | xgb cAUC |
|---|---|---|---|---|
| EFB2 (sum) | −0.0024 | −0.0700 | −0.0087 | −0.0434 |
| EB2_FPmean (mean) | −0.0016 | −0.0635 | −0.0127 | −0.0313 |
| EFB3 | −0.0017 | −0.0677 | −0.0115 | −0.0304 |

Three things follow.

1. **Conclusion 4 above survives, strengthened.** The skill prior does not merely add
   nothing once the demo-derived features are present; out-of-time it actively hurts.
2. **The §五 claim that "XGB extracts incremental signal from FP" is refuted.** Clean XGB
   loses 0.030–0.043 contested-AUC when firepower is added, in every encoding.
3. **Mean encoding is a real but small improvement over sums, and only for XGB**
   (cAUC −0.031 vs −0.043; AUC is slightly worse). It fixes the count confound without
   making the pillar useful — which is conclusion 1 above, now on clean data.

The overfitting signature is visible directly: every FP set scores *higher* than EB2 on CV
(0.8519–0.8520 vs 0.8508 for logreg) and *lower* out-of-time. That gap is the pillar's
whole story.
