# Study 1 — The round-start (t=0) win probability: concrete plan, pilot evidence, problems, and justification

**Date:** 2026-09-12 · **Pilot code:** `src/models/pilot_t0_round_start.py` · **Pilot outputs:**
`outputs/t0_pilot_ladder.csv`, `outputs/t0_pilot_martingale.csv`, `outputs/t0_pilot_openingkill.csv`,
`outputs/t0_round_table.parquet`, log `outputs/t0_pilot.log`.
**Data:** round-level export (4,866 rounds / 220 matches, round-level base rate P(CT) = 0.487) built from
the uploaded supplement; per-second table for the anchor checks. **In-time only.** The 2026 holdout was
not touched by the pilot.

---

## 1. What the pilot established (numbers first)

Round-level logistic regression, 5-fold GroupKFold by match, paired match-block bootstrap (B=500).
Scores rebuilt from round winners (see Problem P1). All deltas are paired ΔAUC vs the buy-state rung R2.

| Rung (nested) | cols | AUC | log-loss | ΔAUC vs R2 (95% CI) |
|---|---:|---:|---:|---|
| R1 format + side + true score | 9 | 0.536 | 0.6913 | −0.146 (−0.165, −0.128) |
| **R2 + buy state** | 37 | **0.682** | 0.6163 | — |
| R3 + recent history (prev result, loss streak, prev equipment) | 45 | 0.687 | 0.6163 | +0.0043 (+0.0002, +0.0085) |
| R4 + team prior, same-year HLTV rank + Inferno form (**leaky**) | 51 | 0.692 | 0.6161 | +0.0101 (+0.0035, +0.0163) |
| R5 + player prior, same-year side-split rating (**leaky**) | 54 | 0.693 | 0.6157 | +0.0103 (+0.0038, +0.0172) |
| R4L team prior, **lagged** (prev-season rank; 2025 matches only) | 53 | 0.685 | 0.6175 | +0.0024 (−0.0032, +0.0079) |
| R5L player prior, **lagged** | 58 | 0.685 | 0.6172 | +0.0025 (−0.0029, +0.0082) |
| priors only, same-year | 11 | 0.536 | 0.6922 | −0.147 |
| priors only, lagged | 15 | 0.493 | 0.6958 | −0.190 |
| per-second model A, value at first snapshot | 17 | 0.692 | 0.6138 | +0.0097 (+0.0027, +0.0158) |
| per-second model EB2, value at first snapshot | 72 | 0.691 | 0.6149 | +0.0086 (+0.0021, +0.0152) |
| set-A columns fitted at round level (no `bomb_planted`) | 16 | 0.691 | 0.6142 | +0.0090 (+0.0033, +0.0146) |
| **STACK: first-snapshot A logit + history** | 9 | **0.695** | 0.6137 | +0.0129 (+0.0056, +0.0191) |
| STACK + history + priors (same-year, leaky) | 18 | 0.700 | 0.6132 | +0.0175 (+0.0100, +0.0249) |
| STACK + history + priors (lagged) | 22 | 0.693 | 0.6150 | +0.0109 (+0.0026, +0.0182) |

Five facts follow.

1. **A single round is close to a coin flip before it starts.** Score and format alone give 0.536.
   The buy state is the whole t=0 signal (0.68); nothing above it moves log-loss by more than 0.003.
2. **Priors are weak at t=0, even leaky ones.** The same-year construction (which uses ratings
   computed partly from the match itself) adds +0.010 AUC; the honest lagged construction adds
   +0.002 and its CI includes zero. Priors alone are at chance (0.49–0.54). This is the same
   conclusion as the firepower pillar, now at the one place where a prior had its best chance.
3. **A dedicated round-level fit is not better than the per-second model's first snapshot.** The
   16 set-A columns fitted on 4,866 rounds reach 0.691, the same as the first-snapshot value of the
   per-second model (0.692); the 37-column buy rung is *worse* (0.682) because the per-player loadout
   columns add noise, and stronger regularisation does not recover it (C=0.01: 0.685).
4. **The right construction is a stack, not a new model.** Using the first-snapshot logit of the
   per-second economy model as a base and adding history (+ priors) gives the best t=0 model
   (0.695; 0.700 with leaky priors), nests the per-second model by construction, and cannot
   double-count economy.
5. **The earlier "duration bias" worry was wrong.** Because `time_elapsed_sec` is a feature, the
   per-second model at t≈0.3 s is conditioning on the first-snapshot subpopulation, which has
   exactly one row per round. Its first-snapshot value is calibrated at round level (ECE 0.020).

### Martingale anchor checks (per-second EB2 curve anchored at the demo-only t=0 model R3)

| Identity | Observed (EB2) | Observed (A, economy only) | Under the null |
|---|---|---|---|
| mean p0 vs mean first-snapshot value | 0.4870 vs 0.4878; mean jump +0.0008, mean \|jump\| 0.041 | — | 0 |
| E[(Y − p0)²] / E[p0(1 − p0)] | 0.2162 / 0.2099 = **1.03** | 1.02 | 1 (calibration) |
| E[jump0² + Σ_t Δp_t² + (Y − p_last)²] / E[(Y − p0)²] | (0.2604 + 0.0214) / 0.2162 = **1.30** | 1.31 | 1 (martingale) |
| lag-k autocorrelation of increments, k = 1 / 2 / 3 / 5 / 10 s | −0.039 / −0.030 / −0.025 / −0.012 / −0.003 | −0.036 / −0.025 / −0.016 / −0.006 / −0.002 | 0 |

The terminal identity holds (the prior is calibrated). The **quadratic-variation identity fails by
30%**: for a discrete-time martingale (Y − p0)² = jump0² + Σ Δp² + (Y − p_last)² + cross terms with
zero-mean cross terms, but the realised cross terms sum to −0.066. The path moves more, in squared
terms, than any honest sequential forecast that starts at p0 and ends at the outcome. The excess is
**short-range mean reversion**: increment autocorrelations are individually small but consistently
negative over lags of 1–5 s and vanish beyond 10 s; summed over ~98 increments per round they
account for the gap. Two things make this robust: (i) the economy-only model shows the same excess
(1.31), so it is about how the state features evolve within a round (damage, timers), not about the
spatial pillars; (ii) the fix is not smoothing — a 3-s moving average drives the ratio to 0.37
(under-dispersed, because it also removes the real kill-driven jumps). Increment shrinkage
(p̃_t = p0 + λ(p_t − p0)) was tested: **λ = 0.85 restores the identity exactly (ratio 1.005) but costs
0.011 log-loss (0.4575 → 0.4687)**, so the extra movement is *informative* second by second. The
reconciliation is that the per-second model is calibrated marginally (each p_t ≈ E[Y | state_t]) but
is not a martingale with respect to its own history: E[Y | p_t, p_{t−1}, …] ≠ p_t, because the state
features carry transient information (damage taken, momentary positions) that the level forecast
uses and then partially retracts. The remedy is therefore a **path-aware forecast** (lagged p_t or a
short exponential history as features, or the sequence models, whose OOF paths can be scored on the
same identity), not shrinkage. This gives the deep-model dead heat a new axis: TCN/Transformer may
tie on AUC yet differ on martingale consistency. This is a new honesty diagnostic that the
extreme-path test did not see (that test found no upper-tail inflation).

### Opening-kill increment (first kill of the round, 4,851 rounds; median at 33 s)

| First death on | n | WP before → after | realised CT win | post-kill ECE |
|---|---:|---|---:|---:|
| CT | 2,068 | 0.398 → 0.216 (jump −0.18) | 0.234 | 0.028 |
| T | 2,783 | 0.516 → 0.728 (jump +0.21) | 0.678 | **0.052** |

The model **over-reacts to a T-side opening death by about 5 points** (in the central p0 bin,
0.730 predicted vs 0.669 realised), while its reaction to a CT death is calibrated. The size of the
jump is state-dependent: ±0.22 in even states, ±0.04 in decided ones. This is the "starting point,
then the first event moves it" object the advisor asked for, measured rather than illustrated.

---

## 2. Definition and the design the data supports

**p0(r) = P(CT wins round r | information at freeze-end of round r)**, one row per round.

**Construction (stacked):**
```
logit p0 = β·z_A + γ·H + δ·Π            z_A = logit of the per-second economy model (set A)
                                             evaluated at the round's first snapshot (OOF)
                                        H  = recent-history block (demo-derived)
                                        Π  = prior block (external, lagged only)
```
`β` is free (a stacked model; expect β ≈ 1, report it as a calibration check). Fitted at round level
with GroupKFold by match on the same folds that produced `z_A`, so no row's `z_A` was predicted by a
model that saw its match.

**Why stack rather than refit:** fact 3 above. The per-second model's economy coefficients are
estimated from 476k rows and are the best available t=0 economy estimate; a round-level refit on the
same columns merely matches it, and any extra buy detail hurts. Stacking makes the t=0 model a strict
superset of what the pathwise benchmark already uses as p0, so every improvement is attributable.

### Feature blocks (exact sources)

| Block | Columns | Source | Note |
|---|---|---|---|
| z_A | `logit(p_A_first)` | `oof_predict(train, FEATURE_SETS["A"], "logreg")`, first snapshot per round | requires the score-feature fix (P1) upstream, or the corrected score passed in |
| H history | `ct_won_prev`, `t_won_prev`, `ct_loss_streak`, `t_loss_streak`, `ct_prev_equip`, `t_prev_equip`, `momentum_diff`, `streak_diff` | `rounds.csv`, team-keyed lags across the halftime swap (implemented in the pilot) | undefined at rounds 1/13 (pistols) and after a missing previous round → 0 plus a flag |
| Π team, lagged | `lrank_diff_lag`, `ct_rank_lag`, `t_rank_lag`, `*_rank_lag_missing`, `ct_form`, `t_form`, `form_diff` | `team_rankings.csv` previous-season snapshot; Inferno form = leave-future-out map win rate from earlier corpus matches (Laplace-shrunk) | lagged rank exists only for 2025+ matches (P4) |
| Π player, lagged | `ct_rating_lag_mean`, `t_rating_lag_mean`, `rating_diff_lag`, `ct_n_lag`, `t_n_lag` | `player_season_stats.csv` previous season, side-split (`rating_ct` for CT players, `rating_t` for T players), mean over the five starters | coverage 0% for 2024 matches, 97% for 2025 (P4) |
| Π opening | `ct_opening_lag`, `t_opening_lag` | HLTV opening-duel scores, already scraped | candidate; ties to the opening-kill study |

Dropped after the pilot: per-player loadout spread (`equip_min/std`, helmets, `n_full`) — noise.

---

## 3. Problems found (each with the fix)

| # | Problem | Evidence | Fix |
|---|---|---|---|
| **P1** | **`ct_score`/`t_score` in the pipeline are cumulative *side* wins, never re-keyed at the halftime swap** (`assemble.py` line 90). In half 2, `ct_score` = starting-CT team's half-1 wins + the other team's half-2 wins. | Agreement with the true current-side score: 95% half 1, 16% half 2, 4% OT. Affects `ECONOMY_COLS` in every published model and the Wyner export, whose README calls it "rounds won by each side before this round". | Rebuild team scores from `round_winner_team_clan` (pilot does this). Patch `assemble.py` to re-key at the swap and add a `score_diff_team` column. The published models are unaffected in practice (`Score` alone is AUC 0.536), but the export documentation must be corrected before Wyner uses it. |
| **P2** | The round-level export mixes round-start state and **round outcomes** under per-second column names: `bomb_planted`, `bomb_site`, `bomb_plant_*`, `reason` are outcomes. | Fitting `ECONOMY_COLS` from the export at round level gave AUC 0.89 (leak). | Never take a t=0 feature from an outcome column; the pilot's `R1/R2` lists are the whitelist. |
| P3 | Missing rounds: 10 matches lack round 1, 11 lack the clinching round, 1 has a gap (validation trimming). | `first_round != 1` in 10; cumulative wins reproduce the curated score in 209/220. | History features carry a `prev_missing` flag; match labels come from `demo_list_final.csv`, never from the last recorded round. |
| **P4** | **No lagged priors for 2024 matches**: `team_rankings.csv` and `player_stats_sided.csv` start in 2024, so "previous season" is undefined for 92 of 220 matches. | Lagged coverage 59% (rank), 57% (players); 0% in 2024, 97% in 2025. | Either (a) Haiwen scrapes the 2023-year-end HLTV top-30 and 2023 side-split stats for the ~130 players appearing in 2024 matches (one afternoon, same schema), or (b) the prior rungs are evaluated on 2025 matches only (128 matches, 2,865 rounds) with the CI widened accordingly. (a) is recommended; (b) is the fallback and must be stated. |
| P5 | Same-year HLTV rank is a year-end snapshot, so for a January match it encodes the whole season to come. | By construction. | Same-year priors are reported only as the leaky upper bound; the headline uses lagged. |
| P6 | Inferno form from the corpus is thin (sampling caps ≤12 maps per team; form is computed from earlier corpus matches only). | Laplace-shrunk form contributes inside R4 with no separable effect. | Keep as a control; do not interpret its coefficient. |
| P7 | The achievable effect is small: a round is ~coin flip given the buy (0.68), and the whole prior block moves it by ≤ +0.01. | Ladder above. | State the minimum detectable effect (§4) before running; frame the study as measurement, not improvement. |
| P8 | 2026 is more CT-sided at round level (0.543 vs 0.487). | Pilot base rates. | Out-of-time report includes calibration slope/intercept and an intercept-recalibrated variant (disclosed). |
| P9 | The 30% excess quadratic variation could reflect OOF fold boundaries, 1 Hz sampling, or the omitted final jump rather than model dynamics. | All snapshots of a round come from one fold, so folds are not the cause. The final jump is now included (it makes the excess larger, 1.21 → 1.30). Smoothing over-corrects (3-s MA → 0.37). Increment autocorrelations are −0.04…−0.01 over lags 1–5 s for both EB2 and A. | Report the identity with the final jump; report multi-lag autocorrelations; test increment shrinkage (λ) as the remedy; restrict a variant to increments at kill ticks to separate event jumps from between-event drift. The identity is exact for any discrete-time martingale, so a persistent excess is real. |
| P10 | 0.06% of rounds have their first snapshot later than 1 s after freeze-end. | Export README. | Define p0 from the first snapshot only when t < 1 s; otherwise drop the round from the anchor checks. |
| P11 | Team-clan attribution glitches (mode of `team_clan_name`) flip the side in a few rounds. | 23 of 4,866 rows disagree with the side schedule. | Use the mode over the half, not the round; flag and exclude the 23 rows from history features. |

---

## 4. Evaluation protocol

- **Unit and folds:** one row per round; 5-fold GroupKFold by match, identical fold assignment for
  `z_A` and the stack. Metrics: AUC, log-loss, Brier, ECE, calibration slope/intercept.
- **Significance:** paired match-block bootstrap (B=500) on every nested comparison. From the pilot,
  the half-width of a paired ΔAUC CI on this table is ≈ 0.004–0.007, and of a paired Δlog-loss CI
  ≈ 0.001–0.003. **Minimum detectable effect ≈ +0.006 AUC / 0.002 log-loss.** A prior block that
  cannot clear that is reported as "no detectable effect", not "no effect".
- **Out-of-time (once, disclosed):** the 564 round-starts of the 27 2026 matches, with every prior
  lagged (2025 rank, 2025 stats). Report the same metrics plus slope/intercept; the primary
  out-of-time claim is calibration, not AUC (564 rounds give a ±0.04 AUC CI).
- **Anchor checks (per-second EB2 vs the t=0 model):** the three identities in §1, by p0 decile,
  with match-block CIs, plus the P9 robustness variants.
- **Opening-kill increment:** by victim side × p_before quintile; report post-kill calibration
  (mean predicted vs realised) with CIs; then the same for the *second* kill to see whether the
  over-reaction persists. Extension: condition on where the death happened (zone) and on whether it
  was a trade within 5 s.

---

## 5. Why this design is statistically sound

1. **Round is the correct unit.** The label is a round outcome; snapshot-level fits weight long
   rounds more. The stack is fitted at round level, and `z_A` is a per-round scalar.
2. **No leakage.** Folds are by match for both stages; `z_A` for a row was produced by a model that
   never saw its match; every prior is lagged to the previous season; history uses only earlier
   rounds of the same match; outcome columns of the export are excluded (P2).
3. **Correct dependence structure.** Rounds within a match are correlated (momentum, economy
   carry-over, team identity); the match-block bootstrap resamples matches, so CIs are valid under
   that correlation. Paired resampling isolates the increment of each rung.
4. **Pre-specified nested ladder.** Each rung adds one block; the increment is the quantity reported;
   there is no post-hoc selection among sets. Same-year and lagged constructions are both reported so
   the leaky upper bound and the honest estimate bracket the truth.
5. **The martingale checks are exact expectations, not heuristics.** For any calibrated forecast,
   E[(Y−p0)²] = E[p0(1−p0)]; for any discrete-time martingale, E[Σ Δp²] = E[(Y−p0)²]. Ratios of 1 are
   the null; a 21% excess is a quantitative statement about the curve, with a match-block CI.
6. **The increment study conditions on the pre-event state.** Comparing p_after to the realised
   outcome within p_before bins removes the confound that first blood is more common for the side
   that was already ahead.

---

## 6. Why the study is worth doing (and what each outcome means)

- **It supplies the object the advisor's benchmark needs.** The extreme-path test anchors at p0; today
  that is the model's own first output. A t=0 prior built from pre-round information is the proper
  anchor, and the pilot shows the two agree (jump +0.001), which itself is a reportable result.
- **It already produced two new honesty findings** in under two minutes of compute: the curve carries
  30% excess quadratic variation from short-range mean reversion (informative, not noise: shrinking
  it costs log-loss), and it over-reacts by 5 points to a T-side opening death. Both slot into the
  existing "honest probabilities" section beside the extreme-path result, and both point to the same
  fix, a path-aware forecast, which can be tested with the same identities and which re-opens the
  deep-model comparison on an axis where sequence models should have an edge.
- **It closes the prior question.** The paper's firepower result says an external skill prior adds
  nothing once in-round state is rich. The t=0 model is the one place with no in-round state. The
  pilot says the prior adds ≤ +0.01 even there (leaky) and ~0 lagged. Reporting that turns a pillar-
  specific negative into a general statement: in this game, pre-round information beyond the buy is
  nearly uninformative about a single round. If the 2023 scrape reverses it, that is equally
  reportable.
- **It is the input Study 2 needs.** The match-level DP consumes per-round probabilities at round
  start; the pilot's A1-lite already shows what the current round's economy adds at match level.
- **It is cheap and reversible.** No re-parse, no nav mesh, no GPU. The only new data is an optional
  2023 scrape.

Expected outcome, stated in advance: STACK + history significant over the first snapshot by
≈ +0.003–0.005 AUC (history is real but small); lagged priors not significant; the 30% QV excess and
the T-side over-reaction replicate on 2026; a path-aware variant (lagged p_t as a feature) brings the
identity toward 1 without the 0.011 log-loss cost that level shrinkage incurs, and the TCN's OOF
path sits closer to 1 than the logistic path. The paper gains a "round-start prior" subsection with one
table (the ladder) and one figure (increment calibration + QV by decile).

---

## 7. Deliverables and effort

- `src/models/round_start_wp.py` (promote the pilot: table build, stack, ladder, CIs, out-of-time
  once, anchor checks, increment study) — 1 day.
- `assemble.py` score fix + export README correction (P1, P2) — 0.5 day; re-run the affected docs.
- Optional 2023 scrape spec for Haiwen (P4) — 0.5 day to write, one afternoon to run.
- Figures: ladder forest; increment calibration by victim side; QV ratio by p0 decile — 1 day.
- Paper subsection + table — 1 day. **Total ≈ 4 days plus the optional scrape.**
