# Study 2 — Match-level win probability: robust plan with pilot evidence

> **Superseded (2026-09-13)** by [plan_round_to_map.md](plan_round_to_map.md) (the polished plan, with the
> football-paper framing and the Betty rungs) and [study_round_to_map_results.md](study_round_to_map_results.md)
> (what was built and measured). Kept for the pilot record.

**Date:** 2026-09-12 · **Pilot code:** `src/models/pilot_match_state.py` · **Pilot outputs:**
`outputs/match_pilot_dp.csv`, `outputs/match_pilot_calibration.csv`, `outputs/match_pilot_meanreversion.csv`,
`outputs/match_pilot_momentum_econ.csv`, log `outputs/match_pilot.log`. **In-time only.**

A "match" is one map (first to 13, MR12 regulation, MR3 overtime). The corpus has one map per series,
so series context is out of scope.

---

## 1. Format facts verified on the data (the DP depends on these)

| Fact | Value | Source |
|---|---|---|
| Maps / rounds | 220 / 4,866 (13–42 rounds per map) | export |
| Overtime maps | 36 (rounds 25+); OT periods of 6 rounds, first to 4 in a period | export |
| Side schedule | Team X = CT in the first half (mode over rounds ≤ 12). X is T for 13–24. **OT: X stays on T for 25–27, then CT 28–33, T 34–39, CT 40–42** (each OT period starts on the side the previous one ended on) | `side_schedule()`; the coded schedule matches 4,843 / 4,866 rows; the 23 mismatches are clan-attribution glitches |
| Map winner | curated `match_winner` (demo list). **11 maps lose their clinching round to validation trimming**, so the last recorded round's winner is wrong for them; 10 maps lack round 1 | pilot label check: curated winner agrees with cumulative round wins on 100% of rows once the missing rounds are accounted for |
| Score columns | `ct_score`/`t_score` in the export and the training table are cumulative **side** wins (not re-keyed at halftime) — unusable after round 12; team scores are rebuilt from round winners | see Study 1, P1 |
| Round-level CT win rate | 0.487 (2024–25); **0.543 in 2026** | export / holdout table |

---

## 2. Pilot: the floor that every rung must beat

Score-state dynamic program (backward induction over (X wins, Y wins, next round), side schedule as
above), evaluated at every round start against the curated map winner; match-block bootstrap.

| Model | AUC | log-loss (95% CI) | Brier |
|---|---:|---|---:|
| A0, side-blind (p = 0.5 every round) | 0.776 | 0.5652 (0.515, 0.619) | 0.1934 |
| A0, constant side rate (p_CT = 0.487) | 0.776 | 0.5654 (0.514, 0.618) | 0.1935 |
| A1-lite: current round from a t=0 logistic (format + buy), future rounds constant | 0.778 | 0.5655 | 0.1933 |

Paired Δlog-loss A1-lite − A0: **+0.0000 (−0.0017, +0.0020)**. The current round's economy adds
nothing at match level; the side asymmetry of Inferno adds nothing in 2024–25 (it will in 2026).

By phase (A0): half 1 AUC 0.728, log-loss 0.607; round 13 AUC 0.847; half 2 AUC 0.846, log-loss
0.494; overtime AUC 0.710, log-loss 0.616.

**The independent-rounds DP is over-confident for the leader at every margin:**

| X − Y at round start | n | A0 predicted | realised X wins map |
|---:|---:|---:|---:|
| −4 | 263 | 0.139 | 0.209 |
| −3 | 358 | 0.214 | 0.279 |
| −2 | 470 | 0.301 | 0.340 |
| +2 | 428 | 0.729 | 0.661 |
| +3 | 311 | 0.815 | 0.720 |
| +4 | 215 | 0.879 | 0.823 |
| +6 | 285 | 0.975 | 0.954 |

Comebacks happen more often than an i.i.d. round model allows. The reason is visible in the round
dynamics:

- **P(X wins round | X won the previous round) = 0.638 vs 0.357** (non-pistol rounds). Rounds are
  strongly positively dependent, which raises the variance of the remaining-rounds sum and hence the
  comeback probability from any given deficit.
- **In the t=0 ladder (Study 1), the previous outcome adds only +0.004 AUC once the buy state is
  known.** The dependence is therefore almost entirely economy carry-over (money and equipment
  persist across rounds), not an unexplained "momentum". The coarse economy class (≥ $3,800 side
  total counts as a full buy) is too crude to show it: conditioning on both teams being "full buy"
  still leaves 0.604 vs 0.384.
- **The next-round win rate rises with the lead** (0.44–0.48 when trailing by 2–6, 0.53–0.63 when
  leading by 2–6): the score reveals team quality. This effect pushes the other way (leads should be
  safer), and the data say the economy dependence dominates.

Conclusion for the design: **the match model's error is structural, not a missing feature.** It needs
round-to-round dependence carried through the economy, and a within-match estimate of team strength.
A bigger classifier on round-start features (route B in the umbrella plan) cannot fix a state-space
error with 220 outcomes; the DP must be extended.

---

## 3. The robust design: a state-augmented Markov chain, built in rungs

Each rung is nested in the next; each is evaluated identically (§4); each must beat the previous by a
paired Δlog-loss CI that excludes zero, or it is dropped.

| Rung | State | Per-round probability | What it adds | Pilot status |
|---|---|---|---|---|
| **A0** | (a, b, k) | constant by side | baseline | done: LL 0.5654 |
| **A1** | (a, b, k) | current round from the Study 1 t=0 model; future rounds constant | economy of the round about to be played | done (lite): +0.0000, drop unless the full t=0 model changes it |
| **A2 economy chain** | (a, b, k, e_X, e_Y) with e = equipment tier of each team, 4 tiers cut at the empirical quartiles of side equipment value (not the current class thresholds) | p(side, e_X, e_Y) estimated from the 4,866 rounds (logistic on side × tiers); transition P(e' \| e, won/lost, pistol-next) estimated empirically | the carry-over that makes leads less safe; the pistol rounds 1 and 13 reset the chain | to build |
| **A3 latent strength** | A2 + a Beta posterior on X's true round-win edge, updated from rounds played so far, with prior mean from the lagged HLTV rank difference (Study 1's `lrank_diff_lag`) and prior strength n0 tuned by CV | the "leader is the better team" effect, and the pre-match prior at 0-0 | to build |
| A4 residual check | GBM on [logit A3, phase, tiers, streaks] | detects any structure the chain misses; if it adds < MDE, the chain is adequate | to build |
| **C per-second chaining** | P_match(t) = p_round(t)·V(a+1, b) + (1 − p_round(t))·V(a, b+1) with V from A3 | the broadcast object: a match-long WP path (~220 paths × ~2,000 seconds) | to build |

A2 keeps the state space small: 13 × 13 × 42 × 4 × 4 ≈ 114k states plus overtime; exact backward
induction in seconds. A3 adds a two-parameter posterior (wins, rounds) per state, so it is solved by
forward simulation (10k paths per round start) rather than exact induction; the simulation is seeded
by the same transition tables, so A2 and A3 differ only in the strength update.

**Team modeling, in cost order and all lagged:** (1) side constants (done, adds nothing in 2024–25);
(2) lagged HLTV rank difference as the A3 prior mean; (3) roster-derived side-specific strength from
previous-season side-split ratings as an alternative prior mean; (4) leave-future-out Inferno form as a
control. Team dummies are excluded (≤ 12 maps per team). An Elo on scraped HLTV results is deferred
unless (2)–(3) leave the 0-0 prior at chance.

---

## 4. Evaluation protocol (what makes it robust)

- **Unit:** every round start (4,866 rows), label = curated map winner. Metrics: log-loss (primary,
  proper), Brier, AUC (secondary), calibration slope/intercept, and **calibration by score margin and
  by phase** (the table in §2 is the headline diagnostic; a correct model flattens it).
- **Significance:** match-block bootstrap (B=500); all comparisons paired. From the pilot, the paired
  Δlog-loss half-width is ≈ 0.002 while the marginal log-loss CI is ±0.05, so **only paired deltas are
  interpretable; marginal CIs at match level are too wide to rank models**. MDE ≈ 0.002 log-loss.
- **Consistency constraints (tested, not assumed):** at 0-0 the output equals the pre-match prior;
  monotone in score difference; symmetric under relabelling X↔Y; V(12,12) at round 25 near 0.5.
- **Martingale checks at match level:** per-round increments mean-zero given the state; realised
  quadratic variation vs p0(1 − p0); the Pipping–Wyner extreme-path benchmark on the match path
  (write-off frequency of the eventual winner at ≤ 5/10/20%, PIT, D_upper). Match paths have ~25–40
  decision points per map (and ~2,000 per-second steps under rung C), so the discrete-time caveat
  that limited the round-level test is much weaker here.
- **Economy carry-over ablation:** A2 − A0 by phase; the gain should concentrate after pistol rounds
  and eco rounds. Report it by round index.
- **Out-of-time (once, disclosed):** the 27 2026 maps (564 round starts). The 2026 side rate is
  0.543, so a 2024–25-fitted side constant is wrong by ≈ 0.056; report the model as fitted, and an
  intercept-recalibrated variant, both labelled. **No AUC headline on 27 outcomes**; log-loss, Brier,
  calibration, and the pathwise test are the out-of-time claims.

---

## 5. Problems and how the plan handles them

| # | Problem | Handling |
|---|---|---|
| Q1 | Score columns in the training table and export are side-cumulative (Study 1 P1). | Team scores rebuilt from winners; `assemble.py` fix upstream; export README corrected. |
| Q2 | 11 maps lack the clinching round; 10 lack round 1; 23 rows with clan glitches. | Labels from the curated list; DP states come from rebuilt cumulative wins so a missing round only removes one row; glitch rows excluded from transition estimation. |
| Q3 | Overtime side pattern differs from the naive assumption. | Encoded from the data and unit-tested against `side_schedule()`; 36 OT maps give 290 OT round starts, enough to check OT calibration but not to tune anything OT-specific. |
| Q4 | Only 220 map outcomes. | Paired deltas only; the state-space model has ~40 parameters (tiers × side, transitions), not thousands; A4 exists to show a flexible learner cannot beat it. |
| Q5 | Economy class thresholds (`≥ $3,800` side total) make almost every round a "full buy". | Tiers re-cut at empirical quartiles of side equipment value for A2; the class columns are not used. |
| Q6 | Lagged team priors are undefined for 2024 (no 2023 tables). | A3's prior mean falls back to 0 (no edge) for 2024 maps with a missing-flag; the 2023 scrape in Study 1 P4 removes this. |
| Q7 | 2026 side shift (0.543). | Disclosed recalibrated variant; the chain's side constant is the one parameter allowed to move. |
| Q8 | Per-second chaining inherits the round model's over-reaction to T-side opening kills (Study 1). | Report the match path's QV ratio; if inflated, apply the increment shrinkage tested in Study 1. |
| Q9 | Series context (map 1/2/3, veto) is unavailable. | Out of scope; stated as a limitation. |

---

## 6. Why it is worth doing

- The pilot already shows a **specific, quantified failure of the standard approach** (an i.i.d. score
  DP is over-confident for leaders by 5–10 points at 2–4 round margins) and its mechanism (economy
  carry-over makes rounds dependent). Fixing that is a clean, testable modelling contribution.
- The match path is the **natural home for the advisor's extreme-path benchmark**: long, many
  decision points, and a base rate that is exactly 0.5 by relabelling. The round-level version of that
  test had to hedge on discreteness; this one does not.
- It reuses everything: the t=0 model (Study 1) for the current round, the per-second EB2 model for
  rung C, the curated match list for labels, and the bootstrap machinery. No new data is required.
- Deliverable for a broadcast: the per-second **match** win probability, which is what viewers
  actually track, produced by chaining a round model whose honesty has already been audited.

---

## 7. Deliverables and effort

- `src/models/match_wp.py`: schedule + OT rules with tests, A0–A3, forward simulation, per-second
  chaining, evaluation battery, one disclosed out-of-time run — 4 days.
- Match-level pathwise benchmark reusing `pathwise_calibration.py` — 1 day.
- Figures: calibration-by-margin before/after A2/A3; prior weight vs round index; one full-map WP path
  with the round curve underneath; extreme-path benchmark at match level — 1.5 days.
- Paper section "From round to match" — 1.5 days. **Total ≈ 8 working days.**
