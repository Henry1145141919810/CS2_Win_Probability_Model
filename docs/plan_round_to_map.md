# From round to map: the map win-probability study (plan, controls, and build status)

**Scope.** One map (de_inferno, MR12, first to 13, MR3 overtime). "Match" below means *this map*.
Best-of-three series are out of scope: the corpus holds one map per series, and series modelling
needs the other maps' data. **Date:** 2026-09-13. **Supersedes** `plan_study2_match_wp.md`.
**Code:** `src/models/map_wp.py` (built rungs), `src/models/pilot_match_state.py` (first pilot).
**Data:** the raw supplement only (per-second first snapshots, `rounds`/`ticks` channels, reference
CSVs) through `src/models/t0_table.py`. The Wyner round-level export is not used.

---

## 1. The object and the idea

We have a per-second round model p_t = P(CT wins the current round | state at t) that is
calibrated, transfers out-of-time, and has been audited for trajectory honesty. The map question is

    P_map(t) = P(team X wins the map | everything known at second t)

for every second of every round, with team X = the side that starts on CT (so the label is symmetric:
X wins 49.1% of maps, and relabelling X↔Y must give 1 − P).

**Carrying the thought from the football paper.** The paper is Brill, Yurko and Wyner, *Exploring the
difficulty of estimating win probability: a simulation study* (JQAS 2026, the file `JQAS WP Model
Football Paper.pdf` at the repo root). Its argument: a *statistical* win-probability model (regression or
boosting from game state to the binary outcome) is fitted on play-by-play rows that all share one draw
of the game's outcome, so its bias and variance are inflated and the effective sample size is roughly
half the nominal one; naive bootstrap intervals under-cover and honest intervals are wide (about 6
points of WP). Its discussion suggests the alternative: a *probabilistic state-space model*, where
transition probabilities are estimated from play-level data (effective sample size = the number of
plays, because transitions are independent observations) and win probability is obtained by
propagating those transitions through the rules of the game. Such a model has lower variance and
higher bias, because it makes stronger simplifying assumptions. Robberechts, Van Haaren and Davis
(KDD 2021) is the same idea carried out for soccer (remaining goals as a Poisson process with
time-varying intensity, W/D/L read off the distribution), and it is the only model in their comparison
that stayed calibrated late in the game.

The map study is that program applied to CS2. The transition model is fitted at the *round* level
(4,866 independent-ish observations rather than 220 map outcomes), the map win probability is the
absorption probability of the first-to-13 rule (computed exactly by backward induction, which is the
limit of "simulating games"), and the rungs below add structure to the transitions one assumption at
a time so the bias can be measured while the variance stays low.

The CS2 analogue is exact and simpler than soccer's:

    remaining rounds ~ a sequence of Bernoulli trials with per-round probability p_k(state_k),
    the map ends by the first-to-13 / overtime rule, and P_map is the absorption probability.

Three things make CS2 harder than the football version and are the reason for the ladder below:
(i) rounds are **not independent**: money carries over, so the loser of a round tends to lose the
next (P(X wins | X won previous) = 0.64 vs 0.36); (ii) the **side** alternates on a fixed schedule,
so p_k depends on k; (iii) at 0–0 the only information is pre-match strength, which in this game is
weak (Study 1: lagged priors add ≈ 0 to a single round).

---

## 2. What is verified on the data (the DP relies on these)

| Fact | Value / source |
|---|---|
| Maps, rounds | 220 maps, 4,866 round starts; 13–42 rounds per map; 36 maps go to overtime |
| Side schedule | X (first-half CT) is CT for rounds 1–12, T for 13–24; in OT X is T for 25–27, then sides alternate every 3 rounds (28–33 CT, 34–39 T, …). Verified on 4,843/4,866 rows; the 23 mismatches are clan-attribution glitches |
| End rule | X wins when a = 13 + 3n and b ≤ 11 + 3n for some n ≥ 0 (13–11, 16–14, 19–17, …); 12–12 (15–15, …) continues. Checked against the curated winner on all 220 maps |
| Scores | pipeline `ct_score`/`t_score` are cumulative *side* wins (never re-keyed at halftime); team scores are rebuilt from round winners. Map winner from the curated demo list (11 maps lose their clinching round to validation trimming) |
| Economy resets | pistol rounds 1 and 13: both sides ≈ $4–5k. Every OT round is a full buy on both sides (median $27–30k), so there is no economy carry-over in OT |
| Economy tiers | side equipment, non-pistol regulation: eco < $6k (7%), force $6–15k (13%), partial $15–22k (14%), full ≥ $22k (66%) |
| Round dependence | P(X wins round k | X won k−1) = 0.64, vs 0.36 if X lost; conditioning on both teams "full buy" by the class column still leaves 0.60 vs 0.38 (the class threshold is too coarse; tiers above are the fix) |
| Score reveals strength | P(X wins next round) rises from 0.44–0.48 when trailing by 2–6 to 0.53–0.63 when leading by 2–6 |
| Base rates | round-level CT win 0.487 (2024–25), 0.543 (2026). X wins the map 49.1% |

---

## 3. The model ladder (each rung nested, each evaluated identically)

| Rung | State | Per-round probability | Parameters | Built? |
|---|---|---|---|---|
| **M0** i.i.d. state model (the football-paper analogue) | (a, b) | constant side rate p_CT | 1 | yes |
| **M1** economy chain | (a, b, e_X, e_Y) with 4 tiers + pistol/OT resets | logistic(side, pistol, tier_X, tier_Y); empirical tier transitions T[won/lost][tier → tier′] | ≈ 12 + 2·(5×4) | yes |
| **M2** + latent strength | M1 + posterior mean of X's neutral edge θ̂(a, b) = (n0·μ + a)/(n0 + a + b), entering p_k as a logit shift; n0 tuned by CV; μ = 0.5 now, lagged-rank prior once 2023 tables exist | M1 + 2 | yes |
| **M3** residual check | GBM on [logit V_M2, a, b, k, tiers, previous outcome, streaks] | ~hundreds | yes (cheap) |
| **M4** per-second chaining | P_map(t) = p_X(t)·E[V(a+1, b, e′)] + (1 − p_X(t))·E[V(a, b+1, e′)], p_X(t) from the per-second EB2 model | 0 new | yes |
| **M5a** round-sequence deep model (Betty) | GRU/Transformer over the map's round history (per-round vectors: outcome, tiers, side, score, kills, plant, duration) → P_map at each round start | 10⁴–10⁵ | planned |
| **M5b** generative simulator (Betty) | autoregressive neural simulator of the remaining rounds (outcome + next economy) rolled out to the end rule; a diffusion model over the remaining-score path is the same idea with a different sampler | 10⁵ | planned, exploratory |
| **M5c** per-second sequence model (Betty) | the existing TCN/Transformer retargeted to the map label (`--target map`), sequence = the whole map | existing code | planned |

**Why the deep rungs are last and gated.** There are 220 map outcomes. M5a/b/c trade the structural
model's low variance for capacity; on this sample they are expected to lose on log-loss and win only
if the chain's remaining bias (§5, calibration by margin) is large. They are worth running because Betty
makes them cheap, and because the generative simulator is the only rung that yields a *distribution
over final scores* (quantiles, expected margin) rather than a single probability. Each deep rung must
(1) beat M2 by a paired Δlog-loss whose CI excludes zero, (2) keep calibration slope within 0.9–1.1,
and (3) show a flatter learning curve than M3 to be reported as anything but a negative result.

---

## 4. Metrics

All at **round-start granularity** (4,866 rows; label = curated map winner), plus per second for M4/M5c.

- **Primary:** log-loss (proper; punishes the leader over-confidence the pilot found). **Secondary:**
  Brier, AUC (AUC is nearly identical across rungs because ranking is dominated by the score state).
- **Calibration:** slope/intercept, ECE, and the headline diagnostic, **calibration by score margin**
  (predicted vs realised X win rate at a − b = −6…+6) and by phase (round 1, half 1, round 13, half 2,
  OT). M0's failure mode is visible here: 0.73 predicted vs 0.66 realised at +2.
- **Trajectory honesty (the advisor's tests, now at map level):** per-round increments mean-zero given
  the state; realised quadratic variation E[Σ ΔP²] vs E[(Y − P0)²]; Pipping–Wyner extreme-path
  benchmark (write-off frequency of the eventual map winner at ≤ 5/10/20% vs the martingale law,
  PIT and one-sided KS). A map path has 20–40 round-level decision points and ~2,000 per-second steps
  under M4, so the discreteness caveat that limited the round-level test is much weaker.
- **Ranked probability score** is not needed (binary outcome, no draws); it is what the football paper
  used for W/D/L.

## 5. Verification, comparison, and controls

| Check | What it establishes |
|---|---|
| **Paired match-block bootstrap** (B=500) on every Δ vs M0 and vs the previous rung | significance; marginal CIs on 220 outcomes are ±0.05 log-loss and cannot rank models, paired deltas can (MDE ≈ 0.002) |
| **Structural constraints** (tested, not assumed) | P(0–0) = pre-match prior; monotone in a − b; symmetric under X↔Y; V(12–12) ≈ 0.5 ± side; V = 1/0 at terminal states |
| **Synthetic negative control** | simulate maps from M0 (i.i.d. rounds, real side schedule); refit M1–M3 on them; they must *not* beat M0. Guards against learning structure that is not there |
| **Label-shift control** | pair each map's states with another map's winner; every rung must fall to 0.693 log-loss |
| **Learning curves** (55/110/165/220 maps, 5 seeds) | the bias–variance picture: M0/M1/M2 should be flat (low variance), M3 and the deep rungs steeper |
| **Economy carry-over ablation** | M1 − M0 by round index; the gain must concentrate after pistol rounds and eco rounds |
| **Strength ablation** | M2 − M1 by round index; the latent edge should matter mid-map (after enough rounds, before the score decides) |
| **Chain consistency** | the per-second M4 path at each round start must equal the M2 value (no jump at freeze end) |
| **Known-truth simulation study (planned; the JQAS methodology)** | generate maps from the fitted M1 chain as ground truth (true WP known at every state by the same DP), refit M0 / M1 / M3 and a direct statistical estimator on replicate datasets of 55–880 maps, decompose MSE against the true WP into bias² and variance, compute the effective sample size of the observational map data, and check bootstrap CI coverage | the paper's bias–variance and ESS argument, measured for CS2 instead of assumed; the synthetic i.i.d. control already built is the truth = M0 special case |
| **Out-of-time, once, disclosed** | 27 maps / 564 round starts, 2026 side rate 0.543: report as fitted and with the side constant recalibrated, both labelled; no AUC headline on 27 outcomes |

## 6. Bias–variance: what "toy around with the state model" means here

- **M0 is the pure state model.** One parameter. It cannot overfit and it is wrong in a known
  direction: the pilot measured 5–10 points of over-confidence for the leader at 2–4 round margins.
- **M1 keeps the structure and adds the one dependence CS2 has that soccer does not**: money. About
  50 parameters, all interpretable (a 5×4 transition table per outcome and a 12-coefficient round
  model). This is where most of the bias should go.
- **M2 adds the football paper's "prior strength matters early, in-game performance later"** in the
  cheapest possible form: a Beta posterior on X's edge that is a function of (a, b) only, so the DP
  stays exact. Its single hyper-parameter n0 says how fast the score is believed.
- **M3 measures what is left.** If a flexible learner on top of the chain's logit cannot beat it by
  the MDE, the remaining bias is below what 220 maps can resolve, and the deep rungs are unlikely to
  help on log-loss.
- The learning curves make the trade visible: a rung whose out-of-fold loss keeps falling with more
  maps is variance-limited; one that is flat is bias-limited.

## 7. Build status

Built and run in-time (2024–25, 5-fold GroupKFold by match) by `src/models/map_wp.py`:
M0, M1, M2 (with n0 grid), M3, M4 chaining, the round-start battery with paired bootstrap,
calibration by margin and phase, structural checks, martingale increments and QV, the extreme-path
benchmark at round-start and per-second resolution, the synthetic negative control, the label-shift
control, and the learning curve. Results: `docs/study_round_to_map_results.md`.

The out-of-time scoring was run once after the design froze (`--holdout`; results §7 of the results doc).

Not built: M5a/b/c (Betty), the lagged-rank prior μ for M2 (needs the 2023 HLTV tables), the loss-bonus
streak state, an era-tracked side constant, series (Bo3) context.

## Sources

Brill, Yurko, Wyner, "Exploring the difficulty of estimating win probability: a simulation study", J. Quant.
Anal. Sports 22(1): 105–115, 2026, https://doi.org/10.1515/jqas-2024-0130 (code: github.com/snoopryan123/fourth_down,
`1_simulation/sim_v3`). Robberechts, Van Haaren, Davis, "A Bayesian Approach to In-Game Win Probability in Soccer", KDD 2021
([arXiv 1906.05029](https://arxiv.org/abs/1906.05029), [ACM DL](https://dl.acm.org/doi/10.1145/3447548.3467194),
[PDF](https://www.janvanhaaren.be/assets/papers/kdd-2021-win-probability.pdf)). Pipping & Wyner 2025/2026
(extreme-path benchmarks; see `notes_pathwise_calibration.md`).
