# Plan: round-start (t=0) win probability, match-level win probability, and the no-economy ablation

> **Superseded in part (2026-09-12, after the data upload and pilots).** The detailed, data-backed
> versions of Studies 1 and 2 are in [plan_study1_t0_prior.md](plan_study1_t0_prior.md) and
> [plan_study2_match_wp.md](plan_study2_match_wp.md); Study 3 was run (`src/models/noecon_ablation.py`,
> results in [study_noecon_results.md](study_noecon_results.md)). Corrections to this document that
> the pilots forced: (1) the "duration bias" argument in Study 1 is wrong — because
> `time_elapsed_sec` is a feature, the per-second model's first-snapshot value is already a
> one-row-per-round estimate and is calibrated at round level; the t=0 model should be a *stack* on
> that value, not a refit; (2) `ct_score`/`t_score` in the pipeline are cumulative side wins, not
> team scores, and must be rebuilt from round winners; (3) the round-level export's `bomb_planted`
> is a round outcome and must never enter a t=0 model; (4) the match-level DP must model economy
> carry-over between rounds (the i.i.d. version is over-confident for leaders by 5–10 points).

Three extensions requested by Prof. Wyner (Sept 2026). Each is scoped so it reuses the existing
pipeline, the round-level export (`exports/cs2_inferno_round_level_v1/`), and the FULL Benchmark
discipline (`docs/FULL_BENCHMARK.md`). None of the three needs the nav mesh or a re-parse: Studies 1
and 2 run on the round-level export plus the parsed `rounds` / `kills` channels; Study 3 runs on the
already-assembled training and 2026 tables.

Status of each question against what exists today:

| Question | Exists? | Closest existing artifact |
|---|---|---|
| t=0 round WP as its own model | **No.** Only implicit: the per-second model's first snapshot, used as p0 in `pathwise_calibration.py` | `pathwise_calibration.py` (p0), time-window AUC at 5..25 s in `train_pipeline.py` |
| Match-level WP | **No.** Nothing beyond `ct_score`/`t_score` as round-model inputs | `exports/.../rounds.csv` (score state, sides, `match_winner`) |
| Non-economy features alone | **No.** Every `FEATURE_SETS` entry includes `ECONOMY_COLS` | Study 1 residual/FWL analysis (economy partialled out); contested-AUC slice |

> Touch-once reminder. All three studies evaluate on the 2026 holdout as **separate, disclosed
> evaluations**: cross-validation on the 220 training matches drives every design choice; the 2026
> set is scored once per study, after the design is frozen, and reported as such.

---

## Study 1 — Round-start win probability (t=0 prior)

### Definition
`p0(r) = P(CT wins round r | information available at freeze-end of round r)`.
Trained **one row per round** (4,866 rounds), not per snapshot. This matters: the snapshot table's
base rate is 0.445 but the round-level base rate is 0.487, because rounds the CTs lose run longer.
A t=0 model trained on snapshots inherits that duration bias.

### Feature ladder (each rung is a nested set; report the marginal lift with paired CIs)
1. **Format / side context** — side, `round_num`, `half`, `is_pistol_round`, overtime flag,
   rounds-to-win for each team, `ct_score`, `t_score`, `score_diff`.
2. **Buy state** (from `rounds.csv` base state and `round_players.csv`): equipment values, economy
   class, armor, helmets, defuse kits, grenade counts by type, AWP presence, per-player loadout
   spread (e.g. number of rifles vs pistols), money-in-bank if recoverable from ticks.
3. **Recent history** (lag within match from `rounds.csv`): previous-round winner, previous-round
   `reason` (elimination / defuse / explode / time), consecutive losses per team (loss-bonus level),
   previous-round equipment (did the loser save?), current win streak, whether the last round was a
   pistol.
4. **Team prior** (must be **lagged**: only information dated before the match): HLTV rank at the
   previous snapshot (`team_rankings.csv`), rank difference, side-specific team rating built from
   the roster's side-split HLTV stats (`rating_ct` vs `rating_t` means, previous season), Inferno
   win rate computed only from earlier matches in the corpus (time-ordered, leave-future-out).
5. **Player prior** (lagged): mean previous-season rating over the ten starters, roster-change
   indicator, opening-duel scores (`opening_ct`/`opening_t` already scraped).
6. **Series context**: map number parsed from `match_id` (`m1`/`m2`/`m3`). Map veto (who picked
   Inferno) is not in the data; scrape from HLTV only if rung 6 shows promise.

**Hypothesis (the interesting one):** priors should matter *more* at t=0 than in the per-second
model. The firepower negative showed a skill prior adds nothing once rich in-round state is present;
at t=0 there is no in-round state, so this is where a prior has its best chance. Either outcome is
reportable and complements the existing firepower result.

### Verification
- **In-time:** 5-fold GroupKFold by match on rounds. AUC, log-loss, Brier, BSS, ECE, calibration
  slope/intercept. Match-level block bootstrap (B=500), paired ΔAUC/Δlog-loss per rung.
- **Out-of-time (disclosed, once):** ~600 round-starts from the 27 2026 matches, with all priors
  lagged (2025 stats, 2025-end rank). Report calibration under the 0.487 → ~0.51 side drift.
- **Consistency with the per-second model (the martingale checks Prof. Wyner's group cares about):**
  1. *Anchor test:* the per-second model's first-snapshot prediction should equal p0 on average
     within p0 bins (no systematic jump at t=0+).
  2. *Quadratic-variation identity:* for a calibrated martingale, E[(Y − p0)²] = E[p0(1−p0)] and,
     with orthogonal increments, E[Σ Δp_t²] = E[(Y − p0)²]. Compare the per-second curve's realised
     total movement against the t=0 model's implied variance, per p0 bin. Excess movement = the
     curve over-reacts; deficit = under-reacts. This extends `notes_pathwise_calibration.md`.
  3. *Re-run the extreme-path benchmark* (`pathwise_calibration.py`) with p0 taken from the t=0
     model instead of the first snapshot; report whether D_upper and the comeback rates move.
  4. *Offset model:* refit the per-second EB2 model with `logit(p0)` as an offset (the Study 1
     stacked-logit pattern). The per-second model then becomes an explicit update of the prior.
     Report its AUC/ECE by time window (5..25 s), where the current model is weakest (0.69 → 0.76).
- **Opening-duel event study ("opening loss"):** from the `kills` channel take the first kill of
  each round. Measure the WP jump at that kill, conditional on p0, which side lost the player, where
  (zone), and when. Check the jump is calibrated: among rounds with p0 in a bin and a T-side opening
  kill, the realised CT win rate should match the post-kill WP. Produces the "starting point, then
  the first event moves it" figure the request describes. Opening-duel win rate is also a candidate
  team prior (rung 5).

### Deliverables
`src/models/round_start_wp.py` (table build + ladder + CIs), `outputs/t0_ladder.csv`,
`outputs/t0_martingale_checks.csv`, figure: ladder + opening-kill increment calibration. Paper: a
Results subsection "The round-start prior", and p0 wording updated in the pathwise subsection.

### Effort
About one week once the data bundle is restored (0.5 d table, 1 d ladder+CIs, 0.5 d out-of-time,
1 d martingale checks, 1 d opening-kill study, 1 d figure/text).

---

## Study 2 — Match-level win probability

Here a "match" is **one map** (first to 13, MR12; overtime MR3, first to 4 per period). Series
(best-of-3) context is out of scope: the corpus holds one map per series.

### Three model routes (build A first; B is the check; C is the deliverable)

**A. Compositional score-state model (primary).** State = (rounds won by team X, rounds won by
team Y, which side X plays now, round index → half / overtime period). Backward induction over the
score lattice gives `P_match(X wins | state)` from per-round win probabilities. Three levels:
- A0: constant side-specific per-round probability (Inferno CT rate by half). The classical
  baseline; every later rung must beat it.
- A1: current round's probability from the **t=0 model (Study 1)**; future rounds from a
  team-and-side prior (A0 rates shifted by the team prior). One-step economy awareness.
- A2 (stretch): Monte-Carlo roll-out with an **economy transition model** (empirical
  P(next economy class | outcome, current class, loss-bonus level)) so pistol-round and eco-round
  effects propagate beyond one round. Verify the transition model on held-out matches first.

**B. Direct model (check).** Logistic/GBM on `P(team wins map | round-start state + score + priors)`.
Only 220 independent outcomes across 4,866 correlated rows, so expect overfitting; use it as a
residual correction on top of A (A's logit as offset), not as a stand-alone model.

**C. Per-second match WP (the broadcast object).**
`P_match(t) = p_round(t)·P_match(score+1) + (1 − p_round(t))·P_match(score, opp+1)` chains the
per-second EB2 round model to A. This yields a match-long WP path (24+ rounds × ~90 s), which is the
most natural object for the Pipping–Wyner extreme-path benchmark: many more decision points than a
single round, so the discrete-conservatism caveat in `notes_pathwise_calibration.md` weakens.

### Team modeling (in order of cost; all lagged, coverage-monitored)
1. Side-specific constants (map-level).
2. HLTV rank difference from `team_rankings.csv` (yearly snapshot; 2025 snapshot for 2026 matches).
3. Roster-derived side-specific strength from `player_stats_sided.csv` (previous season): mean
   `rating_ct` for the CT roster, mean `rating_t` for the T roster; roster-change indicator.
4. Inferno-specific team form from earlier corpus matches only (time-ordered).
5. Elo/Glicko/Bradley–Terry on full HLTV results (needs a results scrape; only if 2–4 leave signal
   on the table). Team dummies are not viable: sampling caps give ≤12 maps per team and 47 ranked
   teams.

The firepower lesson applies directly: every prior is an inference-time dependency. Lag it, monitor
coverage, and evaluate out-of-time. The expectation is the opposite of the round-model result: with
no in-match state at 0-0, the prior is *all* the information, so it should carry weight there and
fade as the score state accumulates. Report the prior's weight as a function of round index.

### Verification
- **In-time:** 5-fold GroupKFold by match; predict `P_match` at every round start (A, B) and every
  second (C). Log-loss / Brier / AUC at round-start granularity with match-block bootstrap; paired
  CIs across the ladder A0 → A1 → +team prior → +B residual → A2.
- **Calibration by score state:** reliability at 0-0, 12-12, 12-9, 6-6 etc. Monotonicity in score
  difference. At 0-0 the output must equal the pre-match prior; at 12-12 it must be near 0.5 shifted
  by side and economy.
- **Martingale / pathwise at match level:** per-round increments mean-zero given the prior;
  realised quadratic variation vs p0(1−p0); extreme-path benchmark on the match path (write-off
  frequency of eventual winners at ≤5/10/20%, D_upper).
- **Economy carry-over ablation:** does A1 beat A0 in log-loss specifically after pistol rounds and
  eco rounds? That is where the one-step economy should pay.
- **Out-of-time (disclosed, once):** 27 maps. Do **not** headline a 27-outcome AUC; report
  round-start log-loss/Brier, calibration slope/intercept, and the pathwise test.

### Data
All from the round-level export plus the parsed `rounds` channel: `ct_score`/`t_score`, sides per
round, `reason`, `match_winner`, overtime present (`n_rounds` up to 42). Overtime rules need one
explicit implementation and a unit test. Team and player priors from `configs/`.

### Deliverables
`src/models/match_wp.py` (score DP + priors + per-second chaining), `outputs/match_wp_ladder.csv`,
`outputs/match_pathwise.csv`, figures: prior-weight vs round index; match WP path example with the
round-level curve; extreme-path benchmark at match level. Paper: new section "From round to match".

### Effort
1.5–2 weeks: DP + OT rules 1 d, priors 1–2 d, per-second chaining 1 d, battery 2 d, match-level
pathwise 1 d, figures/text 2 d. Stretch A2 adds 2–3 d; Elo scrape adds 2–3 d.

---

## Study 3 — How the non-economy pillars do without economy

### Why it is not already answered
Study 1 (residual analysis) answers "what do the spatial pillars add *after* economy". This asks
"how far do they get *alone*", which is the question a reader asks when told economy dominates.

### The definitional trap
`ECONOMY_COLS` = money (`*_equipment_value`, `*_economy_class`, `*_armor_total`, `ct_defuse_kits`)
**plus** combat state (`*_players_alive`, `*_health_total`, `bomb_planted`) **plus** clock and score
(`time_elapsed_sec`, `round_num`, `ct_score`, `t_score`, `score_diff`). Dropping the whole block does
not remove headcount: per-zone player counts in `TACTICAL_COLS` sum to players alive, firepower
sums scale with it, `n_ct_near_bomb` and entropy carry it too. The ablation must make that explicit.

### Ladder (new `FEATURE_SETS` entries; all runs through `train_pipeline.py` with `--bootstrap 500`)
| Set | Contents | Question it answers |
|---|---|---|
| `N0` | `time_elapsed_sec` only | clock-only floor |
| `Money` | equipment value, economy class, armor, kits | money alone |
| `Combat` | alive, health, bomb_planted, time | combat state alone (no money) |
| `Spatial` | Voronoi 9 cols only | can territory alone rank rounds? |
| `SpatialT` | Voronoi + tactical **minus count-type columns** (per-zone player counts, `n_ct_near_bomb`, AWP alive counts) | spatial + utility without headcount leakage |
| `SpatialT+` | Voronoi + full tactical | same, with the headcount leak, to size the leak |
| `Bomb` | bomb geometry + bomb-live + defuse-race only | post-plant geometry alone |
| `NoMoney` | EB2 minus the money columns | everything except money |
| `NoEcon` | EB2 minus all of `ECONOMY_COLS` | everything except the whole block |

Report for each: AUC, contested-AUC, log-loss, ECE, share of the full EB2 lift recovered, and the
**time-window profile** (AUC at 5/10/15/20/25 s and later). Expected shape: `Money` is flat in time
(fixed at buy) while `Spatial` starts near 0.5 and rises as positions become informative. That curve
is also a consistency check for Study 1: the no-economy model at t→0 must sit at chance.

Run out-of-time too (disclosed, once): the non-economy sets are demo-derived and should transfer.

### Effort
About one day: feature-set entries, one pipeline run per model (logreg, xgb), the time-window and
conditional slices, one figure. Needs only `data/training_dataset.parquet` and
`data/test_dataset_2026.parquet`.

---

## Sequencing and prerequisites

1. **Restore the environment first** (this machine has code only): Python 3.11+, the data bundle
   (`training_dataset.parquet`, 2026 tables, parsed `rounds`/`kills` channels, or the Wyner
   export CSVs). No nav mesh or demos are needed for any of the three studies.
2. Study 3 (1 day) → Study 1 (1 week) → Study 2 (2 weeks). Study 1's t=0 model is an input to
   Study 2's A1 rung, and Study 3's time-window curve is a consistency check for Study 1.
3. Paper impact: Study 3 tightens the existing "economy dominates" claim; Study 1 replaces the
   first-snapshot p0 with a principled prior and adds the opening-kill increment analysis; Study 2
   is a new section and the natural home for a match-level extreme-path benchmark. None of it
   changes the shipped round model (EB2, no firepower).
