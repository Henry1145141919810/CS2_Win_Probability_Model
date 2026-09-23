# From round win probability to map win probability

**The question.** The paper's model gives P(CT wins the current round) at every second. A viewer or a
bettor cares about the map: P(team X wins the map | everything known now). This memo builds that
quantity, tests it, and reports what works. "Map" means one Inferno map (first to 13 rounds; 12-12
goes to overtime in periods of six rounds, first to 4). Best-of-three series are out of scope.

**Data and conventions.** 220 maps from 2024-25 (4,866 round starts) for fitting, 5-fold
cross-validation grouped by map; 27 maps from 2026 (564 round starts) held out and scored once. Team
X is the team that starts on CT; it wins 49.1% of maps, so the label is balanced and symmetric. The
state at the start of a round is the score (a, b) = rounds won by X and by Y, the side each team is on
(fixed by the round number), and the two teams' economy.

## How the model works

The idea comes from Brill, Yurko and Wyner (JQAS 2026): instead of fitting a classifier from game
state to the map outcome (a *statistical* model, which is high-variance because every round of a map
shares one outcome), fit the *transitions* of the game at the round level, where there are thousands of
observations, and propagate them through the rules. Concretely:

1. A per-round model gives p(X wins the next round | state).
2. A dynamic program walks backward over the score lattice: V(a, b) = p·V(a+1, b) + (1 − p)·V(a, b+1),
   with V = 1 or 0 once a team has clinched. V(a, b) is the map win probability at that score.
3. Richer rungs add state to step 1 so the transitions are more realistic, one assumption at a time.

The rungs:

- **M0, the pure state model.** Every round is an independent coin flip with one parameter, the
  CT-side win rate (0.487 in 2024-25). This is the low-variance, high-bias baseline of the JQAS paper.
- **M1, the economy chain.** Adds what CS2 has and coin flips do not: money carries over. Each team's
  equipment is put in one of four tiers (eco, force, partial, full). The per-round model is a logistic
  regression on side, pistol round, and the two tiers (12 coefficients); a transition table, estimated
  from the data, says how a team's tier moves after it wins or loses a round (about 40 numbers). Pistol
  rounds (1 and 13) and overtime rounds are economy resets and are handled as such. About 50 parameters,
  all interpretable.
- **M2, latent team strength.** M1 plus a Beta posterior on X's true edge, updated from the score: a
  team that has won more rounds is believed to be better. One hyper-parameter (how fast the score is
  believed), chosen by inner cross-validation.
- **M3, the residual check.** Gradient boosting fitted on top of M2's output plus the raw state. If a
  flexible learner cannot improve on the chain, the chain has captured what 220 maps can support.
- **M4, per-second chaining.** The live object. At every second, P_map = p_t · V(a+1, b) + (1 − p_t) ·
  V(a, b+1), where p_t is the paper's per-second round probability and V comes from the chain. This is
  what a broadcast would show.

## What the ladder shows

Round-start log-loss (lower is better; 0.693 is a coin flip):

| rung | in-time | 2026 (27 maps, one disclosed scoring) | 2026 calibration slope |
|---|---:|---:|---:|
| M0 i.i.d. rounds, one side-rate parameter | 0.5673 | 0.5487 | 0.80 |
| M1 economy chain, ~50 parameters | 0.5624 | 0.5363 | 0.995 |
| M2 + latent team strength | 0.5633 | 0.5366 | 0.95 |
| M3 gradient boosting on top of the chain | 0.5750 | 0.5332 | 1.17 |
| M0 with the side constant re-estimated on 2026 (diagnostic, not a model) | | 0.5297 | 0.91 |
| M4 per-second chaining (scored per second, not per round start) | 0.5566 | 0.5214 | 0.96 |

Reading the table:

- The state model M0 is a strong floor but over-confident: slope 0.80, and at a margin of three rounds
  it says 0.82 where the leader actually wins 0.72. The reason is money: the trailing team's economy
  recovers, so leads are less safe than independent coin flips imply.
- M1 fixes most of that. It removes the bias when X trails, halves it when X leads (0.78 at +3), and its
  2026 slope is 0.995. The gain over M0 is 0.005 in-time and 0.012 on 2026, the right direction and size.
- M2 adds nothing: the score carries no strength information beyond what the economy explains.
- M3 is worse in-time by 0.013 with a confidence interval that excludes zero: with 220 map outcomes a
  flexible learner fits noise, exactly the JQAS prediction. Its 2026 number looks good but is noise.
- The diagnostic row is the largest effect in the table. Re-estimating M0's one constant on 2026 (the CT
  round rate moved from 0.487 to 0.543) beats every honest rung by 0.019. A deployed map model needs
  its side rate tracked per era.
- Nothing in the 2026 column is significant: on 27 maps the intervals are about ±0.05. The 2026 values
  are lower than in-time because those maps were more lopsided, not because the models improved.

## Checks

- **Controls.** Shuffling map winners across maps pushes every rung to log-loss above 0.89. On synthetic
  maps generated from M0, where no structure exists, M1 finds none (Δ +0.002, CI includes zero) while
  M3 invents some. Learning curves are flat for M0-M2 from 110 maps and still falling for M3 at 220:
  the structural rungs are bias-limited, the boosted one variance-limited.
- **Format.** The dynamic program gives 0.50 at 0-0, 0.52 at 12-12, is monotone in wins, and reproduces
  the curated winner on every map whose clinching round survived parsing.
- **Trajectory honesty.** The round-start path of the chain behaves as a martingale (squared movement
  1.07 times terminal variance; the eventual winner is written off less often than the martingale law
  allows). The per-second path (M4) does not: 40% excess movement in-time (CI 1.26-1.55), inherited from
  the round model's own 30% excess and amplified by the value gap between winning and losing a round. On
  the 27 maps of 2026 the excess does not show (1.09); unresolved.
- **A trap found on the way.** Fitting an overtime-specific coefficient must be avoided: X won 53% of
  290 overtime rounds by chance, one coefficient turned that into 0.60 at 12-12, and backward induction
  leaked it into every near-tied state. Overtime is now treated as full-buy regulation rounds.

## Built and not built

Built and run: M0-M4, the battery above, the controls, learning curves, and the single disclosed 2026
scoring. Not built: the loss-bonus streak in the economy state (for the residual leading-side bias), a
lagged-rank prior for M2 (needs 2023 HLTV tables), the known-truth simulation from the JQAS paper
(simulate maps from the fitted chain, decompose each estimator's error into bias² and variance, effective
sample size, bootstrap coverage), sequence and generative models on the cluster, and series context.
