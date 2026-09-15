# The round-start (t = 0) win probability

**Question.** What is the win probability of a round before anyone moves, and is the live per-second
curve consistent with that starting point?

**Setup.** t = 0 is freeze-end: buys complete, nobody has moved. One row per round, 4,866 rounds from 220
maps (2024–25), 5-fold cross-validation grouped by map, paired match-level bootstrap. The 2026 holdout was
not used. Priors come in two constructions: same-year HLTV stats (leaky, an upper bound) and previous-season
stats (honest; undefined for 2024 because no 2023 tables exist).

**Result 1: the buy is the whole signal; priors add almost nothing.**

| Round-level model | AUC | ΔAUC vs buy state (95% CI) |
|---|---:|---|
| format, side, score | 0.536 | |
| + buy state (equipment, armour, kits, utility, AWPs) | 0.682 | reference |
| + recent history (previous result, loss streak) | 0.687 | +0.004 (+0.000, +0.009) |
| + team and player priors, same-year (leaky) | 0.692 | +0.010 (+0.004, +0.016) |
| + team and player priors, lagged (honest) | 0.688 | +0.005 (−0.000, +0.012) |
| priors alone, lagged | 0.504 | at chance |
| stack: per-second model's first value + history + lagged priors | 0.695 | +0.013 (+0.005, +0.021) |

A single round is close to a coin flip before it starts, and external strength information moves it by
at most one point of AUC. This repeats the paper's firepower finding at the one moment where a prior has
no in-round state to compete with. The right construction is not a new model but a stack on the
per-second model's own first value.

**Result 2: the prior is calibrated, the path is not a martingale.** Anchored at the prior,
E[(Y − p0)²] / E[p0(1 − p0)] = 1.02 (calibrated start). But the summed squared movement of the per-second
curve, including the final jump to the outcome, is 1.31 times the terminal variance, for the production
model and for the economy-only model alike. The excess is short-lag mean reversion (increment
autocorrelation −0.04 to −0.01 at 1–5 s, zero beyond 10 s). Shrinking all increments by 0.85 restores the
identity exactly but costs 0.011 log-loss, so the extra movement is informative: the model is calibrated at
each instant but not consistent with its own history. The extreme-path test did not see this because it
looks at the path maximum, not its total variation. Remedy: a path-aware forecast (lagged values as inputs,
or the sequence models scored on this identity).

**Result 3: the first kill and the stronger team.** The first kill moves the curve by about 0.2 in even
states; the reaction to a CT death is calibrated (0.216 predicted, 0.234 realised), the reaction to a T
death overshoots by 5 points (0.728 vs 0.678). At equal buys the higher-ranked team wins 53–59% of rounds
when the rank gap is six or more, but the deployed curve gives the favourite 0.50–0.51 because it has no
team features; the stack with honest lagged priors reaches only 0.51–0.52, the leaky same-year stack
0.53–0.56.

**Status and caveats.** In-time only; not scored on 2026. Lagged priors rest on the 2025 matches alone
until 2023 HLTV tables are scraped. Round-level paired CIs are about ±0.006 AUC, so effects below that are
undetectable, not absent.
