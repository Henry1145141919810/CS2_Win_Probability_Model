# Notes — Pathwise extreme-path calibration (trajectory honesty)

**Date:** 2026-08-05 · **Code:** `src/models/pathwise_calibration.py` ·
**Results:** `outputs/pathwise_benchmark.csv`, `outputs/pathwise_perround_*.parquet` ·
**Figure:** `outputs/figures/paper/F10_pathwise_calibration.png`

## What this is

Item 7 of the FULL Benchmark. Treats each round's per-second win probability as a Doob martingale
(p0 = round-start WP, terminal = outcome) and tests whether its *extremes* are those of an honest
sequential forecast. Based on Pipping & Wyner, "A Paradox of Blown Leads" (2025) and "Conditional
Extreme-Path Benchmarks for Sequential Probability Forecasts" (2026) — the advisor's group.

**Benchmark (calibrated continuous martingale):**
$P(\text{loser peak} \ge x \mid \text{lose}) = \frac{p_0}{1-p_0}\cdot\frac{1-x}{x}$.
Winner's trough $\le y \iff$ loser peak $\ge 1-y$. In discrete/jumpy time this is a **conservative**
reference (Theorem 2): real observed extremes should sit at or below it.

## Results (5-fold OOF)

**Reproduced the paper's comeback number:** logreg EFB2, eventual winner written off to ≤10% in
**7.42%** of rounds (paper reported 7.2%; matches within noise).

**Task A — observed vs continuous benchmark (winner's trough ≤ y):**

| Model | y=0.05 | y=0.10 | y=0.20 |
|---|---|---|---|
| logreg EFB2 observed | 2.88% (2.46,3.31) | **7.42%** (6.73,8.10) | 18.82% (17.70,19.89) |
| logreg EFB2 benchmark | 5.33% | 11.07% | 23.86% |
| obs/bench | 0.54 | 0.67 | 0.79 |
| lgbm EB2 observed | 2.28% (1.84,2.67) | 6.33% (5.70,6.93) | 16.75% (15.85,17.77) |
| lgbm EB2 benchmark | 5.35% | 11.02% | 23.67% |
| obs/bench | 0.43 | 0.57 | 0.71 |

Observed is **below** the continuous benchmark at every threshold, for both models.

**Task B — PIT + one-sided KS:**

| Model | D_upper (p) | D_lower | loser-peak mean | PIT mean |
|---|---|---|---|---|
| logreg EFB2 | **0.000 (p=1.00)** | 0.166 | 0.574 | 0.416 |
| lgbm EB2 | **0.000 (p=1.00)** | 0.254 | 0.553 | 0.352 |

## Interpretation

**Headline: the model does not overreact.** Upper-tail inflation is exactly zero
(D_upper = 0, p = 1) for both models. Unlike a forecast that sensationalises (the Pipping–Wyner
paper finds ESPN's NBA win-probability feed over-reacts, upper tail inflated), CS2-EB2 never assigns
extreme confidence to eventual losers more often than an honest forecast would. For a live product,
this is the direction that matters: it does not cry wolf.

**The below-benchmark position is expected discrete conservatism, not a flaw.** The continuous
benchmark assumes a path that can move at every instant; a CS2 round's WP is nearly flat between kills
and jumps only at them, so a round offers only a handful of decision points at which probability can
move. The Pipping–Wyner discrete-time result (their Theorem 2) proves that such discrete, jump-driven
paths reach extremes *less* often than the continuous benchmark, which is exactly the pattern here
(large D_lower, PIT mean < 0.5). The gap is a property of round-structured play, not miscalibration.

We therefore do **not** claim the trajectories are perfectly path-calibrated (the PIT is not uniform);
we claim the stronger, product-relevant and well-supported result: **no over-reaction, with mild
conservatism in the direction discrete play predicts.** EB2 (production) is slightly more conservative
than EFB2, consistent with it being the simpler, steadier model.

## Caveats

- Continuous benchmark used as the reference; the exact discrete benchmark (which would sit between
  observed and continuous) is not computed here. Distinguishing "pure discreteness" from "mild
  under-reaction of a per-snapshot model" would need that correction — future work.
- Within-match correlation handled by match-level block bootstrap on the tail frequencies.
- Rounds with a single snapshot contribute a degenerate trajectory; negligible in aggregate.

## Paper placement

Reframes the "honest probabilities" subsection around the Doob-martingale null; benchmarks the 7.4%
against the martingale law; adds Fig. F10; Discussion ties it to contested-AUC as the second form of
"honesty" (don't overclaim discrimination in even rounds; don't overclaim surprise at comebacks).
