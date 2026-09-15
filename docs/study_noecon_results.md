# Study 3 — The no-economy ablation: what the non-economy pillars predict on their own

**Date:** 2026-09-12 · **Code:** `src/models/noecon_ablation.py`, figure `src/viz/noecon_figure.py` ·
**Outputs:** `outputs/noecon_ablation.csv` (in-time + paired CIs), `outputs/noecon_holdout.csv`
(out-of-time, one disclosed run), `outputs/noecon_timeprofile.csv`, `outputs/oof_noecon_{logreg,xgb}.parquet`,
`outputs/figures/paper/F12_noecon_ablation.png`, log `outputs/noecon_ablation.log`.
**Data:** canonical training table (476,595 × 135, 220 matches, base 0.445); primary 2026 table
(55,271 × 115, 27 matches, base 0.512). 5-fold GroupKFold by match; B=500 paired match-block bootstrap.

## Question and why it was open

Every published feature set contains `ECONOMY_COLS`, and the residual analysis measures what the other
pillars add *after* economy. This study asks the complementary question: **how far does each block get
alone, with economy absent, and how much of the production model's discrimination do the non-economy
pillars recover by themselves?**

Two definitional issues shape the design. First, the economy block is money **plus** combat state,
clock and score: `MONEY` = equipment value, economy class, armor, kits (7); `COMBAT` = players alive,
health, bomb-planted, time (6); `SCORE` = scores, score difference, round number (4). Second,
headcount leaks into the other pillars through columns that literally count alive players (per-zone
player counts, AWP-alive, `n_ct_near_bomb`, `n_ct_can_defuse`; 14 columns), and the buy leaks through
utility counts (grenades are bought). The ladder therefore splits the economy block, builds
"minus count-like columns" variants, and reports every set on the equal-alive subset, where headcount
carries no information.

## Results — in-time (out-of-fold)

Share = (AUC − 0.5) / (AUC_EB2 − 0.5): the fraction of the production model's above-chance
discrimination a set recovers on its own.

| Set (cols) | logistic AUC (95% CI) | XGBoost AUC | share of EB2 | paired ΔAUC vs EB2, logistic (95% CI) | equal-alive AUC | contested-AUC |
|---|---|---:|---:|---|---:|---:|
| **EB2** production (72) | **0.851** (0.844, 0.857) | 0.850 | 1.00 | — | 0.715 | 0.596 |
| everything except money (65) | 0.841 (0.834, 0.848) | 0.844 | 0.97 | −0.010 (−0.013, −0.007) | 0.695 | 0.579 |
| **A** economy block (17) | 0.847 (0.840, 0.853) | 0.845 | 0.99 | −0.004 (−0.006, −0.003) | 0.710 | 0.587 |
| all non-economy (55) | 0.822 (0.815, 0.828) | 0.830 | 0.92 | −0.029 (−0.033, −0.025) | 0.679 | 0.567 |
| Voronoi + tactical (43) | 0.818 (0.811, 0.824) | 0.828 | 0.91 | −0.033 (−0.037, −0.029) | 0.677 | 0.557 |
| all non-economy, no count columns (41) | 0.806 (0.799, 0.813) | 0.823 | 0.87 | −0.045 (−0.049, −0.040) | 0.677 | 0.550 |
| Voronoi + tactical, no count columns (30) | 0.801 (0.793, 0.808) | 0.820 | 0.86 | −0.050 (−0.055, −0.045) | 0.675 | 0.543 |
| money only (7) | 0.830 (0.822, 0.838) | 0.834 | 0.94 | −0.021 (−0.025, −0.016) | 0.693 | 0.536 |
| combat state only (6) | 0.797 (0.790, 0.804) | 0.800 | 0.85 | −0.054 (−0.060, −0.049) | 0.571 | 0.578 |
| Voronoi control only (9) | 0.723 (0.717, 0.729) | 0.745 | 0.64 | −0.128 (−0.134, −0.121) | 0.585 | 0.538 |
| bomb geometry + defuse race (18) | 0.686 (0.680, 0.691) | 0.695 | 0.53 | −0.165 (−0.172, −0.158) | 0.539 | 0.540 |
| clock only (1) | 0.555 | 0.550 | 0.16 | −0.296 | 0.535 | 0.548 |
| score + round (4) | 0.536 | 0.537 | 0.10 | −0.314 | 0.536 | 0.461 |

Every paired interval excludes zero. Post-plant AUC: bomb block alone 0.935 / 0.940 (logistic / XGBoost)
vs EB2 0.963 / 0.967 and economy 0.950 / 0.955.

## Results — out-of-time (2026, scored once)

| Set | logistic in-time → 2026 | XGBoost in-time → 2026 | 2026 ΔAUC vs A, logistic (95% CI) |
|---|---|---|---|
| EB2 | 0.851 → 0.847 | 0.850 → 0.850 | +0.005 (+0.001, +0.010) |
| everything except money | 0.841 → 0.841 | 0.844 → 0.844 | −0.001 (−0.009, +0.009) |
| A economy | 0.847 → 0.842 | 0.845 → 0.844 | — |
| all non-economy | 0.822 → 0.822 | 0.830 → 0.829 | −0.021 (−0.032, −0.009) |
| Voronoi + tactical | 0.818 → 0.815 | 0.828 → 0.825 | −0.027 (−0.038, −0.015) |
| money only | 0.830 → 0.823 | 0.834 → 0.829 | −0.019 (−0.031, −0.009) |
| combat state only | 0.797 → 0.787 | 0.800 → 0.799 | −0.054 (−0.070, −0.038) |
| Voronoi only | 0.723 → 0.703 | 0.745 → 0.732 | −0.140 (−0.163, −0.118) |
| bomb block only | 0.686 → 0.675 | 0.695 → 0.671 | −0.166 (−0.188, −0.140) |

Every non-economy set transfers within 0.01–0.02 of its in-time value; the ordering is unchanged. The
2026 contested-AUC values are noisier (27 matches) and are not interpreted here.

## Results — when each block becomes informative (logistic, AUC by seconds into the round)

| Set | 0–5 s | 5–10 | 10–20 | 20–40 | 40–60 | 60–90 | 90 s+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| EB2 | 0.691 | 0.690 | 0.713 | 0.780 | 0.844 | 0.898 | 0.963 |
| A economy | 0.693 | 0.694 | 0.714 | 0.779 | 0.842 | 0.892 | 0.953 |
| money only | 0.688 | 0.689 | 0.709 | 0.771 | 0.829 | 0.873 | 0.933 |
| combat state only | 0.481 | 0.482 | 0.563 | 0.693 | 0.799 | 0.875 | 0.949 |
| Voronoi only | 0.483 | 0.498 | 0.578 | 0.638 | 0.733 | 0.792 | 0.864 |
| bomb block only | 0.497 | 0.484 | 0.520 | 0.551 | 0.635 | 0.750 | 0.910 |
| all non-economy | 0.678 | 0.669 | 0.684 | 0.741 | 0.811 | 0.871 | 0.951 |

Bucket sizes: 24k, 24k, 49k, 96k, 91k, 113k, 79k snapshots. XGBoost profiles are within 0.02 of these
in every cell (see the CSV).

## Reading

1. **Money is the single strongest block and the only one informative at second zero.** Seven money
   columns alone give 0.830 (94% of EB2's above-chance AUC), and the money-only curve starts at 0.69
   at t=0 while every other block starts at chance. This is the mechanical reason the economy baseline
   dominates: it is the only information that exists before anyone moves.
2. **The non-economy pillars alone recover 92–95% of the production model's discrimination**
   (0.822 logistic / 0.830 XGBoost), but they do it *late*: at 0–10 s they sit at 0.67–0.68 (which is
   the buy leaking through utility counts and AWPs), and only after 40 s do they approach the economy
   block. Combat state alone ends the round at 0.949, above money (0.933), so by the last 30 s the
   round is read from who is alive, not from what was bought.
3. **The blocks are largely redundant with one another.** Dropping the money block from EB2 costs only
   0.010 (logistic) / 0.006 (XGBoost) AUC, and "everything except money" transfers to 2026 with no
   degradation (0.841 → 0.841). Utility counts, armor and AWP presence re-encode most of the buy, so
   "without economy" cannot be made money-free by column selection alone; this study quantifies the
   overlap rather than pretending it away.
4. **Headcount is worth about 0.017 AUC to the spatial block.** Voronoi + tactical falls from 0.818 to
   0.801 when the 14 count-like columns are removed (all non-economy: 0.822 → 0.806). On the
   equal-alive subset, where headcount is uninformative, every non-economy set sits at 0.68 against
   0.71 for the economy block and EB2, and combat state alone drops to 0.57: the "combat" signal is
   headcount, and once that is held fixed, health and time carry little.
5. **Voronoi control alone is a mid-strength, slow signal**: 0.72 pooled, 0.48 at t=0, 0.86 after 90 s,
   0.585 on equal-alive. It is not an economy proxy (it starts at chance), and it transfers (0.703 in
   2026). The bomb block alone is weak pooled (0.69) because it is silent before the plant, but reaches
   0.91 after 90 s and 0.935 post-plant, consistent with its role as the endgame feature.
6. **Contested-AUC does not move.** No set exceeds 0.60 on contested snapshots in-time; removing economy
   does not reveal hidden even-round signal. This is the ceiling result again.

## Caveats

- The `Score` set uses the pipeline's score columns, which are cumulative *side* wins rather than team
  scores (see `plan_study1_t0_prior.md`, P1). Score alone is at chance either way (0.536), so this
  does not change the reading, but the column should be fixed upstream before it is interpreted.
- "Money-free" sets still see the buy through utility counts and AWP presence; the 0–10 s values of the
  non-economy sets (0.67–0.68) measure exactly that leak.
- The 2026 evaluation is a single disclosed scoring of the touch-once holdout with models fitted on all
  training data; nothing was tuned on it.
- Time-bucket AUCs are computed on the pooled OOF predictions within each bucket (no refit), unlike
  `train_pipeline.py`'s time-window analysis, which refits per second; the two answer different
  questions (this one: how the full model's discrimination varies through the round).

## Paper placement

One paragraph after the model matrix ("How much of the model is economy?") citing three numbers: money
alone 0.83, all non-economy 0.82, everything-but-money 0.84 vs EB2 0.85; the time-profile figure (F12,
panel B) as the visual argument that economy dominates because it is the only information available
early, not because the spatial pillars are weak; the full table in the appendix beside Table B1.
