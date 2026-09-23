# The FULL Benchmark

The complete evaluation suite. **Every model or feature change must be run through all seven groups,
not a subset.** A change can raise AUC yet break calibration, out-of-time transfer, or trajectory
honesty; only the full suite catches every failure mode. (Firepower looked best in-sample and
collapsed out-of-time; that is why nothing ships on a partial benchmark.)

All in-time numbers are 5-fold GroupKFold out-of-fold (never split a match). All confidence intervals
are B=500 match-level block bootstrap unless stated. Base rate P(CT win) = 0.445 (train), 0.512 (2026).

---

## Group 1 — Discrimination and core scores (the model x feature-set matrix)

Run every architecture x every valuable feature set. Code: `src/models/appendix_metrics.py`,
`train_pipeline.py`, `model_report.py`.

| Metric | Meaning | How to get | How to interpret |
|---|---|---|---|
| **AUC** | Probability the model ranks a random CT-win snapshot above a random T-win one. Pure discrimination, ignores calibration. | `roc_auc_score(y, oof_p)` | 0.5 = chance, 1.0 = perfect. Headline ~0.85, but this is inflated by lopsided snapshots (see Group 6). Judge on contested-AUC. |
| **Log-loss** | Proper scoring rule; punishes confident wrong calls heavily. | `log_loss(y, p)` | Lower is better (~0.46). Sensitive to over-confidence. |
| **Brier** | Mean squared error of the probability. Proper scoring rule. | `brier_score_loss(y, p)` | Lower is better (~0.155). Decomposes into reliability/resolution/uncertainty (Group 4). |
| **BSS (Brier Skill Score)** | Brier improvement over always predicting the base rate. | `1 - brier/(base*(1-base))` | Higher is better (~0.37). 0 = no better than base rate. |

Report the full grid (5 classical + TCN + Transformer + GAT + ensemble) x (A, E, EB2, EFB2, ...).

---

## Group 2 — Uncertainty and significance

A difference is real only if its interval excludes zero. Code: `block_bootstrap` in
`train_pipeline.py`.

| Test | Meaning | How to get | How to interpret |
|---|---|---|---|
| **Match-level block bootstrap (B=500)** | Resample whole matches (the independent unit) with replacement, recompute the metric each time. | draw matches -> recompute -> 2.5/97.5 percentiles | Gives a 95% CI on any metric. Treating snapshots as independent would shrink CIs ~10x and fake significance. |
| **Paired delta-AUC CI** | The CI of the *difference* between two models/sets on the same folds. | bootstrap the per-resample difference | Excludes 0 = significant. NOTE: overlapping *marginal* CIs do NOT imply no difference; always test the paired difference. |
| **DeLong test** | Analytic significance test for two correlated AUCs. | DeLong p-value | p < 0.05 = AUCs differ. Corroborates the bootstrap. |

---

## Group 3 — Interpretation (what the model uses)

Code: SHAP/permutation in the viz + eval scripts.

| Tool | Meaning | How to get | How to interpret |
|---|---|---|---|
| **SHAP dependence** | For one feature, how its value pushes the prediction up/down, per snapshot. | `shap` values, plot feature vs SHAP | Shape matters: a *step* = a hard learned boundary (e.g. defuse feasibility at 0); a *smooth ramp* = a graded/soft signal; integer *clusters* = a count in disguise (the firepower confound). |
| **SHAP beeswarm** | Global feature attribution across all snapshots. | `shap` summary | Ranks features by total impact; shows direction and spread. |
| **Permutation importance** | How much AUC drops when one feature is shuffled. | shuffle col -> re-score | Big drop = model relies on it. Beware collinearity: it ranks a proxy above the thing it proxies (rating-sum ranked #1 but was a player counter). Pair with the residual analysis (Group 6). |
| **Logistic coefficients** | The fitted linear model, directly readable. | standardized coefs / odds ratios | Sign and size per SD; the most transparent view of the linear model. |

---

## Group 4 — Calibration (are the probabilities honest?)

The output *is* the product, so calibration matters as much as discrimination. Code:
`src/models/extended_metrics.py`; figures `calibration.png`, `calibration_over_time.png`.

| Metric | Meaning | How to get | How to interpret |
|---|---|---|---|
| **Reliability diagram** | Observed win rate vs predicted probability, binned. | bin p, plot mean-p vs mean-y with CIs | On the diagonal (within CI) = honest. Bows above/below = under/over-confident. |
| **ECE (Expected Calibration Error)** | Average gap between predicted and observed across bins. | 10-bin weighted \|p - y\| | < 0.02 = well calibrated. Sensitive to bin placement, so also report the two below. |
| **Adaptive ECE / KS-cal** | Binning-free calibration error (equal-mass bins / max CDF gap). | equal-mass ECE; KS on (p,y) | Confirms ECE isn't a binning artifact. |
| **Calibration slope + intercept** | Logistic recalibration fit of y on logit(p). | fit y ~ logit(p) | Slope ~1 and intercept ~0 = calibrated. Intercept < 0 = over-confident in CTs (the broken-firepower signature, -0.36); slope < 1 = over-extreme. |
| **Murphy / Brier decomposition** | Splits Brier into reliability (calibration), resolution (discrimination), uncertainty (base). | Murphy partition | A good feature adds *resolution* without hurting *reliability*. |
| **Calibration-over-time** | ECE by seconds into the round. | ECE per time bucket | Catches a model honest on average but dishonest at a phase. Should stay near/under 0.02 throughout. |

---

## Group 5 — Out-of-time generalization (the 2026 holdout)

The single most important gate. Cross-validation cannot see era shift. Code:
`src/models/holdout_2026.py`. **Touch-once**: any re-run is a separate, disclosed evaluation.

| Metric | Meaning | How to get | How to interpret |
|---|---|---|---|
| **In-time vs out-of-time AUC** | Train on 2024-25, predict 27 unseen 2026 matches. | fit all train -> predict 2026 | in ~ out = transfers (demo-derived pillars). Large drop = overfit or an inference-time data dependency (firepower: 0.8519 -> 0.8236). |
| **Out-of-time calibration (slope/intercept, ECE)** | Does the probability stay honest under drift? | same on 2026 | Benign base-rate drift = intercept +0.11 (fixable). A collapse = intercept -0.36, ECE 4x (the firepower failure). |
| **Two variants for any external/skill prior** | same-year (leaky) AND lagged (leak-free, deployment-real). | `FIREPOWER_YEAR_LAG` rebuild | If a prior fails to beat the demo-only model under *both* (and the leaky best case), it is a rigorous negative, not a plumbing bug. |

---

## Group 6 — Beyond-economy signal and the contested-round ceiling

Where the real signal lives, and where it runs out. Code: `residual_analysis.py`,
`contested_study.py`; figures F7, F8.

| Analysis | Meaning | How to get | How to interpret |
|---|---|---|---|
| **Contested-AUC** | AUC restricted to even rounds (equal players alive AND \|equip diff\| <= $1500). | subset then AUC | The honest ceiling. ~0.58 = near chance; the headline 0.85 is carried by lopsided snapshots. Report this, not just pooled AUC. |
| **Residual (FWL / offset) analysis** | Signal a pillar adds *after* partialling economy out. | economy logit as offset; ΔAUC | Paired ΔAUC CI excludes 0 = genuinely new information. FWL economy-R2 low = orthogonal (defuse race, R2 0.15); high = economy in disguise (dist-to-bomb, R2 0.69). |
| **Information-saturation curve** | Contested-AUC vs representation richness (economy -> spatial -> deep). | contested-AUC per feature set/model | Flat (~0.585) = a real ceiling no representation breaks. |
| **Bayes-error matching (model-free)** | Among near-identical even states from *other* matches, how often do outcomes disagree? | kNN, same-match excluded | disagree ~0.5 and oracle-AUC ~0.51 (5v5-even) = irreducibly random. A man-advantage control returns oracle-AUC 0.83, validating the estimator. Separates aleatoric from recoverable. |

---

## Group 7 — Trajectory honesty (pathwise extreme-path calibration)

Are the comebacks and write-offs as dramatic as they look? Based on Pipping & Wyner (advisor's group).
Code: `src/models/pathwise_calibration.py`; figure F10; notes `docs/studies/notes_pathwise_calibration.md`.

| Test | Meaning | How to get | How to interpret |
|---|---|---|---|
| **Winner's-trough / loser's-peak vs martingale benchmark** | How often the eventual winner is written off to <= y, vs the calibrated-martingale law P(loser peak >= x) = (p0/(1-p0))((1-x)/x). | per-round trough; compare observed vs benchmark, thresholds 5/10/20% | Observed >> benchmark = the model over-dramatises (over-reacts). Observed <= benchmark = honest / conservative. Ours: 7.4% vs 11% at <=10%, below at every threshold. |
| **PIT + one-sided KS (D_upper)** | Transform each round's extreme through the benchmark CDF; test for excess extreme values. | PIT U_i; D_upper = sup(t - Fhat(t)) | D_upper > 0 significant = upper-tail inflation = over-reaction (the source paper flags ESPN's NBA feed). Ours: D_upper = 0 = no over-reaction. |
| **Discrete caveat** | The exact law is for continuous paths; CS2 WP jumps at kills. | note only | The continuous benchmark is a conservative reference; observed below it is expected discreteness, not a flaw. |

---

## Situational slices (report alongside the groups above)

- **Post-plant / endgame log-loss** (defuse-race value): post-plant log-loss should fall 7-8% with EB2; calibration preserved.
- **Comeback/tail reliability**: observed win rate when the model says 5% / 10% (0.7% / 6.8%).
- **Alive-state breakdown** (1v1..5v5, contested): localises where signal exists (5v5-even is the coin-flip bulk).
- **Live win-prob curve with bootstrap band**: the actual deliverable; step-like, wide band = honest about what it knows.

---

## One-line summary of the pipeline

Parse -> validate -> assemble (per-second snapshots) -> 5-fold OOF over matches -> Groups 1-4 in-time
-> Group 5 out-of-time (touch-once) -> Groups 6-7 (ceiling + trajectory) -> report the full grid with
match-level CIs. Ship on the configuration that is best **out-of-time**, calibrated, and honest on both
the contested and trajectory axes. Current recommendation: **EB2 (no firepower)**.
