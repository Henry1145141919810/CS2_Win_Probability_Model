# Methodology — Evaluation & Uncertainty Quantification

The git-tracked companion to the proposal. Mirrors the protocol implemented in
`src/models/train_pipeline.py` so the plan lives with the code.

## Snapshots & labels
- Each round is sampled once per second from freeze-end to round end (tickrate 64).
- Every snapshot is one training row, labelled with the round's eventual side winner
  (`ct_won` ∈ {0,1}). The model outputs `P(CT win) = E[ct_won | features]`; the
  probability fluctuates within a round because the features change second-by-second.

## Cross-validation
- **5-fold GroupKFold by `match_id`.** Never split within a match — rounds/snapshots of
  one match are correlated; folding by round leaks and inflates AUC.

## Metrics
- **AUC-ROC** — discrimination (needs only 0/1 labels; rank-correctness).
- **Log-loss + Brier** — calibration (do the probabilities mean what they say).
- **Calibration curve** (reliability diagram, 10 bins) — predicted vs observed win rate;
  apply isotonic regression if miscalibrated.
- **DeLong's test** — each feature set's AUC vs the economy baseline (Model A), on the
  same out-of-fold predictions.

## Feature sets (models A–E)
| Set | Features |
|---|---|
| A | Economy only (Xenopoulos/ESTA baseline) |
| B | Economy + Map control |
| C | Economy + Firepower *(Pillar 3 pending)* |
| D | Economy + Tactical readiness (incl. bomb/rotation) |
| E | All available pillars |

## Time-window analysis (headline)
Re-evaluate every model on snapshots at exactly 5/10/15/20/25 s into the round; plot
AUC vs time. The second at which the full model diverges from Model A is the
**control signal emergence point**.

## Temporal evaluation
- **2026 held-out test set** (15–20 fresh Inferno demos): touched once, at the end, for
  out-of-time generalization. CV drives all development so 2026 stays uncontaminated.
- Auxiliary check: train on 2024, test on 2025.

## Bootstrap & confidence intervals
- **Block bootstrap at the match level** — matches are the independent resampling unit.
  Each iteration draws all matches with replacement (keeping their rounds/snapshots).
- **B = 500** for 95% CIs on (i) each model's AUC and (ii) the AUC difference
  **E − A**; a difference CI excluding 0 ⇒ statistically significant improvement.
- **B = 2000** for the discrete control-signal-emergence-point (stable percentile CIs).
- **Targets:** XGBoost × {A–E} on laptop CPU; GAT/TCN/Transformer/Ensemble × E on a
  cloud A100. Other cells report 5-fold GroupKFold mean ± std. Deep models additionally
  report mean ± std over 20 random seeds (initialisation instability). Cloud loops
  checkpoint every 10 iterations.
- Implemented: `train_pipeline.py --bootstrap B` (metric-level match block bootstrap on
  fixed OOF predictions; the full-retrain variant is the heavier cloud job).

## Current status (June 2026, 22 dev demos)
- Pillars 1, 2, 4 implemented; Pillar 3 (firepower) pending.
- XGBoost: A 0.7745 → B 0.7801 → E 0.7851 (DeLong +0.0105 vs A, p<1e-3).
- Bootstrap (B=500): E−A diff +0.0105 (95% CI +0.0020 to +0.0191, excludes 0).
- Preliminary at this sample size; full study uses ~150 train demos + 2026 holdout.
