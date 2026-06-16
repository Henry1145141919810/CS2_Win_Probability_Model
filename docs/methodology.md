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

## Current status (June 2026, 220 demos / 476,595 snapshots)
- Tier-1-filtered (dropped an ESL qualifier + a women's-team game); 1 demo off-list.
- Map control A/B (XGBoost): A 0.8318; B Voronoi 0.8366; G distance-grey 0.8357;
  BG both 0.8376; E Voronoi+tactical 0.8407; **EG full+grey 0.8416** (best, +0.0098 vs A,
  95% CI +0.0074..+0.0119; all CIs exclude 0).
- Finding: distance-grey alone ≈ Voronoi (slightly below) but COMPLEMENTARY — keep both.
  The grey model's facing(FOV)+LOS components are not yet added (need yaw re-parse + the
  4.4GB LOS matrix); current +0.001 from grey is a floor.
- Memory note: awpy LOS BVH = 4.4GB; build guarded/deferred. Distance-grey is memory-light.

## (Earlier) 208 demos / 449,521 snapshots
- Pillars 1, 2, 4 implemented; Pillar 3 (firepower) pending.
- AUC (5-fold GroupKFold): logreg A 0.8479 → E 0.8511; XGBoost A 0.8319 → E 0.8401.
- DeLong: every set (B/D/E) significant vs A (p<1e-3, both models).
- Bootstrap B=500 (match-level): all CIs exclude 0. XGBoost E−A = +0.0082
  (95% CI +0.0062 to +0.0102). Effect significant but modest -> motivates richer
  spatial features (contested/LOS control, facing, post-plant site focus, plant spots).
- Note: logreg ≥ XGBoost so far (0.851 vs 0.840) — XGBoost tuning pass pending.
- Full study adds the 2026 holdout, Pillar 3, deep models, and the revision features.
