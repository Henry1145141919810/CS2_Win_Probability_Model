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

## LOS/FOV/smoke grey control — negative result (June 2026)
Built a physically-faithful control model: an area is controlled only if a living player
is in range AND has line-of-sight (precomputed 3060x3060 nav visibility matrix, ray-cast
vs the map mesh) AND is facing it (FOV from yaw) AND it is not occluded by an active smoke.
Surface is ~80% "grey" (uncontested) at any instant — realistic for CS.

**Finding: it does NOT improve round-win prediction over simple Voronoi.** XGBoost:
B (Voronoi) +0.0048 vs A; G (full grey) only +0.0021; the full grey is even weaker than
distance-only grey (+0.0039). EG (full+grey) 0.8402 ≈ E (Voronoi+tactical) 0.8407 — grey
adds nothing on top. Interpretation: positional/proximity territory is a more STABLE
predictor of round state than instantaneous sightline control, which flickers tick-to-tick
(yaw is twitchy; visible set changes constantly). A publishable negative result; the LOS/
smoke/facing infrastructure is reusable (e.g. time-smoothed control, retake analysis).
Decision: keep Voronoi as the control feature. Best model = E (econ+Voronoi+tactical) 0.8407.

## Territory control (memory+decay) — recovers realistic control to Voronoi parity
Added a 3rd control model: grey/LOS/FOV/smoke control WITH MEMORY — cleared space stays a
team's for `decay`=15s without re-checking (FOV only gates re-acquiring; close range always
held). Stateful per round. XGBoost: instant-grey alone G=0.8341 < Voronoi B=0.8366; but with
tactical, territory DT=0.8404 ≈ Voronoi+tactical E=0.8407 (>> grey DG=0.8394). Both together
ET=0.8412 (new best, +0.0095 vs A) but only +0.0005 over E (redundant — both measure stable
territory). Conclusion: temporal STABILITY is what makes spatial control predictive;
memory/decay fixes the flickering grey model to match proximity-Voronoi, validating that
"realistic ≠ predictive unless stabilized." Territory is also the most interpretable for viz.
All three control models kept in the dataset (74 cols) for the ablation narrative.

## XGBoost tuning + improved encoding — the spatial lift was partly overfitting
- tune_xgb.py (GroupKFold grid): default depth-6 overfits; tuned depth-3/min_child-10/
  lambda-10 (n_est 600, lr 0.03, subsample/colsample 0.8) → +0.007 on ET. Adopted as the
  pipeline default. Tuned xgb (~0.848) now ~ties logistic (0.849) → signal near-linear.
- CRUCIAL: tuned baseline A jumped 0.8318→0.8443; spatial lift E−A shrank +0.0090→+0.0034.
  So a chunk of the "spatial contribution" was XGBoost OVERFITTING noise. Honest spatial
  lift on a proper model = ~+0.003-0.004 (logistic agreed all along: +0.0028).
- Better encoding (ET+ = per-zone territory deficits + control×economy interactions): did
  NOT help (ET+ ≈ ET, logistic dipped). Redundant with aggregate control + economy. Negative.
- Best model: logistic E/ET 0.8493 (tuned xgb ET 0.8480). Spatial pillars real+significant
  but modest (~+0.003), concentrated in contested rounds (~+0.013, see below). Next lever:
  Pillar 3 firepower.

## Conditional analysis — WHERE map control matters (src/models/conditional_analysis.py)
Aggregate spatial lift is small (+0.009) because it's diluted by easy lopsided snapshots.
Restricting to contested subsets (XGBoost, OOF AUC):
- Economy baseline COLLAPSES in even rounds: ALL 0.832 → equal-alive 0.686 → even-econ 0.674
  → equal-alive & even-econ 0.578 (~coin flip). The "0.83 overall" is carried by easy
  lopsided snapshots; economy knows little in genuinely even rounds.
- Map-control lift is LARGER where it matters: equal-alive (half the data) E−A = +0.0131
  vs +0.0090 overall (~45% bigger); even-econ +0.0107; pre-plant +0.0103.
- Magnitude is bigger but moderate (~+0.013, not +0.02-0.04); the most-contested subset is
  near-coin-flip for ALL features (0.58) = irreducible randomness of an even CS round.
- NEW insight: spatial signal is NONLINEAR — XGBoost extracts the lift in contested subsets,
  Logistic gets ~0/negative there. So logistic wins on easy snapshots; xgb wins where it's hard.
Paper framing: "map control is most informative exactly when economy fails (contested rounds),
and its signal needs nonlinear models." Key figure material.

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
