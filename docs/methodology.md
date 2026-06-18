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
**Primary (comparable to Xenopoulos/ESTA and prior CS win-prob work):**
- **AUC-ROC** — discrimination (needs only 0/1 labels; rank-correctness).
- **Log-loss + Brier** — calibration (do the probabilities mean what they say).

**Complementary (standard outputs of `train_pipeline.py`):**
- **ECE** (Expected Calibration Error, 10 bins) — single-number summary of the reliability
  diagram: average |predicted% − observed win-rate| over probability bins. All our models
  are well-calibrated (ECE < 0.02). Reliability curve in `calibration.py` (with bootstrap CIs).
- **BSS** (Brier Skill Score) = 1 − Brier/Brier_base, where Brier_base uses the constant
  base-rate predictor. Scale-free (0 = no skill, 1 = perfect); easier to read than raw Brier.
- **contested-AUC (cAUC)** — AUC computed ONLY on *contested* snapshots (equal players alive
  AND |equipment diff| ≤ 1500). This is the headline complementary metric. *Rationale:*
  overall AUC is dominated by easy lopsided snapshots (5v2, eco vs full-buy) that economy
  already nails, which DILUTES the spatial signal — the aggregate map-control lift looks
  tiny (+0.003). A "better metric" will NOT inflate that +0.003; it is genuinely small on
  average. But cAUC correctly reports the lift **where it actually matters** (genuinely even
  rounds), where it is ~+0.013 — the honest *and* stronger framing. Economy's own AUC
  collapses from ~0.83 overall to ~0.58 on contested snapshots (near coin-flip), so the
  contested regime is exactly where any added signal is valuable. AUC/Brier stay primary
  (literature-comparable); cAUC/ECE/BSS are reported alongside.
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

## Interpretability — logistic coefficients (src/models/logistic_coefficients.py)
Because the headline model is logistic and ~ties tuned XGBoost (signal near-linear), the
standardized coefficients ARE the explanation. All inputs are z-scored, so coefficient
magnitude = effect size; sign = direction (+ favors CT). Outputs: console table + CSV
(`outputs/logistic_coefficients.csv`) + bar chart (`outputs/figures/logistic_coefficients.png`),
all sets (A / E / ET). Sanity (set ET, top drivers, coef on z-scale):
- Economy/combat dominate and have sensible signs: `min_ct_dist_to_bomb` −0.78 (CTs far
  from bomb = retake = worse), `ct_health_total` +0.72, `ct_equipment_value` +0.71,
  `t_equipment_value` −0.67, `t_players_alive` −0.57, `bomb_planted` −0.31.
- **Map control is the strongest non-economy/non-bomb signal**: `control_deficit` /
  `ct_voronoi_control_pct` +0.27 each (odds ×1.31 per SD), `ct_mid_control` −0.37 (collinear
  proxy; mid is a T-side staging read), `ct_banana_control` −0.16. Territory coefs are tiny
  (`ct_terr_deficit` +0.05) — consistent with territory being redundant with Voronoi once
  Voronoi is in the model (the ablation finding, restated by the linear model).
- Note: `control_deficit` ≡ `ct_voronoi_control_pct` − 0.5, so they're perfectly collinear
  and split one coefficient; report them as a single Voronoi term in the paper.

## Qualitative — where map control flips the call (src/viz/control_shift_examples.py)
To SHOW (not just measure) the spatial signal, fit OOF logreg A (economy) and E (economy +
**map control only**, deliberately excluding tactical/bomb so the delta is purely spatial),
then per snapshot `delta = P_E − P_A` is the win-prob attributable to map control. Rank
rounds by a sustained, contested (both teams alive, ≤1 player apart), correct shift; pick 3
distinct matches. Each yields a win-prob timeline (P_A vs P_E) + the 3-model map at the peak
second. Best example: **b8 vs flyquest m3 r12, a 4v4 with even economy** — economy alone
hugs 0.50 (coin flip) the whole round, but map control consistently and correctly pulls
toward T, who won (peak shift −22% at t+50s). This is the figure that makes the
contested-AUC number concrete: economy is blind in even rounds; control sees the read.

## Three-model control visualization (src/viz/mapcontrol_viz.py)
Renders all three control models stacked over the radar for chosen round/seconds:
Voronoi (greedy total partition) / grey (instantaneous LOS+FOV+smoke, ~60-80% grey, with
facing lines + smokes drawn) / territory (memory+decay 15s, replayed from freeze-end).
Documents the PROCESS and the negative result visually: grey flickers and is mostly
uncontested; territory stabilizes it back toward the Voronoi read.

## Model matrix — 5 architectures × {A, E, ET} (train_pipeline.py, 220 demos / 476,595 snaps)
5-fold GroupKFold OOF; metrics AUC (primary) / ECE / BSS / cAUC; B=500 match-block bootstrap.
| model | A AUC | E AUC | ET AUC | E−A (95% CI) | ECE(E) | cAUC(E) |
|---|---|---|---|---|---|---|
| logreg   | 0.8465 | **0.8493** | 0.8493 | +0.0028 (+0.0014,+0.0042) | 0.016 | 0.590 |
| xgb (tuned) | 0.8443 | 0.8476 | 0.8480 | +0.0034 (+0.0020,+0.0047) | 0.012 | 0.585 |
| lgbm     | 0.8448 | 0.8479 | 0.8476 | +0.0031 (+0.0018,+0.0042) | 0.014 | 0.586 |
| catboost | 0.8448 | 0.8478 | 0.8474 | +0.0030 (+0.0018,+0.0043) | 0.011 | 0.590 |
| rf       | 0.8228 | 0.8426 | 0.8428 | +0.0198 (+0.0165,+0.0228) | 0.011 | 0.585 |

- **Every model, every set: control lift E−A and ET−A is significant** (all bootstrap CIs
  exclude 0; DeLong p≈0). The well-specified models (logreg/xgb/lgbm/catboost) AGREE on the
  honest magnitude: **+0.003** aggregate. ET ≈ E everywhere (territory redundant w/ Voronoi).
- logistic E/ET is the best model (0.8493) and ~ties the GBMs (0.8474–0.8480) → signal is
  near-linear; the three GBMs are interchangeable.
- **RF is the cautionary cell**: its economy baseline UNDERFITS (0.8228, ECE 0.042 — poorly
  calibrated), so adding control "helps" a huge +0.0198 and fixes calibration (ECE→0.011).
  That large lift is a *weak-baseline artifact*, not extra spatial signal — exactly why we
  report the lift across architectures, not from one model. All non-RF cells are well
  calibrated (ECE < 0.02); cAUC ≈ 0.58–0.59 everywhere (economy collapses on contested snaps).
- Time-window AUC (5→25s) rises 0.69→0.76 for all; E/A curves nearly overlap (aggregate
  spatial lift is small and roughly constant in time), consistent with the contested story.

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
