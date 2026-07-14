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

## Standard evaluation protocol — RUN FOR EVERY NEW MODEL/METHOD
A model is not "done" at a bare AUC. To keep results comparable and honest across all
architectures, **every new model or method must pass the same battery** (one command:
`src/models/model_report.py --models <m> --sets <s>`):
1. **Metrics** — AUC / log-loss / Brier (primary) + ECE / BSS / contested-AUC (complementary).
2. **Uncertainty** — match-level block-bootstrap 95% CI on the AUC and the E−A lift; plus a
   per-round **win-prob CI band** (`winprob_chart.py`: logistic = analytic delta-method,
   trees = bootstrap-retrain).
3. **Calibration** — ECE + reliability diagram (bootstrap CI, `calibration.py`) and
   **calibration-over-time** (`calibration_over_time.py`); apply isotonic/Platt if needed.
4. **Interpretation** — model-agnostic **permutation importance** (`permutation_importance.py`,
   collinearity-robust); **SHAP** for tree models (`shap_analysis.py`); standardized
   coefficients for linear models (`logistic_coefficients.py`).
This is a standing requirement, not a one-off (the three interpretation methods must agree
before a feature claim is trusted — see the Voronoi coefficient-collinearity catch below).

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
| F | Economy + Firepower *(Pillar 3 — implemented v1→v2; historically labelled "C")* |
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
- **Map control is the strongest non-economy/non-bomb signal, but second-order.** CAVEAT:
  coefficients *inside* the full control block are collinearity-inflated and unstable —
  `control_deficit` ≡ `ct_voronoi_control_pct` − 0.5 (perfect duplicate, L2 splits → +0.27
  each) and the per-zone controls are correlated (`ct_mid_control` −0.37). Do NOT read +0.27
  as the Voronoi effect size. **Robust standalone fit (economy + one summary feature each):**
  Voronoi `control_deficit` **+0.115**, territory `ct_terr_deficit` **+0.066** (≈ half of
  Voronoi; the two are only r=0.47 correlated, so territory is a related-but-noisier proxy,
  redundant with Voronoi at the AUC level — ET≈E — not an identical signal). For scale,
  economy/combat terms are 5–8× larger.
- **Economy magnitudes (set A, clean):** `t_players_alive` −0.89, `ct_health_total` +0.74,
  `ct_equipment_value` +0.72, `t_equipment_value` −0.61, `bomb_planted` −0.51,
  `t_health_total` −0.43, `ct_players_alive` +0.43; (tactical) `min_ct_dist_to_bomb` −0.78.

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

## Full ablation + robustness/interpretation/uncertainty battery (June 2026)
**Ablation (logreg & xgb × 9 feature sets, `train_pipeline.py`):** A 0.8465 → B (+Voronoi)
0.8485 → E (Voronoi+tactical) **0.8493** (logreg). Grey (G) 0.8467 and territory (Terr)
0.8468 **alone barely beat economy** — the negative result, crisp; the lift comes from
Voronoi + tactical. ET ≈ E (territory redundant); ET+ no better. Voronoi's contested-AUC is
the highest single-pillar cAUC (B 0.594), i.e. it helps most exactly in even rounds.

**Three interpretation methods AGREE (economy ≫ map control; control real but second-order):**
- Permutation importance (`permutation_importance.py`, all 5 models, AUC-drop): top 8 are
  economy/combat (equipment, health, armor) + bomb geometry (`min_ct_dist_to_bomb`); map
  control (`ct_a_site_control`, `control_deficit`, `ct_mid_control`) ranks ~10-12th at
  +0.005–0.010 — consistent across logreg/xgb/lgbm/catboost/rf.
- SHAP (`shap_analysis.py`, TreeSHAP on xgb): same order — `ct_equipment_value` 0.46,
  `t_equipment_value` 0.46, health ~0.35, `min_ct_dist_to_bomb` 0.30, then map control
  (`ct_a_site_control` 0.11, `ct_voronoi_control_pct` 0.10) ~10th.
- Logistic standalone coefficients agree (Voronoi ~+0.11, economy 0.5–0.9). The collinearity
  trap (full-model +0.27 artifact) is exactly why we cross-check with the two methods above.

**Ensemble (`ensemble.py`):** soft-vote of the 5 classical models = AUC **0.8503** (+0.0010
over best single logreg) — the best result so far; the linear+nonlinear mix is mildly
decorrelated. Logit-average ≈ same.

**Uncertainty — win-prob CI band (`winprob_chart.py`):** logistic analytic band is tight
(mean width ~0.03); the xgb bootstrap band is ~7× wider (~0.21) → boosted trees have much
higher epistemic (train-sample) variance. Another reason to prefer logistic: better AUC,
tighter predictions, and interpretable.

**Calibration-over-time (`calibration_over_time.py`):** both headline models stay honest at
every phase (ECE < 0.03 throughout, tightest late as rounds resolve); xgb is slightly better
calibrated mid-round than logreg.

**Worst-prediction audit (`worst_prediction_audit.py`):** only 3.1% of snapshots are
confidently-wrong; the model is NOT overconfident in contested rounds (0.82% conf-wrong
there — it correctly says ~coin flip; contested log-loss 0.67 vs 0.43 elsewhere). Confident
errors are upsets in lopsided situations (80% eco-mismatch, 21% man-advantage) = irreducible
comebacks, not a systematic blind spot. Healthy, well-calibrated behavior.

## Bomb-state features + defuse-race feasibility (June 2026) — best result so far
Added bomb-state tracking (carried / dropped / planted, reconstructed from the bomb event
stream) and features for control AROUND the live bomb, the dropped-bomb scramble, and the
**defuse race** (`defuse_time_margin` = fuse time left − run-to-bomb time − defuse time).
Dataset → 91 cols; bomb states: carried 63% / dropped 18% / planted 19%.

**Benchmark (all 5 models, E vs E+bomb-live, B=300 bootstrap):** significant in EVERY model
(all CIs exclude 0). logreg 0.8493→**0.8506**, xgb 0.8476→0.8492, lgbm +0.0013, catboost
+0.0011, rf +0.0014. `logreg EB 0.8506` is the best single model in the study (ensemble was
0.8503). Log loss drops too (logreg 0.4605→0.4578).

**Concentrated where designed (retake + endgame):** post-plant log loss −7-8% (xgb
0.187→0.172, AUC 0.962→0.967); endgame (plant/30+) log loss 0.401→0.396 across all models.
This is the "better endgame features lower endgame log loss" lever from the calibration study,
realised — and ECE stays ~0.013 (calibration preserved).

**Honest decomposition (permutation importance, xgb):** the win is almost entirely
**`defuse_time_margin` (AUC-drop +0.0090, rank #8/68)** — a top-tier feature. The other new
features are weak: `min_ct_dist_to_bomb_live` #19, then bomb-local control / dropped-bomb /
state all ~#22-59 (near-zero). NEGATIVE result within the positive: **map control *around*
the bomb is redundant** with the existing site-control features, and the **dropped-bomb
scramble barely predicts** the outcome (dropped subset log loss 0.492→0.490) — a loose bomb
is mostly a *consequence* of a lost round, not a cause. Lesson: the defuse-race GEOMETRY is
what matters post-plant, not bomb-neighbourhood ownership. Feature sets EB / EBT added to
train_pipeline; bomb-live promoted into the canonical dataset.

## Defuse-race v2, stacking, error map, SHAP mechanism (June 2026)
**Defuse-race v2 (per-CT kit-aware finish time, defuse-capable count, T-contest window):**
re-assembled (95 cols, sets EB2/EBT2). Result: **≈ v1, no real gain** — logreg EB 0.8506 →
EB2 0.8508 (+0.0002), xgb −0.0003; the simple `defuse_time_margin` already captured the
signal. Honest marginal/negative refinement. Best single model = **logreg EB2 0.8508**
(highest contested-AUC yet, 0.596); EBT2 helps trees slightly (xgb 0.8494, lgbm 0.8497). All
bomb sets significant vs A (DeLong p≈0).

**Ensembling (`stacking.py`, set EB2):** **soft-vote = 0.8518** (ECE 0.014, BSS 0.375) is the
best PRACTICAL model — calibrated. A logistic **stack** edges AUC to 0.8520 but **wrecks
calibration** (ECE 0.046, worse log loss): the meta-learner overfits the probability scale on
correlated OOF preds (xgb gets a negative weight — collinearity). Lesson: prefer soft-vote;
the stack's +0.0002 AUC isn't worth the calibration damage.

**SHAP dependence (`shap_dependence.py`) — HOW the defuse race acts:** `defuse_margin_kit`
swings SHAP from −1.19 (CT can't reach+defuse in time → strongly T) to +0.04 (CT can) around
the feasibility boundary; even larger than v1 (−0.91). `defuse_contest_margin` adds the
T-interruption axis (−0.57→+0.33). Voronoi `control_deficit` SHAP is tiny in xgb (±0.01),
consistent with second-order.

**Error map (`zone_side_error.py`, logreg EB2):** calibration-gap ≈ 0 in every cut (no bias).
Hardest where the round is still open: **Ts in mid (AUC 0.787) or banana (0.808)** vs at a
site (0.93). B-site retakes a touch harder than A (log loss 0.191 vs 0.166). Early (0.615) ≫
endgame (0.396) log loss — the irreducible-uncertainty gradient.

**Sensitivity sweep (`sensitivity_sweep.py`, 35-demo subset, logreg EBT2):** the result is
robust to every design constant — re-assembling under territory decay ∈ {10,15,20}s, Voronoi
weighting {area,count}, bomb-local radius ∈ {400,600,800}u, and CT defuse speed ∈
{215,250,285} moves AUC by **|Δ| ≤ 0.0004** (baseline 0.8509). No result depends on an
arbitrary parameter choice — the constants are not load-bearing.

## Pillar 3 firepower — v1 → v2 (by teammate Leu)

### Firepower v1 (June 2026) — sum-based, 9 features
Firepower = HLTV Rating/ADR/KAST **summed** over alive players + a 1vN **clutch score**, joined
on `(steamid, year)` (skill drifts yearly). 9 cols; sets **F** (econ+firepower), **EF** (4
pillars), and a new all-pillars set **EFB2** (econ+map+tactical+territory+firepower+bomb). Canonical
dataset re-assembled to **104 cols**.

**Full matrix (5 models × {A,B,D,F,E,EF,EB2,EBT2,EFB2}, B=300 bootstrap):** **EFB2 is the new best
model — logreg 0.8515** (vs A +0.0049, 95% CI +0.0031..+0.0069; xgb 0.8498, lgbm 0.8498). Brier
0.1552 / log-loss 0.4559 (best in study); ECE 0.016 (calibration preserved).

**Firepower is the WEAKEST, least-robust pillar — honest result:**
- **F − A significant only on logreg** (+0.0018, CI +0.0004..+0.0032); on xgb/lgbm/catboost the
  bootstrap CI **includes 0** (not significant). Matches Leu's own caveat.
- **EF − E ≈ 0** (logreg +0.0007, CI includes 0; xgb −0.0000; catboost −0.0011) — once the other
  pillars are present, firepower adds essentially nothing to aggregate AUC.
- **Where it does help = contested rounds:** cAUC F−A ≈ **+0.007** across ALL models (logreg
  0.587→0.593, xgb 0.580→0.587, …) — same "matters where economy fails" shape as map control.
- **Clutch (1vN) is NOT where it helps:** AUC ≈ 0.98 for every set there — a 1vN is near-decided by
  the man-advantage; `clutch_score` adds ~nothing.

**CRITICAL confound (the key critique for firepower v2):** `firepower_rating_diff` is permutation-
importance **rank #1** (AUC-drop 0.061) — but it correlates **0.988** with the player-count
advantage `(ct_alive − t_alive)` and predicts `ct_won` *identically* (both r=0.498). Because Rating
is **summed** over alive players, this feature is ~99% a **player-count proxy, not a skill signal**;
its apparent dominance is an artifact (it re-encodes the strongest predictor, already in economy),
which is exactly why its marginal lift (EF−E) is ~0. **Recommendation: firepower v2 should use
*average* rating per alive player (skill-per-capita) to decouple skill from count.** Also, the
logistic firepower coefficients sign-flip (`ct_firepower_adr` −1.66) from collinearity among
rating/ADR/KAST — trust permutation importance + AUC, not raw coefficients.

### Firepower v2 (July 2026) — side-aware + situational gating, 20 features
Leu's v2 (commit fc4f719) is a redesign, kept alongside v1 to show the progression. Changes:
**side-aware stats** (CT-side vs T-side Rating/Firepower/Entrying/Trading/Opening from a new
`player_stats_sided.csv`; each alive player queried for the side they're *currently* playing);
**per-player conditional gates** (lone survivor → Clutching + suppress Entry/Trading; teammates
alive → Entry/Trading; Opening only when both sides 5-alive); **Sniping** exposed as the AWP
holder's role score; **Utility** = HLTV utility skill × current grenade $ carried. `FIREPOWER_COLS`
= **20**; dataset re-assembled to **115 cols**. Same 220 demos / 5-fold OOF / B=500 → directly
comparable to v1.

**v2 result — mixed, and instructive (both kept on the record):**
| metric | v1 | v2 |
|---|---|---|
| logreg F−A | +0.0018 ✅ | +0.0022 ✅ |
| logreg EF−E | +0.0007 (ns) | +0.0010 (ns) |
| **logreg contested-AUC (F)** | 0.593 | **0.603** ⬆ |
| **logreg EFB2 (best overall)** | 0.8515 | **0.8519** ⬆ (study best; cAUC 0.603) |
| xgb/lgbm/catboost EF−E | ≈0 (ns) | **negative** (catboost −0.0026, *sig*) |
| GBM EFB2 (xgb/lgbm/cat) | .8498/.8498/.8483 | .8480/.8479/.8465 ⬇ |
| count confound (rating-diff ↔ player-count) | 0.988 | **0.987 (persists)** |

- **v2 helps the linear / headline model** where it matters: best **contested-AUC (0.603)** and best
  overall AUC (**logreg EFB2 0.8519**) in the whole study — the side-aware/situational signal pays off
  exactly in even rounds.
- **v2 hurts the tree models** (EF < E; catboost significantly): the 20 sparse, NaN-gated features
  (Opening only at 5v5, Clutch only 1vN) give GBMs noise to overfit; v1's 9 dense features were cleaner.
- **The count confound is NOT fixed** — v2 kept **sums**, so `ct_rating_sum − t_rating_sum` is still
  **0.987**-correlated with the player-count advantage. Permutation importance: the sum features still
  lead (`ct_rating_sum` #9, `t_rating_sum` #12), with a few new side-aware ones surfacing
  (`t_trading_sum` #10, `t_awp_sniping_skill` #19).
- **Open: firepower v3** = *average* rating per alive player (per-capita) to finally decouple skill
  from count; and prune the sparse gated features that hurt the GBMs.

**Practical takeaway:** use **v2 for the logistic / headline model** (best contested-AUC + best AUC);
for GBMs, v1 or a pruned v2 is better. Both versions retained in the repo.

## 2026 OUT-OF-TIME HOLDOUT (touch-once) — the final validity gate ⭐
Trained on the full 220-demo (2024-25) set, evaluated **once** on **27 fresh 2026 Inferno matches**
(55,271 snapshots; IEM Kraków / Cologne Major / Atlanta, BLAST Rivals) that appear in no fold.
Base rate shifts **0.445 → 0.512** (2026 Inferno is markedly more CT-sided). B=500 bootstrap CIs
resample the 27 test matches. Script: `src/models/holdout_2026.py`; figure `outputs/figures/holdout_2026.png`.

**Result 1 — the core model GENERALISES (the headline positive).** Without firepower, out-of-time
performance is essentially unchanged (some sets slightly *better*):
| set | model | in-time (OOF) | out-of-time 2026 | Δ |
|---|---|---|---|---|
| E | xgb | 0.8476 | 0.8476 | +0.0000 |
| **EB2** | **lgbm** | 0.8493 | **0.8501** | **+0.0008** |
| EB2 | xgb | 0.8489 | 0.8497 | +0.0007 |
| EB2 | catboost | 0.8491 | 0.8494 | +0.0002 |
| A | xgb | 0.8443 | 0.8441 | −0.0002 |

→ **economy + Voronoi map control + tactical + defuse-race TRANSFER cleanly to a new season.**
Contested-AUC even *improves* out-of-time (0.590 → 0.64-0.65).

**Result 2 — FIREPOWER DOES NOT TRANSFER (the critical negative).** Set EFB2 collapses:
| model | in-time | out-of-time | Δ |
|---|---|---|---|
| logreg EFB2 | **0.8519** (best in-sample) | **0.8236** | **−0.0283** |
| rf EFB2 | 0.8446 | 0.8159 | −0.0287 |
| lgbm EFB2 | 0.8479 | 0.8325 | −0.0155 |
| xgb EFB2 | 0.8480 | 0.8344 | −0.0136 |
Calibration breaks too: logreg EFB2 **ECE 0.016 → 0.071**, calibration intercept **−0.36**.
**The best in-sample model became the worst out-of-time.**

**Diagnosis — a DATA-COVERAGE gap, not a signal failure:**
- **0 of 27** 2026 demos are in `demo_year_map.csv` → `year_for_match()` silently falls back to
  `DEFAULT_YEAR=2024`, so **2024 skill stats are used for 2026 matches**.
- **~30% of 2026 players have no HLTV stats at all** → their firepower contributes **0**. At 5v5,
  mean `ct_rating_sum` is **5.28 in training** but only **3.66 on 2026**; 11.8% of 2026 5v5 snapshots
  have `rating_sum < 3` (vs **0.0%** in training).
- The model therefore reads corrupted firepower as "few/weak players alive" and mispredicts.

**Deployment lesson (a real contribution):** a *skill-prior* pillar creates an **inference-time data
dependency** that the other pillars do not have — economy/map-control/bomb features are computed
from the demo itself and always available, whereas firepower needs an external, per-era player
database. If that database lags the deployment era, the pillar **actively harms** the model.

**Recommended model changes on out-of-time evidence: use EB2 (no firepower), not EFB2.**
Cross-validation ranked EFB2 first; the holdout overturns that. This is exactly why the touch-once
out-of-time gate exists — 5-fold CV could never have seen it.

**Base-rate shift:** on the non-firepower sets the shift (0.445→0.512) costs only mild calibration
drift (ECE 0.013→0.023-0.028; calibration slope ≈1.0, intercept +0.11-0.13 — i.e. the model
under-predicts CT wins by a roughly constant amount). Ranking (AUC) is unaffected; a simple
recalibration (intercept shift / isotonic on recent data) fixes it.

**Fix path (Leu):** add the 27 2026 demos to `demo_year_map.csv` (year=2026) and scrape 2026 HLTV
stats for the missing players into `player_stats_sided.csv`. Any re-evaluation after that fix must be
reported as a **separate, disclosed** holdout run (the first run above stands as the honest
touch-once result for the current pipeline).

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
