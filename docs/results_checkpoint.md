# Results Checkpoint — CS2 Win-Probability Model (de_inferno)

**Date:** 2026-06-24 · **Dataset:** 220 Tier-1 demos / 476,595 snapshots / **104 cols (4 pillars)** · base P(CT win)=0.445
**In-sample best:** logistic on **EFB2** = **0.8519** · 4-model soft-vote **0.8531** · 9-architecture matrix complete
**⭐ OUT-OF-TIME (2026 holdout) OVERTURNS THIS → recommended model is `EB2` (NO firepower): lgbm EB2 = 0.8501 out-of-time (in-time 0.8493).**
**Firepower does NOT transfer** (logreg EFB2 0.8519 → **0.8236**) — a data-coverage gap, see §2026 holdout.

Snapshot of every pillar, model, feature set, test/metric, and the honest verdict on what
works. Companion to `docs/methodology.md` (full protocol + derivations).

---

## PART 1 — Inventory

### A. Evaluation protocol (constant)
- awpy v2.0.2, tickrate 64; one row per second (freeze-end → round end); label `ct_won`.
- **5-fold GroupKFold by `match_id`** (never split a match). All metrics are out-of-fold.

### B. Pillars (all 4 feature pillars implemented; deep models done on Betty)
| Pillar | Status | Feature group (cols) | Encodes |
|---|---|---|---|
| 1. Economy / combat | ✅ | `ECONOMY_COLS` (17) | equipment, health, armor, players-alive, kits, score, time, bomb-planted |
| 2a. Map control — Voronoi | ✅ | `MAPCONTROL_COLS` (9) | nearest-player area ownership; overall + per-zone + trend/volatility |
| 2b. Map control — grey | ✅ | `MAPCONTROL_LOS_COLS` (5) | instantaneous LOS+FOV+smoke (4-state) |
| 2c. Map control — territory | ✅ | `TERRITORY_COLS` (5) + `TERRITORY_ZONE_COLS` (5) | grey + memory/decay 15s; per-zone deficits |
| 3. Firepower (v1→v2, Leu) | ✅ | `FIREPOWER_COLS` (v1: 9 → v2: 20) | HLTV skill: v1 summed Rating/ADR/KAST+clutch; v2 side-aware + situational gating |
| 4a. Tactical readiness | ✅ | `TACTICAL_COLS` (28) | entropy, per-zone counts, AWP, utility, util advantage |
| 4b. Bomb geometry (plant) | ✅ | `BOMB_COLS` (6) | site, plant coords, nearest-CT straight + nav-path dist, CTs near bomb |
| 4c. Bomb-state + defuse race | ✅ NEW | `BOMB_LIVE_COLS` (8) | carried/dropped/planted, control around live bomb, dropped-bomb scramble, `defuse_time_margin` |
| interactions | ✅ | `INTERACTION_COLS` (4) | control×even-economy, control×equal-alive |

### C. Feature sets benchmarked
A (economy) · B (+Voronoi) · G (+grey) · Terr (+territory) · DG (tactical+grey) ·
DT (tactical+territory) · E (Voronoi+tactical) · ET (E+territory) · ET+ (per-zone+interactions) ·
**EB (E+bomb-live)** · EBT (full).

### D. Models tried
| Model | Config | Loss |
|---|---|---|
| Logistic | StandardScaler + LogisticRegression (L2, C=1) | log loss |
| XGBoost | tuned: depth 3, n600, lr .03, min_child 10, λ10, sub/col .8 | log loss |
| LightGBM | depth 3, 15 leaves, n600, lr .03, λ10 | log loss |
| CatBoost | depth 3, n600, lr .03, l2 10 | log loss |
| Random Forest | 300 trees | Gini (no global loss) |
| Ensemble | soft-vote / logit-avg of the 5 | — |
| Deep (GAT/TCN/Transformer) | planned, NOT run (GPU) | — |

### E. Metrics
AUC-ROC, log-loss, Brier (primary) · ECE, BSS, **contested-AUC** (complementary).

### F. Tests / interpretation / uncertainty / calibration
- Significance: DeLong; match-level block bootstrap (B=300–500) 95% CI on AUC & E−A diff.
- Temporal: time-window AUC (5/10/15/20/25s). Conditional: equal-alive / even-econ / contested /
  pre-plant / post-plant / dropped / endgame subsets.
- Interpretation ×3: standardized logistic coefficients; permutation importance (all 5 models,
  collinearity-robust); SHAP (TreeSHAP + LinearExplainer).
- Uncertainty: win-prob CI band — analytic delta-method (logreg) vs bootstrap-retrain (trees).
- Calibration: reliability + bootstrap ECE CI; calibration-over-time; isotonic/temperature; phase-aware.
- Error analysis: worst-prediction audit. Loss study: log loss vs Brier vs focal.
- Unified runner: `model_report.py` (metrics + uncertainty + calibration + interpretation).

---

## PART 2 — Stage verdict: what's good & meaningful

### Master AUC grid (OOF, 5-fold; B=300 bootstrap CI vs A)
| set ↓ \ model → | logreg | xgb | lgbm | catboost | rf | lift vs A (logreg, 95% CI) |
|---|---|---|---|---|---|---|
| A economy | 0.8465 | 0.8443 | 0.8448 | 0.8448 | 0.8228 | — |
| B +Voronoi | 0.8485 | 0.8468 | 0.8469 | 0.8468 | — | +0.0020 (+0.0011,+0.0029) ✅ |
| D +tactical | 0.8487 | 0.8472 | 0.8471 | 0.8473 | — | +0.0021 (+0.0010,+0.0036) ✅ |
| **F +firepower** | 0.8484 | 0.8459 | 0.8454 | 0.8455 | 0.8338 | +0.0018 (+0.0004,+0.0032) ✅ logreg only |
| E +Voronoi+tactical | 0.8493 | 0.8476 | 0.8479 | 0.8478 | 0.8426 | +0.0027 (+0.0015,+0.0042) ✅ |
| EF (4 pillars) | 0.8500 | 0.8475 | 0.8481 | 0.8468 | 0.8426 | +0.0034 (+0.0015,+0.0053) ✅ |
| EB2 +defuse-race | 0.8508 | 0.8489 | 0.8493 | 0.8491 | 0.8446 | +0.0043 (+0.0027,+0.0059) ✅ |
| EBT2 +territory | 0.8508 | 0.8494 | 0.8497 | 0.8489 | 0.8445 | +0.0043 (+0.0028,+0.0059) ✅ |
| **EFB2 (ALL pillars)** | **0.8515** | 0.8498 | 0.8498 | 0.8483 | 0.8442 | **+0.0049 (+0.0031,+0.0069) ✅** |

**Deep models (Betty GPU, 5-fold OOF, B=500 match bootstrap CIs) — a statistical DEAD HEAT:**
| model | AUC | AUC 95% CI | ECE | cAUC |
|---|---|---|---|---|
| classical EFB2 (logreg) | **0.8515** | ~(0.845, 0.858) | 0.016 | **0.596** |
| TCN (sequence, aggregate feats) | 0.8488 | (0.8420, 0.8557) | **0.013** | 0.572 |
| GAT (player self-attention, raw trajectories) | 0.8465 | (0.8396, 0.8534) | 0.014 | 0.576 |

Every model's point estimate falls **inside the others' 95% CIs** → classical, TCN, Transformer, and
GAT are **statistically indistinguishable**; no deep model beats the calibrated classical baseline,
none is worse. TCN tuned best = dropout 0.5/hidden 48/seq-len 160 (seq-len critical: 160→0.849,
100→0.826, 64→0.780); multi-seed 0.8485±0.0004. Transformer (causal encoder) 0.8473, best point
calibration (ECE 0.009). GAT on raw per-player trajectories the weakest (per-snapshot, no temporal,
data-hungry).

**9-ARCHITECTURE MATRIX COMPLETE — full deep + ensemble (OOF, B=500 CIs):**
| model | AUC | AUC 95% CI | ECE | BSS | cAUC |
|---|---|---|---|---|---|
| logreg EFB2 | 0.8515 | (0.844,0.858) | 0.016 | 0.372 | 0.596 |
| xgb EFB2 | 0.8498 | (0.843,0.857) | 0.013 | 0.370 | 0.591 |
| TCN | 0.8489 | (0.842,0.856) | 0.012 | 0.368 | 0.572 |
| Transformer | 0.8473 | (0.840,0.854) | 0.009 | 0.365 | 0.568 |
| GAT | 0.8465 | (0.840,0.853) | 0.014 | 0.361 | 0.576 |
| **SOFT-VOTE (4)** | **0.8531** | (0.846,0.860) | **0.009** | **0.378** | 0.589 |
| logistic-stack | 0.8529 | (0.846,0.860) | 0.042⚠ | 0.368 | 0.590 |

**Takeaways:** (1) on ~220 matches, careful feature engineering + a calibrated classical model
MATCHES sequence (TCN/Transformer) and spatial (GAT) deep learning — all statistically tied.
(2) The **soft-vote of classical+deep is the best overall model (0.8531, best calibration)** — deep
models don't win alone but ADD ENSEMBLE DIVERSITY. (3) soft-vote > logistic-stack (stack overfits the
prob scale, ECE 0.042). (4) Surpassing this clearly needs ~2-5× more data / spatio-temporal models.
Classical/soft-vote are the practical picks. ensemble_oof.py combines saved deep OOF + classical.

*EFB2 (all four pillars + bomb defuse-race) is the best classical model. Firepower is the **weakest
pillar**: F−A significant only on logreg (GBM CIs include 0); EF−E ≈ 0. Its value is conditional
(contested rounds). All non-RF models well-calibrated (ECE < 0.02).*

**Firepower v1 → v2 (both kept to show the process; same 220 demos / 5-fold OOF / B=500):**
| | v1 (9 feats, sums) | v2 (20 feats, side-aware+gated) |
|---|---|---|
| logreg F−A | +0.0018 ✅ | +0.0022 ✅ |
| logreg contested-AUC (F) | 0.593 | **0.603** ⬆ |
| **logreg EFB2 (best overall)** | 0.8515 | **0.8519** ⬆ (study best, cAUC 0.603) |
| xgb/lgbm/catboost EF−E | ≈0 (ns) | **negative** (catboost −0.0026 sig) |
| GBM EFB2 (xgb/lgbm/cat) | .8498/.8498/.8483 | .8480/.8479/.8465 ⬇ |
| count confound (rating-diff↔count) | 0.988 | **0.987 (persists)** |

v2's side-aware + situational design **helps the linear/headline model** (best contested-AUC 0.603
and best overall AUC logreg EFB2 0.8519 in the study) but **hurts the tree models** (sparse NaN-gated
features overfit). The **count confound persists** (v2 kept sums; `rating_sum` diff is 0.987-correlated
with the player-count advantage) — documented as a **limitation, not an open work item**: a per-capita
redesign (v3) was considered and **deliberately not pursued**, since the 2026 holdout showed the
pillar's binding constraint is **data coverage across eras**, not feature encoding (see holdout below).

---

## ⭐ 2026 OUT-OF-TIME HOLDOUT (touch-once) — the final validity gate
Trained on all 220 demos (2024-25), evaluated **once** on **27 fresh 2026 Inferno matches** (55,271
snapshots). Base rate shifts **0.445 → 0.512**. B=500 CIs resample the 27 test matches.
(`src/models/holdout_2026.py`, figure `outputs/figures/holdout_2026.png`.)

**✅ The core model GENERALISES — without firepower it does not degrade at all:**
| set | model | in-time (OOF) | **out-of-time 2026** | Δ |
|---|---|---|---|---|
| **EB2** | **lgbm** | 0.8493 | **0.8501** | **+0.0008** |
| EB2 | xgb | 0.8489 | 0.8497 | +0.0007 |
| EB2 | catboost | 0.8491 | 0.8494 | +0.0002 |
| E | xgb | 0.8476 | 0.8476 | +0.0000 |
| A | xgb | 0.8443 | 0.8441 | −0.0002 |

Contested-AUC even **improves** out-of-time (0.590 → 0.64–0.65) — the spatial signal is, if anything,
stronger in 2026. **Economy + Voronoi + tactical + defuse-race transfer cleanly to a new season.**

**❌ FIREPOWER DOES NOT TRANSFER — EFB2 collapses:**
| model | in-time | out-of-time | Δ |
|---|---|---|---|
| **logreg EFB2** | **0.8519** (best in-sample) | **0.8236** | **−0.0283** |
| rf EFB2 | 0.8446 | 0.8159 | −0.0287 |
| lgbm EFB2 | 0.8479 | 0.8325 | −0.0155 |
| xgb EFB2 | 0.8480 | 0.8344 | −0.0136 |

Calibration breaks too (logreg EFB2 **ECE 0.016 → 0.071**, intercept **−0.36**). **The best in-sample
model became the worst out-of-time.**

**Diagnosis — a DATA-COVERAGE gap, not a signal failure:** **0 of 27** 2026 demos are in
`demo_year_map.csv` (→ 2024 stats used for 2026), and **~30% of 2026 players have no HLTV stats**
(5v5 mean `ct_rating_sum`: **5.28 train vs 3.66 on 2026**; 11.8% of 2026 5v5 snapshots < 3, vs 0.0%
in training). The model reads corrupted firepower as "weak/few players alive."

**Lesson + decision:** a *skill-prior* pillar creates an **inference-time data dependency** the other
pillars don't have (they're computed from the demo itself). **On out-of-time evidence, the recommended
model is `EB2` (no firepower)** — CV ranked EFB2 first; the holdout overturns it. Base-rate shift costs
only mild calibration drift on non-firepower sets (slope ≈1.0, intercept +0.11-0.13; fixable by
recalibration). **Fix path (Leu):** add 2026 demos to `demo_year_map.csv` + scrape 2026 HLTV stats;
any re-run must be reported as a **separate, disclosed** evaluation.

### By model class
1. **Logistic — winner.** Best/tied at every set, tightest CI band (~0.03 vs xgb ~0.21), most
   interpretable, cheapest. Signal is essentially linear.
2. **GBMs (xgb/lgbm/catboost) — interchangeable, a hair behind.** Best calibrated (catboost
   ECE 0.011); slightly better in nonlinear contested subsets; higher epistemic variance.
3. **Random Forest — cautionary only.** Underfits (0.8228), poorly calibrated (ECE 0.042,
   Gini not loss-trained); its "lift" is a weak-baseline artifact.
4. **Ensemble — marginal** (+0.0010 over best single).

### By feature set (honest effect size)
| tier | sets | lift vs A | significant? | notes |
|---|---|---|---|---|
| Baseline | A | — | — | carries the easy lopsided snapshots; collapses to ~0.58 contested |
| Real, modest | B / E | +0.0028 (E) | ✅ all CIs exclude 0 | Voronoi + tactical; ~+0.013 on contested rounds |
| Real, best | **EB** | +0.0041 (logreg) | ✅ all 5 models | ~entirely `defuse_time_margin` (perm #8/68); post-plant log-loss −7-8% |
| Redundant | ET / ET+ / EBT | ≈ E/EB | — | territory adds ~0 once Voronoi present |
| Negative | G / Terr (alone) | ~0 | ✗ | grey & territory alone barely beat economy |

### One-paragraph takeaway
Use **logistic regression on set EB** (economy + Voronoi + tactical + bomb-live) — **0.8506**,
best in the study. Economy dominates (5–8× any spatial feature; collapses to 0.58 in even
rounds). On top, **two pillars carry real, significant signal: Voronoi map control (+0.003,
concentrated +0.013 in contested rounds) and the defuse-race geometry (best single addition
since economy, concentrated post-plant/endgame).** Everything else — grey control, territory,
bomb-neighbourhood ownership, dropped-bomb scramble — is redundant or a negative result. All
models well-calibrated (ECE < 0.02 except RF), honest at every round phase, and the spatial
story is corroborated three independent ways (coefficients, permutation importance, SHAP).

### Open levers (not yet done)
Pillar 3 firepower (HLTV scrape) · deep sequence models (GAT/TCN/Transformer, GPU) ·
more defuse-race detail (per-CT kit-aware timing, interrupt windows) · 2026 out-of-time holdout.
