# Results Checkpoint — CS2 Win-Probability Model (de_inferno)

**Date:** 2026-06-19 · **Dataset:** 220 Tier-1 demos / 476,595 snapshots / 95 cols · base P(CT win)=0.445
**Best single model:** logistic on **EB2** = **0.8508 AUC** · **best practical:** soft-vote (5 models, EB2) = **0.8518** (calibrated)

Snapshot of every pillar, model, feature set, test/metric, and the honest verdict on what
works. Companion to `docs/methodology.md` (full protocol + derivations).

---

## PART 1 — Inventory

### A. Evaluation protocol (constant)
- awpy v2.0.2, tickrate 64; one row per second (freeze-end → round end); label `ct_won`.
- **5-fold GroupKFold by `match_id`** (never split a match). All metrics are out-of-fold.

### B. Pillars (4 of 5 planned; firepower + deep models deliberately skipped)
| Pillar | Status | Feature group (cols) | Encodes |
|---|---|---|---|
| 1. Economy / combat | ✅ | `ECONOMY_COLS` (17) | equipment, health, armor, players-alive, kits, score, time, bomb-planted |
| 2a. Map control — Voronoi | ✅ | `MAPCONTROL_COLS` (9) | nearest-player area ownership; overall + per-zone + trend/volatility |
| 2b. Map control — grey | ✅ | `MAPCONTROL_LOS_COLS` (5) | instantaneous LOS+FOV+smoke (4-state) |
| 2c. Map control — territory | ✅ | `TERRITORY_COLS` (5) + `TERRITORY_ZONE_COLS` (5) | grey + memory/decay 15s; per-zone deficits |
| 3. Firepower | ❌ skipped | — | needs HLTV per-player scrape (manual/ToS) |
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

### Master AUC grid (OOF, 5-fold)
| set ↓ \ model → | logreg | xgb | lgbm | catboost | rf |
|---|---|---|---|---|---|
| A economy | 0.8465 | 0.8443 | 0.8448 | 0.8448 | 0.8228 |
| B +Voronoi | 0.8485 | 0.8468 | — | — | — |
| G +grey | 0.8467 | 0.8451 | — | — | — |
| Terr +territory | 0.8468 | 0.8453 | — | — | — |
| E +Voronoi+tactical | 0.8493 | 0.8476 | 0.8479 | 0.8478 | 0.8426 |
| ET +territory | 0.8493 | 0.8480 | 0.8476 | 0.8474 | 0.8428 |
| **EB +bomb-live** | 0.8506 | 0.8492 | 0.8491 | 0.8489 | 0.8440 |
| **EB2 +defuse-race v2** | **0.8508** | 0.8489 | 0.8493 | 0.8491 | 0.8446 |
| EBT2 full | 0.8508 | 0.8494 | 0.8497 | 0.8489 | 0.8445 |
| **Soft-vote (5 models, EB2)** | — | — | — | — | **0.8518** |
| Logistic stack (EB2) | — | — | — | — | 0.8520 (ECE 0.046 — miscalibrated) |

*Defuse-race v2 (EB2) ≈ v1 (EB): the simple defuse_time_margin already captured it. Soft-vote
is the best practical model (calibrated); the stack's extra +0.0002 AUC ruins calibration.
Model weakest where the round is open (Ts in mid AUC 0.787 / banana 0.808), strong at sites (0.93).*

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
