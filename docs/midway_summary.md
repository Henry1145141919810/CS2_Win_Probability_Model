# CS2 Win-Probability Model — Midway Status (for Leu)

**Project:** live per-second round win-probability for de_inferno (arXiv + MLSA 2027)
**As of:** 2026-06-25 · **Repo:** github.com/Henry1145141919810/CS2_Win_Probability_Model
**TL;DR:** 4 feature pillars built, full 9-architecture model matrix evaluated with bootstrap CIs +
a thorough metric battery. **Best model: 4-model soft-vote ensemble = 0.8531 AUC** (best single:
logistic on all-pillars EFB2 = 0.8515). The modeling phase is essentially done and paper-ready; the
open work is **firepower v2 (you), the 2026 out-of-time holdout, and possibly more data.**

---

## 1. Where the project stands
We predict P(CT win) at every second of a round from the game state. The pipeline: parse demos
(awpy, tickrate 64) → assemble one feature row per per-second snapshot → train/evaluate with
5-fold **GroupKFold by match** (never split a match). Dataset: **220 Tier-1 demos / 476,595
snapshots / 104 columns**, base rate P(CT win)=0.445.

The honest headline: **economy dominates; map control + the defuse-race add small-but-significant
lift; firepower is the weakest pillar; and deep models match (don't beat) a calibrated classical
model at this data size.** It's a rigorous, calibrated, interpretable study — a strong paper even
without a blockbuster AUC.

---

## 2. The four feature pillars (what they are + verdict)

| Pillar | What it encodes | Verdict |
|---|---|---|
| **1. Economy / combat** (17 cols) | equipment, HP, armor, players-alive, kits, score, time, bomb-planted | **Dominant.** Carries the model. Collapses to ~0.58 AUC in *even* rounds (a coin flip). |
| **2. Map control** (3 variants) | Voronoi (proximity), grey (LOS+FOV+smoke), territory (grey+memory/decay) | **Voronoi is real & significant** (+0.003 overall, +0.013 in contested rounds). Grey/territory are **negative/redundant** — "realistic ≠ predictive unless stabilized." |
| **3. Firepower** (your pillar, 9 cols) | HLTV Rating/ADR/KAST summed over alive players + 1vN clutch | **Weakest pillar** — see §5. Has a *count confound* to fix in v2. |
| **4. Tactical + bomb** (28+6+8 cols) | utility, AWP, entropy, zones, bomb geometry, **defuse-race** | **Defuse-race is the best single addition since economy** — see §4. |

---

## 3. The model matrix — 9 architectures (5-fold OOF, B=500 bootstrap CIs)
| model | AUC | 95% CI | ECE | cAUC | notes |
|---|---|---|---|---|---|
| **logreg EFB2** | **0.8515** | (0.844,0.858) | 0.016 | 0.596 | best single; signal is near-linear |
| xgb EFB2 | 0.8498 | (0.843,0.857) | 0.013 | 0.591 | |
| lgbm EFB2 | 0.8498 | — | 0.012 | 0.593 | |
| catboost EFB2 | 0.8483 | — | 0.018 | 0.586 | |
| RF EFB2 | 0.8442 | — | — | 0.574 | underfits economy (cautionary) |
| TCN (sequence) | 0.8489 | (0.842,0.856) | 0.012 | 0.572 | Betty GPU |
| Transformer | 0.8473 | (0.840,0.854) | 0.009 | 0.568 | Betty GPU |
| GAT (raw trajectories) | 0.8465 | (0.840,0.853) | 0.014 | 0.576 | Betty GPU |
| **SOFT-VOTE (4)** | **0.8531** | (0.846,0.860) | 0.009 | 0.589 | **best overall** |
| logistic-stack | 0.8529 | — | 0.042⚠ | 0.590 | overfits prob scale → use soft-vote |

**Every model's point estimate is inside the others' 95% CIs → statistically a dead heat.** No
deep model beats the classical baseline; the ensemble wins because it combines *diverse* learners
(classical + deep). Feature sets: A=economy, E=+Voronoi+tactical, F=+firepower, EF=4 pillars,
EB2=+defuse-race, **EFB2 = all pillars** (the best).

---

## 4. Key scientific findings
1. **Contested-AUC is our novel evaluation lens.** Economy alone collapses to ~0.58 AUC in even
   rounds (equal players + even economy). The spatial pillars earn their value *exactly there*.
2. **Map control (Voronoi) is robustly significant** (CIs exclude 0 across all models), strongest in
   contested rounds. Grey/territory variants are redundant — a clean ablation story.
3. **Defuse-race geometry** (`defuse_time_margin` = fuse-time-left − run-time − defuse-time) is the
   **#8 most important feature overall** and cuts post-plant log-loss ~7–8% — the best single feature
   added since economy. (Bomb-neighborhood control and dropped-bomb features were weak/redundant.)
4. **Deep models tie, don't beat.** TCN/Transformer/GAT all ≈ classical; "momentum" and "raw
   trajectories" don't help at 220 matches — the engineered features already capture the signal.
5. **The accuracy gain is *resolution*, not calibration** (Brier decomposition): economy→all-pillars
   raises Resolution (discrimination) while Reliability stays ~0. The features add skill **without**
   costing calibration.
6. **Honest probabilities + honest comebacks.** All models calibrated (ECE<0.02, robust to binning).
   Tail calibration: model says 5%→side wins 0.7%, says 10%→6.8%. Eventual winner was written off
   (≤10%) in 7.2% of rounds, and those calls are calibrated — the live curve is trustworthy.

---

## 5. Firepower (your pillar) — the v1 result and the v2 fix
**v1 (what you built):** HLTV Rating/ADR/KAST *summed* over alive players + clutch, joined by
`(steamid, year)`. Integrated cleanly; benchmarked across all 5 models.

**Honest v1 result:**
- F − A significant **only on logreg** (xgb/lgbm/catboost bootstrap CIs include 0); **EF − E ≈ 0**
  (adds ~nothing on top of the other pillars).
- Its real value is **conditional**: contested-AUC F − A ≈ **+0.007** across all models. Clutch (1vN)
  is *not* where it helps (man-advantage dominates; AUC there ~0.98 for everything).

**⚠️ The key issue to fix in v2 — a count confound.** `firepower_rating_diff` is permutation-
importance **rank #1**, but it correlates **0.988** with the player-count advantage `(ct_alive −
t_alive)` and predicts the outcome *identically* (both r=0.498). Because Rating is **summed** over
alive players, this feature is ~99% a **player-count proxy, not a skill signal** — which is why its
marginal lift is ~0 (the count is already in economy). The logistic firepower coefficients also
sign-flip (ADR −1.66) from collinearity among rating/ADR/KAST.

**Recommended v2:**
1. Use **average rating per alive player** (skill-per-capita), not the sum → decouples skill from
   count. e.g. `ct_avg_rating`, `firepower_skill_diff = ct_avg_rating − t_avg_rating`.
2. Drop or rethink `clutch_score` (added ~nothing) — consider situational skill: recent multi-kill
   rate, opening-duel win-rate, gated by the same 1vN/entry logic.
3. Keep the year-aware `(steamid, year)` join (that part is good).
4. After v2: re-assemble (`python src/features/assemble.py`, ~35 min) and re-run the standard battery
   `train_pipeline.py --models logreg,xgb,lgbm,catboost,rf --sets A,F,E,EF,EFB2 --bootstrap 500`.

---

## 6. Evaluation framework (full battery — see `docs/metrics.md`)
- **Primary:** AUC, log-loss, Brier. **Complementary:** ECE, BSS, **contested-AUC**.
- **Extended (new):** Brier decomposition (reliability/resolution/uncertainty), sharpness,
  **calibration slope+intercept** (bin-free), **adaptive-ECE + KS-cal** (bin-free), and a
  **comeback/tail honesty** diagnostic.
- **Uncertainty:** match-level block bootstrap (B=500) 95% CIs on every metric; multi-seed std
  for deep models; win-prob CI bands.
- **Interpretation:** standardized coefficients, permutation importance (all models), SHAP.
- **Standing rule:** every new model/method runs the *same* interpretation + uncertainty +
  calibration battery, swept across all models × valuable feature sets.

---

## 7. Infrastructure (where things run)
| Runs on a laptop (local) | Runs on PARCC Betty (GPU) |
|---|---|
| all 5 classical models, ensemble, **all metric computation**, feature assembly | TCN, Transformer, GAT (deep models) |

Deep models train on Betty B200 GPUs via Slurm (env `cs2-rwp`, ~0.3 s/epoch for the TCN). The 81 MB
aggregate dataset + a 68 MB per-player trajectory dataset are uploaded; jobs in `jobs/*.sh`. Classical
work + all metrics need no GPU.

---

## 8. Next steps
1. **Firepower v2 (Leu)** — average-rating-per-player fix (§5). Highest near-term ROI for this pillar.
2. **2026 out-of-time holdout (Henry)** — parse ~15–20 fresh 2026 Inferno demos, run the *frozen*
   EFB2 model once → the last validity gate before writing. (Harness can be built on request.)
3. **More data (together)** — the real lever to make spatial/deep models surpass classical is
   ~2–5× more matches and/or multi-map; also enables trajectory-level features to pay off.
4. **Paper draft** — methods + results are complete; `docs/results_checkpoint.md` + `docs/metrics.md`
   contain everything needed.

---

## 9. Repo map (for follow-up)
| area | files |
|---|---|
| Features | `src/features/{economy,mapcontrol,positional,bomb,firepower,assemble}.py`, `build_trajectory_dataset.py` |
| Classical models / eval | `src/models/{train_pipeline,calibration,phase_calibration,conditional_analysis,logistic_coefficients,permutation_importance,shap_analysis,extended_metrics,ensemble_oof}.py` |
| Deep models (Betty) | `src/models/deep/{tcn,gat,transformer}.py`, jobs `jobs/*.sh`, runbook `docs/cluster_runbook.md` |
| Docs | `docs/{results_checkpoint,metrics,methodology,firepower_pillar,midway_summary}.md` |

**Single best reference for current numbers:** `docs/results_checkpoint.md` (the model matrix) and
`docs/metrics.md` (every metric + values + interpretation).
