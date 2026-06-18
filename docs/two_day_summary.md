# CS2 Win-Probability Model — Two-Day Progress Summary

**For:** project partner · **Author:** Henry (UPenn WSABI) · **Period:** 2026-06-16 → 2026-06-18
**Repo:** https://github.com/Henry1145141919810/CS2_Win_Probability_Model
**Map:** de_inferno · **Target:** live per-second round win-probability, arXiv + MLSA 2027

---

## 1. One-paragraph recap

We are building a live, per-second CS2 round win-probability model for de_inferno that
augments the established economy baseline (Xenopoulos/ESTA-style) with novel **spatial**
feature pillars — Voronoi map control, line-of-sight/territory control, and tactical
readiness — and evaluates them rigorously across model architectures. Over these two days
the project went from a small pilot to an at-scale, fully-evaluated study: **220 pro demos /
476,595 per-second snapshots**, three competing map-control formulations, a five-architecture
model matrix with bootstrap confidence intervals, calibration testing, interpretability, and
qualitative case studies. The headline scientific result is honest and defensible: **map
control adds a small but statistically robust lift overall (~+0.003 AUC), concentrated in
genuinely contested rounds (~+0.013) where the economy baseline collapses to a coin flip.**

---

## 2. Dataset & pipeline

- **220 Tier-1 demos** (de_inferno), de-duplicated and tier-filtered (dropped a qualifier and
  a women's-team game; one demo held off-list). Parsed with awpy v2.0.2, tickrate forced to 64.
- **476,595 snapshots**, one row per second from freeze-end to round end; label `ct_won`
  (base rate 0.445). Grouped by `match_id` for cross-validation (never split a match).
- Five parquet channels per demo (ticks / kills / rounds / bomb / smokes). Grenades dropped
  (446 MB, unused). Memory-guarded parsing (awpy peaks ~6.7 GB/demo on a 16.8 GB laptop).
- **Feature pillars implemented:** (1) economy/combat, (2) map control (3 variants, below),
  (4) tactical readiness + bomb/rotation. Pillar (3) firepower is the main remaining gap.

---

## 3. The three map-control models (the core scientific contribution)

We built and compared three increasingly realistic ways to quantify "who controls the map":

| Model | Definition | Verdict |
|---|---|---|
| **Voronoi** | Each nav area → nearest living player's team (area-weighted). Greedy/total. | **Best single predictor; kept.** Robustly significant in both linear and nonlinear models. |
| **Grey (LOS+FOV+smoke)** | An area is held only if a living player is in range **and** has line-of-sight (3060×3060 nav visibility matrix) **and** is facing it (FOV from yaw) **and** it isn't smoked. ~60-80% of the map is "grey"/uncontested at any instant — realistic for CS. | **Negative result.** Physically faithful but *flickers* tick-to-tick (yaw is twitchy), so it does **not** beat simple Voronoi at predicting the round outcome. |
| **Territory (memory + decay)** | The grey model **with memory**: cleared space stays a team's for 15 s without re-checking; FOV only gates *re-acquiring* neglected areas. Stateful per round. | **Recovers Voronoi-level predictiveness.** Confirms the key insight below. |

**Key insight — "realistic ≠ predictive unless stabilized."** The instantaneous, physically
correct sightline model is a *worse* predictor than crude proximity, because round outcomes
depend on *stable* territorial state, not on where 10 crosshairs happen to point this tick.
Adding memory/decay fixes the flicker and restores the signal. This is a clean, publishable
finding, and all three models are retained in the dataset for the ablation narrative.

A 3-row visualization (`src/viz/mapcontrol_viz.py`) renders all three side-by-side over the
radar (with facing lines and smokes drawn) to show this process visually.

---

## 4. Model matrix — 5 architectures × {A, E, ET}

5-fold GroupKFold out-of-fold; primary metric AUC; B=500 match-level block bootstrap.
**A** = economy only (baseline). **E** = economy + Voronoi + tactical/bomb. **ET** = E + territory.

| Model | A (AUC) | E (AUC) | ET (AUC) | Lift E−A (95% CI) | ECE (E) |
|---|---|---|---|---|---|
| **Logistic** | 0.8465 | **0.8493** | 0.8493 | +0.0028 (+0.0014, +0.0042) | 0.016 |
| XGBoost (tuned) | 0.8443 | 0.8476 | 0.8480 | +0.0034 (+0.0020, +0.0047) | 0.012 |
| LightGBM | 0.8448 | 0.8479 | 0.8476 | +0.0031 (+0.0018, +0.0042) | 0.014 |
| CatBoost | 0.8448 | 0.8478 | 0.8474 | +0.0030 (+0.0018, +0.0043) | 0.011 |
| Random Forest | 0.8228 | 0.8426 | 0.8428 | +0.0198 (+0.0165, +0.0228) | 0.011 |

**Takeaways**
- **Every model, every feature set: the control lift is statistically significant** — all
  bootstrap CIs exclude 0, DeLong p ≈ 0.
- The four well-specified models (logistic + the three GBMs) **agree** on the honest
  magnitude: **~+0.003 AUC** aggregate. ET ≈ E everywhere → territory is *redundant* with
  Voronoi once Voronoi is in the model.
- **Logistic regression is the best model (0.8493)** and ties the GBMs → the signal is
  essentially **linear**; the gradient-boosted models are interchangeable.
- **Random Forest is the cautionary cell.** Its economy baseline *underfits* (0.8228, poorly
  calibrated at ECE 0.042), so control "helps" a huge +0.0198 and fixes its calibration
  (ECE → 0.011). That large lift is a **weak-baseline artifact, not extra spatial signal** —
  which is exactly why we report the lift *across architectures* rather than trusting one model.

---

## 5. Metrics — primary + a metric that tells the honest story

- **Primary (literature-comparable):** AUC-ROC, log-loss, Brier.
- **Complementary (now standard outputs of the harness):** ECE (calibration), BSS (Brier
  Skill Score), and **contested-AUC (cAUC)** — AUC computed only on genuinely *contested*
  snapshots (equal players alive **and** |equipment diff| ≤ 1500; 58,368 snapshots).
- **Why cAUC matters:** overall AUC is dominated by easy lopsided snapshots (5-v-2, eco vs
  full-buy) that economy already nails, which *dilutes* the spatial signal and makes the
  aggregate lift look tiny (+0.003). A "better metric" will **not** inflate that number — it
  is genuinely small on average. But cAUC correctly reports the lift **where it actually
  matters**: in even rounds the economy baseline's own AUC **collapses from ~0.85 to ~0.58**
  (near coin-flip), and the contested regime is exactly where added signal is valuable. This
  is the honest *and* stronger framing for the paper.

---

## 6. Where map control matters — conditional analysis

Restricting evaluation to contested subsets (XGBoost, OOF AUC):
- Economy baseline collapses in even rounds: all snapshots 0.832 → equal-alive 0.686 →
  even-econ 0.674 → equal-alive **and** even-econ **0.578** (~coin flip).
- Map-control lift is **larger where it matters**: equal-alive (half the data) E−A = **+0.0131**
  vs +0.009 overall; even-econ +0.0107; pre-plant +0.0103.
- The signal is **nonlinear in the hardest subsets** — XGBoost extracts the contested lift,
  logistic gets ~0 there (logistic wins on easy snapshots, XGBoost wins where it's hard).

Paper framing: *"map control is most informative exactly when economy fails (contested
rounds), and its hardest signal needs nonlinear models."*

---

## 7. Interpretability — what the model learned

`src/models/logistic_coefficients.py` emits standardized coefficients (z-scored inputs, so
magnitude = effect size, sign = direction toward CT win). Signs are all sensible, and the
*relative scale* is the real story — **economy dominates; map control is real but second-order.**
- **Economy/combat dominate (set A, ~0.5–0.9):** `t_players_alive` −0.89, `ct_health_total`
  +0.74, `ct_equipment_value` +0.72, `t_equipment_value` −0.61, `bomb_planted` −0.51; plus
  the single largest term `min_ct_dist_to_bomb` −0.78 (CTs far from bomb = retake = bad).
- **Map control fit on its own ≈ +0.11** (Voronoi `control_deficit`) — the strongest
  non-economy/non-bomb signal, but ~5–8× smaller than economy.
- **Voronoi vs territory, fit separately:** Voronoi deficit **+0.115** vs territory deficit
  **+0.066** — territory is *directionally the same but about half as strong*, and the two are
  only moderately correlated (r = 0.47). So territory is a related-but-noisier proxy, redundant
  with Voronoi at the AUC level (ET ≈ E), **not** an identical or stronger signal.
- **Caveat:** coefficients *inside* the full model are collinearity-inflated (e.g.
  `control_deficit` shows +0.27 there only because a duplicate column splits the weight) —
  the standalone fits above are the trustworthy effect sizes.

Output: console table + `outputs/logistic_coefficients.csv` + a bar chart.

---

## 8. Qualitative case studies — map control flipping the call

`src/viz/control_shift_examples.py` fits OOF logistic A (economy) vs E (economy + **map
control only**, deliberately excluding tactical/bomb so the shift is purely spatial), then
finds contested rounds where control moves the win probability **toward the eventual winner**.

**Headline example — b8 vs FlyQuest (map 3), round 12, a 4-v-4 with even economy:** the
economy-only model hugs 0.50 (a coin flip) for the entire round, but the map-control model
consistently and correctly pulls toward T, who won (peak shift −22% at t+50 s). This single
figure makes the contested-AUC number concrete: *economy is blind in even rounds; control
sees the read.* Two more examples (one CT-favoring, one 1-v-1 clutch) are included, each with
a win-probability timeline plus the 3-model map at the key moment.

---

## 9. Calibration — are the probabilities honest?

`src/models/calibration.py` produces a reliability diagram with **per-bin bootstrap error
bars and a bootstrap 95% CI on ECE** (resampling whole matches, B=200). Result: **all
non-RF models are well-calibrated (ECE < 0.02)** — "the model said 70%" really means CT won
~70% of the time. This is now a standard, repeatable check in the pipeline, not a one-off.

---

## 10. Evaluation protocol (so the numbers are trustworthy)

- **5-fold GroupKFold by match** — never split a match (rounds/snapshots within a match are
  correlated; folding by round leaks and inflates AUC).
- **DeLong's test** for each feature set vs the economy baseline on identical OOF predictions.
- **Match-level block bootstrap** (matches are the independent unit), B=500 for AUC and the
  E−A difference CI; a difference CI excluding 0 ⇒ significant.
- **Time-window analysis** at 5/10/15/20/25 s for the control-signal-emergence story.
- **Held out for the end:** a 2026 out-of-time test set (touched once) — CV drives all
  development so the holdout stays uncontaminated.

---

## 11. What's in the repo (for your review)

| Area | Files |
|---|---|
| Features | `src/features/{economy,mapcontrol,positional,bomb,assemble}.py`, `visibility.py` (LOS matrix) |
| Models / eval | `src/models/{train_pipeline,calibration,conditional_analysis,tune_xgb,logistic_coefficients}.py` |
| Visualization | `src/viz/{mapcontrol_viz,control_shift_examples,winprob_chart}.py` |
| Docs | `docs/methodology.md` (full protocol + every result), `docs/map_control_models.md`, this summary |

**Reproduce the headline table:**
`python src/models/train_pipeline.py --models logreg,xgb,rf,lgbm,catboost --sets A,E,ET --bootstrap 500`

> **Note for the partner:** figures live in `outputs/figures/` and are git-ignored
> (regenerable from the scripts above on the parsed dataset). The parsed dataset and demos
> are shared separately (not in git). If you want any specific figure committed to the repo,
> just ask.

---

## 12. Open items / next steps

1. **Pillar 3 — firepower** (per-player rating / aim stats). The clearest remaining lever for
   a larger lift; requires an HLTV per-player scrape + name mapping (manual, ToS-limited).
2. **Deep models on E** — GAT / TCN / Transformer / ensemble (cloud A100), mean ± std over
   20 seeds; the laptop covers the classical matrix.
3. **2026 out-of-time holdout** — final generalization check.
4. **Honest positioning of the spatial result** — small aggregate lift, robustly significant,
   concentrated in contested rounds; the three-control-model ablation and the calibration /
   contested-AUC story are the methodological contributions even if the raw AUC gain is modest.
