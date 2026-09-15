# Round to map: results of the built rungs (in-time, 2024–25)

**Date:** 2026-09-13 · **Plan:** [plan_round_to_map.md](plan_round_to_map.md) · **Code:**
`src/models/map_wp.py` (ladder + battery + controls), `src/viz/map_wp_figure.py` (F14) ·
**Outputs:** `outputs/map_wp_ladder.csv`, `map_wp_calibration_margin.csv`, `map_wp_phase.csv`,
`map_wp_by_round.csv`, `map_wp_trajectory.csv`, `map_wp_learning_curve.csv`, `map_wp_roundstart.parquet`,
`map_wp_persecond.parquet`, log `outputs/map_wp_full.log` · **Figure:** `outputs/figures/paper/F14_map_wp.png`.
**Data:** raw supplement only (per-second first snapshots, `rounds` + `ticks` channels, reference CSVs).
5-fold GroupKFold by map; paired match-block bootstrap B=500; 4,866 round starts, 220 maps, 476,595
snapshots. **The 2026 holdout was scored once, after the in-time design was frozen (§7).**

## 1. The ladder at round-start resolution

| Rung | log-loss | Brier | AUC | cal. slope | paired Δlog-loss vs M0 (95% CI) |
|---|---:|---:|---:|---:|---|
| **M0** i.i.d. rounds, side rate only (state model) | 0.5673 | 0.1936 | 0.777 | 0.73 | — |
| **M1** economy chain | **0.5624** | **0.1930** | 0.773 | 0.88 | −0.0048 (−0.0183, +0.0058) |
| **M2** + latent strength (n0 = 5–20 by inner CV) | 0.5633 | 0.1933 | 0.774 | 0.84 | −0.0039 (−0.0151, +0.0049) |
| **M2c** M2 recalibrated (Platt on the chain logit) | 0.5645 | 0.1939 | 0.770 | 0.96 | −0.0026 (−0.0188, +0.0120) |
| **M3** XGBoost on the chain logit + state | 0.5750 | 0.1977 | 0.760 | 0.90 | +0.0080 (−0.0091, +0.0234); vs M1 **+0.0127 (+0.0048, +0.0208)** |

Reading: the economy chain lowers log-loss by 0.005 over the pure state model, which is the right
direction and the right size but **not significant on 220 map outcomes** (the paired CI half-width at
this level is ≈ 0.012, six times wider than at the round level). The latent-strength term adds nothing
(n0 small, Δ ≈ 0). The flexible learner on top is **significantly worse** than the chain: with 220
outcomes it fits noise, as predicted. AUC barely moves across rungs because ranking is set by the score.

## 2. Where the state model is biased, and how much the chain repairs

Calibration by score margin (X − Y at round start), realised X map-win rate vs predicted:

| margin | n | realised | M0 | M1 | M2c |
|---:|---:|---:|---:|---:|---:|
| −6 | 296 | 0.054 | 0.030 | 0.062 | 0.076 |
| −4 | 263 | 0.209 | 0.138 | 0.190 | 0.205 |
| −2 | 470 | 0.340 | 0.297 | 0.326 | 0.330 |
| 0 | 738 | 0.504 | 0.510 | 0.510 | 0.498 |
| +2 | 428 | 0.661 | 0.735 | 0.708 | 0.682 |
| +3 | 311 | 0.720 | 0.817 | 0.778 | 0.750 |
| +4 | 215 | 0.823 | 0.880 | 0.839 | 0.812 |
| +6 | 285 | 0.954 | 0.975 | 0.954 | 0.940 |

- The i.i.d. model is over-confident at every margin (slope 0.73): 7–10 points at ±3–4.
- The chain removes the bias on the trailing side entirely and halves it on the leading side
  (+3: 0.817 → 0.778 vs 0.720 realised). The residual leading-side over-confidence (3–6 points) is a
  candidate for the next refinement: the loss-bonus streak, which the current state (equipment tier
  only) cannot see, makes a trailing team's *future* economy better than its present tier implies.
- Recalibration (M2c) flattens the margin curve (slope 0.96) but does not lower log-loss: the
  remaining error is not a monotone distortion.

By phase, the chain beats the state model in half 1, at round 13, and in half 2 (0.602 vs 0.607;
0.484 vs 0.490; 0.488 vs 0.497) and loses at round 1 and in overtime (0.697 vs 0.693; 0.641 vs 0.621).
Both losses are small-sample effects: at 0–0 the chain's value moves 1–3 points off 0.5 through the fitted
pistol/side coefficients, and the 290 overtime rounds are too few to estimate anything OT-specific
(see §5, the overtime lesson).

## 3. Trajectory honesty at map level

Round-start paths (one value per round, 220 paths):

| model | increment mean | QV ratio | write-off ≤ 5% obs / bench | ≤ 10% | ≤ 20% | D_upper (p) |
|---|---:|---:|---|---|---|---|
| M0 | −0.000 | 0.86 | 5.5% / 5.3% | 10.5% / 11.1% | 25.9% / 25.0% | 0.032 (0.65) |
| M2 | −0.000 | 1.07 | 3.2% / 5.3% | 8.2% / 11.2% | 21.4% / 25.2% | 0.011 (0.94) |
| M3 | +0.002 | 1.17 | 0.0% / 5.4% | 10.5% / 11.3% | 19.1% / 25.4% | 0.000 (1.00) |

The chain's round-start path is an honest martingale: zero-mean increments, realised quadratic
variation within 7% of the terminal variance, and the eventual winner is written off *less* often than
the calibrated-martingale law allows (no upper-tail inflation).

Per-second path (M4, chaining the production round model into the chain, 476,595 snapshots):

| metric | value |
|---|---|
| log-loss / Brier / AUC (per snapshot, vs map winner) | 0.5566 / 0.1903 / 0.781 |
| calibration slope / intercept | 0.84 / −0.06 |
| freeze-end consistency: mean \|P_map(first snapshot) − M2\| | 0.010 |
| **QV ratio** | **1.40** |
| write-off ≤ 5% / 10% / 20%, observed vs benchmark | 7.3% vs 5.3% / 13.2% vs 11.3% / 30.0% vs 25.3% |
| D_upper (p) | 0.080 (0.078) |

The per-second map curve uses within-round information (log-loss 0.557 vs 0.563 at round starts) but
**over-reacts**: 40% excess quadratic variation (match-bootstrap 95% CI 1.26–1.55 on the ratio) and more
write-offs of the eventual winner than the martingale law permits, at every threshold. The round-start
path of the same chain sits at 1.07, so the excess is created inside rounds. This is the round model's own 30% excess (Study 1)
propagated and amplified through the chain: every within-round wobble of p_t is multiplied by the
value gap V(a+1, b) − V(a, b+1). It is the clearest case yet for a path-aware round forecast, and it
means the broadcast object should not be shipped as a raw chain of the current per-second model.

## 4. Controls

| Control | Result | Verdict |
|---|---|---|
| Label shift (each map's states paired with another map's winner) | log-loss rises from 0.56 to 0.89–1.06 for every rung | the fit is to the real labels, not to the pipeline |
| Synthetic i.i.d. maps from M0 (real schedule and end rule, tiers independent of outcomes) | M1 +0.0021 (−0.0011, +0.0059), M2 +0.0021 vs M0; M3 +0.0132 (+0.0060, +0.0213) | the chain finds no structure where none exists; the GBM invents some |
| Learning curve, OOF log-loss at 55 / 110 / 165 / 220 maps | M0 0.605 / 0.569 / 0.562 / 0.567; M1 0.602 / 0.563 / 0.557 / 0.562; M2 0.602 / 0.564 / 0.558 / 0.563; M3 0.642 / 0.576 / 0.575 / 0.575 | the structural rungs are flat from 110 maps (bias-limited, low variance); M3 is variance-limited and still 0.013 behind at 220 |
| Structural checks (fold 0) | V(0–0) 0.50, V(12–12) 0.52, V(12–0) 0.999, V(0–12) 0.001, monotone in a at every b < 12 | the DP respects the format |
| Format rules vs data | side schedule 4,843/4,866; end rule reproduces the curated winner on all maps whose clinching round survived trimming (209/220) | |

## 5. Lessons recorded

1. **Never fit an overtime-specific coefficient.** X (the first-half CT team) won 53% of the 290 OT
   rounds by chance; a fitted OT term turned that into V(12–12) = 0.60 and, through backward
   induction, biased every regulation state near a tie toward X (at level scores in half 2 the chain
   said 0.54 against a realised 0.51). Overtime rounds are full buys with no carry-over; they are
   treated as regulation full-vs-full rounds with the side effect only. After the fix V(12–12) is
   0.47–0.53 across folds.
2. **The state model's bias is directional and known.** It is over-confident for the leader because it
   ignores that money carries over and that the loser's economy recovers (loss bonus). The chain fixes
   the trailing side; the leading side needs the streak in the state.
3. **220 outcomes cannot rank models by log-loss.** Paired CIs at map level are ±0.012; the honest
   claims are the calibration-by-margin table, the trajectory identities, and the controls. Any deep
   rung must be judged the same way, and the pre-registered bar (Δlog-loss CI excluding zero, slope
   0.9–1.1, flatter learning curve than M3) is unlikely to be met at this sample.
4. **The map curve amplifies the round curve's dishonesty.** Fixing the round model's excess variation
   is a prerequisite for a per-second map product.

## 7. Out-of-time 2026: the single disclosed evaluation

Everything fitted on all 220 training maps (p_CT = 0.487, n0 = 20, V(0–0) = 0.49); scored once on the
27 maps / 564 round starts of 2026. Nothing was tuned afterwards. 2026 is CT-sided (CT round rate
0.543) and the CT-first team X won 59% of the maps, so the 2024–25 side constant is off by 0.056.

| rung | log-loss | Brier | AUC | cal. slope | intercept | paired Δlog-loss vs M0 (95% CI, 27 maps) |
|---|---:|---:|---:|---:|---:|---|
| M0 i.i.d., 2024–25 side rate | 0.5487 | 0.1771 | 0.796 | 0.80 | −0.14 | — |
| **M1** economy chain | **0.5363** | 0.1760 | 0.802 | **0.995** | −0.16 | −0.0124 (−0.059, +0.018) |
| M2 + strength | 0.5366 | 0.1758 | 0.802 | 0.95 | −0.17 | −0.0122 (−0.049, +0.011) |
| M2c recalibrated | 0.5346 | 0.1766 | 0.802 | 1.11 | −0.12 | −0.0142 (−0.084, +0.033) |
| M3 XGBoost on the chain | 0.5332 | 0.1764 | 0.807 | 1.17 | −0.11 | −0.0154 (−0.089, +0.040) |
| M0 with the 2026 side rate (diagnostic; uses the test labels) | 0.5297 | 0.1723 | 0.807 | 0.91 | +0.08 | −0.0187 (−0.069, +0.029) |
| M4 per-second chaining (55,271 snapshots) | 0.5214 | 0.1700 | 0.811 | 0.96 | −0.13 | |

- The chain's advantage over the i.i.d. model **transfers** (−0.012 out-of-time vs −0.005 in-time) and its
  calibration slope is 0.995 on unseen maps. Nothing is significant on 27 outcomes (CIs ±0.05), and
  the M3 point estimate leading is noise (it was significantly worse in-time).
- The **side shift alone is worth 0.019**: re-estimating one constant on 2026 beats every rung. A
  production system needs the side rate tracked per era (it is one number), exactly the
  intercept-drift lesson from the round-level holdout.
- By margin, 2026 realised rates at −4 … −2 (0.03 / 0.12 / 0.19, n = 25–31 each) sit below every rung
  (M1 0.12 / 0.26 / 0.32): the trailing CT-first team came back less often than in 2024–25. With
  30 rows per bucket this is not interpretable beyond "the shift is in the side constant".
- Trajectory: round-start paths under-dispersed (QV 0.66–0.87) and write-offs below the law; the
  per-second path shows **no over-reaction on 2026** (QV 1.09, D_upper p = 0.89) against 1.40 in-time.
  27 paths cannot settle this; the in-time excess stands on its own CI and the discrepancy is an
  open item for the path-aware round model.

Absolute levels are lower than in-time (0.536 vs 0.562) because the 2026 maps are more lopsided (59%
map wins for X, CT-sided rounds), not because the model improved; log-loss is not comparable across
sets with different base rates.

## 8. Built / not built

Built and run: M0, M1, M2 (with inner-CV n0), M2c, M3, M4 per-second chaining, the round-start battery
with paired bootstrap, calibration by margin and phase, structural checks, martingale increments and
quadratic variation, the extreme-path benchmark at both resolutions, the label-shift control, the
synthetic i.i.d. control, the learning curve, and figure F14.

Also run: the single disclosed out-of-time scoring (§7; `--holdout`, log `outputs/map_wp_holdout2026.log`).

Not built: the loss-bonus streak in the economy state (M1b); the lagged-rank prior μ for M2 (needs
2023 HLTV tables); the Betty rungs M5a (round-sequence GRU/Transformer), M5b (autoregressive or
diffusion simulator of the remaining rounds), M5c (per-second TCN/Transformer retargeted to the map
label); an era-tracked side constant; series (Bo3) context.
