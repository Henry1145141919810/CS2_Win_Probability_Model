# Metrics — full glossary, values, and interpretation

Every metric used to evaluate the CS2 win-probability models, what it means, and the current
values across the matrix. All metrics are computed on **out-of-fold (OOF)** predictions:
5-fold GroupKFold by match → each snapshot is predicted by a model that never trained on its
match (leak-free, uses all data). Code: `train_pipeline.py` (primary), `extended_metrics.py`
(extended), `calibration.py` / `phase_calibration.py` (calibration), `conditional_analysis.py`.

---

## 1. Primary metrics (literature-comparable)
| metric | range (best) | measures | reading |
|---|---|---|---|
| **AUC-ROC** | 0.5–1 (↑) | discrimination: ranks a random CT-win above a random T-win; ignores calibration | ~0.85 strong |
| **Log-loss** | 0–∞ (↓) | proper score; punishes confident-wrong unboundedly; the training objective | ~0.46 (base-rate predictor 0.69) |
| **Brier** | 0–1 (↓) | proper score = MSE of probability vs 0/1 outcome | ~0.155 (base-rate 0.247) |

## 2. Complementary metrics (standard outputs)
| metric | range (best) | measures | reading |
|---|---|---|---|
| **ECE** | 0–1 (↓) | calibration: avg |pred% − actual%| over 10 equal-width bins | <0.02 = honest |
| **BSS** (Brier Skill Score) | ≤1 (↑) | Brier skill vs base-rate predictor | ~0.37 = 37% of baseline error removed |
| **contested-AUC** | 0.5–1 (↑) | AUC on even rounds only (equal alive & |Δequip|≤1500) — where economy collapses | ~0.58–0.60; our novel lens |

## 3. Extended metrics (added June 2026, `extended_metrics.py`)
| metric | range (best) | measures | reading |
|---|---|---|---|
| **Reliability (REL)** | 0–1 (↓) | Brier-decomposition calibration term = mean-sq (pred − observed) per bin | ~0 for all = calibrated |
| **Resolution (RES)** | 0–unc (↑) | Brier-decomposition discrimination term = how far bin outcomes spread from base rate | higher = more skill |
| **Uncertainty (UNC)** | const | irreducible base-rate variance = p̄(1−p̄) = 0.247; dataset constant | not model-specific |
| | | *Identity:* **Brier = REL − RES + UNC** | |
| **Sharpness** | (↑, if calibrated) | std of predictions; a model always predicting the base rate is calibrated but useless | ~0.30 |
| **Calibration slope** | 1 = perfect | fit outcome ~ a + b·logit(p); b<1 overconfident, b>1 underconfident (bin-free, TRIPOD) | 0.92–1.03 |
| **Calibration intercept** | 0 = perfect | the `a` above; net bias | ~0 |
| **adaptECE** | 0–1 (↓) | ECE with equal-MASS bins (removes the equal-width binning artifact) | ≈ ECE (so ECE is robust) |
| **KS-cal error** | 0–1 (↓) | bin-FREE calibration = max |cumsum(y−p)|/n (Kolmogorov–Smirnov style) | ~0.003–0.008 |

*Motivation for the bin-free trio (slope/intercept, adaptECE, KS): binned ECE wobbles with bin
count; the esports-eval paper (Choi et al. 2023, arXiv:2309.06248, "Balance score") argues for a
bin-free calibration estimate. We use the standard bin-free measures above (the exact Balance-score
formula wasn't recoverable from the paper); they confirm our calibration conclusions are not a
binning artifact.*

## 4. Comeback / tail honesty (`extended_metrics.py`)
For a LIVE win-prob model, how trustworthy are extreme calls ("this side is losing")?
- **Tail calibration:** observed win-rate inside the extreme probability bins ([0,5%], …, [95,100%]).
- **Comeback rate:** per round, the lowest win-prob the model ever gave the *eventual winner*;
  what fraction of rounds dip ≤20/10/5%.
- **Comeback calibration:** of snapshots where a side is ≤10%, how often it actually comes back.

## 5. Tests & uncertainty
- **DeLong's test** — p-value for AUC vs baseline on the same OOF preds.
- **Match-level block bootstrap (B=500)** — resample the 220 matches → 95% CI on every metric
  (all 9 architectures).
- **Multi-seed mean±std** — deep-model training instability.
- **Reliability diagram, calibration-over-time, temperature/isotonic** — calibration diagnostics.
- **Time-window AUC (5–25 s)** — control-signal emergence. **Conditional/subset AUC** — where signal lives.
- **Win-prob CI band** — epistemic uncertainty on the live per-second curve.

---

## 6. Current values — extended metrics across the matrix (OOF)
| model | AUC | REL↓ | RES↑ | sharp | cal-slope | ECE | adaptECE | KScal | BSS | cAUC |
|---|---|---|---|---|---|---|---|---|---|---|
| logreg/A | .8465 | .0006 | .0888 | .303 | 0.987 | .0192 | .0188 | .0072 | .361 | .587 |
| logreg/E | .8493 | .0004 | .0901 | .307 | 0.970 | .0157 | .0171 | .0071 | .367 | .590 |
| logreg/EFB2 | .8515 | .0005 | .0915 | .310 | 0.960 | .0159 | .0167 | .0072 | .372 | .596 |
| xgb/EFB2 | .8498 | .0003 | .0910 | .308 | 0.972 | .0129 | .0134 | .0059 | .370 | .591 |
| lgbm/EFB2 | .8498 | .0003 | .0911 | .307 | 0.972 | .0118 | .0110 | .0055 | .370 | .593 |
| catboost/EFB2 | .8483 | .0004 | .0901 | .304 | 1.001 | .0177 | .0167 | .0058 | .367 | .586 |
| rf/EFB2 | .8442 | .0004 | .0885 | .300 | 0.992 | .0139 | .0145 | .0061 | .360 | .574 |
| transformer | .8473 | .0002 | .0896 | .302 | 0.991 | .0091 | .0108 | .0072 | .365 | .568 |
| tcn | .8489 | .0002 | .0903 | .310 | 0.919 | .0118 | .0121 | .0061 | .368 | .572 |
| **SOFT-VOTE(4)** | **.8531** | **.0001** | **.0926** | .304 | 1.033 | **.0090** | **.0082** | **.0028** | **.378** | .589 |

(Full per-cell CSV: `outputs/extended_metrics.csv`. Classical AUC/log-loss/Brier/ECE/BSS/cAUC for
all 5 models × 9 sets: `outputs/full_matrix_firepower.log`.)

## 7. Key insights from the extended metrics
1. **The accuracy gain is all RESOLUTION, not calibration.** economy→all-pillars: Reliability stays
   ~0 (already calibrated) while Resolution rises 0.0888→0.0915 → the pillars add *discriminative
   skill without costing honesty*. The cleanest framing of "the features help."
2. **Calibration claims are robust to binning** — adaptECE & KS-cal track ECE; not a binning artifact.
3. **Calibration slope adds direction:** all models 0.92–1.03 (near-perfect). TCN mildly overconfident
   (0.919); the soft-vote mildly under-confident (1.033, regression-to-mean of averaging); catboost
   dead-on (1.001).
4. **The ensemble wins on BOTH reliability and resolution** (REL 0.0001, RES 0.0926, KS-cal 0.0028) —
   why it's the best overall model, by construction not luck.
5. **Comeback/tail honesty:** says 5%→wins 0.7%, says 10%→6.8%, says 15%→15.4% (honest write-offs).
   Eventual winner dipped ≤10% in 7.2% of rounds (real comebacks); of ≤10% states, 1.9% came back vs
   ~2.3% predicted (calibrated even at the extreme). Mild overconfidence only in the 80–90% band.

## 8. Modern-practice alignment
The field's standard battery for win-prob models = AUC + log-loss + Brier + calibration (ECE /
reliability), and the esports-eval literature explicitly warns accuracy/AUC alone is insufficient
because stakeholders use the probabilities directly — which is exactly why we lead with calibration.
Our additions (Brier decomposition, calibration slope/intercept, bin-free calibration, comeback/tail)
are standard in forecast verification (URR decomposition) and clinical prediction (TRIPOD slope), and
the comeback/tail honesty is a live-win-prob-specific contribution rarely reported.
References: Choi et al. 2023 (arXiv:2309.06248); Xenopoulos/ESTA (arXiv:2011.01324, 2209.09861);
Bröcker reliability/resolution decomposition (arXiv:0806.0813).
