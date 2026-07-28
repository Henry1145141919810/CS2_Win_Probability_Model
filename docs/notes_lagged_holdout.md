# Notes — the firepower holdout, all THREE variants (2026)

> **2026-07-25 update — the same-year variant is in, and it settles the question.**
> Leu completed the 2026 same-year stats (82/82). We now have all three constructions of the skill
> prior, and **firepower fails to beat the demo-only model (EB2) under every one of them** — including
> the leaky best-case (same-year 2026, full coverage + oracle same-era knowledge). Figure:
> `outputs/figures/paper/F6_firepower_three_variants.png`.
>
> | EFB2 out-of-time AUC | broken | lagged-2025 | same-year 2026 | EB2 (no fp) |
> |---|---|---|---|---|
> | logreg | 0.8236 | 0.8423 | 0.8450 | **0.8474** |
> | xgb | 0.8344 | 0.8401 | 0.8399 | **0.8497** |
> | lgbm | 0.8325 | 0.8393 | 0.8386 | **0.8501** |
> | catboost | 0.8386 | 0.8377 | 0.8391 | **0.8494** |
> | rf | 0.8159 | 0.8378 | 0.8422 | **0.8439** |
>
> **Same-year EFB2 beats EB2 on NO model** (−0.002 to −0.011). Contested-AUC with firepower is worse
> in every variant (0.55–0.59 vs EB2's 0.63–0.66). **Calibration:** only the broken build is
> miscalibrated (intercept −0.36 → −0.01); both coverage-complete builds sit near zero. **Coverage is
> the whole calibration story; the year of stats barely matters** (same-year ≈ lagged — consistent with
> the count confound, r=0.987: the feature mostly counts players, so which season's ratings you sum is
> nearly irrelevant). **Verdict: ship EB2, no firepower — settled across all three variants.**
>
> ---
>
> ## (original note, broken vs lagged — kept for the record)
>
# Notes — the lagged-prior firepower holdout (2026, leak-free)

**Date:** 2026-07-25 · **Run:** `src/models/holdout_2026.py --test data/test_dataset_2026_lag2025.parquet --tag lag2025`
**Result files:** `outputs/holdout_2026_lag2025.csv`, `outputs/figures/holdout_2026_lag2025.png`,
`outputs/figures/paper/F5_firepower_recovery.png`

## What we did

Leu's scrape (commit `bca7f26`) completed the **2025** stats for the 17 previously-missing players →
**82/82 coverage**. We added an opt-in `FIREPOWER_YEAR_LAG` env var to the firepower pillar and
rebuilt the 2026 holdout so that each 2026 match looks up the **previous season's (2025)** skill
stats — the leak-free, deployment-realistic construction (a live system at match time only knows last
season). Then re-ran the full classical holdout (5 models × 4 sets + ensemble, B=500 CIs) as a
**separate, disclosed** evaluation (the original touch-once same-year run stands unchanged).

Task A (2026 same-year stats) was **not** done by Leu, so the same-year "fixed" variant is not
available — but the lagged variant is the more important one, so this is fine.

## The coverage gap is closed

| 5v5 `ct_rating_sum` | training | broken same-year 2026 | **fixed lagged-2025 2026** |
|---|---|---|---|
| mean | 5.28 | 3.66 | **5.36** |
| % below 3 | 0.0% | 11.8% | **0.0%** |
| % non-zero | 100% | 91.6% | **100%** |

The feature now means the same thing in 2026 as in training.

## Headline result (be precise — this is nuanced)

**The lagged fix repairs the catastrophe, but firepower still does not earn its place.**

### 1. Calibration FULLY recovers
The broken run's smoking gun is gone. EFB2 calibration intercept (negative = over-confident in CTs):

| model | broken | **fixed** | ECE broken → fixed |
|---|---|---|---|
| logreg | −0.359 | **+0.089** | 0.071 → 0.021 |
| xgb | −0.118 | **+0.074** | 0.032 → 0.028 |
| lgbm | −0.157 | **+0.068** | 0.040 → 0.031 |
| catboost | −0.062 | **+0.050** | 0.026 → 0.027 |
| rf | −0.008 | **+0.071** | 0.035 → 0.013 |

Every model flips from over-confident to the **benign +0.05…+0.09 base-rate drift** that the
no-firepower sets show (the 0.445 → 0.512 shift). ECE returns to healthy. The calibration disaster was
**entirely** a data-coverage artifact.

### 2. AUC collapse LARGELY repairs
EFB2 out-of-time AUC, broken → fixed: logreg **0.8236 → 0.8423** (+0.019), rf **0.8159 → 0.8378**
(+0.022). The catastrophic −0.028 drop shrinks to about −0.010.

### 3. But firepower STILL does not transfer as a net positive
Even with correct, leak-free data, **EFB2 < EB2 on every model out-of-time**:

| model | EB2 (no fp) out | EFB2 (fp) out | firepower costs |
|---|---|---|---|
| logreg | 0.8474 | 0.8423 | −0.0051 |
| xgb | 0.8497 | 0.8401 | −0.0096 |
| lgbm | **0.8501** | 0.8393 | −0.0107 |
| catboost | 0.8494 | 0.8377 | −0.0117 |
| rf | 0.8439 | 0.8378 | −0.0062 |

And contested-AUC is notably worse with firepower: EFB2 **0.589** vs EB2 **0.637** out-of-time.
The EFB2 in→out delta is still negative (−0.007 to −0.010); EB2 is flat/positive (0.000 to +0.001).

## Interpretation (for the paper)

The in-sample edge of firepower (logreg EFB2 0.8519, the study's best in-sample) is an **in-sample
artifact that reverses out-of-time**. Two distinct lessons, which the two variants separate cleanly:

1. **A skill prior's inference-time data dependency can fail catastrophically and silently.** The
   coverage gap (unfixed) → −0.028 AUC and a −0.36 calibration intercept. This is a *plumbing* failure
   and it is fixable: a lagged, coverage-complete construction repairs calibration completely and most
   of the AUC.
2. **Even correctly fed, the skill prior adds no out-of-time value here.** EB2 (no firepower) remains
   the best and most robust configuration on every model and on contested rounds. The demo-derived
   pillars already capture what matters; the external skill prior is not worth its operational cost
   (an external per-era database that must be scraped, coverage-monitored, and lagged).

**Recommendation stands and is now sharper: ship EB2 (no firepower).** Not because firepower's data is
unfixable — we fixed it — but because a correctly-constructed, leak-free skill prior still does not beat
the demo-only model out-of-time.

## Why this is a *stronger* paper result than before

A reviewer seeing only the broken run would say "just fix the data." We fixed it, disclosed both
variants, and firepower **still** fails to transfer. That converts a plumbing bug into a rigorous,
general negative result about externally-sourced priors — which is the most transferable contribution
of the paper.

## Caveats / honesty

- **Mild train/serve skew:** the EFB2 models were *trained* on same-year firepower and *tested* on
  lagged firepower. This is the realistic deployment setup (train on history, serve with last season),
  and the lagged 2026 distribution (5.36) matches training (5.28) closely, so the skew is small. Worth
  one sentence in the paper.
- Deep models (TCN/Transformer) out-of-time on this holdout are pending on Betty (`docs/BETTY_benchmark_guide.md`).
- GAT out-of-time deferred (needs a 2026 trajectory dataset).
