# Defuse-progress — full benchmark results (is it worthwhile?)

**Date:** 2026-08-30 · **Data:** re-parsed `training_dataset_defuse.parquet` (476,595 × 135, 220
matches, 0 overlap with 2026, 4,608 defusing rows = 0.97%) + `test_dataset_2026_defuse.parquet`
(55,271 × 135, 631 defusing rows). Run: `src/models/defuse_full_benchmark.py`. Figure: F11.

## Verdict: worthwhile, but as a curve-honesty fix, NOT an AUC improvement

The feature fires on **<1% of snapshots**, so by construction it cannot move a pooled metric — and it
does not. The value is entirely on the defusing rows, where it is large and unambiguous.

### 1. Pooled metrics barely move (expected)
EB2 vs EB2D, in-time / out-of-time AUC essentially identical (e.g. lgbm EB2 0.8493/0.8501 vs EB2D
0.8498/0.8491). Paired ΔAUC (EB2D−EB2) is tiny and mixed: a few marginally significant positives
(logreg IN +0.0003, OUT +0.0007), a few marginally significant negatives out-of-time (catboost
−0.0011, rf on EFB2D −0.0035) — all consistent with noise on a feature that touches ~1% of rows.
Contested-AUC unchanged. **Reading this as "no value" is the trap; the feature is not a discrimination
feature.**

### 2. Curve honesty on the defusing rows (the point) — decisive
Logistic, 2026 test, 631 defusing rows:

| defuse progress | actual CT win | EB2 (no feature) | EB2D (+ feature) |
|---|---|---|---|
| 0–20% | 0.909 | 0.910 | 0.904 |
| 20–40% | 0.924 | 0.904 | 0.920 |
| 40–60% | 0.951 | 0.917 | 0.954 |
| 60–80% | 0.975 | 0.926 | 0.971 |
| 80–100% | 0.983 | **0.924** | **0.979** |

EB2 is **flat** — a CT is 80% through defusing and the model has not moved off ~0.92. EB2D **tracks the
empirical curve** almost exactly. Aggregate on defusing rows:

- **log-loss −60%** (0.1580 → 0.0627), **Brier −62%** (0.0451 → 0.0172) for logistic.
- xgb: log-loss −36%, Brier −35% (trees learn part of it from `defuse_beats_fuse` already, but not all).

### 3. Not a `ct_won` relabel
39.6% of the 1,032 training defuse attempts are interrupted; on those EB2D stays well-calibrated too,
so the feature encodes the *live* state, not the outcome.

## Recommendation for the paper

**Include it, framed honestly as a per-second calibration (trajectory-honesty) case study**, not as an
accuracy gain. It is the same methodological argument the paper already makes twice:

1. **Pooled metrics hide what matters** — exactly the contested-AUC argument, now applied to a live
   state instead of an even round. A feature invisible in pooled AUC removes a 6-point calibration
   error at the most consequential moment a broadcast overlay shows.
2. **It is a concrete instance of the extreme-path / trajectory-honesty section** (v10): the model was
   systematically wrong on a specific trajectory segment, and a physics-derived feature fixes it.

Placement: a short subsection near the calibration / trajectory-honesty results, with Fig. F11. Keep
the feature quarantined (EB2D/EFB2D) and say so — folding an endgame-certainty signal into the
map-control sets would inflate them.

Honest caveats to state: fires on ~1% of rows; pooled AUC and contested-AUC do not move; a couple of
out-of-time paired deltas are marginally negative (noise). The claim is calibration on the defusing
rows, nothing more.

## Bonus from the re-parse
The re-parsed training table is a clean superset of the published one (same 476,595 × 220, base rate
0.445, plus the 4 defuse columns + 2 coverage diagnostics). It can become the canonical table if we
adopt the feature.

## Firepower v4 (Leu's separate exploration) — no paper change
Mean-encoding (fixes the count confound) is a fourth failed firepower construction: CV 0.8520
(marginal over EB2 0.8508), out-of-time 0.8458 < EB2 0.8474. Consistent with the existing firepower
negative; noted, no change needed.
