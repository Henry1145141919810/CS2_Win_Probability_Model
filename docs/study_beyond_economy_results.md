# Studies 1 & 2 — Beyond-Economy Signal and the Contested Ceiling

Process and results. Plan: [plan_beyond_economy.md](plan_beyond_economy.md). Code:
`src/models/residual_analysis.py` (Study 1), `src/models/contested_study.py` (Study 2).
All evaluation is 5-fold GroupKFold by match; uncertainty is match-level bootstrap.

Status: methodology recorded; results tables filled from the run logs (see below).

---

## Study 1 — Residual analysis (beyond-economy signal)

### Method
1. **Economy out-of-fold prediction.** For each model, `p_A = P(ct\_won | economy)` via 5-fold
   GroupKFold, then the economy logit `z = logit(p_A)`.
2. **Stacked-logit comparison.** A model on `[z]` alone recovers economy; a model on `[z, X_spatial]`
   adds the spatial block, which can therefore only act through the residual economy leaves behind.
   `X_spatial` = Voronoi map control + tactical + bomb-live + defuse-race (set EB2 minus economy;
   **no firepower**), 60 columns.
3. **Metrics and significance.** AUC, log-loss, Brier, BSS, contested-AUC, in-time (OOF) and
   out-of-time (2026 same-year holdout). The beyond-economy contribution is the **paired** match-level
   bootstrap CI of the difference `(+spatial) - (base)`; significant when the ΔAUC CI excludes zero.
4. **FWL orthogonalization (diagnostic).** Partial economy out of the outcome (`y_res = y - p_A`) and
   of each spatial feature (`s_res = s - E[s|economy]`, OOF linear regression), then rank features by
   the partial correlation `corr(s_res, y_res)`. `economy_r2` reports how much of each feature economy
   already explains (high = economy re-encoded, as with the firepower count confound).

### Results

**Beyond-economy signal is significant and transfers out-of-time.** Adding the spatial/bomb block on
top of the economy logit (so it can only act through the economy residual):

| Model | in-time base → +spatial | in-time ΔAUC (95% CI) | out-of-time ΔAUC (95% CI) |
|---|---|---|---|
| Logistic | 0.8461 → 0.8496 | +0.0035 (+0.0018, +0.0053) | +0.0048 (+0.0009, +0.0087) |
| XGBoost | 0.8428 → 0.8484 | +0.0056 (+0.0038, +0.0076) | +0.0033 (+0.0011, +0.0058) |
| LightGBM | 0.8423 → 0.8485 | +0.0061 (+0.0045, +0.0078) | +0.0033 (+0.0010, +0.0053) |
| CatBoost | 0.8428 → 0.8480 | +0.0051 (+0.0035, +0.0067) | +0.0024 (+0.0001, +0.0046) |

**Every interval excludes zero, in-time and out-of-time, on all four models.** After removing what
economy already explains, the spatial and bomb features add a small but robust and transferable
+0.004 to +0.006 AUC in-time and +0.002 to +0.005 out-of-time. The contested-AUC delta, by contrast,
is **not** significant (for example logistic dcAUC +0.005, CI −0.015 to +0.025): the beyond-economy
signal helps in partially-decided rounds, not in even ones. This is consistent with Study 2, where the
even-round outcome is near-irreducible.

**FWL orthogonalization: the defuse-race features carry the most economy-orthogonal signal.** Top
features by partial correlation of the economy-residualised feature with the economy-residualised
outcome, with `economy_r2` (how much economy already explains the feature):

| Feature | partial corr | economy_r² |
|---|---|---|
| `defuse_margin_kit` | 0.047 | **0.15** |
| `defuse_time_margin` | 0.046 | **0.15** |
| `defuse_contest_margin` | 0.045 | 0.65 |
| `min_ct_dist_to_bomb` | −0.045 | 0.69 |
| `min_ct_path_to_bomb` | −0.043 | 0.68 |
| `ct_bomb_local_control` | 0.041 | 0.42 |

The two strongest beyond-economy features are the defuse-race margins, and they have the **lowest**
`economy_r2` (≈ 0.15), meaning they are largely orthogonal to economy and carry genuinely new
information. By contrast, `min_ct_dist_to_bomb` has a high `economy_r2` (≈ 0.69): its apparent
importance is substantially economy re-encoded. This corroborates the paper's defuse-race finding with
a rigorous partialling-out, and mirrors the firepower count-confound diagnostic.

*Figure: F7 (`outputs/figures/paper/F7_residual.png`).*

---

## Study 2 — The contested-round ceiling

All three analyses run at the **snapshot level**, because the round-level contested slice is
underpowered (a prior run gave a ±0.05 AUC CI on ~200 even rounds). Match-level bootstrap throughout.

### (A) Contested-AUC by alive-state and graded evenness
- **By alive-state:** the even-and-even-economy subset is split by headcount into 1v1 / 2v2 / 3v3 /
  4v4 / 5v5 (`ct_alive == t_alive` and `ct_alive + t_alive ∈ {2,4,6,8,10}`, and `|Δequip| ≤ $1500`).
  Contested-AUC is the generalist xgb-EB2 OOF restricted to each subset, with match-bootstrap CIs and
  the snapshot/round/match counts.
- **Graded evenness:** with equal alive fixed, sweep the equipment-difference threshold
  ($10000 → $250) to trace how predictability falls as rounds become more even.

### (B) Information-saturation curve
Contested-AUC across increasingly rich representations, on the same contested subset:
economy (A) → +spatial (E) → +bomb (EB2) → +firepower (EFB2), all xgb, plus the deep models
(TCN, Transformer) via their saved OOF predictions joined on `(match_id, tick)`. A plateau that holds
even for the models with the most information indicates a ceiling rather than a feature-engineering
limitation.

### (C) Matching-based Bayes-error estimate (the model-free ceiling)

**This is the direct test of "are even rounds irreducibly random?".** For each even snapshot we find
its nearest neighbours in observable-state space and measure how often near-identical states end
differently.

**Snapshot selection (documented exactly):**
- **State descriptor** for "how similar are two situations": the observable state =
  economy + Voronoi map control + tactical/bomb columns (`ECONOMY_COLS + MAPCONTROL_COLS + TACTICAL`),
  standardised. This captures headcount, equipment, health, map control (aggregated positions),
  utility, AWP presence, bomb geometry, and time.
- **Neighbours from DIFFERENT matches only.** Snapshots from the same round share the round label, so
  same-match neighbours are excluded; otherwise disagreement would be trivially zero. We over-fetch
  `K_BUF = 200` candidates and keep the first `K = 25` from other matches.
- **Sampling for runtime:** up to `POOL_CAP = 25000` snapshots form the neighbour pool per category
  (subsampled if larger) and up to `N_QUERY = 2500` queries are drawn per category.
- **Categories (labelled):** the five even alive-states (1v1 … 5v5, each with even economy), the
  pooled "all even (contested)" set, and a **control**: a one-player man-advantage state
  (`|ct_alive − t_alive| = 1`), which should show LOW disagreement, validating that the method detects
  determinism where it genuinely exists.

**Metrics per category:**
- `disagree_rate` D = mean over near-twin pairs of P(neighbour outcome ≠ query outcome). D → 0.5 means
  the state does not determine the outcome (aleatoric); D → 0 means it does.
- `oracle_AUC` = AUC of the local neighbour outcome-frequency as a predictor of the true outcome. A
  model that perfectly memorised the local outcome distribution cannot exceed this, so it is an
  empirical ceiling on achievable AUC for that category. Reported with match-bootstrap CI.
- Snapshot, round, and match counts per category (for power/transparency).

### Results

**Headline: the "0.58 contested ceiling" is dominated by pre-engagement 5v5 rounds, which are provably
near-irreducible. No representation or model breaks it.**

**(A) Contested-AUC by alive-state (generalist xgb-EB2 OOF).** Even-and-even-economy snapshots, split
by headcount:

| State | n snapshots | n rounds | contested-AUC (95% CI) | CT win-rate |
|---|---|---|---|---|
| 1v1 even | 3{,}515 | 413 | 0.693 (0.635, 0.749) | 0.402 |
| 2v2 even | 3{,}905 | 432 | 0.734 (0.669, 0.791) | 0.289 |
| 3v3 even | 3{,}633 | 477 | 0.706 (0.651, 0.767) | 0.302 |
| 4v4 even | 5{,}977 | 577 | 0.637 (0.557, 0.711) | 0.424 |
| **5v5 even** | **41{,}338** | 1{,}209 | **0.525 (0.490, 0.558)** | 0.483 |

The 5v5-even state is the least predictable (AUC 0.525, CT win-rate 0.483, essentially a coin flip) and
is **71% of the contested set** (41{,}338 of 58{,}368 snapshots). The pooled contested-AUC of ~0.58 is
therefore dominated by the symmetric, pre-engagement phase of rounds. Smaller even states (1v1-3v3),
where positional and bomb information has accumulated, are considerably more predictable (0.69-0.73).

**Graded evenness** (equal alive, sweeping the equipment threshold): contested-AUC declines
monotonically as the economy evens out, 0.612 (|Δequip| ≤ $10k) → 0.558 (≤ $250).

**(B) Information saturation (contested set, xgb classical + deep).**

| Representation | contested-AUC (95% CI) |
|---|---|
| economy (A) | 0.580 (0.552, 0.608) |
| +spatial (E) | 0.585 (0.557, 0.613) |
| +bomb (EB2) | 0.586 (0.559, 0.613) |
| +firepower (EFB2) | 0.589 (0.562, 0.619) |
| TCN (deep) | 0.572 (0.549, 0.597) |
| Transformer (deep) | 0.568 (0.547, 0.591) |

Contested-AUC is flat near 0.585 across all representations, and the deep sequence models are slightly
worse. Adding information does not break the ceiling.

**(C) Matching-based Bayes-error (model-free ceiling).** State descriptor: 60 observable columns
(economy + Voronoi control + tactical/bomb), standardised; K = 25 neighbours drawn from **different
matches** (same-match neighbours excluded because they share the round label); up to 2{,}500 queries and
25{,}000 pool per category.

| Category | n snapshots | n matches | near-twin disagreement | oracle-AUC (ceiling, 95% CI) |
|---|---|---|---|---|
| 1v1 even | 3{,}515 | 182 | 0.431 | 0.608 (0.551, 0.666) |
| 2v2 even | 3{,}905 | 192 | 0.401 | 0.592 (0.532, 0.648) |
| 3v3 even | 3{,}633 | 195 | 0.364 | 0.675 (0.611, 0.743) |
| 4v4 even | 5{,}977 | 203 | 0.476 | 0.514 (0.451, 0.577) |
| **5v5 even** | **41{,}338** | 220 | **0.495** | **0.511 (0.480, 0.545)** |
| all even (pooled) | 58{,}368 | 220 | 0.476 | 0.548 (0.518, 0.578) |
| **CONTROL: 1-man-adv** | **129{,}717** | 220 | **0.314** | **0.828 (0.811, 0.846)** |

Interpretation of the two well-powered anchors:
- **5v5 even (n = 41{,}338):** near-identical states end differently 49.5% of the time (a fair coin is
  50%), and a predictor with perfect knowledge of the local outcome frequency reaches only AUC 0.511.
  The outcome is not determined by the observable state. The generalist model already achieves 0.525,
  i.e. it is at the aleatoric ceiling.
- **Man-advantage control (n = 129{,}717):** disagreement drops to 0.314 and the oracle reaches 0.828,
  confirming that the estimator detects strong signal where the state genuinely determines the outcome.
  This validates the method: the near-coin-flip result for 5v5-even is a property of the game, not an
  artifact of the estimator.

Intermediate even states (1v1-3v3) show recoverable structure (oracle 0.59-0.68); 4v4 is noisier and
closer to the 5v5 coin flip.

---

## Interpretation and paper placement

**The contested ceiling is real and largely aleatoric, and it is concentrated in the pre-engagement
5v5 phase.** Three independent lines of evidence agree: (i) contested-AUC is flat at ~0.585 across all
representations including deep models; (ii) 5v5-even, which is 71% of contested snapshots, sits at a
coin flip for both the model (0.525) and a model-free matching oracle (0.511); (iii) the matching
estimator recovers AUC 0.83 on a man-advantage control, so its near-0.5 verdict for 5v5-even reflects
the game, not the method. The signal that does exist in even rounds lives in the smaller, later states
(2v2, 3v3), where positional and bomb information has accumulated.

This reframes the contested-AUC contribution: even rounds are hard not because our features are weak
but because the dominant even state, the symmetric opening, is close to irreducibly random, and this
can be demonstrated model-free.

**Paper placement:** extends contribution C1. New Results subsection "The contested-round ceiling"
(Study 2), with Fig. F8 (saturation curve + alive-state model-vs-oracle). Discussion gains a paragraph
distinguishing aleatoric from epistemic uncertainty in even rounds. The duel-level sub-study remains
future work for the extended version.

**Caveats.** The matching oracle is a k-NN estimate and is noisy for the smaller categories (wide CIs
on 1v1-4v4); the claim rests on the two well-powered anchors (5v5-even and the control) and the
saturation curve. The contested-specialist-vs-generalist comparison and the next-engagement features
(plan Study 2A) are not yet run; the alive-state decomposition already localises the ceiling, which was
the key question.
