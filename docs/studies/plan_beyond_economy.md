# Plan: Beyond-Economy Signal — two studies to add to the project and paper

Two additions that deepen the paper's central question (where does the predictive signal live,
beyond economy). Both are designed to be rigorous, implementable on the existing pipeline, and to
slot into the current paper as new contributions rather than side notes.

- **Study 1 — Residual analysis.** Fit economy alone, then measure the signal the spatial/tactical
  pillars add *orthogonally* to economy. Cleanly separates "new information" from "economy proxy."
- **Study 2 — The contested-round ceiling.** Model even rounds directly, attempt to raise the 0.58
  ceiling, and decompose the remaining gap into irreducible aleatoric randomness versus recoverable
  signal.

> **Load-bearing caveat (read before Study 2):** we already found that the contested slice is
> **underpowered at the round level** (a ±0.05 AUC CI on ~200 even rounds). A ceiling claim made at the
> round level is not defensible. Study 2 is therefore designed to run at the **snapshot and duel
> level**, where the effective sample is large enough, and to report ceiling estimates that pool across
> contested snapshots rather than relying on a single round-level AUC point.

---

## Study 1 — Residual analysis (beyond-economy signal, cleanly isolated)

### Motivation
The ablation shows the spatial/tactical pillars add roughly +0.003 to +0.005 AUC over economy, but a
nested AUC comparison does not separate *new* information from *re-encoded* economy. A residual
(partialling-out) analysis isolates the component of the spatial features that is orthogonal to
economy and measures only that. This answers a sharper question than the ablation: do the spatial
pillars carry information economy does not already contain, and how much?

### Method

**Primary: stacked-logit (offset) model.**
1. Compute economy out-of-fold predictions `p_A = P(ct_won | economy)` via the existing
   `oof_predict(df, FEATURE_SETS["A"], "logreg")` (GroupKFold by match, no leakage).
2. Form the economy logit `z = logit(clip(p_A, 1e-6, 1-1e-6))`.
3. Fit two GroupKFold models on the same folds:
   - **base:** target `ct_won`, single feature `z` (recovers economy).
   - **+spatial:** target `ct_won`, features `[z, X_spatial]` where `X_spatial` is the map-control +
     tactical + bomb columns (set E/EB2 minus economy).
   The improvement of **+spatial** over **base** is the beyond-economy contribution: the spatial
   features can only act through the residual left by economy.
4. Report ΔAUC, Δlog-loss, ΔBrier with **paired match-level bootstrap CIs** (reuse
   `block_bootstrap`). Run both **in-time** (OOF) and **out-of-time** (fit on all training, apply the
   economy logit + spatial model to the 2026 holdout).

**Diagnostic: feature orthogonalization (Frisch-Waugh-Lovell / double-ML residuals).**
1. For each spatial feature `s`, fit `E[s | economy]` (OOF) and take the residual
   `s_res = s - \hat{s}`.
2. Take the outcome residual `y_res = y - p_A`.
3. Rank spatial features by the partial predictive power of `s_res` on `y_res` (e.g. OOF univariate
   AUC of `s_res`, or the coefficient in a regression of `y_res` on all `s_res`).
   This produces a **beyond-economy importance ranking**: which spatial features are genuinely
   orthogonal to economy, and which are largely economy in disguise (mirrors the firepower count
   confound, `r = 0.987`).

### Deliverables
- **Table:** economy-only vs economy+spatial (offset model), ΔAUC / Δlog-loss / ΔBrier with paired
  CIs, in-time and out-of-time.
- **Figure:** beyond-economy importance ranking (bar chart of `s_res → y_res` partial AUC per
  feature), grouped by pillar. Expected: defuse-race and Voronoi control rank high; some tactical
  counts collapse toward zero once economy is partialled out.
- **Optional second panel:** Δlog-loss from the offset model, sliced by round phase and by
  contested-vs-not, to show *where* the orthogonal signal lives (expected: post-plant and contested).

### Paper placement
New Results subsection, "Beyond economy: a residual analysis," immediately after the model matrix and
before/with contested-AUC. It converts the pillar-contribution claim from a nested-AUC delta into a
partialling-out result, which is more rigorous and more convincing to a statistical reviewer. It also
gives a clean, quotable sentence: "controlling for economy, the spatial and bomb pillars add X AUC
(paired 95% CI ...), concentrated in post-plant and contested states."

### Caveats
- The stacked-logit uses OOF `p_A` as an input to a second OOF loop on the **same folds**; this is
  leak-free provided the fold structure is identical (each round's `p_A` was predicted by a model that
  did not see it). Verify identical fold assignment.
- Letting `z` float (a free coefficient) is a stacked model; fixing its coefficient to 1 is a true
  offset. Report the free-coefficient version as primary (standard), and confirm the coefficient is
  near 1 (a calibration sanity check).

### Task list
- [ ] `src/models/residual_analysis.py`: OOF economy logit, stacked/offset model, paired-bootstrap deltas
- [ ] FWL feature orthogonalization + beyond-economy importance ranking
- [ ] Out-of-time variant (2026 holdout)
- [ ] Figure(s) on the validated palette; table for the paper
- [ ] Write the Results subsection

---

## Study 2 — The contested-round ceiling

### Motivation
Contested-AUC sits near 0.58 for every model. Two questions follow: can a model specialised to even
rounds beat 0.58, and how much of the 0.42 gap to perfect prediction is irreducible aleatoric
randomness versus recoverable signal? Answering the second reframes a apparent weakness ("we are only
0.58 on even rounds") into a characterisation of the game ("even rounds are close to irreducibly
random, and here is how close"), which is a genuine contribution.

### Part A — Contested-specialist modeling
- Define contested precisely (equal players alive AND `|Δequip| ≤ $1500`), and also study a **graded**
  definition by sweeping the equipment threshold and the alive-equality constraint (how the AUC
  changes as rounds become more even). This produces an "evenness vs predictability" curve.
- Train models **only on contested snapshots** (GroupKFold) and compare against the **generalist
  evaluated on contested** (the current 0.58). Question: does removing easy rounds from training help
  or hurt on the hard subset?
- Add candidate **next-engagement features** (positional isolation, trade availability, utility for the
  next fight, man-advantage in the specific contested zone) and test whether any lifts contested-AUC
  above 0.58. These are the features most likely to carry even-round signal, per the prior-vs-posterior
  argument (the state encodes who is alive, not who wins the next duel).

### Part B — Ceiling decomposition (the contribution)
Two complementary estimators of the irreducible component.

**(i) Information-saturation curve.** Compute contested-AUC across increasingly rich representations:
economy → +spatial → +bomb → +firepower → deep sequence (TCN/Transformer) → raw player-graph (GAT).
Plot contested-AUC versus representation richness. If it asymptotes (for example near 0.60) regardless
of added information, that plateau is the practical ceiling. The GAT is important here: it sees raw
per-player positions/velocities/angles that the aggregates discard, so if even the GAT does not lift
contested-AUC, the ceiling is unlikely to be a feature-engineering artifact.

**(ii) Matching-based Bayes-error estimate.** For each contested snapshot, find its nearest neighbours
in a standardised, economy-controlled feature space and measure the **outcome disagreement rate** among
neighbours. Near-identical even states that nonetheless split ~50/50 in outcome imply a high Bayes
error and hence a low achievable AUC. Report the implied ceiling AUC as a function of neighbourhood
size, with sensitivity to the distance metric.

Combine: "the richest representation we could construct saturates contested-AUC at ≈X; a
matching-based Bayes-error estimate implies an irreducible ceiling of ≈Z. Of the gap from 0.50 to
perfect, ≈A is aleatoric and ≈B is potentially recoverable."

### The power problem, and the duel-level reframing (mandatory)
Round-level contested analysis is underpowered (±0.05 AUC CI on ~200 even rounds). Two responses,
both used:
1. **Report at the snapshot level with match-block bootstrap**, and state the effective sample and CI
   honestly. The saturation curve and Bayes-error estimate pool across ~58k contested snapshots, so
   they are far better powered than a single round-level AUC.
2. **Reframe toward duels where the claim needs power.** The kills channel
   (`data/.../kills/*.parquet`) lets us define an **engagement/duel dataset**: for each gunfight in a
   contested round, the context (who has position, utility, numbers, angle) and the label (who won the
   duel). There are many more duels than even rounds, so a duel-win-probability model has the
   statistical power a round-level ceiling claim lacks. This is the power-correct venue for the
   even-round signal, and it connects directly to Study 1's residual framing (duel outcome controlling
   for economy). Scope: a self-contained sub-study; larger than Part A/B but the rigorous home for the
   ceiling claim.

### Deliverables
- **Figure:** the information-saturation curve (contested-AUC vs representation richness) with the
  Bayes-error ceiling band overlaid.
- **Figure/table:** the graded evenness-vs-predictability curve.
- **Headline number:** the estimated irreducible fraction of the even-round gap, with sensitivity.
- **(Duel sub-study):** duel-win-probability AUC and the beyond-economy residual at duel level.

### Paper placement
Extends contribution C1 (contested-AUC) from "we propose the metric" to "we propose it and characterise
the ceiling it exposes." New Results subsection "The contested-round ceiling," plus a Discussion
paragraph on aleatoric versus epistemic uncertainty in even rounds. The duel-level sub-study can be
included if space allows, or held for the extended arXiv version and the follow-up paper.

### Caveats
- Bayes-error via matching is sensitive to the distance metric, standardisation, and the curse of
  dimensionality with ~100 features. Mitigation: run matching in a reduced space (economy-controlled,
  or a low-dimensional embedding), and report sensitivity to neighbourhood size.
- Contested-specialist training loses ~88% of the data; the comparison to the generalist must control
  for training-set size.
- Keep the touch-once discipline: any contested/duel analysis on the 2026 holdout is a separate,
  disclosed evaluation.

### Task list
- [ ] `src/models/contested_study.py`: graded-evenness curve; contested-specialist vs generalist
- [ ] Next-engagement candidate features; test lift over 0.58
- [ ] Information-saturation curve (needs deep OOF preds on the contested subset)
- [ ] Matching-based Bayes-error estimator + sensitivity
- [ ] (Sub-study) duel dataset from the kills channel; duel-win-probability model
- [ ] Figures on the validated palette; Results subsection + Discussion paragraph

---

## How these change the paper (summary)

| Study | New contribution | Section | Risk |
|---|---|---|---|
| 1. Residual analysis | Beyond-economy signal isolated by partialling out economy; more rigorous than the nested ablation | new Results subsection | low; reuses existing pipeline |
| 2A. Contested specialist | Whether an even-round specialist beats 0.58 | Results | low |
| 2B. Ceiling decomposition | Quantifies the irreducible fraction of the even-round gap; reframes the 0.58 finding | Results + Discussion | medium; matching estimator needs care |
| 2-duel. Duel-level | Power-correct home for the even-round signal | extended/arXiv or follow-up | higher; new data engineering |

Sequencing suggestion: Study 1 first (small, high-rigor, reuses everything), then Study 2A/2B, then
the duel sub-study if the ceiling result warrants it. None of this blocks the Oct 1 abstract; it
strengthens the Dec 4 full paper and the arXiv extended version.
