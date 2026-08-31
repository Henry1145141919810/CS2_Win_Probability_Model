# Firepower rewrite — checklist

**Decision:** restructure the firepower narrative from a chronological v1→v2→v3 account into
parallel encodings evaluated on one baseline, matching the shape of the map-control section
(§4.2 / §7.3). **EFB2 keeps the v2 (summed + gated) encoding** — no main table, appendix
table, heatmap or deep-model result is rerun. The encoding ablation is scoped to the five
classical models, which is where an encoding question belongs.

**Why the narrative changes.** The version history is not the logic. The count confound was
found in v1 but only repaired in v4; v2 (situational gates) and v3 (team-rank weighting) were
both built on the *unrepaired* summed encoding in between. Presented chronologically the
reader cannot tell what each step establishes. Presented as "repair the defect, then explore
two directions from the repaired base", each rung is attributable.

**Evidence.** `outputs/firepower_ladder.csv` — 25 cells (5 encodings × 5 classical models),
CV and out-of-time, AUC and contested-AUC. Training `training_dataset_clean220.parquet`
(220 matches), test `test_dataset_2026_defuse.parquet` (27 matches).

Key numbers, mean Δ vs EB2 over the five models:

| rung | ΔCV cAUC | ΔOOT cAUC | OOT wins |
|---|---|---|---|
| V1 summed rating | +0.0030 | −0.0153 | 0/5 |
| V4 mean rating | −0.0008 | −0.0056 | 1/5 |
| V4.2 mean × team rank | **+0.0105** | −0.0006 | 3/5 |
| V4.3 mean + gates | +0.0005 | −0.0487 | 0/5 |

Cross-validation — the only legitimate selection criterion — picks **V4.2**. Out-of-time it
still does not beat the skill-free model. That is the robustness result that closes the
"you picked a weak encoding" objection without redefining EFB2.

---

## A. `tex/main.tex` §4.4 Pillar 4 description (lines 419–452)

- [x] **A1** (424–425) "We built three versions and report all of them, because the
      progression is part of the finding." → reframe as four parallel encodings, one defect
      addressed per rung.
- [x] **A2** (427–430) `\paragraph{Firepower v1 (9 features).}` → **(a) summed rating**, the
      original encoding and the source of the count confound.
- [x] **A3** NEW paragraph → **(b) mean rating**: divide by the number of alive players that
      have an HLTV entry (`n_with_stats`, not the headcount — players missing from the table
      never entered the sum, so dividing by the headcount penalises their teammates).
- [x] **A4** (441–452) v3 paragraph → **(c) mean × team-rank weight**, $w=1/\log_2(r+1)$.
      Keep the rationale and the alternatives ($1/r$, linear); it now sits on the repaired base.
- [x] **A5** (432–439) v2 paragraph → **(d) mean + situational gates**. Keep the gate
      descriptions verbatim; they are unchanged, only the aggregation under them changed.
- [x] **A6** (450–452) forward reference "no version (v1, v2, or v3)" → "no encoding".
- [x] **A7** NEW one sentence: EFB2 throughout the paper uses the summed+gated encoding;
      §7.5 reports the ablation over all four and the robustness check.

## B. `tex/main.tex` §7.5 Results (lines 878–947)

- [x] **B1** (878) `\subsection{Firepower: anatomy of a confound}` → title covering both the
      confound and the repair (the confound is now diagnosed *and* fixed).
- [x] **B2** (883–908) v1 paragraph + `fig:shapfirepower` → **keep essentially as is.** The
      permutation-importance artifact and the integer-clustering figure are the best part of
      the section and are unaffected. Relabel "v1" → "the summed encoding".
- [x] **B3** (910–916) `\paragraph{v2, which half-works.}` → rewrite. The claim "because v2
      kept the sums, the count confound survives" stays true and now leads into the repair.
- [x] **B4** (918–928) `\paragraph{v3: opponent-quality weighting does not help either.}` →
      rewrite as rung (c) on the repaired base. Note it is the CV-selected best.
- [x] **B5** NEW → the ladder table (5 encodings × CV/OOT), and the reading: **cross-validation
      rewards every rung, out-of-time rejects every rung.**
- [x] **B6** (930–937) Summary → keep the conclusion (content, not encoding, is the binding
      constraint) but reach it from the ablation rather than by assertion.
- [x] **B7** NEW → robustness sentence: the CV-best encoding (V4.2) was re-tested
      out-of-time and still does not beat the skill-free model (mean Δ contested-AUC
      −0.0006 across five models).
- [x] **B8** (939–947) `fig:fpenc` caption → rewrite for the new figure.

## C. `tex/main.tex` Limitations (lines 1405–1409)

- [x] **C1** "We report rather than repair it" → **must change.** It was repaired: mean
      normalisation removes the confound (OOT contested-AUC −0.0153 → −0.0056) and the pillar
      still adds nothing. The stronger statement replaces the weaker one.

## D. `tex/main.tex` cross-references

- [x] **D1** (285–286) feature-set glossary → state which firepower encoding EFB2 carries.
- [x] **D2** (1292) "$r = 0.987$ with the player count" in §7.8 → verified at 0.9866 on the
      clean training table; keep, check the surrounding wording still fits.
- [x] **D3** (100–105) abstract → check "correctly constructed" still reads right now that
      four constructions are reported rather than three.

## E. Figures

- [x] **E1** `F9_firepower_encodings.png` (`fig:fpenc`) → **regenerate.** Currently plots v2
      raw sum + three v3 weightings from `outputs/firepower_v3_full.csv`. Must plot the four
      rungs from `outputs/firepower_ladder.csv`, and should show CV and out-of-time side by
      side — the reversal between them is the point.
- [x] **E2** `src/viz/firepower_v3_figure.py` → rewrite (or add a new script) for the above.
- [x] **E3** `shap_dependence_ct_firepower_rating.png` → **no change.** Still correct.
- [x] **E4** `F4_datagap.png`, `F6_firepower_three_variants.png` → **no change.** Coverage and
      the three data constructions are independent of the encoding question.

## F. `docs/paper/draft.md` (keep the markdown draft in sync)

- [x] **F1** (186–197) §4.4 → mirror A.
- [x] **F2** (325–337) §7.5 → mirror B.
- [x] **F3** (~464) Limitations item 4 → mirror C.

## Not touched

Main model matrix, appendix metric batteries, `F3_heatmap.png`, the out-of-time section's
tables and figures, every deep-model number, and the `EFB2` column list itself. No GPU work
and no retraining of any kind.
