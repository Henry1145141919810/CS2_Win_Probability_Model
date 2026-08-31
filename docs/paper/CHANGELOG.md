# Paper draft — version log

How versioning works, so an Overleaf update is never guesswork.

- The version lives in **one place**: `\newcommand{\draftversion}{vN}` at the top of
  [tex/main.tex](tex/main.tex). It is stamped on the title page, so a printed or emailed PDF
  is never ambiguous about which draft it is.
- Every substantive revision bumps `vN`, adds a row below, and ships a fresh
  `CS2_winprob_overleaf_vN.zip`.
- Each row lists **exactly which files changed**, so you can re-upload only those to Overleaf
  instead of re-uploading the whole project.
- Each version is also a **git tag** (`paper-vN`), so `git diff paper-v1 paper-v2 -- docs/paper/`
  shows precisely what moved.

## Updating Overleaf

**Small change (text only):** re-upload `main.tex`. Nothing else.

**Figures changed:** re-upload `main.tex` plus only the figures named in the row below.

**Big change / unsure:** delete the Overleaf project contents and upload
`CS2_winprob_overleaf_vN.zip` fresh. Cheapest and safest.

---

## v1 — 2026-07-13 — first full draft

Tag: `paper-v1` · Zip: `CS2_winprob_overleaf_v1.zip` · Class: `article` (LNCS conversion
deferred to submission)

**Content.** First complete draft. Four contributions: contested-AUC; the three-way map-control
ablation (realistic ≠ predictive unless temporally stabilised); the defuse-race feature; the
nine-architecture dead heat plus the 2026 out-of-time holdout, where the demo-derived pillars
transfer (0.8493 → 0.8501) and the firepower pillar collapses (0.8519 → 0.8236) on an
inference-time data dependency. Plus a Chronology section recording the ten steps in the order
they actually happened.

**Files.** `main.tex`, `refs.bib`, and 14 figures in `figures/`.

**Known gaps carried into v2.**

- `refs.bib` is a **scaffold**. Every entry is marked `% VERIFY`; several venue/year fields are
  recollection, not fact. Nothing in it is confirmed.
- Leu's affiliation is a `\todo{}`.
- The PARCC acknowledgement is a `\todo{}` and is *required* by their terms of use.
- Appendix B (full metric battery) is a `\todo{}` — the numbers exist in `outputs/`, it is a
  formatting job.
- Deep models not evaluated out-of-time.
- **Blocked on Leu:** the 2026 HLTV scrape. When it lands, the holdout gets re-run for both the
  same-era and the lagged-prior variants — and that re-run must be reported as a *separate,
  disclosed* evaluation, because the 2026 holdout was touch-once.

---

## v2 — 2026-07-25 — lagged-prior firepower result

Tag: `paper-v2` · Zip: `CS2_winprob_overleaf_v2.zip` · Class: `article`

**Changed.** Adds the leak-free lagged-prior firepower holdout (the experiment that was "in
progress" in v1). Leu's scrape completed 2025 stats to 82/82 coverage; we rebuilt the 2026 holdout
so each match uses the previous season's skill, and re-ran the classical holdout as a separate,
disclosed evaluation. New Sect. 7.9 ("The lagged-prior variant"), Table 7, and Fig. F5. Abstract,
Limitations, and Conclusion updated. Key finding: fixing the coverage gap restores calibration
completely (intercept −0.36 → +0.07) and most of the AUC, **but firepower still does not beat the
demo-only EB2 out-of-time on any model** — turning a plumbing bug into a rigorous negative result
about external skill priors. Recommendation (ship EB2, no firepower) stands, now for a sharper
reason.

**Re-upload to Overleaf.** main.tex + one new figure: `F5_firepower_recovery.png`

**Figures regenerated.** F5 is new (`src/viz/firepower_recovery.py`). No existing figure changed.

**Still open.** refs.bib still an unverified scaffold; Leu's affiliation; PARCC acknowledgement;
Appendix B metric tables; deep-model out-of-time (TCN/Transformer queued on Betty — see
`docs/BETTY_benchmark_guide.md`); GAT out-of-time (needs a 2026 trajectory dataset). Task A (2026
same-year stats) not done by Leu, so only the lagged variant is reported — which is the one that
should headline anyway.

---

## v3 — 2026-07-25 — the same-year variant completes the firepower triptych

Tag: `paper-v3` · Zip: `CS2_winprob_overleaf_v3.zip` · Class: `article`

**Changed.** Leu delivered the 2026 same-year stats (82/82). Ran the third and final holdout variant
(same-year 2026: full coverage + same-era knowledge, but leaky — the best case a skill prior could
have). Sect. 7.9 rewritten from "two variants" to the full three-construction spectrum; Table 7 now
shows all three vs EB2; **Fig. F5 replaced by F6** (three-variant comparison). Abstract, Conclusion
updated. **Result: firepower beats the demo-only EB2 under NO construction** — broken, leak-free
lagged, or leaky same-era oracle — trailing by 0.002–0.011 AUC on every model, with worse
contested-AUC everywhere. Same-year ≈ lagged confirms the count confound (which season's ratings you
sum barely matters). The negative verdict is now triangulated and pre-empts every reviewer objection.

**Re-upload to Overleaf.** main.tex + one new figure: `F6_firepower_three_variants.png`. (F5 removed
from the project — delete it from Overleaf too if you uploaded v2.)

**Figures regenerated.** F6 new (`src/viz/firepower_three_variants.py`). F5 retired.

**Still open.** Same as v2: refs.bib unverified; Leu's affiliation; PARCC acknowledgement; Appendix B
metric tables; deep-model out-of-time on Betty (TCN/Transformer queued — now runnable against either
the lagged or same-year holdout); GAT out-of-time (needs 2026 trajectory dataset). **Firepower work is
now complete — no further scrapes or variants.**

---

## v4 — 2026-07-29 — deep models evaluated out-of-time (closes a limitation)

Tag: `paper-v4` · Zip: `CS2_winprob_overleaf_v4.zip` · Class: `article`

**Changed.** Ran the TCN + Transformer out-of-time on both the lagged and same-year 2026 holdouts
(Betty). Result: they **tie the classical models out-of-time too** — TCN 0.8443, Transformer 0.8448
(same-year), landing exactly on classical EFB2 (0.8450), all CIs overlapping. The in-time dead heat is
not an in-sample artifact. They degrade *less* than classical EFB2 (−0.003/−0.005 vs −0.007/−0.009),
and — because they consume all features incl. firepower — sit below the no-firepower EB2. Sect. 7.6
gains a "dead heat persists out-of-time" paragraph; the Limitations "deep-model holdout" item is
retired and replaced with the finding; abstract gains a clause. No new figure (numbers are in-text +
`outputs/holdout_deep_summary.csv`).

Also fixed a real bug along the way: the deep `--holdout` path crashed on Betty because the cluster's
`training_dataset.parquet` was a stale firepower-v1 build while the holdouts are v2 — added a
schema-mismatch guard to the deep scripts (commit 717b953). Root fix was re-syncing the data file.

**Re-upload to Overleaf.** main.tex only (no figure change).

**Figures regenerated.** None.

**Still open.** refs.bib unverified; Leu's affiliation; PARCC acknowledgement; Appendix B metric
tables; GAT out-of-time (still needs a 2026 trajectory dataset). Firepower + deep benchmarks now
COMPLETE.

---

## v5 — 2026-07-29 — professionalization pass (part 1) + figure de-editorialization

Tag: `paper-v5` · Zip: `CS2_winprob_overleaf_v5.zip` · Class: `article`

**Changed.** Begins the professionalization pass toward submission register (formal, result-first,
concise; no colloquialisms; no em-dash asides). Completed in this pass:
- **Abstract** rewritten to formal register.
- **Introduction** rewritten; removed the "A note on what this draft is" lab-notebook subsection.
- **Evaluation → new "Uncertainty and significance testing" subsection** that states the method
  precisely: match-level *paired* block bootstrap on the difference; a lift is significant when the
  paired ΔAUC interval excludes zero; overlapping marginal CIs do NOT imply non-significance. This is
  the rigorous basis for "pillars beat baseline (significant)" vs "architectures tie (not)".
- **All paper figure titles de-editorialized** (F1, F2, F4, F6): titles now state what the figure
  shows; the interpretive message lives in the caption, as in a published paper. Figures regenerated.

**Deferred to part 2 (after firepower v3 lands):** results and discussion sections still carry the
conversational register, and the "Chronology: how we actually got here" section should be cut or moved
to an appendix. These are deferred deliberately because the firepower results subsection will change
when v3 (team-ranking-weighted) is benchmarked, so finalizing its prose now would be premature.

**Re-upload to Overleaf.** main.tex + regenerated figures F1, F2, F4, F6.

**Still open.** refs.bib unverified; Leu's affiliation; PARCC acknowledgement; Appendix B tables;
professionalization part 2 (results/discussion + cut Chronology); GAT out-of-time.

---

## v6 — 2026-07-29 — two new studies: residual analysis + contested-round ceiling

Tag: `paper-v6` · Zip: `CS2_winprob_overleaf_v6.zip` · Class: `article`

**Changed.** Adds two Results subsections (both in submission register) that deepen the beyond-economy
question, with two new figures.

- **Sect. "Beyond economy: a residual analysis" (Study 1, Fig. F7, Table).** Partials economy out (a
  stacked-logit / offset model) and measures the orthogonal signal the spatial/bomb block adds. It is
  significant and transfers out-of-time on all four models (paired ΔAUC CI excludes zero in-time and
  out-of-time). An FWL orthogonalization ranks the defuse-race margins as the strongest and most
  economy-orthogonal features (economy R² ≈ 0.15), while nearest-CT-distance is largely economy
  re-encoded (R² ≈ 0.69). Contested-AUC gain is NOT significant, dovetailing with the ceiling result.
- **Sect. "The contested-round ceiling" (Study 2, Fig. F8).** Three converging analyses at snapshot
  level: (i) contested-AUC is flat ~0.585 across all representations incl. deep; (ii) it is dominated
  by 5v5-even snapshots (71% of contested), which sit at a coin flip (AUC 0.525); (iii) a model-free
  matching estimator shows near-identical 5v5-even states disagree 49.5% of the time (oracle AUC 0.511),
  while a man-advantage control returns oracle AUC 0.828, validating the method. Even-round outcomes are
  largely aleatoric, and this is shown model-free.

Full method + results + documented snapshot selection: `docs/study_beyond_economy_results.md`. Code:
`src/models/residual_analysis.py`, `src/models/contested_study.py`. Result CSVs in `outputs/`.

**Re-upload to Overleaf.** main.tex + two new figures: F7_residual.png, F8_contested_ceiling.png.

**Still open.** refs.bib unverified; Leu's affiliation; PARCC acknowledgement; Appendix B tables;
professionalization part 2 (results/discussion register + cut Chronology); GAT out-of-time; the
contested-specialist/next-engagement and duel-level sub-studies (plan Study 2A / duel).

---

## v7 — 2026-07-29 — arXiv candidate: firepower v3, verified citations, professionalization

Tag: `paper-v7` · Zip: `CS2_winprob_overleaf_v7.zip` · Class: `article`

**This is the first arXiv-candidate draft.** Combines several strands:

- **Firepower v3 (team-ranking-weighted), full benchmark, Fig. F9.** Leu added a v3 encoding that
  weights each player's stats by their team's HLTV world ranking. The full paired-CI benchmark
  (`src/models/firepower_v3_full.py`, all 5 models x 5 encodings on a common EB2 base) shows **no
  firepower encoding (v2 or any v3 weighting) beats the skill-free EB2 model in-sample** — every paired
  ΔAUC-vs-EB2 interval crosses zero — and v3 is indistinguishable from v2. Firepower section updated;
  the negative verdict is now comprehensive (no encoding helps in-sample, all degrade out-of-time).
- **All citations verified** against arXiv/DBLP/publisher; fabricated `choi2023esports` removed and the
  claim re-cited to Guo et al.; LightGBM/CatBoost added; `\cite` wired through related work, models,
  evaluation.
- **Professionalization:** cut the "Chronology" lab-notebook section; removed all 156 em-dashes to
  submission register (LaTeX environments verified balanced).
- Studies 1 & 2 (residual analysis, contested ceiling) integrated in v6 remain (Figs F7, F8).

**Re-upload to Overleaf.** main.tex + refs.bib + one new figure F9_firepower_encodings.png (F7/F8
already present since v6). Or upload the fresh zip.

**Still open before submission.** Leu's affiliation + author order; PARCC acknowledgement; Appendix B
full metric tables (currently a \todo); a final read-through. Optional: GAT out-of-time; duel-level
sub-study.

---

## v8 — 2026-08-05 — Appendix B full metric battery (real numbers)

Tag: `paper-v8` · Zip: `CS2_winprob_overleaf_v8.zip`

**Changed.** Replaced the Appendix B \todo with two populated tables from freshly computed results:
Table B1 (in-time 5-fold OOF battery: AUC, log-loss, Brier, ECE, BSS, contested-AUC for all 5 classical
models x 4 feature sets, plus TCN/Transformer/ensemble) and Table B2 (out-of-time 2026 same-year
battery for EB2 vs EFB2 with calibration slope/intercept). Numbers from
`src/models/appendix_metrics.py` (in-time), `outputs/extended_metrics.csv` (deep/ensemble), and
`outputs/holdout_2026_sameyr2026.csv` (out-of-time). No fabricated values.

**Re-upload to Overleaf.** main.tex only.

**Still open (4 \todo).** Leu's affiliation; PARCC acknowledgement; exact deep wall-clock/epochs;
Appendix C supplementary-figure placement (optional).

---

## v9 — 2026-08-05 — proofreading pass (clarity, firepower v3, acknowledgements)

Tag: `paper-v9` · Zip: `CS2_winprob_overleaf_v9.zip`

**Changed (from Henry's proofread).**
- Map-control figure regenerated to the 3-row Voronoi/grey/territory version (`mapcontrol_viz.py`);
  caption and surrounding text updated to describe the territory row.
- Added the rationale for the 15 s territory memory (map geometry: opponents cannot traverse cleared
  ground instantly).
- Renamed Pillar 1 "economy and combat state" and clarified it carries players-alive/health/time, not
  only money; added a footnote at the residual analysis explaining why nearest-CT-distance has high
  economy R² (headcount/phase, shown by Fig. 2's colouring).
- Added Firepower v3 (team-ranking-weighted) to Sect. 4.4, with the benchmark verdict (no encoding
  beats the skill-free model).
- Tightened the GAT description so it no longer implies the other models lack HP/equipment (they have
  team-aggregated totals; the GAT adds per-player + position).
- Explained `defuse_time_margin` and permutation importance in plain words before the "rank #8 of 68"
  claim; clarified that the seven features above it are the economy/combat baseline.
- Added a "Match selection (inclusion criteria)" subsection: the fixed Tier-1 event list, take-every-
  Inferno-map rule, 264→220 via validation, released list. A reproducible standard.
- De-colloquialised the "dead heat" section title and phrases; glossed TCN (and LightGBM/XGBoost/
  CatBoost already glossed).
- Filled Appendix B \todo earlier (v8); now filled acknowledgements (PARCC allocation), deep-model
  wall-clock/epochs, and placed two supplementary figures (logistic coefficients, SHAP beeswarm) in
  Appendix C.

**Re-upload to Overleaf.** main.tex + regenerated `mapcontrol_compare_faze-vs-g2-m1-inferno_r5.png`
(+ `logistic_coefficients.png`, `shap_beeswarm_ET_xgb.png` if not already uploaded).

**Still open (2 \todo, both need Henry).** Leu's affiliation/email; confirm exact PARCC/Betty
acknowledgement wording against their code-of-conduct page.

---

## v10 — 2026-08-05 — trajectory-level (extreme-path) calibration benchmark

Tag: `paper-v10` · Zip: `CS2_winprob_overleaf_v10.zip`

**Changed.** Adds the pathwise extreme-path calibration benchmark (now item 7 of the "FULL Benchmark"),
based on the advisor's group (Pipping & Wyner 2025/2026). Recasts the "honest probabilities"
subsection around the Doob-martingale null: each round's per-second WP is a martingale (p0 = round
start, terminal = outcome), and its extremes obey P(loser peak ≥ x | lose) = (p0/(1-p0))((1-x)/x).

Results (`src/models/pathwise_calibration.py`, Fig. F10): reproduced the 7.2% comeback number (now
7.4%); across all thresholds and both models the write-off frequency sits BELOW the continuous
benchmark (obs/bench 0.43–0.79); PIT/KS shows D_upper = 0 (no upper-tail inflation), i.e. the model
does not over-react (unlike ESPN's NBA feed, which the source paper flags as over-confident). The
below-benchmark position is the expected discrete conservatism (few kill-driven decision points per
round). Added: Fig. F10, a Discussion "two forms of honesty" paragraph (contested-AUC + path
extremes), a Limitations item on discrete/jumpy paths, and refs to Pipping–Wyner (1–2).

**Re-upload to Overleaf.** main.tex + refs.bib + one new figure F10_pathwise_calibration.png.

**Notes/data.** `docs/notes_pathwise_calibration.md`, `outputs/pathwise_benchmark.csv`,
`outputs/pathwise_perround_*.parquet`.

**Still open.** Leu's affiliation; exact discrete benchmark (future work); optional full PIT/KS with
discrete null calibration for the extended version.

---

## v11 — 2026-08-31 — defuse-progress feature (curve honesty) + complete feature table

Tag: `paper-v11` · Zip: `CS2_winprob_overleaf_v11.zip`

**Changed.** Adds (1) a new Results subsection "Per-second honesty during a defuse" (Sect. defuseprogress)
with Fig. F11 and Table tab:defusecal, and (2) a complete feature-list appendix (Appendix, longtable)
plus a per-set composition table.

The defuse-progress feature (4 columns measuring an ACTUAL running defuse, from a re-parse that adds the
per-tick is_defusing flag) is a curve-honesty fix, not an accuracy gain: it fires on ~1% of snapshots so
pooled AUC and contested-AUC do not move, but on the defusing rows it cuts log-loss 60% (0.158 -> 0.063,
logistic) and makes the predicted curve track the empirical one (EB2 stays flat as a defuse completes;
EB2D tracks it). 39.6% of attempts are interrupted, so it is not a ct_won relabel. Framed as the
contested-AUC argument in miniature. The four columns EXTEND the bomb-geometry pillar (not a pillar of
their own); EB2D/EFB2D are an ablation label (EB2/EFB2 + those columns), kept out of the headline sets
because a near-complete defuse is close to an endgame certainty.

**Deep-model corroboration (Betty, out-of-time on the 2026 holdout).** TCN and Transformer, which ingest
all columns incl. the 4 defuse ones, tie the classical models on pooled AUC (0.843 / 0.838, CIs overlap
~0.85 -> the dead heat persists) and track the defusing-row curve out-of-time. Table tab:defusecal ranks
defusing-row calibration: logistic EB2D best (log-loss 0.063), then TCN (0.083), Transformer (0.124), all
far below the featureless EB2 (0.158) -- feature, not model capacity.

**Data.** The TRAINING table is adopted as the canonical superset (`data/training_dataset.parquet`,
476,595 x 135) — byte-identical to the published table on all 115 prior columns (firepower included), so
every in-time number is unchanged (re-confirmed by re-running the in-time classical benchmark). The 2026
TEST table is deliberately NOT swapped: the re-parse populated 2026 firepower (coverage 91.6% -> 99.99%),
which would silently convert the paper's firepower-collapse finding (EFB2 out-of-time 0.8236) into the
same-year variant (0.8450). `data/test_dataset_2026.parquet` stays the original coverage-gap table; the
defuse benchmark uses a separate `test_dataset_2026_defuse.parquet` (firepower-free EB2/EB2D curve, so
unaffected). Old tables backed up as `data/*_predefuse_backup.parquet`.

**Re-upload to Overleaf.** main.tex + one new figure `F11_defuse_curve.png`.

---

## vNext — template (copy this block, don't edit v1)

```
## vN — YYYY-MM-DD — <one-line summary>

Tag: `paper-vN` · Zip: `CS2_winprob_overleaf_vN.zip`

**Changed.** <what actually changed and why>

**Re-upload to Overleaf.** main.tex + <exact figure filenames, or "none">

**Figures regenerated.** <which, and by which script>

**Still open.** <carried-forward gaps>
```
