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

## vNext — template (copy this block, don't edit v1)

```
## vN — YYYY-MM-DD — <one-line summary>

Tag: `paper-vN` · Zip: `CS2_winprob_overleaf_vN.zip`

**Changed.** <what actually changed and why>

**Re-upload to Overleaf.** main.tex + <exact figure filenames, or "none">

**Figures regenerated.** <which, and by which script>

**Still open.** <carried-forward gaps>
```
