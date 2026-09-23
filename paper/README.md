# Paper

| File | What it is |
|---|---|
| [CS2_Win_Probability_preprint.pdf](CS2_Win_Probability_preprint.pdf) | The current preprint (working draft v13.1), compiled from `tex/` |
| [tex/main.tex](tex/main.tex), [tex/refs.bib](tex/refs.bib), [tex/figures/](tex/figures) | LaTeX source; everything Overleaf needs |
| [CHANGELOG.md](CHANGELOG.md) | Every version since v1, with what changed and which files to re-upload |
| [draft.md](draft.md) | Markdown mirror of the paper's content (earlier register) |
| `tex/main_fp_rewrite.tex`, `draft_fp_rewrite.md`, `TODO_firepower_rewrite.md` | The firepower-section rewrite, merged into v12; kept for the record |

**Build.** Any LaTeX distribution works (`pdflatex main.tex` + `bibtex` or `latexmk -pdf main.tex`), as
does Overleaf (upload the contents of `tex/`) or Tectonic (`tectonic main.tex`). The version stamped on the
title page lives in one place: `\newcommand{\draftversion}{...}` at the top of `main.tex`.

**Versions.** Each version is a git tag `paper-vN`. The paper lived in `docs/paper/` up to v13 and in
`paper/` from v13.1, so compare across the move with
`git diff paper-v12 paper-v13.1 -M --stat`.
