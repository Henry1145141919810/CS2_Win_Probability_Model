# Collaborator Setup & Catch-Up Guide

Everything you need to get to parity with the current state of the **CS2 Round Win
Probability** project — environment, the shared data bundle, the pipeline, what's been
built, the results, and where we're going. Read top to bottom once; it's ~15 min.

---

## 0. What this project is (1 paragraph)

A live, per-second **round win-probability model for CS2 (de_inferno)**. We parse pro GOTV
demos into per-second snapshots, engineer features in four "pillars" (economy, map control,
firepower, tactical), and train models (logistic, XGBoost, …) to predict `P(CT wins the
round)` at every second. The novel angle is **spatial map control** (territorial dominance),
benchmarked against the economy-only baseline from the literature (Xenopoulos/ESTA). Target:
arXiv + MLSA 2027. Full rationale in [`docs/methodology.md`](docs/methodology.md).

---

## 1. Prerequisites
- **Python 3.11+** (we use 3.12).
- **git**.
- ~2 GB disk for the data bundle; 8 GB+ free RAM recommended for training.
- You do **NOT** need the raw demos (300 GB) or to parse anything — the bundle has the
  parsed data.

## 2. Setup (one time)

```bash
# 1. clone the code
git clone https://github.com/Henry1145141919810/CS2_Win_Probability_Model.git
cd CS2_Win_Probability_Model

# 2. virtual env + deps
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on mac/linux
pip install -r requirements.txt   # torch/torch-geometric are optional (deep models, later)

# 3. awpy map data (nav mesh for control features + radar image for plots)
awpy get navs
awpy get maps
#   (skip `awpy get tris` (101 MB) unless you want to REBUILD the LOS matrix —
#    the prebuilt matrix is already in the bundle.)
```

## 3. The shared data bundle (Henry sends a Drive link → `cs2_share_bundle.zip`, ~180 MB)

**Unzip it into the repo root** — it keeps the right folder paths:

```bash
unzip cs2_share_bundle.zip          # creates data/parquet/..., data/training_dataset.parquet, etc.
```

What's inside:
| Path | What |
|---|---|
| `data/parquet/{ticks,kills,rounds,bomb,smokes,infernos}/` | parsed demos, 1 file per demo (per-second ticks; events). **No grenades** (446 MB, unused). |
| `data/training_dataset.parquet` | the model-ready table: **476,595 snapshots × 69 features**, 220 demos. |
| `configs/demo_list_final.csv` | the exact 220 demos in the model (event, teams, score, winner). |
| `.cache/visibility_de_inferno.npy` | precomputed line-of-sight matrix (saves a ~37 min build). |

That's everything — **bundle + repo + `awpy get navs maps` = full parity.** Nothing else to
download or share.

---

## 4. Repo structure

```
src/data/        # data pipeline (one stage per file)
  extract_demos.py        rar -> dem (needs 7-Zip; you won't need this)
  batch_parse.py          dem -> 5 parquet channels (awpy)   [you have the output]
  validate_parquet.py     round-label QA (halftime swap, phantom-round trimming)
  assemble.py             parquet -> training_dataset.parquet (computes all features)
  liquipedia_inferno.py   builds the match list; reconcile/craft_final_list = list bookkeeping
  awpy_patch.py           fixes an awpy crash on some demos (int-encoded winner)
src/features/    # the feature pillars
  economy.py              Pillar 1 (equipment, hp, alive, bomb, score, time)
  mapcontrol.py           Pillar 2 (Voronoi control + the LOS/FOV/smoke "grey" control)
  positional.py           Pillar 4a (callout entropy, players-per-zone, AWP, utility)
  bomb.py                 Pillar 4b (bomb site/coords, nearest-CT nav-path distance)
  build_zone_map.py       maps nav areas -> named zones (banana/mid/sites)
  visibility.py           builds/loads the line-of-sight matrix
src/models/
  train_pipeline.py       5-fold GroupKFold CV, DeLong, block bootstrap, time-window
src/viz/
  mapcontrol_viz.py       control-surface plots (Voronoi vs grey) + animation
  winprob_chart.py        per-second win-probability chart (BLAST.tv style)
  plant_spots.py          bomb-plant cluster map
configs/         # match lists + validation reports (small, tracked in git)
docs/methodology.md, docs/sampling.md   # the "why" docs
```

---

## 5. The pipeline (how data becomes results)

```
demos (.rar) ──extract──▶ .dem ──batch_parse──▶ data/parquet/{ticks,kills,rounds,bomb,smokes,infernos}
                                                          │
                                            validate_parquet (QA, drops bad demos)
                                                          │
                                                     assemble.py
                                                          ▼
                                          data/training_dataset.parquet  (features + label)
                                                          │
                                                   train_pipeline.py
                                                          ▼
                                        AUC / DeLong / bootstrap CIs / time-window
```

**You start at `assemble` / `train`** — the bundle already has the parsed parquet AND the
assembled `training_dataset.parquet`.

Reproduce the headline numbers:
```bash
python src/models/train_pipeline.py --models logreg,xgb --sets A,B,G,BG,E,EG --bootstrap 500
```
Rebuild the dataset from the parquet (e.g. after changing a feature):
```bash
python src/features/assemble.py            # ~30-60 min; writes training_dataset.parquet
```

---

## 6. Key data facts (so features make sense)

- **Tickrate = 64.** (awpy's default of 128 is wrong; we force 64.) We sample **1 snapshot
  per second** from `freeze_end` to round end.
- **`side` column is lowercase `'ct'`/`'t'`**; `winner` is `'ct'`/`'t'` too.
- **Teams swap sides at halftime** → first-to-13 is per *team*, not per side. `validate_parquet`
  reconstructs the real team score (via `team_clan_name`) and trims phantom rounds.
- **Label** = the per-round *side* winner (`ct_won` = 1 if CT won that round). Every snapshot
  of a round shares that round's label; the probability fluctuates because the *features*
  change second to second.
- **Eval:** 5-fold **GroupKFold by match** (never split a match across folds — that leaks).
  Metrics: AUC (ranking), log-loss + Brier (calibration), DeLong (significance), match-level
  **block bootstrap** (CIs).
- Dataset: **220 demos** (Tier-1, 2024–25), 476,595 snapshots, label base-rate 0.445.

---

## 7. The feature pillars + the map-control story

- **Pillar 1 — Economy** (`economy.py`): equipment value, HP/armor/alive totals, defuse kits,
  bomb-planted, time, score diff. This is **Model A**, the literature baseline.
- **Pillar 2 — Map control** (`mapcontrol.py`): two models —
  - **Voronoi** (`control_features`): each nav area → nearest living player's team; area-weighted
    control %, per-zone, trend, volatility. *This is the one we keep.*
  - **Grey/LOS** (`contest_control`): a 4-state model (CT/T/contested/grey) where a player only
    controls an area if in range **AND** has line-of-sight **AND** is facing it (FOV) **AND** it
    isn't behind an active smoke. Physically realistic (~80 % grey), but **see §8 — it did NOT
    improve prediction.**
- **Pillar 4 — Tactical** (`positional.py`, `bomb.py`): per-side positional entropy, players-per-
  zone, AWP alive/zone, utility counts (smokes/flashes/fire), bomb site/coords, nearest-CT
  nav-path distance to bomb.
- **Pillar 3 — Firepower** (HLTV player ratings): **not built yet** — the main remaining pillar.

Feature sets in `train_pipeline.py`: `A`=econ, `B`=+Voronoi, `G`=+grey, `BG`=both controls,
`D`=econ+tactical, `E`=econ+Voronoi+tactical, `EG`=E+grey.

---

## 8. Current results (XGBoost, 5-fold GroupKFold, B=500 bootstrap)

| Set | AUC | Δ vs economy | sig (CI excludes 0) |
|---|---|---|---|
| A — economy (baseline) | 0.8318 | — | — |
| B — + Voronoi | 0.8366 | +0.0048 | ✅ |
| G — + grey (LOS/FOV/smoke) | 0.8339 | +0.0021 | ✅ |
| **E — econ+Voronoi+tactical** | **0.8407** | **+0.0090** | ✅ |
| EG — E + grey | 0.8402 | +0.0085 | ✅ |

**Key finding (negative result, worth publishing):** the realistic **grey/LOS control does
not beat Voronoi.** Instantaneous "what each player can see" flickers tick-to-tick, whereas
**positional territory (Voronoi) is a more stable predictor** of round outcome. So we keep
Voronoi; the LOS/smoke/facing infrastructure stays for other analyses (time-smoothed control,
retake studies). **Best model = E (0.8407).** Effects are significant but modest — Pillar 3
(firepower) is the next likely lever.

Logistic regression is currently slightly ahead of XGBoost on this feature set (~0.849) — a
tuning item, possibly a genuine finding.

---

## 9. Visualizations

```bash
python src/viz/winprob_chart.py --match faze-vs-g2-m1-inferno     # per-second win prob, one round
python src/viz/mapcontrol_viz.py --match faze-vs-g2-m1-inferno --round 5   # Voronoi vs grey surface
python src/viz/plant_spots.py                                     # bomb-plant clusters
```
Figures land in `outputs/figures/` (git-ignored). The control viz shows player positions,
**facing lines (yaw)**, **active smokes**, and the 4-state grey surface side-by-side with Voronoi.

---

## 10. Gotchas / memory notes
- awpy parsing **one** demo peaks ~6.7 GB RAM; `batch_parse` waits if free RAM < 7 GB. (You
  won't parse — but FYI.)
- The LOS-matrix *build* needs ~4.4 GB (one-time); the prebuilt matrix is in the bundle, so
  you don't rebuild unless you run `awpy get tris` + `python src/features/visibility.py`.
- `process.md` and `preprocess.md` (the running dev log / planning) are **local/git-ignored** —
  this file + `docs/methodology.md` are the shared sources of truth.

## 11. What's done vs next
- **Done:** data pipeline, 220-demo dataset, Pillars 1/2/4, Voronoi-vs-grey study, eval harness
  (DeLong + bootstrap + time-window), figures.
- **Next:** Pillar 3 firepower (HLTV scrape + name mapping); time-smoothed grey experiment;
  2026 hold-out test; deep models (TCN/Transformer/GAT); paper figures + write-up.

Questions → ping Henry, or read `docs/methodology.md`.
