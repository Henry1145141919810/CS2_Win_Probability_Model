# CS2 Round Win Probability Model

Map-Control and Firepower-Augmented Round Win Probability Model for Counter-Strike 2.

A live, per-second round win probability model for CS2 that augments the traditional
economy baseline (Xenopoulos et al., ESTA 2022) with three novel feature pillars:
Voronoi-derived **map control**, player-skill **firepower**, and **tactical readiness**.
Nine model architectures are evaluated across five competing feature sets.

> Status: **scaffold** — repository structure only. Implementation in progress.

## Repository structure

```
.
├── src/                  # Source code
│   ├── data/             # Demo parsing & validation (batch_parse, validate_parquet)
│   ├── features/         # Four feature pillars (economy, mapcontrol, firepower, tactical)
│   ├── models/           # Training pipeline & model architectures
│   ├── eval/             # CV, DeLong test, bootstrap, time-window analysis
│   └── viz/              # Figures (Voronoi heatmaps, per-second prob charts)
├── demos/                # GOTV demo files (git-ignored)
│   ├── raw/              # .rar archives from HLTV
│   └── extracted/        # .dem files
├── data/
│   └── parquet/          # Parsed channels: ticks/ kills/ rounds/ bomb/ grenades/
├── features/             # Per-match engineered features per pillar (git-ignored)
│   ├── economy/  mapcontrol/  firepower/  tactical/  combined/
├── models/               # Saved trained models / checkpoints (git-ignored)
├── outputs/figures/      # Paper figures (git-ignored)
├── notebooks/            # EDA & exploration
├── configs/              # Experiment configs
└── paper/                # Springer LNCS manuscript
```

Large artifacts (demos, parquet, features, models, figures) are **not** tracked in git;
only code and small metadata CSVs are versioned. Directory layout is preserved via
`.gitkeep` placeholders.

## The four feature pillars

| Pillar | Question | Source |
|---|---|---|
| 1. Economy | What did each team buy? | `dem.ticks` |
| 2. Map control | Where is each team right now? | `dem.ticks` + awpy nav mesh (Voronoi) |
| 3. Firepower | How skilled are the alive players? | HLTV stats join |
| 4. Tactical readiness | How prepared for the engagement? | `dem.ticks`, `dem.grenades`, `dem.bomb` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # requires Python 3.11+
```

## Scope

- Map: `de_inferno`
- Train: pro GOTV demos Jan 2024 – Dec 2025 (~150 maps)
- Test: fresh demos Jan 2026 onwards (15–20 maps, held out)

## License

TBD.
