# Documentation

Everything that is not code or the paper itself. The paper is the authoritative account; these documents
record how each result was reached, with the numbers, the dead ends, and the checks.

## methodology/ — how everything is evaluated

| File | What it covers |
|---|---|
| [methodology.md](methodology/methodology.md) | The full protocol and the chronological record of every experiment with its numbers |
| [FULL_BENCHMARK.md](methodology/FULL_BENCHMARK.md) | The seven-group evaluation checklist every model change must pass |
| [metrics.md](methodology/metrics.md) | Definitions and reading guide for every metric used |
| [map_control_models.md](methodology/map_control_models.md) | The three map-control representations, explained |
| [feature_table_draft.md](methodology/feature_table_draft.md) | Every feature, by pillar, and the feature-set composition |
| [sampling.md](methodology/sampling.md) | How the matches were chosen |

## pillars/ — the two feature pillars with the longest history

| File | What it covers |
|---|---|
| [firepower_pillar.md](pillars/firepower_pillar.md) | Player-skill prior: data acquisition, v1 and v2 design, results |
| [notes_lagged_holdout.md](pillars/notes_lagged_holdout.md) | The skill prior under three data regimes out-of-time |
| [notes_firepower_v3.md](pillars/notes_firepower_v3.md) | Team-ranking weighting (negative) |
| [firepower_v4_exploration.md](pillars/firepower_v4_exploration.md) | Mean encoding and the count confound (in Chinese) |
| [defuse_progress_README.md](pillars/defuse_progress_README.md) | The live defuse-progress feature: what and why |
| [notes_defuse_progress.md](pillars/notes_defuse_progress.md), [notes_defuse_results.md](pillars/notes_defuse_results.md) | Its derivation, validation, and benchmark |

## studies/ — analyses beyond the main model matrix

| File | What it covers |
|---|---|
| [results_checkpoint.md](studies/results_checkpoint.md) | One-page inventory of every result and the overall verdict |
| [plan_beyond_economy.md](studies/plan_beyond_economy.md), [study_beyond_economy_results.md](studies/study_beyond_economy_results.md) | Residual analysis and the contested-round ceiling |
| [notes_pathwise_calibration.md](studies/notes_pathwise_calibration.md) | Trajectory (extreme-path) calibration |
| [plan_t0_match_noecon.md](studies/plan_t0_match_noecon.md) | Umbrella plan for the three September 2026 extensions |
| [study_noecon_results.md](studies/study_noecon_results.md) | The no-economy ablation (paper v13) |
| [plan_study1_t0_prior.md](studies/plan_study1_t0_prior.md), [memo_t0_prior.md](studies/memo_t0_prior.md) | The round-start prior (paper v13): plan, pilot, one-page summary |
| [plan_round_to_map.md](studies/plan_round_to_map.md), [study_round_to_map_results.md](studies/study_round_to_map_results.md), [memo_round_to_map.md](studies/memo_round_to_map.md) | From round to map win probability: plan, results, one-page summary (not yet in the paper) |
| [plan_study2_match_wp.md](studies/plan_study2_match_wp.md) | Superseded first plan for the map model, kept for the record |

## cluster/ — running the GPU models

| File | What it covers |
|---|---|
| [PARCC_betty_guide.md](cluster/PARCC_betty_guide.md) | Logging in, storage, Slurm rules, data transfer on the Betty cluster |
| [cluster_runbook.md](cluster/cluster_runbook.md) | First TCN run, step by step |
| [BETTY_benchmark_guide.md](cluster/BETTY_benchmark_guide.md), [BETTY_defuse_guide.md](cluster/BETTY_defuse_guide.md) | Out-of-time deep-model runs |
| [defuse_runbook.md](cluster/defuse_runbook.md) | Re-parsing demos for the defuse-progress feature |

## project/ — project history

| File | What it covers |
|---|---|
| [roadmap.md](project/roadmap.md) | Milestones and venue plan |
| [submission_requirements.md](project/submission_requirements.md) | Venue rules and deadlines |
| [collaborator_onboarding.md](project/collaborator_onboarding.md) | The original collaborator guide (June 2026; parts are outdated) |
| [two_day_summary.md](project/two_day_summary.md), [midway_summary.md](project/midway_summary.md) | Status summaries written for collaborators |
| [TODO_leu_2026_scrape.md](project/TODO_leu_2026_scrape.md) | Task spec for the 2026 HLTV scrape |

**Known data defect, recorded here for anyone reusing the tables:** the `ct_score` / `t_score` columns in
the modelling tables are cumulative *side* wins, never re-keyed at the halftime swap. They are correct in
the first half only. The round-level studies rebuild team scores from round winners; see
[plan_study1_t0_prior.md](studies/plan_study1_t0_prior.md), problem P1.
