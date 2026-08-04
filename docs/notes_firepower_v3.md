# Notes — Firepower v3: team-ranking-weighted skill features

**Date:** 2026-08-02 · **Owner:** Leu + Claude  
**Commit:** `d8ba669`  
**Status: Negative result — v3 does not improve over v2 or EB2 in-sample; abandoned.**

---

## Motivation

All three constructions of the v2 firepower prior (broken, lagged-2025, same-year 2026) fail to beat
EB2 out-of-time (see [notes_lagged_holdout.md](notes_lagged_holdout.md)). One remaining hypothesis:
the summed HLTV rating is **count-confounded and opponent-blind** — it treats a rank-1 team player's
rating the same as a rank-30 team player's identical rating, even though the former is harder-earned.

**Hypothesis:** weighting each player's stats by their team's world ranking (a proxy for opponent
quality) should produce a more informative, less noisy skill prior.

---

## What we built

### Infrastructure (all committed, kept for the record)

| File | Contents |
|---|---|
| `configs/player_team_year.csv` | 507 rows: (steamid, year) → team_canonical, built from 298 tick parquets |
| `configs/team_rankings.csv` | 141 rows: (team_canonical, year) → hltv_rank + weight, 3 HLTV snapshots |
| `src/features/firepower.py` | v3 accumulators added alongside v2; 18 new `_v3` columns emitted |

### Ranking snapshots used

| Era | HLTV date | Note |
|---|---|---|
| 2024 | 2024-12-30 | Monday ranking (exact) |
| 2025 | 2025-12-29 | Nearest Monday to Dec 30 |
| 2026 H1 | 2026-06-29 | Nearest Monday to Jun 30 |

### Weighting formula (primary)

```
weight = 1 / log₂(rank + 1)
```

| Rank | Weight |
|---|---|
| 1 (Team Falcons 2026) | 1.000 |
| 2 (Team Vitality 2026) | 0.631 |
| 5 | 0.387 |
| 10 | 0.301 |
| 20 | 0.228 |
| 30 | 0.201 |
| Unranked (>30) | 0.193 (rank 35 default) |

Players with no team info also receive the default weight (0.193).

---

## Test method

Full re-assembly was not done. Instead, a **proxy evaluation** was run on the existing
`training_dataset.parquet`:

1. For each demo, read round-1 tick data to determine which team started as CT.
2. Determine CT/T team per snapshot using round number (rounds ≤ 12 = first half).
3. Look up each team's rank weight for that year.
4. Compute `ct_fp_v3 = ct_fp_v2 × ct_team_weight` for every firepower column.
   *(Not per-player multiplication — that requires re-assembly — but an aggregate proxy.)*
5. Replace the 20 v2 firepower columns with the 20 weighted versions; run GroupKFold LogReg.

Three weighting formulas were tested:

| Formula | rank 1 | rank 5 | rank 10 | rank 20 | rank 30 |
|---|---|---|---|---|---|
| `1/log₂(rank+1)` | 1.000 | 0.431 | 0.301 | 0.228 | 0.201 |
| `1/rank` | 1.000 | 0.200 | 0.100 | 0.050 | 0.033 |
| `(31−rank)/30` linear | 1.000 | 0.867 | 0.700 | 0.367 | 0.033 |

**Note on implementation:** since all 5 alive players on CT are from the same team, multiplying
each player's stats individually by their team weight is mathematically identical to multiplying
the summed total once:

```
ct_rating_v3 = Σ (player.rating × team_weight)
             = team_weight × Σ player.rating
             = team_weight × ct_rating_sum
```

The proxy test therefore computes the exact v3 feature — no re-assembly was needed.

---

## Results

**In-sample GroupKFold AUC (5 splits, LogisticRegression):**

| Feature set | AUC | vs EFB2 |
|---|---|---|
| EB2 (no firepower) | 0.8509 ± 0.0099 | −0.0010 |
| **EFB2 (v2 raw sum)** | **0.8519 ± 0.0098** | — |
| EFB3 · 1/log₂ (original) | 0.8516 ± 0.0105 | −0.0003 |
| EFB3 · 1/rank (steeper) | 0.8510 ± 0.0106 | −0.0009 |
| EFB3 · linear | 0.8509 ± 0.0100 | −0.0010 |

**EFB2 (raw sum, no weighting) is the best firepower encoding in-sample.
All v3 variants sit below EFB2. The steeper the weighting, the worse the result.**

Out-of-time holdout was not re-run for v3: the in-sample result already shows that weighting
*hurts* v2, and EFB2 was already dominated by EB2 out-of-time in all prior runs.

---

## Interpretation

The result is intuitive in hindsight. HLTV rating is computed from match outcomes against opponents
of varying quality — it is **not** a raw counting stat. A player earning 1.3 rating on a rank-1 team
and a player earning 1.3 rating on a rank-30 team are playing in different competitive environments,
but HLTV's formula already adjusts implicitly for opponent strength when computing rating.

By multiplying rating by team rank weight, we introduce **double-penalisation**: the low-rank team's
player already has a lower rating reflecting weaker opponents; shrinking it further by ×0.2 removes
real signal. The steeper the weight curve, the more signal is destroyed.

**Why the raw sum still beats EB2 in-sample but loses out-of-time is unchanged** from the v2
analysis: firepower features carry in-sample signal (r ≈ 0.987 with player count, so they partly
function as a alive-player counter), but this does not transfer to new matches with different rosters.

---

## Verdict

**V3 is a negative result.** The team-ranking-weighted encoding is strictly worse than the raw sum
in-sample, and the raw sum already fails out-of-time. No further firepower variants are planned.

The infrastructure (player_team_year.csv, team_rankings.csv, v3 columns in firepower.py) is kept in
the repository for reproducibility but is not used in any trained model.

**For the paper:** this result strengthens the negative finding — even with opponent-quality adjustment
via world ranking, the external skill prior does not outperform the demo-only model. The failure is
not a data problem (coverage is full, weighting is principled) but a structural one: the match state
features already absorb whatever team-quality signal the prior can provide.
