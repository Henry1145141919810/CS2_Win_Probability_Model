# Defuse progress — what it is and why it exists

**Status:** implemented and validated on the 27-match 2026 set, assembled through the normal
pipeline. Needs the *training* set re-parsed before it can be evaluated on the paper's data.
Branch `exp/defuse-time`.
**Deep dive:** [`notes_defuse_progress.md`](notes_defuse_progress.md) · **How to run:**
[`defuse_runbook.md`](defuse_runbook.md)

## The gap

Every bomb feature in the model is **counterfactual**. `defuse_time_margin`,
`defuse_margin_kit`, `n_ct_can_defuse`, `defuse_contest_margin` all ask the same question
from geometry: *could* a CT get to the bomb and finish in time? None of them can see a
defuse that is **actually happening**.

The cost is measurable. On the 27-match 2026 set, using the full `EB2` set — which carries
that entire bomb block — the model's predicted win probability while a defuse runs:

| defuse progress | actual CT win rate | EB2 predicts |
|---|---|---|
| 0–20% | 0.909 | 0.894 |
| 20–40% | 0.924 | 0.885 |
| 40–60% | 0.951 | 0.892 |
| 60–80% | 0.975 | 0.895 |
| 80–100% | 0.983 | **0.897** |

The curve is **flat**. A CT is 80% of the way through defusing and the model has not moved.

## The feature

Four columns, 0 whenever nobody is defusing:

| column | meaning |
|---|---|
| `defuse_elapsed_sec` | seconds into the current attempt (real-valued, 1/64 s precision) |
| `defuse_progress_frac` | elapsed / required, kit-aware (5 s with a kit, 10 s without), [0,1] |
| `defuse_in_progress` | is anyone on the bomb right now |
| `defuse_beats_fuse` | will the remaining defuse time fit inside the remaining fuse |

Feature sets: **`EB2D`** = EB2 + these four. **`EFB2D`** = EFB2 + these four.

With them, the same model tracks the empirical curve almost exactly:

| defuse progress | actual | EB2 | **EB2D** |
|---|---|---|---|
| 0–20% | 0.909 | 0.894 | 0.898 |
| 20–40% | 0.924 | 0.885 | 0.929 |
| 40–60% | 0.951 | 0.892 | 0.960 |
| 60–80% | 0.975 | 0.896 | 0.978 |
| 80–100% | 0.983 | 0.897 | **0.988** |

Log-loss over the defusing rows drops **65%** (0.1760 → 0.0613), Brier **72%**. Overall AUC
moves +0.0006, which is the expected non-result — see "What it will and will not move".

## Where the data comes from

The parsed bundle contains no defuse-start information at all, and it cannot be recovered
from what is there:

- `awpy.parsers.bomb.parse_bomb()` hardcodes five events — drop / pickup / plant / detonate
  / defuse — which is exactly the vocabulary in `data/parquet/bomb/`.
- `bomb_begindefuse` and `bomb_abortdefuse` come back **empty from demoparser2 even when
  requested explicitly**. That route is dead.
- Back-deriving a start from a completed defuse (`defuse_tick − 5 s`) would only ever see
  defuses that *succeeded* — i.e. it would encode the label.

What does work is the per-tick player prop **`is_defusing`** (`CCSPlayerPawn.m_bIsDefusing`),
a one-line addition to `PLAYER_PROPS`. **This is why the demos have to be re-parsed.**

`batch_parse` then derives a small `defuse` channel from the full 64 Hz tick stream *before*
downsampling — one row per attempt (`round_num, steamid, start_tick, end_tick, had_kit,
completed`). It plays the same role the bomb table plays for the plant tick, so
`defuse_elapsed_sec` is computed the way every other time feature in the project is:
`(snapshot_tick − reference_tick) / 64`. Deriving it after the 1 Hz downsample would leave
only integer seconds with a ±1 s error on the start.

## Is it just relabelling `ct_won`?

No. A defuse that completes *is* the CT win, so this was the first thing checked. Across the
31 re-parsed 2026 demos:

```
attempts     135
completed     87
INTERRUPTED   48   (35.6%)
```

Over a third of attempts fail, and four of them got past 75% progress before dying — the most
extreme a no-kit defuse interrupted at **9.98 s of the 10 s** it needed. The model does not
memorise "defusing ⇒ CT wins": on interrupted attempts it predicts **0.406**, against 0.501
for the baseline and a true 0.0.

**Quarantine anyway.** These columns live only in `EB2D` / `EFB2D`, never in `E` / `EB2` /
`EFB2`. Folding endgame certainty into the map-control sets would credit the spatial pillar
for something it did not do.

## What it will and will not move

It fires on ~1.2% of snapshots, so it **cannot** move a global metric — overall AUC moves
+0.0009 — and reading that as "no value" is the trap. It is a **per-second curve honesty**
fix, in the same family as the pathwise-calibration work, and it should be judged on the
defusing rows and on the shape of the curve above.

## Files

| file | what |
|---|---|
| `src/features/defuse.py` | the feature (polars only, no nav mesh, no awpy) |
| `src/data/batch_parse.py` | `is_defusing` prop + `_defuse_attempts()` + the `defuse` channel |
| `src/features/assemble.py` | loads the `defuse` channel into the snapshot loop |
| `src/models/train_pipeline.py` | `EB2D`, `EFB2D` |
| `src/data/defuse_report.py` | attempts / interrupted / duration check for a parsed tree |
| `src/data/parse_from_zip.py` | macOS/Linux demo extraction (bsdtar, no 7-Zip) |
| `tests/test_defuse_progress.py` | 6 synthetic cases, no demos needed |
