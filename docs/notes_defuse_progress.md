# Notes — Defuse progress ("how many seconds into the defuse")

**Date:** 2026-08-26 · **Branch:** `exp/defuse-time` (not merged, not pushed) ·
**Code:** `src/features/defuse.py`, `src/data/batch_parse.py` (`_defuse_attempts`),
`src/data/parse_from_zip.py`, `src/data/defuse_report.py`,
`src/features/assemble_defuse_pilot.py`, `src/models/pilot_defuse_lr.py` ·
**Data:** `data/holdout2026/parquet_defuse/` (32 demos, re-parsed), `data/holdout2026/pilot_defuse.parquet`

## What this is

A per-snapshot feature for the state the model is currently blind to: a CT is **on the bomb right
now**, and the defuse bar is *N* seconds along. The existing bomb features are all counterfactual —
`defuse_time_margin` asks "*could* a CT get there in time?" from geometry — and say nothing about a
defuse that is actually happening.

Scope is deliberately narrow. The feature fires on ~1% of snapshots and **cannot** move a global
metric; it is a **per-second WP curve honesty** fix, in the same family as the pathwise-calibration
work (`notes_pathwise_calibration.md`), not an accuracy play. See "Quarantine" below.

## Why `is_defusing` and not a defuse-start event

The parsed bundle has no defuse-start information at all. Two findings, both verified:

1. **awpy never asks for it.** `awpy.parsers.bomb.parse_bomb()` hardcodes five events —
   `bomb_dropped / bomb_pickup / bomb_planted / bomb_exploded / bomb_defused` — which is exactly the
   `drop/pickup/plant/detonate/defuse` vocabulary in `data/parquet/bomb/`. `DEFAULT_EVENT_LIST` omits
   `bomb_begindefuse` and `DEFAULT_PLAYER_PROPS` omits `is_defusing`.
2. **The events are empty even when requested.** `Demo.parse()` accepts a custom `events=` list;
   asking for `bomb_begindefuse` / `bomb_abortdefuse` on a real 2026 demo returned **0 rows** for
   both, while `bomb_defused` returned 4. Route A is dead.

`is_defusing` (engine field `CCSPlayerPawn.m_bIsDefusing`) does work, and is a one-line addition to
`PLAYER_PROPS`.

**Do NOT back-derive a start from a completed defuse** (`defuse_tick − 5 s`). That only ever sees
defuses that *succeeded*, which turns the feature into a relabelling of `ct_won`.

## Why a derived channel and not just a tick column

The project has one uniform time convention, with no exceptions:

```
time_elapsed_sec = (tick - round_row["freeze_end"]) / 64.0     economy.py:37
time_left        = 40.0 - (tick - plant["tick"]) / 64.0        bomb.py:200,244
time_elapsed_sec = (tick - rr["freeze_end"]) / 64.0            build_trajectory_dataset.py:55
```

`(snapshot tick − reference tick) / 64`, where the reference always lives in a table that is **not**
downsampled (`rounds.freeze_end`, the plant tick in `bomb`). Ticks are saved at 1 Hz (`--stride 64`),
so `is_defusing` alone would only ever yield integer seconds with a ±1 s error on the start.

So `batch_parse` derives a `defuse` channel from the **full 64 Hz stream, before downsampling** — one
row per attempt: `round_num, steamid, start_tick, end_tick, had_kit, completed`. It plays exactly the
role the bomb table plays for the plant tick, and `defuse_elapsed_sec` lands at the same 1/64 s
precision as every other time feature. The table is ~5 rows per demo.

That this matters is visible in the data: the five attempts in one demo start at fractional offsets
**.89 / .22 / .64 / .61 / .72 s**. At 1 Hz all five would have read `0,1,2,3,4`.

## Features (`BOMB_PROGRESS_COLS`)

| column | meaning |
|---|---|
| `defuse_in_progress` | someone is on the bomb at this snapshot |
| `defuse_elapsed_sec` | seconds into the current attempt |
| `defuse_progress_frac` | elapsed / required, kit-aware (5 s with a defuse kit, 10 s without) |
| `defuse_beats_fuse` | remaining defuse time fits inside the remaining fuse |
| `defuse_attempts_so_far` | attempts started this round so far (>1 means an earlier one failed) |

All five are **0** outside an attempt — the truthful value, so `nan_to_num` in `train_pipeline`
cannot invent an "about to finish" state.

## Data validation (32 re-parsed 2026 demos)

Three independent checks, all pass:

- **Attempt count cross-checks.** 87 attempts flagged `completed` — identical to the 87 `defuse`
  events counted separately from the `bomb` channel.
- **Durations sit exactly on the game constants.** Completed with a kit: n=54, min = median = max =
  **4.984 s** (= 319/64; inclusive of both endpoints it is 320 ticks = 5.000 s). Without a kit:
  median 9.984 s.
- **No flicker.** Every attempt is a perfectly contiguous tick run, and the `bomb_defused` event
  fires on `end_tick + 1` every time.

## Feasibility — is it a relabelling of `ct_won`?

**No.** Across 31 demos (one match had zero defuses):

```
attempts    135
completed    87
INTERRUPTED  48   (35.6%)
```

Progress reached by the interrupted attempts is bimodal: 26 of 48 died under 10% (a CT taps the bomb
and lets go, or is shot instantly), but **4 got past 75%** — the most extreme being a no-kit defuse
interrupted at **9.984 s of the 10 s** it needed. Those are the rows that teach the model a full bar
can still lose.

Coverage: 677 snapshot-seconds over 31 demos = **1.22%** of the 2026 table, of which 79 s come from
interrupted attempts.

## Pilot result (LR, 2026 split in half by match, 2-fold cross-fitting)

`src/models/pilot_defuse_lr.py`. Nav-free feature sets (see "Blocked on" below): `A` = economy,
`AD` = economy + defuse progress.

**Mean predicted P(CT win) by defuse progress, defusing rows only:**

| progress | actual | A (economy) | AD (+ defuse) |
|---|---|---|---|
| 0–20% | 0.909 | 0.805 | **0.897** |
| 20–40% | 0.924 | 0.798 | **0.932** |
| 40–60% | 0.951 | 0.807 | **0.957** |
| 60–80% | 0.975 | 0.818 | **0.972** |
| 80–100% | 0.983 | 0.824 | **0.979** |

**The baseline is flat** (.805 → .824) while a defuse runs to 80% — that is the defect, measured
rather than assumed. With the feature the curve tracks the empirical one.

| set | AUC | log-loss | Brier | log-loss (defusing rows) | Brier (defusing) |
|---|---|---|---|---|---|
| A (17) | 0.8296 | 0.4998 | 0.1678 | 0.2833 | 0.0830 |
| AD (22) | 0.8306 | 0.4983 | 0.1673 | **0.1173** (−59%) | **0.0306** (−63%) |

Overall AUC moves **+0.0010** — the expected non-result at 1.14% coverage. Judging this feature by
global AUC would be the wrong test.

**Leakage check passes.** On interrupted attempts the model predicts **0.562**, not the 0.967 it
predicts on completed ones: it has not memorised "defusing ⇒ CT wins". (0.562 against a true 0.0 is
still badly calibrated, but whether a T kills the defuser is not predictable from board state, so
that is the honest ceiling.)

## Limits attached to the pilot numbers

1. **Half the touch-once 2026 holdout was trained on.** These numbers must never be quoted beside the
   official 27-match out-of-time result. Output is isolated at `data/holdout2026/pilot_defuse.parquet`.
2. **The sample cannot support wide feature sets.** `ATF` (65 features) scores 0.8196, *below* `A`'s
   0.8296, on 13 training matches.
3. **Only 34 interrupted defusing rows survive 1 Hz sampling** (most interrupted attempts last well
   under a second).
4. **completed/interrupted is proxied by the round label** rather than the channel's `completed`
   column. A strict version should use the column.
5. **No map-control or bomb-geometry features** (see below), so nothing here is comparable to any
   published number.

## Blocked on

- **`~/.awpy/navs/de_inferno.json`.** awpy's data mirror (`awpycs.com`) is dead — every resource 404s
  and the domain 301s to the GitHub repo; 2.0.2 is the latest release, so `awpy get navs` cannot be
  recovered. `assemble.py` fails at import without it (`mapcontrol.load_nav`). Needs a copy from a
  machine that fetched it while the mirror was up. The awpy 1.x pip package bundles a `de_inferno.txt`,
  but it is a CS:GO-era coordinate list in a different format — **not** a substitute.
- **The 220 training demos re-parsed with `is_defusing`.** Until then the feature can only be studied
  on the 2026 set, which costs half the holdout. Estimated 4.4 h from the existing bundle's file
  timestamps (~1 min/demo); the raw demos never need to leave that machine, only the ~170 MB parquet.

## Quarantine

`BOMB_PROGRESS_COLS` lives only in the `EB3` / `EFB3` sets in `train_pipeline.py`. A defuse that runs
to completion *is* the CT win, so folding these columns into `E` / `EFB2` would let endgame certainty
be credited to the map-control contribution. The map-control headline numbers must keep coming from
sets that exclude them.

## Tooling added along the way

- `src/data/parse_from_zip.py` — streams demos out of an HLTV zip bundle on macOS/Linux using the
  system `bsdtar` (libarchive ≥3.4 reads RAR5), no 7-Zip needed. One archive at a time, inferno only,
  parse, delete: peak extra disk ~1.5 GB instead of the ~70 GB a full unpack of the 23 GB bundle
  would need. ~15 s per archive.
- `src/data/defuse_report.py` — attempts / interrupted / duration distribution for a parsed tree.
- `src/data/inspect_bomb_events.py` — bomb-event vocabulary probe (how finding 1 above was made).

**Pipeline reproduction check.** Re-parsing the 2026 demos from raw `.dem` and assembling through the
pilot path produced **55,271 rows over 27 matches at ct_won 0.512** — the same shape as the official
`data/test_dataset_2026.parquet`, by an independent route.
