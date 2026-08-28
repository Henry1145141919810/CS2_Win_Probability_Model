# Runbook — re-parse for defuse progress, then evaluate

For whoever holds the raw `.dem` files. Nothing here needs the nav mesh until the assemble
step, and nothing needs a GPU. Background: [`defuse_progress_README.md`](defuse_progress_README.md).

**Why a re-parse at all:** the defuse timer comes from the per-tick `is_defusing` prop, and
no saved parquet has it. Everything else in the pipeline is unchanged.

**Budget:** ~1 min per demo (measured from the existing bundle's file timestamps), so ~4 h for
220 + ~30 min for the 2026 set, plus ~40 min of assemble. Peak RAM ~6.7 GB for one demo;
`batch_parse` waits if free RAM is under 7 GB. Extra disk: one more parquet tree, ~180 MB.

---

## 0. Get the code

```bash
git fetch origin
git checkout exp/defuse-time
python -c "import sys; sys.path.insert(0,'src'); from features.defuse import BOMB_PROGRESS_COLS; print(BOMB_PROGRESS_COLS)"
# -> ['defuse_in_progress', 'defuse_elapsed_sec', 'defuse_progress_frac', 'defuse_beats_fuse']
python tests/test_defuse_progress.py        # 6 cases, needs only polars
```

## 1. ⚠️ Separate the 2026 demos first

**The 2026 out-of-time demos are currently inside the training tree.** `data/parquet/ticks/`
holds 32 files with `2026` in the name, 27 of which are the holdout, and they are in the
shipped `training_dataset.parquet` (247 matches instead of 220) — see
`notes_firepower_v4.md`, Finding 5. If the same thing happens on the re-parse, the new table
inherits the same leak.

```bash
ls demos/extracted | grep -c 2026          # are the 2026 .dem in the training demo dir?
mkdir -p demos/extracted_2026
mv demos/extracted/*2026* demos/extracted_2026/
ls demos/extracted | wc -l                 # should now be the training demos only
```

`assemble.py` will refuse to build a training table containing anything listed in
`configs/demo_list_2026_test.csv`, so a mistake here stops with an error rather than silently
inflating the scores. Do not pass `--allow-test-demos`.

## 2. Re-parse the training demos → a NEW tree

```bash
python src/data/batch_parse.py \
    --raw-dir demos/extracted \
    --out     data/parquet_defuse
```

Never `--out data/parquet` — that overwrites the tree every published number came from.
Progress lines now report the derived channel:

```
    defuse: 5 attempts (4 completed, 1 interrupted)
    <demo>: ok (12s)
```

**Try 3 demos first** (`--limit 3`) and confirm those lines appear before committing 4 hours.
A match where nobody ever touched the bomb legitimately writes no defuse file.

## 3. Check the parse before going further

```bash
python src/data/defuse_report.py --tree data/parquet_defuse
```

Three things must hold, or something is wrong upstream:

- **completed attempts == the count of `defuse` events** in the bomb channel;
- **durations sit on the game constants** — completed with a kit is 4.984 s at min, median
  and max (= 319/64; 320 ticks inclusive = 5.000 s), without a kit ~9.98 s;
- **INTERRUPTED is a real fraction**, ~35% on the 2026 set. If it is near zero the feature is
  a relabelling of `ct_won` and should not be used.

## 4. Re-parse the 2026 holdout → its own tree

```bash
python src/data/batch_parse.py \
    --raw-dir demos/extracted_2026 \
    --out     data/holdout2026/parquet_defuse
python src/data/defuse_report.py --tree data/holdout2026/parquet_defuse
```

## 5. Assemble both tables

`assemble.py` reads `data/parquet` by default, so both the input tree and the output file
have to be given explicitly — otherwise it rebuilds the published table from the old tree.

```bash
python src/features/assemble.py \
    --parquet-root data/parquet_defuse \
    --out          data/training_dataset_defuse.parquet

python src/features/assemble.py \
    --parquet-root data/holdout2026/parquet_defuse \
    --out          data/test_dataset_2026_defuse.parquet
```

Verify before training:

```bash
python - <<'PY'
import polars as pl
tr = pl.read_parquet("data/training_dataset_defuse.parquet")
te = pl.read_parquet("data/test_dataset_2026_defuse.parquet")
print(tr.shape, tr["match_id"].n_unique(), "matches")
print("overlap with test:", len(set(tr["match_id"].unique()) & set(te["match_id"].unique())), "(must be 0)")
print("defusing rows:", int(tr["defuse_in_progress"].sum()),
      f"({tr['defuse_in_progress'].mean()*100:.2f}%)")
PY
```

Expect ~220 matches, **0 overlap**, and roughly 1% defusing rows (~3,700 on the training set).

## 6. Train

```bash
python src/models/train_pipeline.py \
    --data   data/training_dataset_defuse.parquet \
    --sets   EB2,EB2D,EFB2,EFB2D \
    --models logreg,xgb \
    --bootstrap 500
```

`--data` matters: without it the run reads `data/training_dataset.parquet` and the
comparison is against the wrong table.

## 7. Read the result correctly

**Do not judge this on overall AUC.** The feature fires on ~1% of rows and moves pooled AUC
by ~+0.001. That is expected, and it is the whole reason the paper argues contested-AUC
rather than pooled AUC is the honest metric.

Judge it on the defusing rows and on the shape of the curve:

```bash
python src/models/pilot_defuse_lr.py \
    --data data/training_dataset_defuse.parquet --sets EB2,EB2D
```

The table to look at is *mean predicted P(CT win) by defuse progress*. On the 2026 pilot,
`EB2` stayed flat at 0.894 → 0.897 while the empirical rate went 0.909 → 0.983, and `EB2D`
tracked it at 0.897 → 0.988.

## 8. Two things worth re-testing on the full data

The pilot trained on 13–14 matches, which is too small to settle either:

- **Does XGB learn the fuse threshold on its own?** In the pilot it learned nothing from these
  columns in any encoding (log-loss ~0.22 throughout, baseline curve sloping the wrong way) —
  the defusing regime was ~315 rows against `min_child_weight=10`. With ~3,700 it may behave
  differently, in which case `defuse_beats_fuse` could be replaced by a raw `fuse_time_left`.
- **`fuse_time_left` as a feature in its own right.** How much fuse is left is a basic
  post-plant state variable that no current feature exposes: `time_elapsed_sec` counts from
  freeze-end, `bomb_planted` is a flag, and `defuse_time_margin` folds the fuse into a
  distance term.

## Troubleshooting

| symptom | cause |
|---|---|
| `REFUSING TO ASSEMBLE: N out-of-time test demos ...` | step 1 was skipped — the guard working |
| `no defuse channel in <tree>` from defuse_report | parsed with old code; check `is_defusing` is in `PLAYER_PROPS` |
| every demo re-parses on a second run | fixed; the skip-check no longer requires the defuse file, which a bomb-free match never writes |
| `ColumnNotFoundError: official_end` on one demo | a split (`-p2`) demo fragment with incomplete rounds; the existing bundle drops it too |
| `FileNotFoundError: ~/.awpy/navs/de_inferno.json` | assemble needs the nav mesh; awpy's mirror is dead, so copy the file from a machine that already has it |
