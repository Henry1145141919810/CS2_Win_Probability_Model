# Betty guide — deep models with the defuse-progress feature

For the deep-model half of the defuse-progress benchmark. The classical half runs locally; this is the
GPU part you run on Betty once the training table is re-parsed. Companion:
`docs/defuse_progress_README.md`, `docs/BETTY_benchmark_guide.md`.

## What changed and why it needs a GPU re-run

The re-parse adds 4 columns (`defuse_in_progress, defuse_elapsed_sec, defuse_progress_frac,
defuse_beats_fuse`) to the training and 2026 tables. The deep models (TCN, Transformer, GAT) consume
**every** feature column, so once they train on the re-parsed table they pick up the defuse block
automatically. No code change is needed for that; only the data file changes.

## Prereqs — get the new tables onto Betty

Both tables are gitignored (data files), so scp them up. From your LOCAL machine, after the local
re-parse + assemble finish:

```bash
cd /mnt/e/CS2_Win_Prob_Model   # (WSL path; adjust if different)
scp data/training_dataset_defuse.parquet data/test_dataset_2026_defuse.parquet \
    hyhuang@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
```

On Betty, pull the branch that has the parsing/defuse code:

```bash
cd /vast/projects/ajw/wharton/cs2-rwp
git fetch origin && git checkout feat/defuse-progress   # or whatever branch it lands on
```

## 1. Smoke test first (always)

> **Cluster CLI-filter rule (since 2026-08-03):** `sbatch` rejects a wrong CPU:GPU ratio
> (`CPUS_PER_GPU_MISMATCH`). Required CPUs/GPU: `b200-mig45` = **6**, `b200-mig90` = 14,
> `dgx-b200` = 28 (mem cap 8 GB/CPU). All job scripts here use 6 on mig45.

```bash
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=6 --mem=32G --time=00:20:00 --pty bash
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
cd /vast/projects/ajw/wharton/cs2-rwp
python src/models/deep/tcn.py \
    --data data/training_dataset_defuse.parquet \
    --holdout data/test_dataset_2026_defuse.parquet \
    --epochs 3 --limit-matches 20
```

Expect a `TCN OUT-OF-TIME [...]` line. The feature-column count should now be **+4** vs the previous
run (the schema-mismatch guard added earlier will otherwise print exactly which columns differ).

## 2. Out-of-time deep runs (the ones that matter)

```bash
sbatch jobs/tcn_holdout.sh          # edit --data/--holdout to the _defuse tables first
sbatch jobs/transformer_holdout.sh
```

Edit each job's `--data` to `data/training_dataset_defuse.parquet` and `--holdout` to
`data/test_dataset_2026_defuse.parquet`, and change `--save-oof` to a `*_defuse.parquet` name so it
does not overwrite the earlier predictions.

## 3. What to report back

For each model: the `OUT-OF-TIME` metric line + bootstrap CI, and (this is the point of the feature)
the **per-second curve honesty on defusing rows**. Save the holdout predictions
(`--save-oof outputs/holdout_tcn_defuse.parquet`) and scp them back, or run this on Betty:

```python
import polars as pl, numpy as np
from sklearn.metrics import log_loss, brier_score_loss
te = pl.read_parquet("data/test_dataset_2026_defuse.parquet")
oof = pl.read_parquet("outputs/holdout_tcn_defuse.parquet")   # match_id, y, p_tcn
m = te.join(oof, on="match_id")  # or on (match_id, tick) if saved
d = m.filter(pl.col("defuse_in_progress") == 1)
y, p = d["ct_won"].to_numpy().astype(float), d["p_tcn"].to_numpy()
print("defusing rows:", d.height, "log-loss", log_loss(y, p, labels=[0,1]), "brier", brier_score_loss(y, p))
for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]:
    b = d.filter((pl.col("defuse_progress_frac")>=lo) & (pl.col("defuse_progress_frac")<hi))
    if b.height: print(f"  {lo:.1f}-{hi:.1f}: n={b.height} actual {b['ct_won'].mean():.3f} pred {b['p_tcn'].mean():.3f}")
```

## 4. Isolating the feature in a deep model (optional, secondary)

Deep models ingest all columns, so unlike the classical EB2-vs-EB2D comparison they cannot toggle the
4 columns via a feature-set name. To measure the deep marginal effect cleanly, run the same job twice:
once on the full `_defuse` table, once on a copy with the 4 defuse columns dropped:

```bash
python - <<'PY'
import polars as pl
for f in ["data/training_dataset_defuse.parquet","data/test_dataset_2026_defuse.parquet"]:
    pl.read_parquet(f).drop(["defuse_in_progress","defuse_elapsed_sec","defuse_progress_frac","defuse_beats_fuse"]).write_parquet(f.replace(".parquet","_nodefuse.parquet"))
PY
```

Then compare the two runs on the defusing-row curve above. This is lower priority: the feature fires
on ~1% of rows and (like the classical case) will barely move pooled AUC; the value is the curve, not
the headline number. Only worth doing if the classical result says the feature earns a place in the
paper.

## Cluster rules reminder
All compute via Slurm; deep jobs on `b200-mig45` (small) or `dgx-b200`; data in project space, never
`/vast/home`; `conda deactivate` locally before `kinit`.
