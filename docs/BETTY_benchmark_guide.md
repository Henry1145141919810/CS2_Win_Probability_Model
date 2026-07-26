# Betty benchmark guide — deep-model out-of-time run

**Purpose:** run the deep models (TCN, Transformer, GAT) with the *corrected* firepower data, and
get their **out-of-time** numbers on the 2026 lagged-prior holdout — the one thing that needs a GPU
and therefore Betty. Everything else (all classical models, both holdout variants) is done locally;
see `outputs/holdout_2026_lag2025.csv` and the notes in `docs/results_checkpoint.md`.

> **Context in one line:** the classical benchmark shows firepower **recovers** out-of-time once it
> is fed lagged-2025 stats (the coverage gap that broke it is closed). The open question Betty
> answers: do the sequence models degrade *more* than the classical ones, or do they also recover?

---

## What is local vs what needs Betty

| Work | Where | Status |
|---|---|---|
| Leu's 2026/2025 HLTV scrape | — | done (Task B; 2025 = 82/82) |
| `FIREPOWER_YEAR_LAG` mechanism + rebuild lagged holdout | local | **done** (`data/test_dataset_2026_lag2025.parquet`) |
| Classical 5 models × 4 sets, in-time + out-of-time, CIs | local | **done** |
| Visualizations + notes | local | **done** |
| **Deep models (TCN/Transformer) out-of-time on lagged holdout** | **Betty** | **← your job** |
| GAT out-of-time | Betty | **deferred** (needs a 2026 trajectory dataset — see §5) |

You only need to run **two sbatch jobs**. Optionally also refresh the in-time deep OOF (§4).

---

## 0. Prereqs — get the new file onto Betty

The holdout parquet was rebuilt locally and is **not** in git (data files are gitignored). Copy it
to Betty's project space, and pull the code that has the new `--holdout` flag.

```bash
# on Betty, in the repo:
cd /vast/projects/ajw/wharton/cs2-rwp
git pull                                   # gets the --holdout flag in src/models/deep/*.py + jobs/*_holdout.sh

# from your LOCAL machine, copy the rebuilt holdout parquet up (scp or Globus):
scp data/test_dataset_2026_lag2025.parquet \
    <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
```

Confirm `data/training_dataset.parquet` is already on Betty (it is, from the earlier deep runs) and
that it still has the same-year firepower (training is **not** lagged — only the holdout is).

---

## 1. Smoke test FIRST (always)

Never submit a full job untested. Grab a debug GPU for 30 min and run a tiny version:

```bash
srun --partition=b200-mig45 --gpus=1 --cpus-per-task=4 --mem=16G --time=00:20:00 --pty bash
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
cd /vast/projects/ajw/wharton/cs2-rwp

# 3 epochs, tiny — just proves the --holdout path runs end to end and prints a metric line:
python src/models/deep/tcn.py \
    --data data/training_dataset.parquet \
    --holdout data/test_dataset_2026_lag2025.parquet \
    --epochs 3 --limit-matches 20
```

Expect a line like `TCN OUT-OF-TIME [test_dataset_2026_lag2025]  AUC 0.8x ...`. If it prints, exit
the interactive session and submit the real jobs.

> ⚠️ The `--holdout` code path was written locally but **could not be GPU-tested** here (no CUDA on
> the laptop). The smoke test is how we catch any column/shape mismatch. If it errors, send me the
> traceback — the most likely culprit is a feature-column mismatch between the two parquets, which
> the smoke test will surface immediately.

---

## 2. Submit the two holdout jobs

```bash
sbatch jobs/tcn_holdout.sh
sbatch jobs/transformer_holdout.sh
squeue -u $USER
```

Each trains on all 2024-25 data and evaluates once on the 2026 lagged holdout, printing the full
metric battery + B=500 match-level bootstrap CIs, and saving per-snapshot holdout predictions to
`outputs/holdout_{tcn,transformer}_lag2025.parquet`.

Logs: `tail -f /vast/projects/ajw/wharton/cs2-rwp/logs/cs2-tcn-holdout_<JOBID>.out`

---

## 3. What to send back

For each model, from the log:

- the `... OUT-OF-TIME [...]` line: **AUC, log-loss, Brier, ECE, BSS, cAUC**
- the bootstrap CI block (95% AUC / ECE / cAUC interval)
- and the two `outputs/holdout_*_lag2025.parquet` files (scp back, or leave on Betty and tell me)

Then I fold them into the paper's holdout table next to the classical numbers, and we can finally
answer "do deep models transfer out-of-time?" with data instead of a limitation footnote.

---

## 4. Optional — refresh the in-time deep OOF

The in-time deep numbers in the paper (TCN 0.8488, Transformer 0.8473) predate Leu's data. If you
want them re-confirmed on the current data, re-run the existing CV jobs (already tested, no code
change):

```bash
sbatch jobs/tcn_cv.sh
sbatch jobs/transformer_cv.sh
```

Low priority — they should be unchanged, since training firepower didn't move.

---

## 5. Why GAT is deferred

The GAT reads `data/trajectory_dataset.parquet` — **raw per-player trajectories**, a different data
product from the assembled feature table the TCN/Transformer/classical models use. There is no 2026
trajectory dataset yet; building one means running a trajectory-assembly pass over the isolated 2026
tick channels (`data/holdout2026/parquet/ticks`). That is a separate task. Given the GAT is the
weakest in-time model (0.8465) and inherits the same firepower dependency, its out-of-time number is
the least informative of the three — so we skip it for now and note it as future work. If the
TCN/Transformer results are interesting, I'll build the 2026 trajectory set and add a GAT `--holdout`
path.

---

## Cluster rules reminder (from CLAUDE_CONTEXT.md)

- **All compute via Slurm.** Never run training on a login node.
- Deep jobs → `b200-mig45` (small) or `dgx-b200` (full). Data → project space, never `/vast/home`.
- `kinit` fails if local conda is active → `conda deactivate` locally first.
