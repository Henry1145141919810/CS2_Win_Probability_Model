# PARCC Betty — Deep-Learning Test Runbook (TCN)

Goal: run the first deep-learning model (causal **TCN**, per-second win prob) on Betty, the
safe way. Full cluster reference: `e:/CLAUDE_CONTEXT.md`. **Golden rule: never run `python`
on a login node — everything compute goes through Slurm (`sbatch`/`srun`).**

The model + jobs are in this repo: `src/models/deep/tcn.py`, `jobs/tcn_smoke.sh`,
`jobs/tcn_train.sh`. Strategy: **smoke test first** (20 matches, 3 epochs, minutes), then full.

---

## 0. One-time setup on Betty

**(a) Connect** (from local; deactivate local conda first so `kinit` works):
```bash
conda deactivate
kinit <PennKey>@UPENN.EDU                       # uppercase realm; 10h ticket; Duo prompt
ssh <PennKey>@login.betty.parcc.upenn.edu
hostname                                         # note login0X (for tmux reattach)
```

**(b) Get the code** into the project space (1TB), not home:
```bash
cd /vast/projects/ajw/wharton/cs2-rwp           # == ~/projects/cs2-rwp
git clone https://github.com/Henry1145141919810/CS2_Win_Probability_Model.git .
# (or: git pull, if already cloned)
mkdir -p logs checkpoints data
```

**(c) Create the conda env** — do this inside an *interactive Slurm session*, NOT the login node:
```bash
srun --partition=genoa-std-mem --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
module load anaconda3 && source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p "$HOME/envs/cs2-rwp" python=3.11 uv -c conda-forge
conda activate "$HOME/envs/cs2-rwp"
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # Betty B200 = cu128
uv pip install polars numpy scikit-learn pyarrow
exit                                              # leave the interactive session
```
(The TCN script needs only torch/polars/numpy/scikit-learn/pyarrow — light. awpy/xgboost/etc.
are only needed if you also rebuild features on the cluster.)

---

## 1. Get the data onto Betty (81 MB → plain scp is fine, <1GB rule)
From your **local laptop** (new terminal, repo root):
```bash
scp data/training_dataset.parquet \
    <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/data/
```
(If you'd rather rebuild on the cluster, you'd also need the parsed parquet/ demos + awpy —
but for a model test, shipping the 81 MB training file is far simpler.)

---

## 2. GPU sanity check (30 seconds)
```bash
srun -p b200-mig45 --gpus=1 -t 00:02:00 nvidia-smi      # should print a GPU
```
If that shows a GPU, the allocation + drivers work.

---

## 3. Smoke test (cheap end-to-end validation)
```bash
cd /vast/projects/ajw/wharton/cs2-rwp
sbatch jobs/tcn_smoke.sh
squeue -u $USER                                          # watch state: PD -> R -> CD
tail -f logs/cs2-tcn-smoke_<JOBID>.out                   # live log
```
Expect: `device=cuda (NVIDIA B200 ...)`, a few epochs printing `val_AUC`, and a saved
checkpoint. AUC will be mediocre (only 20 matches, 3 epochs) — that's fine; you're testing the
**pipeline**, not performance. If it crashes, the log tells you what to fix (usually a missing
package or a path).

---

## 4. Full run (only after the smoke test passes)
```bash
sbatch jobs/tcn_train.sh                                  # dgx-b200, 30 epochs, ~minutes-1h
tail -f logs/cs2-tcn_<JOBID>.out
```
Success criterion for the experiment: **val AUC approaching/beating the classical baseline
(logreg EB2 ≈ 0.851)**. The script prints best val AUC and saves `checkpoints/tcn.pt`.

---

## 5. Retrieve results to local (small files only)
```bash
# from local laptop:
scp <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/logs/cs2-tcn_<JOBID>.out .
scp <PennKey>@login.betty.parcc.upenn.edu:/vast/projects/ajw/wharton/cs2-rwp/checkpoints/tcn.pt .
```

---

## Cluster rules (don't get suspended)
- **Never** `python ...` on a login node — only `sbatch`/`srun`, editing, `git`, `scp` (<1GB).
- Big transfers → **Globus**; here scp is OK because the file is 81 MB.
- **Home = 50 GB, code only.** All data/checkpoints/logs → `/vast/projects/ajw/wharton/cs2-rwp/`.
- **No `pip install --user`** — only the conda env.
- Check before big jobs: `parcc_quota.py`, `parcc_sfree.py`, `parcc_sqos.py`.
- Long bootstrap loops must checkpoint every 10 iterations (the TCN already checkpoints best epoch).

---

## Troubleshooting
| symptom | fix |
|---|---|
| `kinit` fails | local conda still active → `conda deactivate`; realm must be UPPERCASE |
| job stuck `PD` | `scontrol show job <ID>` (reason); try `b200-mig45` for the smoke test |
| `CUDA not available` in log | wrong env / non-GPU partition; confirm `--partition=dgx-b200`/`b200-mig45` + `--gpus=1` |
| `ModuleNotFoundError` | re-run the `uv pip install` step inside the env |
| OOM | lower `--batch` (e.g. 32) or `--seq-len` |

## Next deep models (same pattern)
GAT (player-graph), Transformer encoder, and the XGBoost+Transformer ensemble reuse this
scaffold — copy `jobs/tcn_train.sh`, swap the script. Build them once the TCN test is green.
