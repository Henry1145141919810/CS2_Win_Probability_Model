#!/bin/bash
# Fast GAT smoke test (20 matches, single split, 5 epochs) -> catch bugs cheaply before the full run.
# Submit:  sbatch jobs/gat_smoke.sh
#SBATCH --job-name=cs2-gat-smoke
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:15:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root; submit with `sbatch` from there (logs/ must exist)
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"; nvidia-smi

python src/models/deep/gat.py --data "$PROJ/data/trajectory_dataset.parquet" \
    --limit-matches 20 --epochs 5
