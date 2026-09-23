#!/bin/bash
# GAT (player self-attention on raw trajectories) with 5-fold OOF -> full metric suite.
# Needs data/trajectory_dataset.parquet uploaded first. Submit:  sbatch jobs/gat_cv.sh
#SBATCH --job-name=cs2-gat-cv
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root; submit with `sbatch` from there (logs/ must exist)
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"
nvidia-smi

python src/models/deep/gat.py --data "$PROJ/data/trajectory_dataset.parquet" \
    --cv --epochs 40 --patience 6 --dim 64 --heads 4 --layers 2 --dropout 0.3
