#!/bin/bash
# FULL TCN training (all matches, 30 epochs). Run only AFTER the smoke test passes.
# Submit:  sbatch jobs/tcn_train.sh
#SBATCH --job-name=cs2-tcn
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"

PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-$PWD}}"   # repo root; submit with `sbatch` from there (logs/ must exist)
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"
nvidia-smi

python src/models/deep/tcn.py \
    --data "$PROJ/data/training_dataset.parquet" \
    --epochs 30 --batch 64 --hidden 64 --dropout 0.2 --lr 1e-3 \
    --checkpoint "$PROJ/checkpoints/tcn.pt"
