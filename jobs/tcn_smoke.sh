#!/bin/bash
# SMOKE TEST — validate the whole GPU pipeline cheaply (20 matches, 3 epochs, ~minutes).
# Submit from a LOGIN NODE with:  sbatch jobs/tcn_smoke.sh   (NEVER run python on the login node)
#SBATCH --job-name=cs2-tcn-smoke
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"

PROJ=/vast/projects/ajw/wharton/cs2-rwp
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"
nvidia-smi

python src/models/deep/tcn.py \
    --data "$PROJ/data/training_dataset.parquet" \
    --limit-matches 20 --epochs 3 \
    --checkpoint "$PROJ/checkpoints/tcn_smoke.pt"
