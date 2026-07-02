#!/bin/bash
# Fast GAT smoke test (20 matches, single split, 5 epochs) -> catch bugs cheaply before the full run.
# Submit:  sbatch jobs/gat_smoke.sh
#SBATCH --job-name=cs2-gat-smoke
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
PROJ=/vast/projects/ajw/wharton/cs2-rwp
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"; nvidia-smi

python src/models/deep/gat.py --data "$PROJ/data/trajectory_dataset.parquet" \
    --limit-matches 20 --epochs 5
