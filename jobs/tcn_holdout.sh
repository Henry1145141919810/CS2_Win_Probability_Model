#!/bin/bash
# TCN OUT-OF-TIME: train on ALL 2024-25 training data, evaluate on the 2026 lagged-prior holdout.
# Answers whether the sequence model degrades out-of-time MORE than the classical models do,
# now that firepower is populated correctly (lagged 2025 stats). Submit:  sbatch jobs/tcn_holdout.sh
#
# PREREQ: sync data/test_dataset_2026_lag2025.parquet to Betty first (see docs/BETTY_benchmark_guide.md).
#SBATCH --job-name=cs2-tcn-holdout
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
    --holdout "$PROJ/data/test_dataset_2026_lag2025.parquet" \
    --epochs 40 --patience 6 \
    --save-oof "$PROJ/outputs/holdout_tcn_lag2025.parquet"
