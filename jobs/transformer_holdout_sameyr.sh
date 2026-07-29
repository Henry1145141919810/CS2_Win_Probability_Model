#!/bin/bash
# Transformer OUT-OF-TIME (SAME-YEAR variant): train on ALL 2024-25 data, evaluate on the 2026
# SAME-YEAR holdout (full coverage + same-era firepower -- the leaky best case). Companion to
# jobs/transformer_holdout.sh (lagged). Submit:  sbatch jobs/transformer_holdout_sameyr.sh
#
# PREREQ: sync data/test_dataset_2026_sameyr.parquet to Betty first (see docs/BETTY_benchmark_guide.md).
#SBATCH --job-name=cs2-tf-holdout-sy
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"

PROJ=/vast/projects/ajw/wharton/cs2-rwp
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"
nvidia-smi

python src/models/deep/transformer.py \
    --data "$PROJ/data/training_dataset.parquet" \
    --holdout "$PROJ/data/test_dataset_2026_sameyr.parquet" \
    --epochs 40 --patience 6 --dim 64 --heads 4 --layers 2 --dropout 0.1 \
    --save-oof "$PROJ/outputs/holdout_transformer_sameyr.parquet"
