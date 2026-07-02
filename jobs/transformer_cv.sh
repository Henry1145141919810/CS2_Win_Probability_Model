#!/bin/bash
# Causal Transformer, 5-fold OOF -> full metrics + B=500 CIs, and SAVE OOF preds for the ensemble.
# Submit:  sbatch jobs/transformer_cv.sh
#SBATCH --job-name=cs2-tf-cv
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
echo "host=$(hostname)  date=$(date)"; nvidia-smi

python src/models/deep/transformer.py --data "$PROJ/data/training_dataset.parquet" \
    --cv --epochs 40 --patience 6 --dim 64 --heads 4 --layers 2 --dropout 0.1 \
    --save-oof "$PROJ/outputs/oof_transformer.parquet"
