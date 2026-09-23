#!/bin/bash
# Multi-seed TCN (5-fold OOF per seed) -> deep-model uncertainty (mean +/- std across seeds).
# This is the deep-model CI for the standard eval protocol. Submit:  sbatch jobs/tcn_seeds.sh
#SBATCH --job-name=cs2-tcn-seeds
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

# Use the best config from the sweep here (edit dropout/hidden if the sweep finds better).
for s in 0 1 2 3 4; do
  echo ""; echo "SEED $s"
  python src/models/deep/tcn.py --cv --seed "$s" --epochs 40 --patience 6 \
      --dropout 0.4 --hidden 48 | grep -E "OOF|baseline"
done
echo ""; echo "-> average the 5 'TCN (5-fold OOF)' AUCs for mean +/- std (the deep-model CI)"
