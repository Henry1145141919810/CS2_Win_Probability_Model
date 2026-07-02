#!/bin/bash
# Hyperparameter sweep for the TCN (5-fold OOF each). Finds the best config; ~1 min/config on B200.
# Submit:  sbatch jobs/tcn_sweep.sh   then read:  grep -E "CONFIG|OOF" logs/cs2-tcn-sweep_*.out
#SBATCH --job-name=cs2-tcn-sweep
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=b200-mig45
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/cs2-rwp"
PROJ=/vast/projects/ajw/wharton/cs2-rwp
cd "$PROJ"
echo "host=$(hostname)  date=$(date)"

# dropout hidden lr seqlen
CONFIGS=(
  "0.3 48 1e-3 160"
  "0.4 48 1e-3 160"
  "0.5 48 1e-3 160"
  "0.4 32 1e-3 160"
  "0.4 64 1e-3 160"
  "0.4 48 5e-4 160"
  "0.4 48 1e-3 100"
  "0.4 48 1e-3 64"
)
for c in "${CONFIGS[@]}"; do
  set -- $c
  echo ""; echo "CONFIG dropout=$1 hidden=$2 lr=$3 seqlen=$4"
  python src/models/deep/tcn.py --cv --epochs 40 --patience 6 \
      --dropout "$1" --hidden "$2" --lr "$3" --seq-len "$4" | grep -E "OOF|baseline"
done
