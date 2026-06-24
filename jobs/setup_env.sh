#!/bin/bash
# ONE-TIME environment build, run on a COMPUTE node (not the login node) via Slurm.
# Submit:  sbatch jobs/setup_env.sh    then watch:  tail -f logs/cs2-setup-env_<JOBID>.out
#SBATCH --job-name=cs2-setup-env
#SBATCH --output=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.out
#SBATCH --error=/vast/projects/ajw/wharton/cs2-rwp/logs/%x_%j.err
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00

set -euo pipefail
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -p "$HOME/envs/cs2-rwp" python=3.11 uv -c conda-forge
conda activate "$HOME/envs/cs2-rwp"
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # Betty B200 = cu128
uv pip install polars numpy scikit-learn pyarrow

python -c "import torch; print('torch', torch.__version__)"
echo "ENV READY"
