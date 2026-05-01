#!/usr/bin/env bash
#SBATCH --job-name=backdoor_detect
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

# Anandi / IITB HPC SLURM job script
# Submit with: sbatch scripts/run_slurm.sh [--test]

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"

# Load modules if available (adjust for your cluster)
module load cuda/12.1 2>/dev/null || true
module load python/3.10 2>/dev/null || true

# Activate conda env if it exists
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate safety_algn 2>/dev/null || true
fi

# Set HF cache
export HF_HOME="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME" logs

# Run
MODE=${1:---full}
bash scripts/run_experiment.sh $MODE

echo "End time: $(date)"
