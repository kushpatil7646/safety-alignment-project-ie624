#!/bin/bash
#SBATCH --job-name=backdoor_detect
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --output=/users/student/idddp/kushpatil/Safety_algn/logs/backdoor_detect_%j.out

set -e

echo "======================================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "======================================================"

source /users/student/idddp/kushpatil/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd /users/student/idddp/kushpatil/Safety_algn
mkdir -p logs results

# Install umap-learn if not present
python3 -c "import umap" 2>/dev/null || pip install umap-learn --quiet

# HF cache on scratch (fast NVMe, avoid quota issues)
export HF_HOME="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

echo ""
echo "Python : $(python3 --version)"
echo "CUDA   : $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "HF cache: $HF_HOME"
echo ""

# ── Run the full experiment ────────────────────────────────────────────────────
python3 -m src.evaluate \
    --config configs/config.yaml \
    --log-level INFO

echo ""
echo "======================================================"
echo "Experiment complete: $(date)"
echo "Results in: /users/student/idddp/kushpatil/Safety_algn/results/"
echo "======================================================"
