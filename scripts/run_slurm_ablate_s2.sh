#!/bin/bash
#SBATCH --job-name=bdoor_ablate_s2
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn3_anandi
#SBATCH --qos=anandi

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/users/student/idddp/kushpatil/Safety_algn/logs/ablate_s2_%j.out

set -e

echo "======================================================"
echo "ABLATION: skip Stage 2 (Representation)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "======================================================"

source /users/student/idddp/kushpatil/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd /users/student/idddp/kushpatil/Safety_algn
mkdir -p logs results_gpu_ablate_s2

python3 -c "import umap" 2>/dev/null || pip install umap-learn --quiet

export HF_HOME="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
# HF_TOKEN must be set in your environment before sbatch (e.g. export HF_TOKEN=<your_token>)
: "${HF_TOKEN:?HF_TOKEN is not set — run: export HF_TOKEN=<your_token>}"
mkdir -p "$HF_HOME"

echo ""
echo "Python : $(python3 --version)"
echo "CUDA   : $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

python3 -m src.evaluate \
    --config configs/config.yaml \
    --skip-stages 2 \
    --output-dir results_gpu_ablate_s2 \
    --log-level INFO

echo ""
echo "======================================================"
echo "Ablation skip-S1 complete: $(date)"
echo "Results in: results_gpu_ablate_s2/"
echo "======================================================"
