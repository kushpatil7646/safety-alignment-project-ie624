#!/bin/bash
#SBATCH --job-name=backdoor_test
#SBATCH --account=cminds_anandi
#SBATCH --partition=cn4_mangala
#SBATCH --qos=mangala
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --output=/users/student/idddp/kushpatil/Safety_algn/logs/backdoor_test_%j.out

set -e

echo "======================================================"
echo "TEST RUN (GPT-2) — pipeline sanity check"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURM_NODELIST"
echo "Start time : $(date)"
echo "======================================================"

source /users/student/idddp/kushpatil/miniconda3/etc/profile.d/conda.sh
conda activate myenv

cd /users/student/idddp/kushpatil/Safety_algn
mkdir -p logs results

python3 -c "import umap" 2>/dev/null || pip install umap-learn --quiet

export HF_HOME="/scratch/$USER/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

echo "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"

python3 -m src.evaluate \
    --config configs/config.yaml \
    --test \
    --log-level INFO

echo "Test complete: $(date)"
