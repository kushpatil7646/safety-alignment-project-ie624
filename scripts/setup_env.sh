#!/usr/bin/env bash
# Set up conda environment for the BackdoorLLM detection project
# Run once on a new machine: bash scripts/setup_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="safety_algn"

echo "=== Setting up environment: $ENV_NAME ==="

# Create conda env
conda create -n "$ENV_NAME" python=3.10 -y
conda activate "$ENV_NAME" || source activate "$ENV_NAME"

# PyTorch with CUDA (adjust cuda version for your GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -r requirements.txt

echo ""
echo "=== Environment '$ENV_NAME' ready ==="
echo "Activate with: conda activate $ENV_NAME"
echo "Then run     : bash scripts/run_experiment.sh --test"
