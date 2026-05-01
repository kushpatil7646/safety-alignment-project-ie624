#!/usr/bin/env bash
# Full experiment runner for the BackdoorLLM Detection Pipeline
# Usage:
#   ./scripts/run_experiment.sh [--test] [--config configs/config.yaml]
#
# --test  : use GPT-2 (small, fast) to verify the pipeline end-to-end
# --full  : use BackdoorLLM models from HuggingFace (needs ~40GB GPU RAM)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

CONFIG="configs/config.yaml"
MODE="full"
LOG_LEVEL="INFO"

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)   MODE="test"; shift ;;
        --full)   MODE="full"; shift ;;
        --config) CONFIG="$2"; shift 2 ;;
        --debug)  LOG_LEVEL="DEBUG"; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Environment ───────────────────────────────────────────────────────────────
echo "=== BackdoorLLM Detection Pipeline ==="
echo "Mode     : $MODE"
echo "Config   : $CONFIG"
echo "Date     : $(date)"
echo "Host     : $(hostname)"
echo "Python   : $(python3 --version)"

if python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())" 2>/dev/null; then
    python3 -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || true
fi
echo ""

# ── Install dependencies if needed ────────────────────────────────────────────
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing dependencies ..."
    pip install -r requirements.txt --quiet
fi

# Set HF cache to scratch space if available
if [ -d "/scratch" ]; then
    export HF_HOME="/scratch/hf_cache"
    export TRANSFORMERS_CACHE="/scratch/hf_cache"
    mkdir -p "$HF_HOME"
    echo "HF cache: $HF_HOME"
fi

# ── Run evaluation ────────────────────────────────────────────────────────────
mkdir -p results logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_${MODE}_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

if [ "$MODE" = "test" ]; then
    echo ">>> Running in TEST mode (GPT-2 only, quick pipeline check) <<<"
    python3 -m src.evaluate \
        --config "$CONFIG" \
        --test \
        --log-level "$LOG_LEVEL" \
        2>&1 | tee "$LOG_FILE"
else
    echo ">>> Running in FULL mode (BackdoorLLM models from HuggingFace) <<<"
    python3 -m src.evaluate \
        --config "$CONFIG" \
        --log-level "$LOG_LEVEL" \
        2>&1 | tee "$LOG_FILE"
fi

echo ""
echo "=== Experiment complete ==="
echo "Results saved to: results/"
echo "Log saved to: $LOG_FILE"

# ── Print summary ─────────────────────────────────────────────────────────────
if [ -f "results/summary.json" ]; then
    echo ""
    echo "=== SUMMARY ==="
    python3 -c "
import json, sys
with open('results/summary.json') as f:
    s = json.load(f)
ours = s.get('our_method', {})
print(f\"Our Method  | AUROC={ours.get('auroc', 'N/A'):.4f}  FPR@95TPR={ours.get('fpr_at_95tpr', 'N/A'):.4f}  Acc={ours.get('accuracy', 'N/A'):.4f}\")
for name, m in s.get('baselines', {}).items():
    print(f\"{name:20s}| AUROC={m.get('auroc','N/A'):.4f}  FPR@95TPR={m.get('fpr_at_95tpr','N/A'):.4f}  Acc={m.get('accuracy','N/A'):.4f}\")
"
fi
