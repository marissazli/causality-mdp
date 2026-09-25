#!/usr/bin/env bash
# Qwen3-8B replay run on financial_article_writing (same settings as the FAW baseline, --cf-mode replay).
# Usage: bash run_qwen_faw_replay.sh [safe] [extra args, e.g. --start-id 12]
# GPU defaults to 1; override with GPU=0 bash run_qwen_faw_replay.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs results_v2_qwen_replay

SAFE_FLAG=()
LOG=logs/run_faw_replay.log
if [[ "${1:-}" == "safe" ]]; then
    SAFE_FLAG=(--safe)
    LOG=logs/run_faw_safe_replay.log
    shift
fi

CUDA_VISIBLE_DEVICES=${GPU:-1} python run_experiments_v2.py \
    --backend hf \
    --hf-model-id Qwen/Qwen3-8B \
    --environment financial_article_writing \
    --adversarial-agent CHIEF_EDITOR \
    --cf-agent RESEARCHER \
    --cf-selection-rule last_pre_target_agent \
    --cf-mode replay \
    --cf-samples 1 \
    --export-tape \
    --max-new-tokens 512 \
    "${SAFE_FLAG[@]}" \
    --output-root results_v2_qwen_replay \
    "$@" 2>&1 | tee "$LOG"
