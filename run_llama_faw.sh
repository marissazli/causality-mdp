#!/usr/bin/env bash
# Llama-3.1-8B-Instruct baseline run on financial_article_writing.
# Usage: bash run_llama_faw.sh [safe] [extra args, e.g. --start-id 12]
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs results_v2_llama_baseline

SAFE_FLAG=()
LOG=logs/run_llama_faw.log
if [[ "${1:-}" == "safe" ]]; then
    SAFE_FLAG=(--safe)
    LOG=logs/run_llama_faw_safe.log
    shift
fi

CUDA_VISIBLE_DEVICES=1 python run_experiments_v2.py \
    --backend hf \
    --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --hf-dtype bfloat16 \
    --environment financial_article_writing \
    --adversarial-agent CHIEF_EDITOR \
    --cf-agent RESEARCHER \
    --cf-selection-rule last_pre_target_agent \
    --cf-mode baseline \
    --cf-samples 1 \
    --export-tape \
    --max-new-tokens 512 \
    "${SAFE_FLAG[@]}" \
    --output-root results_v2_llama_baseline \
    "$@" 2>&1 | tee "$LOG"
