#!/usr/bin/env bash
# Llama-3.1-8B-Instruct run with the same settings as the Qwen runs.
# Usage: bash run_llama_env.sh <tp|mad|cg|faw> [safe] [extra args, e.g. --start-id 12]
# Replay instead of baseline: CF_MODE=replay bash run_llama_env.sh <env> [safe]
#   (writes to results_v2_llama_replay, logs as run_llama_<env>[_safe]_replay.log)
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
CF_MODE=${CF_MODE:-baseline}
[[ "$CF_MODE" == baseline || "$CF_MODE" == replay ]] || { echo "CF_MODE must be baseline or replay"; exit 1; }
OUTPUT_ROOT=results_v2_llama_${CF_MODE}
LOG_SUFFIX=$([[ "$CF_MODE" == replay ]] && echo _replay || true)
mkdir -p logs "$OUTPUT_ROOT"

ENV_KEY="${1:?usage: run_llama_env.sh <tp|mad|cg|faw> [safe] [extra args]}"
shift
# max-new-tokens: Qwen runs used the default (256) everywhere except FAW (512)
case "$ENV_KEY" in
    tp)  ENVIRONMENT=travel_planning;           ADV=PLANNER_AGENT; CF_AGENT=PLANNER_AGENT; MAX_TOKENS=256 ;;
    mad) ENVIRONMENT=multi_agent_debate;        ADV=agent_4;       CF_AGENT=agent_0;       MAX_TOKENS=256 ;;
    cg)  ENVIRONMENT=code_generation;           ADV=CEO;           CF_AGENT=CEO;           MAX_TOKENS=256 ;;
    faw) ENVIRONMENT=financial_article_writing; ADV=CHIEF_EDITOR;  CF_AGENT=RESEARCHER;    MAX_TOKENS=512 ;;
    *) echo "unknown environment key: $ENV_KEY (use tp, mad, cg, faw)"; exit 1 ;;
esac

SAFE_FLAG=()
LOG=logs/run_llama_${ENV_KEY}${LOG_SUFFIX}.log
if [[ "${1:-}" == "safe" ]]; then
    SAFE_FLAG=(--safe)
    LOG=logs/run_llama_${ENV_KEY}_safe${LOG_SUFFIX}.log
    shift
fi

CUDA_VISIBLE_DEVICES=${GPU:-1} python run_experiments_v2.py \
    --backend hf \
    --hf-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --hf-dtype bfloat16 \
    --environment "$ENVIRONMENT" \
    --adversarial-agent "$ADV" \
    --cf-agent "$CF_AGENT" \
    --cf-selection-rule last_pre_target_agent \
    --cf-mode "$CF_MODE" \
    --cf-samples 1 \
    --export-tape \
    --max-new-tokens "$MAX_TOKENS" \
    "${SAFE_FLAG[@]}" \
    --output-root "$OUTPUT_ROOT" \
    "$@" 2>&1 | tee "$LOG"
