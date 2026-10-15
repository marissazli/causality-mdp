#!/usr/bin/env bash
# Rerun specific financial_article_writing rows into a fresh *_rerun folder, with the
# same settings as the original A6000 runs. Used for:
#   - Qwen non-safe baseline rows that came from other GPUs with seed 49 instead of 49+row
#   - rows where generate_image ran while its ID was random (now a prompt hash)
# Each row uses --start-id N --end-id N so seed = 49+N and output lands in row=N.
# Rows already finished in the rerun folder are skipped, so the script is restartable.
#
# Usage: bash run_faw_rerun.sh <qwen|llama> <baseline|replay> <nonsafe|safe> ROW [ROW ...]
# GPU defaults to 1; GPU=0 to override.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

MODEL_KEY=${1:?usage: run_faw_rerun.sh <qwen|llama> <baseline|replay> <nonsafe|safe> ROW...}
CF_MODE=${2:?missing baseline|replay}
SAFETY=${3:?missing nonsafe|safe}
shift 3
ROWS=("$@")
[[ ${#ROWS[@]} -gt 0 ]] || { echo "no rows given"; exit 1; }

case "$MODEL_KEY" in
    # Qwen FAW runs used the default float16; Llama runs used bfloat16
    qwen)  MODEL_ID=Qwen/Qwen3-8B;                         MODEL_DIR=Qwen3-8B;                   DTYPE=float16 ;;
    llama) MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct; MODEL_DIR=Meta-Llama-3.1-8B-Instruct; DTYPE=bfloat16 ;;
    *) echo "model must be qwen or llama"; exit 1 ;;
esac
[[ "$CF_MODE" == baseline || "$CF_MODE" == replay ]] || { echo "mode must be baseline or replay"; exit 1; }
SAFE_FLAG=(); SAFE_DIR=""; SAFE_TAG=""
case "$SAFETY" in
    nonsafe) ;;
    safe) SAFE_FLAG=(--safe); SAFE_DIR="/safe=True"; SAFE_TAG="_safe" ;;
    *) echo "safety must be nonsafe or safe"; exit 1 ;;
esac

OUTPUT_ROOT=results_v2_${MODEL_KEY}_${CF_MODE}_rerun
ROW_DIR=$OUTPUT_ROOT/env=financial_article_writing/model=$MODEL_DIR/adv=CHIEF_EDITOR$SAFE_DIR
LOG=logs/run_faw_rerun_${MODEL_KEY}_${CF_MODE}${SAFE_TAG}.log
mkdir -p logs "$OUTPUT_ROOT"

for r in "${ROWS[@]}"; do
    if [[ -f "$ROW_DIR/row=$r/index.json" ]]; then
        echo "[rerun] $MODEL_KEY $CF_MODE $SAFETY row $r already done, skipping" | tee -a "$LOG"
        continue
    fi
    echo "[rerun] $MODEL_KEY $CF_MODE $SAFETY row $r starting $(date '+%F %T')" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=${GPU:-1} python run_experiments_v2.py \
        --backend hf \
        --hf-model-id "$MODEL_ID" \
        --hf-dtype "$DTYPE" \
        --environment financial_article_writing \
        --adversarial-agent CHIEF_EDITOR \
        --cf-agent RESEARCHER \
        --cf-selection-rule last_pre_target_agent \
        --cf-mode "$CF_MODE" \
        --cf-samples 1 \
        --export-tape \
        --max-new-tokens 512 \
        "${SAFE_FLAG[@]}" \
        --output-root "$OUTPUT_ROOT" \
        --start-id "$r" --end-id "$r" 2>&1 | tee -a "$LOG"
    echo "[rerun] $MODEL_KEY $CF_MODE $SAFETY row $r finished $(date '+%F %T') exit=${PIPESTATUS[0]}" | tee -a "$LOG"
done
