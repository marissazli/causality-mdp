#!/usr/bin/env bash
# (B) no-adversary queue, ordered cheapest first so the long Qwen FAW runs are last:
#   MAD, TP, CG (both models), then Llama FAW, then Qwen FAW.
# If WAIT_FOR_A=1, first waits for the (A) benign runs to finish (Qwen FAW safe is
# the last one), then stops the old ablation_queue session so its own (B) loop
# never starts. Every step skips finished rows, so this is safe to restart.
set -uo pipefail
cd "$(dirname "$0")/.."

if [[ "${WAIT_FOR_A:-0}" == 1 ]]; then
    A_LAST=results/results_v3_qwen_benign/env=financial_article_writing/model=Qwen3-8B/adv=CHIEF_EDITOR/safe=True
    echo "[noadv-queue] waiting for (A) to finish: $A_LAST $(date '+%F %T')"
    until [[ $(ls "$A_LAST"/row=*/index.json 2>/dev/null | wc -l) -ge 66 ]]; do sleep 60; done
    # let (A) write its manifest, then stop the old queue before its (B) loop gets going
    sleep 30
    tmux kill-session -t ablation_queue 2>/dev/null && echo "[noadv-queue] stopped ablation_queue"
    sleep 15
fi

for e in mad tp cg; do
    for m in llama qwen; do
        KV_CACHE=1 bash scripts/run_noadv.sh $m $e
        KV_CACHE=1 bash scripts/run_noadv.sh $m $e safe
    done
done
for m in llama qwen; do
    KV_CACHE=1 bash scripts/run_noadv.sh $m faw
    KV_CACHE=1 bash scripts/run_noadv.sh $m faw safe
done
echo "[noadv-queue] all done $(date '+%F %T')"
