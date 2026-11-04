#!/usr/bin/env bash
# One-click RoboBeats evaluation on SSS-Bench: gate -> memory -> reader -> grading.
#
#   SSS_DATA_ROOT=/data/sss_videos LLM_MODEL=/models/Qwen3.8-27B GATE=2B bash sss_bench/scripts/run_robobeats.sh
#
# env:
#   SSS_DATA_ROOT  (required) source videos, layout in sss_bench/README.md / sss_bench/data/videos.csv
#   LLM_MODEL      (required unless LLM_BASE_URLS is set) Qwen3.8-27B weights; used as writer, reader and judge
#   GATE           0.8B | 2B | 4B                      (default 2B)
#   GATE_MODEL     HF id or local path                 (default superbigsaw/RoboBeats-$GATE)
#   GATE_GPUS      GPUs for the gate, one shard each   (default 0)
#   LLM_GPUS       GPUs for 27B vLLM, one server each  (default 1,2,3,4,5,6,7)
#   LLM_BASE_URLS  use existing OpenAI-compatible endpoints (comma-separated) instead of launching vLLM
#   READER_MODE    hard10 (default; one look of <= 10 frames) | memoryonly
#   WORK           output dir                          (default runs/robobeats_$GATE)
#   KEEP_SERVERS=1 leave the vLLM servers running at the end
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SSS_DATA_ROOT:?set SSS_DATA_ROOT to the video root}"
GATE=${GATE:-2B}; GATE_MODEL=${GATE_MODEL:-superbigsaw/RoboBeats-$GATE}
GATE_GPUS=${GATE_GPUS:-0}; LLM_GPUS=${LLM_GPUS:-1,2,3,4,5,6,7}
READER_MODE=${READER_MODE:-hard10}; WORK=${WORK:-runs/robobeats_$GATE}; SERVED=Qwen3.8-27B
PY=${PYTHON:-python}
mkdir -p "$WORK/logs"
export SSS_DATA_ROOT

echo "== 0/4 checking videos"
$PY sss_bench/scripts/check_videos.py "$SSS_DATA_ROOT"

PIDS=()
cleanup() { if [ "${KEEP_SERVERS:-0}" != 1 ] && [ ${#PIDS[@]} -gt 0 ]; then kill "${PIDS[@]}" 2>/dev/null || true; fi; }
trap cleanup EXIT

launch_llm() {
  : "${LLM_MODEL:?set LLM_MODEL (Qwen3.8-27B weights) or LLM_BASE_URLS}"
  local urls=() i=0
  for g in ${LLM_GPUS//,/ }; do
    local port=$((8100 + i))
    CUDA_VISIBLE_DEVICES=$g nohup vllm serve "$LLM_MODEL" --served-model-name $SERVED --host 127.0.0.1 --port $port \
      --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.90 --max-num-seqs 2 \
      --limit-mm-per-prompt '{"image":80,"video":0}' --mm-processor-kwargs '{"max_pixels":200704,"min_pixels":3136}' \
      > "$WORK/logs/vllm_$port.log" 2>&1 &
    PIDS+=($!); urls+=("http://127.0.0.1:$port/v1"); i=$((i + 1))
  done
  LLM_BASE_URLS=$(IFS=,; echo "${urls[*]}")
}

wait_llm() {
  for u in ${LLM_BASE_URLS//,/ }; do
    until curl -sf "${u%/v1}/health" >/dev/null 2>&1 || curl -sf "$u/models" >/dev/null 2>&1; do sleep 10; done
    echo "   ready: $u"
  done
}

run_gate() {
  echo "== 1/4 RoboBeats-$GATE gate on GPUs $GATE_GPUS"
  local gpus=(${GATE_GPUS//,/ }) n=${#gpus[@]} gp=()
  for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} $PY -m sss_bench.reproduce.pipeline gate --work "$WORK" --gate-model "$GATE_MODEL" \
      --shard $i --nshard $n > "$WORK/logs/gate_$i.log" 2>&1 &
    gp+=($!)
  done
  for p in "${gp[@]}"; do wait $p || { echo "gate shard failed, see $WORK/logs/gate_*.log"; exit 1; }; done
}

overlap=0
if [ -z "${LLM_BASE_URLS:-}" ]; then
  for g in ${GATE_GPUS//,/ }; do [[ ",$LLM_GPUS," == *",$g,"* ]] && overlap=1; done
fi
if [ -n "${LLM_BASE_URLS:-}" ]; then
  run_gate
elif [ $overlap = 1 ]; then          # shared GPUs: gate first, then the 27B servers
  run_gate; launch_llm
else                                 # disjoint GPUs: start the 27B servers while the gate runs
  launch_llm; run_gate
fi
echo "== 2/4 waiting for 27B endpoints"; wait_llm

echo "== 3/4 memory writing + reader ($READER_MODE)"
$PY -m sss_bench.reproduce.pipeline answer --work "$WORK" --llm-base-urls "$LLM_BASE_URLS" --llm-model $SERVED --reader-mode $READER_MODE

echo "== 4/4 grading"
$PY -m sss_bench.evaluate --pred "$WORK/predictions_$READER_MODE.jsonl" --data-root "$SSS_DATA_ROOT" \
  --judge-base-url "$LLM_BASE_URLS" --judge-model $SERVED --out "$WORK/results_$READER_MODE"
echo "done: $WORK/results_$READER_MODE/summary.json"
