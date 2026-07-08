#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${VLLM_WORKDIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPT="${PROMPT:-Explain unified memory in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-0}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
USE_UVM="${USE_UVM:-0}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_moe_stream_trace.$(date -u +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-${VLLM_WORKDIR}/.venv/bin/python}"

TRACE_FILE="${RUN_ROOT}/moe_stream_trace.jsonl"
RESULT_JSON="${RUN_ROOT}/generate_result.json"
SUMMARY_TXT="${RUN_ROOT}/stream_summary.txt"
SERVER_LOG="${RUN_ROOT}/generate.log"

mkdir -p "${RUN_ROOT}"

echo "run_root=${RUN_ROOT}"
echo "model=${MODEL}"
echo "use_uvm=${USE_UVM}"
echo "trace_file=${TRACE_FILE}"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/nvidia_smi_before.txt"

(
  cd "${VLLM_WORKDIR}" || exit 1
  env \
    VLLM_USE_UVM="${USE_UVM}" \
    VLLM_UVM_MOE_STREAM_TRACE=1 \
    VLLM_UVM_MOE_STREAM_TRACE_FILE="${TRACE_FILE}" \
    VLLM_UVM_MOE_STREAM_TRACE_MARKER=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" "${SCRIPT_DIR}/vllm_generate_once.py" \
      --model "${MODEL}" \
      --prompt "${PROMPT}" \
      --max-tokens "${MAX_TOKENS}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
      --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
      --output-json "${RESULT_JSON}"
) 2>&1 | tee "${SERVER_LOG}"

python3 "${SCRIPT_DIR}/analyze_vllm_moe_stream_trace.py" "${TRACE_FILE}" \
  | tee "${SUMMARY_TXT}"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/nvidia_smi_after.txt"

echo "summary=${SUMMARY_TXT}"
