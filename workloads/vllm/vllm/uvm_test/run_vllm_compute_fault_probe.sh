#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPT="${PROMPT:-Explain unified memory in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
VLLM_PY="${VLLM_PY:-${VLLM_WORKDIR}/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_compute_fault_probe.$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_ROOT}"

FAULT_TRACE="${RUN_ROOT}/uvm_fault_trace.log"
GEN_RESULT="${RUN_ROOT}/generate_result.json"
GEN_LOG="${RUN_ROOT}/generate.log"
SUMMARY="${RUN_ROOT}/fault_summary.txt"
SUMMARY_JSON="${RUN_ROOT}/fault_summary.json"
TRACE_MARKER="${TRACE_MARKER:-/sys/kernel/tracing/trace_marker}"
TRACE_MARKER_MODE=""

cleanup() {
  set +e
  if [[ -n "${TRACE_PGID:-}" ]]; then
    sudo kill -- "-${TRACE_PGID}" >/dev/null 2>&1 || true
    kill -- "-${TRACE_PGID}" >/dev/null 2>&1 || true
    wait "${TRACE_PID:-}" >/dev/null 2>&1 || true
  elif [[ -n "${TRACE_PID:-}" ]]; then
    kill "${TRACE_PID}" >/dev/null 2>&1 || true
    wait "${TRACE_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses ]]; then
    echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses >/dev/null 2>&1 || true
  fi
  if [[ -e /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination ]]; then
    echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination >/dev/null 2>&1 || true
  fi
  if [[ -e /sys/kernel/tracing/tracing_on ]]; then
    echo 0 | sudo tee /sys/kernel/tracing/tracing_on >/dev/null 2>&1 || true
  fi
  if [[ -n "${TRACE_MARKER_MODE}" && -e "${TRACE_MARKER}" ]]; then
    sudo chmod "${TRACE_MARKER_MODE}" "${TRACE_MARKER}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "run_root=${RUN_ROOT}"
echo "model=${MODEL}"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/nvidia_smi_before.txt"

echo "[1/3] start ftrace UVM fault capture"
sudo mount -t tracefs nodev /sys/kernel/tracing >/dev/null 2>&1 || true
if [[ -e "${TRACE_MARKER}" ]]; then
  TRACE_MARKER_MODE="$(stat -c '%a' "${TRACE_MARKER}" 2>/dev/null || true)"
  sudo chmod a+w "${TRACE_MARKER}" >/dev/null 2>&1 || true
fi
echo 0 | sudo tee /sys/kernel/tracing/tracing_on >/dev/null
: | sudo tee /sys/kernel/tracing/trace >/dev/null || true
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_destination >/dev/null
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_perf_fault_log_addresses >/dev/null
echo 1 | sudo tee /sys/kernel/tracing/tracing_on >/dev/null
setsid bash -c \
  'sudo cat /sys/kernel/tracing/trace_pipe \
    | grep --line-buffered -E "Replayable fault GPU|VLLM_UVM_COMPUTE_START"' \
  > "${FAULT_TRACE}" &
TRACE_PID=$!
TRACE_PGID="$(ps -o pgid= -p "${TRACE_PID}" | tr -d '[:space:]')"

echo "[2/3] run UVM alloc-only vLLM request"
(
  cd "${VLLM_WORKDIR}" || exit 1
  env \
    VLLM_USE_UVM=1 \
    VLLM_UVM_ADDRESS_LOG_ENABLE=0 \
    VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
    VLLM_UVM_MOE_STREAM_TRACE=0 \
    VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_ENABLE=0 \
    VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_ENABLE=0 \
    VLLM_UVM_COMPUTE_START_TRACE_MARKER_ENABLE=1 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
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
      --output-json "${GEN_RESULT}"
) 2>&1 | tee "${GEN_LOG}"

sleep 2
cleanup
trap - EXIT

grep -m1 "initialize_kv_cache begin allocator=" "${GEN_LOG}" \
  | tee "${RUN_ROOT}/kv_allocator.txt" >/dev/null 2>&1 || true

echo "[3/3] analyze compute-time faults"
python3 "${SCRIPT_DIR}/analyze_uvm_compute_faults.py" \
  --trace-log "${FAULT_TRACE}" \
  --summary-json "${SUMMARY_JSON}" \
  | tee "${SUMMARY}"

cat "${GEN_RESULT}"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/nvidia_smi_after.txt"

echo "fault_summary=${SUMMARY}"
