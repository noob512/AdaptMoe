#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_WORKDIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPT="${PROMPT:-Explain unified memory in one sentence.}"
MAX_TOKENS="${MAX_TOKENS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-0}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
VLLM_PY="${VLLM_PY:-${VLLM_WORKDIR}/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_uvm_fault_expert_access.$(date -u +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_ROOT}"

ROUTING_TRACE="${RUN_ROOT}/routing_trace.jsonl"
PLAN_FILE="${RUN_ROOT}/offload_plan.json"
STREAM_TRACE="${RUN_ROOT}/moe_stream_trace.jsonl"
WEIGHT_MAP="${RUN_ROOT}/weight_map.jsonl"
FAULT_TRACE="${RUN_ROOT}/uvm_fault_trace.log"
GEN_RESULT="${RUN_ROOT}/generate_result.json"
GEN_LOG="${RUN_ROOT}/generate.log"
FAULT_SUMMARY="${RUN_ROOT}/fault_summary.txt"
MANAGED_CHECK="${RUN_ROOT}/managed_check.txt"
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

echo "[1/4] routing warmup to choose active experts"
(
  cd "${VLLM_WORKDIR}" || exit 1
  env \
    VLLM_USE_UVM=0 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${ROUTING_TRACE}" \
    VLLM_UVM_MOE_STREAM_TRACE=0 \
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
      --output-json "${RUN_ROOT}/routing_generate_result.json"
) 2>&1 | tee "${RUN_ROOT}/routing_generate.log"

python3 "${SCRIPT_DIR}/build_offload_plan_from_routing_trace.py" \
  --routing-trace "${ROUTING_TRACE}" \
  --output "${PLAN_FILE}" \
  --max-layers "${OFFLOAD_PLAN_MAX_LAYERS:-12}" \
  --max-experts-per-layer "${OFFLOAD_PLAN_MAX_EXPERTS_PER_LAYER:-1}"

echo "[2/4] start ftrace UVM fault capture"
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
    | grep --line-buffered -E "Replayable fault GPU|vllm_moe_stream"' \
  > "${FAULT_TRACE}" &
TRACE_PID=$!
TRACE_PGID="$(ps -o pgid= -p "${TRACE_PID}" | tr -d '[:space:]')"

echo "[3/4] run managed UVM offload/access probe"
(
  cd "${VLLM_WORKDIR}" || exit 1
  env \
    VLLM_USE_UVM=1 \
    VLLM_UVM_MOE_STREAM_TRACE=1 \
    VLLM_UVM_MOE_STREAM_TRACE_FILE="${STREAM_TRACE}" \
    VLLM_UVM_MOE_STREAM_TRACE_MARKER=1 \
    VLLM_UVM_WEIGHT_MAP_ENABLE=1 \
    VLLM_UVM_WEIGHT_MAP_FILE="${WEIGHT_MAP}" \
    VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
    VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=offload_cpu \
    VLLM_UVM_INIT_WEIGHT_SYNC=1 \
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN_FILE}" \
    VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down \
    VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE=1 \
    VLLM_UVM_WEIGHT_EXECUTOR=driver_batch \
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

echo "[4/4] analyze fault access"
python3 - "${WEIGHT_MAP}" > "${MANAGED_CHECK}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

path = Path(sys.argv[1])
reasons = Counter()
slices = 0
with path.open(encoding="utf-8", errors="replace") as f:
    for line in f:
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("is_moe_expert") and rec.get("is_logical_expert_slice"):
            slices += 1
            reasons[str(rec.get("driver_capability_reason") or "unknown")] += 1
print(f"moe_expert_slices={slices}")
for reason, count in reasons.most_common():
    print(f"{reason}={count}")
PY
cat "${MANAGED_CHECK}"

python3 "${SCRIPT_DIR}/analyze_uvm_fault_expert_access.py" \
  --fault-trace "${FAULT_TRACE}" \
  --weight-map "${WEIGHT_MAP}" \
  --late-gap "${LATE_GAP:-2}" \
  | tee "${FAULT_SUMMARY}"

nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv,noheader,nounits | tee "${RUN_ROOT}/nvidia_smi_after.txt"

echo "fault_summary=${FAULT_SUMMARY}"
