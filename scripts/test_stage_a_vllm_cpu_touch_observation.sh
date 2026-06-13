#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}/workloads/vllm"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"
ROUTE_LOG_PATH="${ROUTE_LOG_PATH:-/proc/driver/nvidia-uvm/phase_f_route_log}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

RUN_ROOT="${RUN_ROOT:-${VLLM_DIR}/log/stage_a_vllm_cpu_touch_observation_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-4}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-2}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.80}"
DEVICE="${DEVICE:-0}"

SUMMARY="${RUN_ROOT}/stage_a_cpu_touch_summary.txt"

COUNTER_KEYS=(
  stage_a_cpu_faults_total
  stage_a_cpu_faults_registered
  stage_a_cpu_faults_candidate
  stage_a_cpu_read_faults
  stage_a_cpu_write_faults
  stage_a_cpu_faults_weight
  stage_a_cpu_faults_kv
  stage_a_cpu_faults_scratch
  stage_a_cpu_faults_other
  stage_a_cpu_fault_fallbacks
)

mkdir -p "${RUN_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${path}. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi
}

write_param() {
  local name="$1"
  local value="$2"
  local path="${PARAM_ROOT}/${name}"

  require_file "${path}"
  if [[ ! -w "${path}" && "${EUID}" -ne 0 ]]; then
    echo "${value}" | sudo tee "${path}" >/dev/null
  else
    echo "${value}" > "${path}"
  fi
}

reset_counters() {
  require_file "${STATS_RESET_PATH}"
  if [[ ! -w "${STATS_RESET_PATH}" && "${EUID}" -ne 0 ]]; then
    echo 1 | sudo tee "${STATS_RESET_PATH}" >/dev/null
  else
    echo 1 > "${STATS_RESET_PATH}"
  fi
}

counter_value() {
  local file="$1"
  local key="$2"
  awk -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) print 0 }' "${file}"
}

emit_delta() {
  local before="$1"
  local after="$2"
  local key="$3"
  local b a

  b="$(counter_value "${before}" "${key}")"
  a="$(counter_value "${after}" "${key}")"
  printf '%s=%s\n' "${key}" "$((a - b))"
}

main() {
  require_file "${STATS_PATH}"
  require_file "${ROUTE_LOG_PATH}"
  require_file "${PARAM_ROOT}/uvm_stage_a_cpu_fault_observe_enable"
  require_file "${PARAM_ROOT}/uvm_stage_a_cpu_fault_log_enable"

  write_param uvm_stage_a_cpu_fault_observe_enable 1
  write_param uvm_stage_a_cpu_fault_log_enable 1
  reset_counters

  cat "${STATS_PATH}" > "${RUN_ROOT}/latency_stats_before.txt"
  cat "${ROUTE_LOG_PATH}" > "${RUN_ROOT}/phase_f_route_log_before.txt" 2>/dev/null || true

  (
    cd "${VLLM_DIR}"
    RUN_ROOT="${RUN_ROOT}/vllm_smoke" \
    RESET_COUNTERS=0 \
    PROMPTS="${PROMPTS}" \
    REQUEST_RATE="${REQUEST_RATE}" \
    OUTPUT_LEN="${OUTPUT_LEN}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS}" \
    SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION}" \
    VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
    UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
    UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
    UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
    DRIVER_POOL_STATE_SNAPSHOT_DEVICE="${DEVICE}" \
    ./run_stage_n_tag_only_three_pool_smoke.sh
  )

  cat "${STATS_PATH}" > "${RUN_ROOT}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${RUN_ROOT}/phase_f_route_log_after.txt" 2>/dev/null || true
  grep 'stage_a_cpu_fault' "${RUN_ROOT}/phase_f_route_log_after.txt" > "${RUN_ROOT}/stage_a_cpu_fault_events.log" || true

  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "output_len=${OUTPUT_LEN}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo
    echo "counter_deltas:"
    for key in "${COUNTER_KEYS[@]}"; do
      emit_delta "${RUN_ROOT}/latency_stats_before.txt" "${RUN_ROOT}/latency_stats_after.txt" "${key}"
    done
    echo
    echo "decision:"
    weight_delta="$(emit_delta "${RUN_ROOT}/latency_stats_before.txt" "${RUN_ROOT}/latency_stats_after.txt" stage_a_cpu_faults_weight | cut -d= -f2)"
    kv_delta="$(emit_delta "${RUN_ROOT}/latency_stats_before.txt" "${RUN_ROOT}/latency_stats_after.txt" stage_a_cpu_faults_kv | cut -d= -f2)"
    if [[ "${weight_delta}" == "0" && "${kv_delta}" == "0" ]]; then
      echo "stage_a_hypothesis=passes_for_weight_and_kv"
    else
      echo "stage_a_hypothesis=needs_investigation"
    fi
    echo "weight_cpu_faults=${weight_delta}"
    echo "kv_cpu_faults=${kv_delta}"
    echo
    echo "stage_a_event_log=${RUN_ROOT}/stage_a_cpu_fault_events.log"
    echo "vllm_run_root=${RUN_ROOT}/vllm_smoke"
  } | tee "${SUMMARY}"

  echo
  echo "Done."
  echo "Summary: ${SUMMARY}"
}

main "$@"
