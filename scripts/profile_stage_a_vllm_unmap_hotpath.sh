#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}/workloads/vllm"

RUN_ROOT="${RUN_ROOT:-${VLLM_DIR}/log/stage_a_vllm_unmap_hotpath_$(date -u +%Y%m%d_%H%M%S)}"
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${ROOT_DIR}/${RUN_ROOT}"
fi

PROMPTS="${PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-4}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.90}"
DEVICE="${DEVICE:-0}"

PERF_BIN="${PERF_BIN:-perf}"
PERF_ENABLED="${PERF_ENABLED:-1}"
PERF_FREQ="${PERF_FREQ:-99}"
PERF_CALLGRAPH="${PERF_CALLGRAPH:-fp}"
PERF_SYSTEM_WIDE="${PERF_SYSTEM_WIDE:-1}"
PERF_USE_SUDO="${PERF_USE_SUDO:-auto}"
PERF_EXTRA_ARGS="${PERF_EXTRA_ARGS:-}"

STAGE_A_SCRIPT="${ROOT_DIR}/scripts/test_stage_a_vllm_cpu_touch_observation.sh"
SUMMARY="${RUN_ROOT}/unmap_hotpath_summary.txt"
PERF_DATA="${RUN_ROOT}/perf.data"
PERF_REPORT="${RUN_ROOT}/perf_report_stdio.txt"
PERF_SCRIPT="${RUN_ROOT}/perf_script.txt"
PERF_FOCUS="${RUN_ROOT}/perf_unmap_focus.txt"

mkdir -p "${RUN_ROOT}"

perf_cmd_prefix=()
stage_env=()

append_env_if_set() {
  local name="$1"

  if [[ "${!name+x}" == "x" ]]; then
    stage_env+=("${name}=${!name}")
  fi
}

append_passthrough_env() {
  local name
  local passthrough_names=(
    MODEL
    STARTUP_TIMEOUT
    PORT
    RESET_COUNTERS
    MODE
    WITH_ADDRESS_LOG
    UVM_PHASE_F_ROUTE_LOG_DELAY_MS
    KV_BUDGET_BYTES
    WEIGHT_BUDGET_BYTES
    SCRATCH_TEMP_BUDGET_BYTES
    DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY
    DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_EVERY
    DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_MAX
    REQUIRE_PHASE_N_AUTO_HOOK
    REQUIRE_PHASE_N_AUTO_RECLAIM
    REQUIRE_STAGE_N_HARD_PARTITION
    HARD_PARTITION_TOLERANCE_BYTES
    DRIVER_POOL_BUDGET_MODE
    UVM_PHASE_N_BLOCK_QUOTA_ENABLE
    UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY
    UVM_PHASE_N_NATIVE_BLOCK_ENABLE
    UVM_PHASE_N_AUTO_RECLAIM_ENABLE
    UVM_PHASE_N_AUTO_RECLAIM_EXECUTE
    UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY
    UVM_PHASE_N_AUTO_RECLAIM_EVERY
    UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY
    UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS
    UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS
    UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS
    VLLM_UVM_DRIVER_KV_TAG_ENABLE
    VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE
    VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP
  )

  for name in "${passthrough_names[@]}"; do
    append_env_if_set "${name}"
  done
}

setup_perf_prefix() {
  perf_cmd_prefix=()

  if [[ "${PERF_ENABLED}" != "1" ]]; then
    return
  fi

  if ! command -v "${PERF_BIN}" >/dev/null 2>&1; then
    echo "WARN: perf not found: ${PERF_BIN}; disabling perf" | tee -a "${RUN_ROOT}/warnings.log"
    PERF_ENABLED=0
    return
  fi

  if [[ "${PERF_USE_SUDO}" == "1" ||
        ( "${PERF_USE_SUDO}" == "auto" && "${EUID}" -ne 0 && -r /proc/sys/kernel/perf_event_paranoid &&
          "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 1 ) ]]; then
    if command -v sudo >/dev/null 2>&1; then
      perf_cmd_prefix=(sudo)
    fi
  fi
}

run_with_perf() {
  local stage_run_root="${RUN_ROOT}/stage_a"
  local stdout_path="${RUN_ROOT}/stage_a_stdout.log"
  local stderr_path="${RUN_ROOT}/stage_a_stderr.log"
  local exit_code=0
  local perf_args=()

  mkdir -p "${stage_run_root}"

  stage_env=(
    env
    "RUN_ROOT=${stage_run_root}"
    "PROMPTS=${PROMPTS}"
    "REQUEST_RATE=${REQUEST_RATE}"
    "OUTPUT_LEN=${OUTPUT_LEN}"
    "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
    "SERVER_MAX_NUM_SEQS=${SERVER_MAX_NUM_SEQS}"
    "SERVER_GPU_MEMORY_UTILIZATION=${SERVER_GPU_MEMORY_UTILIZATION}"
    "DEVICE=${DEVICE}"
  )
  append_passthrough_env

  if [[ "${PERF_ENABLED}" == "1" ]]; then
    perf_args=(record)
    if [[ "${PERF_SYSTEM_WIDE}" == "1" ]]; then
      perf_args+=(-a)
    fi
    perf_args+=(-F "${PERF_FREQ}" -g "--call-graph=${PERF_CALLGRAPH}" -o "${PERF_DATA}")

    set +e
    "${perf_cmd_prefix[@]}" "${PERF_BIN}" "${perf_args[@]}" ${PERF_EXTRA_ARGS} -- "${stage_env[@]}" "${STAGE_A_SCRIPT}" \
      > "${stdout_path}" \
      2> "${stderr_path}"
    exit_code=$?
    set -e
  else
    set +e
    "${stage_env[@]}" "${STAGE_A_SCRIPT}" \
      > "${stdout_path}" \
      2> "${stderr_path}"
    exit_code=$?
    set -e
  fi

  echo "${exit_code}" > "${RUN_ROOT}/stage_a_exit_code.txt"
}

emit_perf_reports() {
  if [[ "${PERF_ENABLED}" != "1" || ! -f "${PERF_DATA}" ]]; then
    : > "${PERF_REPORT}"
    : > "${PERF_SCRIPT}"
    : > "${PERF_FOCUS}"
    return
  fi

  "${perf_cmd_prefix[@]}" "${PERF_BIN}" report -i "${PERF_DATA}" --stdio --no-children \
    > "${PERF_REPORT}" 2> "${RUN_ROOT}/perf_report.stderr" || true

  "${perf_cmd_prefix[@]}" "${PERF_BIN}" script -i "${PERF_DATA}" \
    > "${PERF_SCRIPT}" 2> "${RUN_ROOT}/perf_script.stderr" || true

  grep -E "block_unmap_cpu|uvm_va_block_unmap|uvm_va_block_unmap_mask|unmap_mapping_range|zap_page_range|tlb_flush|flush_tlb|native_flush_tlb|on_each_cpu|smp_call_function" \
    "${PERF_REPORT}" "${PERF_SCRIPT}" \
    > "${PERF_FOCUS}" || true
}

summary_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) print "NA" }' "${file}"
}

count_pattern() {
  local file="$1"
  local pattern="$2"
  if [[ ! -f "${file}" ]]; then
    echo 0
    return
  fi
  grep -E -c "${pattern}" "${file}" || true
}

write_summary() {
  local stage_summary="${RUN_ROOT}/stage_a/stage_a_cpu_touch_summary.txt"
  local exit_code
  local weight_faults="NA"
  local kv_faults="NA"
  local candidate_faults="NA"
  local total_faults="NA"
  local hypothesis="missing_summary"
  local report_unmap_count=0
  local script_unmap_count=0
  local focus_lines=0
  local go_no_go

  exit_code="$(cat "${RUN_ROOT}/stage_a_exit_code.txt" 2>/dev/null || echo 127)"

  if [[ -f "${stage_summary}" ]]; then
    hypothesis="$(summary_value "${stage_summary}" stage_a_hypothesis)"
    weight_faults="$(summary_value "${stage_summary}" weight_cpu_faults)"
    kv_faults="$(summary_value "${stage_summary}" kv_cpu_faults)"
    candidate_faults="$(summary_value "${stage_summary}" stage_a_cpu_faults_candidate)"
    total_faults="$(summary_value "${stage_summary}" stage_a_cpu_faults_total)"
  fi

  report_unmap_count="$(count_pattern "${PERF_REPORT}" "block_unmap_cpu|unmap_mapping_range")"
  script_unmap_count="$(count_pattern "${PERF_SCRIPT}" "block_unmap_cpu|unmap_mapping_range")"
  focus_lines="$(awk 'END { print NR + 0 }' "${PERF_FOCUS}" 2>/dev/null || echo 0)"

  if [[ "${exit_code}" == "0" &&
        "${weight_faults}" == "0" &&
        "${kv_faults}" == "0" &&
        "${candidate_faults}" == "0" &&
        "${focus_lines}" == "0" ]]; then
    go_no_go="native_uvm_sufficient_for_this_window"
  elif [[ "${exit_code}" == "0" &&
          "${weight_faults}" == "0" &&
          "${kv_faults}" == "0" &&
          "${focus_lines}" != "0" ]]; then
    go_no_go="investigate_non_cpu_fault_cpu_unmap_path"
  elif [[ "${exit_code}" == "0" ]]; then
    go_no_go="investigate_cpu_touch_or_registration_gap"
  else
    go_no_go="run_failed"
  fi

  {
    echo "Stage A vLLM CPU-touch + CPU-unmap hotpath profile"
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "config:"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "output_len=${OUTPUT_LEN}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo "device=${DEVICE}"
    echo "perf_enabled=${PERF_ENABLED}"
    echo "perf_freq=${PERF_FREQ}"
    echo "perf_callgraph=${PERF_CALLGRAPH}"
    echo "perf_system_wide=${PERF_SYSTEM_WIDE}"
    echo
    echo "stage_a:"
    echo "exit_code=${exit_code}"
    echo "stage_a_hypothesis=${hypothesis}"
    echo "weight_cpu_faults=${weight_faults}"
    echo "kv_cpu_faults=${kv_faults}"
    echo "stage_a_cpu_faults_candidate=${candidate_faults}"
    echo "stage_a_cpu_faults_total=${total_faults}"
    echo
    echo "perf_unmap_hotpath:"
    echo "perf_report_unmap_symbol_lines=${report_unmap_count}"
    echo "perf_script_unmap_symbol_lines=${script_unmap_count}"
    echo "perf_focus_lines=${focus_lines}"
    echo "perf_focus=${PERF_FOCUS}"
    echo "perf_report=${PERF_REPORT}"
    echo "perf_script=${PERF_SCRIPT}"
    echo "perf_data=${PERF_DATA}"
    echo
    echo "decision:"
    echo "go_no_go=${go_no_go}"
    case "${go_no_go}" in
      native_uvm_sufficient_for_this_window)
        echo "interpretation=No weight/KV CPU fault and no sampled CPU unmap hotpath in this observation window."
        ;;
      investigate_non_cpu_fault_cpu_unmap_path)
        echo "interpretation=Weight/KV CPU faults are absent, but CPU unmap symbols appear in perf; inspect whether migration/eviction/prefetch-to-CPU created CPU mappings."
        ;;
      investigate_cpu_touch_or_registration_gap)
        echo "interpretation=Stage A counters are non-zero or ambiguous; inspect registered region and CPU fault event logs first."
        ;;
      *)
        echo "interpretation=The workload/profiler run failed; inspect stdout/stderr."
        ;;
    esac
    echo
    echo "stage_a_summary=${stage_summary}"
    echo "stage_a_stdout=${RUN_ROOT}/stage_a_stdout.log"
    echo "stage_a_stderr=${RUN_ROOT}/stage_a_stderr.log"
  } | tee "${SUMMARY}"

  if [[ "${go_no_go}" == "run_failed" ]]; then
    return 1
  fi
}

main() {
  setup_perf_prefix
  run_with_perf
  emit_perf_reports
  write_summary

  echo
  echo "Done."
  echo "Summary: ${SUMMARY}"
}

main "$@"
