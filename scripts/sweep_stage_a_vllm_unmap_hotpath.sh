#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}/workloads/vllm"

RUN_ROOT="${RUN_ROOT:-${VLLM_DIR}/log/stage_a_vllm_unmap_hotpath_sweep_$(date -u +%Y%m%d_%H%M%S)}"
if [[ "${RUN_ROOT}" != /* ]]; then
  RUN_ROOT="${ROOT_DIR}/${RUN_ROOT}"
fi

SCENARIOS="${SCENARIOS:-steady pressure_reclaim}"
OUTPUT_LEN_VALUES="${OUTPUT_LEN_VALUES:-32 128 512}"
MAX_MODEL_LEN_VALUES="${MAX_MODEL_LEN_VALUES:-2048 4096}"
PROMPTS="${PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-2}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-4}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.90}"
DEVICE="${DEVICE:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

PRESSURE_PROMPTS="${PRESSURE_PROMPTS:-${PROMPTS}}"
PRESSURE_REQUEST_RATE="${PRESSURE_REQUEST_RATE:-${REQUEST_RATE}}"
PRESSURE_SERVER_MAX_NUM_SEQS="${PRESSURE_SERVER_MAX_NUM_SEQS:-${SERVER_MAX_NUM_SEQS}}"
PRESSURE_SERVER_GPU_MEMORY_UTILIZATION="${PRESSURE_SERVER_GPU_MEMORY_UTILIZATION:-0.97}"
PRESSURE_KV_BUDGET_BYTES="${PRESSURE_KV_BUDGET_BYTES:-1073741824}"
PRESSURE_WEIGHT_BUDGET_BYTES="${PRESSURE_WEIGHT_BUDGET_BYTES:-21474836480}"
PRESSURE_SCRATCH_TEMP_BUDGET_BYTES="${PRESSURE_SCRATCH_TEMP_BUDGET_BYTES:-16777216}"
PRESSURE_AUTO_RECLAIM_EVERY="${PRESSURE_AUTO_RECLAIM_EVERY:-1}"
PRESSURE_AUTO_RECLAIM_SCAN_EVERY="${PRESSURE_AUTO_RECLAIM_SCAN_EVERY:-0}"
PRESSURE_AUTO_RECLAIM_MAX_VICTIMS="${PRESSURE_AUTO_RECLAIM_MAX_VICTIMS:-128}"
PRESSURE_AUTO_RECLAIM_COOLDOWN_EPOCHS="${PRESSURE_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"
PRESSURE_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${PRESSURE_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-0}"

PERF_ENABLED="${PERF_ENABLED:-1}"
PERF_FREQ="${PERF_FREQ:-99}"
PERF_CALLGRAPH="${PERF_CALLGRAPH:-fp}"
PERF_SYSTEM_WIDE="${PERF_SYSTEM_WIDE:-1}"
PERF_USE_SUDO="${PERF_USE_SUDO:-auto}"
PERF_EXTRA_ARGS="${PERF_EXTRA_ARGS:-}"

SUMMARY_CSV="${RUN_ROOT}/sweep_summary.csv"
VERDICT="${RUN_ROOT}/sweep_verdict.txt"
LOG="${RUN_ROOT}/sweep.log"

mkdir -p "${RUN_ROOT}"

summary_value() {
  local file="$1"
  local key="$2"

  awk -F= -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) print "NA" }' "${file}"
}

append_case_summary() {
  local scenario="$1"
  local output_len="$2"
  local max_model_len="$3"
  local case_root="$4"
  local exit_code="$5"
  local summary="${case_root}/unmap_hotpath_summary.txt"
  local go_no_go="missing_summary"
  local stage_a_hypothesis="missing_summary"
  local weight_faults="NA"
  local kv_faults="NA"
  local candidate_faults="NA"
  local total_faults="NA"
  local focus_lines="NA"
  local report_unmap_lines="NA"
  local script_unmap_lines="NA"

  if [[ -f "${summary}" ]]; then
    go_no_go="$(summary_value "${summary}" go_no_go)"
    stage_a_hypothesis="$(summary_value "${summary}" stage_a_hypothesis)"
    weight_faults="$(summary_value "${summary}" weight_cpu_faults)"
    kv_faults="$(summary_value "${summary}" kv_cpu_faults)"
    candidate_faults="$(summary_value "${summary}" stage_a_cpu_faults_candidate)"
    total_faults="$(summary_value "${summary}" stage_a_cpu_faults_total)"
    focus_lines="$(summary_value "${summary}" perf_focus_lines)"
    report_unmap_lines="$(summary_value "${summary}" perf_report_unmap_symbol_lines)"
    script_unmap_lines="$(summary_value "${summary}" perf_script_unmap_symbol_lines)"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${scenario}" \
    "${output_len}" \
    "${max_model_len}" \
    "${exit_code}" \
    "${go_no_go}" \
    "${stage_a_hypothesis}" \
    "${weight_faults}" \
    "${kv_faults}" \
    "${candidate_faults}" \
    "${total_faults}" \
    "${focus_lines}" \
    "${report_unmap_lines}" \
    "${script_unmap_lines}" \
    >> "${SUMMARY_CSV}"
}

run_case() {
  local scenario="$1"
  local output_len="$2"
  local max_model_len="$3"
  local case_name="${scenario}_o${output_len}_m${max_model_len}"
  local case_root="${RUN_ROOT}/${case_name}"
  local exit_code=0
  local case_prompts="${PROMPTS}"
  local case_request_rate="${REQUEST_RATE}"
  local case_max_num_seqs="${SERVER_MAX_NUM_SEQS}"
  local case_gpu_memory_utilization="${SERVER_GPU_MEMORY_UTILIZATION}"
  local -a case_env=()

  mkdir -p "${case_root}"

  case_env=(
    env
    "RUN_ROOT=${case_root}"
    "PROMPTS=${case_prompts}"
    "REQUEST_RATE=${case_request_rate}"
    "OUTPUT_LEN=${output_len}"
    "MAX_MODEL_LEN=${max_model_len}"
    "SERVER_MAX_NUM_SEQS=${case_max_num_seqs}"
    "SERVER_GPU_MEMORY_UTILIZATION=${case_gpu_memory_utilization}"
    "DEVICE=${DEVICE}"
    "PERF_ENABLED=${PERF_ENABLED}"
    "PERF_FREQ=${PERF_FREQ}"
    "PERF_CALLGRAPH=${PERF_CALLGRAPH}"
    "PERF_SYSTEM_WIDE=${PERF_SYSTEM_WIDE}"
    "PERF_USE_SUDO=${PERF_USE_SUDO}"
    "PERF_EXTRA_ARGS=${PERF_EXTRA_ARGS}"
  )

  if [[ "${scenario}" == "pressure_reclaim" ]]; then
    case_prompts="${PRESSURE_PROMPTS}"
    case_request_rate="${PRESSURE_REQUEST_RATE}"
    case_max_num_seqs="${PRESSURE_SERVER_MAX_NUM_SEQS}"
    case_gpu_memory_utilization="${PRESSURE_SERVER_GPU_MEMORY_UTILIZATION}"
    case_env=(
      env
      "RUN_ROOT=${case_root}"
      "PROMPTS=${case_prompts}"
      "REQUEST_RATE=${case_request_rate}"
      "OUTPUT_LEN=${output_len}"
      "MAX_MODEL_LEN=${max_model_len}"
      "SERVER_MAX_NUM_SEQS=${case_max_num_seqs}"
      "SERVER_GPU_MEMORY_UTILIZATION=${case_gpu_memory_utilization}"
      "DEVICE=${DEVICE}"
      "PERF_ENABLED=${PERF_ENABLED}"
      "PERF_FREQ=${PERF_FREQ}"
      "PERF_CALLGRAPH=${PERF_CALLGRAPH}"
      "PERF_SYSTEM_WIDE=${PERF_SYSTEM_WIDE}"
      "PERF_USE_SUDO=${PERF_USE_SUDO}"
      "PERF_EXTRA_ARGS=${PERF_EXTRA_ARGS}"
      "REQUIRE_PHASE_N_AUTO_HOOK=1"
      "REQUIRE_PHASE_N_AUTO_RECLAIM=1"
      "DRIVER_POOL_BUDGET_MODE=soft"
      "KV_BUDGET_BYTES=${PRESSURE_KV_BUDGET_BYTES}"
      "WEIGHT_BUDGET_BYTES=${PRESSURE_WEIGHT_BUDGET_BYTES}"
      "SCRATCH_TEMP_BUDGET_BYTES=${PRESSURE_SCRATCH_TEMP_BUDGET_BYTES}"
      "UVM_PHASE_N_AUTO_RECLAIM_ENABLE=1"
      "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE=1"
      "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY=1"
      "UVM_PHASE_N_AUTO_RECLAIM_EVERY=${PRESSURE_AUTO_RECLAIM_EVERY}"
      "UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY=${PRESSURE_AUTO_RECLAIM_SCAN_EVERY}"
      "UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS=${PRESSURE_AUTO_RECLAIM_MAX_VICTIMS}"
      "UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS=${PRESSURE_AUTO_RECLAIM_COOLDOWN_EPOCHS}"
      "UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS=${PRESSURE_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS}"
    )
  elif [[ "${scenario}" != "steady" ]]; then
    echo "ERROR: unknown scenario '${scenario}'. Supported: steady pressure_reclaim" >&2
    return 2
  fi

  echo "Running ${case_name}" | tee -a "${LOG}"
  {
    echo "scenario=${scenario}"
    echo "output_len=${output_len}"
    echo "max_model_len=${max_model_len}"
    echo "prompts=${case_prompts}"
    echo "request_rate=${case_request_rate}"
    echo "server_max_num_seqs=${case_max_num_seqs}"
    echo "server_gpu_memory_utilization=${case_gpu_memory_utilization}"
  } > "${case_root}/case_config.txt"

  set +e
  "${case_env[@]}" "${ROOT_DIR}/scripts/profile_stage_a_vllm_unmap_hotpath.sh" \
    > "${case_root}.stdout.log" \
    2> "${case_root}.stderr.log"
  exit_code=$?
  set -e

  append_case_summary "${scenario}" "${output_len}" "${max_model_len}" "${case_root}" "${exit_code}"
  return "${exit_code}"
}

write_verdict() {
  local failures=0
  local cases=0
  local native_ok=0
  local non_cpu_fault_unmap=0
  local cpu_touch_or_gap=0
  local missing=0

  {
    echo "Stage A vLLM CPU-touch + CPU-unmap hotpath sweep verdict"
    echo "run_root=${RUN_ROOT}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "config:"
    echo "scenarios=${SCENARIOS}"
    echo "output_len_values=${OUTPUT_LEN_VALUES}"
    echo "max_model_len_values=${MAX_MODEL_LEN_VALUES}"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo "pressure_prompts=${PRESSURE_PROMPTS}"
    echo "pressure_request_rate=${PRESSURE_REQUEST_RATE}"
    echo "pressure_server_max_num_seqs=${PRESSURE_SERVER_MAX_NUM_SEQS}"
    echo "pressure_server_gpu_memory_utilization=${PRESSURE_SERVER_GPU_MEMORY_UTILIZATION}"
    echo "pressure_kv_budget_bytes=${PRESSURE_KV_BUDGET_BYTES}"
    echo "device=${DEVICE}"
    echo "perf_enabled=${PERF_ENABLED}"
    echo "perf_freq=${PERF_FREQ}"
    echo "perf_callgraph=${PERF_CALLGRAPH}"
    echo "perf_system_wide=${PERF_SYSTEM_WIDE}"
    echo
    echo "case_results:"
  } > "${VERDICT}"

  while IFS=, read -r scenario output_len max_model_len exit_code go_no_go hypothesis weight_faults kv_faults candidate_faults total_faults focus_lines report_unmap_lines script_unmap_lines; do
    if [[ "${scenario}" == "scenario" ]]; then
      continue
    fi

    cases=$((cases + 1))

    if [[ "${exit_code}" != "0" ]]; then
      echo "FAIL ${scenario}_o${output_len}_m${max_model_len}: exit=${exit_code} go_no_go=${go_no_go}" >> "${VERDICT}"
      failures=$((failures + 1))
      continue
    fi

    case "${go_no_go}" in
      native_uvm_sufficient_for_this_window)
        echo "PASS_NATIVE ${scenario}_o${output_len}_m${max_model_len}: weight=${weight_faults} kv=${kv_faults} candidate=${candidate_faults} perf_focus=${focus_lines}" >> "${VERDICT}"
        native_ok=$((native_ok + 1))
        ;;
      investigate_non_cpu_fault_cpu_unmap_path)
        echo "KEEP_RESEARCH ${scenario}_o${output_len}_m${max_model_len}: weight=${weight_faults} kv=${kv_faults} candidate=${candidate_faults} perf_focus=${focus_lines} report_unmap=${report_unmap_lines} script_unmap=${script_unmap_lines}" >> "${VERDICT}"
        non_cpu_fault_unmap=$((non_cpu_fault_unmap + 1))
        ;;
      investigate_cpu_touch_or_registration_gap)
        echo "INSPECT_CPU_TOUCH ${scenario}_o${output_len}_m${max_model_len}: hypothesis=${hypothesis} weight=${weight_faults} kv=${kv_faults} candidate=${candidate_faults} total=${total_faults}" >> "${VERDICT}"
        cpu_touch_or_gap=$((cpu_touch_or_gap + 1))
        ;;
      *)
        echo "MISSING_OR_AMBIGUOUS ${scenario}_o${output_len}_m${max_model_len}: go_no_go=${go_no_go} weight=${weight_faults} kv=${kv_faults} perf_focus=${focus_lines}" >> "${VERDICT}"
        missing=$((missing + 1))
        ;;
    esac
  done < "${SUMMARY_CSV}"

  {
    echo
    echo "cases=${cases}"
    echo "native_uvm_sufficient_cases=${native_ok}"
    echo "non_cpu_fault_cpu_unmap_cases=${non_cpu_fault_unmap}"
    echo "cpu_touch_or_registration_gap_cases=${cpu_touch_or_gap}"
    echo "missing_or_ambiguous_cases=${missing}"
    echo "failures=${failures}"
    if [[ "${failures}" -ne 0 ]]; then
      echo "overall_verdict=rerun_or_fix_failed_cases"
      echo "next_step=inspect per-case stdout/stderr and unmap_hotpath_summary.txt"
    elif [[ "${non_cpu_fault_unmap}" -ne 0 ]]; then
      echo "overall_verdict=continue_cpu_no_user_mapping_research"
      echo "next_step=inspect perf_unmap_focus.txt and correlate CPU unmap stacks with eviction/prefetch/reclaim paths"
    elif [[ "${cpu_touch_or_gap}" -ne 0 || "${missing}" -ne 0 ]]; then
      echo "overall_verdict=inspect_cpu_touch_or_registration_before_policy_work"
      echo "next_step=inspect stage_a_cpu_fault_events.log and region registration logs"
    else
      echo "overall_verdict=native_uvm_sufficient_for_covered_windows"
      echo "next_step=do not treat GPU-only as primary optimization for these covered vLLM windows"
    fi
  } >> "${VERDICT}"

  cat "${VERDICT}"

  if [[ "${failures}" -ne 0 ]]; then
    return 1
  fi
}

main() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "scenarios=${SCENARIOS}"
    echo "output_len_values=${OUTPUT_LEN_VALUES}"
    echo "max_model_len_values=${MAX_MODEL_LEN_VALUES}"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo "pressure_server_gpu_memory_utilization=${PRESSURE_SERVER_GPU_MEMORY_UTILIZATION}"
    echo "pressure_kv_budget_bytes=${PRESSURE_KV_BUDGET_BYTES}"
    echo "device=${DEVICE}"
    echo "perf_enabled=${PERF_ENABLED}"
    echo "perf_freq=${PERF_FREQ}"
    echo "stop_on_failure=${STOP_ON_FAILURE}"
  } | tee "${LOG}"

  printf 'scenario,output_len,max_model_len,exit_code,go_no_go,stage_a_hypothesis,weight_cpu_faults,kv_cpu_faults,stage_a_cpu_faults_candidate,stage_a_cpu_faults_total,perf_focus_lines,perf_report_unmap_symbol_lines,perf_script_unmap_symbol_lines\n' > "${SUMMARY_CSV}"

  for scenario in ${SCENARIOS}; do
    for output_len in ${OUTPUT_LEN_VALUES}; do
      for max_model_len in ${MAX_MODEL_LEN_VALUES}; do
        set +e
        run_case "${scenario}" "${output_len}" "${max_model_len}"
        exit_code=$?
        set -e

        if [[ "${exit_code}" -ne 0 && "${STOP_ON_FAILURE}" == "1" ]]; then
          echo "Stopping after ${scenario}_o${output_len}_m${max_model_len} failure." | tee -a "${LOG}"
          break 3
        fi
      done
    done
  done

  echo "Summary: ${SUMMARY_CSV}" | tee -a "${LOG}"
  column -s, -t "${SUMMARY_CSV}" || cat "${SUMMARY_CSV}"
  echo
  write_verdict
  echo "Verdict: ${VERDICT}"
}

main "$@"
