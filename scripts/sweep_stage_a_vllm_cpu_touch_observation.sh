#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_DIR="${ROOT_DIR}/workloads/vllm"

RUN_ROOT="${RUN_ROOT:-${VLLM_DIR}/log/stage_a_vllm_cpu_touch_sweep_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_LEN_VALUES="${OUTPUT_LEN_VALUES:-32 128 512}"
MAX_MODEL_LEN_VALUES="${MAX_MODEL_LEN_VALUES:-2048 4096}"
PROMPTS="${PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-2}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-4}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.90}"
DEVICE="${DEVICE:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

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
  local case_name="$1"
  local output_len="$2"
  local max_model_len="$3"
  local case_root="$4"
  local exit_code="$5"
  local summary="${case_root}/stage_a_cpu_touch_summary.txt"
  local hypothesis="missing_summary"
  local weight_faults="NA"
  local kv_faults="NA"
  local candidate_faults="NA"
  local registered_faults="NA"
  local total_faults="NA"

  if [[ -f "${summary}" ]]; then
    hypothesis="$(summary_value "${summary}" stage_a_hypothesis)"
    weight_faults="$(summary_value "${summary}" weight_cpu_faults)"
    kv_faults="$(summary_value "${summary}" kv_cpu_faults)"
    candidate_faults="$(summary_value "${summary}" stage_a_cpu_faults_candidate)"
    registered_faults="$(summary_value "${summary}" stage_a_cpu_faults_registered)"
    total_faults="$(summary_value "${summary}" stage_a_cpu_faults_total)"
  fi

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${case_name}" \
    "${output_len}" \
    "${max_model_len}" \
    "${exit_code}" \
    "${hypothesis}" \
    "${weight_faults}" \
    "${kv_faults}" \
    "${candidate_faults}" \
    "${registered_faults}" \
    "${total_faults}" \
    >> "${SUMMARY_CSV}"
}

write_verdict() {
  local failures=0
  local cases=0

  {
    echo "Stage A vLLM CPU touch sweep verdict"
    echo "run_root=${RUN_ROOT}"
    echo "summary_csv=${SUMMARY_CSV}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "config:"
    echo "output_len_values=${OUTPUT_LEN_VALUES}"
    echo "max_model_len_values=${MAX_MODEL_LEN_VALUES}"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo "device=${DEVICE}"
    echo
    echo "case_results:"
  } > "${VERDICT}"

  while IFS=, read -r case_name output_len max_model_len exit_code hypothesis weight_faults kv_faults candidate_faults registered_faults total_faults; do
    if [[ "${case_name}" == "case" ]]; then
      continue
    fi

    cases=$((cases + 1))

    if [[ "${exit_code}" == "0" &&
          "${hypothesis}" == "passes_for_weight_and_kv" &&
          "${weight_faults}" == "0" &&
          "${kv_faults}" == "0" &&
          "${candidate_faults}" == "0" ]]; then
      echo "PASS ${case_name}: output_len=${output_len} max_model_len=${max_model_len} weight=${weight_faults} kv=${kv_faults} candidate=${candidate_faults}" >> "${VERDICT}"
    else
      echo "FAIL ${case_name}: exit=${exit_code} hypothesis=${hypothesis} weight=${weight_faults} kv=${kv_faults} candidate=${candidate_faults} registered=${registered_faults} total=${total_faults}" >> "${VERDICT}"
      failures=$((failures + 1))
    fi
  done < "${SUMMARY_CSV}"

  {
    echo
    echo "cases=${cases}"
    echo "failures=${failures}"
    if [[ "${failures}" -eq 0 ]]; then
      echo "stage_a_vllm_cpu_touch_sweep=pass"
      echo "next_step=begin_soft_gpu_only_driver_manual_allocation"
    else
      echo "stage_a_vllm_cpu_touch_sweep=fail"
      echo "next_step=inspect_stage_a_cpu_fault_events_before_driver_policy"
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
    echo "output_len_values=${OUTPUT_LEN_VALUES}"
    echo "max_model_len_values=${MAX_MODEL_LEN_VALUES}"
    echo "prompts=${PROMPTS}"
    echo "request_rate=${REQUEST_RATE}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "server_gpu_memory_utilization=${SERVER_GPU_MEMORY_UTILIZATION}"
    echo "device=${DEVICE}"
    echo "stop_on_failure=${STOP_ON_FAILURE}"
  } | tee "${LOG}"

  printf 'case,output_len,max_model_len,exit_code,stage_a_hypothesis,weight_cpu_faults,kv_cpu_faults,stage_a_cpu_faults_candidate,stage_a_cpu_faults_registered,stage_a_cpu_faults_total\n' > "${SUMMARY_CSV}"

  for output_len in ${OUTPUT_LEN_VALUES}; do
    for max_model_len in ${MAX_MODEL_LEN_VALUES}; do
      local_case="o${output_len}_m${max_model_len}"
      case_root="${RUN_ROOT}/${local_case}"

      echo "Running ${local_case}" | tee -a "${LOG}"
      set +e
      RUN_ROOT="${case_root}" \
      PROMPTS="${PROMPTS}" \
      REQUEST_RATE="${REQUEST_RATE}" \
      OUTPUT_LEN="${output_len}" \
      MAX_MODEL_LEN="${max_model_len}" \
      SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS}" \
      SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION}" \
      DEVICE="${DEVICE}" \
      "${ROOT_DIR}/scripts/test_stage_a_vllm_cpu_touch_observation.sh" \
        > "${case_root}.stdout.log" \
        2> "${case_root}.stderr.log"
      exit_code=$?
      set -e

      append_case_summary "${local_case}" "${output_len}" "${max_model_len}" "${case_root}" "${exit_code}"

      if [[ "${exit_code}" -ne 0 && "${STOP_ON_FAILURE}" == "1" ]]; then
        echo "Stopping after ${local_case} failure. See ${case_root}.stderr.log" | tee -a "${LOG}"
        break 2
      fi
    done
  done

  echo "Summary: ${SUMMARY_CSV}" | tee -a "${LOG}"
  column -s, -t "${SUMMARY_CSV}" || cat "${SUMMARY_CSV}"
  echo
  write_verdict
  echo "Verdict: ${VERDICT}"
}

main "$@"
