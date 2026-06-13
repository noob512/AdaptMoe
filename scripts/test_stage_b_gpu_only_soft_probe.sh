#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NVCC_ARCH="${NVCC_ARCH:-}"

if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi

STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"
ROUTE_LOG_PATH="${ROUTE_LOG_PATH:-/proc/driver/nvidia-uvm/phase_f_route_log}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

PAGES="${PAGES:-1024}"
DEVICE="${DEVICE:-0}"
CASES="${CASES:-none weight kv both}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/stage_b_gpu_only_soft_probe_$(date -u +%Y%m%d_%H%M%S)}"
VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS:-1}"

SRC="${ROOT_DIR}/tools/uvm_stage_b_gpu_only_soft_probe.cu"
BIN="${RUN_ROOT}/uvm_stage_b_gpu_only_soft_probe"
SUMMARY="${RUN_ROOT}/summary.csv"
META="${RUN_ROOT}/metadata.txt"
VALIDATION="${RUN_ROOT}/validation.txt"

COUNTER_KEYS=(
  gpu_only_apply_calls
  gpu_only_apply_pages
  gpu_only_cpu_unmap_pages
  gpu_only_gpu_map_pages
  gpu_only_clear_calls
  gpu_only_clear_pages
  gpu_only_cpu_faults
  gpu_only_cpu_read_faults
  gpu_only_cpu_write_faults
  gpu_only_soft_breaks
  gpu_only_soft_break_pages
  gpu_only_failed_pages
  stage_a_cpu_faults_registered
  stage_a_cpu_faults_candidate
  stage_a_cpu_read_faults
  stage_a_cpu_write_faults
  stage_a_cpu_faults_weight
  stage_a_cpu_faults_kv
)

mkdir -p "${RUN_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${path}. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi
}

write_param_if_exists() {
  local name="$1"
  local value="$2"
  local path="${PARAM_ROOT}/${name}"

  if [[ ! -e "${path}" ]]; then
    return
  fi

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

build_probe() {
  echo "Building ${BIN}"
  "${NVCC}" -O2 -std=c++17 ${NVCC_ARCH} \
    -I"${ROOT_DIR}/kernel-module/nvidia-module/kernel-open/nvidia-uvm" \
    -I"${ROOT_DIR}/kernel-module/nvidia-module/kernel-open/common/inc" \
    "${SRC}" -o "${BIN}"
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "nvcc=${NVCC}"
    echo "nvcc_arch=${NVCC_ARCH}"
    echo "pages=${PAGES}"
    echo "device=${DEVICE}"
    echo "cases=${CASES}"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
  } | tee "${META}"
}

run_case() {
  local case_name="$1"
  local case_dir="${RUN_ROOT}/${case_name}"
  mkdir -p "${case_dir}"

  reset_counters
  cat "${STATS_PATH}" > "${case_dir}/latency_stats_before.txt"

  "${BIN}" \
    --pages "${PAGES}" \
    --device "${DEVICE}" \
    --cpu-touch "${case_name}" \
    > "${case_dir}/probe_stdout.txt" \
    2> "${case_dir}/probe_stderr.txt"

  cat "${STATS_PATH}" > "${case_dir}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${case_dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  grep -E "phase_f_gpu_only_soft|gpu_only_soft_break" \
    "${case_dir}/phase_f_route_log_after.txt" \
    > "${case_dir}/gpu_only_events.log" || true

  printf '%s' "${case_name}" >> "${SUMMARY}"
  for key in "${COUNTER_KEYS[@]}"; do
    local before after
    before="$(counter_value "${case_dir}/latency_stats_before.txt" "${key}")"
    after="$(counter_value "${case_dir}/latency_stats_after.txt" "${key}")"
    printf ',%s' "$((after - before))" >> "${SUMMARY}"
  done
  printf '\n' >> "${SUMMARY}"
}

case_was_run() {
  local case_name="$1"

  awk -F, -v case_name="${case_name}" 'NR > 1 && $1 == case_name { found = 1 } END { exit found ? 0 : 1 }' "${SUMMARY}"
}

summary_delta() {
  local case_name="$1"
  local key="$2"

  awk -F, -v case_name="${case_name}" -v column="${key}_delta" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) {
        if ($i == column)
          column_index = i
      }
      next
    }
    $1 == case_name {
      found = 1
      if (column_index > 0 && $column_index ~ /^-?[0-9]+$/)
        print $column_index
      else
        print "NA"
      exit
    }
    END {
      if (!found)
        print "NA"
    }
  ' "${SUMMARY}"
}

validate_expectations() {
  local failures=0

  check_eq_zero() {
    local case_name="$1"
    local key="$2"
    local value

    value="$(summary_delta "${case_name}" "${key}")"
    if [[ "${value}" == "0" ]]; then
      echo "PASS ${case_name}: ${key}_delta == 0"
    else
      echo "FAIL ${case_name}: expected ${key}_delta == 0, got ${value}"
      failures=$((failures + 1))
    fi
  }

  check_gt_zero() {
    local case_name="$1"
    local key="$2"
    local value

    value="$(summary_delta "${case_name}" "${key}")"
    if [[ "${value}" =~ ^-?[0-9]+$ && "${value}" -gt 0 ]]; then
      echo "PASS ${case_name}: ${key}_delta > 0 (${value})"
    else
      echo "FAIL ${case_name}: expected ${key}_delta > 0, got ${value}"
      failures=$((failures + 1))
    fi
  }

  check_stdout_has() {
    local case_name="$1"
    local pattern="$2"
    local stdout_path="${RUN_ROOT}/${case_name}/probe_stdout.txt"

    if grep -q "${pattern}" "${stdout_path}"; then
      echo "PASS ${case_name}: probe_stdout contains ${pattern}"
    else
      echo "FAIL ${case_name}: probe_stdout missing ${pattern}"
      failures=$((failures + 1))
    fi
  }

  validate_common() {
    local case_name="$1"

    check_stdout_has "${case_name}" "adopted_existing_uvm_fd=1"
    check_stdout_has "${case_name}" "phase_f_action name=gpu_only_read_soft region=weight status=0"
    check_stdout_has "${case_name}" "phase_f_action name=gpu_only_rw_soft region=kv status=0"
    check_gt_zero "${case_name}" gpu_only_apply_calls
    check_gt_zero "${case_name}" gpu_only_apply_pages
    check_gt_zero "${case_name}" gpu_only_cpu_unmap_pages
    check_gt_zero "${case_name}" gpu_only_gpu_map_pages
    check_eq_zero "${case_name}" gpu_only_failed_pages
  }

  validate_case() {
    local case_name="$1"

    if ! case_was_run "${case_name}"; then
      echo "SKIP ${case_name}: case not requested"
      return
    fi

    validate_common "${case_name}"

    case "${case_name}" in
      none)
        check_eq_zero none gpu_only_cpu_faults
        check_eq_zero none gpu_only_cpu_read_faults
        check_eq_zero none gpu_only_cpu_write_faults
        check_eq_zero none gpu_only_soft_breaks
        check_gt_zero none gpu_only_clear_pages
        ;;
      weight)
        check_gt_zero weight gpu_only_cpu_faults
        check_gt_zero weight gpu_only_cpu_read_faults
        check_eq_zero weight gpu_only_cpu_write_faults
        check_gt_zero weight gpu_only_soft_breaks
        check_gt_zero weight gpu_only_soft_break_pages
        check_gt_zero weight stage_a_cpu_faults_weight
        check_eq_zero weight stage_a_cpu_faults_kv
        ;;
      kv)
        check_gt_zero kv gpu_only_cpu_faults
        check_gt_zero kv gpu_only_cpu_write_faults
        check_gt_zero kv gpu_only_soft_breaks
        check_gt_zero kv gpu_only_soft_break_pages
        check_gt_zero kv stage_a_cpu_faults_kv
        check_eq_zero kv stage_a_cpu_faults_weight
        ;;
      both)
        check_gt_zero both gpu_only_cpu_faults
        check_gt_zero both gpu_only_cpu_read_faults
        check_gt_zero both gpu_only_cpu_write_faults
        check_gt_zero both gpu_only_soft_breaks
        check_gt_zero both gpu_only_soft_break_pages
        check_gt_zero both stage_a_cpu_faults_weight
        check_gt_zero both stage_a_cpu_faults_kv
        ;;
      *)
        echo "FAIL ${case_name}: unknown case"
        failures=$((failures + 1))
        ;;
    esac
  }

  {
    echo "Stage B soft GPU-only manual probe validation"
    echo "run_root=${RUN_ROOT}"
    echo "summary=${SUMMARY}"
    echo
    validate_case none
    validate_case weight
    validate_case kv
    validate_case both
    echo
    if [[ "${failures}" -eq 0 ]]; then
      echo "stage_b_gpu_only_soft_probe=pass"
    else
      echo "stage_b_gpu_only_soft_probe=fail"
      echo "failure_count=${failures}"
    fi
  } > "${VALIDATION}"

  cat "${VALIDATION}"

  if [[ "${failures}" -ne 0 ]]; then
    return 1
  fi
}

main() {
  require_file "${STATS_PATH}"
  require_file "${ROUTE_LOG_PATH}"

  write_param_if_exists uvm_stage_a_cpu_fault_observe_enable 1
  write_param_if_exists uvm_stage_a_cpu_fault_log_enable 1

  write_metadata
  build_probe

  {
    printf 'case'
    for key in "${COUNTER_KEYS[@]}"; do
      printf ',%s_delta' "${key}"
    done
    printf '\n'
  } > "${SUMMARY}"

  for case_name in ${CASES}; do
    echo "Running case=${case_name}"
    run_case "${case_name}"
  done

  echo
  echo "Done."
  echo "Run root: ${RUN_ROOT}"
  echo "Summary: ${SUMMARY}"
  column -s, -t "${SUMMARY}" || cat "${SUMMARY}"

  if [[ "${VALIDATE_EXPECTATIONS}" == "1" ]]; then
    echo
    validate_expectations
    echo "Validation: ${VALIDATION}"
  fi
}

main "$@"
