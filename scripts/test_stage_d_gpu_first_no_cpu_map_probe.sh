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
CASES="${CASES:-none touch_after_no_map}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/stage_d_gpu_first_no_cpu_map_probe_$(date -u +%Y%m%d_%H%M%S)}"
VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS:-1}"

SRC="${ROOT_DIR}/tools/uvm_stage_d_gpu_first_no_cpu_map_probe.cu"
BIN="${RUN_ROOT}/uvm_stage_d_gpu_first_no_cpu_map_probe"
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
  stage_a_cpu_faults_kv
)

STDOUT_KEYS=(
  bytes
  adopted_existing_uvm_fd
  cpu_touch_after_cpu_no_map
  initial_apply_ms
  initial_apply_gpu_bytes
  initial_apply_cpu_unmap_bytes
  gpu_first_write_ms
  query_after_gpu_first_gpu_bytes
  query_after_gpu_first_cpu_bytes
  query_after_gpu_first_on_gpu
  query_after_gpu_first_on_cpu
  prefetch_cpu_no_map_ms
  prefetch_cpu_no_map_gpu_bytes
  prefetch_cpu_no_map_cpu_bytes
  query_cpu_gpu_bytes
  query_cpu_cpu_bytes
  query_cpu_on_cpu
  query_cpu_on_gpu
  cpu_touch_ms
  post_no_map_apply_ms
  post_no_map_apply_gpu_bytes
  post_no_map_apply_cpu_unmap_bytes
  gpu_second_write_ms
  query_final_gpu_bytes
  query_final_cpu_bytes
  query_final_on_gpu
  query_final_on_cpu
  clear_gpu_bytes
  clear_ms
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

stdout_value() {
  local stdout_path="$1"
  local key="$2"

  awk -v key="${key}" '
    $1 == "stage_d_gpu_first_no_cpu_map_probe" {
      for (i = 1; i <= NF; ++i) {
        split($i, kv, "=")
        if (kv[1] == key) {
          print kv[2]
          found = 1
          exit
        }
      }
    }
    END {
      if (!found)
        print "NA"
    }
  ' "${stdout_path}"
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
  local stdout_path="${case_dir}/probe_stdout.txt"
  local extra_args=()

  mkdir -p "${case_dir}"

  if [[ "${case_name}" == "touch_after_no_map" || "${case_name}" == "touch" ]]; then
    extra_args+=(--cpu-touch-after-cpu-no-map)
  elif [[ "${case_name}" != "none" ]]; then
    echo "ERROR: unsupported case ${case_name}; use none or touch_after_no_map" >&2
    exit 1
  fi

  reset_counters
  cat "${STATS_PATH}" > "${case_dir}/latency_stats_before.txt"

  "${BIN}" \
    --pages "${PAGES}" \
    --device "${DEVICE}" \
    "${extra_args[@]}" \
    > "${stdout_path}" \
    2> "${case_dir}/probe_stderr.txt"

  cat "${STATS_PATH}" > "${case_dir}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${case_dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  grep -E "phase_f_gpu_only_soft|phase_f_resident_bytes|gpu_only_soft_break|stage_a_cpu_fault" \
    "${case_dir}/phase_f_route_log_after.txt" \
    > "${case_dir}/gpu_only_events.log" || true

  printf '%s,%s' "${case_name}" "${PAGES}" >> "${SUMMARY}"

  for key in "${STDOUT_KEYS[@]}"; do
    printf ',%s' "$(stdout_value "${stdout_path}" "${key}")" >> "${SUMMARY}"
  done

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

summary_value() {
  local case_name="$1"
  local column="$2"

  awk -F, -v case_name="${case_name}" -v column="${column}" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) {
        if ($i == column)
          column_index = i
      }
      next
    }
    $1 == case_name {
      found = 1
      if (column_index > 0)
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

summary_delta() {
  local case_name="$1"
  local key="$2"
  summary_value "${case_name}" "${key}_delta"
}

validate_expectations() {
  local failures=0
  local expected_bytes=$((PAGES * 4096))

  check_eq() {
    local case_name="$1"
    local column="$2"
    local expected="$3"
    local value

    value="$(summary_value "${case_name}" "${column}")"
    if [[ "${value}" == "${expected}" ]]; then
      echo "PASS ${case_name}: ${column} == ${expected}"
    else
      echo "FAIL ${case_name}: expected ${column} == ${expected}, got ${value}"
      failures=$((failures + 1))
    fi
  }

  check_delta_eq() {
    local case_name="$1"
    local key="$2"
    local expected="$3"
    check_eq "${case_name}" "${key}_delta" "${expected}"
  }

  check_delta_gt_zero() {
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

  validate_common() {
    local case_name="$1"

    check_eq "${case_name}" adopted_existing_uvm_fd 1
    check_eq "${case_name}" initial_apply_cpu_unmap_bytes 0
    check_eq "${case_name}" query_cpu_cpu_bytes "${expected_bytes}"
    check_eq "${case_name}" query_cpu_on_cpu 1
    check_eq "${case_name}" post_no_map_apply_gpu_bytes "${expected_bytes}"
    check_eq "${case_name}" query_final_gpu_bytes "${expected_bytes}"
    check_eq "${case_name}" query_final_on_gpu 1
    check_delta_eq "${case_name}" gpu_only_failed_pages 0
  }

  {
    echo "Stage D GPU-first no-CPU-map probe validation"
    echo "run_root=${RUN_ROOT}"
    echo "summary=${SUMMARY}"
    echo

    if case_was_run none; then
      validate_common none
      check_eq none cpu_touch_after_cpu_no_map 0
      check_eq none post_no_map_apply_cpu_unmap_bytes 0
      check_delta_eq none gpu_only_cpu_unmap_pages 0
      check_delta_eq none gpu_only_cpu_faults 0
      check_delta_eq none gpu_only_soft_breaks 0
      check_delta_eq none stage_a_cpu_faults_kv 0
    fi

    if case_was_run touch_after_no_map; then
      validate_common touch_after_no_map
      check_eq touch_after_no_map cpu_touch_after_cpu_no_map 1
      check_eq touch_after_no_map post_no_map_apply_cpu_unmap_bytes "${expected_bytes}"
      check_delta_eq touch_after_no_map gpu_only_cpu_unmap_pages "${PAGES}"
      check_delta_gt_zero touch_after_no_map gpu_only_cpu_faults
      check_delta_gt_zero touch_after_no_map gpu_only_cpu_read_faults
      check_delta_gt_zero touch_after_no_map gpu_only_soft_breaks
      check_delta_gt_zero touch_after_no_map stage_a_cpu_faults_kv
    fi

    echo
    if [[ "${failures}" -eq 0 ]]; then
      echo "stage_d_gpu_first_no_cpu_map_probe=pass"
    else
      echo "stage_d_gpu_first_no_cpu_map_probe=fail"
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
    printf 'case,pages'
    for key in "${STDOUT_KEYS[@]}"; do
      printf ',%s' "${key}"
    done
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
  fi
}

main "$@"
