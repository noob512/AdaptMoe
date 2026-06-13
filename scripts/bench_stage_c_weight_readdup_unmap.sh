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

PAGES_VALUES="${PAGES_VALUES:-1024 8192 65536}"
ITERATIONS="${ITERATIONS:-10}"
DEVICE="${DEVICE:-0}"
MODES="${MODES:-baseline_prefetch readdup_soft}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/stage_c_weight_readdup_unmap_bench_$(date -u +%Y%m%d_%H%M%S)}"
VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS:-1}"

SRC="${ROOT_DIR}/tools/uvm_stage_c_weight_readdup_unmap_bench.cu"
BIN="${RUN_ROOT}/uvm_stage_c_weight_readdup_unmap_bench"
SUMMARY="${RUN_ROOT}/summary.csv"
META="${RUN_ROOT}/metadata.txt"
VALIDATION="${RUN_ROOT}/validation.txt"

COUNTER_KEYS=(
  gpu_only_apply_calls
  gpu_only_apply_pages
  gpu_only_cpu_unmap_pages
  gpu_only_gpu_map_pages
  gpu_only_readdup_apply_calls
  gpu_only_readdup_apply_pages
  gpu_only_cpu_preserved_pages
  gpu_only_clear_calls
  gpu_only_clear_pages
  gpu_only_cpu_faults
  gpu_only_cpu_read_faults
  gpu_only_cpu_write_faults
  gpu_only_soft_breaks
  gpu_only_soft_break_pages
  gpu_only_failed_pages
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
    $1 == "stage_c_weight_unmap_bench" {
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

build_bench() {
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
    echo "pages_values=${PAGES_VALUES}"
    echo "iterations=${ITERATIONS}"
    echo "device=${DEVICE}"
    echo "modes=${MODES}"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
  } | tee "${META}"
}

run_case() {
  local mode="$1"
  local pages="$2"
  local case_name="${mode}_p${pages}"
  local case_dir="${RUN_ROOT}/${case_name}"
  local stdout_path="${case_dir}/stdout.txt"

  mkdir -p "${case_dir}"

  reset_counters
  cat "${STATS_PATH}" > "${case_dir}/latency_stats_before.txt"

  "${BIN}" \
    --mode "${mode}" \
    --pages "${pages}" \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    > "${stdout_path}" \
    2> "${case_dir}/stderr.txt"

  cat "${STATS_PATH}" > "${case_dir}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${case_dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  grep -E "phase_f_gpu_only_soft|phase_f_resident_bytes|gpu_only_soft_break" \
    "${case_dir}/phase_f_route_log_after.txt" \
    > "${case_dir}/gpu_only_events.log" || true

  printf '%s,%s,%s' "${case_name}" "${mode}" "${pages}" >> "${SUMMARY}"
  for key in \
    bytes iterations adopted_existing_uvm_fd init_ms apply_ms \
    avg_cpu_restore_ms avg_gpu_prefetch_ms avg_gpu_read_ms min_gpu_read_ms \
    apply_gpu_bytes apply_cpu_bytes query_gpu_bytes query_cpu_bytes query_on_cpu query_on_gpu; do
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

validate_expectations() {
  local failures=0

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

  check_gt_zero() {
    local case_name="$1"
    local column="$2"
    local value

    value="$(summary_value "${case_name}" "${column}")"
    if [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      if awk -v value="${value}" 'BEGIN { exit value > 0 ? 0 : 1 }'; then
        echo "PASS ${case_name}: ${column} > 0 (${value})"
        return
      fi
    fi

    echo "FAIL ${case_name}: expected ${column} > 0, got ${value}"
    failures=$((failures + 1))
  }

  {
    echo "Stage C weight read-dup unmap benchmark validation"
    echo "run_root=${RUN_ROOT}"
    echo "summary=${SUMMARY}"
    echo

    for pages in ${PAGES_VALUES}; do
      local readdup_case="readdup_soft_p${pages}"
      local baseline_case="baseline_prefetch_p${pages}"

      if grep -q "^${readdup_case}," "${SUMMARY}"; then
        check_eq "${readdup_case}" gpu_only_readdup_apply_pages_delta "${pages}"
        check_eq "${readdup_case}" gpu_only_cpu_preserved_pages_delta "${pages}"
        check_eq "${readdup_case}" gpu_only_cpu_unmap_pages_delta "${pages}"
        check_eq "${readdup_case}" gpu_only_cpu_faults_delta 0
        check_eq "${readdup_case}" gpu_only_soft_breaks_delta 0
        check_eq "${readdup_case}" gpu_only_failed_pages_delta 0
        check_eq "${readdup_case}" query_on_cpu 1
        check_eq "${readdup_case}" query_on_gpu 1
        check_gt_zero "${readdup_case}" apply_ms
        check_gt_zero "${readdup_case}" avg_gpu_read_ms
      fi

      if grep -q "^${baseline_case}," "${SUMMARY}"; then
        check_gt_zero "${baseline_case}" avg_gpu_prefetch_ms
        check_gt_zero "${baseline_case}" avg_cpu_restore_ms
      fi
    done

    echo
    if [[ "${failures}" -eq 0 ]]; then
      echo "stage_c_weight_readdup_unmap_bench=pass"
    else
      echo "stage_c_weight_readdup_unmap_bench=fail"
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
  build_bench

  {
    printf 'case,mode,pages'
    for key in \
      bytes iterations adopted_existing_uvm_fd init_ms apply_ms \
      avg_cpu_restore_ms avg_gpu_prefetch_ms avg_gpu_read_ms min_gpu_read_ms \
      apply_gpu_bytes apply_cpu_bytes query_gpu_bytes query_cpu_bytes query_on_cpu query_on_gpu; do
      printf ',%s' "${key}"
    done
    for key in "${COUNTER_KEYS[@]}"; do
      printf ',%s_delta' "${key}"
    done
    printf '\n'
  } > "${SUMMARY}"

  for pages in ${PAGES_VALUES}; do
    for mode in ${MODES}; do
      echo "Running mode=${mode} pages=${pages}"
      run_case "${mode}" "${pages}"
    done
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
