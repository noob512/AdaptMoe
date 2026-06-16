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
MODES="${MODES:-baseline_cpu_map cpu_no_map}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/stage_d_rw_cpu_storage_no_map_bench_$(date -u +%Y%m%d_%H%M%S)}"
VALIDATE_EXPECTATIONS="${VALIDATE_EXPECTATIONS:-1}"

SRC="${ROOT_DIR}/tools/uvm_stage_d_rw_cpu_storage_no_map_bench.cu"
BIN="${RUN_ROOT}/uvm_stage_d_rw_cpu_storage_no_map_bench"
SUMMARY="${RUN_ROOT}/summary.csv"
COMPARISON="${RUN_ROOT}/comparison.csv"
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
  iterations
  adopted_existing_uvm_fd
  init_ms
  setup_apply_ms
  setup_apply_gpu_bytes
  setup_apply_cpu_unmap_bytes
  setup_write_ms
  setup_cpu_migrate_ms
  setup_query_cpu_bytes
  setup_query_gpu_bytes
  setup_query_on_cpu
  setup_query_on_gpu
  avg_apply_ms
  min_apply_ms
  p50_apply_ms
  p90_apply_ms
  p99_apply_ms
  max_apply_ms
  avg_gpu_write_ms
  min_gpu_write_ms
  max_gpu_write_ms
  avg_cpu_migrate_ms
  min_cpu_migrate_ms
  p50_cpu_migrate_ms
  p90_cpu_migrate_ms
  p99_cpu_migrate_ms
  max_cpu_migrate_ms
  total_apply_gpu_bytes
  total_apply_cpu_unmap_bytes
  final_query_cpu_bytes
  final_query_gpu_bytes
  final_query_on_cpu
  final_query_on_gpu
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
    $1 == "stage_d_rw_cpu_storage_no_map_bench" {
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
  local iter_csv="${case_dir}/iterations.csv"

  mkdir -p "${case_dir}"

  reset_counters
  cat "${STATS_PATH}" > "${case_dir}/latency_stats_before.txt"

  "${BIN}" \
    --mode "${mode}" \
    --pages "${pages}" \
    --iterations "${ITERATIONS}" \
    --device "${DEVICE}" \
    --iteration-csv "${iter_csv}" \
    > "${stdout_path}" \
    2> "${case_dir}/stderr.txt"

  cat "${STATS_PATH}" > "${case_dir}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${case_dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  grep -E "phase_f_gpu_only_soft|phase_f_resident_bytes|phase_n_.*reclaim|gpu_only_soft_break|stage_a_cpu_fault" \
    "${case_dir}/phase_f_route_log_after.txt" \
    > "${case_dir}/gpu_only_events.log" || true

  printf '%s,%s,%s' "${case_name}" "${mode}" "${pages}" >> "${SUMMARY}"

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

write_comparison() {
  {
    echo "pages,baseline_avg_apply_ms,no_map_avg_apply_ms,apply_speedup,baseline_avg_cpu_migrate_ms,no_map_avg_cpu_migrate_ms,cpu_migrate_speedup,baseline_avg_cycle_ms,no_map_avg_cycle_ms,cycle_speedup,baseline_total_unmap_bytes,no_map_total_unmap_bytes,unmap_bytes_reduction,baseline_unmap_pages_delta,no_map_unmap_pages_delta,unmap_pages_reduction"
    for pages in ${PAGES_VALUES}; do
      local baseline_case="baseline_cpu_map_p${pages}"
      local no_map_case="cpu_no_map_p${pages}"

      if ! grep -q "^${baseline_case}," "${SUMMARY}" ||
         ! grep -q "^${no_map_case}," "${SUMMARY}"; then
        continue
      fi

      local baseline_apply no_map_apply baseline_migrate no_map_migrate baseline_write no_map_write
      local baseline_unmap_bytes no_map_unmap_bytes
      local baseline_unmap_pages no_map_unmap_pages
      baseline_apply="$(summary_value "${baseline_case}" avg_apply_ms)"
      no_map_apply="$(summary_value "${no_map_case}" avg_apply_ms)"
      baseline_migrate="$(summary_value "${baseline_case}" avg_cpu_migrate_ms)"
      no_map_migrate="$(summary_value "${no_map_case}" avg_cpu_migrate_ms)"
      baseline_write="$(summary_value "${baseline_case}" avg_gpu_write_ms)"
      no_map_write="$(summary_value "${no_map_case}" avg_gpu_write_ms)"
      baseline_unmap_bytes="$(summary_value "${baseline_case}" total_apply_cpu_unmap_bytes)"
      no_map_unmap_bytes="$(summary_value "${no_map_case}" total_apply_cpu_unmap_bytes)"
      baseline_unmap_pages="$(summary_value "${baseline_case}" gpu_only_cpu_unmap_pages_delta)"
      no_map_unmap_pages="$(summary_value "${no_map_case}" gpu_only_cpu_unmap_pages_delta)"

      awk -v pages="${pages}" \
          -v ba="${baseline_apply}" \
          -v na="${no_map_apply}" \
          -v bm="${baseline_migrate}" \
          -v nm="${no_map_migrate}" \
          -v bw="${baseline_write}" \
          -v nw="${no_map_write}" \
          -v bub="${baseline_unmap_bytes}" \
          -v nub="${no_map_unmap_bytes}" \
          -v bup="${baseline_unmap_pages}" \
          -v nup="${no_map_unmap_pages}" '
        BEGIN {
          speedup = (na > 0 ? ba / na : 0);
          migrate_speedup = (nm > 0 ? bm / nm : 0);
          baseline_cycle = ba + bm + bw;
          no_map_cycle = na + nm + nw;
          cycle_speedup = (no_map_cycle > 0 ? baseline_cycle / no_map_cycle : 0);
          byte_reduction = (bub > 0 ? (bub - nub) / bub : 0);
          page_reduction = (bup > 0 ? (bup - nup) / bup : 0);
          printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%.6f,%s,%s,%.6f\n",
                 pages, ba, na, speedup,
                 bm, nm, migrate_speedup,
                 baseline_cycle, no_map_cycle, cycle_speedup,
                 bub, nub, byte_reduction, bup, nup, page_reduction;
        }'
    done
  } > "${COMPARISON}"
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
    if [[ "${value}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] &&
       awk -v value="${value}" 'BEGIN { exit value > 0 ? 0 : 1 }'; then
      echo "PASS ${case_name}: ${column} > 0 (${value})"
    else
      echo "FAIL ${case_name}: expected ${column} > 0, got ${value}"
      failures=$((failures + 1))
    fi
  }

  check_lt() {
    local left_case="$1"
    local right_case="$2"
    local column="$3"
    local left right

    left="$(summary_value "${left_case}" "${column}")"
    right="$(summary_value "${right_case}" "${column}")"
    if [[ "${left}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] &&
       [[ "${right}" =~ ^-?[0-9]+([.][0-9]+)?$ ]] &&
       awk -v left="${left}" -v right="${right}" 'BEGIN { exit left < right ? 0 : 1 }'; then
      echo "PASS ${left_case}: ${column} (${left}) < ${right_case} (${right})"
    else
      echo "FAIL ${left_case}: expected ${column} < ${right_case}, got ${left} vs ${right}"
      failures=$((failures + 1))
    fi
  }

  {
    echo "Stage D RW CPU storage no-map benchmark validation"
    echo "run_root=${RUN_ROOT}"
    echo "summary=${SUMMARY}"
    echo "comparison=${COMPARISON}"
    echo

    for pages in ${PAGES_VALUES}; do
      local baseline_case="baseline_cpu_map_p${pages}"
      local no_map_case="cpu_no_map_p${pages}"
      local expected_setup_bytes=$((pages * 4096))

      if grep -q "^${baseline_case}," "${SUMMARY}"; then
        check_eq "${baseline_case}" adopted_existing_uvm_fd 1
        check_eq "${baseline_case}" setup_apply_cpu_unmap_bytes "${expected_setup_bytes}"
        check_eq "${baseline_case}" setup_query_cpu_bytes "${expected_setup_bytes}"
        check_eq "${baseline_case}" setup_query_on_cpu 1
        check_eq "${baseline_case}" final_query_cpu_bytes "${expected_setup_bytes}"
        check_eq "${baseline_case}" final_query_on_cpu 1
        check_gt_zero "${baseline_case}" total_apply_cpu_unmap_bytes
        check_gt_zero "${baseline_case}" avg_apply_ms
        check_eq "${baseline_case}" gpu_only_failed_pages_delta 0
      fi

      if grep -q "^${no_map_case}," "${SUMMARY}"; then
        check_eq "${no_map_case}" adopted_existing_uvm_fd 1
        check_eq "${no_map_case}" setup_apply_cpu_unmap_bytes "${expected_setup_bytes}"
        check_eq "${no_map_case}" setup_query_cpu_bytes "${expected_setup_bytes}"
        check_eq "${no_map_case}" setup_query_on_cpu 1
        check_eq "${no_map_case}" final_query_cpu_bytes "${expected_setup_bytes}"
        check_eq "${no_map_case}" final_query_on_cpu 1
        check_eq "${no_map_case}" total_apply_cpu_unmap_bytes 0
        check_eq "${no_map_case}" gpu_only_cpu_faults_delta 0
        check_eq "${no_map_case}" gpu_only_soft_breaks_delta 0
        check_eq "${no_map_case}" gpu_only_failed_pages_delta 0
        check_gt_zero "${no_map_case}" avg_apply_ms
      fi

      if grep -q "^${baseline_case}," "${SUMMARY}" &&
         grep -q "^${no_map_case}," "${SUMMARY}"; then
        check_lt "${no_map_case}" "${baseline_case}" gpu_only_cpu_unmap_pages_delta
      fi
    done

    echo
    if [[ "${failures}" -eq 0 ]]; then
      echo "stage_d_rw_cpu_storage_no_map_bench=pass"
    else
      echo "stage_d_rw_cpu_storage_no_map_bench=fail"
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
    for key in "${STDOUT_KEYS[@]}"; do
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

  write_comparison

  echo
  echo "Done."
  echo "Run root: ${RUN_ROOT}"
  echo "Summary: ${SUMMARY}"
  column -s, -t "${SUMMARY}" || cat "${SUMMARY}"
  echo
  echo "Comparison: ${COMPARISON}"
  column -s, -t "${COMPARISON}" || cat "${COMPARISON}"

  if [[ "${VALIDATE_EXPECTATIONS}" == "1" ]]; then
    echo
    validate_expectations
  fi
}

main "$@"
