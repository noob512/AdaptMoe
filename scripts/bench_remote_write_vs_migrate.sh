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
PASSES="${PASSES:-3}"
DEVICE="${DEVICE:-0}"
MODES="${MODES:-remote_cpu prefetch_gpu fault_migrate}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/remote_write_vs_migrate_bench_$(date -u +%Y%m%d_%H%M%S)}"

SRC="${ROOT_DIR}/tools/uvm_remote_write_vs_migrate_bench.cu"
BIN="${RUN_ROOT}/uvm_remote_write_vs_migrate_bench"
SUMMARY="${RUN_ROOT}/summary.csv"
META="${RUN_ROOT}/metadata.txt"

mkdir -p "${RUN_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${path}. Load the modified nvidia_uvm module first." >&2
    exit 1
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

csv_value() {
  local stdout_path="$1"
  local key="$2"

  awk -F, -v key="${key}" '
    NR == 1 {
      for (i = 1; i <= NF; ++i) {
        if ($i == key)
          column_index = i
      }
      next
    }
    NR == 2 {
      if (column_index > 0)
        print $column_index
      else
        print "NA"
      found = 1
      exit
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
    echo "passes=${PASSES}"
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
    --passes "${PASSES}" \
    --device "${DEVICE}" \
    > "${stdout_path}" \
    2> "${case_dir}/stderr.txt"

  cat "${STATS_PATH}" > "${case_dir}/latency_stats_after.txt"
  cat "${ROUTE_LOG_PATH}" > "${case_dir}/phase_f_route_log_after.txt" 2>/dev/null || true

  printf '%s,%s,%s' "${case_name}" "${mode}" "${pages}" >> "${SUMMARY}"
  for key in setup_ms prepare_ms wall_ms event_ms wall_gib_s event_gib_s query_before_cpu_bytes query_before_gpu_bytes query_after_cpu_bytes query_after_gpu_bytes; do
    printf ',%s' "$(csv_value "${stdout_path}" "${key}")" >> "${SUMMARY}"
  done

  printf ',%s,%s\n' \
    "$(( $(counter_value "${case_dir}/latency_stats_after.txt" gpu_only_apply_pages) - $(counter_value "${case_dir}/latency_stats_before.txt" gpu_only_apply_pages) ))" \
    "$(( $(counter_value "${case_dir}/latency_stats_after.txt" gpu_only_cpu_unmap_pages) - $(counter_value "${case_dir}/latency_stats_before.txt" gpu_only_cpu_unmap_pages) ))" \
    >> "${SUMMARY}"
}

{
  echo "case,mode,pages,setup_ms,prepare_ms,wall_ms,event_ms,wall_gib_s,event_gib_s,query_before_cpu_bytes,query_before_gpu_bytes,query_after_cpu_bytes,query_after_gpu_bytes,gpu_only_apply_pages_delta,gpu_only_cpu_unmap_pages_delta"
} > "${SUMMARY}"

build_bench
write_metadata

for mode in ${MODES}; do
  for pages in ${PAGES_VALUES}; do
    echo "Running ${mode}_p${pages}"
    run_case "${mode}" "${pages}"
  done
done

echo "Summary: ${SUMMARY}"
