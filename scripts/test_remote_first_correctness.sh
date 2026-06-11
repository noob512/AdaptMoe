#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NVCC_ARCH="${NVCC_ARCH:--arch=native}"

if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi

STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

PAGES="${PAGES:-4096}"
WRITE_PAGES="${WRITE_PAGES:-512}"
PASSES="${PASSES:-4}"
OPS_PER_PAGE="${OPS_PER_PAGE:-16}"
STREAMS="${STREAMS:-4}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
REMOTE_FIRST="${REMOTE_FIRST:-1}"
BG_PROMOTE="${BG_PROMOTE:-1}"
PREPARE_REMOTE="${PREPARE_REMOTE:-0}"
PREPARE_REMOTE_BG="${PREPARE_REMOTE_BG:-0}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/remote_first_correctness_$(date -u +%Y%m%d_%H%M%S)}"

SRC="${ROOT_DIR}/tools/uvm_remote_first_correctness_probe.cu"
BIN="${RUN_ROOT}/uvm_remote_first_correctness_probe"
CSV="${RUN_ROOT}/correctness_raw.csv"
SUMMARY="${RUN_ROOT}/correctness_summary.md"
META="${RUN_ROOT}/metadata.txt"
STATS_BEFORE="${RUN_ROOT}/latency_stats_before.txt"
STATS_AFTER="${RUN_ROOT}/latency_stats_after.txt"
STATS_DELTA="${RUN_ROOT}/latency_stats_delta.txt"

COUNTER_KEYS=(
  remote_first_faults
  remote_first_pages
  bg_promote_enqueued_pages
  bg_promote_copied_pages
  bg_promote_committed_pages
  bg_promote_cancelled_pages
  remote_map_pages
  bg_promote_demand_pages_enqueued
  bg_promote_prefetch_pages_enqueued
  bg_promote_demand_pages_promoted
  bg_promote_prefetch_pages_promoted
  pte_switch_pages
  bg_promote_worker_rounds
  bg_promote_retry_count
  bg_promote_failed_not_cpu_resident
  bg_promote_failed_already_gpu_resident
  bg_promote_failed_allocation
  bg_promote_failed_copy
  bg_promote_failed_unmap
  bg_promote_failed_map
  bg_promote_failed_block_dead
  sync_fallback_pages
  remote_prepare_calls
  remote_prepare_pages
  remote_prepare_mapped_pages
  bg_promote_delayed_schedules
  bg_promote_immediate_schedules
  bg_promote_timer_fires
  bg_promote_deferred_pages
  bg_promote_start_pages
)

mkdir -p "${RUN_ROOT}"

write_param() {
  local name="$1"
  local value="$2"
  local path="${PARAM_ROOT}/${name}"

  if [[ ! -e "${path}" ]]; then
    echo "ERROR: ${path} is missing. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi

  if [[ ! -w "${path}" && "${EUID}" -ne 0 ]]; then
    echo "${value}" | sudo tee "${path}" >/dev/null
  else
    echo "${value}" > "${path}"
  fi
}

reset_counters_if_requested() {
  if [[ "${RESET_COUNTERS}" != "1" ]]; then
    return
  fi

  if [[ ! -e "${STATS_RESET_PATH}" ]]; then
    echo "WARNING: ${STATS_RESET_PATH} is missing; continuing with before/after deltas." >&2
    return
  fi

  if [[ ! -w "${STATS_RESET_PATH}" && "${EUID}" -ne 0 ]]; then
    echo 1 | sudo tee "${STATS_RESET_PATH}" >/dev/null
  else
    echo 1 > "${STATS_RESET_PATH}"
  fi
}

capture_stats_quiet() {
  local out="$1"
  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi
  cat "${STATS_PATH}" > "${out}"
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "stats_path=${STATS_PATH}"
    echo "stats_reset_path=${STATS_RESET_PATH}"
    echo "nvcc=${NVCC}"
    echo "nvcc_arch=${NVCC_ARCH}"
    echo "pages=${PAGES}"
    echo "write_pages=${WRITE_PAGES}"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "streams=${STREAMS}"
    echo "device=${DEVICE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "remote_first=${REMOTE_FIRST}"
    echo "bg_promote=${BG_PROMOTE}"
    echo "prepare_remote=${PREPARE_REMOTE}"
    echo "prepare_remote_bg=${PREPARE_REMOTE_BG}"
    echo "reset_counters=${RESET_COUNTERS}"
    echo
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    echo "UVM module parameters:"
    for p in \
      "${PARAM_ROOT}/uvm_latency_remote_first_enable" \
      "${PARAM_ROOT}/uvm_latency_bg_promote_enable" \
      "${PARAM_ROOT}/uvm_latency_bg_promote_max_pages" \
      "${PARAM_ROOT}/uvm_latency_bg_promote_delay_ms" \
      "${PARAM_ROOT}/uvm_latency_bg_promote_reschedule_delay_ms" \
      "${PARAM_ROOT}/uvm_perf_prefetch_enable" \
      "${PARAM_ROOT}/uvm_perf_access_counter_migration_enable" \
      "${PARAM_ROOT}/uvm_perf_access_counter_threshold" \
      "${PARAM_ROOT}/uvm_ats_mode"; do
      [[ -r "${p}" ]] && printf '  %-48s %s\n' "$(basename "${p}")" "$(cat "${p}")"
    done
  } | tee "${META}"
}

counter_value() {
  local file="$1"
  local key="$2"
  awk -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) print 0 }' "${file}"
}

summarize() {
  python3 - "${CSV}" "${STATS_BEFORE}" "${STATS_AFTER}" "${STATS_DELTA}" "${SUMMARY}" "${META}" "${COUNTER_KEYS[@]}" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
before_path = Path(sys.argv[2])
after_path = Path(sys.argv[3])
delta_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
meta_path = Path(sys.argv[6])
keys = sys.argv[7:]

def read_stats(path):
    values = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return values

before = read_stats(before_path)
after = read_stats(after_path)
deltas = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
delta_path.write_text("".join(f"{key} {deltas[key]}\n" for key in keys))

rows = list(csv.DictReader(csv_path.open(newline="")))
failed = [row for row in rows if row.get("status") != "PASS"]

lines = []
lines.append("# Remote-First Correctness Probe")
lines.append("")
lines.append(f"- Raw CSV: `{csv_path}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append(f"- Counter delta: `{delta_path}`")
lines.append("")
lines.append("## Correctness")
lines.append("")
lines.append("| Test | Status | Observed | Expected | Wall ms |")
lines.append("|---|---|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| `{row['test']}` | {row['status']} | {row['observed']} | {row['expected']} | {float(row['wall_ms']):.3f} |"
    )

lines.append("")
lines.append("## Counter Delta")
lines.append("")
lines.append("| Counter | Delta |")
lines.append("|---|---:|")
for key in keys:
    lines.append(f"| `{key}` | {deltas[key]} |")

lines.append("")
lines.append("## Interpretation")
lines.append("")
if failed:
    lines.append(f"- FAILED: {len(failed)} correctness checks failed. Inspect raw CSV and dmesg before trusting performance numbers.")
else:
    lines.append("- All correctness checks passed for GPU read, CPU read, CPU-write break, GPU-write fallback, and multi-stream read.")

if deltas.get("remote_first_faults", 0) > 0:
    lines.append("- Remote-first fault handling was exercised.")
if deltas.get("bg_promote_committed_pages", 0) > 0:
    lines.append("- Background read-dup promotion committed pages.")
if deltas.get("pte_switch_pages", 0) > 0:
    lines.append("- PTE switch to local GPU read mapping was observed.")
if deltas.get("bg_promote_cancelled_pages", 0) > 0:
    lines.append("- Some background pages were cancelled or skipped; check failure counters.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

echo "run_root=${RUN_ROOT}"

write_param uvm_latency_remote_first_enable "${REMOTE_FIRST}"
write_param uvm_latency_bg_promote_enable "${BG_PROMOTE}"
write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"

reset_counters_if_requested
write_metadata

echo "Building ${BIN}"
"${NVCC}" -O3 -std=c++17 ${NVCC_ARCH} \
  -I"${ROOT_DIR}/kernel-module/nvidia-module/kernel-open/nvidia-uvm" \
  -I"${ROOT_DIR}/kernel-module/nvidia-module/kernel-open/common/inc" \
  "${SRC}" -o "${BIN}"

capture_stats_quiet "${STATS_BEFORE}"

probe_args=(
  --pages "${PAGES}"
  --write-pages "${WRITE_PAGES}"
  --passes "${PASSES}"
  --ops-per-page "${OPS_PER_PAGE}"
  --streams "${STREAMS}"
  --device "${DEVICE}"
)

if [[ "${PREPARE_REMOTE}" == "1" && "${PREPARE_REMOTE_BG}" == "1" ]]; then
  echo "ERROR: PREPARE_REMOTE and PREPARE_REMOTE_BG are mutually exclusive." >&2
  exit 1
elif [[ "${PREPARE_REMOTE}" == "1" ]]; then
  probe_args+=(--prepare-remote)
elif [[ "${PREPARE_REMOTE_BG}" == "1" ]]; then
  probe_args+=(--prepare-remote-bg)
fi

"${BIN}" \
  "${probe_args[@]}" | tee "${CSV}"

capture_stats_quiet "${STATS_AFTER}"
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Raw CSV: ${CSV}"
echo "Summary: ${SUMMARY}"
echo "Counter delta: ${STATS_DELTA}"
