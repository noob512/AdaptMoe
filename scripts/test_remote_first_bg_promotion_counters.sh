#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NVCC_ARCH="${NVCC_ARCH:--arch=native}"

if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi

UVM_KO="${UVM_KO:-${ROOT_DIR}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

PAGES="${PAGES:-16384}"
PASSES="${PASSES:-6}"
OPS_PER_PAGE="${OPS_PER_PAGE:-128}"
COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE:-512}"
REPEATS="${REPEATS:-2}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
CASES="${CASES:-remote_first_bg remote_first_only readonly_dup_sync counter_migrate local_prefetch default_uvm}"
RELOAD_UVM="${RELOAD_UVM:-0}"
RESET_COUNTERS="${RESET_COUNTERS:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/remote_first_counter_probe_$(date -u +%Y%m%d_%H%M%S)}"

SRC="${ROOT_DIR}/tools/uvm_remote_first_counter_probe.cu"
BIN="${RUN_ROOT}/uvm_remote_first_counter_probe"
CSV="${RUN_ROOT}/probe_raw.csv"
CASE_STATS_CSV="${RUN_ROOT}/counter_delta_by_case.csv"
SUMMARY="${RUN_ROOT}/probe_summary.md"
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
  bg_promote_delayed_schedules
  bg_promote_immediate_schedules
  bg_promote_timer_fires
  bg_promote_deferred_pages
  bg_promote_start_pages
)

mkdir -p "${RUN_ROOT}"

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

write_param() {
  local name="$1"
  local value="$2"
  local path="${PARAM_ROOT}/${name}"

  if [[ ! -w "${path}" && "${EUID}" -ne 0 ]]; then
    echo "${value}" | sudo tee "${path}" >/dev/null
  else
    echo "${value}" > "${path}"
  fi
}

reload_uvm_if_requested() {
  if [[ "${RELOAD_UVM}" != "1" ]]; then
    return
  fi

  if [[ ! -f "${UVM_KO}" ]]; then
    echo "ERROR: UVM_KO not found: ${UVM_KO}" >&2
    exit 1
  fi

  echo "Reloading nvidia_uvm from ${UVM_KO}"
  run_sudo modprobe -r nvidia_uvm
  run_sudo insmod "${UVM_KO}" \
    uvm_latency_remote_first_enable=1 \
    uvm_latency_bg_promote_enable=1 \
    uvm_latency_bg_promote_max_pages="${BG_MAX_PAGES}" \
    uvm_latency_bg_promote_delay_ms="${BG_DELAY_MS}" \
    uvm_latency_bg_promote_reschedule_delay_ms="${BG_RESCHEDULE_DELAY_MS}" \
    uvm_phase_f_route_log_delay_ms="${PHASE_F_ROUTE_LOG_DELAY_MS}"
}

check_runtime_params() {
  local required=(
    uvm_latency_remote_first_enable
    uvm_latency_bg_promote_enable
    uvm_latency_bg_promote_max_pages
    uvm_latency_bg_promote_delay_ms
    uvm_latency_bg_promote_reschedule_delay_ms
  )

  for name in "${required[@]}"; do
    if [[ ! -e "${PARAM_ROOT}/${name}" ]]; then
      echo "ERROR: ${PARAM_ROOT}/${name} is missing." >&2
      echo "The currently loaded nvidia_uvm module is probably not the newly built one." >&2
      echo "Run with RELOAD_UVM=1, or load ${UVM_KO} manually." >&2
      exit 1
    fi
  done

  write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
  write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
  write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"
}

set_driver_case_params() {
  local remote_first="$1"
  local bg_promote="$2"

  write_param uvm_latency_remote_first_enable "${remote_first}"
  write_param uvm_latency_bg_promote_enable "${bg_promote}"
  write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
  write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
  write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"
}

capture_stats() {
  local out="$1"

  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing." >&2
    echo "The procfs counter node is not available. Load the newly built nvidia-uvm.ko first." >&2
    exit 1
  fi

  cat "${STATS_PATH}" | tee "${out}"
}

capture_stats_quiet() {
  local out="$1"

  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing." >&2
    exit 1
  fi

  cat "${STATS_PATH}" > "${out}"
}

reset_counters_if_requested() {
  if [[ "${RESET_COUNTERS}" != "1" ]]; then
    return
  fi

  if [[ ! -e "${STATS_RESET_PATH}" ]]; then
    echo "WARNING: ${STATS_RESET_PATH} is missing; using before/after deltas without reset." >&2
    return
  fi

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

append_case_delta() {
  local case_name="$1"
  local profile="$2"
  local repeat="$3"
  local ops="$4"
  local remote_first="$5"
  local bg_promote="$6"
  local before="$7"
  local after="$8"

  printf '%s,%s,%s,%s,%s,%s' "${case_name}" "${profile}" "${repeat}" "${ops}" "${remote_first}" "${bg_promote}" >> "${CASE_STATS_CSV}"
  for key in "${COUNTER_KEYS[@]}"; do
    local b
    local a
    b="$(counter_value "${before}" "${key}")"
    a="$(counter_value "${after}" "${key}")"
    printf ',%s' "$((a - b))" >> "${CASE_STATS_CSV}"
  done
  printf '\n' >> "${CASE_STATS_CSV}"
}

build_probe() {
  echo "Building ${BIN}"
  "${NVCC}" -O3 -std=c++17 ${NVCC_ARCH} "${SRC}" -o "${BIN}"
}

case_profile() {
  case "$1" in
    remote_first_bg) echo "remote_first_bg" ;;
    remote_first_only) echo "remote_first_only" ;;
    readonly_dup_sync) echo "readonly_dup_sync" ;;
    counter_migrate) echo "counter_migrate" ;;
    local_prefetch) echo "local_prefetch" ;;
    default_uvm) echo "default_uvm" ;;
    *) echo "ERROR: unknown case $1" >&2; exit 1 ;;
  esac
}

case_remote_first() {
  case "$1" in
    remote_first_bg|remote_first_only) echo 1 ;;
    readonly_dup_sync|counter_migrate|local_prefetch|default_uvm) echo 0 ;;
    *) echo "ERROR: unknown case $1" >&2; exit 1 ;;
  esac
}

case_bg_promote() {
  case "$1" in
    remote_first_bg) echo 1 ;;
    remote_first_only|readonly_dup_sync|counter_migrate|local_prefetch|default_uvm) echo 0 ;;
    *) echo "ERROR: unknown case $1" >&2; exit 1 ;;
  esac
}

case_ops_per_page() {
  case "$1" in
    counter_migrate) echo "${COUNTER_OPS_PER_PAGE}" ;;
    *) echo "${OPS_PER_PAGE}" ;;
  esac
}

run_probe_matrix() {
  local first=1

  {
    printf 'case,profile,repeat,ops_per_page,remote_first_enable,bg_promote_enable'
    for key in "${COUNTER_KEYS[@]}"; do
      printf ',%s_delta' "${key}"
    done
    printf '\n'
  } > "${CASE_STATS_CSV}"

  for repeat in $(seq 0 $((REPEATS - 1))); do
    for case_name in ${CASES}; do
      local profile
      local remote_first
      local bg_promote
      local ops
      local before
      local after

      profile="$(case_profile "${case_name}")"
      remote_first="$(case_remote_first "${case_name}")"
      bg_promote="$(case_bg_promote "${case_name}")"
      ops="$(case_ops_per_page "${case_name}")"
      before="${RUN_ROOT}/latency_stats_${case_name}_repeat${repeat}_before.txt"
      after="${RUN_ROOT}/latency_stats_${case_name}_repeat${repeat}_after.txt"

      echo "Running repeat=${repeat} case=${case_name} profile=${profile} remote_first=${remote_first} bg_promote=${bg_promote} ops_per_page=${ops}"
      set_driver_case_params "${remote_first}" "${bg_promote}"
      capture_stats_quiet "${before}"

      local args=(
        --profile "${profile}"
        --pages "${PAGES}"
        --passes "${PASSES}"
        --ops-per-page "${ops}"
        --repeat "${repeat}"
        --device "${DEVICE}"
      )

      if [[ "${first}" == "1" ]]; then
        "${BIN}" "${args[@]}" >> "${CSV}"
        first=0
      else
        "${BIN}" "${args[@]}" --no-header >> "${CSV}"
      fi

      capture_stats_quiet "${after}"
      append_case_delta "${case_name}" "${profile}" "${repeat}" "${ops}" "${remote_first}" "${bg_promote}" "${before}" "${after}"
    done
  done
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "uvm_ko=${UVM_KO}"
    echo "stats_path=${STATS_PATH}"
    echo "stats_reset_path=${STATS_RESET_PATH}"
    echo "nvcc=${NVCC}"
    echo "nvcc_arch=${NVCC_ARCH}"
    echo "pages=${PAGES}"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "counter_ops_per_page=${COUNTER_OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "phase_f_route_log_delay_ms=${PHASE_F_ROUTE_LOG_DELAY_MS}"
    echo "cases=${CASES}"
    echo "reload_uvm=${RELOAD_UVM}"
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
      "${PARAM_ROOT}/uvm_phase_f_route_log_delay_ms" \
      "${PARAM_ROOT}/uvm_perf_prefetch_enable" \
      "${PARAM_ROOT}/uvm_perf_access_counter_migration_enable" \
      "${PARAM_ROOT}/uvm_perf_access_counter_threshold" \
      "${PARAM_ROOT}/uvm_ats_mode"; do
      [[ -r "${p}" ]] && printf '  %-48s %s\n' "$(basename "${p}")" "$(cat "${p}")"
    done
  } | tee "${META}"
}

summarize() {
  python3 - "${CSV}" "${STATS_BEFORE}" "${STATS_AFTER}" "${STATS_DELTA}" "${SUMMARY}" "${META}" "${CASE_STATS_CSV}" <<'PY'
import csv
import statistics as stats
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
before_path = Path(sys.argv[2])
after_path = Path(sys.argv[3])
delta_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
meta_path = Path(sys.argv[6])
case_stats_path = Path(sys.argv[7])

keys = [
    "remote_first_faults",
    "remote_first_pages",
    "bg_promote_enqueued_pages",
    "bg_promote_copied_pages",
    "bg_promote_committed_pages",
    "bg_promote_cancelled_pages",
    "remote_map_pages",
    "bg_promote_demand_pages_enqueued",
    "bg_promote_prefetch_pages_enqueued",
    "bg_promote_demand_pages_promoted",
    "bg_promote_prefetch_pages_promoted",
    "pte_switch_pages",
    "bg_promote_worker_rounds",
    "bg_promote_retry_count",
    "bg_promote_failed_not_cpu_resident",
    "bg_promote_failed_already_gpu_resident",
    "bg_promote_failed_allocation",
    "bg_promote_failed_copy",
    "bg_promote_failed_unmap",
    "bg_promote_failed_map",
    "bg_promote_failed_block_dead",
    "sync_fallback_pages",
    "bg_promote_delayed_schedules",
    "bg_promote_immediate_schedules",
    "bg_promote_timer_fires",
]

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

def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return 0.0

def i(row, key):
    try:
        return int(float(row[key]))
    except Exception:
        return 0

def median(values):
    values = list(values)
    return stats.median(values) if values else 0.0

before = read_stats(before_path)
after = read_stats(after_path)
deltas = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
delta_path.write_text("".join(f"{key} {deltas[key]}\n" for key in keys))

rows = list(csv.DictReader(csv_path.open(newline="")))
case_rows = list(csv.DictReader(case_stats_path.open(newline="")))

profiles = list(OrderedDict((row["profile"], None) for row in rows).keys())
by_profile = defaultdict(list)
for row in rows:
    by_profile[row["profile"]].append(row)

perf = {}
for profile in profiles:
    profile_rows = by_profile[profile]
    last_pass = max(i(r, "pass") for r in profile_rows)
    first_rows = [r for r in profile_rows if i(r, "pass") == 0]
    last_rows = [r for r in profile_rows if i(r, "pass") == last_pass]
    perf[profile] = {
        "runs": len({i(r, "repeat") for r in profile_rows}),
        "ops": i(profile_rows[0], "ops_per_page"),
        "pass0_ms": median(f(r, "wall_ms") for r in first_rows),
        "last_ms": median(f(r, "wall_ms") for r in last_rows),
        "median_ms": median(f(r, "wall_ms") for r in profile_rows),
        "gib_s_last": median(f(r, "gib_s") for r in last_rows),
    }

case_order = list(OrderedDict((row["case"], None) for row in case_rows).keys())
case_totals = {}
for case in case_order:
    selected = [row for row in case_rows if row["case"] == case]
    case_totals[case] = {
        "profile": selected[0]["profile"],
        "remote_first_enable": selected[0]["remote_first_enable"],
        "bg_promote_enable": selected[0]["bg_promote_enable"],
        "ops": selected[0]["ops_per_page"],
    }
    for key in keys:
        case_totals[case][key] = sum(i(row, f"{key}_delta") for row in selected)

lines = []
lines.append("# Remote-First Background Promotion Counter Probe")
lines.append("")
lines.append(f"- Raw CSV: `{csv_path}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append(f"- Counter before: `{before_path}`")
lines.append(f"- Counter after: `{after_path}`")
lines.append(f"- Counter delta: `{delta_path}`")
lines.append(f"- Per-case counter delta: `{case_stats_path}`")
lines.append("")
lines.append("## Total Counter Delta")
lines.append("")
lines.append("| Counter | Delta |")
lines.append("|---|---:|")
for key in keys:
    lines.append(f"| `{key}` | {deltas[key]} |")

lines.append("")
lines.append("## Per-Case Counter Delta")
lines.append("")
lines.append("| Case | Profile | RF | BG | Ops/page | RF faults | RF pages | Remote map pages | BG enqueued | Demand enq | Prefetch enq | BG copied | BG committed | PTE switch | BG cancelled |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for case in case_order:
    s = case_totals[case]
    lines.append(
        f"| `{case}` | `{s['profile']}` | {s['remote_first_enable']} | {s['bg_promote_enable']} | {s['ops']} | "
        f"{s['remote_first_faults']} | {s['remote_first_pages']} | {s['remote_map_pages']} | "
        f"{s['bg_promote_enqueued_pages']} | {s['bg_promote_demand_pages_enqueued']} | "
        f"{s['bg_promote_prefetch_pages_enqueued']} | {s['bg_promote_copied_pages']} | "
        f"{s['bg_promote_committed_pages']} | {s['pte_switch_pages']} | {s['bg_promote_cancelled_pages']} |"
    )

lines.append("")
lines.append("## Per-Case Worker And Failure Detail")
lines.append("")
lines.append("| Case | Demand promoted | Prefetch promoted | Worker rounds | Delayed schedules | Immediate schedules | Timer fires | Retries | Not CPU resident | Already GPU resident | Alloc fail | Copy fail | Unmap fail | Map fail | Dead block | Sync fallback pages |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for case in case_order:
    s = case_totals[case]
    lines.append(
        f"| `{case}` | {s['bg_promote_demand_pages_promoted']} | {s['bg_promote_prefetch_pages_promoted']} | "
        f"{s['bg_promote_worker_rounds']} | {s['bg_promote_delayed_schedules']} | "
        f"{s['bg_promote_immediate_schedules']} | {s['bg_promote_timer_fires']} | "
        f"{s['bg_promote_retry_count']} | "
        f"{s['bg_promote_failed_not_cpu_resident']} | {s['bg_promote_failed_already_gpu_resident']} | "
        f"{s['bg_promote_failed_allocation']} | {s['bg_promote_failed_copy']} | "
        f"{s['bg_promote_failed_unmap']} | {s['bg_promote_failed_map']} | "
        f"{s['bg_promote_failed_block_dead']} | {s['sync_fallback_pages']} |"
    )

lines.append("")
lines.append("## Performance")
lines.append("")
lines.append("| Profile | Runs | Ops/page | Pass0 ms | Last-pass ms | Median ms | Last-pass GiB/s |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for profile in profiles:
    s = perf[profile]
    lines.append(
        f"| `{profile}` | {s['runs']} | {s['ops']} | {s['pass0_ms']:.3f} | "
        f"{s['last_ms']:.3f} | {s['median_ms']:.3f} | {s['gib_s_last']:.3f} |"
    )

lines.append("")
lines.append("## Interpretation")
lines.append("")

rf_bg = case_totals.get("remote_first_bg")
rf_only = case_totals.get("remote_first_only")
rd_sync = perf.get("readonly_dup_sync")
counter = perf.get("counter_migrate")
default = perf.get("default_uvm")
local = perf.get("local_prefetch")
rf_bg_perf = perf.get("remote_first_bg")
rf_only_perf = perf.get("remote_first_only")

if rf_bg:
    lines.append("- `remote_first_bg` should exercise the new full path: remote-first fault service plus background COPY/read-dup promotion.")
    if rf_bg["remote_first_faults"] > 0 and rf_bg["bg_promote_committed_pages"] > 0:
        lines.append("  Observed: both remote-first and committed background promotion counters increased.")
    else:
        lines.append("  Warning: expected remote-first/background counters did not both increase.")
    if rf_bg["bg_promote_delayed_schedules"] > 0:
        lines.append("  Observed: delayed background promotion scheduling was used.")

if rf_only:
    lines.append("- `remote_first_only` disables background promotion to isolate foreground remote-first latency.")
    if rf_only["remote_first_faults"] > 0 and rf_only["bg_promote_enqueued_pages"] == 0:
        lines.append("  Observed: remote-first counters increased while background counters stayed zero.")

if rd_sync and rf_bg_perf:
    lines.append(f"- Native read-dup first pass vs remote-first+BG first pass: `{rd_sync['pass0_ms']:.3f} ms` vs `{rf_bg_perf['pass0_ms']:.3f} ms`.")

if rf_only_perf and rf_bg_perf:
    lines.append(f"- Remote-first only first pass vs remote-first+BG first pass: `{rf_only_perf['pass0_ms']:.3f} ms` vs `{rf_bg_perf['pass0_ms']:.3f} ms`.")

if counter:
    lines.append(f"- `counter_migrate` uses CPU preferred + AccessedBy with `{counter['ops']}` ops/page to test access-counter driven promotion/migration behavior.")

if default and rf_bg_perf:
    lines.append(f"- Default UVM first pass vs remote-first+BG first pass: `{default['pass0_ms']:.3f} ms` vs `{rf_bg_perf['pass0_ms']:.3f} ms`.")

if local and rf_bg_perf:
    lines.append(f"- Local prefetch last pass vs remote-first+BG last pass: `{local['last_ms']:.3f} ms` vs `{rf_bg_perf['last_ms']:.3f} ms`.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

reload_uvm_if_requested
check_runtime_params
reset_counters_if_requested
write_metadata
build_probe

echo
echo "Counter snapshot before:"
capture_stats "${STATS_BEFORE}"

run_probe_matrix

echo
echo "Counter snapshot after:"
capture_stats "${STATS_AFTER}"

summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Raw CSV: ${CSV}"
echo "Summary: ${SUMMARY}"
echo "Counter delta: ${STATS_DELTA}"
echo "Per-case counter delta: ${CASE_STATS_CSV}"
