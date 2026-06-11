#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NVCC_ARCH="${NVCC_ARCH:--arch=native}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

if [[ ! -x "${NVCC}" ]]; then
  NVCC="$(command -v nvcc)"
fi

PAGES="${PAGES:-8192}"
PASSES="${PASSES:-2}"
OPS_PER_PAGE="${OPS_PER_PAGE:-128}"
COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE:-512}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote uva_pinned_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first hybrid_remote_first_only hybrid_remote_first_bg readonly_dup direct_migrate counter_migrate}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/readonly_access_mechanisms_$(date -u +%Y%m%d_%H%M%S)}"

# Isolate the new remote-first/background promotion path: enabled only for
# hybrid_remote_first_bg and disabled for all other mechanisms.
DISABLE_REMOTE_FIRST="${DISABLE_REMOTE_FIRST:-1}"
RESTORE_PARAMS="${RESTORE_PARAMS:-1}"

SRC="${ROOT_DIR}/tools/uvm_readonly_access_mechanism_bench.cu"
BIN="${RUN_ROOT}/uvm_readonly_access_mechanism_bench"
CSV="${RUN_ROOT}/readonly_access_mechanisms_raw.csv"
SUMMARY="${RUN_ROOT}/readonly_access_mechanisms_summary.md"
META="${RUN_ROOT}/metadata.txt"

mkdir -p "${RUN_ROOT}"

declare -A OLD_PARAMS=()
ACCESS_COUNTER_CONTROL_WARNED=0

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

param_path() {
  printf '%s/%s' "${PARAM_ROOT}" "$1"
}

read_param() {
  local name="$1"
  local path
  path="$(param_path "${name}")"
  [[ -r "${path}" ]] || return 1
  cat "${path}"
}

write_param() {
  local name="$1"
  local value="$2"
  local path
  path="$(param_path "${name}")"
  [[ -e "${path}" ]] || return 0

  local current
  current="$(read_param "${name}" 2>/dev/null || true)"
  if [[ "${current}" == "${value}" ]]; then
    return 0
  fi

  if [[ ! -w "${path}" && "${EUID}" -ne 0 ]]; then
    if ! echo "${value}" | sudo tee "${path}" >/dev/null; then
      echo "WARNING: could not write ${name}=${value}; continuing with current value $(read_param "${name}" 2>/dev/null || echo unknown)" >&2
      return 0
    fi
  else
    if ! { echo "${value}" > "${path}"; } 2>/dev/null; then
      echo "WARNING: could not write ${name}=${value}; continuing with current value $(read_param "${name}" 2>/dev/null || echo unknown)" >&2
      return 0
    fi
  fi
}

save_param() {
  local name="$1"
  if [[ -n "${OLD_PARAMS[$name]+set}" ]]; then
    return
  fi
  if value="$(read_param "${name}")"; then
    OLD_PARAMS["${name}"]="${value}"
  fi
}

restore_params() {
  if [[ "${RESTORE_PARAMS}" != "1" ]]; then
    return
  fi
  for name in "${!OLD_PARAMS[@]}"; do
    write_param "${name}" "${OLD_PARAMS[${name}]}" || true
  done
}

trap restore_params EXIT

disable_remote_first_path() {
  if [[ "${DISABLE_REMOTE_FIRST}" != "1" ]]; then
    return
  fi

  for name in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable; do
    save_param "${name}"
    write_param "${name}" 0
  done
}

set_remote_first_for_mechanism() {
  local mechanism="$1"

  if [[ "${DISABLE_REMOTE_FIRST}" != "1" ]]; then
    return
  fi

  for name in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable \
    uvm_latency_bg_promote_max_pages \
    uvm_latency_bg_promote_delay_ms \
    uvm_latency_bg_promote_reschedule_delay_ms \
    uvm_phase_f_route_log_delay_ms; do
    save_param "${name}"
  done

  case "${mechanism}" in
    hybrid_remote_first_only)
      write_param uvm_latency_remote_first_enable 1
      write_param uvm_latency_bg_promote_enable 0
      write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
      write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
      write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"
      write_param uvm_phase_f_route_log_delay_ms "${PHASE_F_ROUTE_LOG_DELAY_MS}"
      ;;
    hybrid_remote_first_bg)
      write_param uvm_latency_remote_first_enable 1
      write_param uvm_latency_bg_promote_enable 1
      write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
      write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
      write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"
      write_param uvm_phase_f_route_log_delay_ms "${PHASE_F_ROUTE_LOG_DELAY_MS}"
      ;;
    *)
      write_param uvm_latency_remote_first_enable 0
      write_param uvm_latency_bg_promote_enable 0
      write_param uvm_latency_bg_promote_max_pages "${BG_MAX_PAGES}"
      write_param uvm_latency_bg_promote_delay_ms "${BG_DELAY_MS}"
      write_param uvm_latency_bg_promote_reschedule_delay_ms "${BG_RESCHEDULE_DELAY_MS}"
      ;;
  esac
}

set_access_counter_for_mechanism() {
  local mechanism="$1"
  local name="uvm_perf_access_counter_migration_enable"
  local path
  path="$(param_path "${name}")"
  [[ -e "${path}" ]] || return

  if [[ "$(stat -c %A "${path}")" != *w* ]]; then
    if [[ "${ACCESS_COUNTER_CONTROL_WARNED}" == "0" ]]; then
      echo "WARNING: ${name} is not writable; using currently loaded value $(read_param "${name}" 2>/dev/null || echo unknown)." >&2
      echo "         For uvm_pure_remote, keep OPS_PER_PAGE * PASSES below the access-counter threshold if you need a stricter remote-only run." >&2
      ACCESS_COUNTER_CONTROL_WARNED=1
    fi
    return
  fi

  save_param "${name}"
  case "${mechanism}" in
    counter_migrate)
      write_param "${name}" "${ACCESS_COUNTER_ENABLE_FOR_COUNTER:-${OLD_PARAMS[${name}]:--1}}"
      ;;
    *)
      # Keep uvm_pure_remote remote instead of letting access counters promote it.
      write_param "${name}" 0
      ;;
  esac
}

ops_for_mechanism() {
  case "$1" in
    counter_migrate) echo "${COUNTER_OPS_PER_PAGE}" ;;
    *) echo "${OPS_PER_PAGE}" ;;
  esac
}

build_probe() {
  echo "Building ${BIN}"
  "${NVCC}" -O3 -std=c++17 ${NVCC_ARCH} \
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
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "counter_ops_per_page=${COUNTER_OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "phase_f_route_log_delay_ms=${PHASE_F_ROUTE_LOG_DELAY_MS}"
    echo "mechanisms=${MECHANISMS}"
    echo "disable_remote_first=${DISABLE_REMOTE_FIRST}"
    echo "restore_params=${RESTORE_PARAMS}"
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

run_matrix() {
  local first=1

  for repeat in $(seq 0 $((REPEATS - 1))); do
    for mechanism in ${MECHANISMS}; do
      local ops
      ops="$(ops_for_mechanism "${mechanism}")"

      echo "Running repeat=${repeat} mechanism=${mechanism} ops_per_page=${ops}"
      set_remote_first_for_mechanism "${mechanism}"
      set_access_counter_for_mechanism "${mechanism}"

      local args=(
        --mechanism "${mechanism}"
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
    done
  done
}

summarize() {
  python3 - "${CSV}" "${SUMMARY}" "${META}" <<'PY'
import csv
import statistics as stats
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

csv_path = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])

rows = list(csv.DictReader(csv_path.open(newline="")))

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

by_mech = defaultdict(list)
for row in rows:
    by_mech[row["mechanism"]].append(row)

order = list(OrderedDict((row["mechanism"], None) for row in rows).keys())
summary = {}
for mech in order:
    selected = by_mech[mech]
    last_pass = max(i(r, "pass") for r in selected)
    pass0 = [r for r in selected if i(r, "pass") == 0]
    pass1 = [r for r in selected if i(r, "pass") == 1]
    last = [r for r in selected if i(r, "pass") == last_pass]
    summary[mech] = {
        "runs": len({i(r, "repeat") for r in selected}),
        "ops": i(selected[0], "ops_per_page"),
        "bytes": i(selected[0], "bytes"),
        "setup_ms": median(f(r, "setup_ms") for r in selected),
        "prepare_ms": median(f(r, "prepare_ms") for r in selected),
        "prepare_mapped_mib": median(f(r, "prepare_mapped_mib") for r in selected),
        "pass0_ms": median(f(r, "wall_ms") for r in pass0),
        "pass1_ms": median(f(r, "wall_ms") for r in pass1) if pass1 else 0.0,
        "last_ms": median(f(r, "wall_ms") for r in last),
        "pass0_gib_s": median(f(r, "gib_s") for r in pass0),
        "last_gib_s": median(f(r, "gib_s") for r in last),
    }

remote0 = summary.get("uvm_pure_remote", {}).get("pass0_ms", 0.0)
remote_last = summary.get("uvm_pure_remote", {}).get("last_ms", 0.0)

lines = []
lines.append("# Read-Only Access Mechanism Comparison")
lines.append("")
lines.append(f"- Raw CSV: `{csv_path}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Mechanisms")
lines.append("")
lines.append("| Mechanism | Meaning |")
lines.append("|---|---|")
lines.append("| `uvm_pure_remote` | Managed memory, CPU preferred + GPU AccessedBy. The script tries to disable access-counter migration for this case; if the module parameter is read-only, keep `OPS_PER_PAGE * PASSES` below the threshold for a stricter remote-only run. |")
lines.append("| `uva_pinned_remote` | vLLM-style pinned CPU tensor exposed to GPU with UVA / `cudaHostGetDevicePointer`. No UVM managed page migration. |")
lines.append("| `readmostly_prepare_remote` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark manually calls Phase F `PREPARE_REMOTE_READ` to pre-build GPU->CPU READ mappings outside the fault path. |")
lines.append("| `readmostly_prepare_remote_bg` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark manually calls Phase F `PREPARE_REMOTE_READ_BG`, which pre-builds GPU->CPU READ mappings and enqueues background read-dup promotion. |")
lines.append("| `readmostly_prepare_remote_bg_after_first` | Managed memory initially CPU-resident, `ReadMostly`; before the first read the benchmark calls Phase F `PREPARE_REMOTE_READ_BG_DEFERRED`, which pre-builds GPU->CPU READ mappings and only marks promotion pending. After pass0 completes it calls `START_BG_PROMOTION` to launch the worker. |")
lines.append("| `hybrid_remote_first_only` | Managed memory initially CPU-resident, `ReadMostly`; script enables modified driver remote-first READ mapping and disables background promotion. |")
lines.append("| `hybrid_remote_first_bg` | Managed memory initially CPU-resident, `ReadMostly`; script enables the modified driver remote-first READ mapping plus background read-dup promotion for this case only. |")
lines.append("| `readonly_dup` | Managed memory initially CPU-resident, `cudaMemAdviseSetReadMostly`; first GPU read may build a GPU read duplicate. |")
lines.append("| `direct_migrate` | Managed memory initially CPU-resident with no advice; normal on-demand migration on first GPU read. |")
lines.append("| `counter_migrate` | Managed memory CPU preferred + GPU AccessedBy with high ops/page to trigger access-counter behavior. |")
lines.append("")
lines.append("## Performance")
lines.append("")
lines.append("| Mechanism | Runs | Ops/page | Data MiB | Setup ms | Prepare ms | Prepared MiB | First pass ms | Second/last pass ms | First GiB/s | Last GiB/s | First vs pure remote | Last vs pure remote |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for mech in order:
    s = summary[mech]
    data_mib = s["bytes"] / (1024 * 1024)
    first_speed = remote0 / s["pass0_ms"] if remote0 > 0 and s["pass0_ms"] > 0 else 0.0
    last_speed = remote_last / s["last_ms"] if remote_last > 0 and s["last_ms"] > 0 else 0.0
    lines.append(
        f"| `{mech}` | {s['runs']} | {s['ops']} | {data_mib:.1f} | "
        f"{s['setup_ms']:.3f} | {s['prepare_ms']:.3f} | {s['prepare_mapped_mib']:.1f} | "
        f"{s['pass0_ms']:.3f} | {s['last_ms']:.3f} | "
        f"{s['pass0_gib_s']:.3f} | {s['last_gib_s']:.3f} | "
        f"{first_speed:.2f}x | {last_speed:.2f}x |"
    )

lines.append("")
lines.append("## How To Read This")
lines.append("")
lines.append("- Compare `uvm_pure_remote` and `uva_pinned_remote` to separate UVM remote mapping overhead from vLLM-style pinned UVA remote reads.")
lines.append("- Compare `readmostly_prepare_remote` with `uvm_pure_remote` and `hybrid_remote_first_only` to test whether pre-building ReadMostly remote READ mappings outside the fault path removes the expensive first-touch mapping cost.")
lines.append("- Compare `readmostly_prepare_remote_bg` with `readmostly_prepare_remote` to isolate the extra prepare-time enqueue and background read-dup promotion effect.")
lines.append("- Compare `readmostly_prepare_remote_bg_after_first` with `readmostly_prepare_remote_bg` to test whether deferring worker start until after pass0 protects first-pass remote-read latency from background copy interference.")
lines.append("- The target result for `readmostly_prepare_remote_bg` is first-pass latency close to `uvm_pure_remote`, then second/last-pass latency close to `readonly_dup` after the background promotion commits.")
lines.append("- The stricter target for `readmostly_prepare_remote_bg_after_first` is pass0 close to `uvm_pure_remote`; promotion should begin only after pass0 and later passes should converge toward `readonly_dup` as the worker commits pages.")
lines.append("- Compare `hybrid_remote_first_only` and `hybrid_remote_first_bg` to isolate the cost of background promotion.")
lines.append("- Compare `hybrid_remote_first_bg` with `uvm_pure_remote` for first-pass latency and with `readonly_dup` for second-pass local-read convergence.")
lines.append("- Compare first pass and second/last pass of `readonly_dup` and `direct_migrate` to see whether the expensive first touch turns into fast GPU-local reuse.")
lines.append("- `counter_migrate` intentionally uses `COUNTER_OPS_PER_PAGE`, so its ops/page may differ from the other rows; compare latency and bandwidth with that in mind.")
lines.append("- If `readonly_dup` or `direct_migrate` has a slow first pass but a much faster second pass, the mechanism paid migration/read-dup cost once and then used GPU-local data.")
lines.append("- If `uvm_pure_remote` remains slow on the second pass, that is the expected cost of repeatedly reading CPU-resident data over the interconnect.")
lines.append("")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

disable_remote_first_path
write_metadata
build_probe
run_matrix
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Raw CSV: ${CSV}"
echo "Summary: ${SUMMARY}"
echo "Metadata: ${META}"
