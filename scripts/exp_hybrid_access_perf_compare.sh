#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/hybrid_access_perf_compare_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY="${RUN_ROOT}/hybrid_access_perf_compare_summary.md"
META="${RUN_ROOT}/metadata.txt"

DEVICE="${DEVICE:-0}"
REPEATS="${REPEATS:-3}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE:-512}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-1}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"

RUN_MAIN="${RUN_MAIN:-1}"
MAIN_PAGES="${MAIN_PAGES:-16384}"
MAIN_PASSES="${MAIN_PASSES:-8}"
MAIN_MECHANISMS="${MAIN_MECHANISMS:-uvm_pure_remote uva_pinned_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first hybrid_remote_first_only hybrid_remote_first_bg readonly_dup direct_migrate counter_migrate}"

RUN_SCALE="${RUN_SCALE:-1}"
SCALE_PAGES_VALUES="${SCALE_PAGES_VALUES:-4096 16384 65536 131072}"
SCALE_PASSES="${SCALE_PASSES:-4}"
SCALE_MECHANISMS="${SCALE_MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first hybrid_remote_first_bg readonly_dup direct_migrate}"

RUN_BG_SWEEP="${RUN_BG_SWEEP:-1}"
BG_SWEEP_PAGES="${BG_SWEEP_PAGES:-131072}"
BG_SWEEP_PASSES="${BG_SWEEP_PASSES:-4}"
BG_MAX_VALUES="${BG_MAX_VALUES:-16 64 512}"
BG_SWEEP_MECHANISMS="${BG_SWEEP_MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first hybrid_remote_first_bg readonly_dup}"

mkdir -p "${RUN_ROOT}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing ${path}" >&2
    exit 1
  fi
}

require_modified_uvm() {
  require_file "${ROOT_DIR}/scripts/test_readonly_access_mechanisms.sh"
  require_file "${ROOT_DIR}/scripts/exp_prepare_remote_bg_scale_sweep.sh"
  require_file "${ROOT_DIR}/scripts/exp_prepare_remote_bg_interference_sweep.sh"
  require_file "${ROOT_DIR}/tools/uvm_readonly_access_mechanism_bench.cu"

  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi

  for param in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable \
    uvm_latency_bg_promote_max_pages \
    uvm_latency_bg_promote_delay_ms \
    uvm_latency_bg_promote_reschedule_delay_ms; do
    if [[ ! -e "${PARAM_ROOT}/${param}" ]]; then
      echo "ERROR: ${PARAM_ROOT}/${param} is missing. This does not look like the modified UVM module." >&2
      exit 1
    fi
  done
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "device=${DEVICE}"
    echo "repeats=${REPEATS}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "counter_ops_per_page=${COUNTER_OPS_PER_PAGE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "phase_f_route_log_delay_ms=${PHASE_F_ROUTE_LOG_DELAY_MS}"
    echo "run_main=${RUN_MAIN}"
    echo "main_pages=${MAIN_PAGES}"
    echo "main_passes=${MAIN_PASSES}"
    echo "main_mechanisms=${MAIN_MECHANISMS}"
    echo "run_scale=${RUN_SCALE}"
    echo "scale_pages_values=${SCALE_PAGES_VALUES}"
    echo "scale_passes=${SCALE_PASSES}"
    echo "scale_mechanisms=${SCALE_MECHANISMS}"
    echo "run_bg_sweep=${RUN_BG_SWEEP}"
    echo "bg_sweep_pages=${BG_SWEEP_PAGES}"
    echo "bg_sweep_passes=${BG_SWEEP_PASSES}"
    echo "bg_max_values=${BG_MAX_VALUES}"
    echo "bg_sweep_mechanisms=${BG_SWEEP_MECHANISMS}"
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

run_main_comparison() {
  local child_root="${RUN_ROOT}/main_comparison"

  echo
  echo "== Main comparison: pages=${MAIN_PAGES}, passes=${MAIN_PASSES} =="
  RUN_ROOT="${child_root}" \
  PAGES="${MAIN_PAGES}" \
  PASSES="${MAIN_PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${BG_MAX_PAGES}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS}" \
  MECHANISMS="${MAIN_MECHANISMS}" \
  DISABLE_REMOTE_FIRST=1 \
  "${ROOT_DIR}/scripts/test_readonly_access_mechanisms.sh"
}

run_scale_sweep() {
  local child_root="${RUN_ROOT}/scale_sweep"

  echo
  echo "== Scale sweep: pages=${SCALE_PAGES_VALUES}, passes=${SCALE_PASSES} =="
  RUN_ROOT="${child_root}" \
  PAGES_VALUES="${SCALE_PAGES_VALUES}" \
  PASSES="${SCALE_PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${BG_MAX_PAGES}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  MECHANISMS="${SCALE_MECHANISMS}" \
  "${ROOT_DIR}/scripts/exp_prepare_remote_bg_scale_sweep.sh"
}

run_bg_sweep() {
  local child_root="${RUN_ROOT}/bg_sweep"

  echo
  echo "== Background sweep: pages=${BG_SWEEP_PAGES}, BG_MAX_VALUES=${BG_MAX_VALUES} =="
  RUN_ROOT="${child_root}" \
  BG_MAX_VALUES="${BG_MAX_VALUES}" \
  PAGES="${BG_SWEEP_PAGES}" \
  PASSES="${BG_SWEEP_PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  MECHANISMS="${BG_SWEEP_MECHANISMS}" \
  "${ROOT_DIR}/scripts/exp_prepare_remote_bg_interference_sweep.sh"
}

write_summary() {
  python3 - "${RUN_ROOT}" "${SUMMARY}" "${META}" <<'PY'
import csv
import statistics as stats
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])

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

def summarize_raw(raw_path):
    if not raw_path.exists():
        return []
    raw = list(csv.DictReader(raw_path.open(newline="")))
    out = []
    for mech in dict.fromkeys(row["mechanism"] for row in raw):
        selected = [row for row in raw if row["mechanism"] == mech]
        last_pass = max(i(row, "pass") for row in selected)
        out.append({
            "mech": mech,
            "bytes": i(selected[0], "bytes"),
            "ops": i(selected[0], "ops_per_page"),
            "prepare": median(f(row, "prepare_ms") for row in selected),
            "prepared": median(f(row, "prepare_mapped_mib") for row in selected),
            "pass0": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 0),
            "pass1": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 1),
            "last": median(f(row, "wall_ms") for row in selected if i(row, "pass") == last_pass),
        })
    return out

main_raw = run_root / "main_comparison" / "readonly_access_mechanisms_raw.csv"
main_summary = run_root / "main_comparison" / "readonly_access_mechanisms_summary.md"
scale_summary = run_root / "scale_sweep" / "scale_sweep_summary.md"
bg_summary = run_root / "bg_sweep" / "interference_sweep_summary.md"
main_rows = summarize_raw(main_raw)

lines = []
lines.append("# Hybrid Access Performance Comparison")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Outputs")
lines.append("")
if main_summary.exists():
    lines.append(f"- Main comparison summary: `{main_summary}`")
    lines.append(f"- Main raw CSV: `{main_raw}`")
if scale_summary.exists():
    lines.append(f"- Scale sweep summary: `{scale_summary}`")
if bg_summary.exists():
    lines.append(f"- Background interference summary: `{bg_summary}`")
lines.append("")

if main_rows:
    lines.append("## Main Comparison Medians")
    lines.append("")
    lines.append("| Mechanism | Data MiB | Ops/page | Prepare ms | Prepared MiB | Pass0 ms | Pass1 ms | Last ms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in main_rows:
        lines.append(
            f"| `{row['mech']}` | {row['bytes'] / (1024 * 1024):.1f} | {row['ops']} | "
            f"{row['prepare']:.3f} | {row['prepared']:.1f} | {row['pass0']:.3f} | "
            f"{row['pass1']:.3f} | {row['last']:.3f} |"
        )
    lines.append("")

lines.append("## Expected Signals")
lines.append("")
lines.append("- `uvm_pure_remote` and `uva_pinned_remote` should be close. This verifies that prepared UVM remote mapping is not slower than vLLM-style pinned UVA remote read.")
lines.append("- `readmostly_prepare_remote` should have pass0 close to pure remote and later passes still remote-speed, because it only prepares remote READ PTEs.")
lines.append("- `readmostly_prepare_remote_bg` should show pass0 near remote and later passes near `readonly_dup`/local if background promotion commits in time.")
lines.append("- `readmostly_prepare_remote_bg_after_first` is the stricter version: pass0 should be closest to pure remote because the worker starts after pass0.")
lines.append("- `hybrid_remote_first_only` shows the cost of building remote READ mappings inside replayable fault service. If it is much slower than `readmostly_prepare_remote`, the fault path is still the bottleneck.")
lines.append("- `hybrid_remote_first_bg` should improve over synchronous `readonly_dup` on pass0, and converge to local speed later if background promotion succeeds.")
lines.append("- `readonly_dup` and `direct_migrate` are synchronization-heavy baselines: pass0 is expensive, later passes should become local.")
lines.append("- In scale and BG sweeps, rising pass0 for background variants means copy/PTE switch is interfering with remote reads; rising prepare time means work leaked into the prepare phase.")
lines.append("")
lines.append("## Useful Overrides")
lines.append("")
lines.append("```bash")
lines.append("# Add 1 GiB to scale sweep")
lines.append("SCALE_PAGES_VALUES='16384 65536 131072 262144' ./scripts/exp_hybrid_access_perf_compare.sh")
lines.append("")
lines.append("# Only run the main comparison")
lines.append("RUN_SCALE=0 RUN_BG_SWEEP=0 ./scripts/exp_hybrid_access_perf_compare.sh")
lines.append("")
lines.append("# Focus on delayed after-first promotion")
lines.append("MAIN_MECHANISMS='uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg_after_first readonly_dup' ./scripts/exp_hybrid_access_perf_compare.sh")
lines.append("```")
lines.append("")

summary_path.write_text("\n".join(lines))
print(summary_path)
PY
}

require_modified_uvm
write_metadata

if [[ "${RUN_MAIN}" == "1" ]]; then
  run_main_comparison
fi

if [[ "${RUN_SCALE}" == "1" ]]; then
  run_scale_sweep
fi

if [[ "${RUN_BG_SWEEP}" == "1" ]]; then
  run_bg_sweep
fi

write_summary

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
