#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

PAGES_VALUES="${PAGES_VALUES:-4096 16384 65536 262144}"
PASSES="${PASSES:-2}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first readonly_dup direct_migrate}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/prepare_remote_bg_scale_sweep_$(date -u +%Y%m%d_%H%M%S)}"

SUMMARY="${RUN_ROOT}/scale_sweep_summary.md"
META="${RUN_ROOT}/metadata.txt"

COUNTER_KEYS=(
  remote_prepare_mapped_pages
  bg_promote_enqueued_pages
  bg_promote_copied_pages
  bg_promote_committed_pages
  pte_switch_pages
  bg_promote_worker_rounds
  bg_promote_retry_count
  bg_promote_cancelled_pages
  bg_promote_delayed_schedules
  bg_promote_immediate_schedules
  bg_promote_timer_fires
  bg_promote_deferred_pages
  bg_promote_start_pages
)

mkdir -p "${RUN_ROOT}"

require_uvm() {
  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi
}

reset_counters() {
  [[ -e "${STATS_RESET_PATH}" ]] || return 0
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

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "pages_values=${PAGES_VALUES}"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "mechanisms=${MECHANISMS}"
  } | tee "${META}"
}

run_case() {
  local pages="$1"
  local child_root="${RUN_ROOT}/pages_${pages}"

  mkdir -p "${child_root}"
  reset_counters
  cat "${STATS_PATH}" > "${child_root}/latency_stats_before.txt"

  echo
  echo "Running pages=${pages} data_mib=$((pages * 4096 / 1024 / 1024))"
  RUN_ROOT="${child_root}" \
  PAGES="${pages}" \
  PASSES="${PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${BG_MAX_PAGES}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  MECHANISMS="${MECHANISMS}" \
  DISABLE_REMOTE_FIRST=1 \
  "${ROOT_DIR}/scripts/test_readonly_access_mechanisms.sh"

  cat "${STATS_PATH}" > "${child_root}/latency_stats_after.txt"
  {
    printf 'counter,delta\n'
    for key in "${COUNTER_KEYS[@]}"; do
      local before after
      before="$(counter_value "${child_root}/latency_stats_before.txt" "${key}")"
      after="$(counter_value "${child_root}/latency_stats_after.txt" "${key}")"
      printf '%s,%s\n' "${key}" "$((after - before))"
    done
  } > "${child_root}/counter_delta.csv"
}

summarize() {
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

def deltas(path):
    if not path.exists():
        return {}
    return {row["counter"]: int(row["delta"]) for row in csv.DictReader(path.open(newline=""))}

rows = []
for child in sorted(run_root.glob("pages_*"), key=lambda p: int(p.name.split("_")[1])):
    raw_path = child / "readonly_access_mechanisms_raw.csv"
    if not raw_path.exists():
        continue
    pages = int(child.name.split("_")[1])
    data_mib = pages * 4096 / (1024 * 1024)
    raw = list(csv.DictReader(raw_path.open(newline="")))
    delta = deltas(child / "counter_delta.csv")
    for mech in sorted({row["mechanism"] for row in raw}):
        selected = [row for row in raw if row["mechanism"] == mech]
        last_pass = max(i(row, "pass") for row in selected)
        rows.append({
            "data_mib": data_mib,
            "mech": mech,
            "setup": median(f(row, "setup_ms") for row in selected),
            "prepare": median(f(row, "prepare_ms") for row in selected),
            "prepared_mib": median(f(row, "prepare_mapped_mib") for row in selected),
            "pass0": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 0),
            "pass1": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 1),
            "last": median(f(row, "wall_ms") for row in selected if i(row, "pass") == last_pass),
            "committed": delta.get("bg_promote_committed_pages", 0),
            "rounds": delta.get("bg_promote_worker_rounds", 0),
            "retries": delta.get("bg_promote_retry_count", 0),
            "delayed": delta.get("bg_promote_delayed_schedules", 0),
            "immediate": delta.get("bg_promote_immediate_schedules", 0),
            "timer": delta.get("bg_promote_timer_fires", 0),
        })

lines = []
lines.append("# Prepare-Remote-BG Scale Sweep")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("| Data MiB | Mechanism | Setup ms | Prepare ms | Prepared MiB | Pass0 ms | Pass1 ms | Last ms | BG committed pages | Worker rounds | Delayed schedules | Immediate schedules | Timer fires | Retries |")
lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['data_mib']:.0f} | `{row['mech']}` | {row['setup']:.3f} | {row['prepare']:.3f} | "
        f"{row['prepared_mib']:.1f} | {row['pass0']:.3f} | {row['pass1']:.3f} | {row['last']:.3f} | "
        f"{row['committed']} | {row['rounds']} | {row['delayed']} | {row['immediate']} | {row['timer']} | {row['retries']} |"
    )

lines.append("")
lines.append("## How To Read This")
lines.append("")
lines.append("- `readmostly_prepare_remote` should scale like remote access in both passes.")
lines.append("- `readmostly_prepare_remote_bg` should keep pass0 near remote while pass1/last approaches `readonly_dup` if promotion finishes.")
lines.append("- Watch prepare time, worker rounds, and retries for non-linear growth at 256 MiB and 1 GiB.")
lines.append("- Non-zero `BG_DELAY_MS` / `BG_RESCHEDULE_DELAY_MS` should reduce prepare-time copy pressure at the cost of slower convergence to local reads.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

require_uvm
write_metadata
for pages in ${PAGES_VALUES}; do
  run_case "${pages}"
done
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
