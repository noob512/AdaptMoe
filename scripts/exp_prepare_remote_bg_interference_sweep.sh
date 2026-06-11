#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

BG_MAX_VALUES="${BG_MAX_VALUES:-16 64 512 2048}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PAGES="${PAGES:-131072}"
PASSES="${PASSES:-3}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-0}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first readonly_dup}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/prepare_remote_bg_interference_sweep_$(date -u +%Y%m%d_%H%M%S)}"

SUMMARY="${RUN_ROOT}/interference_sweep_summary.md"
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
  [[ -e "${PARAM_ROOT}/uvm_latency_bg_promote_max_pages" ]] || {
    echo "ERROR: modified UVM module parameters are missing." >&2
    exit 1
  }
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
    echo "bg_max_values=${BG_MAX_VALUES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "pages=${PAGES}"
    echo "data_mib=$((PAGES * 4096 / 1024 / 1024))"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "mechanisms=${MECHANISMS}"
  } | tee "${META}"
}

run_case() {
  local bg_max="$1"
  local child_root="${RUN_ROOT}/bg_${bg_max}"

  mkdir -p "${child_root}"
  reset_counters
  cat "${STATS_PATH}" > "${child_root}/latency_stats_before.txt"

  echo
  echo "Running BG_MAX_PAGES=${bg_max}"
  RUN_ROOT="${child_root}" \
  PAGES="${PAGES}" \
  PASSES="${PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${bg_max}" \
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
for child in sorted(run_root.glob("bg_*"), key=lambda p: int(p.name.split("_")[1])):
    raw_path = child / "readonly_access_mechanisms_raw.csv"
    if not raw_path.exists():
        continue
    bg = int(child.name.split("_")[1])
    raw = list(csv.DictReader(raw_path.open(newline="")))
    delta = deltas(child / "counter_delta.csv")
    for mech in sorted({row["mechanism"] for row in raw}):
        selected = [row for row in raw if row["mechanism"] == mech]
        last_pass = max(i(row, "pass") for row in selected)
        rows.append({
            "bg": bg,
            "mech": mech,
            "prepare": median(f(row, "prepare_ms") for row in selected),
            "pass0": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 0),
            "pass1": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 1),
            "last": median(f(row, "wall_ms") for row in selected if i(row, "pass") == last_pass),
            "committed": delta.get("bg_promote_committed_pages", 0),
            "rounds": delta.get("bg_promote_worker_rounds", 0),
            "retries": delta.get("bg_promote_retry_count", 0),
            "cancelled": delta.get("bg_promote_cancelled_pages", 0),
            "delayed": delta.get("bg_promote_delayed_schedules", 0),
            "immediate": delta.get("bg_promote_immediate_schedules", 0),
            "timer": delta.get("bg_promote_timer_fires", 0),
        })

lines = []
lines.append("# Prepare-Remote-BG Background Copy Interference Sweep")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("| BG_MAX | Mechanism | Prepare ms | Pass0 ms | Pass1 ms | Last ms | BG committed pages | Worker rounds | Delayed schedules | Immediate schedules | Timer fires | Retries | Cancelled |")
lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['bg']} | `{row['mech']}` | {row['prepare']:.3f} | {row['pass0']:.3f} | "
        f"{row['pass1']:.3f} | {row['last']:.3f} | {row['committed']} | {row['rounds']} | "
        f"{row['delayed']} | {row['immediate']} | {row['timer']} | {row['retries']} | {row['cancelled']} |"
    )

lines.append("")
lines.append("## How To Read This")
lines.append("")
lines.append("- Compare `readmostly_prepare_remote_bg` pass0 across BG_MAX values; rising pass0 indicates background copy is interfering with urgent remote reads.")
lines.append("- Compare pass1/last against `readonly_dup`; slow later passes indicate promotion did not finish quickly enough.")
lines.append("- Worker rounds should rise when BG_MAX is small; retries/cancelled should stay near zero.")
lines.append("- `Delayed schedules` and `Timer fires` should increase when `BG_DELAY_MS` or `BG_RESCHEDULE_DELAY_MS` is non-zero; `Immediate schedules` should dominate when both are zero.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

require_uvm
write_metadata
for bg in ${BG_MAX_VALUES}; do
  run_case "${bg}"
done
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
