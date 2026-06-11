#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

PAGES_VALUES="${PAGES_VALUES:-131072 262144}"
BG_MAX_VALUES="${BG_MAX_VALUES:-16 32 64}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PASSES="${PASSES:-3}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
REPEATS="${REPEATS:-1}"
DEVICE="${DEVICE:-0}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.02}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first readonly_dup}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/prepare_remote_bg_remote_during_copy_$(date -u +%Y%m%d_%H%M%S)}"

SUMMARY="${RUN_ROOT}/remote_during_copy_summary.md"
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
  for p in uvm_latency_bg_promote_max_pages uvm_latency_remote_first_enable uvm_latency_bg_promote_enable; do
    if [[ ! -e "${PARAM_ROOT}/${p}" ]]; then
      echo "ERROR: ${PARAM_ROOT}/${p} is missing. The loaded nvidia_uvm is not the modified module." >&2
      exit 1
    fi
  done
}

reset_counters() {
  [[ -e "${STATS_RESET_PATH}" ]] || return 0
  if [[ ! -w "${STATS_RESET_PATH}" && "${EUID}" -ne 0 ]]; then
    echo 1 | sudo tee "${STATS_RESET_PATH}" >/dev/null
  else
    echo 1 > "${STATS_RESET_PATH}"
  fi
}

sample_counters() {
  local out="$1"
  printf 'unix_ms' > "${out}"
  for key in "${COUNTER_KEYS[@]}"; do
    printf ',%s' "${key}" >> "${out}"
  done
  printf '\n' >> "${out}"

  while true; do
    local now
    now="$(date +%s%3N)"
    printf '%s' "${now}" >> "${out}"
    for key in "${COUNTER_KEYS[@]}"; do
      local value
      value="$(awk -v key="${key}" '$1 == key { print $2; found = 1 } END { if (!found) print 0 }' "${STATS_PATH}")"
      printf ',%s' "${value}" >> "${out}"
    done
    printf '\n' >> "${out}"
    sleep "${SAMPLE_INTERVAL_SEC}"
  done
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "pages_values=${PAGES_VALUES}"
    echo "bg_max_values=${BG_MAX_VALUES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "sample_interval_sec=${SAMPLE_INTERVAL_SEC}"
    echo "mechanisms=${MECHANISMS}"
    echo
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
  } | tee "${META}"
}

run_case() {
  local pages="$1"
  local bg_max="$2"
  local child_root="${RUN_ROOT}/pages_${pages}_bg_${bg_max}"
  local samples="${child_root}/counter_samples.csv"

  mkdir -p "${child_root}"
  reset_counters

  echo
  echo "Running pages=${pages} bg_max=${bg_max}"
  sample_counters "${samples}" &
  local sampler_pid="$!"
  set +e
  RUN_ROOT="${child_root}" \
  PAGES="${pages}" \
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
  local status="$?"
  set -e
  kill "${sampler_pid}" 2>/dev/null || true
  wait "${sampler_pid}" 2>/dev/null || true
  cat "${STATS_PATH}" > "${child_root}/latency_stats_after.txt"
  return "${status}"
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

def max_sample(path, key):
    if not path.exists():
        return 0
    rows = list(csv.DictReader(path.open(newline="")))
    return max((i(row, key) for row in rows), default=0)

rows = []
for child in sorted(run_root.glob("pages_*_bg_*")):
    raw_path = child / "readonly_access_mechanisms_raw.csv"
    sample_path = child / "counter_samples.csv"
    if not raw_path.exists():
        continue
    parts = child.name.split("_")
    pages = int(parts[1])
    bg = int(parts[3])
    raw = list(csv.DictReader(raw_path.open(newline="")))
    for mech in sorted({row["mechanism"] for row in raw}):
        selected = [row for row in raw if row["mechanism"] == mech]
        last_pass = max(i(row, "pass") for row in selected)
        entry = {
            "pages": pages,
            "data_mib": pages * 4096 / (1024 * 1024),
            "bg": bg,
            "mechanism": mech,
            "prepare_ms": median(f(row, "prepare_ms") for row in selected),
            "pass0_ms": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 0),
            "pass1_ms": median(f(row, "wall_ms") for row in selected if i(row, "pass") == 1),
            "last_ms": median(f(row, "wall_ms") for row in selected if i(row, "pass") == last_pass),
            "committed_max": max_sample(sample_path, "bg_promote_committed_pages"),
            "copied_max": max_sample(sample_path, "bg_promote_copied_pages"),
            "switch_max": max_sample(sample_path, "pte_switch_pages"),
            "delayed_max": max_sample(sample_path, "bg_promote_delayed_schedules"),
            "immediate_max": max_sample(sample_path, "bg_promote_immediate_schedules"),
            "timer_max": max_sample(sample_path, "bg_promote_timer_fires"),
            "summary": child / "readonly_access_mechanisms_summary.md",
            "samples": sample_path,
        }
        rows.append(entry)

lines = []
lines.append("# Prepare-Remote-BG Remote-During-Copy Experiment")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Results")
lines.append("")
lines.append("| Data MiB | BG_MAX | Mechanism | Prepare ms | Pass0 ms | Pass1 ms | Last ms | Max copied pages | Max committed pages | Max PTE switch pages | Max delayed schedules | Max immediate schedules | Max timer fires |")
lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row['data_mib']:.0f} | {row['bg']} | `{row['mechanism']}` | "
        f"{row['prepare_ms']:.3f} | {row['pass0_ms']:.3f} | {row['pass1_ms']:.3f} | {row['last_ms']:.3f} | "
        f"{row['copied_max']} | {row['committed_max']} | {row['switch_max']} | "
        f"{row['delayed_max']} | {row['immediate_max']} | {row['timer_max']} |"
    )

lines.append("")
lines.append("## How To Read This")
lines.append("")
lines.append("- The desired pattern is `readmostly_prepare_remote_bg` pass0 close to `readmostly_prepare_remote` / `uvm_pure_remote`, then pass1/pass2 moving toward `readonly_dup`.")
lines.append("- Counter sample CSV files under each child run show whether `bg_promote_committed_pages` grows while the benchmark is running.")
lines.append("- If pass0 is already local-fast, background promotion completed before or during pass0; reduce `BG_MAX_VALUES`, increase `PAGES_VALUES`, or reduce `OPS_PER_PAGE` to stretch the remote-during-copy window.")
lines.append("- With non-zero `BG_DELAY_MS`, pass0 should stay closer to remote because worker startup is intentionally moved after prepare.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

require_uvm
write_metadata

for pages in ${PAGES_VALUES}; do
  for bg_max in ${BG_MAX_VALUES}; do
    run_case "${pages}" "${bg_max}"
  done
done

summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
