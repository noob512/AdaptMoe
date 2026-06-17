#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/start_bg_after_pass_sweep_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY="${RUN_ROOT}/start_bg_after_pass_sweep_summary.md"
META="${RUN_ROOT}/metadata.txt"

PAGES_VALUES="${PAGES_VALUES:-16384 65536 131072}"
START_BG_AFTER_PASS_VALUES="${START_BG_AFTER_PASS_VALUES:-0 1 2}"
PASSES="${PASSES:-5}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-1}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote uva_pinned_remote readmostly_prepare_remote readmostly_prepare_remote_bg_after_first readonly_dup}"

mkdir -p "${RUN_ROOT}"

require_uvm() {
  if [[ ! -r "${STATS_PATH}" ]]; then
    echo "ERROR: ${STATS_PATH} is missing. Load the modified nvidia_uvm module first." >&2
    exit 1
  fi

  for param in \
    uvm_latency_bg_promote_max_pages \
    uvm_latency_bg_promote_delay_ms \
    uvm_latency_bg_promote_reschedule_delay_ms; do
    if [[ ! -e "${PARAM_ROOT}/${param}" ]]; then
      echo "ERROR: ${PARAM_ROOT}/${param} is missing. This does not look like the modified UVM module." >&2
      exit 1
    fi
  done
}

validate_start_passes() {
  local start
  for start in ${START_BG_AFTER_PASS_VALUES}; do
    if (( start < 0 || start >= PASSES )); then
      echo "ERROR: START_BG_AFTER_PASS=${start} must be in [0, PASSES - 1]. PASSES=${PASSES}" >&2
      exit 1
    fi
  done
}

write_metadata() {
  {
    echo "run_root=${RUN_ROOT}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root_dir=${ROOT_DIR}"
    echo "pages_values=${PAGES_VALUES}"
    echo "start_bg_after_pass_values=${START_BG_AFTER_PASS_VALUES}"
    echo "passes=${PASSES}"
    echo "ops_per_page=${OPS_PER_PAGE}"
    echo "repeats=${REPEATS}"
    echo "device=${DEVICE}"
    echo "bg_max_pages=${BG_MAX_PAGES}"
    echo "bg_delay_ms=${BG_DELAY_MS}"
    echo "bg_reschedule_delay_ms=${BG_RESCHEDULE_DELAY_MS}"
    echo "mechanisms=${MECHANISMS}"
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
      "${PARAM_ROOT}/uvm_perf_access_counter_migration_enable" \
      "${PARAM_ROOT}/uvm_perf_access_counter_threshold" \
      "${PARAM_ROOT}/uvm_ats_mode"; do
      [[ -r "${p}" ]] && printf '  %-48s %s\n' "$(basename "${p}")" "$(cat "${p}")"
    done
  } | tee "${META}"
}

run_case() {
  local pages="$1"
  local start_after_pass="$2"
  local child_root="${RUN_ROOT}/pages_${pages}/start_after_${start_after_pass}"

  echo
  echo "Running pages=${pages} data_mib=$((pages * 4096 / 1024 / 1024)) START_BG_AFTER_PASS=${start_after_pass}"

  RUN_ROOT="${child_root}" \
  PAGES="${pages}" \
  PASSES="${PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${BG_MAX_PAGES}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  START_BG_AFTER_PASS="${start_after_pass}" \
  MECHANISMS="${MECHANISMS}" \
  DISABLE_REMOTE_FIRST=1 \
  "${ROOT_DIR}/scripts/test_readonly_access_mechanisms.sh"
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

rows = []
max_pass = 0
for child in sorted(run_root.glob("pages_*/start_after_*"),
                    key=lambda p: (int(p.parent.name.split("_")[1]), int(p.name.split("_")[-1]))):
    raw_path = child / "readonly_access_mechanisms_raw.csv"
    if not raw_path.exists():
        continue

    pages = int(child.parent.name.split("_")[1])
    start_after = int(child.name.split("_")[-1])
    data_mib = pages * 4096 / (1024 * 1024)
    raw = list(csv.DictReader(raw_path.open(newline="")))
    if not raw:
        continue

    max_pass = max(max_pass, max(i(row, "pass") for row in raw))
    for mech in dict.fromkeys(row["mechanism"] for row in raw):
        selected = [row for row in raw if row["mechanism"] == mech]
        pass_ms = {
            p: median(f(row, "wall_ms") for row in selected if i(row, "pass") == p)
            for p in sorted({i(row, "pass") for row in selected})
        }
        rows.append({
            "data_mib": data_mib,
            "start_after": start_after,
            "mech": mech,
            "prepare": median(f(row, "prepare_ms") for row in selected),
            "prepared_mib": median(f(row, "prepare_mapped_mib") for row in selected),
            "passes": pass_ms,
            "raw": raw_path,
        })

remote_by_case = {}
readonly_by_case = {}
for row in rows:
    key = (row["data_mib"], row["start_after"])
    if row["mech"] == "uvm_pure_remote":
        remote_by_case[key] = row
    elif row["mech"] == "readonly_dup":
        readonly_by_case[key] = row

lines = []
lines.append("# Start-BG-After-Pass Sweep")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("This experiment keeps the existing `readmostly_prepare_remote_bg_after_first` kernel path, but changes the user-space benchmark decision of when to call `START_BG_PROMOTION`.")
lines.append("")
lines.append("- `START_BG_AFTER_PASS=0`: current behavior; start background promotion immediately after pass0.")
lines.append("- `START_BG_AFTER_PASS=1`: let pass0 and pass1 use the pre-built remote mapping, then start background promotion.")
lines.append("- `START_BG_AFTER_PASS=2`: let pass0/pass1/pass2 use remote reads, then start background promotion.")
lines.append("")
lines.append("If pass1 slowdown is caused by background copy/PTE switch competing with remote reads, then moving `START_BG_AFTER_PASS` from 0 to 1 should make pass1 return to the `uvm_pure_remote` level, and shift the disturbance to the next pass.")
lines.append("")

lines.append("## Key View")
lines.append("")
lines.append("| Data MiB | Start BG after pass | Remote p1 ms | After-first p0 ms | After-first p1 ms | After-first p2 ms | After-first last ms | p1 vs remote | Local last ms |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    if row["mech"] != "readmostly_prepare_remote_bg_after_first":
        continue
    key = (row["data_mib"], row["start_after"])
    remote = remote_by_case.get(key, {}).get("passes", {})
    local = readonly_by_case.get(key, {}).get("passes", {})
    remote_p1 = remote.get(1, 0.0)
    p1 = row["passes"].get(1, 0.0)
    ratio = p1 / remote_p1 if remote_p1 > 0 else 0.0
    lines.append(
        f"| {row['data_mib']:.0f} | {row['start_after']} | {remote_p1:.3f} | "
        f"{row['passes'].get(0, 0.0):.3f} | {p1:.3f} | {row['passes'].get(2, 0.0):.3f} | "
        f"{row['passes'].get(max(row['passes']), 0.0):.3f} | {ratio:.2f}x | "
        f"{local.get(max(local), 0.0):.3f} |"
    )

headers = ["Data MiB", "Start BG after pass", "Mechanism", "Prepare ms", "Prepared MiB"]
headers += [f"Pass{p} ms" for p in range(max_pass + 1)]
lines.append("")
lines.append("## Full Median Table")
lines.append("")
lines.append("| " + " | ".join(headers) + " |")
lines.append("|" + "|".join("---:" if h != "Mechanism" else "---" for h in headers) + "|")
for row in rows:
    values = [
        f"{row['data_mib']:.0f}",
        str(row["start_after"]),
        f"`{row['mech']}`",
        f"{row['prepare']:.3f}",
        f"{row['prepared_mib']:.1f}",
    ]
    for p in range(max_pass + 1):
        values.append(f"{row['passes'].get(p, 0.0):.3f}")
    lines.append("| " + " | ".join(values) + " |")

lines.append("")
lines.append("## Raw Outputs")
lines.append("")
for raw_path in sorted({row["raw"] for row in rows}):
    lines.append(f"- `{raw_path}`")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY
}

require_uvm
validate_start_passes
write_metadata
for pages in ${PAGES_VALUES}; do
  for start_after_pass in ${START_BG_AFTER_PASS_VALUES}; do
    run_case "${pages}" "${start_after_pass}"
  done
done
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
