#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/runtime_gate_bg_promotion_$(date -u +%Y%m%d_%H%M%S)}"
SUMMARY="${RUN_ROOT}/runtime_gate_bg_promotion_summary.md"
META="${RUN_ROOT}/metadata.txt"

PAGES_VALUES="${PAGES_VALUES:-16384 65536 131072}"
BG_IDLE_WINDOW_MS_VALUES="${BG_IDLE_WINDOW_MS_VALUES:-0 1 2 5}"
PASSES="${PASSES:-5}"
OPS_PER_PAGE="${OPS_PER_PAGE:-64}"
REPEATS="${REPEATS:-3}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
MECHANISMS="${MECHANISMS:-uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg_after_first readmostly_prepare_remote_bg_runtime_gate readonly_dup}"

COUNTER_KEYS=(
  remote_prepare_mapped_pages
  bg_promote_enqueued_pages
  bg_promote_copied_pages
  bg_promote_committed_pages
  pte_switch_pages
  bg_promote_worker_rounds
  bg_promote_retry_count
  bg_promote_cancelled_pages
  bg_promote_deferred_pages
  bg_promote_start_pages
  bg_promote_pause_calls
  bg_promote_resume_calls
  bg_promote_paused_blocks
  bg_promote_resumed_blocks
  bg_promote_gate_skips
)

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
    echo "bg_idle_window_ms_values=${BG_IDLE_WINDOW_MS_VALUES}"
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
  local idle_ms="$2"
  local child_root="${RUN_ROOT}/pages_${pages}/idle_${idle_ms}ms"

  mkdir -p "${child_root}"
  reset_counters
  cat "${STATS_PATH}" > "${child_root}/latency_stats_before.txt"

  echo
  echo "Running pages=${pages} data_mib=$((pages * 4096 / 1024 / 1024)) BG_IDLE_WINDOW_MS=${idle_ms}"

  RUN_ROOT="${child_root}" \
  PAGES="${pages}" \
  PASSES="${PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${BG_MAX_PAGES}" \
  BG_DELAY_MS="${BG_DELAY_MS}" \
  BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
  START_BG_AFTER_PASS=0 \
  BG_IDLE_WINDOW_MS="${idle_ms}" \
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
max_pass = 0
for child in sorted(run_root.glob("pages_*/idle_*ms"),
                    key=lambda p: (int(p.parent.name.split("_")[1]), int(p.name.split("_")[1][:-2]))):
    raw_path = child / "readonly_access_mechanisms_raw.csv"
    if not raw_path.exists():
        continue

    pages = int(child.parent.name.split("_")[1])
    idle_ms = int(child.name.split("_")[1][:-2])
    data_mib = pages * 4096 / (1024 * 1024)
    raw = list(csv.DictReader(raw_path.open(newline="")))
    delta = deltas(child / "counter_delta.csv")
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
            "idle_ms": idle_ms,
            "mech": mech,
            "prepare": median(f(row, "prepare_ms") for row in selected),
            "prepared_mib": median(f(row, "prepare_mapped_mib") for row in selected),
            "passes": pass_ms,
            "committed": delta.get("bg_promote_committed_pages", 0),
            "rounds": delta.get("bg_promote_worker_rounds", 0),
            "pause": delta.get("bg_promote_pause_calls", 0),
            "resume": delta.get("bg_promote_resume_calls", 0),
            "paused_blocks": delta.get("bg_promote_paused_blocks", 0),
            "resumed_blocks": delta.get("bg_promote_resumed_blocks", 0),
            "gate_skips": delta.get("bg_promote_gate_skips", 0),
            "raw": raw_path,
        })

remote_by_case = {}
readonly_by_case = {}
for row in rows:
    key = (row["data_mib"], row["idle_ms"])
    if row["mech"] == "uvm_pure_remote":
        remote_by_case[key] = row
    elif row["mech"] == "readonly_dup":
        readonly_by_case[key] = row

lines = []
lines.append("# Runtime-Gated Background Promotion")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Purpose")
lines.append("")
lines.append("This experiment tests explicit runtime gating: background read-dup promotion is paused during each measured remote-read pass, then resumed in the pass gap. `BG_IDLE_WINDOW_MS` controls how much host-side idle time is given to the worker after each pass.")
lines.append("")
lines.append("## Key View")
lines.append("")
lines.append("| Data MiB | Idle window ms | Remote p1 ms | Runtime-gate p0 | Runtime-gate p1 | Runtime-gate p2 | Runtime-gate last | p1 vs remote | Local last ms | BG committed pages | Gate skips |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    if row["mech"] != "readmostly_prepare_remote_bg_runtime_gate":
        continue
    key = (row["data_mib"], row["idle_ms"])
    remote = remote_by_case.get(key, {}).get("passes", {})
    local = readonly_by_case.get(key, {}).get("passes", {})
    remote_p1 = remote.get(1, 0.0)
    p1 = row["passes"].get(1, 0.0)
    ratio = p1 / remote_p1 if remote_p1 > 0 else 0.0
    lines.append(
        f"| {row['data_mib']:.0f} | {row['idle_ms']} | {remote_p1:.3f} | "
        f"{row['passes'].get(0, 0.0):.3f} | {p1:.3f} | {row['passes'].get(2, 0.0):.3f} | "
        f"{row['passes'].get(max(row['passes']), 0.0):.3f} | {ratio:.2f}x | "
        f"{local.get(max(local), 0.0):.3f} | {row['committed']} | {row['gate_skips']} |"
    )

headers = ["Data MiB", "Idle ms", "Mechanism", "Prepare ms", "Prepared MiB"]
headers += [f"Pass{p} ms" for p in range(max_pass + 1)]
headers += ["Committed pages", "Worker rounds", "Pause calls", "Resume calls", "Paused blocks", "Resumed blocks", "Gate skips"]

lines.append("")
lines.append("## Full Median Table")
lines.append("")
lines.append("| " + " | ".join(headers) + " |")
lines.append("|" + "|".join("---:" if h != "Mechanism" else "---" for h in headers) + "|")
for row in rows:
    values = [
        f"{row['data_mib']:.0f}",
        str(row["idle_ms"]),
        f"`{row['mech']}`",
        f"{row['prepare']:.3f}",
        f"{row['prepared_mib']:.1f}",
    ]
    for p in range(max_pass + 1):
        values.append(f"{row['passes'].get(p, 0.0):.3f}")
    values += [
        str(row["committed"]),
        str(row["rounds"]),
        str(row["pause"]),
        str(row["resume"]),
        str(row["paused_blocks"]),
        str(row["resumed_blocks"]),
        str(row["gate_skips"]),
    ]
    lines.append("| " + " | ".join(values) + " |")

lines.append("")
lines.append("## Expected Signal")
lines.append("")
lines.append("- Passes should stay close to `uvm_pure_remote` while promotion is paused.")
lines.append("- Larger `BG_IDLE_WINDOW_MS` should increase committed pages and move later passes toward `readonly_dup` local speed.")
lines.append("- `bg_promote_gate_skips` should be non-zero when workers/timers wake during paused remote-read windows.")
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
write_metadata
for pages in ${PAGES_VALUES}; do
  for idle_ms in ${BG_IDLE_WINDOW_MS_VALUES}; do
    run_case "${pages}" "${idle_ms}"
  done
done
summarize

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
