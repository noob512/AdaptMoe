#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

BG_MAX_VALUES="${BG_MAX_VALUES:-64 128 256 512 1024 0}"
PAGES="${PAGES:-16384}"
PASSES="${PASSES:-6}"
OPS_PER_PAGE="${OPS_PER_PAGE:-128}"
COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE:-512}"
REPEATS="${REPEATS:-2}"
DEVICE="${DEVICE:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/remote_first_bg_limit_sweep_$(date -u +%Y%m%d_%H%M%S)}"

SUMMARY="${RUN_ROOT}/bg_limit_sweep_summary.md"
META="${RUN_ROOT}/metadata.txt"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "root_dir=${ROOT_DIR}"
  echo "bg_max_values=${BG_MAX_VALUES}"
  echo "pages=${PAGES}"
  echo "passes=${PASSES}"
  echo "ops_per_page=${OPS_PER_PAGE}"
  echo "counter_ops_per_page=${COUNTER_OPS_PER_PAGE}"
  echo "repeats=${REPEATS}"
  echo "device=${DEVICE}"
  echo "phase_f_route_log_delay_ms=${PHASE_F_ROUTE_LOG_DELAY_MS}"
  echo "reset_counters=${RESET_COUNTERS}"
} | tee "${META}"

for bg_max in ${BG_MAX_VALUES}; do
  child_root="${RUN_ROOT}/bg_max_${bg_max}"
  echo
  echo "Running BG_MAX_PAGES=${bg_max}"
  RUN_ROOT="${child_root}" \
  PAGES="${PAGES}" \
  PASSES="${PASSES}" \
  OPS_PER_PAGE="${OPS_PER_PAGE}" \
  COUNTER_OPS_PER_PAGE="${COUNTER_OPS_PER_PAGE}" \
  REPEATS="${REPEATS}" \
  DEVICE="${DEVICE}" \
  BG_MAX_PAGES="${bg_max}" \
  PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS}" \
  CASES="remote_first_bg" \
  RESET_COUNTERS="${RESET_COUNTERS}" \
  "${ROOT_DIR}/scripts/test_remote_first_bg_promotion_counters.sh"
done

python3 - "${RUN_ROOT}" "${SUMMARY}" "${META}" ${BG_MAX_VALUES} <<'PY'
import csv
import statistics as stats
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
values = sys.argv[4:]

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
for value in values:
    child = run_root / f"bg_max_{value}"
    raw_path = child / "probe_raw.csv"
    delta_path = child / "counter_delta_by_case.csv"
    if not raw_path.exists() or not delta_path.exists():
        rows.append({"bg_max": value, "missing": True})
        continue

    raw = list(csv.DictReader(raw_path.open(newline="")))
    deltas = list(csv.DictReader(delta_path.open(newline="")))
    last_pass = max(i(r, "pass") for r in raw)
    pass0 = [r for r in raw if i(r, "pass") == 0]
    last = [r for r in raw if i(r, "pass") == last_pass]

    agg = {
        "bg_max": value,
        "missing": False,
        "pass0_ms": median(f(r, "wall_ms") for r in pass0),
        "last_ms": median(f(r, "wall_ms") for r in last),
        "median_ms": median(f(r, "wall_ms") for r in raw),
        "last_gib_s": median(f(r, "gib_s") for r in last),
        "summary": child / "probe_summary.md",
    }

    for key in [
        "remote_first_pages",
        "remote_map_pages",
        "bg_promote_enqueued_pages",
        "bg_promote_demand_pages_promoted",
        "bg_promote_prefetch_pages_promoted",
        "bg_promote_committed_pages",
        "pte_switch_pages",
        "bg_promote_worker_rounds",
        "bg_promote_retry_count",
        "bg_promote_cancelled_pages",
        "sync_fallback_pages",
    ]:
        agg[key] = sum(i(r, f"{key}_delta") for r in deltas)

    rows.append(agg)

lines = []
lines.append("# Remote-First BG Promotion Limit Sweep")
lines.append("")
lines.append(f"- Run root: `{run_root}`")
lines.append(f"- Metadata: `{meta_path}`")
lines.append("")
lines.append("## Performance And Counters")
lines.append("")
lines.append("| BG_MAX_PAGES | Pass0 ms | Last-pass ms | Median ms | Last GiB/s | RF pages | Remote map | BG enq | Demand promoted | Prefetch promoted | BG committed | PTE switch | Worker rounds | Retries | Cancelled | Fallback |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for row in rows:
    if row.get("missing"):
        lines.append(f"| {row['bg_max']} | MISSING | MISSING | MISSING | MISSING | | | | | | | | | | | |")
        continue
    lines.append(
        f"| {row['bg_max']} | {row['pass0_ms']:.3f} | {row['last_ms']:.3f} | {row['median_ms']:.3f} | "
        f"{row['last_gib_s']:.3f} | {row['remote_first_pages']} | {row['remote_map_pages']} | "
        f"{row['bg_promote_enqueued_pages']} | {row['bg_promote_demand_pages_promoted']} | "
        f"{row['bg_promote_prefetch_pages_promoted']} | {row['bg_promote_committed_pages']} | "
        f"{row['pte_switch_pages']} | {row['bg_promote_worker_rounds']} | {row['bg_promote_retry_count']} | "
        f"{row['bg_promote_cancelled_pages']} | {row['sync_fallback_pages']} |"
    )

lines.append("")
lines.append("## How To Read This")
lines.append("")
lines.append("- If `BG_MAX_PAGES` is too large, pass0 may rise because background copy competes with urgent remote reads.")
lines.append("- If `BG_MAX_PAGES` is too small, worker rounds rise and later passes may remain remote for longer.")
lines.append("- A useful setting keeps pass0 low while making `BG committed` and `PTE switch` reach the working set quickly.")

summary_path.write_text("\n".join(lines) + "\n")
print(summary_path)
PY

echo
echo "Done."
echo "Run root: ${RUN_ROOT}"
echo "Summary: ${SUMMARY}"
