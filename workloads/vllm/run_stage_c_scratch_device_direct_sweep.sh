#!/usr/bin/env bash
# Validate and sweep Stage C/C2 device-direct allocation for runtime scratch buffers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-2}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_c_scratch_device_direct_sweep_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/log/stage_c_scratch_device_direct_sweep.log}"
BACKENDS="${BACKENDS:-cuda_malloc_async}"
BUDGET_BYTES_LIST="${BUDGET_BYTES_LIST:-1048576,4194304,16777216,67108864,268435456}"
MAX_BYTES_LIST="${MAX_BYTES_LIST:-262144,1048576,4194304}"
MIN_BYTES="${MIN_BYTES:-4096}"
TARGET_PHASES="${TARGET_PHASES:-enabled:attention}"
TARGET_CLASS="${TARGET_CLASS:-gap_hot_runtime_scratch}"
TARGET_GAP="${TARGET_GAP:-2}"
POOL_RELEASE_THRESHOLD="${POOL_RELEASE_THRESHOLD:-1048576}"
MAX_RUNS="${MAX_RUNS:-0}"
REQUIRE_REDUCTION_PCT="${REQUIRE_REDUCTION_PCT:-50}"
SKIP_C2_CHECK="${SKIP_C2_CHECK:-0}"
C2_BUDGET_BYTES="${C2_BUDGET_BYTES:-1048576}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_c_scratch_device_direct_sweep.sh [options]

Options:
  --run-root <path>             Output root. Default: /tmp/vllm_stage_c_scratch_device_direct_sweep_<utc_stamp>
  --log-file <path>             Full shell output log. Default: log/stage_c_scratch_device_direct_sweep.log
  --model <name>                Model name. Default: Qwen/Qwen3-30B-A3B-FP8
  --prompts <int>               Benchmark prompts. Default: 4
  --output-len <int>            Generated tokens per request. Default: 256
  --request-rate <float>        Benchmark request rate. Default: 2
  --backends <csv>              Device-direct backends. Default: cuda_malloc_async
  --budgets <csv>               Max live device-direct bytes to sweep.
                                Default: 1048576,4194304,16777216,67108864,268435456
  --max-bytes-list <csv>        Max single allocation bytes to sweep. Default: 262144,1048576,4194304
  --min-bytes <n>               Min eligible allocation bytes. Default: 4096
  --target-phases <csv>         Eligible phase prefixes. Default: enabled:attention
  --target-class <name>         Auto gap target class. Default: gap_hot_runtime_scratch
  --target-gap <n>              Auto gap index to target. Default: 2
  --pool-release-threshold <n>  cuda_malloc_async mempool release threshold. Default: 1048576
  --max-runs <n>                Cap sweep combinations after C2 check. 0 means no cap.
  --require-reduction-pct <n>   Mark strong reduction at or above this pct. Default: 50
  --skip-c2-check               Skip the cuda_malloc vs cuda_malloc_async implementation check.
  -h, --help                    Show this help.

This script tests temporary runtime scratch buffers, not Stage L KV cache policy:
  1. Optional C2 gate: proves device_direct can allocate through cuda_malloc_async.
  2. A/B baseline: managed UVM scratch path with device_direct_trace.
  3. Device-direct run: same workload, but target scratch allocations use GPU allocation.
  4. Sweep: varies backend, total live byte budget, and max eligible allocation size.

Main outputs:
  <run-root>/summary.csv
  <run-root>/summary.md
  <run-root>/runs/*/vllm_stage_c_attention_p<prompts>_ab_comparison.json
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --backends) BACKENDS="$2"; shift 2 ;;
    --budgets) BUDGET_BYTES_LIST="$2"; shift 2 ;;
    --max-bytes-list) MAX_BYTES_LIST="$2"; shift 2 ;;
    --min-bytes) MIN_BYTES="$2"; shift 2 ;;
    --target-phases) TARGET_PHASES="$2"; shift 2 ;;
    --target-class) TARGET_CLASS="$2"; shift 2 ;;
    --target-gap) TARGET_GAP="$2"; shift 2 ;;
    --pool-release-threshold) POOL_RELEASE_THRESHOLD="$2"; shift 2 ;;
    --max-runs) MAX_RUNS="$2"; shift 2 ;;
    --require-reduction-pct) REQUIRE_REDUCTION_PCT="$2"; shift 2 ;;
    --skip-c2-check) SKIP_C2_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage C scratch device-direct sweep started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

echo "==========================================================="
echo " Stage C/C2 Scratch Device-Direct Sweep"
echo " run_root: $RUN_ROOT"
echo " log_file: $LOG_FILE"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " backends: $BACKENDS"
echo " budgets: $BUDGET_BYTES_LIST"
echo " max_bytes_list: $MAX_BYTES_LIST"
echo " min_bytes: $MIN_BYTES"
echo " target_phases: $TARGET_PHASES"
echo " target_class: $TARGET_CLASS"
echo " target_gap: $TARGET_GAP"
echo " pool_release_threshold: ${POOL_RELEASE_THRESHOLD:-<cuda-default>}"
echo " max_runs: $MAX_RUNS"
echo " require_reduction_pct: $REQUIRE_REDUCTION_PCT"
echo " skip_c2_check: $SKIP_C2_CHECK"
echo "==========================================================="

if [ "$SKIP_C2_CHECK" = "0" ]; then
  echo "==========================================================="
  echo " [0] C2 implementation gate: cuda_malloc_async device-direct"
  echo "==========================================================="
  "$SCRIPT_DIR/check_stage_c2_success.py" \
    --out-dir "$RUN_ROOT/c2_success_check" \
    --output-json "$RUN_ROOT/c2_success_check/check.json" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --budget-bytes "$C2_BUDGET_BYTES" \
    --pool-release-threshold "$POOL_RELEASE_THRESHOLD" \
    --no-require-effectiveness-vs-trace
else
  echo "Skipping C2 implementation gate."
fi

IFS=',' read -r -a backend_array <<< "$BACKENDS"
IFS=',' read -r -a budget_array <<< "$BUDGET_BYTES_LIST"
IFS=',' read -r -a max_bytes_array <<< "$MAX_BYTES_LIST"

combo=0
for backend in "${backend_array[@]}"; do
  backend="${backend//[[:space:]]/}"
  [ -n "$backend" ] || continue
  for budget in "${budget_array[@]}"; do
    budget="${budget//[[:space:]]/}"
    [ -n "$budget" ] || continue
    for max_bytes in "${max_bytes_array[@]}"; do
      max_bytes="${max_bytes//[[:space:]]/}"
      [ -n "$max_bytes" ] || continue
      combo=$((combo + 1))
      if [ "$MAX_RUNS" != "0" ] && [ "$combo" -gt "$MAX_RUNS" ]; then
        echo "Reached --max-runs=$MAX_RUNS; stopping sweep."
        break 3
      fi

      run_name="backend_${backend}_budget_${budget}_max_${max_bytes}"
      out_dir="$RUN_ROOT/runs/$run_name"
      echo "==========================================================="
      echo " [$combo] Scratch A/B: $run_name"
      echo " out_dir: $out_dir"
      echo "==========================================================="
      MODEL="$MODEL" \
      PROMPTS="$PROMPTS" \
      REQUEST_RATE="$REQUEST_RATE" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      DEVICE_DIRECT_MIN_BYTES="$MIN_BYTES" \
      DEVICE_DIRECT_MAX_BYTES="$max_bytes" \
      DEVICE_DIRECT_MAX_TOTAL_BYTES="$budget" \
      DEVICE_DIRECT_BACKEND="$backend" \
      DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="$POOL_RELEASE_THRESHOLD" \
      DEVICE_DIRECT_TARGET_PHASES="$TARGET_PHASES" \
      AUTO_GAP_WATCH_TARGET_GAP="$TARGET_GAP" \
      AUTO_GAP_WATCH_TARGET_CLASS="$TARGET_CLASS" \
      OUT_DIR="$out_dir" \
      RUN_ID="$run_name" \
      "$SCRIPT_DIR/run_stage_c_attention_p20_ab.sh"
    done
  done
done

echo "==========================================================="
echo " Building scratch device-direct sweep summary"
echo "==========================================================="
python3 - "$RUN_ROOT" "$PROMPTS" "$REQUIRE_REDUCTION_PCT" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
prompts = sys.argv[2]
require_reduction_pct = float(sys.argv[3])
runs_root = run_root / "runs"
summary_csv = run_root / "summary.csv"
summary_md = run_root / "summary.md"

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def pct(value):
    if value is None:
        return None
    return float(value) * 100.0

def fmt(value, digits=2):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)

def pick(data: dict, *keys):
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

rows = []
if runs_root.is_dir():
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        report = run_dir / f"vllm_stage_c_attention_p{prompts}_ab_comparison.json"
        if not report.is_file():
            rows.append({
                "run": run_dir.name,
                "status": "missing_report",
                "report": str(report),
            })
            continue
        data = load_json(report)
        comp = data.get("comparison", {})
        trace = data.get("trace_only", {})
        device = data.get("stage_c_attention_device_direct", {})
        device_metrics = device.get("metrics", {})
        device_bench = device.get("bench_main", {})
        trace_bench = trace.get("bench_main", {})

        gap_delta_pct = pct(comp.get("gap_fault_delta_pct"))
        unknown_delta_pct = pct(comp.get("unknown_fault_delta_pct"))
        throughput_delta_pct = pct(comp.get("output_throughput_delta_pct"))
        tpot_delta_pct = pct(comp.get("mean_tpot_delta_pct"))
        scratch_reduction_pct = None
        if gap_delta_pct is not None:
            scratch_reduction_pct = -gap_delta_pct

        rows.append({
            "run": run_dir.name,
            "status": "ok",
            "success_signal": comp.get("success_signal"),
            "effectiveness_signal": comp.get("effectiveness_signal"),
            "strong_scratch_reduction": (
                scratch_reduction_pct is not None
                and scratch_reduction_pct >= require_reduction_pct
            ),
            "trace_gap_faults": pick(trace, "main_gap", "faults"),
            "device_gap_faults": pick(device, "main_gap", "faults"),
            "scratch_fault_reduction_pct": scratch_reduction_pct,
            "unknown_fault_delta_pct": unknown_delta_pct,
            "trace_unknown_faults": trace.get("total_unknown_faults"),
            "device_unknown_faults": device.get("total_unknown_faults"),
            "device_direct_actual_records": device_metrics.get("device_direct_actual_records"),
            "device_direct_eligible_records": device_metrics.get("device_direct_eligible_records"),
            "device_direct_budget_reject_records": device_metrics.get("device_direct_budget_reject_records"),
            "device_direct_backend_counts": json.dumps(device_metrics.get("device_direct_backend_counts") or {}, sort_keys=True),
            "placement_backend_counts": json.dumps(device_metrics.get("placement_backend_counts") or {}, sort_keys=True),
            "device_direct_max_total_bytes": device_metrics.get("device_direct_max_total_bytes"),
            "device_direct_peak_live_bytes_observed": device_metrics.get("device_direct_peak_live_bytes_observed"),
            "device_direct_min_budget_remaining_observed": device_metrics.get("device_direct_min_budget_remaining_observed"),
            "pool_release_threshold_set": device_metrics.get("device_direct_pool_release_threshold_set"),
            "pool_release_threshold": device_metrics.get("device_direct_pool_release_threshold"),
            "pool_config_success": device_metrics.get("device_direct_pool_config_success"),
            "trace_output_tok_s": trace_bench.get("output_throughput_tok_s"),
            "device_output_tok_s": device_bench.get("output_throughput_tok_s"),
            "output_throughput_delta_pct": throughput_delta_pct,
            "trace_mean_tpot_ms": trace_bench.get("mean_tpot_ms"),
            "device_mean_tpot_ms": device_bench.get("mean_tpot_ms"),
            "mean_tpot_delta_pct": tpot_delta_pct,
            "failed_requests": device_bench.get("failed_requests"),
            "report": str(report),
        })

fieldnames = [
    "run",
    "status",
    "success_signal",
    "effectiveness_signal",
    "strong_scratch_reduction",
    "trace_gap_faults",
    "device_gap_faults",
    "scratch_fault_reduction_pct",
    "unknown_fault_delta_pct",
    "trace_unknown_faults",
    "device_unknown_faults",
    "device_direct_actual_records",
    "device_direct_eligible_records",
    "device_direct_budget_reject_records",
    "device_direct_backend_counts",
    "placement_backend_counts",
    "device_direct_max_total_bytes",
    "device_direct_peak_live_bytes_observed",
    "device_direct_min_budget_remaining_observed",
    "pool_release_threshold_set",
    "pool_release_threshold",
    "pool_config_success",
    "trace_output_tok_s",
    "device_output_tok_s",
    "output_throughput_delta_pct",
    "trace_mean_tpot_ms",
    "device_mean_tpot_ms",
    "mean_tpot_delta_pct",
    "failed_requests",
    "report",
]
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

ok_rows = [r for r in rows if r.get("status") == "ok"]
successful = [r for r in ok_rows if r.get("success_signal") is True]
effective = [r for r in successful if r.get("scratch_fault_reduction_pct") is not None]
best = sorted(
    effective,
    key=lambda r: (
        r.get("scratch_fault_reduction_pct") or -1,
        -(r.get("mean_tpot_delta_pct") or 0),
    ),
    reverse=True,
)

lines = []
lines.append("# Stage C/C2 Scratch Device-Direct Sweep")
lines.append("")
lines.append(f"- run_root: `{run_root}`")
lines.append(f"- total_runs: {len(rows)}")
lines.append(f"- successful_device_direct_runs: {len(successful)}")
lines.append(f"- strong_reduction_threshold: {require_reduction_pct:.1f}%")
lines.append("")
lines.append("## What This Tests")
lines.append("")
lines.append("- Baseline run keeps temporary scratch allocations on the managed UVM path and records `device_direct_trace` eligibility.")
lines.append("- Device run changes matching `gap_hot_runtime_scratch` allocations to direct GPU allocation through Stage C/C2 `device_direct`.")
lines.append("- The key fault metric is `scratch_fault_reduction_pct`, derived from the hot scratch gap fault delta.")
lines.append("- `device_direct_actual_records` and backend counts prove whether the implementation actually used the GPU direct allocation path.")
lines.append("")
lines.append("## Best Runs")
lines.append("")
if best:
    lines.append("| rank | run | scratch fault reduction | unknown fault delta | actual records | budget rejects | output delta | TPOT delta |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for idx, row in enumerate(best[:10], start=1):
        lines.append(
            "| "
            f"{idx} | `{row['run']}` | "
            f"{fmt(row.get('scratch_fault_reduction_pct'))}% | "
            f"{fmt(row.get('unknown_fault_delta_pct'))}% | "
            f"{fmt(row.get('device_direct_actual_records'))} | "
            f"{fmt(row.get('device_direct_budget_reject_records'))} | "
            f"{fmt(row.get('output_throughput_delta_pct'))}% | "
            f"{fmt(row.get('mean_tpot_delta_pct'))}% |"
        )
else:
    lines.append("No successful device-direct run with parseable scratch fault reduction was found.")
lines.append("")
lines.append("## How To Read The Result")
lines.append("")
lines.append("- A good parameter set has `success_signal=True`, many `device_direct_actual_records`, near-zero budget rejects, and high positive `scratch_fault_reduction_pct`.")
lines.append("- If scratch faults fall but throughput/TPOT worsens, the direct allocation path may be paying allocation/synchronization overhead; compare `cuda_malloc` vs `cuda_malloc_async` and increase the async pool threshold/budget.")
lines.append("- If actual records are zero, the phase/class/size filters did not match the temporary buffers.")
lines.append("- If budget rejects are high, raise `--budgets` or lower `--max-bytes-list` only if the large allocations are not the fault source.")
lines.append("")
lines.append(f"Full CSV: `{summary_csv}`")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"summary_csv={summary_csv}")
print(f"summary_md={summary_md}")
if best:
    top = best[0]
    print(
        "best_run="
        f"{top['run']} scratch_fault_reduction_pct={fmt(top.get('scratch_fault_reduction_pct'))} "
        f"actual_records={fmt(top.get('device_direct_actual_records'))} "
        f"budget_rejects={fmt(top.get('device_direct_budget_reject_records'))}"
    )
PY

echo "==========================================================="
echo " Done"
echo " Summary CSV: $RUN_ROOT/summary.csv"
echo " Summary report: $RUN_ROOT/summary.md"
echo " Raw runs: $RUN_ROOT/runs"
echo "==========================================================="
