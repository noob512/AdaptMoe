#!/usr/bin/env bash
# Diagnose whether Stage L reduces UVM faults and where performance overhead appears.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REPEAT="${REPEAT:-3}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_l_fault_perf_diag_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
RUN_L7_CROSS="${RUN_L7_CROSS:-0}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_l_fault_perf_diagnosis.sh [options]

Options:
  --run-root <path>       Output root. Default: /tmp/vllm_stage_l_fault_perf_diag_<utc_stamp>
  --log-file <path>       Full shell output log. Default: <run-root>/fault_perf_diagnosis.log
  --model <name>          Model name. Default: Qwen/Qwen3-30B-A3B-FP8
  --prompts <int>         Benchmark prompts. Default: 4
  --output-len <int>      Generated tokens per request. Default: 256
  --request-rate <float>  Benchmark request rate. Default: 2
  --repeat <int>          Repeats per variant. Default: 3
  --with-l7-cross         Also run the heavier cross-pool L7 variant.
  -h, --help              Show this help.

This script runs a layered A/B diagnosis:
  baseline_pure_uvm       pure UVM-style run with Stage L/J/K disabled and VLLM_UVM_POLICY_ENABLE=0
  l5_metadata             KV metadata tracking only
  l6_decision_trace       KV safe-offload decisions in trace-only action mode
  l6_real_prefetch        KV safe offload/prefetch-back with real prefetch calls
  l7_coordinator          L6 real prefetch routed through Stage K coordinator
  l7_cross_pool           optional KV + weights cross-pool coordinator

It writes per-run logs plus summary.csv and summary.md under the run root.
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
    --repeat) REPEAT="$2"; shift 2 ;;
    --with-l7-cross) RUN_L7_CROSS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/fault_perf_diagnosis.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage L fault/performance diagnosis started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

echo "==========================================================="
echo " Stage L Fault/Performance Diagnosis"
echo " run_root: $RUN_ROOT"
echo " log_file: $LOG_FILE"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " repeat: $REPEAT"
echo " run_l7_cross: $RUN_L7_CROSS"
echo " pure_uvm_policy: VLLM_UVM_POLICY_ENABLE=0"
echo "==========================================================="

run_variant() {
  local name="$1"
  shift
  local rep
  for rep in $(seq 1 "$REPEAT"); do
    local run_dir="$RUN_ROOT/runs/${name}_rep${rep}"
    mkdir -p "$run_dir"
    echo "==========================================================="
    echo " Running ${name} rep ${rep}/${REPEAT}"
    echo " run_dir: $run_dir"
    echo "==========================================================="
    VLLM_UVM_POLICY_ENABLE=0 "$@" "$run_dir" "$rep"
  done
}

run_baseline() {
  local run_dir="$1"
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --allocator-log "$run_dir/vllm_uvm_allocator_trace_baseline.log" \
    --trace-log "$run_dir/uvm_kv_fault_stats_baseline.log" \
    --address-log "$run_dir/vllm_uvm_address_regions_baseline.log" \
    --server-log "$run_dir/vllm_server_baseline.log" \
    --bench-log "$run_dir/vllm_bench_baseline.log" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE"
}

run_l5() {
  local run_dir="$1"
  "$SCRIPT_DIR/check_stage_l5_success.py" \
    --run-dir "$run_dir" \
    --output-json "$run_dir/check.json" \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --mode trace_only
}

run_l6_decision_trace() {
  local run_dir="$1"
  "$SCRIPT_DIR/check_stage_l6_success.py" \
    --run-dir "$run_dir" \
    --output-json "$run_dir/check.json" \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --mode trace_only \
    --data-offload-mode trace_only \
    --prefetch-back-mode trace_only
}

run_l6_real_prefetch() {
  local run_dir="$1"
  "$SCRIPT_DIR/check_stage_l6_success.py" \
    --run-dir "$run_dir" \
    --output-json "$run_dir/check.json" \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --mode trace_only \
    --data-offload-mode prefetch_cpu \
    --prefetch-back-mode prefetch \
    --strict-issued
}

run_l7_coordinator() {
  local run_dir="$1"
  "$SCRIPT_DIR/check_stage_l7_success.py" \
    --run-dir "$run_dir" \
    --output-json "$run_dir/check.json" \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --mode trace_only \
    --coordinator-mode trace_only
}

run_l7_cross_pool() {
  local run_dir="$1"
  "$SCRIPT_DIR/check_stage_l7_cross_pool_success.py" \
    --run-dir "$run_dir" \
    --output-json "$run_dir/check.json" \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --mode trace_only \
    --coordinator-mode trace_only \
    --priority kv,scratch,weights
}

run_variant baseline_pure_uvm run_baseline
run_variant l5_metadata run_l5
run_variant l6_decision_trace run_l6_decision_trace
run_variant l6_real_prefetch run_l6_real_prefetch
run_variant l7_coordinator run_l7_coordinator
if [ "$RUN_L7_CROSS" = "1" ]; then
  run_variant l7_cross_pool run_l7_cross_pool
fi

echo "==========================================================="
echo " Building diagnosis summary"
echo "==========================================================="
python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
runs_dir = run_root / "runs"

bench_patterns = {
    "successful_requests": r"Successful requests:\s+(\d+)",
    "failed_requests": r"Failed requests:\s+(\d+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
}
fault_keys = (
    "delta_faults",
    "delta_duplicates",
    "delta_after_dedup",
    "delta_kv_faults",
    "delta_kv_duplicates",
    "delta_kv_after_dedup",
)

def number(raw: str):
    return int(raw) if raw.isdigit() else float(raw)

def parse_bench(path: Path) -> dict:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pattern in bench_patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            out[key] = number(matches[-1])
    return out

def parse_faults(logs: list[Path]) -> dict:
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in logs
        if p.is_file()
    )
    out = {}
    for key in fault_keys:
        matches = re.findall(rf"\b{key}=([0-9]+)", text)
        if matches:
            out[key] = int(matches[-1])
    return out

def read_check(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None

def summarize_check(data: dict) -> dict:
    out = {}
    checks = data.get("checks") or []
    if checks:
        out["checks_failed"] = sum(1 for item in checks if not item.get("passed"))
    out["passed"] = data.get("passed")

    trace = data.get("trace_summary") or {}
    residency = data.get("residency_summary") or data.get("residency") or {}
    coordinator = data.get("coordinator_summary") or data.get("coordinator") or {}
    pointer_map = data.get("pointer_map_summary") or data.get("pointer_map") or {}
    weight_trace = data.get("weight_trace") or {}

    for source in (trace, residency):
        for key in ("offload_issued", "prefetch_issued", "offload_fallbacks", "unsafe_offloads", "trace_records", "records"):
            value = source.get(key)
            if isinstance(value, list):
                value = len(value)
            if value is not None:
                out[f"residency_{key}"] = value
        summary = source.get("summary")
        if isinstance(summary, dict):
            for key in (
                "kv_data_offload_attempt_records",
                "kv_data_offload_issued_records",
                "kv_prefetch_back_issued_records",
                "kv_data_offload_fallback_records",
                "kv_prefetch_back_fallback_records",
                "kv_safe_candidate_records",
                "kv_unsafe_candidate_records",
                "real_action_records",
                "state_updates",
            ):
                if key in summary:
                    out[key] = summary[key]

    for key in ("grants", "denies", "requests", "priority_records", "priority_safety_records"):
        if key in coordinator:
            out[f"coordinator_{key}"] = coordinator[key]
    if "actions" in coordinator:
        out["coordinator_actions"] = json.dumps(coordinator["actions"], sort_keys=True)
    if "pools" in coordinator:
        out["coordinator_pools"] = json.dumps(coordinator["pools"], sort_keys=True)
    for key in ("mapped_blocks", "records"):
        if key in pointer_map:
            out[f"pointer_map_{key}"] = pointer_map[key]
    for key in ("records", "prefetch_candidates", "offload_candidates"):
        if key in weight_trace:
            out[f"weight_trace_{key}"] = weight_trace[key]
    return out

rows = []
for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
    variant = re.sub(r"_rep\d+$", "", run_dir.name)
    rep_match = re.search(r"_rep(\d+)$", run_dir.name)
    bench = first_existing(
        [
            run_dir / "vllm_bench_baseline.log",
            run_dir / "vllm_bench_stage_l5.log",
            run_dir / "vllm_bench_stage_l6.log",
            run_dir / "vllm_bench_stage_l7.log",
            run_dir / "vllm_bench_stage_l7_cross_pool.log",
        ]
    )
    check = read_check(run_dir / "check.json")
    logs = [
        run_dir / "stage_l5_gpu_runner.log",
        run_dir / "stage_l6_gpu_runner.log",
        run_dir / "stage_l7_gpu_runner.log",
        run_dir / "stage_l7_cross_pool_runner.log",
    ]
    if variant == "baseline_pure_uvm":
        logs.append(run_dir / "stage_baseline_runner.log")
    row = {
        "variant": variant,
        "rep": int(rep_match.group(1)) if rep_match else "",
        "run_dir": str(run_dir),
        "bench_log": str(bench) if bench else "",
        "check_json": str(run_dir / "check.json") if (run_dir / "check.json").is_file() else "",
    }
    row.update(parse_bench(bench) if bench else {})
    row.update(parse_faults(logs + [run_dir / "uvm_kv_fault_stats_baseline.log"]))
    row.update(summarize_check(check))
    for path in run_dir.iterdir():
        if path.is_file() and path.suffix in (".log", ".jsonl"):
            row[f"bytes_{path.name}"] = path.stat().st_size
    rows.append(row)

baseline_rows = [r for r in rows if r["variant"] == "baseline_pure_uvm"]
baseline_medians = {}
for key in ("output_tok_s", "mean_tpot_ms", "mean_ttft_ms", "delta_kv_faults", "delta_kv_after_dedup", "delta_after_dedup"):
    vals = [r[key] for r in baseline_rows if isinstance(r.get(key), (int, float))]
    if vals:
        baseline_medians[key] = statistics.median(vals)

for row in rows:
    for key, base in baseline_medians.items():
        val = row.get(key)
        if isinstance(val, (int, float)) and base:
            row[f"{key}_vs_baseline_pct"] = round((val - base) * 100.0 / base, 3)
    if isinstance(row.get("delta_kv_after_dedup"), (int, float)) and baseline_medians.get("delta_kv_after_dedup"):
        base = baseline_medians["delta_kv_after_dedup"]
        row["kv_dedup_fault_reduction_pct"] = round((base - row["delta_kv_after_dedup"]) * 100.0 / base, 3)

summary_csv = run_root / "summary.csv"
fieldnames = sorted({key for row in rows for key in row})
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups = {}
for row in rows:
    groups.setdefault(row["variant"], []).append(row)

def med(rows: list[dict], key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else ""

lines = [
    "# Stage L Fault/Performance Diagnosis",
    "",
    f"- run_root: `{run_root}`",
    f"- runs: {len(rows)}",
    f"- baseline_medians: `{json.dumps(baseline_medians, sort_keys=True)}`",
    "",
    "## Median By Variant",
    "",
    "| variant | failed | output_tok_s | mean_tpot_ms | mean_ttft_ms | kv_faults | kv_dedup_faults | kv_dedup_reduction_vs_baseline | offload | prefetch | coordinator_requests |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for variant in sorted(groups):
    rs = groups[variant]
    failed = med(rs, "failed_requests")
    kv_reduction = med(rs, "kv_dedup_fault_reduction_pct")
    offload = med(rs, "kv_data_offload_issued_records") or med(rs, "residency_offload_issued")
    prefetch = med(rs, "kv_prefetch_back_issued_records") or med(rs, "residency_prefetch_issued")
    lines.append(
        f"| {variant} | {failed} | {med(rs, 'output_tok_s')} | {med(rs, 'mean_tpot_ms')} | "
        f"{med(rs, 'mean_ttft_ms')} | {med(rs, 'delta_kv_faults')} | {med(rs, 'delta_kv_after_dedup')} | "
        f"{kv_reduction}% | {offload} | {prefetch} | {med(rs, 'coordinator_requests')} |"
    )

lines.extend(
    [
        "",
        "## How To Read This",
        "",
        "- `baseline_pure_uvm` is the pure-UVM reference with Stage L/J/K disabled and `VLLM_UVM_POLICY_ENABLE=0`.",
        "- `l5_metadata` isolates metadata/runtime candidate overhead.",
        "- `l6_decision_trace` isolates safe-candidate and offload decision overhead without real CPU/GPU prefetch actions.",
        "- `l6_real_prefetch` adds real KV prefetch/offload calls.",
        "- `l7_coordinator` adds Stage K action-budget coordinator overhead.",
        "- If KV faults drop but `mean_tpot_ms` rises, compare the first variant where TPOT rises sharply; that is the likely overhead layer.",
        "",
        f"- CSV: `{summary_csv}`",
    ]
)
(run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
PY

echo "==========================================================="
echo " Done"
echo " Summary:"
echo " - $RUN_ROOT/summary.md"
echo " - $RUN_ROOT/summary.csv"
echo " Full log:"
echo " - $LOG_FILE"
echo "==========================================================="
