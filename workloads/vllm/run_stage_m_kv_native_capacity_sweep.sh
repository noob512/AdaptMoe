#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_kv_native_capacity_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-8 16 32}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-0}"
KV_BYTES_LIST="${KV_BYTES_LIST:-6442450944 7516192768 8589934592 9663676416}"
KV_ALLOC_MODES="${KV_ALLOC_MODES:-uvm native}"
KV_DEVICE_DIRECT_BACKEND="${KV_DEVICE_DIRECT_BACKEND:-cuda_malloc}"
KV_DEVICE_DIRECT_MIN_BYTES="${KV_DEVICE_DIRECT_MIN_BYTES:-0}"
KV_DEVICE_DIRECT_MAX_BYTES="${KV_DEVICE_DIRECT_MAX_BYTES:-0}"
KV_DEVICE_DIRECT_MAX_TOTAL_BYTES="${KV_DEVICE_DIRECT_MAX_TOTAL_BYTES:-0}"
TOP_N="${TOP_N:-20}"
FAULTS_CSV_ENABLE="${FAULTS_CSV_ENABLE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  run_stage_m_kv_native_capacity_sweep.sh [options]

Sweeps explicit KV cache capacity and compares:
  allocation=uvm    : KV stays in the global UVM managed allocator
  allocation=native : initialize_kv_cache/KvPersistent is replaced by native
                      cudaMalloc/cudaMallocAsync through the UVM allocator's
                      KV device-direct path

Environment defaults:
  MODEL=Qwen/Qwen3-30B-A3B-FP8
  PROMPTS=32 OUTPUT_LEN=64 REQUEST_RATE=8
  SERVER_GPU_MEMORY_UTILIZATION=0.95
  MAX_NUM_SEQS_LIST="8 16 32"
  MAX_NUM_BATCHED_TOKENS_LIST="0"
  KV_BYTES_LIST="6442450944 7516192768 8589934592 9663676416"
  KV_ALLOC_MODES="uvm native"
  KV_DEVICE_DIRECT_BACKEND=cuda_malloc

Examples:
  # Small smoke/dry run.
  RUN_ROOT=/tmp/kv_native_smoke DRY_RUN=1 \
  KV_BYTES_LIST="6442450944 7516192768" MAX_NUM_SEQS_LIST="8" \
  workloads/vllm/run_stage_m_kv_native_capacity_sweep.sh

  # Focused H1 matrix.
  RUN_ROOT=log/kv_native_capacity_h1 \
  KV_BYTES_LIST="6442450944 7516192768 8589934592" \
  MAX_NUM_SEQS_LIST="8 16 32" \
  workloads/vllm/run_stage_m_kv_native_capacity_sweep.sh

Options:
  --model <name>
  --run-root <path>
  --prompts <n>
  --output-len <n>
  --request-rate <rps>
  --server-gpu-memory-utilization <f>
  --max-num-seqs-list "8 16 32"
  --max-num-batched-tokens-list "0 1024"
  --kv-bytes-list "6442450944 7516192768"
  --kv-alloc-modes "uvm native"
  --kv-device-direct-backend <cuda_malloc|cuda_malloc_async>
  --dry-run <0|1>
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-num-seqs-list) MAX_NUM_SEQS_LIST="$2"; shift 2 ;;
    --max-num-batched-tokens-list) MAX_NUM_BATCHED_TOKENS_LIST="$2"; shift 2 ;;
    --kv-bytes-list) KV_BYTES_LIST="$2"; shift 2 ;;
    --kv-alloc-modes) KV_ALLOC_MODES="$2"; shift 2 ;;
    --kv-device-direct-backend) KV_DEVICE_DIRECT_BACKEND="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$REPO_ROOT/$RUN_ROOT" ;;
esac

[[ "$DRY_RUN" =~ ^[01]$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$KV_DEVICE_DIRECT_BACKEND" =~ ^(cuda_malloc|cuda_malloc_async)$ ]] || {
  echo "KV_DEVICE_DIRECT_BACKEND must be cuda_malloc or cuda_malloc_async" >&2
  exit 2
}

mkdir -p "$RUN_ROOT"
COMMANDS_LOG="$RUN_ROOT/commands.log"
CASES_CSV="$RUN_ROOT/cases.csv"
SUMMARY_CSV="$RUN_ROOT/kv_native_capacity_summary.csv"
: > "$COMMANDS_LOG"
printf 'case,mode,target_kv_bytes,max_num_seqs,max_num_batched_tokens\n' > "$CASES_CSV"

append_command() {
  printf 'command:' >> "$COMMANDS_LOG"
  printf ' %q' "$@" >> "$COMMANDS_LOG"
  printf '\n' >> "$COMMANDS_LOG"
}

run_case() {
  local mode="$1"
  local kv_bytes="$2"
  local seqs="$3"
  local bt="$4"
  local case_name="${mode}_kv${kv_bytes}_s${seqs}_bt${bt}"
  local case_dir="$RUN_ROOT/$case_name"
  mkdir -p "$case_dir"

  printf '%s,%s,%s,%s,%s\n' "$case_name" "$mode" "$kv_bytes" "$seqs" "$bt" >> "$CASES_CSV"

  local -a extra_bt_args=()
  if [ "$bt" != "0" ]; then
    extra_bt_args=(--server-max-num-batched-tokens "$bt")
  fi

  local -a server_kv_args=()
  if [ "$kv_bytes" != "0" ]; then
    server_kv_args=(--server-kv-cache-memory-bytes "$kv_bytes")
  fi

  local -a kv_direct_args=()
  if [ "$mode" = "native" ]; then
    kv_direct_args=(
      --uvm-kv-device-direct-enable 1
      --uvm-kv-device-direct-min-bytes "$KV_DEVICE_DIRECT_MIN_BYTES"
      --uvm-kv-device-direct-max-bytes "$KV_DEVICE_DIRECT_MAX_BYTES"
      --uvm-kv-device-direct-max-total-bytes "$KV_DEVICE_DIRECT_MAX_TOTAL_BYTES"
      --uvm-kv-range-config-enable 0
      --uvm-device-direct-enable 0
      --uvm-device-direct-backend "$KV_DEVICE_DIRECT_BACKEND"
    )
  elif [ "$mode" = "uvm" ]; then
    kv_direct_args=(
      --uvm-kv-device-direct-enable 0
      --uvm-kv-range-config-enable 1
      --uvm-device-direct-enable 0
      --uvm-device-direct-backend "$KV_DEVICE_DIRECT_BACKEND"
    )
  else
    echo "Unknown KV allocation mode: $mode" >&2
    exit 2
  fi

  local -a run_cmd=(
    "$SCRIPT_DIR/run_kv_fault_ratio.sh"
    --model "$MODEL"
    --mode trace
    --with-address-log
    --allocator-log "$case_dir/vllm_uvm_allocator.log"
    --trace-log "$case_dir/uvm_fault_stats.log"
    --address-trace-log "$case_dir/uvm_fault_addrs.log"
    --address-log "$case_dir/vllm_uvm_address_regions.log"
    --server-log "$case_dir/vllm_server.log"
    --bench-log "$case_dir/vllm_bench.log"
    --uvm-policy-enable 1
    "${kv_direct_args[@]}"
    --uvm-residency-enable 0
    --uvm-kv-runtime-enable 0
    --uvm-weight-prefetch-enable 0
    --uvm-weight-offload-enable 0
    --uvm-weight-batch-executor-enable 0
    --uvm-pool-coordinator-enable 0
    --uvm-pool-registry-enable 0
    --uvm-scratch-pool-enable 0
    --uvm-moe-routing-trace-enable 0
    --server-max-num-seqs "$seqs"
    "${extra_bt_args[@]}"
    "${server_kv_args[@]}"
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION"
    --prompts "$PROMPTS"
    --request-rate "$REQUEST_RATE"
    --output-len "$OUTPUT_LEN"
  )

  append_command "${run_cmd[@]}"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] %s\n' "$case_name"
    return
  fi

  "${run_cmd[@]}"

  local -a analyze_cmd=(
    python3 "$SCRIPT_DIR/analyze_uvm_fault_addresses.py"
    --address-log "$case_dir/vllm_uvm_address_regions.log"
    --fault-log "$case_dir/uvm_fault_addrs.log"
    --stats-log "$case_dir/uvm_fault_stats.log"
    --dedup-mode global-page
    --summary-json "$case_dir/fault_address_summary.json"
    --regions-csv "$case_dir/fault_regions.csv"
    --top-n "$TOP_N"
  )
  if [ "$FAULTS_CSV_ENABLE" = "1" ]; then
    analyze_cmd+=(--faults-csv "$case_dir/fault_classification.csv")
  fi

  append_command "${analyze_cmd[@]}"
  "${analyze_cmd[@]}" > "$case_dir/fault_address_report.txt"
}

aggregate_results() {
  python3 - "$RUN_ROOT" "$CASES_CSV" "$SUMMARY_CSV" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases_csv = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])


def pct(num, den):
    if num is None or not den:
        return None
    return num * 100.0 / den


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def first_float(text: str, pattern: str):
    match = re.search(pattern, text)
    return float(match.group(1).replace(",", "")) if match else None


def first_int(text: str, pattern: str):
    value = first_float(text, pattern)
    return int(value) if value is not None else None


def allocator_kv_placement(path: Path):
    counts = {"device_direct": 0, "managed": 0, "other": 0}
    bytes_by_backend = {"device_direct": 0, "managed": 0, "other": 0}
    reasons = {}
    if not path.exists():
        return counts, bytes_by_backend, reasons
    for line in path.read_text(errors="replace").splitlines():
        if "predicted_class=kv_persistent" not in line:
            continue
        backend_match = re.search(r"placement_backend=([^ ]+)", line)
        backend = backend_match.group(1) if backend_match else "other"
        if backend not in counts:
            backend = "other"
        counts[backend] += 1
        size_match = re.search(r"size_bytes=(\d+)", line)
        if size_match:
            bytes_by_backend[backend] += int(size_match.group(1))
        reason_match = re.search(r"device_direct_reason=([^ ]+)", line)
        if reason_match:
            reason = reason_match.group(1)
            reasons[reason] = reasons.get(reason, 0) + 1
    return counts, bytes_by_backend, reasons


rows = []
with cases_csv.open(newline="", encoding="utf-8") as handle:
    for meta in csv.DictReader(handle):
        case = meta["case"]
        case_dir = root / case
        summary_path = case_dir / "fault_address_summary.json"
        data = {}
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
        fault = data.get("fault_summary", {})
        stats = data.get("stats_summary") or {}
        region = data.get("region_summary", {})
        region_types = region.get("types", {})
        kv_region = region_types.get("kv_cache", {})

        counts = fault.get("classification_counts", {})
        unique_fault_pages = fault.get("classified_fault_records")
        weight_fault_pages = counts.get("weight", 0)
        kv_fault_pages = counts.get("kv_cache", 0)
        unknown_fault_pages = counts.get("unknown", 0)
        known_fault_pages = weight_fault_pages + kv_fault_pages

        server_text = (case_dir / "vllm_server.log").read_text(errors="replace") if (case_dir / "vllm_server.log").exists() else ""
        bench_text = (case_dir / "vllm_bench.log").read_text(errors="replace") if (case_dir / "vllm_bench.log").exists() else ""
        allocator_counts, allocator_bytes, allocator_reasons = allocator_kv_placement(case_dir / "vllm_uvm_allocator.log")

        rows.append({
            **meta,
            "actual_kv_covered_bytes": kv_region.get("merged_covered_bytes"),
            "kv_region_count": kv_region.get("region_count"),
            "available_kv_cache_gib": first_float(server_text, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
            "gpu_kv_cache_tokens": first_int(server_text, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
            "output_tok_s": first_float(bench_text, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
            "mean_ttft_ms": first_float(bench_text, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
            "mean_tpot_ms": first_float(bench_text, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
            "unique_fault_pages": unique_fault_pages,
            "weight_fault_pages": weight_fault_pages,
            "kv_fault_pages": kv_fault_pages,
            "unknown_fault_pages": unknown_fault_pages,
            "weight_fault_pct": pct(weight_fault_pages, unique_fault_pages),
            "kv_fault_pct": pct(kv_fault_pages, unique_fault_pages),
            "unknown_fault_pct": pct(unknown_fault_pages, unique_fault_pages),
            "weight_known_fault_pct": pct(weight_fault_pages, known_fault_pages),
            "kv_known_fault_pct": pct(kv_fault_pages, known_fault_pages),
            "delta_after_dedup": stats.get("delta_after_dedup"),
            "delta_kv_after_dedup": stats.get("delta_kv_after_dedup"),
            "delta_kv_after_dedup_ratio_pct": stats.get("delta_kv_after_dedup_ratio_pct"),
            "kv_alloc_records_device_direct": allocator_counts["device_direct"],
            "kv_alloc_records_managed": allocator_counts["managed"],
            "kv_alloc_bytes_device_direct": allocator_bytes["device_direct"],
            "kv_alloc_bytes_managed": allocator_bytes["managed"],
            "kv_device_direct_reasons": ";".join(
                f"{key}:{value}" for key, value in sorted(allocator_reasons.items())
            ),
        })

fieldnames = [
    "case", "mode", "target_kv_bytes", "max_num_seqs", "max_num_batched_tokens",
    "actual_kv_covered_bytes", "available_kv_cache_gib", "gpu_kv_cache_tokens",
    "kv_region_count", "kv_alloc_records_device_direct", "kv_alloc_records_managed",
    "kv_alloc_bytes_device_direct", "kv_alloc_bytes_managed", "kv_device_direct_reasons",
    "output_tok_s", "mean_ttft_ms", "mean_tpot_ms",
    "unique_fault_pages", "weight_fault_pages", "kv_fault_pages", "unknown_fault_pages",
    "weight_fault_pct", "kv_fault_pct", "unknown_fault_pct",
    "weight_known_fault_pct", "kv_known_fault_pct",
    "delta_after_dedup", "delta_kv_after_dedup", "delta_kv_after_dedup_ratio_pct",
]

with summary_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: fmt(row.get(name)) for name in fieldnames})

print(f"Wrote {summary_csv}")
PY
}

echo "==========================================================="
echo "Stage M KV Native Capacity Sweep"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "kv_bytes=$KV_BYTES_LIST"
echo "kv_alloc_modes=$KV_ALLOC_MODES"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS_LIST"
echo "dry_run=$DRY_RUN"
echo "==========================================================="

for mode in $KV_ALLOC_MODES; do
  for kv_bytes in $KV_BYTES_LIST; do
    for seqs in $MAX_NUM_SEQS_LIST; do
      for bt in $MAX_NUM_BATCHED_TOKENS_LIST; do
        run_case "$mode" "$kv_bytes" "$seqs" "$bt"
      done
    done
  done
done

if [ "$DRY_RUN" = "0" ]; then
  aggregate_results
else
  echo "Dry run complete. Commands: $COMMANDS_LOG"
fi

echo "Outputs:"
echo "- cases: $CASES_CSV"
echo "- commands: $COMMANDS_LOG"
echo "- summary: $SUMMARY_CSV"
