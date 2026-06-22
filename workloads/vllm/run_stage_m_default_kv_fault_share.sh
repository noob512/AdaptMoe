#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_default_kv_fault_share_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-8}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-0}"
TOP_N="${TOP_N:-20}"
FAULTS_CSV_ENABLE="${FAULTS_CSV_ENABLE:-0}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  run_stage_m_default_kv_fault_share.sh [options]

Runs a pure-UVM/default-vLLM-KV fault-share experiment:
  - does not pass --kv-cache-memory-bytes
  - does not enable init/runtime KV sizing overrides
  - lets vLLM allocate weights first, then size KV cache from remaining memory
  - captures UVM replayable fault stats and fault addresses
  - classifies fault pages against weight/KV address maps

Environment defaults:
  MODEL=Qwen/Qwen3-30B-A3B-FP8
  PROMPTS=32 OUTPUT_LEN=64 REQUEST_RATE=8
  SERVER_GPU_MEMORY_UTILIZATION=0.70
  MAX_NUM_SEQS_LIST="8"
  MAX_NUM_BATCHED_TOKENS_LIST="0"

Examples:
  # Focused reproduction similar to s8_bt0.
  RUN_ROOT=log/default_kv_fault_s8 \
  MAX_NUM_SEQS_LIST="8" MAX_NUM_BATCHED_TOKENS_LIST="0" \
  workloads/vllm/run_stage_m_default_kv_fault_share.sh

  # Full matrix. Warning: fault address logs can be very large.
  RUN_ROOT=log/default_kv_fault_matrix \
  MAX_NUM_SEQS_LIST="1 4 8 16 32 64" \
  MAX_NUM_BATCHED_TOKENS_LIST="0 1024 2048" \
  workloads/vllm/run_stage_m_default_kv_fault_share.sh

Options:
  --model <name>
  --run-root <path>
  --prompts <n>
  --output-len <n>
  --request-rate <rps>
  --server-gpu-memory-utilization <f>
  --max-num-seqs-list "1 4 8"
  --max-num-batched-tokens-list "0 1024"
  --top-n <n>
  --faults-csv-enable <0|1>
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
    --top-n) TOP_N="$2"; shift 2 ;;
    --faults-csv-enable) FAULTS_CSV_ENABLE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$REPO_ROOT/$RUN_ROOT" ;;
esac

mkdir -p "$RUN_ROOT"
COMMANDS_LOG="$RUN_ROOT/commands.log"
: > "$COMMANDS_LOG"

append_command() {
  printf 'command:' >> "$COMMANDS_LOG"
  printf ' %q' "$@" >> "$COMMANDS_LOG"
  printf '\n' >> "$COMMANDS_LOG"
}

bt_args() {
  local bt="$1"
  if [ "$bt" != "0" ]; then
    printf '%s\0%s\0' --server-max-num-batched-tokens "$bt"
  fi
}

run_case() {
  local seqs="$1"
  local bt="$2"
  local case_name="s${seqs}_bt${bt}"
  local case_dir="$RUN_ROOT/$case_name"
  mkdir -p "$case_dir"

  local -a extra_bt_args=()
  if [ "$bt" != "0" ]; then
    extra_bt_args=(--server-max-num-batched-tokens "$bt")
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
    --uvm-policy-enable 0
    --uvm-device-direct-enable 0
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
  python3 - "$RUN_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def pct(num, den):
    if den in (None, 0) or num is None:
        return None
    return num * 100.0 / den


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def first_float(text, pattern):
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def first_int(text, pattern):
    value = first_float(text, pattern)
    return int(value) if value is not None else None


rows = []
for summary_path in sorted(root.glob("*/fault_address_summary.json")):
    case_dir = summary_path.parent
    case = case_dir.name
    data = json.loads(summary_path.read_text())
    fault = data.get("fault_summary", {})
    stats = data.get("stats_summary") or {}
    counts = fault.get("classification_counts", {})
    parsed = fault.get("parsed_fault_records") or 0
    classified = fault.get("classified_fault_records") or 0
    weight = counts.get("weight", 0)
    kv = counts.get("kv_cache", 0)
    unknown = counts.get("unknown", 0)
    unknown_kv_span = counts.get("unknown_in_kv_summary_span", 0)
    known = weight + kv
    server_log = case_dir / "vllm_server.log"
    server_text = server_log.read_text(errors="replace") if server_log.exists() else ""
    bench_log = case_dir / "vllm_bench.log"
    bench_text = bench_log.read_text(errors="replace") if bench_log.exists() else ""
    rows.append(
        {
            "case": case,
            "output_tok_s": first_float(bench_text, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
            "mean_ttft_ms": first_float(bench_text, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
            "mean_tpot_ms": first_float(bench_text, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
            "available_kv_cache_gib": first_float(server_text, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
            "gpu_kv_cache_tokens": first_int(server_text, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
            "raw_fault_records": parsed,
            "unique_fault_pages": classified,
            "duplicate_records_skipped": fault.get("duplicate_records_skipped"),
            "unique_page_pct_of_raw": pct(classified, parsed),
            "weight_pages": weight,
            "kv_cache_pages": kv,
            "unknown_pages": unknown,
            "unknown_kv_span_pages": unknown_kv_span,
            "weight_page_pct": pct(weight, classified),
            "kv_cache_page_pct": pct(kv, classified),
            "unknown_page_pct": pct(unknown + unknown_kv_span, classified),
            "known_weight_pct": pct(weight, known),
            "known_kv_cache_pct": pct(kv, known),
            "delta_faults": stats.get("delta_faults"),
            "delta_after_dedup": stats.get("delta_after_dedup"),
            "after_dedup_pct_of_faults": pct(stats.get("delta_after_dedup"), stats.get("delta_faults")),
            "delta_kv_after_dedup": stats.get("delta_kv_after_dedup"),
            "kv_after_dedup_pct": stats.get("delta_kv_after_dedup_ratio_pct"),
        }
    )

fieldnames = [
    "case",
    "output_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "available_kv_cache_gib",
    "gpu_kv_cache_tokens",
    "raw_fault_records",
    "unique_fault_pages",
    "duplicate_records_skipped",
    "unique_page_pct_of_raw",
    "weight_pages",
    "kv_cache_pages",
    "unknown_pages",
    "unknown_kv_span_pages",
    "weight_page_pct",
    "kv_cache_page_pct",
    "unknown_page_pct",
    "known_weight_pct",
    "known_kv_cache_pct",
    "delta_faults",
    "delta_after_dedup",
    "after_dedup_pct_of_faults",
    "delta_kv_after_dedup",
    "kv_after_dedup_pct",
]
csv_path = root / "default_kv_fault_share_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

md_path = root / "default_kv_fault_share_summary.md"
lines = [
    "# Default KV Fault Share Summary",
    "",
    "- This run does not pass `--kv-cache-memory-bytes`.",
    "- vLLM allocates/model-loads weights first, then sizes KV cache from remaining memory.",
    f"- summary_csv: `{csv_path}`",
    "",
    "| case | tok/s | TPOT ms | KV GiB | KV tokens | unique/raw % | weight page % | KV page % | unknown page % | after/raw % | KV after % |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {tok} | {tpot} | {kv_gib} | {kv_tokens} | {unique} | {weight} | {kv} | {unknown} | {after} | {kv_after} |".format(
            case=row["case"],
            tok=fmt(row["output_tok_s"]),
            tpot=fmt(row["mean_tpot_ms"]),
            kv_gib=fmt(row["available_kv_cache_gib"]),
            kv_tokens=fmt(row["gpu_kv_cache_tokens"]),
            unique=fmt(row["unique_page_pct_of_raw"]),
            weight=fmt(row["weight_page_pct"]),
            kv=fmt(row["kv_cache_page_pct"]),
            unknown=fmt(row["unknown_page_pct"]),
            after=fmt(row["after_dedup_pct_of_faults"]),
            kv_after=fmt(row["kv_after_dedup_pct"]),
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Summary: {md_path}")
PY
}

echo "==========================================================="
echo "Stage M Default KV Fault Share"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS_LIST"
echo "gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "dry_run=$DRY_RUN"
echo "==========================================================="

for seqs in $MAX_NUM_SEQS_LIST; do
  for bt in $MAX_NUM_BATCHED_TOKENS_LIST; do
    run_case "$seqs" "$bt"
  done
done

if [ "$DRY_RUN" != "1" ]; then
  aggregate_results
fi

echo "Done. Results: $RUN_ROOT"
