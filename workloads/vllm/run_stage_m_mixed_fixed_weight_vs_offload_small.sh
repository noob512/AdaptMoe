#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/mixed_fixed_weight_vs_offload_small_$(date -u +%Y%m%d_%H%M%S)}"
KV_BYTES="${KV_BYTES:-7516192768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PROMPTS="${PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
KV_DEVICE_DIRECT_BACKEND="${KV_DEVICE_DIRECT_BACKEND:-cuda_malloc}"
FIXED_WEIGHT_DEVICE_DIRECT_MAX_TOTAL_BYTES="${FIXED_WEIGHT_DEVICE_DIRECT_MAX_TOTAL_BYTES:-0}"
PORT_BASE="${PORT_BASE:-8930}"
TOP_N="${TOP_N:-20}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  run_stage_m_mixed_fixed_weight_vs_offload_small.sh

Compares two small-sample modes under the same explicit KV cache capacity:
  1. mixed_fixed_native_expert_uvm:
     global UVM enabled, expert weights remain cudaMallocManaged,
     non-expert fixed weights are rematerialized through device-direct
     cudaMalloc, and KV cache is device-direct cudaMalloc.
  2. native_cpu_offload:
     ordinary vLLM CUDA allocator with --cpu-offload-gb and the same
     --kv-cache-memory-bytes.

Environment defaults:
  KV_BYTES=7516192768  # 7 GiB
  MAX_NUM_SEQS=32
  PROMPTS=16 OUTPUT_LEN=16 REQUEST_RATE=8
  CPU_OFFLOAD_GB=8
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --kv-bytes) KV_BYTES="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$REPO_ROOT/$RUN_ROOT" ;;
esac

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
COMMANDS_LOG="$RUN_ROOT/commands.log"
: > "$COMMANDS_LOG"

append_command() {
  printf 'command:' >> "$COMMANDS_LOG"
  printf ' %q' "$@" >> "$COMMANDS_LOG"
  printf '\n' >> "$COMMANDS_LOG"
}

run_mixed_case() {
  local case_dir="$RUN_ROOT/mixed_fixed_native_expert_uvm"
  mkdir -p "$case_dir"

  local fixed_trace="$case_dir/vllm_uvm_fixed_weight_device_direct.jsonl"
  local weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"

  local extra_env
  extra_env="HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=1 \
VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MIN_BYTES=0 \
VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MAX_BYTES=0 \
VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_MAX_TOTAL_BYTES=$FIXED_WEIGHT_DEVICE_DIRECT_MAX_TOTAL_BYTES \
VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_TRACE_FILE='$fixed_trace' \
VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=0"

  local -a cmd=(
    "$SCRIPT_DIR/run_kv_fault_ratio.sh"
    --model "$MODEL"
    --mode trace
    --port "$PORT_BASE"
    --allocator-log "$case_dir/vllm_uvm_allocator.log"
    --trace-log "$case_dir/uvm_fault_stats.log"
    --address-log "$case_dir/vllm_uvm_address_regions.log"
    --server-log "$case_dir/vllm_server.log"
    --bench-log "$case_dir/vllm_bench.log"
    --uvm-policy-enable 1
    --uvm-policy-mode trace_only
    --uvm-kv-device-direct-enable 1
    --uvm-kv-device-direct-min-bytes 0
    --uvm-kv-device-direct-max-bytes 0
    --uvm-kv-device-direct-max-total-bytes 0
    --uvm-kv-range-config-enable 0
    --uvm-device-direct-enable 0
    --uvm-device-direct-backend "$KV_DEVICE_DIRECT_BACKEND"
    --uvm-residency-enable 0
    --uvm-kv-runtime-enable 0
    --uvm-weight-map-enable 1
    --uvm-weight-map-file "$weight_map"
    --uvm-weight-prefetch-enable 0
    --uvm-weight-offload-enable 0
    --uvm-weight-batch-executor-enable 0
    --uvm-pool-coordinator-enable 0
    --uvm-pool-registry-enable 0
    --uvm-scratch-pool-enable 0
    --uvm-moe-routing-trace-enable 0
    --server-max-num-seqs "$MAX_NUM_SEQS"
    --max-model-len "$MAX_MODEL_LEN"
    --server-kv-cache-memory-bytes "$KV_BYTES"
    --server-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --server-extra-env "$extra_env"
    --prompts "$PROMPTS"
    --request-rate "$REQUEST_RATE"
    --output-len "$OUTPUT_LEN"
  )

  append_command "${cmd[@]}"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] mixed_fixed_native_expert_uvm\n'
    return
  fi
  "${cmd[@]}"

  local -a weight_analyze_cmd=(
    python3 "$SCRIPT_DIR/analyze_vllm_weight_address_map.py"
    --weight-map "$weight_map"
    --stats-log "$case_dir/uvm_fault_stats.log"
    --out-dir "$case_dir/weight_address_analysis"
    --top-n "$TOP_N"
  )
  append_command "${weight_analyze_cmd[@]}"
  "${weight_analyze_cmd[@]}" > "$case_dir/weight_address_analysis.log"
}

run_offload_case() {
  local case_dir="$RUN_ROOT/native_cpu_offload_same_kv"
  mkdir -p "$case_dir"
  local -a cmd=(
    env
    "RUN_ROOT=$case_dir"
    "MODEL=$MODEL"
    "KV_CACHE_BYTES=$KV_BYTES"
    "MAX_NUM_SEQS_LIST=$MAX_NUM_SEQS"
    "MAX_MODEL_LEN=$MAX_MODEL_LEN"
    "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
    "CPU_OFFLOAD_GB=$CPU_OFFLOAD_GB"
    "PROMPTS=$PROMPTS"
    "OUTPUT_LEN=$OUTPUT_LEN"
    "REQUEST_RATE=$REQUEST_RATE"
    "PORT_BASE=$((PORT_BASE + 10))"
    "$SCRIPT_DIR/run_stage_m_native_small_kv_concurrency_sweep.sh"
  )
  append_command "${cmd[@]}"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run] native_cpu_offload_same_kv\n'
    return
  fi
  "${cmd[@]}"
}

aggregate_results() {
  python3 - "$RUN_ROOT" "$KV_BYTES" "$MAX_NUM_SEQS" "$PROMPTS" "$OUTPUT_LEN" "$REQUEST_RATE" "$CPU_OFFLOAD_GB" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
target_kv = int(sys.argv[2])
max_num_seqs = int(sys.argv[3])
prompts = int(sys.argv[4])
output_len = int(sys.argv[5])
request_rate = float(sys.argv[6])
cpu_offload_gb = float(sys.argv[7])

def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(text: str, pattern: str):
    m = re.search(pattern, text)
    return float(m.group(1).replace(",", "")) if m else None

def first_int(text: str, pattern: str):
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None

def kv_covered_bytes(address_log: Path):
    text = read(address_log)
    tensor_bytes = [
        int(value)
        for value in re.findall(
            r"^kv_cache,[^,]+,0x[0-9a-fA-F]+,0x[0-9a-fA-F]+,([0-9]+),",
            text,
            flags=re.MULTILINE,
        )
    ]
    if tensor_bytes:
        return sum(tensor_bytes)
    matches = re.findall(
        r"kv_cache:(?:contiguous_range|span_range),all_layers,0x[0-9a-fA-F]+,0x[0-9a-fA-F]+,([0-9]+),",
        text,
    )
    return int(matches[-1]) if matches else None

def fixed_trace_summary(path: Path):
    out = {
        "converted_count": 0,
        "failed_count": 0,
        "converted_bytes": 0,
        "converted_gib": 0.0,
        "materialized_records": 0,
        "new_nonmanaged_records": 0,
        "old_managed_records": 0,
    }
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        event = rec.get("event")
        if event == "fixed_weight_device_direct_summary":
            out["converted_count"] = int(rec.get("converted_count") or 0)
            out["failed_count"] = int(rec.get("failed_count") or 0)
            out["converted_bytes"] = int(rec.get("converted_bytes") or 0)
            out["converted_gib"] = float(rec.get("converted_gib") or 0.0)
        elif event == "fixed_weight_device_direct_materialized":
            out["materialized_records"] += 1
            if rec.get("new_driver_capable") is False:
                out["new_nonmanaged_records"] += 1
            if rec.get("old_driver_capable") is True:
                out["old_managed_records"] += 1
    return out

def weight_summary(path: Path):
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    b = summary.get("bytes", {})
    return {
        "weight_total_gib": (b.get("total_accounted_bytes") or 0) / 1024**3,
        "weight_fixed_gib": (b.get("fixed_weight_bytes") or 0) / 1024**3,
        "weight_expert_gib": (b.get("moe_expert_bytes") or 0) / 1024**3,
        "weight_managed_gib": (b.get("managed_bytes") or 0) / 1024**3,
        "weight_unmanaged_gib": (b.get("unmanaged_or_nonmanaged_bytes") or 0) / 1024**3,
        "driver_reason_counts": b.get("driver_capability_reason_counts", {}),
        "bytes_by_driver_reason": b.get("bytes_by_driver_capability_reason", {}),
        "bytes_by_semantic": b.get("bytes_by_semantic_type", {}),
    }

def parse_uvm_stats(path: Path):
    text = read(path)
    rows = []
    for line in text.splitlines():
        fields = dict((k, int(v)) for k, v in re.findall(r"\b([A-Za-z_]+)=([0-9]+)", line))
        if fields:
            rows.append(fields)
    if not rows:
        return {"stats_records": 0, "delta_after_dedup": None, "delta_kv_after_dedup": None}
    first, last = rows[0], rows[-1]
    total_key = next((k for k in ("total_after_dedup", "after_dedup", "total_replayable_after_dedup") if k in last), None)
    kv_key = next((k for k in ("total_kv_after_dedup", "kv_after_dedup") if k in last), None)
    total = int(last.get(total_key, 0)) - int(first.get(total_key, 0)) if total_key else None
    kv = int(last.get(kv_key, 0)) - int(first.get(kv_key, 0)) if kv_key else None
    return {"stats_records": len(rows), "delta_after_dedup": total, "delta_kv_after_dedup": kv}

def parse_bench(server_log: Path, bench_log: Path):
    server = read(server_log)
    bench = read(bench_log)
    return {
        "model_load_gib": first_float(server, r"Model loading took\s+([0-9.]+)\s+GiB memory"),
        "gpu_kv_cache_tokens": first_int(server, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
        "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
        "max_concurrency_1024": first_float(server, r"Maximum concurrency for 1,024 tokens per request:\s+([0-9.]+)x"),
        "successful_requests": first_int(bench, r"Successful requests:\s+([0-9,]+)"),
        "failed_requests": first_int(bench, r"Failed requests:\s+([0-9,]+)"),
        "request_throughput_req_s": first_float(bench, r"Request throughput \(req/s\):\s+([0-9.]+)"),
        "output_tok_s": first_float(bench, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
        "total_tok_s": first_float(bench, r"Total Token throughput \(tok/s\):\s+([0-9.]+)"),
        "peak_concurrent_requests": first_float(bench, r"Peak concurrent requests:\s+([0-9.]+)"),
        "mean_ttft_ms": first_float(bench, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
        "median_ttft_ms": first_float(bench, r"Median TTFT \(ms\):\s+([0-9.]+)"),
        "p99_ttft_ms": first_float(bench, r"P99 TTFT \(ms\):\s+([0-9.]+)"),
        "mean_tpot_ms": first_float(bench, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
        "median_tpot_ms": first_float(bench, r"Median TPOT \(ms\):\s+([0-9.]+)"),
        "p99_tpot_ms": first_float(bench, r"P99 TPOT \(ms\):\s+([0-9.]+)"),
    }

def allocator_counts(path: Path):
    text = read(path)
    return {
        "kv_device_direct_allocs": len(re.findall(r"kv_device_direct_enabled", text)),
        "fixed_weight_device_direct_allocs": len(re.findall(r"fixed_weight_device_direct_enabled", text)),
        "device_direct_backend_records": len(re.findall(r"placement_backend=device_direct", text)),
        "managed_backend_records": len(re.findall(r"placement_backend=managed", text)),
    }

rows = []
mixed_dir = root / "mixed_fixed_native_expert_uvm"
mixed_weight = weight_summary(mixed_dir / "weight_address_analysis" / "weight_address_map_summary.json")
mixed_fixed = fixed_trace_summary(mixed_dir / "vllm_uvm_fixed_weight_device_direct.jsonl")
mixed = {
    "case": "mixed_fixed_native_expert_uvm",
    "mechanism": "global_uvm_expert_weight + native_fixed_weight + native_kv",
    "target_kv_bytes": target_kv,
    "actual_kv_covered_bytes": kv_covered_bytes(mixed_dir / "vllm_uvm_address_regions.log"),
    **parse_bench(mixed_dir / "vllm_server.log", mixed_dir / "vllm_bench.log"),
    **parse_uvm_stats(mixed_dir / "uvm_fault_stats.log"),
    **allocator_counts(mixed_dir / "vllm_uvm_allocator.log"),
    **mixed_fixed,
    **mixed_weight,
}
rows.append(mixed)

offload_dir = root / "native_cpu_offload_same_kv" / f"s{max_num_seqs}"
if not offload_dir.is_dir():
    parent = root / "native_cpu_offload_same_kv"
    candidates = sorted(parent.glob("s*")) if parent.is_dir() else []
    offload_dir = candidates[0] if candidates else offload_dir
offload = {
    "case": "native_cpu_offload_same_kv",
    "mechanism": f"native_cuda + vllm_cpu_offload_{cpu_offload_gb:g}GiB",
    "target_kv_bytes": target_kv,
    "actual_kv_covered_bytes": None,
    **parse_bench(offload_dir / "vllm_server.log", offload_dir / "vllm_bench.log"),
    "stats_records": None,
    "delta_after_dedup": None,
    "delta_kv_after_dedup": None,
    "kv_device_direct_allocs": None,
    "fixed_weight_device_direct_allocs": None,
    "device_direct_backend_records": None,
    "managed_backend_records": None,
    "converted_count": None,
    "failed_count": None,
    "converted_bytes": None,
    "converted_gib": None,
    "materialized_records": None,
    "new_nonmanaged_records": None,
    "old_managed_records": None,
}
rows.append(offload)

fieldnames = sorted({k for row in rows for k in row})
csv_path = root / "mixed_fixed_weight_vs_offload_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

json_path = root / "mixed_fixed_weight_vs_offload_summary.json"
json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

def fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

def gib(v):
    return None if v is None else float(v) / 1024**3

lines = [
    "# Mixed Fixed Weight vs CPU Offload Small Sample",
    "",
    f"- run_root: `{root}`",
    f"- KV target: `{target_kv}` bytes ({target_kv / 1024**3:.3f} GiB)",
    f"- max_num_seqs: `{max_num_seqs}`",
    f"- prompts/output/request_rate: `{prompts}` / `{output_len}` / `{request_rate:g}`",
    "",
    "## Performance",
    "",
    "| case | KV GiB | KV tokens | success/fail | out tok/s | req/s | TTFT mean ms | TPOT mean ms | peak conc |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {kv_gib} | {kv_tokens} | {ok}/{fail} | {out_tok} | {req_s} | {ttft} | {tpot} | {peak} |".format(
            case=row["case"],
            kv_gib=fmt(gib(row.get("actual_kv_covered_bytes") or target_kv)),
            kv_tokens=fmt(row.get("gpu_kv_cache_tokens")),
            ok=fmt(row.get("successful_requests"), 0),
            fail=fmt(row.get("failed_requests"), 0),
            out_tok=fmt(row.get("output_tok_s")),
            req_s=fmt(row.get("request_throughput_req_s")),
            ttft=fmt(row.get("mean_ttft_ms")),
            tpot=fmt(row.get("mean_tpot_ms")),
            peak=fmt(row.get("peak_concurrent_requests")),
        )
    )

lines += [
    "",
    "## Mixed Allocation Evidence",
    "",
    "| evidence | value |",
    "|---|---:|",
    f"| fixed weights converted | {fmt(mixed.get('converted_count'), 0)} |",
    f"| fixed conversion failures | {fmt(mixed.get('failed_count'), 0)} |",
    f"| fixed converted GiB | {fmt(mixed.get('converted_gib'))} |",
    f"| fixed materialized records with new non-managed ptr | {fmt(mixed.get('new_nonmanaged_records'), 0)} |",
    f"| materialized records whose old ptr was managed | {fmt(mixed.get('old_managed_records'), 0)} |",
    f"| final weight managed GiB | {fmt(mixed.get('weight_managed_gib'))} |",
    f"| final weight non-managed GiB | {fmt(mixed.get('weight_unmanaged_gib'))} |",
    f"| final fixed-weight GiB | {fmt(mixed.get('weight_fixed_gib'))} |",
    f"| final expert-weight GiB | {fmt(mixed.get('weight_expert_gib'))} |",
    f"| allocator fixed_weight_device_direct hits | {fmt(mixed.get('fixed_weight_device_direct_allocs'), 0)} |",
    f"| allocator kv_device_direct hits | {fmt(mixed.get('kv_device_direct_allocs'), 0)} |",
    f"| UVM after-dedup faults | {fmt(mixed.get('delta_after_dedup'), 0)} |",
    f"| UVM KV after-dedup faults | {fmt(mixed.get('delta_kv_after_dedup'), 0)} |",
    "",
    "## Files",
    "",
    f"- summary_csv: `{csv_path}`",
    f"- summary_json: `{json_path}`",
    f"- mixed_weight_report: `{mixed_dir / 'weight_address_analysis' / 'weight_address_map_report.md'}`",
    f"- mixed_fixed_trace: `{mixed_dir / 'vllm_uvm_fixed_weight_device_direct.jsonl'}`",
    f"- commands: `{root / 'commands.log'}`",
]

md_path = root / "mixed_fixed_weight_vs_offload_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY
}

{
  echo "Stage M mixed fixed/native weight vs CPU offload small comparison"
  echo "run_root=$RUN_ROOT"
  echo "model=$MODEL"
  echo "kv_bytes=$KV_BYTES"
  echo "max_num_seqs=$MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"
  echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
  echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
  echo "dry_run=$DRY_RUN"
} | tee "$RUN_ROOT/experiment_header.txt"

run_mixed_case
run_offload_case
if [ "$DRY_RUN" != "1" ]; then
  aggregate_results
fi

echo "Stage M mixed fixed/native weight vs CPU offload comparison finished."
