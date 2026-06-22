#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/small_kv_one_prompt_fault_share_$(date -u +%Y%m%d_%H%M%S)}"
CASES="${CASES:-hybrid_auto_kv}"
CLIENT_MODE="${CLIENT_MODE:-bench}"
STRICT_PROMPT="${STRICT_PROMPT:-Explain UVM page faults in one short sentence.}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
KV_CACHE_BYTES_0P5G="${KV_CACHE_BYTES_0P5G:-536870912}"
CAPTURE_STARTUP_FAULTS="${CAPTURE_STARTUP_FAULTS:-0}"
TOP_N="${TOP_N:-20}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  run_stage_m_small_kv_one_prompt_fault_share.sh [options]

Runs focused one-prompt UVM fault attribution for small-KV conditions.

Default case:
  hybrid_auto_kv
    Reproduces the old hybrid eager path:
    VLLM_CPU_OFFLOAD_BACKEND=isolated_uvm_hybrid_remote,
    --cpu-offload-gb=8, no --kv-cache-memory-bytes. KV is auto-sized.

Other supported cases:
  uvm_auto_kv
  uvm_explicit_0p5g
  hybrid_explicit_0p5g

Note:
  uvm_explicit_0p5g is diagnostic only. In the current global-UVM path, vLLM may
  override explicit --kv-cache-memory-bytes and set KV to the UVM default size.

Environment defaults:
  PROMPTS=1 OUTPUT_LEN=1 REQUEST_RATE=1
  MAX_MODEL_LEN=1024 MAX_NUM_SEQS=1 GPU_MEMORY_UTILIZATION=0.95
  CAPTURE_STARTUP_FAULTS=0

Examples:
  RUN_ROOT=log/small_kv_one_prompt_fault_share_first \
  CASES="hybrid_auto_kv" \
  workloads/vllm/run_stage_m_small_kv_one_prompt_fault_share.sh

  # Include model-load/KV-init faults in the trace stream too.
  CAPTURE_STARTUP_FAULTS=1 \
  RUN_ROOT=log/small_kv_one_prompt_fault_share_startup \
  workloads/vllm/run_stage_m_small_kv_one_prompt_fault_share.sh

Options:
  --model <name>
  --run-root <path>
  --cases "hybrid_auto_kv uvm_explicit_0p5g"
  --client-mode <bench|strict_one_prompt>
  --strict-prompt <text>
  --prompts <n>
  --output-len <n>
  --request-rate <rps>
  --max-model-len <n>
  --max-num-seqs <n>
  --max-num-batched-tokens <n>
  --gpu-memory-utilization <f>
  --cpu-offload-gb <f>
  --kv-cache-bytes-0p5g <n>
  --capture-startup-faults <0|1>
  --top-n <n>
  --dry-run <0|1>
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --cases) CASES="$2"; shift 2 ;;
    --client-mode) CLIENT_MODE="$2"; shift 2 ;;
    --strict-prompt) STRICT_PROMPT="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --cpu-offload-gb) CPU_OFFLOAD_GB="$2"; shift 2 ;;
    --kv-cache-bytes-0p5g) KV_CACHE_BYTES_0P5G="$2"; shift 2 ;;
    --capture-startup-faults) CAPTURE_STARTUP_FAULTS="$2"; shift 2 ;;
    --top-n) TOP_N="$2"; shift 2 ;;
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

hybrid_extra_env() {
  local manifest="$1"
  printf '%s' \
    "VLLM_USE_UVM=0 " \
    "VLLM_CPU_OFFLOAD_BACKEND=isolated_uvm_hybrid_remote " \
    "VLLM_CPU_OFFLOAD_UVM_SYNC=1 " \
    "VLLM_CPU_OFFLOAD_MANIFEST_FILE=$manifest " \
    "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 " \
    "VLLM_UVM_PHASE_F_IOCTL_MODE=policy " \
    "VLLM_UVM_PHASE_F_FD_MODE=adopt " \
    "VLLM_UVM_PHASE_F_CPP_HELPER=1 " \
    "VLLM_UVM_HYBRID_ROUND_PAGES=1 " \
    "VLLM_UVM_HYBRID_ROUND_READMOSTLY=1 " \
    "VLLM_UVM_HYBRID_START_MODE=eager " \
    "VLLM_UVM_HYBRID_GATE_ENABLE=0"
}

run_case() {
  local case_name="$1"
  local case_dir="$RUN_ROOT/$case_name"
  local extra_env=""
  local -a server_extra_args=()
  local -a max_bt_args=()

  mkdir -p "$case_dir"

  if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
    max_bt_args=(--server-max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi

  case "$case_name" in
    hybrid_auto_kv)
      extra_env="$(hybrid_extra_env "$case_dir/offload_manifest.jsonl")"
      server_extra_args=(--server-cpu-offload-gb "$CPU_OFFLOAD_GB")
      ;;
    hybrid_explicit_0p5g)
      extra_env="$(hybrid_extra_env "$case_dir/offload_manifest.jsonl")"
      server_extra_args=(
        --server-cpu-offload-gb "$CPU_OFFLOAD_GB"
        --server-kv-cache-memory-bytes "$KV_CACHE_BYTES_0P5G"
      )
      ;;
    uvm_explicit_0p5g)
      server_extra_args=(--server-kv-cache-memory-bytes "$KV_CACHE_BYTES_0P5G")
      ;;
    uvm_auto_kv)
      server_extra_args=()
      ;;
    *)
      echo "ERROR: unsupported case: $case_name" >&2
      exit 2
      ;;
  esac

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
    --uvm-weight-map-enable 1
    --uvm-weight-prefetch-enable 0
    --uvm-weight-offload-enable 0
    --uvm-weight-batch-executor-enable 0
    --uvm-pool-coordinator-enable 0
    --uvm-pool-registry-enable 0
    --uvm-scratch-pool-enable 0
    --uvm-moe-routing-trace-enable 0
    --capture-startup-faults "$CAPTURE_STARTUP_FAULTS"
    --server-max-num-seqs "$MAX_NUM_SEQS"
    "${max_bt_args[@]}"
    --server-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --prompts "$PROMPTS"
    --request-rate "$REQUEST_RATE"
    --output-len "$OUTPUT_LEN"
    "${server_extra_args[@]}"
  )

  if [ -n "$extra_env" ]; then
    run_cmd+=(--server-extra-env "$extra_env")
  fi
  if [ "$CLIENT_MODE" = "strict_one_prompt" ]; then
    run_cmd+=(
      --
      python3 -c
      'import json, sys, time, urllib.request
model = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
payload = {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
data = json.dumps(payload).encode()
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=data,
    headers={"Content-Type": "application/json"},
)
start = time.time()
response = urllib.request.urlopen(request, timeout=120)
body = response.read().decode()
print("strict_one_prompt_status", response.status)
print("strict_one_prompt_ms", round((time.time() - start) * 1000, 2))
print(body[:1000])'
      "$MODEL"
      "$STRICT_PROMPT"
      "$OUTPUT_LEN"
      --num-prompts "$PROMPTS"
    )
  fi

  {
    echo "case=$case_name"
    echo "model=$MODEL"
    echo "prompts=$PROMPTS"
    echo "client_mode=$CLIENT_MODE"
    echo "strict_prompt=$STRICT_PROMPT"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "kv_cache_bytes_0p5g=$KV_CACHE_BYTES_0P5G"
    echo "capture_startup_faults=$CAPTURE_STARTUP_FAULTS"
    echo "server_extra_env=$extra_env"
  } > "$case_dir/case_metadata.txt"

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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def first_float(text: str, pattern: str):
    match = re.search(pattern, text)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def first_int(text: str, pattern: str):
    value = first_float(text, pattern)
    return int(value) if value is not None else None


def pct(num, den):
    if num is None or not den:
        return None
    return 100.0 * num / den


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def manifest_summary(path: Path):
    out = {
        "manifest_params": 0,
        "manifest_bytes": 0,
        "manifest_managed": 0,
        "manifest_eager_start_events": 0,
        "manifest_eager_start_ranges": 0,
    }
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        event = rec.get("event")
        if event == "cpu_offload_parameter":
            out["manifest_params"] += 1
            out["manifest_bytes"] += int(rec.get("bytes") or 0)
            out["manifest_managed"] += int(rec.get("uvm_managed") is True)
        elif event == "cpu_offload_hybrid_remote_eager_start":
            out["manifest_eager_start_events"] += 1
            out["manifest_eager_start_ranges"] += int(rec.get("start_success") or 0)
    return out


rows = []
for summary_path in sorted(root.glob("*/fault_address_summary.json")):
    case_dir = summary_path.parent
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    fault = data.get("fault_summary", {})
    stats = data.get("stats_summary") or {}
    counts = fault.get("classification_counts", {})
    classified = fault.get("classified_fault_records") or 0
    parsed = fault.get("parsed_fault_records") or 0
    weight = counts.get("weight", 0)
    kv = counts.get("kv_cache", 0)
    unknown = counts.get("unknown", 0) + counts.get("unknown_in_kv_summary_span", 0)
    known = weight + kv
    server_text = read(case_dir / "vllm_server.log")
    bench_text = read(case_dir / "vllm_bench.log")
    available_kv = first_float(
        server_text, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"
    )
    reserved_kv = first_float(
        server_text, r"reserved\s+([0-9.]+)\s+GiB memory for KV Cache"
    )
    strict_status = first_int(bench_text, r"strict_one_prompt_status\s+([0-9]+)")
    strict_ms = first_float(bench_text, r"strict_one_prompt_ms\s+([0-9.]+)")
    successful_requests = first_float(
        bench_text, r"Successful requests:\s+([0-9.]+)"
    )
    if successful_requests is None and strict_status == 200:
        successful_requests = 1.0
    mean_ttft_ms = first_float(bench_text, r"Mean TTFT \(ms\):\s+([0-9.]+)")
    row = {
        "case": case_dir.name,
        "model_load_gib": first_float(
            server_text, r"Model loading took\s+([0-9.]+)\s+GiB memory"
        ),
        "kv_cache_gib": available_kv if available_kv is not None else reserved_kv,
        "kv_cache_source": "available_auto" if available_kv is not None else ("reserved_explicit" if reserved_kv is not None else ""),
        "gpu_kv_cache_tokens": first_int(
            server_text, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"
        ),
        "successful_requests": successful_requests,
        "strict_one_prompt_ms": strict_ms,
        "latency_ms": mean_ttft_ms if mean_ttft_ms is not None else strict_ms,
        "output_tok_s": first_float(
            bench_text, r"Output token throughput \(tok/s\):\s+([0-9.]+)"
        ),
        "mean_ttft_ms": mean_ttft_ms,
        "mean_tpot_ms": first_float(bench_text, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
        "raw_fault_records": parsed,
        "unique_fault_pages": classified,
        "duplicate_records_skipped": fault.get("duplicate_records_skipped"),
        "unique_page_pct_of_raw": pct(classified, parsed),
        "weight_pages": weight,
        "kv_cache_pages": kv,
        "unknown_pages": unknown,
        "weight_page_pct": pct(weight, classified),
        "kv_cache_page_pct": pct(kv, classified),
        "unknown_page_pct": pct(unknown, classified),
        "known_weight_pct": pct(weight, known),
        "known_kv_cache_pct": pct(kv, known),
        "delta_faults": stats.get("delta_faults"),
        "delta_after_dedup": stats.get("delta_after_dedup"),
        "after_dedup_pct_of_faults": pct(
            stats.get("delta_after_dedup"), stats.get("delta_faults")
        ),
        "delta_kv_faults": stats.get("delta_kv_faults"),
        "delta_kv_after_dedup": stats.get("delta_kv_after_dedup"),
        "kv_fault_pct": stats.get("delta_kv_ratio_pct"),
        "kv_after_dedup_pct": stats.get("delta_kv_after_dedup_ratio_pct"),
        "top1_region": (fault.get("top_matched_names") or [["", 0]])[0][0],
        "top1_region_pages": (fault.get("top_matched_names") or [["", 0]])[0][1],
    }
    row.update(manifest_summary(case_dir / "offload_manifest.jsonl"))
    rows.append(row)

fieldnames = [
    "case",
    "model_load_gib",
    "kv_cache_gib",
    "kv_cache_source",
    "gpu_kv_cache_tokens",
    "successful_requests",
    "strict_one_prompt_ms",
    "latency_ms",
    "output_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "raw_fault_records",
    "unique_fault_pages",
    "duplicate_records_skipped",
    "unique_page_pct_of_raw",
    "weight_pages",
    "kv_cache_pages",
    "unknown_pages",
    "weight_page_pct",
    "kv_cache_page_pct",
    "unknown_page_pct",
    "known_weight_pct",
    "known_kv_cache_pct",
    "delta_faults",
    "delta_after_dedup",
    "after_dedup_pct_of_faults",
    "delta_kv_faults",
    "delta_kv_after_dedup",
    "kv_fault_pct",
    "kv_after_dedup_pct",
    "manifest_params",
    "manifest_bytes",
    "manifest_managed",
    "manifest_eager_start_events",
    "manifest_eager_start_ranges",
    "top1_region",
    "top1_region_pages",
]

csv_path = root / "small_kv_one_prompt_fault_share_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: fmt(row.get(key)) for key in fieldnames})

md_path = root / "small_kv_one_prompt_fault_share_summary.md"
lines = [
    "# Small KV One-Prompt Fault Share",
    "",
    f"- run_root: `{root}`",
    f"- summary_csv: `{csv_path}`",
    "",
    "| case | KV GiB | KV tokens | req ok | latency ms | unique/raw % | weight page % | KV page % | unknown % | KV after % | top region |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(
        "| {case} | {kv_gib} | {kv_tokens} | {ok} | {ttft} | {unique} | {weight} | {kv} | {unknown} | {kv_after} | {top} |".format(
            case=row["case"],
            kv_gib=fmt(row["kv_cache_gib"]),
            kv_tokens=fmt(row["gpu_kv_cache_tokens"]),
            ok=fmt(row["successful_requests"]),
            ttft=fmt(row["latency_ms"]),
            unique=fmt(row["unique_page_pct_of_raw"]),
            weight=fmt(row["weight_page_pct"]),
            kv=fmt(row["kv_cache_page_pct"]),
            unknown=fmt(row["unknown_page_pct"]),
            kv_after=fmt(row["kv_after_dedup_pct"]),
            top=row["top1_region"],
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path)
PY
}

{
  echo "run_root=$RUN_ROOT"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=$MODEL"
  echo "cases=$CASES"
  echo "client_mode=$CLIENT_MODE"
  echo "strict_prompt=$STRICT_PROMPT"
  echo "prompts=$PROMPTS"
  echo "output_len=$OUTPUT_LEN"
  echo "request_rate=$REQUEST_RATE"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
  echo "kv_cache_bytes_0p5g=$KV_CACHE_BYTES_0P5G"
  echo "capture_startup_faults=$CAPTURE_STARTUP_FAULTS"
} | tee "$RUN_ROOT/metadata.txt"

for case_name in $CASES; do
  run_case "$case_name"
done

if [ "$DRY_RUN" != "1" ]; then
  aggregate_results
fi

echo "Done. Results: $RUN_ROOT"
