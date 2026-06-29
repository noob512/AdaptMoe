#!/usr/bin/env bash
# Sweep ordinary vLLM CUDA serving with an explicit small KV cache.
#
# This script intentionally unsets VLLM_USE_UVM and LD_PRELOAD for every server
# run. It answers how much concurrency a native/full-GPU-weight run can sustain
# when KV is kept around 0.5 GiB.

set -u -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/native_small_kv_concurrency_sweep_$(date -u +%Y%m%d_%H%M%S)}"
KV_CACHE_BYTES="${KV_CACHE_BYTES:-536870912}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 2 4 8 16 32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SEED="${SEED:-42}"
PORT_BASE="${PORT_BASE:-8860}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-2}"

mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"
summary_csv="$RUN_ROOT/native_small_kv_concurrency_summary.csv"
summary_md="$RUN_ROOT/native_small_kv_concurrency_summary.md"
commands_log="$RUN_ROOT/commands.log"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$commands_log"
}

cleanup_gpu() {
  pkill -f "vllm bench serve" >/dev/null 2>&1 || true
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >/dev/null 2>&1 || true
}

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before readiness. Log tail:" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$STARTUP_TIMEOUT" ]]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

run_case() {
  local seqs="$1"
  local idx="$2"
  local port=$((PORT_BASE + idx))
  local case_dir="$RUN_ROOT/s${seqs}"
  local server_log="$case_dir/vllm_server.log"
  local bench_log="$case_dir/vllm_bench.log"
  local meta="$case_dir/case_metadata.txt"
  local status=0
  local server_pid=""

  mkdir -p "$case_dir"
  : > "$server_log"
  : > "$bench_log"

  log "Running native small-KV case max_num_seqs=$seqs port=$port"
  cleanup_gpu
  sleep 2

  local serve_cmd=(
    uv run vllm serve "$MODEL"
    --enforce-eager
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$seqs"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --kv-cache-memory-bytes "$KV_CACHE_BYTES"
    --port "$port"
  )
  if [[ "$CPU_OFFLOAD_GB" != "0" && "$CPU_OFFLOAD_GB" != "0.0" ]]; then
    serve_cmd+=(--cpu-offload-gb "$CPU_OFFLOAD_GB")
  fi
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    serve_cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi

  {
    printf 'serve_command: env -u VLLM_USE_UVM -u LD_PRELOAD'
    printf ' %q' "${serve_cmd[@]}"
    echo
  } | tee -a "$commands_log" >"$case_dir/command.txt"

  (
    cd "$SCRIPT_DIR" || exit 1
    env -u VLLM_USE_UVM -u LD_PRELOAD \
      HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
      TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
      HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
      "${serve_cmd[@]}" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    log "Server ready for max_num_seqs=$seqs; idling ${POST_READY_IDLE_SEC}s"
    sleep "$POST_READY_IDLE_SEC"
    local bench_cmd=(
      uv run --directory "$SCRIPT_DIR" vllm bench serve
      --model "$MODEL"
      --dataset-name sharegpt
      --dataset-path "$DATASET_PATH"
      --num-prompts "$PROMPTS"
      --sharegpt-output-len "$OUTPUT_LEN"
      --seed "$SEED"
      --request-rate "$REQUEST_RATE"
      --port "$port"
    )
    {
      echo "===== Benchmark phase: main ====="
      printf 'bench_command:'
      printf ' %q' "${bench_cmd[@]}"
      echo
    } | tee -a "$bench_log" "$commands_log" >/dev/null
    set +e
    (
      cd "$SCRIPT_DIR" || exit 1
      HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
      TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
      HF_DATASETS_OFFLINE="$HF_DATASETS_OFFLINE" \
      "${bench_cmd[@]}"
    ) 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
  else
    status=1
  fi

  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 3
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_gpu

  {
    echo "case=s${seqs}"
    echo "max_num_seqs=$seqs"
    echo "kv_cache_bytes=$KV_CACHE_BYTES"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "status=$status"
  } > "$meta"
}

summarize() {
  python3 - "$RUN_ROOT" "$summary_csv" "$summary_md" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])

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

rows = []
for case_dir in sorted(root.glob("s*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 10**9):
    meta = {}
    for line in read(case_dir / "case_metadata.txt").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k] = v
    server = read(case_dir / "vllm_server.log")
    bench = read(case_dir / "vllm_bench.log")
    row = {
        "case": case_dir.name,
        "max_num_seqs": int(meta.get("max_num_seqs", case_dir.name[1:] or 0)),
        "kv_cache_bytes": int(meta.get("kv_cache_bytes", 0) or 0),
        "kv_cache_gib": (int(meta.get("kv_cache_bytes", 0) or 0) / (1024 ** 3)),
        "max_model_len": int(meta.get("max_model_len", 0) or 0),
        "cpu_offload_gb": float(meta.get("cpu_offload_gb", 0) or 0),
        "status": int(meta.get("status", 1) or 1),
        "gpu_kv_cache_tokens": first_int(server, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
        "max_concurrency_reported": first_float(server, r"Maximum concurrency for [0-9,]+ tokens per request:\s+([0-9.]+)x"),
        "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
        "reserved_kv_cache_gib": first_float(server, r"reserved\s+([0-9.]+)\s+GiB memory for KV Cache"),
        "successful_requests": first_int(bench, r"Successful requests:\s+([0-9,]+)"),
        "failed_requests": first_int(bench, r"Failed requests:\s+([0-9,]+)"),
        "request_throughput_req_s": first_float(bench, r"Request throughput \(req/s\):\s+([0-9.]+)"),
        "output_tok_s": first_float(bench, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
        "peak_output_tok_s": first_float(bench, r"Peak output token throughput \(tok/s\):\s+([0-9.]+)"),
        "peak_concurrent_requests": first_float(bench, r"Peak concurrent requests:\s+([0-9.]+)"),
        "total_token_throughput_tok_s": first_float(bench, r"Total Token throughput \(tok/s\):\s+([0-9.]+)"),
        "mean_ttft_ms": first_float(bench, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
        "median_ttft_ms": first_float(bench, r"Median TTFT \(ms\):\s+([0-9.]+)"),
        "p99_ttft_ms": first_float(bench, r"P99 TTFT \(ms\):\s+([0-9.]+)"),
        "mean_tpot_ms": first_float(bench, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
        "median_tpot_ms": first_float(bench, r"Median TPOT \(ms\):\s+([0-9.]+)"),
        "p99_tpot_ms": first_float(bench, r"P99 TPOT \(ms\):\s+([0-9.]+)"),
    }
    rows.append(row)

fields = [
    "case", "max_num_seqs", "status", "kv_cache_gib", "gpu_kv_cache_tokens",
    "max_concurrency_reported", "successful_requests", "failed_requests",
    "peak_concurrent_requests", "request_throughput_req_s", "output_tok_s",
    "peak_output_tok_s", "total_token_throughput_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "p99_ttft_ms", "p99_tpot_ms",
]
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fields})

def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)

lines = [
    "# Native Small-KV Concurrency Sweep",
    "",
    f"- cases: {len(rows)}",
    f"- kv_cache_bytes: {rows[0]['kv_cache_bytes'] if rows else ''}",
    f"- max_model_len: {rows[0]['max_model_len'] if rows else ''}",
    f"- cpu_offload_gb: {rows[0]['cpu_offload_gb'] if rows else ''}",
    "",
    "| max_num_seqs | status | KV tokens | reported max concurrency | success/failed | peak conc | req/s | out tok/s | TTFT ms | TPOT ms |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['max_num_seqs']} | {r['status']} | {fmt(r['gpu_kv_cache_tokens'])} | "
        f"{fmt(r['max_concurrency_reported'])} | {fmt(r['successful_requests'])}/{fmt(r['failed_requests'])} | "
        f"{fmt(r['peak_concurrent_requests'])} | {fmt(r['request_throughput_req_s'])} | "
        f"{fmt(r['output_tok_s'])} | {fmt(r['mean_ttft_ms'])} | {fmt(r['mean_tpot_ms'])} |"
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(lines))
PY
}

{
  echo "==========================================================="
  echo "Native Small-KV Concurrency Sweep"
  echo "run_root=$RUN_ROOT"
  echo "model=$MODEL"
  echo "kv_cache_bytes=$KV_CACHE_BYTES"
  echo "max_num_seqs_list=$MAX_NUM_SEQS_LIST"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
  echo "VLLM_USE_UVM unset; LD_PRELOAD unset"
  echo "==========================================================="
} | tee -a "$commands_log"

idx=0
for seqs in $MAX_NUM_SEQS_LIST; do
  run_case "$seqs" "$idx"
  summarize | tee -a "$commands_log"
  idx=$((idx + 1))
done

log "Summary CSV: $summary_csv"
log "Summary MD:  $summary_md"
