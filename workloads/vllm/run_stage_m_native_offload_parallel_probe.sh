#!/usr/bin/env bash
# Single-case native vLLM CPU offload probe with higher server parallelism.
#
# Defaults keep the p50 workload shape and only raise MAX_NUM_SEQS:
#   PROMPTS=50 OUTPUT_LEN=64 REQUEST_RATE=1 MAX_NUM_SEQS=32 CPU_OFFLOAD_GB=8

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/native_offload_parallel_probe_s${MAX_NUM_SEQS:-32}_p${PROMPTS:-50}_o${OUTPUT_LEN:-64}_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-50}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
PORT="${PORT:-8821}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-3}"
VLLM_BIN="${VLLM_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"

mkdir -p "$RUN_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
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

build_vllm_cmd() {
  if [[ -n "$VLLM_BIN" ]]; then
    VLLM_CMD=("$VLLM_BIN")
  elif [[ -x "$SCRIPT_DIR/.venv/bin/vllm" ]]; then
    VLLM_CMD=("$SCRIPT_DIR/.venv/bin/vllm")
  elif [[ "$USE_UV_RUN" == "1" ]]; then
    VLLM_CMD=(uv run --directory "$SCRIPT_DIR" --no-sync vllm)
  else
    echo "ERROR: no vllm executable found. Set VLLM_BIN or USE_UV_RUN=1." >&2
    exit 1
  fi
}

require_env() {
  if [[ ! -s "$DATASET_PATH" ]]; then
    echo "ERROR: missing dataset: $DATASET_PATH" >&2
    exit 1
  fi
}

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
server_log = root / "server.log"
bench_log = root / "bench.log"
manifest = root / "offload_manifest.jsonl"

def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(text, pattern):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def first_int(text, pattern):
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None

def manifest_stats(path):
    out = {"params": 0, "bytes": 0, "pinned": 0}
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") != "cpu_offload_parameter":
            continue
        out["params"] += 1
        out["bytes"] += int(rec.get("bytes") or 0)
        out["pinned"] += int(rec.get("cpu_data_pinned") is True)
    return out

server = read(server_log)
bench = read(bench_log)
stats = manifest_stats(manifest)
try:
    status = int(read(root / "status.txt").strip())
except Exception:
    status = None

row = {
    "status": status,
    "model_load_gib": first_float(server, r"Model loading took\s+([0-9.]+)\s+GiB memory"),
    "model_load_s": first_float(server, r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"),
    "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
    "gpu_kv_cache_tokens": first_int(server, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
    "max_concurrency_1024": first_float(server, r"Maximum concurrency for 1,024 tokens per request:\s+([0-9.]+)x"),
    "successful_requests": first_float(bench, r"Successful requests:\s+([0-9.]+)"),
    "failed_requests": first_float(bench, r"Failed requests:\s+([0-9.]+)"),
    "benchmark_duration_s": first_float(bench, r"Benchmark duration \(s\):\s+([0-9.]+)"),
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
    "manifest_params": stats["params"],
    "manifest_bytes": stats["bytes"],
    "manifest_gib": stats["bytes"] / 1024**3,
    "manifest_pinned_params": stats["pinned"],
}

csv_path = root / "native_offload_parallel_probe_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)

def fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

lines = [
    "# Native CPU Offload Parallel Probe",
    "",
    f"- run_root: `{root}`",
    f"- summary_csv: `{csv_path}`",
    "",
    "| status | model GiB | KV GiB | KV tokens | max conc @1024 | ok/fail | req/s | tok/s | TTFT ms | TPOT ms | peak conc |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    (
        f"| {fmt(row['status'])} | {fmt(row['model_load_gib'])} | "
        f"{fmt(row['available_kv_cache_gib'])} | {fmt(row['gpu_kv_cache_tokens'])} | "
        f"{fmt(row['max_concurrency_1024'])} | "
        f"{fmt(row['successful_requests'])}/{fmt(row['failed_requests'])} | "
        f"{fmt(row['request_throughput_req_s'])} | {fmt(row['output_tok_s'])} | "
        f"{fmt(row['mean_ttft_ms'])} | {fmt(row['mean_tpot_ms'])} | "
        f"{fmt(row['peak_concurrent_requests'])} |"
    ),
    "",
    "## Offload Manifest",
    "",
    f"- params: `{row['manifest_params']}`",
    f"- bytes: `{row['manifest_bytes']}`",
    f"- GiB: `{row['manifest_gib']:.4f}`",
    f"- pinned params: `{row['manifest_pinned_params']}`",
]

md_path = root / "native_offload_parallel_probe_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY
}

require_env
build_vllm_cmd
cleanup_gpu

SERVER_LOG="$RUN_ROOT/server.log"
BENCH_LOG="$RUN_ROOT/bench.log"
MANIFEST="$RUN_ROOT/offload_manifest.jsonl"
: > "$SERVER_LOG"
: > "$BENCH_LOG"
: > "$MANIFEST"

{
  echo "run_root=$RUN_ROOT"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=$MODEL"
  echo "dataset_path=$DATASET_PATH"
  echo "prompts=$PROMPTS"
  echo "output_len=$OUTPUT_LEN"
  echo "request_rate=$REQUEST_RATE"
  echo "seed=$SEED"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
  echo "backend=uva"
  echo
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
} | tee "$RUN_ROOT/metadata.txt"

log "Starting native CPU offload server with MAX_NUM_SEQS=$MAX_NUM_SEQS"
(
  cd "$SCRIPT_DIR" || exit 1
  env -u VLLM_USE_UVM -u LD_PRELOAD \
    VLLM_CPU_OFFLOAD_BACKEND=uva \
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="$MANIFEST" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    "${VLLM_CMD[@]}" serve "$MODEL" \
      --enforce-eager \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb "$CPU_OFFLOAD_GB" \
      --port "$PORT" >"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

status=0
if wait_for_server "$PORT" "$SERVER_PID" "$SERVER_LOG"; then
  log "Server ready; idling ${POST_READY_IDLE_SEC}s before benchmark"
  sleep "$POST_READY_IDLE_SEC"
  set +e
  (
    cd "$SCRIPT_DIR" || exit 1
    "${VLLM_CMD[@]}" bench serve \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$PROMPTS" \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      --request-rate "$REQUEST_RATE" \
      --port "$PORT"
  ) 2>&1 | tee -a "$BENCH_LOG"
  status=${PIPESTATUS[0]}
  set -u
else
  status=1
fi

echo "$status" > "$RUN_ROOT/status.txt"

if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
  sleep 3
  kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
fi
cleanup_gpu

summarize 2>&1 | tee "$RUN_ROOT/summary.log"

log "Done: $RUN_ROOT"
log "Summary: $RUN_ROOT/native_offload_parallel_probe_summary.md"
exit "$status"
