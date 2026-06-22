#!/usr/bin/env bash
# Single-case hybrid probe with a hard KV-cache reservation.
#
# Default intent:
#   - keep the same CPU-offloaded managed weight set (8 GiB)
#   - force vLLM to reserve 8 GiB KV cache
#   - prepare Phase-F remote-read/bg metadata
#   - do not eager-start full bg-promotion, so offloaded weights should not
#     immediately consume the KV cache budget

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
RESET_PATH="${RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/hybrid_fixed_kv_probe_kv${KV_CACHE_GB:-8}_s${MAX_NUM_SEQS:-32}_p${PROMPTS:-50}_o${OUTPUT_LEN:-64}_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-50}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
KV_CACHE_GB="${KV_CACHE_GB:-8}"
PORT="${PORT:-8922}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-3}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
HYBRID_START_MODE="${HYBRID_START_MODE:-prepare_only}"
HYBRID_GATE_ENABLE="${HYBRID_GATE_ENABLE:-0}"
VLLM_UVM_ALLOCATOR_SO="${VLLM_UVM_ALLOCATOR_SO:-$SCRIPT_DIR/vllm/uvm_test/uvm_allocator.so}"
VLLM_BIN="${VLLM_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"

mkdir -p "$RUN_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

kv_cache_bytes() {
  python3 - "$KV_CACHE_GB" <<'PY'
import sys
print(int(float(sys.argv[1]) * 1024 ** 3))
PY
}

write_param() {
  local name="$1"
  local value="$2"
  local path="$PARAM_ROOT/$name"

  if [[ ! -e "$path" ]]; then
    log "WARN: missing UVM param $name"
    return 0
  fi
  if [[ -w "$path" ]]; then
    printf '%s\n' "$value" > "$path" || log "WARN: failed to write $name=$value"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    printf '%s\n' "$value" | sudo -n tee "$path" >/dev/null || log "WARN: sudo write failed for $name=$value"
    return 0
  fi
  log "WARN: cannot write $name=$value; current value $(cat "$path" 2>/dev/null || echo unknown)"
}

configure_uvm() {
  write_param uvm_latency_remote_first_enable 0
  write_param uvm_latency_bg_promote_enable 0
  write_param uvm_latency_bg_promote_max_pages "$BG_MAX_PAGES"
  write_param uvm_latency_bg_promote_delay_ms "$BG_DELAY_MS"
  write_param uvm_latency_bg_promote_reschedule_delay_ms "$BG_RESCHEDULE_DELAY_MS"
  write_param uvm_phase_f_route_log_delay_ms "$PHASE_F_ROUTE_LOG_DELAY_MS"
}

reset_stats() {
  if [[ -w "$RESET_PATH" ]]; then
    printf '1\n' > "$RESET_PATH" || true
  elif command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    printf '1\n' | sudo -n tee "$RESET_PATH" >/dev/null || true
  else
    log "WARN: cannot reset latency stats; deltas may include earlier runs"
  fi
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
      tail -n 160 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$STARTUP_TIMEOUT" ]]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 160 "$server_log" >&2 2>/dev/null || true
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
  if [[ ! -r "$STATS_PATH" ]]; then
    echo "ERROR: $STATS_PATH is not readable. Load the patched nvidia_uvm module first." >&2
    exit 1
  fi
  if [[ ! -s "$VLLM_UVM_ALLOCATOR_SO" ]]; then
    echo "ERROR: missing UVM allocator: $VLLM_UVM_ALLOCATOR_SO" >&2
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

def stats(path):
    out = {}
    for line in read(path).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out

def manifest(path):
    out = {
        "params": 0,
        "bytes": 0,
        "managed": 0,
        "readmostly": 0,
        "remote_prepare_records": 0,
        "remote_prepare_failures": 0,
        "eager_start_events": 0,
        "eager_start_ranges": 0,
        "gate_events": 0,
    }
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ev = rec.get("event")
        if ev == "cpu_offload_parameter":
            out["params"] += 1
            out["bytes"] += int(rec.get("bytes") or 0)
            out["managed"] += int(rec.get("uvm_managed") is True)
            out["readmostly"] += int((rec.get("uvm_readmostly_result") or 0) > 0)
            if int(rec.get("uvm_remote_read_bg_success") or 0) > 0:
                out["remote_prepare_records"] += 1
            out["remote_prepare_failures"] += int(rec.get("uvm_remote_read_bg_failure") or 0)
        elif ev == "cpu_offload_hybrid_remote_eager_start":
            out["eager_start_events"] += 1
            out["eager_start_ranges"] += int(rec.get("start_success") or 0)
        elif ev == "cpu_offload_hybrid_remote_gate":
            out["gate_events"] += 1
    return out

server = read(root / "server.log")
bench = read(root / "bench.log")
before = stats(root / "latency_stats_before.txt")
after = stats(root / "latency_stats_after.txt")
manifest_stats = manifest(root / "offload_manifest.jsonl")
try:
    status = int(read(root / "status.txt").strip())
except Exception:
    status = None

counter_keys = [
    "remote_prepare_pages",
    "remote_prepare_mapped_pages",
    "bg_promote_enqueued_pages",
    "bg_promote_copied_pages",
    "bg_promote_committed_pages",
    "bg_promote_cancelled_pages",
    "bg_promote_failed_already_gpu_resident",
    "bg_promote_failed_not_cpu_resident",
    "bg_promote_start_pages",
    "bg_promote_pause_calls",
    "bg_promote_resume_calls",
]

row = {
    "status": status,
    "model_load_gib": first_float(server, r"Model loading took\s+([0-9.]+)\s+GiB memory"),
    "model_load_s": first_float(server, r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"),
    "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
    "reserved_kv_cache_gib": first_float(server, r"reserved\s+([0-9.]+)\s+GiB memory for KV Cache"),
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
    "mean_tpot_ms": first_float(bench, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
}
for key in counter_keys:
    row[key] = after.get(key, 0) - before.get(key, 0)
row.update({f"manifest_{k}": v for k, v in manifest_stats.items()})

csv_path = root / "hybrid_fixed_kv_probe_summary.csv"
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

kv_gib = row["reserved_kv_cache_gib"] if row["reserved_kv_cache_gib"] is not None else row["available_kv_cache_gib"]
lines = [
    "# Hybrid Fixed KV Probe",
    "",
    f"- run_root: `{root}`",
    f"- summary_csv: `{csv_path}`",
    "",
    "| status | model GiB | KV GiB | KV tokens | max conc @1024 | ok/fail | req/s | tok/s | TTFT ms | TPOT ms | peak conc |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    (
        f"| {fmt(row['status'])} | {fmt(row['model_load_gib'])} | {fmt(kv_gib)} | "
        f"{fmt(row['gpu_kv_cache_tokens'])} | {fmt(row['max_concurrency_1024'])} | "
        f"{fmt(row['successful_requests'])}/{fmt(row['failed_requests'])} | "
        f"{fmt(row['request_throughput_req_s'])} | {fmt(row['output_tok_s'])} | "
        f"{fmt(row['mean_ttft_ms'])} | {fmt(row['mean_tpot_ms'])} | "
        f"{fmt(row['peak_concurrent_requests'])} |"
    ),
    "",
    "## Phase-F / BG Delta",
    "",
    f"- remote_prepare_pages: `{row['remote_prepare_pages']}`",
    f"- bg_promote_enqueued_pages: `{row['bg_promote_enqueued_pages']}`",
    f"- bg_promote_start_pages: `{row['bg_promote_start_pages']}`",
    f"- bg_promote_committed_pages: `{row['bg_promote_committed_pages']}`",
    f"- bg_committed_gib: `{row['bg_promote_committed_pages'] * 4096 / 1024**3:.4f}`",
    f"- pause/resume: `{row['bg_promote_pause_calls']}/{row['bg_promote_resume_calls']}`",
    "",
    "## Manifest",
    "",
    f"- params: `{row['manifest_params']}`",
    f"- offload GiB: `{row['manifest_bytes'] / 1024**3:.4f}`",
    f"- managed: `{row['manifest_managed']}`",
    f"- readmostly: `{row['manifest_readmostly']}`",
    f"- remote_prepare_records: `{row['manifest_remote_prepare_records']}`",
    f"- remote_prepare_failures: `{row['manifest_remote_prepare_failures']}`",
    f"- eager_start_events/ranges: `{row['manifest_eager_start_events']}/{row['manifest_eager_start_ranges']}`",
    f"- gate_events: `{row['manifest_gate_events']}`",
]

md_path = root / "hybrid_fixed_kv_probe_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY
}

require_env
configure_uvm
build_vllm_cmd
cleanup_gpu
reset_stats

KV_CACHE_BYTES="$(kv_cache_bytes)"
SERVER_LOG="$RUN_ROOT/server.log"
BENCH_LOG="$RUN_ROOT/bench.log"
MANIFEST="$RUN_ROOT/offload_manifest.jsonl"
: > "$SERVER_LOG"
: > "$BENCH_LOG"
: > "$MANIFEST"
cp "$STATS_PATH" "$RUN_ROOT/latency_stats_before.txt" 2>/dev/null || true

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
  echo "kv_cache_gb=$KV_CACHE_GB"
  echo "kv_cache_bytes=$KV_CACHE_BYTES"
  echo "backend=isolated_uvm_hybrid_remote"
  echo "hybrid_start_mode=$HYBRID_START_MODE"
  echo "hybrid_gate_enable=$HYBRID_GATE_ENABLE"
  echo "bg_max_pages=$BG_MAX_PAGES"
  echo
  nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
} | tee "$RUN_ROOT/metadata.txt"

log "Starting hybrid fixed-KV server"
(
  cd "$SCRIPT_DIR" || exit 1
  env -u VLLM_USE_UVM -u LD_PRELOAD \
    VLLM_CPU_OFFLOAD_BACKEND=isolated_uvm_hybrid_remote \
    VLLM_CPU_OFFLOAD_UVM_SYNC=1 \
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="$MANIFEST" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_UVM_LOG_FILE="$RUN_ROOT/uvm_allocator.log" \
    VLLM_UVM_POLICY_ENABLE=0 \
    VLLM_UVM_POLICY_RUNTIME_ENABLE=0 \
    VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
    VLLM_UVM_PHASE_F_FD_MODE=adopt \
    VLLM_UVM_PHASE_F_CPP_HELPER=1 \
    VLLM_UVM_HYBRID_ROUND_PAGES=1 \
    VLLM_UVM_HYBRID_ROUND_READMOSTLY=1 \
    VLLM_UVM_HYBRID_START_MODE="$HYBRID_START_MODE" \
    VLLM_UVM_HYBRID_GATE_ENABLE="$HYBRID_GATE_ENABLE" \
    "${VLLM_CMD[@]}" serve "$MODEL" \
      --enforce-eager \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb "$CPU_OFFLOAD_GB" \
      --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
      --port "$PORT" >"$SERVER_LOG" 2>&1
) &
SERVER_PID=$!

status=0
if wait_for_server "$PORT" "$SERVER_PID" "$SERVER_LOG"; then
  log "Server ready; idling ${POST_READY_IDLE_SEC}s before benchmark"
  sleep "$POST_READY_IDLE_SEC"
  cp "$STATS_PATH" "$RUN_ROOT/latency_stats_after_ready_idle.txt" 2>/dev/null || true
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

cp "$STATS_PATH" "$RUN_ROOT/latency_stats_after.txt" 2>/dev/null || true
echo "$status" > "$RUN_ROOT/status.txt"

if kill -0 "$SERVER_PID" 2>/dev/null; then
  kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
fi
cleanup_gpu

summarize 2>&1 | tee "$RUN_ROOT/summary.log"

log "Done: $RUN_ROOT"
log "Summary: $RUN_ROOT/hybrid_fixed_kv_probe_summary.md"
exit "$status"
