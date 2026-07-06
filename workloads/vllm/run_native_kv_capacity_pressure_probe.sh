#!/usr/bin/env bash
# Native vLLM KV capacity and pressure probe.
#
# This intentionally disables UVM/LD_PRELOAD/offload and runs vLLM with normal
# cudaMalloc-backed GPU memory. It verifies:
#   1. how much KV cache vLLM reports after model load;
#   2. whether long requests can trigger clear KV-capacity errors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/native_kv_capacity_pressure_$(date -u +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8899}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-12288}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
VLLM_BIN="${VLLM_BIN:-$SCRIPT_DIR/.venv/bin/vllm}"
PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"

# If vLLM reports roughly 10.9K token KV capacity for Qwen3 FP8, these defaults
# should produce one request that fits and one request that exceeds capacity.
FIT_PROMPT_TOKENS="${FIT_PROMPT_TOKENS:-9000}"
OVER_PROMPT_TOKENS="${OVER_PROMPT_TOKENS:-12000}"
MAX_TOKENS="${MAX_TOKENS:-16}"
CONCURRENT_PRESSURE="${CONCURRENT_PRESSURE:-0}"
CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-4}"
CONCURRENT_PROMPT_TOKENS="${CONCURRENT_PROMPT_TOKENS:-4096}"
UNIQUE_PROMPTS="${UNIQUE_PROMPTS:-0}"

SERVER_LOG="$RUN_ROOT/server.log"
CLIENT_LOG="$RUN_ROOT/client.log"
SUMMARY_JSON="$RUN_ROOT/native_kv_capacity_pressure_summary.json"
SUMMARY_TXT="$RUN_ROOT/native_kv_capacity_pressure_summary.txt"

mkdir -p "$RUN_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RUN_ROOT/run.log"
}

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    kill "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 2
    kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
}
trap cleanup EXIT

require_env() {
  [[ -x "$VLLM_BIN" ]] || {
    echo "ERROR: vLLM binary not found/executable: $VLLM_BIN" >&2
    exit 1
  }
  [[ -x "$PYTHON_BIN" ]] || {
    echo "ERROR: Python binary not found/executable: $PYTHON_BIN" >&2
    exit 1
  }
}

wait_for_server() {
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server exited before readiness. Log tail:" >&2
      tail -n 160 "$SERVER_LOG" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$STARTUP_TIMEOUT" ]]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 160 "$SERVER_LOG" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

start_server() {
  log "Starting native vLLM: model=$MODEL max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  : > "$SERVER_LOG"
  local extra_args=()
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
    extra_args+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ -n "$ENABLE_PREFIX_CACHING" ]]; then
    case "${ENABLE_PREFIX_CACHING,,}" in
      1|true|yes|on)
        extra_args+=(--enable-prefix-caching)
        ;;
      0|false|no|off)
        extra_args+=(--no-enable-prefix-caching)
        ;;
      *)
        echo "ERROR: ENABLE_PREFIX_CACHING must be true/false, got: $ENABLE_PREFIX_CACHING" >&2
        exit 1
        ;;
    esac
  fi
  env -u VLLM_USE_UVM -u LD_PRELOAD \
    VLLM_UVM_POLICY_ENABLE=0 \
    VLLM_UVM_POLICY_RUNTIME_ENABLE=0 \
    VLLM_UVM_RESIDENCY_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=0 \
    "$VLLM_BIN" serve "$MODEL" \
      --enforce-eager \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb 0 \
      --port "$PORT" \
      "${extra_args[@]}" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  wait_for_server
  log "Server ready on port $PORT"
}

run_requests() {
  "$PYTHON_BIN" - "$PORT" "$MODEL" "$FIT_PROMPT_TOKENS" "$OVER_PROMPT_TOKENS" "$MAX_TOKENS" "$CLIENT_LOG" "$CONCURRENT_PRESSURE" "$CONCURRENT_REQUESTS" "$CONCURRENT_PROMPT_TOKENS" "$UNIQUE_PROMPTS" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

port = int(sys.argv[1])
model = sys.argv[2]
fit_prompt_tokens = int(sys.argv[3])
over_prompt_tokens = int(sys.argv[4])
max_tokens = int(sys.argv[5])
client_log = sys.argv[6]
concurrent_pressure = sys.argv[7].strip().lower() in ("1", "true", "yes", "on")
concurrent_requests = int(sys.argv[8])
concurrent_prompt_tokens = int(sys.argv[9])
unique_prompts = sys.argv[10].strip().lower() in ("1", "true", "yes", "on")

def request_case(name, prompt_tokens, unique_seed=0):
    # Qwen tokenizers generally split repeated single-character text compactly,
    # but whitespace-separated "hello" is stable enough to create long prompts.
    if unique_prompts:
        prompt = " ".join([f"hello_{unique_seed}_{idx}" for idx in range(prompt_tokens)])
    else:
        prompt = " ".join(["hello"] * prompt_tokens)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    record = {
        "case": name,
        "requested_prompt_tokens_hint": prompt_tokens,
        "max_tokens": max_tokens,
        "ok": False,
        "status": None,
        "latency_s": None,
        "error": None,
        "usage": None,
    }
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            record.update({
                "ok": 200 <= resp.status < 300,
                "status": resp.status,
                "latency_s": time.time() - start,
                "usage": parsed.get("usage"),
            })
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        record.update({
            "ok": False,
            "status": exc.code,
            "latency_s": time.time() - start,
            "error": body[:2000],
        })
    except Exception as exc:
        record.update({
            "ok": False,
            "latency_s": time.time() - start,
            "error": repr(exc),
        })
    return record

records = []
if concurrent_pressure:
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [
            executor.submit(
                request_case,
                f"concurrent_pressure_{idx}",
                concurrent_prompt_tokens,
                idx,
            )
            for idx in range(concurrent_requests)
        ]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda item: item["case"])
else:
    records = [
        request_case("fit_request", fit_prompt_tokens),
        request_case("over_capacity_request", over_prompt_tokens),
    ]
with open(client_log, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
print(json.dumps(records, indent=2, sort_keys=True, ensure_ascii=False))
PY
}

summarize() {
  "$PYTHON_BIN" - "$RUN_ROOT" "$MODEL" "$SERVER_LOG" "$CLIENT_LOG" "$SUMMARY_JSON" "$SUMMARY_TXT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
model = sys.argv[2]
server_log = Path(sys.argv[3])
client_log = Path(sys.argv[4])
summary_json = Path(sys.argv[5])
summary_txt = Path(sys.argv[6])

server = server_log.read_text(encoding="utf-8", errors="replace")

def first_float(pattern):
    m = re.search(pattern, server)
    return float(m.group(1)) if m else None

def first_int(pattern):
    m = re.search(pattern, server)
    return int(m.group(1).replace(",", "")) if m else None

client_records = []
if client_log.exists():
    for line in client_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            client_records.append(json.loads(line))
        except Exception:
            pass

signals = []
for pattern in [
    r"KV cache",
    r"No available memory",
    r"out of memory",
    r"OOM",
    r"maximum model length",
    r"longer than",
    r"preempt",
    r"offload",
    r"swap",
]:
    if re.search(pattern, server, re.IGNORECASE):
        signals.append(pattern)

summary = {
    "run_root": str(root),
    "model": model,
    "server_log": str(server_log),
    "client_log": str(client_log),
    "available_kv_cache_gib": first_float(r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
    "reserved_kv_cache_gib": first_float(r"reserved\s+([0-9.]+)\s+GiB memory for KV Cache"),
    "gpu_kv_cache_tokens": first_int(r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
    "max_concurrency_1024": first_float(r"Maximum concurrency for 1,024 tokens per request:\s+([0-9.]+)x"),
    "model_loading_gib": first_float(r"Model loading took\s+([0-9.]+)\s+GiB memory"),
    "signals_in_server_log": signals,
    "client_records": client_records,
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

lines = [
    f"run_root={root}",
    f"model={model}",
    f"available_kv_cache_gib={summary['available_kv_cache_gib']}",
    f"reserved_kv_cache_gib={summary['reserved_kv_cache_gib']}",
    f"gpu_kv_cache_tokens={summary['gpu_kv_cache_tokens']}",
    f"max_concurrency_1024={summary['max_concurrency_1024']}",
    f"model_loading_gib={summary['model_loading_gib']}",
    f"signals_in_server_log={','.join(signals)}",
    "",
    "client_records:",
]
for rec in client_records:
    lines.append(json.dumps(rec, sort_keys=True, ensure_ascii=False))
summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_txt.read_text(encoding="utf-8"))
PY
}

require_env
cleanup
start_server
run_requests | tee "$RUN_ROOT/client_stdout.log"
summarize | tee "$RUN_ROOT/summary_stdout.log"
log "Done: $RUN_ROOT"
