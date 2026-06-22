#!/usr/bin/env bash
# Compare vLLM KV-cache capacity under identical serving settings:
#   1. native vLLM pinned-UVA CPU offload
#   2. isolated UVM hybrid remote offload with eager bg-promotion
#
# This probe does not run requests. It waits until each server is ready, then
# parses server.log for model memory and KV-cache capacity.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
RESET_PATH="${RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_kv_cache_native_vs_hybrid_eager_$(date -u +%Y%m%d_%H%M%S)}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PORT_BASE="${PORT_BASE:-8620}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-3}"
HYBRID_POST_READY_IDLE_SEC="${HYBRID_POST_READY_IDLE_SEC:-8}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
VLLM_UVM_ALLOCATOR_SO="${VLLM_UVM_ALLOCATOR_SO:-$SCRIPT_DIR/vllm/uvm_test/uvm_allocator.so}"
VLLM_BIN="${VLLM_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"

mkdir -p "$RUN_ROOT"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
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

require_env() {
  if [[ ! -r "$STATS_PATH" ]]; then
    echo "ERROR: $STATS_PATH is not readable. Load the patched nvidia_uvm module first." >&2
    exit 1
  fi
  if [[ ! -s "$VLLM_UVM_ALLOCATOR_SO" ]]; then
    echo "ERROR: missing UVM allocator: $VLLM_UVM_ALLOCATOR_SO" >&2
    exit 1
  fi
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

write_metadata() {
  {
    echo "run_root=$RUN_ROOT"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model=$MODEL"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "bg_max_pages=$BG_MAX_PAGES"
    echo "bg_delay_ms=$BG_DELAY_MS"
    echo "bg_reschedule_delay_ms=$BG_RESCHEDULE_DELAY_MS"
    echo "hybrid_start_mode=eager"
    echo "hybrid_gate_enable=0"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    echo "UVM module parameters:"
    for p in \
      "$PARAM_ROOT/uvm_latency_remote_first_enable" \
      "$PARAM_ROOT/uvm_latency_bg_promote_enable" \
      "$PARAM_ROOT/uvm_latency_bg_promote_max_pages" \
      "$PARAM_ROOT/uvm_latency_bg_promote_delay_ms" \
      "$PARAM_ROOT/uvm_latency_bg_promote_reschedule_delay_ms" \
      "$PARAM_ROOT/uvm_perf_prefetch_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_migration_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_threshold" \
      "$PARAM_ROOT/uvm_ats_mode"; do
      [[ -r "$p" ]] && printf '  %-48s %s\n' "$(basename "$p")" "$(cat "$p")"
    done
  } | tee "$RUN_ROOT/metadata.txt"
}

run_case() {
  local case_name="$1"
  local backend="$2"
  local port="$3"
  local idle_sec="$4"
  local case_root="$RUN_ROOT/$case_name"
  local server_log="$case_root/server.log"
  local manifest="$case_root/offload_manifest.jsonl"
  local server_pid=""
  local status=0

  mkdir -p "$case_root"
  : > "$server_log"
  : > "$manifest"

  log "Running $case_name backend=$backend"
  cleanup_gpu
  sleep 2
  reset_stats
  cp "$STATS_PATH" "$case_root/latency_stats_before.txt" 2>/dev/null || true

  (
    cd "$SCRIPT_DIR" || exit 1
    env -u VLLM_USE_UVM -u LD_PRELOAD \
      VLLM_CPU_OFFLOAD_BACKEND="$backend" \
      VLLM_CPU_OFFLOAD_UVM_SYNC=1 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="$manifest" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_UVM_LOG_FILE="$case_root/uvm_allocator.log" \
      VLLM_UVM_POLICY_ENABLE=0 \
      VLLM_UVM_POLICY_RUNTIME_ENABLE=0 \
      VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
      VLLM_UVM_PHASE_F_FD_MODE=adopt \
      VLLM_UVM_PHASE_F_CPP_HELPER=1 \
      VLLM_UVM_HYBRID_ROUND_PAGES=1 \
      VLLM_UVM_HYBRID_ROUND_READMOSTLY=1 \
      VLLM_UVM_HYBRID_START_MODE=eager \
      VLLM_UVM_HYBRID_GATE_ENABLE=0 \
      "${VLLM_CMD[@]}" serve "$MODEL" \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --cpu-offload-gb "$CPU_OFFLOAD_GB" \
        --port "$port" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    log "$case_name ready; idling ${idle_sec}s before sampling"
    sleep "$idle_sec"
  else
    status=1
  fi

  cp "$STATS_PATH" "$case_root/latency_stats_after.txt" 2>/dev/null || true
  echo "$status" > "$case_root/status.txt"

  if kill -0 "$server_pid" 2>/dev/null; then
    kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_gpu
}

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = ["native_cpu_offload", "hybrid_eager_offload"]

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
        "remote_prepare_records": 0,
        "remote_prepare_failures": 0,
        "eager_start_events": 0,
        "eager_start_ranges": 0,
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
            if int(rec.get("uvm_remote_read_bg_success") or 0) > 0:
                out["remote_prepare_records"] += 1
            out["remote_prepare_failures"] += int(rec.get("uvm_remote_read_bg_failure") or 0)
        elif ev == "cpu_offload_hybrid_remote_eager_start":
            out["eager_start_events"] += 1
            out["eager_start_ranges"] += int(rec.get("start_success") or 0)
    return out

rows = []
for case in cases:
    case_root = root / case
    log = read(case_root / "server.log")
    before = stats(case_root / "latency_stats_before.txt")
    after = stats(case_root / "latency_stats_after.txt")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in [
        "remote_prepare_pages",
        "bg_promote_enqueued_pages",
        "bg_promote_copied_pages",
        "bg_promote_committed_pages",
        "bg_promote_cancelled_pages",
        "bg_promote_failed_already_gpu_resident",
    ]}
    try:
        status = int(read(case_root / "status.txt").strip())
    except Exception:
        status = None
    row = {
        "case": case,
        "status": status,
        "model_load_gib": first_float(log, r"Model loading took\s+([0-9.]+)\s+GiB memory"),
        "model_load_s": first_float(log, r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"),
        "available_kv_cache_gib": first_float(log, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
        "gpu_kv_cache_tokens": first_int(log, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
        "max_concurrency_1024": first_float(log, r"Maximum concurrency for 1,024 tokens per request:\s+([0-9.]+)x"),
    }
    row.update(delta)
    row.update({f"manifest_{k}": v for k, v in manifest(case_root / "offload_manifest.jsonl").items()})
    rows.append(row)

csv_path = root / "kv_cache_native_vs_hybrid_eager_summary.csv"
fields = list(rows[0].keys()) if rows else []
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

base = rows[0] if rows else {}
hyb = rows[1] if len(rows) > 1 else {}
def diff(key):
    if base.get(key) is None or hyb.get(key) is None:
        return None
    return hyb[key] - base[key]

lines = [
    "# KV Cache: Native CPU Offload vs Hybrid Eager",
    "",
    f"- run_root: `{root}`",
    f"- summary_csv: `{csv_path}`",
    "",
    "| case | status | model load GiB | load s | available KV GiB | GPU KV tokens | max concurrency @1024 | remote prepare pages | bg committed pages |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['case']} | {r['status']} | {r['model_load_gib']} | {r['model_load_s']} | "
        f"{r['available_kv_cache_gib']} | {r['gpu_kv_cache_tokens']} | "
        f"{r['max_concurrency_1024']} | {r['remote_prepare_pages']} | "
        f"{r['bg_promote_committed_pages']} |"
    )
lines.extend([
    "",
    "## Difference: hybrid_eager - native",
    "",
    f"- model_load_gib_delta: `{diff('model_load_gib')}`",
    f"- available_kv_cache_gib_delta: `{diff('available_kv_cache_gib')}`",
    f"- gpu_kv_cache_tokens_delta: `{diff('gpu_kv_cache_tokens')}`",
])
md_path = root / "kv_cache_native_vs_hybrid_eager_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY
}

require_env
configure_uvm
build_vllm_cmd
write_metadata

run_case native_cpu_offload uva "$PORT_BASE" "$POST_READY_IDLE_SEC"
run_case hybrid_eager_offload isolated_uvm_hybrid_remote "$((PORT_BASE + 1))" "$HYBRID_POST_READY_IDLE_SEC"
summarize 2>&1 | tee "$RUN_ROOT/summary.log"

log "Done: $RUN_ROOT"
log "Summary: $RUN_ROOT/kv_cache_native_vs_hybrid_eager_summary.md"
