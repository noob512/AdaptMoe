#!/usr/bin/env bash
# Probe whether Phase-F hybrid remote-read + background promotion really takes
# effect inside vLLM. This intentionally runs only the hybrid backend.
#
# Success criteria:
#   - remote-bg prepare succeeds for the offloaded managed weights
#   - remote_prepare_pages and bg_promote_enqueued_pages increase
#   - bg_promote_committed_pages or bg_promote_copied_pages increases
#
# The experiment starts bg-promotion eagerly after prepare and disables the
# per-forward pause/resume gate so the worker can run before warmup/first query
# consumes the pages through the ordinary UVM path.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
RESET_PATH="${RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_hybrid_effectiveness_probe_$(date -u +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8523}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-8}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
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

  for param in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable \
    uvm_latency_bg_promote_max_pages \
    uvm_latency_bg_promote_delay_ms \
    uvm_latency_bg_promote_reschedule_delay_ms; do
    if [[ ! -e "$PARAM_ROOT/$param" ]]; then
      echo "ERROR: missing $PARAM_ROOT/$param; this does not look like patched UVM." >&2
      exit 1
    fi
  done

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
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "post_ready_idle_sec=$POST_READY_IDLE_SEC"
    echo "hybrid_start_mode=eager"
    echo "hybrid_gate_enable=0"
    echo "bg_max_pages=$BG_MAX_PAGES"
    echo "bg_delay_ms=$BG_DELAY_MS"
    echo "bg_reschedule_delay_ms=$BG_RESCHEDULE_DELAY_MS"
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

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = root / "offload_manifest.jsonl"
bench = (root / "bench.log").read_text(encoding="utf-8", errors="replace") if (root / "bench.log").exists() else ""

def read_stats(path):
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return out

def first(pattern):
    m = re.search(pattern, bench)
    return float(m.group(1)) if m else None

before = read_stats(root / "latency_stats_before.txt")
after = read_stats(root / "latency_stats_after.txt")
keys = [
    "remote_prepare_calls",
    "remote_prepare_pages",
    "remote_prepare_mapped_pages",
    "bg_promote_enqueued_pages",
    "bg_promote_copied_pages",
    "bg_promote_committed_pages",
    "bg_promote_cancelled_pages",
    "bg_promote_failed_already_gpu_resident",
    "bg_promote_failed_not_cpu_resident",
    "bg_promote_worker_rounds",
    "bg_promote_pause_calls",
    "bg_promote_resume_calls",
    "bg_promote_gate_skips",
]
delta = {k: after.get(k, 0) - before.get(k, 0) for k in keys}

manifest_counts = {
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
for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines() if manifest.exists() else []:
    try:
        rec = json.loads(line)
    except Exception:
        continue
    ev = rec.get("event")
    if ev == "cpu_offload_parameter":
        manifest_counts["params"] += 1
        manifest_counts["bytes"] += int(rec.get("bytes") or 0)
        manifest_counts["managed"] += int(rec.get("uvm_managed") is True)
        manifest_counts["readmostly"] += int((rec.get("uvm_readmostly_result") or 0) > 0)
        if int(rec.get("uvm_remote_read_bg_success") or 0) > 0:
            manifest_counts["remote_prepare_records"] += 1
        manifest_counts["remote_prepare_failures"] += int(rec.get("uvm_remote_read_bg_failure") or 0)
    elif ev == "cpu_offload_hybrid_remote_eager_start":
        manifest_counts["eager_start_events"] += 1
        manifest_counts["eager_start_ranges"] += int(rec.get("start_success") or 0)
    elif ev == "cpu_offload_hybrid_remote_gate":
        manifest_counts["gate_events"] += 1

ok = (
    manifest_counts["params"] > 0
    and manifest_counts["remote_prepare_records"] == manifest_counts["params"]
    and manifest_counts["remote_prepare_failures"] == 0
    and delta["remote_prepare_pages"] > 0
    and delta["bg_promote_enqueued_pages"] > 0
    and delta["bg_promote_committed_pages"] > 0
)

lines = [
    "# vLLM Phase-F Hybrid Effectiveness Probe",
    "",
    f"- run_root: `{root}`",
    f"- verdict: `{'PASS' if ok else 'FAIL'}`",
    "",
    "## Benchmark",
    "",
    f"- successful_requests: `{first(r'Successful requests:\\s+([0-9.]+)')}`",
    f"- failed_requests: `{first(r'Failed requests:\\s+([0-9.]+)')}`",
    f"- output_tok_s: `{first(r'Output token throughput \\(tok/s\\):\\s+([0-9.]+)')}`",
    f"- mean_tpot_ms: `{first(r'Mean TPOT \\(ms\\):\\s+([0-9.]+)')}`",
    "",
    "## Manifest",
    "",
]
for k, v in manifest_counts.items():
    lines.append(f"- {k}: `{v}`")

lines.extend(["", "## Kernel Counter Delta", "", "| counter | delta |", "|---|---:|"])
for k in keys:
    lines.append(f"| {k} | {delta[k]} |")

lines.extend([
    "",
    "## Required Signal",
    "",
    "- `remote_prepare_pages > 0`: remote-read PTE prepare touched real pages.",
    "- `bg_promote_enqueued_pages > 0`: pages entered the bg-promotion queue.",
    "- `bg_promote_committed_pages > 0`: the bg worker actually copied and committed pages to GPU.",
    "- `gate_events == 0`: this run disables gate overhead to isolate bg-promotion effectiveness.",
])

(root / "hybrid_effectiveness_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "hybrid_effectiveness_summary.md").read_text(encoding="utf-8"))
PY
}

require_env
configure_uvm
build_vllm_cmd
write_metadata
cleanup_gpu
reset_stats
cp "$STATS_PATH" "$RUN_ROOT/latency_stats_before.txt" 2>/dev/null || true

SERVER_LOG="$RUN_ROOT/server.log"
BENCH_LOG="$RUN_ROOT/bench.log"
MANIFEST="$RUN_ROOT/offload_manifest.jsonl"
: > "$SERVER_LOG"
: > "$BENCH_LOG"
: > "$MANIFEST"

log "Starting hybrid-only vLLM server"
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
    VLLM_UVM_HYBRID_START_MODE=eager \
    VLLM_UVM_HYBRID_GATE_ENABLE=0 \
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
  log "Server ready; idling ${POST_READY_IDLE_SEC}s for eager bg-promotion"
  sleep "$POST_READY_IDLE_SEC"
  cp "$STATS_PATH" "$RUN_ROOT/latency_stats_after_ready_idle.txt" 2>/dev/null || true

  log "Running one short benchmark request"
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
  kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" >/dev/null 2>&1 || true
fi
cleanup_gpu

cp "$STATS_PATH" "$RUN_ROOT/latency_stats_after.txt" 2>/dev/null || true
summarize 2>&1 | tee "$RUN_ROOT/summary.log"

log "Done: $RUN_ROOT"
log "Summary: $RUN_ROOT/hybrid_effectiveness_summary.md"
exit "$status"
