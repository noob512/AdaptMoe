#!/usr/bin/env bash
# Small vLLM probe for Phase-F without CUDA ReadMostly or AccessedBy advice.
#
# Cases:
#   A. isolated_uvm_remote
#   C. isolated_uvm_prepare_remote_noreadmostly_noaccessedby
#   D. isolated_uvm_hybrid_remote_noreadmostly_noaccessedby

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
RESET_PATH="${RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PROMPTS="${PROMPTS:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
KV_CACHE_GB="${KV_CACHE_GB:-8}"
PORT_BASE="${PORT_BASE:-8990}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
POST_READY_IDLE_SEC="${POST_READY_IDLE_SEC:-3}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
VLLM_UVM_ALLOCATOR_SO="${VLLM_UVM_ALLOCATOR_SO:-$SCRIPT_DIR/vllm/uvm_test/uvm_allocator.so}"
VLLM_BIN="${VLLM_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/uvm_phase_f_noreadmostly_noaccessedby_kv${KV_CACHE_GB}_s${MAX_NUM_SEQS}_p${PROMPTS}_o${OUTPUT_LEN}_$(date -u +%Y%m%d_%H%M%S)}"

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
  if [[ "$PROMPTS" -gt 10 ]]; then
    echo "ERROR: this isolation script intentionally caps PROMPTS at 10." >&2
    exit 1
  fi
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

write_metadata() {
  local path="$1"
  local case_name="$2"
  local backend="$3"
  {
    echo "case=$case_name"
    echo "backend=$backend"
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
    echo "hybrid_start_mode=prepare_only"
    echo "hybrid_gate_enable=0"
    echo "bg_promote_enable=0"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    for p in \
      uvm_latency_remote_first_enable \
      uvm_latency_bg_promote_enable \
      uvm_latency_bg_promote_max_pages \
      uvm_latency_bg_promote_delay_ms \
      uvm_latency_bg_promote_reschedule_delay_ms \
      uvm_perf_access_counter_migration_enable \
      uvm_perf_access_counter_threshold \
      uvm_ats_mode; do
      if [[ -r "$PARAM_ROOT/$p" ]]; then
        printf '  %-48s %s\n' "$p" "$(cat "$PARAM_ROOT/$p" 2>/dev/null || true)"
      fi
    done
  } > "$path"
}

run_case() {
  local case_name="$1"
  local backend="$2"
  local port="$3"
  local case_root="$RUN_ROOT/$case_name"
  local server_log="$case_root/server.log"
  local bench_log="$case_root/bench.log"
  local manifest="$case_root/offload_manifest.jsonl"
  local status=0
  local round_pages=0
  local round_readmostly=0

  if [[ "$backend" == "isolated_uvm_prepare_remote" || \
        "$backend" == "isolated_uvm_prepare_remote_noreadmostly" || \
        "$backend" == "isolated_uvm_prepare_remote_noreadmostly_noaccessedby" || \
        "$backend" == "isolated_uvm_hybrid_remote" || \
        "$backend" == "isolated_uvm_hybrid_remote_noreadmostly" || \
        "$backend" == "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby" ]]; then
    round_pages=1
    round_readmostly=1
  fi

  mkdir -p "$case_root"
  : > "$server_log"
  : > "$bench_log"
  : > "$manifest"

  log "Running $case_name backend=$backend port=$port"
  cleanup_gpu
  configure_uvm
  reset_stats
  cp "$STATS_PATH" "$case_root/latency_stats_before.txt" 2>/dev/null || true
  write_metadata "$case_root/metadata.txt" "$case_name" "$backend"

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
      VLLM_UVM_HYBRID_ROUND_PAGES="$round_pages" \
      VLLM_UVM_HYBRID_ROUND_READMOSTLY="$round_readmostly" \
      VLLM_UVM_HYBRID_START_MODE=prepare_only \
      VLLM_UVM_HYBRID_GATE_ENABLE=0 \
      "${VLLM_CMD[@]}" serve "$MODEL" \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --cpu-offload-gb "$CPU_OFFLOAD_GB" \
        --kv-cache-memory-bytes "$KV_CACHE_BYTES" \
        --port "$port" >"$server_log" 2>&1
  ) &
  local server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    log "$case_name ready; idling ${POST_READY_IDLE_SEC}s before benchmark"
    sleep "$POST_READY_IDLE_SEC"
    cp "$STATS_PATH" "$case_root/latency_stats_after_ready_idle.txt" 2>/dev/null || true
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
        --port "$port"
    ) 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
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
  return 0
}

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = [
    "A_isolated_uvm_remote",
    "C_prepare_remote_read_noreadmostly_noaccessedby",
    "D_deferred_bg_pending_noreadmostly_noaccessedby",
]

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
        "readmostly_rounded": 0,
        "accessed_by": 0,
        "preferred_cpu": 0,
        "remote_prepare_records": 0,
        "remote_prepare_failures": 0,
        "phase_f_rounded": 0,
        "phase_f_bytes": 0,
        "gate_events": 0,
        "eager_start_events": 0,
        "names": [],
    }
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ev = rec.get("event")
        if ev == "cpu_offload_hybrid_remote_gate":
            out["gate_events"] += 1
            continue
        if ev == "cpu_offload_hybrid_remote_eager_start":
            out["eager_start_events"] += 1
            continue
        if ev != "cpu_offload_parameter":
            continue
        out["params"] += 1
        out["bytes"] += int(rec.get("bytes") or 0)
        out["managed"] += int(rec.get("uvm_managed") is True)
        out["readmostly"] += int((rec.get("uvm_readmostly_result") or 0) > 0)
        out["readmostly_rounded"] += int((rec.get("uvm_readmostly_rounded_result") or 0) > 0)
        out["accessed_by"] += int((rec.get("uvm_accessed_by_result") or 0) > 0)
        out["preferred_cpu"] += int((rec.get("uvm_preferred_cpu_result") or 0) > 0)
        out["remote_prepare_records"] += int((rec.get("uvm_remote_read_bg_success") or 0) > 0)
        out["remote_prepare_failures"] += int(rec.get("uvm_remote_read_bg_failure") or 0)
        out["phase_f_rounded"] += int(rec.get("uvm_phase_f_rounded") is True)
        out["phase_f_bytes"] += int(rec.get("uvm_phase_f_bytes") or 0)
        name = rec.get("full_parameter_name")
        if isinstance(name, str):
            out["names"].append(name)
    return out

counter_keys = [
    "remote_prepare_calls",
    "remote_prepare_pages",
    "remote_prepare_mapped_pages",
    "bg_promote_enqueued_pages",
    "bg_promote_start_pages",
    "bg_promote_copied_pages",
    "bg_promote_committed_pages",
    "bg_promote_cancelled_pages",
    "bg_promote_failed_already_gpu_resident",
    "bg_promote_failed_not_cpu_resident",
    "bg_promote_pause_calls",
    "bg_promote_resume_calls",
]

rows = []
for case in cases:
    case_root = root / case
    server = read(case_root / "server.log")
    bench = read(case_root / "bench.log")
    before = stats(case_root / "latency_stats_before.txt")
    after = stats(case_root / "latency_stats_after.txt")
    m = manifest(case_root / "offload_manifest.jsonl")
    try:
        status = int(read(case_root / "status.txt").strip())
    except Exception:
        status = None
    row = {
        "case": case,
        "status": status,
        "backend": first_int("", r"never"),
        "model_load_gib": first_float(server, r"Model loading took\s+([0-9.]+)\s+GiB memory"),
        "model_load_s": first_float(server, r"Model loading took\s+[0-9.]+\s+GiB memory and\s+([0-9.]+)\s+seconds"),
        "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
        "reserved_kv_cache_gib": first_float(server, r"reserved\s+([0-9.]+)\s+GiB memory for KV Cache"),
        "gpu_kv_cache_tokens": first_int(server, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
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
        "manifest_params": m["params"],
        "manifest_gib": m["bytes"] / 1024**3,
        "manifest_managed": m["managed"],
        "manifest_readmostly": m["readmostly"],
        "manifest_readmostly_rounded": m["readmostly_rounded"],
        "manifest_preferred_cpu": m["preferred_cpu"],
        "manifest_accessed_by": m["accessed_by"],
        "manifest_remote_prepare_records": m["remote_prepare_records"],
        "manifest_remote_prepare_failures": m["remote_prepare_failures"],
        "manifest_phase_f_rounded": m["phase_f_rounded"],
        "manifest_phase_f_gib": m["phase_f_bytes"] / 1024**3,
        "manifest_gate_events": m["gate_events"],
        "manifest_eager_start_events": m["eager_start_events"],
        "_names": m["names"],
    }
    for key in counter_keys:
        row[key] = after.get(key, 0) - before.get(key, 0)
    rows.append(row)

base = rows[0] if rows else {}
base_names = list(base.get("_names") or [])
for row in rows:
    row["same_offload_names_as_A"] = row.get("_names") == base_names
    row["tok_s_ratio_vs_A"] = (
        row["output_tok_s"] / base["output_tok_s"]
        if row.get("output_tok_s") is not None and base.get("output_tok_s")
        else None
    )
    row["tpot_ratio_vs_A"] = (
        row["mean_tpot_ms"] / base["mean_tpot_ms"]
        if row.get("mean_tpot_ms") is not None and base.get("mean_tpot_ms")
        else None
    )
    row.pop("_names", None)

fields = list(rows[0]) if rows else []
csv_path = root / "uvm_phase_f_noreadmostly_noaccessedby_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

lines = [
    "# UVM Phase-F No-ReadMostly No-AccessedBy Probe",
    "",
    f"- run_root: `{root}`",
    f"- summary_csv: `{csv_path}`",
    "- fixed KV: all cases use `--kv-cache-memory-bytes`",
    "- C/D config: no `cudaMemAdviseSetReadMostly`, no `cudaMemAdviseSetAccessedBy`; Phase-F must build GPU->CPU READ PTEs itself.",
    "- D config: `VLLM_UVM_HYBRID_START_MODE=prepare_only`, `VLLM_UVM_HYBRID_GATE_ENABLE=0`, `uvm_latency_bg_promote_enable=0`",
    "",
    "| case | status | ok/fail | peak conc | tok/s | TTFT ms | TPOT ms | tok/s/A | TPOT/A | params | offload GiB | managed | PrefCPU | AccessedBy | ReadMostly | RM rounded | prepare recs | remote pages | bg enq | bg start | bg commit | pause/resume | same names |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for r in rows:
    lines.append(
        f"| {r['case']} | {fmt(r['status'])} | {fmt(r['successful_requests'], 0)}/{fmt(r['failed_requests'], 0)} | "
        f"{fmt(r['peak_concurrent_requests'])} | {fmt(r['output_tok_s'])} | {fmt(r['mean_ttft_ms'])} | "
        f"{fmt(r['mean_tpot_ms'])} | {fmt(r['tok_s_ratio_vs_A'])} | {fmt(r['tpot_ratio_vs_A'])} | "
        f"{r['manifest_params']} | {fmt(r['manifest_gib'], 4)} | {r['manifest_managed']} | "
        f"{r['manifest_preferred_cpu']} | {r['manifest_accessed_by']} | {r['manifest_readmostly']} | "
        f"{r['manifest_readmostly_rounded']} | {r['manifest_remote_prepare_records']} | "
        f"{r['remote_prepare_pages']} | {r['bg_promote_enqueued_pages']} | {r['bg_promote_start_pages']} | "
        f"{r['bg_promote_committed_pages']} | {r['bg_promote_pause_calls']}/{r['bg_promote_resume_calls']} | "
        f"{r['same_offload_names_as_A']} |"
    )

lines.extend([
    "",
    "## Interpretation Guide",
    "",
    "- If C is close to A and `remote_prepare_pages > 0`, removing ReadMostly fixed the Phase-F remote-PTE-only path.",
    "- If D is close to C and `bg_promote_enqueued_pages > 0`, long-lived bg pending is not the main slowdown without ReadMostly.",
    "- If D is slower than C, pending state still has a secondary cost.",
])

md_path = root / "uvm_phase_f_noreadmostly_noaccessedby_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY
}

require_env
build_vllm_cmd
configure_uvm
KV_CACHE_BYTES="$(kv_cache_bytes)"

{
  echo "run_root=$RUN_ROOT"
  echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "model=$MODEL"
  echo "prompts=$PROMPTS"
  echo "output_len=$OUTPUT_LEN"
  echo "request_rate=$REQUEST_RATE"
  echo "max_model_len=$MAX_MODEL_LEN"
  echo "max_num_seqs=$MAX_NUM_SEQS"
  echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
  echo "kv_cache_gb=$KV_CACHE_GB"
  echo "kv_cache_bytes=$KV_CACHE_BYTES"
  echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
  echo "vllm_cmd=${VLLM_CMD[*]}"
} | tee "$RUN_ROOT/metadata.txt"

run_case A_isolated_uvm_remote isolated_uvm_remote "$PORT_BASE"
run_case C_prepare_remote_read_noreadmostly_noaccessedby isolated_uvm_prepare_remote_noreadmostly_noaccessedby "$((PORT_BASE + 1))"
run_case D_deferred_bg_pending_noreadmostly_noaccessedby isolated_uvm_hybrid_remote_noreadmostly_noaccessedby "$((PORT_BASE + 2))"
summarize 2>&1 | tee "$RUN_ROOT/summary.log"

status_a="$(cat "$RUN_ROOT/A_isolated_uvm_remote/status.txt" 2>/dev/null || echo 1)"
status_c="$(cat "$RUN_ROOT/C_prepare_remote_read_noreadmostly_noaccessedby/status.txt" 2>/dev/null || echo 1)"
status_d="$(cat "$RUN_ROOT/D_deferred_bg_pending_noreadmostly_noaccessedby/status.txt" 2>/dev/null || echo 1)"
log "Done: $RUN_ROOT"
log "Summary: $RUN_ROOT/uvm_phase_f_noreadmostly_noaccessedby_summary.md"
if [[ "$status_a" != "0" || "$status_c" != "0" || "$status_d" != "0" ]]; then
  exit 1
fi
