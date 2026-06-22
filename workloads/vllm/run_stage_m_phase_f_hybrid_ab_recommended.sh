#!/usr/bin/env bash
# Recommended Phase-F hybrid end-to-end A/B runner.
#
# This is a thin wrapper around run_stage_m_isolated_uvm_offload_remote.sh.
# It compares the same vLLM CPU-offloaded weight set across:
#   1. native vLLM pinned-UVA CPU offload
#   2. isolated UVM migration offload
#   3. isolated UVM remote-read offload
#   4. isolated UVM hybrid remote-read + deferred bg promotion

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"
STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"

PROFILE="${PROFILE:-perf_lite}"
case "$PROFILE" in
  smoke)
    DEFAULT_PROMPTS=1
    DEFAULT_OUTPUT_LEN=16
    DEFAULT_MAX_MODEL_LEN=1024
    ;;
  perf_lite)
    DEFAULT_PROMPTS=5
    DEFAULT_OUTPUT_LEN=16
    DEFAULT_MAX_MODEL_LEN=1024
    ;;
  perf)
    DEFAULT_PROMPTS=20
    DEFAULT_OUTPUT_LEN=64
    DEFAULT_MAX_MODEL_LEN=2048
    ;;
  long)
    DEFAULT_PROMPTS=50
    DEFAULT_OUTPUT_LEN=128
    DEFAULT_MAX_MODEL_LEN=4096
    ;;
  *)
    echo "ERROR: unknown PROFILE=$PROFILE; use smoke, perf_lite, perf, or long." >&2
    exit 2
    ;;
esac

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_phase_f_hybrid_ab_${PROFILE}_$(date -u +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-$DEFAULT_PROMPTS}"
OUTPUT_LEN="${OUTPUT_LEN:-$DEFAULT_OUTPUT_LEN}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}"
REQUEST_RATE="${REQUEST_RATE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PORT_BASE="${PORT_BASE:-8420}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
RUN_PING_IOCTL="${RUN_PING_IOCTL:-1}"
RESET_LATENCY_STATS="${RESET_LATENCY_STATS:-1}"

BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"

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

  log "WARN: cannot write $name=$value; continuing with current value $(cat "$path" 2>/dev/null || echo unknown)"
}

require_modified_uvm() {
  if [[ ! -r "$STATS_PATH" ]]; then
    echo "ERROR: $STATS_PATH is not readable. Load the patched nvidia_uvm module first." >&2
    exit 1
  fi

  local missing=0
  for param in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable \
    uvm_latency_bg_promote_max_pages \
    uvm_latency_bg_promote_delay_ms \
    uvm_latency_bg_promote_reschedule_delay_ms; do
    if [[ ! -e "$PARAM_ROOT/$param" ]]; then
      echo "ERROR: missing $PARAM_ROOT/$param; this does not look like the patched UVM module." >&2
      missing=1
    fi
  done
  [[ "$missing" == "0" ]] || exit 1
}

configure_uvm_params() {
  # Keep global experiments quiet. The hybrid path uses explicit Phase-F actions.
  write_param uvm_latency_remote_first_enable 0
  write_param uvm_latency_bg_promote_enable 0
  write_param uvm_latency_bg_promote_max_pages "$BG_MAX_PAGES"
  write_param uvm_latency_bg_promote_delay_ms "$BG_DELAY_MS"
  write_param uvm_latency_bg_promote_reschedule_delay_ms "$BG_RESCHEDULE_DELAY_MS"
  write_param uvm_phase_f_route_log_delay_ms "$PHASE_F_ROUTE_LOG_DELAY_MS"
}

snapshot_environment() {
  {
    echo "run_root=$RUN_ROOT"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "profile=$PROFILE"
    echo "model=$MODEL"
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "port_base=$PORT_BASE"
    echo "bg_max_pages=$BG_MAX_PAGES"
    echo "bg_delay_ms=$BG_DELAY_MS"
    echo "bg_reschedule_delay_ms=$BG_RESCHEDULE_DELAY_MS"
    echo "phase_f_route_log_delay_ms=$PHASE_F_ROUTE_LOG_DELAY_MS"
    echo "vllm_uvm_phase_f_ioctl_mode=policy"
    echo "vllm_uvm_phase_f_fd_mode=adopt"
    echo "vllm_uvm_phase_f_cpp_helper=1"
    echo "vllm_uvm_hybrid_round_pages=1"
    echo "vllm_uvm_hybrid_round_readmostly=1"
    echo
    echo "nvidia-smi:"
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    echo "UVM module parameters:"
    for p in \
      "$PARAM_ROOT/uvm_latency_remote_first_enable" \
      "$PARAM_ROOT/uvm_latency_bg_promote_enable" \
      "$PARAM_ROOT/uvm_latency_bg_promote_max_pages" \
      "$PARAM_ROOT/uvm_latency_bg_promote_delay_ms" \
      "$PARAM_ROOT/uvm_latency_bg_promote_reschedule_delay_ms" \
      "$PARAM_ROOT/uvm_phase_f_route_log_delay_ms" \
      "$PARAM_ROOT/uvm_perf_prefetch_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_migration_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_threshold" \
      "$PARAM_ROOT/uvm_ats_mode"; do
      [[ -r "$p" ]] && printf '  %-48s %s\n' "$(basename "$p")" "$(cat "$p")"
    done
  } | tee "$RUN_ROOT/metadata.txt"
}

run_ping_ioctl() {
  if [[ "$RUN_PING_IOCTL" != "1" ]]; then
    return 0
  fi

  log "Running Phase-F ping-ioctl sanity check"
  set +e
  VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
  VLLM_UVM_PHASE_F_FD_MODE=adopt \
  "$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
    --no-enable-tests \
    --step ping-ioctl 2>&1 | tee "$RUN_ROOT/ping_ioctl.log"
  local status=${PIPESTATUS[0]}
  set -u

  if [[ "$status" != "0" ]]; then
    echo "ERROR: ping-ioctl failed. See $RUN_ROOT/ping_ioctl.log" >&2
    exit "$status"
  fi
}

reset_latency_stats() {
  if [[ "$RESET_LATENCY_STATS" != "1" ]]; then
    return 0
  fi

  local reset_path="/proc/driver/nvidia-uvm/latency_stats_reset"
  if [[ ! -e "$reset_path" ]]; then
    log "WARN: missing $reset_path; kernel deltas may include earlier runs"
    return 0
  fi

  log "Resetting UVM latency stats before A/B"
  if [[ -w "$reset_path" ]]; then
    printf '1\n' > "$reset_path" || log "WARN: failed to reset UVM latency stats"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    printf '1\n' | sudo -n tee "$reset_path" >/dev/null || log "WARN: sudo reset failed for UVM latency stats"
    return 0
  fi

  log "WARN: cannot reset UVM latency stats; kernel deltas may include earlier runs"
}

write_extra_summary() {
  python3 - "$RUN_ROOT" "$STATS_PATH" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_path = root / "phase_f_hybrid_ab_extra_summary.md"
cases = [
    "native_cpu_offload",
    "isolated_uvm_migration_offload",
    "isolated_uvm_remote_offload",
    "isolated_uvm_hybrid_remote_offload",
]
bench_patterns = {
    "ok": r"Successful requests:\s+([0-9.]+)",
    "fail": r"Failed requests:\s+([0-9.]+)",
    "tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "tpot": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "ttft": r"Mean TTFT \(ms\):\s+([0-9.]+)",
}

def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(blob, pattern):
    m = re.search(pattern, blob)
    return float(m.group(1)) if m else None

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def parse_stats(path):
    stats = {}
    for line in read(path).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                stats[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return stats

def manifest_stats(path):
    s = {
        "params": 0,
        "bytes": 0,
        "managed": 0,
        "readmostly": 0,
        "readmostly_rounded": 0,
        "remote_success_records": 0,
        "remote_success_ranges": 0,
        "remote_failures": 0,
        "rounded": 0,
        "gate_events": 0,
        "pause_events": 0,
        "resume_events": 0,
        "start_events": 0,
        "start_ranges": 0,
    }
    for line in read(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        ev = rec.get("event")
        if ev == "cpu_offload_hybrid_remote_gate":
            s["gate_events"] += 1
            if int(rec.get("pause_success") or 0) > 0:
                s["pause_events"] += 1
            if int(rec.get("resume_success") or 0) > 0:
                s["resume_events"] += 1
            start = int(rec.get("start_success") or 0)
            if start > 0:
                s["start_events"] += 1
                s["start_ranges"] += start
            continue
        if ev != "cpu_offload_parameter":
            continue
        s["params"] += 1
        s["bytes"] += int(rec.get("bytes") or 0)
        s["managed"] += int(rec.get("uvm_managed") is True)
        s["rounded"] += int(rec.get("uvm_phase_f_rounded") is True)
        s["readmostly"] += int((rec.get("uvm_readmostly_result") or 0) > 0)
        s["readmostly_rounded"] += int((rec.get("uvm_readmostly_rounded_result") or 0) > 0)
        success = int(rec.get("uvm_remote_read_bg_success") or 0)
        failure = int(rec.get("uvm_remote_read_bg_failure") or 0)
        if success > 0:
            s["remote_success_records"] += 1
            s["remote_success_ranges"] += success
        s["remote_failures"] += failure
    return s

rows = []
for case in cases:
    case_root = root / case
    bench = read(case_root / "bench.log")
    row = {"case": case}
    try:
        row["exit"] = int(read(case_root / "status.txt").strip())
    except Exception:
        row["exit"] = None
    for key, pattern in bench_patterns.items():
        row[key] = first_float(bench, pattern)
    row.update(manifest_stats(case_root / "offload_manifest.jsonl"))
    rows.append(row)

native_tok = rows[0].get("tok_s") if rows else None
native_tpot = rows[0].get("tpot") if rows else None
for row in rows:
    row["tok_ratio"] = row.get("tok_s") / native_tok if row.get("tok_s") is not None and native_tok else None
    row["tpot_ratio"] = row.get("tpot") / native_tpot if row.get("tpot") is not None and native_tpot else None

before = parse_stats(root / "latency_stats_before.txt")
after = parse_stats(root / "latency_stats_after.txt")
interesting = [
    "remote_prepare_calls",
    "remote_prepare_pages",
    "remote_prepare_mapped_pages",
    "bg_promote_enqueued_pages",
    "bg_promote_copied_pages",
    "bg_promote_committed_pages",
    "bg_promote_cancelled_pages",
    "bg_promote_failed_already_gpu_resident",
    "bg_promote_failed_not_cpu_resident",
    "bg_promote_pause_calls",
    "bg_promote_resume_calls",
    "bg_promote_gate_skips",
]

lines = [
    "# Phase-F Hybrid A/B Extra Summary",
    "",
    f"- run_root: `{root}`",
    f"- primary summary: `{root / 'isolated_uvm_offload_summary.md'}`",
    "",
    "## Benchmark And Manifest",
    "",
    "| case | exit | ok/fail | tok/s | TPOT ms | TTFT ms | tok/native | TPOT/native | params | managed | bytes GiB | ReadMostly | remote-bg records | remote-bg failures | gate events | gate start ranges |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['case']} | {fmt(r['exit'])} | {fmt(r.get('ok'),0)}/{fmt(r.get('fail'),0)} | "
        f"{fmt(r.get('tok_s'))} | {fmt(r.get('tpot'))} | {fmt(r.get('ttft'))} | "
        f"{fmt(r.get('tok_ratio'))} | {fmt(r.get('tpot_ratio'))} | "
        f"{r['params']} | {r['managed']} | {r['bytes'] / (1024**3):.3f} | "
        f"{r['readmostly']} | {r['remote_success_records']} | {r['remote_failures']} | "
        f"{r['gate_events']} | {r['start_ranges']} |"
    )

lines.extend(["", "## Kernel Counter Delta", ""])
lines.append("| counter | before | after | delta |")
lines.append("|---|---:|---:|---:|")
for key in interesting:
    b = before.get(key, 0)
    a = after.get(key, 0)
    lines.append(f"| {key} | {b} | {a} | {a - b} |")

lines.extend([
    "",
    "## Reading The Result",
    "",
    "- First validity check: all four cases should have failed requests = 0.",
    "- Hybrid validity check: `remote-bg records` should match the offloaded parameter count, and `remote-bg failures` should be 0.",
    "- Mechanism signal: `remote_prepare_pages` and bg-promotion counters should increase in the kernel delta.",
    "- Performance signal: compare `tok/native` and `TPOT/native`; values above 1 for tok/s or below 1 for TPOT favor the tested backend.",
])

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_path.read_text(encoding="utf-8"))
PY
}

log "Checking patched UVM module"
require_modified_uvm
configure_uvm_params
snapshot_environment
run_ping_ioctl
reset_latency_stats

cp "$STATS_PATH" "$RUN_ROOT/latency_stats_before.txt" 2>/dev/null || true

log "Starting isolated offload A/B"
set +e
RUN_ROOT="$RUN_ROOT" \
MODEL="$MODEL" \
PROMPTS="$PROMPTS" \
OUTPUT_LEN="$OUTPUT_LEN" \
REQUEST_RATE="$REQUEST_RATE" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" \
MAX_NUM_SEQS="$MAX_NUM_SEQS" \
CPU_OFFLOAD_GB="$CPU_OFFLOAD_GB" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
PORT_BASE="$PORT_BASE" \
STARTUP_TIMEOUT="$STARTUP_TIMEOUT" \
VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
VLLM_UVM_PHASE_F_FD_MODE=adopt \
VLLM_UVM_PHASE_F_CPP_HELPER=1 \
VLLM_UVM_HYBRID_ROUND_PAGES=1 \
VLLM_UVM_HYBRID_ROUND_READMOSTLY=1 \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="$PHASE_F_ROUTE_LOG_DELAY_MS" \
"$SCRIPT_DIR/run_stage_m_isolated_uvm_offload_remote.sh" 2>&1 | tee "$RUN_ROOT/driver.log"
status=${PIPESTATUS[0]}
set -u

cp "$STATS_PATH" "$RUN_ROOT/latency_stats_after.txt" 2>/dev/null || true

log "Writing extra summary"
write_extra_summary 2>&1 | tee "$RUN_ROOT/extra_summary.log"

log "Done: $RUN_ROOT"
log "Primary summary: $RUN_ROOT/isolated_uvm_offload_summary.md"
log "Extra summary: $RUN_ROOT/phase_f_hybrid_ab_extra_summary.md"
exit "$status"
