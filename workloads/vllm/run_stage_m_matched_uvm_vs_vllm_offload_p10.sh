#!/usr/bin/env bash
# Stage M matched parameter-set A/B:
#   1. Run native vLLM CPU offload and record the exact offloaded parameters.
#   2. Run pure UVM managed memory with no custom placement policy.
#   3. Run UVM managed memory with cpu_offload_gb=0 and apply CPU placement to
#      the same parameter set from the native manifest.
#
# This isolates the question:
#   pure cudaMallocManaged UVM vs vLLM pinned-CPU UVA offload vs
#   cudaMallocManaged UVM CPU residency for the same offloaded parameter set.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_matched_uvm_vs_vllm_offload_p10_run}"
PROMPTS="${PROMPTS:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-8}"
NATIVE_SERVER_GPU_MEMORY_UTILIZATION="${NATIVE_SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
UVM_SERVER_GPU_MEMORY_UTILIZATION="${UVM_SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
UVM_MATCHED_CPU_OFFLOAD_MODE="${UVM_MATCHED_CPU_OFFLOAD_MODE:-offload_cpu}"
UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="${UVM_MATCHED_CPU_OFFLOAD_EXECUTOR:-auto}"
UVM_MATCHED_CPU_OFFLOAD_SYNC="${UVM_MATCHED_CPU_OFFLOAD_SYNC:-1}"
UVM_TRACE_MIN_BYTES="${UVM_TRACE_MIN_BYTES:-1048576}"
LAYER_TIMING_ENABLE="${LAYER_TIMING_ENABLE:-1}"
LAYER_TIMING_SYNC="${LAYER_TIMING_SYNC:-1}"
LAYER_TIMING_PATTERN="${LAYER_TIMING_PATTERN:-model\\.layers\\.[0-9]+$|model\\.layers\\.[0-9]+\\.(self_attn|mlp)$}"
LAYER_TIMING_MIN_MS="${LAYER_TIMING_MIN_MS:-0}"
LAYER_TIMING_MAX_RECORDS="${LAYER_TIMING_MAX_RECORDS:-0}"
PORT_BASE="${PORT_BASE:-8000}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

SUMMARY_JSON="$RUN_ROOT/matched_uvm_vs_vllm_offload_summary.json"
SUMMARY_CSV="$RUN_ROOT/matched_uvm_vs_vllm_offload_summary.csv"
SUMMARY_MD="$RUN_ROOT/matched_uvm_vs_vllm_offload_summary.md"
MANIFEST="$RUN_ROOT/native_cpu_offload/offload_manifest.jsonl"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M pure/matched UVM vs vLLM CPU offload A/B/C"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "cpu_offload_gb=$NATIVE_CPU_OFFLOAD_GB"
echo "uvm_matched_mode=$UVM_MATCHED_CPU_OFFLOAD_MODE executor=$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR sync=$UVM_MATCHED_CPU_OFFLOAD_SYNC"
echo "layer_timing_enable=$LAYER_TIMING_ENABLE sync=$LAYER_TIMING_SYNC pattern=$LAYER_TIMING_PATTERN"
echo "==========================================================="

cleanup_leftovers() {
  pkill -f "vllm bench serve" >/dev/null 2>&1 || true
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
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
    if [ "$elapsed" -ge "$STARTUP_TIMEOUT" ]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

run_native_offload_case() {
  local case_root="$RUN_ROOT/native_cpu_offload"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local run_log="$case_root/run.log"
  local layer_timing_log="$case_root/layer_timing.jsonl"
  local port="$PORT_BASE"
  local server_pid=""
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$server_log"
  : > "$bench_log"
  : > "$MANIFEST"
  : > "$layer_timing_log"

  echo "==========================================================="
  echo "Run native_cpu_offload: record exact vLLM offloaded parameters"
  echo "==========================================================="

  cleanup_leftovers
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >>"$run_log" 2>&1 || true
  sleep 2

  (
    cd "$SCRIPT_DIR" || exit 1
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="$MANIFEST" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_LAYER_TIMING_ENABLE="$LAYER_TIMING_ENABLE" \
    VLLM_LAYER_TIMING_FILE="$layer_timing_log" \
    VLLM_LAYER_TIMING_SYNC="$LAYER_TIMING_SYNC" \
    VLLM_LAYER_TIMING_PATTERN="$LAYER_TIMING_PATTERN" \
    VLLM_LAYER_TIMING_MIN_MS="$LAYER_TIMING_MIN_MS" \
    VLLM_LAYER_TIMING_MAX_RECORDS="$LAYER_TIMING_MAX_RECORDS" \
    uv run vllm serve "$MODEL" \
      --enforce-eager \
      --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$NATIVE_SERVER_GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb "$NATIVE_CPU_OFFLOAD_GB" \
      --port "$port" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    {
      echo "===== Benchmark phase: main ====="
      echo "command: uv run --directory $SCRIPT_DIR vllm bench serve --model $MODEL --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts $PROMPTS --sharegpt-output-len $OUTPUT_LEN --seed $SEED --request-rate $REQUEST_RATE --port $port"
    } | tee -a "$bench_log" >/dev/null
    set +e
    uv run --directory "$SCRIPT_DIR" vllm bench serve \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$PROMPTS" \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      --request-rate "$REQUEST_RATE" \
      --port "$port" 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
  else
    status=1
  fi

  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 2
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_leftovers
  echo "$status" > "$case_root/status.txt"
  return 0
}

analyze_uvm_case() {
  local case_name="$1"
  local case_root="$RUN_ROOT/$case_name"
  local analysis_dir="$case_root/analysis"
  local address_log="$case_root/vllm_uvm_address_regions.log"
  local addrs_log="$case_root/uvm_fault_addrs.log"
  local allocator_log="$case_root/vllm_uvm_allocator_trace.log"

  mkdir -p "$analysis_dir"
  if [ ! -s "$address_log" ] || [ ! -s "$addrs_log" ]; then
    echo "Skip UVM fault address analysis for $case_name: missing address/fault log" >&2
    return 0
  fi

  python3 "$SCRIPT_DIR/analyze_uvm_fault_addresses.py" \
    --address-log "$address_log" \
    --fault-log "$addrs_log" \
    --regions-csv "$analysis_dir/regions.csv" \
    --faults-csv "$analysis_dir/faults.csv" \
    --summary-json "$analysis_dir/addr_summary.json" \
    --top-n 20 >>"$case_root/run.log" 2>&1 || true

  local allocator_args=()
  if [ -s "$allocator_log" ]; then
    allocator_args=(--allocator-log "$allocator_log")
  fi
  python3 "$SCRIPT_DIR/deep_dive_uvm_faults.py" \
    --address-log "$address_log" \
    --fault-log "$addrs_log" \
    "${allocator_args[@]}" \
    --top-gaps 20 \
    --gap-csv "$analysis_dir/gaps.csv" \
    --summary-json "$analysis_dir/deep_summary.json" >>"$case_root/run.log" 2>&1 || true
}

run_pure_uvm_case() {
  local case_root="$RUN_ROOT/pure_uvm"
  local run_log="$case_root/run.log"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local stats_log="$case_root/uvm_fault_stats.log"
  local addrs_log="$case_root/uvm_fault_addrs.log"
  local address_log="$case_root/vllm_uvm_address_regions.log"
  local allocator_log="$case_root/vllm_uvm_allocator_trace.log"
  local layer_timing_log="$case_root/layer_timing.jsonl"
  local port="$PORT_BASE"
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$stats_log"
  : > "$bench_log"
  : > "$server_log"
  : > "$allocator_log"
  : > "$address_log"
  : > "$addrs_log"
  : > "$layer_timing_log"

  echo "==========================================================="
  echo "Run pure_uvm: managed allocations without custom UVM placement policy"
  echo "==========================================================="

  set +e
  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  VLLM_LAYER_TIMING_ENABLE="$LAYER_TIMING_ENABLE" \
  VLLM_LAYER_TIMING_FILE="$layer_timing_log" \
  VLLM_LAYER_TIMING_SYNC="$LAYER_TIMING_SYNC" \
  VLLM_LAYER_TIMING_PATTERN="$LAYER_TIMING_PATTERN" \
  VLLM_LAYER_TIMING_MIN_MS="$LAYER_TIMING_MIN_MS" \
  VLLM_LAYER_TIMING_MAX_RECORDS="$LAYER_TIMING_MAX_RECORDS" \
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --trace-log "$stats_log" \
    --with-address-log \
    --address-trace-log "$addrs_log" \
    --allocator-log "$allocator_log" \
    --address-log "$address_log" \
    --server-log "$server_log" \
    --bench-log "$bench_log" \
    --uvm-trace-min-bytes "$UVM_TRACE_MIN_BYTES" \
    --uvm-policy-enable 0 \
    --uvm-policy-runtime-enable 0 \
    --uvm-device-direct-enable 0 \
    --uvm-residency-enable 0 \
    --uvm-kv-runtime-enable 0 \
    --uvm-kv-active-protect-enable 0 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-prefetch-trace-enable 0 \
    --uvm-weight-prefetch-hit-trace-enable 0 \
    --uvm-weight-prefetch-hit-driver-query-enable 0 \
    --uvm-weight-batch-executor-enable 0 \
    --uvm-weight-offload-enable 0 \
    --uvm-init-weight-offload-enable 0 \
    --uvm-init-weight-hot-prefetch-enable 0 \
    --uvm-pool-coordinator-enable 0 \
    --uvm-pool-registry-enable 0 \
    --uvm-scratch-pool-enable 0 \
    --uvm-moe-routing-trace-enable 0 \
    --uvm-weight-map-enable 0 \
    --uvm-unknown-detail-enable 0 \
    --auto-gap-watch-enable 0 \
    --uvm-matched-cpu-offload-enable 0 \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$UVM_SERVER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --port "$port" 2>&1 | tee "$run_log"
  status=${PIPESTATUS[0]}
  set -u
  echo "$status" > "$case_root/status.txt"
  cleanup_leftovers
  analyze_uvm_case "pure_uvm"
  return 0
}

run_uvm_matched_case() {
  local case_root="$RUN_ROOT/uvm_matched_cpu_offload"
  local run_log="$case_root/run.log"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local stats_log="$case_root/uvm_fault_stats.log"
  local addrs_log="$case_root/uvm_fault_addrs.log"
  local address_log="$case_root/vllm_uvm_address_regions.log"
  local allocator_log="$case_root/vllm_uvm_allocator_trace.log"
  local matched_trace="$case_root/uvm_matched_cpu_offload_trace.jsonl"
  local layer_timing_log="$case_root/layer_timing.jsonl"
  local port="$PORT_BASE"
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$matched_trace"
  : > "$stats_log"
  : > "$bench_log"
  : > "$server_log"
  : > "$allocator_log"
  : > "$address_log"
  : > "$addrs_log"
  : > "$layer_timing_log"

  echo "==========================================================="
  echo "Run uvm_matched_cpu_offload: same manifest, UVM managed CPU residency"
  echo "==========================================================="

  set +e
  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  VLLM_LAYER_TIMING_ENABLE="$LAYER_TIMING_ENABLE" \
  VLLM_LAYER_TIMING_FILE="$layer_timing_log" \
  VLLM_LAYER_TIMING_SYNC="$LAYER_TIMING_SYNC" \
  VLLM_LAYER_TIMING_PATTERN="$LAYER_TIMING_PATTERN" \
  VLLM_LAYER_TIMING_MIN_MS="$LAYER_TIMING_MIN_MS" \
  VLLM_LAYER_TIMING_MAX_RECORDS="$LAYER_TIMING_MAX_RECORDS" \
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --trace-log "$stats_log" \
    --with-address-log \
    --address-trace-log "$addrs_log" \
    --allocator-log "$allocator_log" \
    --address-log "$address_log" \
    --server-log "$server_log" \
    --bench-log "$bench_log" \
    --uvm-trace-min-bytes "$UVM_TRACE_MIN_BYTES" \
    --uvm-policy-enable 0 \
    --uvm-policy-runtime-enable 0 \
    --uvm-device-direct-enable 0 \
    --uvm-residency-enable 0 \
    --uvm-kv-runtime-enable 0 \
    --uvm-kv-active-protect-enable 0 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-prefetch-trace-enable 0 \
    --uvm-weight-prefetch-hit-trace-enable 0 \
    --uvm-weight-prefetch-hit-driver-query-enable 0 \
    --uvm-weight-batch-executor-enable 1 \
    --uvm-weight-executor "$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR" \
    --uvm-weight-offload-enable 0 \
    --uvm-init-weight-offload-enable 0 \
    --uvm-init-weight-hot-prefetch-enable 0 \
    --uvm-pool-coordinator-enable 0 \
    --uvm-pool-registry-enable 0 \
    --uvm-scratch-pool-enable 0 \
    --uvm-moe-routing-trace-enable 0 \
    --uvm-weight-map-enable 0 \
    --uvm-unknown-detail-enable 0 \
    --auto-gap-watch-enable 0 \
    --uvm-matched-cpu-offload-enable 1 \
    --uvm-matched-cpu-offload-manifest-file "$MANIFEST" \
    --uvm-matched-cpu-offload-trace-file "$matched_trace" \
    --uvm-matched-cpu-offload-mode "$UVM_MATCHED_CPU_OFFLOAD_MODE" \
    --uvm-matched-cpu-offload-executor "$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR" \
    --uvm-matched-cpu-offload-sync "$UVM_MATCHED_CPU_OFFLOAD_SYNC" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$UVM_SERVER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --port "$port" 2>&1 | tee "$run_log"
  status=${PIPESTATUS[0]}
  set -u
  echo "$status" > "$case_root/status.txt"
  cleanup_leftovers
  analyze_uvm_case "uvm_matched_cpu_offload"
  return 0
}

run_native_offload_case
if [ ! -s "$MANIFEST" ]; then
  echo "ERROR: native offload manifest is empty: $MANIFEST" >&2
  exit 1
fi
run_pure_uvm_case
run_uvm_matched_case

python3 - "$RUN_ROOT" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])

cases = [
    ("native_cpu_offload", "vllm_native_cpu_offload"),
    ("pure_uvm", "uvm_managed_no_custom_policy"),
    ("uvm_matched_cpu_offload", "uvm_matched_cpu_residency"),
]

bench_patterns = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}
fault_patterns = {
    "delta_faults": r"\bdelta_faults=([0-9]+)",
    "delta_after_dedup": r"\bdelta_after_dedup=([0-9]+)",
    "delta_kv_faults": r"\bdelta_kv_faults=([0-9]+)",
    "delta_kv_after_dedup": r"\bdelta_kv_after_dedup=([0-9]+)",
}

def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def status(path: Path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def first_float(s: str, pattern: str):
    m = re.search(pattern, s)
    return float(m.group(1)) if m else None

def first_int(s: str, pattern: str):
    m = re.search(pattern, s)
    return int(m.group(1)) if m else None

def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return (new - base) * 100.0 / base

def manifest_stats(path: Path):
    count = 0
    total = 0
    by_prefix = {}
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "cpu_offload_parameter":
            continue
        count += 1
        total += int(rec.get("bytes") or 0)
        name = str(rec.get("full_parameter_name") or "")
        prefix = ".".join(name.split(".")[:3]) if "." in name else name
        by_prefix[prefix] = by_prefix.get(prefix, 0) + 1
    return count, total, by_prefix

def matched_trace_stats(path: Path):
    matched_names = set()
    attempted = 0
    missing = 0
    batch_successes = None
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = rec.get("event")
        if event == "matched_cpu_offload_parameter":
            name = rec.get("parameter_name")
            if isinstance(name, str):
                matched_names.add(name)
            attempted += int(rec.get("bytes") or 0)
            if batch_successes is None and "batch_successes" in rec:
                batch_successes = rec.get("batch_successes")
        elif event == "matched_cpu_offload_missing_parameter":
            missing += 1
    return len(matched_names), batch_successes, attempted, missing

def layer_timing_stats(path: Path):
    by_phase = {}
    top_modules = {}
    records = 0
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") != "layer_timing":
            continue
        phase = str(rec.get("phase") or "unknown")
        module_name = str(rec.get("module_name") or "")
        elapsed = float(rec.get("elapsed_ms") or 0.0)
        records += 1
        phase_stats = by_phase.setdefault(
            phase,
            {
                "records": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "step_ids": set(),
            },
        )
        phase_stats["records"] += 1
        phase_stats["total_ms"] += elapsed
        phase_stats["max_ms"] = max(phase_stats["max_ms"], elapsed)
        step_id = rec.get("step_id")
        if step_id is not None:
            phase_stats["step_ids"].add(step_id)
        module_stats = top_modules.setdefault(
            module_name,
            {
                "records": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "phase_counts": {},
            },
        )
        module_stats["records"] += 1
        module_stats["total_ms"] += elapsed
        module_stats["max_ms"] = max(module_stats["max_ms"], elapsed)
        phase_counts = module_stats["phase_counts"]
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    normalized = {}
    for phase, stats in by_phase.items():
        steps = len(stats["step_ids"])
        normalized[phase] = {
            "records": stats["records"],
            "steps": steps,
            "total_ms": stats["total_ms"],
            "avg_record_ms": (
                stats["total_ms"] / stats["records"]
                if stats["records"]
                else None
            ),
            "avg_step_ms": stats["total_ms"] / steps if steps else None,
            "max_ms": stats["max_ms"],
        }
    top = []
    for module_name, stats in top_modules.items():
        top.append(
            {
                "module_name": module_name,
                "records": stats["records"],
                "total_ms": stats["total_ms"],
                "avg_record_ms": (
                    stats["total_ms"] / stats["records"]
                    if stats["records"]
                    else None
                ),
                "max_ms": stats["max_ms"],
                "phase_counts": stats["phase_counts"],
            }
        )
    top.sort(key=lambda item: item["total_ms"], reverse=True)
    return {
        "records": records,
        "by_phase": normalized,
        "top_modules": top[:20],
    }

def fault_class_stats(path: Path):
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    classes = (
        report.get("fault_analysis", {})
        .get("class_summaries", {})
    )
    result = {}
    for name in ("weight", "kv_cache", "unknown"):
        item = classes.get(name) or {}
        result[name] = {
            "faults": item.get("faults"),
            "unique_pages": item.get("unique_pages"),
            "write_ratio": item.get("write_ratio"),
            "read_ratio": item.get("read_ratio"),
        }
    result["parsed_fault_records"] = report.get(
        "fault_analysis", {}
    ).get("parsed_fault_records")
    return result

manifest_count, manifest_bytes, manifest_by_prefix = manifest_stats(
    root / "native_cpu_offload" / "offload_manifest.jsonl"
)
matched_count, matched_batch_successes, matched_bytes, matched_missing = matched_trace_stats(
    root / "uvm_matched_cpu_offload" / "uvm_matched_cpu_offload_trace.jsonl"
)

matched_case_name = "uvm_matched_cpu_offload"

rows = []
for case, kind in cases:
    case_root = root / case
    bench = text(case_root / "vllm_bench.log")
    run_log = text(case_root / "run.log")
    server = text(case_root / "vllm_server.log")
    row = {
        "case": case,
        "kind": kind,
        "exit_code": status(case_root / "status.txt"),
    }
    timing = layer_timing_stats(case_root / "layer_timing.jsonl")
    fault_classes = fault_class_stats(case_root / "analysis" / "deep_summary.json")
    for key, pat in bench_patterns.items():
        row[key] = first_float(bench, pat)
    for key, pat in fault_patterns.items():
        row[key] = first_int(run_log, pat)
    load = re.search(
        r"Model loading took\s+([0-9.]+)\s+GiB memory and\s+([0-9.]+)\s+seconds",
        server,
    )
    row["model_load_gib"] = float(load.group(1)) if load else None
    row["model_load_s"] = float(load.group(2)) if load else None
    row["manifest_parameter_count"] = manifest_count
    row["manifest_bytes"] = manifest_bytes
    row["layer_timing_records"] = timing["records"]
    row["prefill_layer_avg_step_ms"] = (
        timing["by_phase"].get("prefill", {}).get("avg_step_ms")
    )
    row["decode_layer_avg_step_ms"] = (
        timing["by_phase"].get("decode", {}).get("avg_step_ms")
    )
    row["mixed_layer_avg_step_ms"] = (
        timing["by_phase"].get("mixed", {}).get("avg_step_ms")
    )
    row["layer_timing"] = timing
    row["fault_classes"] = fault_classes
    row["fault_weight_faults"] = (
        fault_classes.get("weight", {}).get("faults")
        if fault_classes
        else None
    )
    row["fault_kv_cache_faults"] = (
        fault_classes.get("kv_cache", {}).get("faults")
        if fault_classes
        else None
    )
    row["fault_unknown_faults"] = (
        fault_classes.get("unknown", {}).get("faults")
        if fault_classes
        else None
    )
    is_matched_case = case == matched_case_name
    row["uvm_matched_parameter_count"] = matched_count if is_matched_case else None
    row["uvm_matched_batch_successes"] = matched_batch_successes if is_matched_case else None
    row["uvm_matched_coverage_pct"] = (
        matched_count * 100.0 / manifest_count
        if is_matched_case and manifest_count
        else None
    )
    row["uvm_matched_bytes"] = matched_bytes if is_matched_case else None
    row["uvm_matched_missing_count"] = matched_missing if is_matched_case else None
    rows.append(row)

base = rows[0] if rows else {}
for row in rows:
    row["tpot_pct_vs_native_offload"] = pct(
        row.get("mean_tpot_ms"), base.get("mean_tpot_ms")
    )
    row["tok_s_pct_vs_native_offload"] = pct(
        row.get("output_tok_s"), base.get("output_tok_s")
    )
    row["ttft_pct_vs_native_offload"] = pct(
        row.get("mean_ttft_ms"), base.get("mean_ttft_ms")
    )

summary = {
    "run_root": str(root),
    "rows": rows,
    "manifest": {
        "parameter_count": manifest_count,
        "bytes": manifest_bytes,
        "top_prefix_counts": sorted(
            manifest_by_prefix.items(), key=lambda kv: kv[1], reverse=True
        )[:20],
    },
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

fields = [
    "case", "kind", "exit_code", "successful_requests", "failed_requests",
    "output_tok_s", "mean_tpot_ms", "mean_ttft_ms", "mean_itl_ms",
    "tok_s_pct_vs_native_offload", "tpot_pct_vs_native_offload",
    "ttft_pct_vs_native_offload", "delta_faults", "delta_after_dedup",
    "delta_kv_faults", "fault_weight_faults", "fault_unknown_faults",
    "fault_kv_cache_faults", "layer_timing_records",
    "prefill_layer_avg_step_ms", "decode_layer_avg_step_ms",
    "mixed_layer_avg_step_ms", "manifest_parameter_count", "manifest_bytes",
    "uvm_matched_parameter_count", "uvm_matched_coverage_pct",
    "uvm_matched_batch_successes",
    "uvm_matched_bytes", "uvm_matched_missing_count",
]
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fields})

lines = [
    "# Stage M Pure/Matched UVM vs vLLM CPU Offload",
    "",
    f"- run_root: `{root}`",
    f"- native_manifest: `{root / 'native_cpu_offload' / 'offload_manifest.jsonl'}`",
    f"- manifest_parameters: `{manifest_count}`",
    f"- manifest_bytes: `{manifest_bytes}`",
    "",
    "| case | tok/s | TPOT ms | TTFT ms | failed | raw UVM faults | dedup UVM faults | weight faults | unknown faults | KV faults | prefill layer ms/step | decode layer ms/step | matched/manifest | tok/s vs native | TPOT vs native |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    matched = ""
    batch_successes = ""
    if row["case"] == matched_case_name:
        matched = f"{row.get('uvm_matched_parameter_count')}/{row.get('manifest_parameter_count')}"
        batch_successes = str(row.get("uvm_matched_batch_successes"))
    lines.append(
        "| {case} | {output_tok_s} | {mean_tpot_ms} | {mean_ttft_ms} | "
        "{failed_requests} | {delta_faults} | {delta_after_dedup} | "
        "{weight_faults} | {unknown_faults} | {kv_faults} | "
        "{prefill_layer_ms} | {decode_layer_ms} | {matched} | "
        "{tok_delta} | {tpot_delta} |".format(
            case=row.get("case"),
            output_tok_s=row.get("output_tok_s"),
            mean_tpot_ms=row.get("mean_tpot_ms"),
            mean_ttft_ms=row.get("mean_ttft_ms"),
            failed_requests=row.get("failed_requests"),
            delta_faults=row.get("delta_faults"),
            delta_after_dedup=row.get("delta_after_dedup"),
            weight_faults=row.get("fault_weight_faults"),
            unknown_faults=row.get("fault_unknown_faults"),
            kv_faults=row.get("fault_kv_cache_faults"),
            prefill_layer_ms=row.get("prefill_layer_avg_step_ms"),
            decode_layer_ms=row.get("decode_layer_avg_step_ms"),
            matched=matched,
            tok_delta=row.get("tok_s_pct_vs_native_offload"),
            tpot_delta=row.get("tpot_pct_vs_native_offload"),
        )
    )
lines.extend([
    "",
    "## Top Layer Timing Modules",
    "",
])
for row in rows:
    lines.append(f"### {row.get('case')}")
    top = row.get("layer_timing", {}).get("top_modules", [])[:10]
    if not top:
        lines.append("")
        lines.append("- no layer timing records")
        lines.append("")
        continue
    lines.append("")
    lines.append("| module | records | total ms | avg ms | max ms |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in top:
        lines.append(
            "| {module} | {records} | {total:.3f} | {avg:.3f} | {maxv:.3f} |".format(
                module=item.get("module_name"),
                records=item.get("records"),
                total=float(item.get("total_ms") or 0.0),
                avg=float(item.get("avg_record_ms") or 0.0),
                maxv=float(item.get("max_ms") or 0.0),
            )
        )
    lines.append("")
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"summary_json: {summary_json}")
print(f"summary_csv: {summary_csv}")
print(f"summary_md: {summary_md}")
PY

echo "Stage M matched UVM vs vLLM CPU offload finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
