#!/usr/bin/env bash
# Stage M native-vs-UVM full-ish GPU A/B.
#
# Purpose:
#   Answer whether UVM managed allocations are already slower than ordinary
#   vLLM CUDA allocations when no cold-weight policy is active.
#
# Cases:
#   native_full_gpu
#     VLLM_USE_UVM is unset. No UVM trace/offload/prefetch/residency.
#
#   uvm_fullish_gpu
#     VLLM_USE_UVM=1. No cold offload, no runtime prefetch, no active eviction,
#     and most expensive diagnostic traces are disabled.
#
#   uvm_fullish_gpu_diag
#     Same placement as uvm_fullish_gpu, but with allocator/address/fault
#     diagnostics enabled so the clean-vs-diag overhead is visible.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_native_vs_uvm_fullish_ab_run}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NATIVE_SERVER_GPU_MEMORY_UTILIZATION="${NATIVE_SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
UVM_SERVER_GPU_MEMORY_UTILIZATION="${UVM_SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-8}"
RUN_NATIVE_CPU_OFFLOAD="${RUN_NATIVE_CPU_OFFLOAD:-1}"
PORT_BASE="${PORT_BASE:-8000}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

# Keep the clean UVM case intentionally quiet. It still captures fault stats,
# but avoids per-fault address logging and allocator/detail traces.
UVM_TRACE_MIN_BYTES="${UVM_TRACE_MIN_BYTES:-1048576}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M Native Full-GPU vs UVM Full-ish GPU A/B"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN"
echo "native_gpu_memory_utilization=$NATIVE_SERVER_GPU_MEMORY_UTILIZATION"
echo "uvm_gpu_memory_utilization=$UVM_SERVER_GPU_MEMORY_UTILIZATION"
echo "native_cpu_offload_gb=$NATIVE_CPU_OFFLOAD_GB run_native_cpu_offload=$RUN_NATIVE_CPU_OFFLOAD"
echo "==========================================================="

cleanup_leftovers() {
  pkill -f "vllm bench serve" >/dev/null 2>&1 || true
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
}

wait_for_native_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Native server exited before readiness. Log tail:" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [ "$elapsed" -ge "$STARTUP_TIMEOUT" ]; then
      echo "Timed out waiting for native server readiness after ${elapsed}s" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

run_native_case() {
  local case_name="$1"
  local cpu_offload_gb="$2"
  local case_root="$RUN_ROOT/$case_name"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local run_log="$case_root/run.log"
  local port="$PORT_BASE"
  local server_pid=""
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$server_log"
  : > "$bench_log"

  echo "==========================================================="
  echo "Run $case_name: ordinary vLLM CUDA allocations cpu_offload_gb=$cpu_offload_gb"
  echo "==========================================================="

  cleanup_leftovers
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >>"$run_log" 2>&1 || true
  sleep 2

  (
    cd "$SCRIPT_DIR" || exit 1
    local serve_cmd=(
      uv run vllm serve "$MODEL"
      --enforce-eager \
      --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$NATIVE_SERVER_GPU_MEMORY_UTILIZATION" \
      --port "$port"
    )
    if [ "$cpu_offload_gb" != "0" ]; then
      serve_cmd+=(--cpu-offload-gb "$cpu_offload_gb")
    fi
    "${serve_cmd[@]}" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_native_server "$port" "$server_pid" "$server_log"; then
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

run_uvm_case() {
  local case_name="$1"
  local diag="$2"
  local case_root="$RUN_ROOT/$case_name"
  local run_log="$case_root/run.log"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local allocator_log="$case_root/vllm_uvm_allocator_trace.log"
  local address_log="$case_root/vllm_uvm_address_regions.log"
  local stats_log="$case_root/uvm_fault_stats.log"
  local addrs_log="$case_root/uvm_fault_addrs.log"
  local port="$PORT_BASE"
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"

  echo "==========================================================="
  echo "Run $case_name: UVM full-ish GPU diag=$diag"
  echo "==========================================================="

  cleanup_leftovers

  local address_args=()
  local allocator_args=()
  local unknown_detail=0
  local weight_map=0
  local pool_registry=0
  if [ "$diag" = "1" ]; then
    address_args=(--with-address-log --address-trace-log "$addrs_log")
    allocator_args=(--allocator-log "$allocator_log")
    unknown_detail=1
    weight_map=1
    pool_registry=1
  fi

  set +e
  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --trace-log "$stats_log" \
    "${address_args[@]}" \
    "${allocator_args[@]}" \
    --address-log "$address_log" \
    --server-log "$server_log" \
    --bench-log "$bench_log" \
    --uvm-trace-min-bytes "$UVM_TRACE_MIN_BYTES" \
    --uvm-policy-enable 0 \
    --uvm-device-direct-enable 0 \
    --uvm-residency-enable 0 \
    --uvm-kv-runtime-enable 0 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-prefetch-trace-enable 0 \
    --uvm-weight-prefetch-hit-trace-enable 0 \
    --uvm-weight-prefetch-hit-driver-query-enable 0 \
    --uvm-weight-batch-executor-enable 0 \
    --uvm-weight-executor "$WEIGHT_EXECUTOR" \
    --uvm-weight-offload-enable 0 \
    --uvm-init-weight-offload-enable 0 \
    --uvm-init-weight-hot-prefetch-enable 0 \
    --uvm-pool-coordinator-enable 0 \
    --uvm-pool-coordinator-trace-enable 0 \
    --uvm-pool-registry-enable "$pool_registry" \
    --uvm-scratch-pool-enable 0 \
    --uvm-moe-routing-trace-enable 0 \
    --uvm-weight-map-enable "$weight_map" \
    --uvm-unknown-detail-enable "$unknown_detail" \
    --uvm-unknown-detail-min-bytes 4096 \
    --auto-gap-watch-enable 0 \
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
  return 0
}

run_native_case "native_full_gpu" 0
if [ "$RUN_NATIVE_CPU_OFFLOAD" = "1" ]; then
  run_native_case "native_cpu_offload" "$NATIVE_CPU_OFFLOAD_GB"
fi
run_uvm_case "uvm_fullish_gpu" 0
run_uvm_case "uvm_fullish_gpu_diag" 1

python3 - "$RUN_ROOT" <<'PY'
import csv
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

CASES = [
    ("native_full_gpu", "native"),
    ("native_cpu_offload", "native_offload"),
    ("uvm_fullish_gpu", "uvm_clean"),
    ("uvm_fullish_gpu_diag", "uvm_diag"),
]

BENCH_PATTERNS = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}

FAULT_PATTERNS = {
    "delta_faults": r"\bdelta_faults=([0-9]+)",
    "delta_after_dedup": r"\bdelta_after_dedup=([0-9]+)",
    "delta_kv_faults": r"\bdelta_kv_faults=([0-9]+)",
    "delta_kv_after_dedup": r"\bdelta_kv_after_dedup=([0-9]+)",
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def first_float(text: str, pattern: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def first_int(text: str, pattern: str):
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def status(path: Path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return (new - base) * 100.0 / base


rows = []
for case, kind in CASES:
    case_root = root / case
    if not case_root.exists():
        continue
    bench = read_text(case_root / "vllm_bench.log")
    server = read_text(case_root / "vllm_server.log")
    run_log = read_text(case_root / "run.log")

    row = {
        "case": case,
        "kind": kind,
        "exit_code": status(case_root / "status.txt"),
    }
    for key, pat in BENCH_PATTERNS.items():
        row[key] = first_float(bench, pat)

    load_match = re.search(
        r"Model loading took\s+([0-9.]+)\s+GiB memory and\s+([0-9.]+)\s+seconds",
        server,
    )
    row["model_load_gib"] = float(load_match.group(1)) if load_match else None
    row["model_load_s"] = float(load_match.group(2)) if load_match else None

    for key, pat in FAULT_PATTERNS.items():
        row[key] = first_int(run_log, pat)

    rows.append(row)

native = rows[0] if rows else {}
for row in rows:
    row["tpot_pct_vs_native"] = pct(row.get("mean_tpot_ms"), native.get("mean_tpot_ms"))
    row["tok_s_pct_vs_native"] = pct(row.get("output_tok_s"), native.get("output_tok_s"))

fields = [
    "case",
    "kind",
    "exit_code",
    "successful_requests",
    "failed_requests",
    "output_tok_s",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "mean_itl_ms",
    "total_tok_s",
    "model_load_gib",
    "model_load_s",
    "delta_faults",
    "delta_after_dedup",
    "delta_kv_faults",
    "delta_kv_after_dedup",
    "tpot_pct_vs_native",
    "tok_s_pct_vs_native",
]

csv_path = root / "native_vs_uvm_fullish_summary.csv"
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


md_path = root / "native_vs_uvm_fullish_summary.md"
lines = [
    "# Stage M Native Full-GPU vs UVM Full-ish GPU",
    "",
    f"- summary_csv: `{csv_path}`",
    "- note: native has no UVM fault counters by design.",
    "",
    "| case | exit | tok/s | TPOT ms | TTFT ms | faults dedup | model GiB | load s | TPOT vs native | tok/s vs native |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {exit} | {tok} | {tpot} | {ttft} | {faults} | {gib} | {load} | {tpot_pct} | {tok_pct} |".format(
            case=row["case"],
            exit=fmt(row.get("exit_code")),
            tok=fmt(row.get("output_tok_s")),
            tpot=fmt(row.get("mean_tpot_ms")),
            ttft=fmt(row.get("mean_ttft_ms")),
            faults=fmt(row.get("delta_after_dedup")),
            gib=fmt(row.get("model_load_gib")),
            load=fmt(row.get("model_load_s")),
            tpot_pct=fmt(row.get("tpot_pct_vs_native")),
            tok_pct=fmt(row.get("tok_s_pct_vs_native")),
        )
    )
lines.extend(
    [
        "",
        "Interpretation:",
        "",
        "- If `uvm_fullish_gpu` is much slower than `native_full_gpu`, the base managed-memory path is already costly before any hotset policy.",
        "- If `uvm_fullish_gpu_diag` is much slower than `uvm_fullish_gpu`, diagnostics are contaminating performance runs and should be split from measurement.",
        "- If `uvm_fullish_gpu` is close to native, focus next on decode-only fault windows and two-phase placement.",
    ]
)
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(md_path.read_text(encoding="utf-8"))
print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
PY

echo "Native vs UVM full-ish A/B finished."
echo "summary_csv: $RUN_ROOT/native_vs_uvm_fullish_summary.csv"
echo "summary_md: $RUN_ROOT/native_vs_uvm_fullish_summary.md"
