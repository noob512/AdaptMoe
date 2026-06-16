#!/usr/bin/env bash
# Reproduce the README experiments that do not require a custom nvidia_uvm
# module and do not use eBPF:
#   1) vLLM native CPU offload
#   2) vLLM UVM allocator baseline

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/results/readme_no_ebpf_repro_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-100}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-5}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
CPU_PORT="${CPU_PORT:-8120}"
UVM_PORT="${UVM_PORT:-8121}"
CPU_MAX_NUM_SEQS="${CPU_MAX_NUM_SEQS:-100}"
UVM_MAX_NUM_SEQS="${UVM_MAX_NUM_SEQS:-16}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p "$RUN_ROOT"

cleanup_gpu() {
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

write_metadata() {
  local case_root="$1"
  local case_name="$2"
  local mode="$3"
  {
    echo "case=$case_name"
    echo "mode=$mode"
    echo "model=$MODEL"
    echo "dataset_path=$DATASET_PATH"
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "seed=$SEED"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "cpu_max_num_seqs=$CPU_MAX_NUM_SEQS"
    echo "uvm_max_num_seqs=$UVM_MAX_NUM_SEQS"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    modinfo nvidia_uvm 2>/dev/null | sed -n '1,20p' || true
  } > "$case_root/metadata.txt"
}

run_case() {
  local case_name="$1"
  local mode="$2"
  local port="$3"
  shift 3

  local case_root="$RUN_ROOT/$case_name"
  local server_log="$case_root/server.log"
  local bench_log="$case_root/bench.log"
  local run_log="$case_root/run.log"
  local status=0
  local server_pid=""

  mkdir -p "$case_root"
  : > "$server_log"
  : > "$bench_log"
  : > "$run_log"
  write_metadata "$case_root" "$case_name" "$mode"

  echo "==========================================================="
  echo "Run $case_name ($mode)"
  echo "logs=$case_root"
  echo "==========================================================="

  cleanup_gpu
  sleep 2

  if [[ "$mode" == "uvm" ]]; then
    (
      cd "$SCRIPT_DIR" || exit 1
      VLLM_USE_UVM=1 \
      VLLM_UVM_LOG_FILE="$case_root/vllm_uvm_allocations.log" \
      uv run vllm serve "$MODEL" "$@" --port "$port" >"$server_log" 2>&1
    ) &
  else
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run vllm serve "$MODEL" "$@" --port "$port" >"$server_log" 2>&1
    ) &
  fi
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    set +e
    (
      cd "$SCRIPT_DIR" || exit 1
      uv run vllm bench serve \
        --model "$MODEL" \
        --dataset-name sharegpt \
        --num-prompts "$PROMPTS" \
        --dataset-path "$DATASET_PATH" \
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

  echo "$status" > "$case_root/status.txt"
  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 3
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_gpu
}

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
patterns = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "peak_output_tok_s": r"Peak output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}

def read(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

rows = []
for name in ("cpu_offload", "uvm_baseline"):
    case = root / name
    text = read(case / "bench.log")
    row = {"case": name}
    try:
        row["exit_code"] = int((case / "status.txt").read_text().strip())
    except Exception:
        row["exit_code"] = None
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        row[key] = float(match.group(1)) if match else None
    rows.append(row)

(root / "summary.json").write_text(
    json.dumps({"run_root": str(root), "rows": rows}, indent=2, sort_keys=True),
    encoding="utf-8",
)

lines = [
    "# README no-eBPF/no-custom-module reproduction",
    "",
    "| case | ok/failed | tok/s | TPOT ms | TTFT ms | duration s | peak tok/s | exit |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    ok = row.get("successful_requests")
    fail = row.get("failed_requests")
    lines.append(
        f"| {row['case']} | {ok if ok is not None else 'n/a'}/{fail if fail is not None else 'n/a'} "
        f"| {row.get('output_tok_s') if row.get('output_tok_s') is not None else 'n/a'} "
        f"| {row.get('mean_tpot_ms') if row.get('mean_tpot_ms') is not None else 'n/a'} "
        f"| {row.get('mean_ttft_ms') if row.get('mean_ttft_ms') is not None else 'n/a'} "
        f"| {row.get('duration_s') if row.get('duration_s') is not None else 'n/a'} "
        f"| {row.get('peak_output_tok_s') if row.get('peak_output_tok_s') is not None else 'n/a'} "
        f"| {row.get('exit_code')} |"
    )

(root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((root / "summary.md").read_text(encoding="utf-8"))
PY
}

echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"

run_case cpu_offload native "$CPU_PORT" \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$CPU_MAX_NUM_SEQS" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB"

run_case uvm_baseline uvm "$UVM_PORT" \
  --enforce-eager \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$UVM_MAX_NUM_SEQS"

summarize
echo "summary_md=$RUN_ROOT/summary.md"
