#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_uvm_kv12g_s16_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-${VLLM_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-8897}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1288490189}"

CASE_DIR="${RUN_ROOT}/uvm_expert_kv12g_s16"
mkdir -p "${CASE_DIR}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used
  for _ in $(seq 1 90); do
    used="$(gpu_used_mib || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU is not clean before ${label}: ${used} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

stop_process_tree() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
    sleep 5
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    pkill -KILL -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

wait_for_server() {
  local server_pid="$1"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 2
}

start_mem_monitor() {
  local out="$1"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
      sleep 1
    done
  ) > "${out}" 2>&1 &
  echo "$!"
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${PY}" - "${RUN_ROOT}" "${CASE_DIR}" "${KV_CACHE_MEMORY_BYTES}" <<'PY'
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
case_dir = Path(sys.argv[2])
kv_bytes = int(sys.argv[3])
server_log = (case_dir / "server.log").read_text(errors="replace") if (case_dir / "server.log").exists() else ""
bench_log = (case_dir / "bench.log").read_text(errors="replace") if (case_dir / "bench.log").exists() else ""
status = (case_dir / "status.txt").read_text(errors="replace").strip() if (case_dir / "status.txt").exists() else "missing"

def first_float(pattern: str, text: str):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None

def first_int(pattern: str, text: str):
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else None

def metric_float(name: str):
    match = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", bench_log)
    return float(match.group(1)) if match else None

def metric_int(name: str):
    value = metric_float(name)
    return int(value) if value is not None else None

def max_int(pattern: str, text: str):
    values = [int(v.replace(",", "")) for v in re.findall(pattern, text)]
    return max(values) if values else None

def max_float(pattern: str, text: str):
    values = [float(v) for v in re.findall(pattern, text)]
    return max(values) if values else None

def parse_mem_monitor(path: Path):
    used, free, util = [], [], []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            try:
                used.append(int(parts[1]))
                free.append(int(parts[2]))
                util.append(int(parts[3]))
            except ValueError:
                continue
    return {
        "samples": len(used),
        "memory_used_mib_max": max(used) if used else None,
        "memory_free_mib_min": min(free) if free else None,
        "gpu_utilization_pct_max": max(util) if util else None,
    }

bench_json_metrics = {}
bench_json_path = case_dir / "bench_result.json"
if bench_json_path.exists():
    try:
        data = json.loads(bench_json_path.read_text(errors="replace"))
    except json.JSONDecodeError:
        data = {}
    for key in (
        "completed",
        "failed",
        "duration",
        "total_input_tokens",
        "total_output_tokens",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "max_concurrent_requests",
        "max_output_tokens_per_s",
    ):
        if key in data:
            bench_json_metrics[key] = data[key]

summary = {
    "run_root": str(run_root),
    "case": "uvm_expert_kv12g_s16",
    "status": status,
    "config": {
        "use_uvm": False,
        "expert_weights_managed": True,
        "kv_cache_managed": True,
        "cpu_offload_gb": 0,
        "kv_cache_memory_bytes": kv_bytes,
        "kv_cache_memory_gib": kv_bytes / 1024**3,
        "num_prompts": 16,
        "sharegpt_output_len": 512,
        "max_num_seqs": 16,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "request_rate": "inf",
    },
    "server": {
        "model_loading_gib": first_float(r"Model loading took ([0-9.]+) GiB memory", server_log),
        "initial_free_memory_gib": first_float(r"Initial free memory ([0-9.]+) GiB", server_log),
        "reserved_kv_gib": first_float(r"reserved ([0-9.]+) GiB memory for KV Cache", server_log),
        "expert_managed_bytes": first_int(r"managed_bytes=([0-9]+)", server_log),
        "expert_gpu_bytes_after_load": first_int(r"gpu_bytes=([0-9]+)", server_log),
        "expert_cpu_bytes_after_load": first_int(r"cpu_bytes=([0-9]+)", server_log),
        "kv_tokens": first_int(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", server_log),
        "max_concurrency_for_4096": first_float(r"Maximum concurrency for 4,096 tokens per request:\s+([0-9.]+)x", server_log),
        "kv_allocator_default": "initialize_kv_cache end allocator=default" in server_log,
        "kv_allocator_uvm": "initialize_kv_cache end allocator=kv_cache_uvm" in server_log,
        "running_max": max_int(r"Running:\s+([0-9]+)\s+reqs", server_log),
        "waiting_max": max_int(r"Waiting:\s+([0-9]+)\s+reqs", server_log),
        "kv_usage_pct_max": max_float(r"GPU KV cache usage:\s+([0-9.]+)%", server_log),
        "has_cuda_oom": "out of memory" in server_log.lower() or "cudaerrormemoryallocation" in server_log.lower(),
        "has_illegal_memory_access": "illegal memory access" in server_log.lower(),
        "has_xid": "xid" in server_log.lower(),
        "has_uvm_fatal": "uvm fatal" in server_log.lower(),
        "has_mmu": "mmu" in server_log.lower(),
    },
    "bench": {
        "successful_requests": metric_int("Successful requests"),
        "failed_requests": metric_int("Failed requests"),
        "benchmark_duration_s": metric_float("Benchmark duration (s)"),
        "request_throughput_req_s": metric_float("Request throughput (req/s)"),
        "output_tok_s": metric_float("Output token throughput (tok/s)"),
        "total_tok_s": metric_float("Total Token throughput (tok/s)"),
        "mean_ttft_ms": metric_float("Mean TTFT (ms)"),
        "median_ttft_ms": metric_float("Median TTFT (ms)"),
        "p99_ttft_ms": metric_float("P99 TTFT (ms)"),
        "mean_tpot_ms": metric_float("Mean TPOT (ms)"),
        "median_tpot_ms": metric_float("Median TPOT (ms)"),
        "p99_tpot_ms": metric_float("P99 TPOT (ms)"),
        "mean_itl_ms": metric_float("Mean ITL (ms)"),
        "median_itl_ms": metric_float("Median ITL (ms)"),
        "p99_itl_ms": metric_float("P99 ITL (ms)"),
        "json_metrics": bench_json_metrics,
    },
    "nvidia_smi_monitor": parse_mem_monitor(case_dir / "nvidia_smi_memory.csv"),
}

(run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

cleanup() {
  set +e
  if [[ -n "${mem_pid:-}" ]]; then
    stop_pid "${mem_pid}"
  fi
  if [[ -n "${server_pid:-}" ]]; then
    stop_process_tree "${server_pid}"
  fi
}
trap cleanup EXIT

echo "run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
echo "case=uvm_expert_kv12g_s16 kv_bytes=${KV_CACHE_MEMORY_BYTES} port=${PORT}" \
  | tee -a "${RUN_ROOT}/run.log"

wait_for_clean_gpu "before_uvm_expert_kv12g_s16"

(
  cd "${VLLM_ROOT}"
  env \
    VLLM_USE_UVM=0 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_KV_CACHE_MANAGED=1 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
    VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0 \
    VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
    VLLM_UVM_ADDRESS_LOG_ENABLE=0 \
    VLLM_UVM_KV_INIT_FAULT_PROBE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
    VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
    "${PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
      --port "${PORT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --seed 0 \
      --enforce-eager \
      --disable-log-requests
) > "${CASE_DIR}/server.log" 2>&1 &
server_pid=$!
echo "${server_pid}" > "${CASE_DIR}/server.pid"

if ! wait_for_server "${server_pid}"; then
  echo "server_failed" | tee "${CASE_DIR}/status.txt"
  tail -160 "${CASE_DIR}/server.log" >&2 || true
  write_summary
  echo "summary=${RUN_ROOT}/summary.json"
  exit 0
fi

mem_pid="$(start_mem_monitor "${CASE_DIR}/nvidia_smi_memory.csv")"

set +e
(
  cd "${VLLM_ROOT}"
  timeout "${TIMEOUT_SEC}s" "${PY}" -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --model "${MODEL}" \
    --dataset-name sharegpt \
    --dataset-path "${DATASET}" \
    --num-prompts "${NUM_PROMPTS}" \
    --sharegpt-output-len "${OUTPUT_LEN}" \
    --seed 42 \
    --temperature 0 \
    --ignore-eos \
    --request-rate "${REQUEST_RATE}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --endpoint /v1/completions \
    --ready-check-timeout-sec 600 \
    --save-result \
    --save-detailed \
    --result-dir "${CASE_DIR}" \
    --result-filename bench_result.json
) > "${CASE_DIR}/bench.log" 2>&1
bench_rc=$?
set -e
echo "${bench_rc}" | tee "${CASE_DIR}/status.txt"

stop_pid "${mem_pid}"
unset mem_pid
stop_process_tree "${server_pid}"
unset server_pid
wait_for_clean_gpu "after_uvm_expert_kv12g_s16" || true
write_summary
echo "summary=${RUN_ROOT}/summary.json"
