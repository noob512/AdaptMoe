#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_expert_uvm_cuda_kv6g_s16_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-${VLLM_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT="${PORT:-8895}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-6442450944}"

mkdir -p "${RUN_ROOT}"

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

cleanup_port() {
  local port="$1"
  pkill -f "vllm.entrypoints.cli.main serve.*--port ${port}" 2>/dev/null || true
  pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
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
  local port="$1"
  local server_pid="$2"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
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
    "${PY}" - "${RUN_ROOT}" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_dir = root / "expert_uvm_cuda_kv6g_s16"
server_log = (case_dir / "server.log").read_text(errors="replace") if (case_dir / "server.log").exists() else ""
bench_log = (case_dir / "bench.log").read_text(errors="replace") if (case_dir / "bench.log").exists() else ""
status = (case_dir / "status.txt").read_text(errors="replace").strip() if (case_dir / "status.txt").exists() else "missing"

def fmetric(text: str, name: str):
    m = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(m.group(1)) if m else None

def imetric(text: str, name: str):
    value = fmetric(text, name)
    return int(value) if value is not None else None

def last_float(pattern: str, text: str):
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None

def last_int(pattern: str, text: str):
    value = last_float(pattern, text)
    return int(value) if value is not None else None

def max_int(pattern: str, text: str):
    values = [int(v.replace(",", "")) for v in re.findall(pattern, text)]
    return max(values) if values else None

def parse_mem_monitor(path: Path):
    result = {
        "samples": 0,
        "memory_used_mib_max": None,
        "memory_free_mib_min": None,
        "gpu_utilization_max_pct": None,
    }
    if not path.exists():
        return result
    used = []
    free = []
    util = []
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
    result["samples"] = len(used)
    result["memory_used_mib_max"] = max(used) if used else None
    result["memory_free_mib_min"] = min(free) if free else None
    result["gpu_utilization_max_pct"] = max(util) if util else None
    return result

bench_json_path = case_dir / "bench_result.json"
bench_json = None
if bench_json_path.exists():
    try:
        bench_json = json.loads(bench_json_path.read_text(errors="replace"))
    except json.JSONDecodeError:
        bench_json = None

summary = {
    "run_root": str(root),
    "case": "expert_uvm_cuda_kv6g_s16",
    "status": status,
    "config": {
        "expert_weights_managed": True,
        "kv_cache_managed": False,
        "kv_cache_memory_bytes": 6442450944,
        "num_prompts": 16,
        "output_len": 512,
        "request_rate": "inf",
        "max_num_seqs": 16,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
    },
    "bench": {
        "successful_requests": imetric(bench_log, "Successful requests"),
        "failed_requests": imetric(bench_log, "Failed requests"),
        "benchmark_duration_s": fmetric(bench_log, "Benchmark duration \\(s\\)") or fmetric(bench_log, "Benchmark duration (s)"),
        "request_throughput_req_s": fmetric(bench_log, "Request throughput \\(req/s\\)") or fmetric(bench_log, "Request throughput (req/s)"),
        "output_tok_s": fmetric(bench_log, "Output token throughput \\(tok/s\\)") or fmetric(bench_log, "Output token throughput (tok/s)"),
        "total_tok_s": fmetric(bench_log, "Total Token throughput \\(tok/s\\)") or fmetric(bench_log, "Total Token throughput (tok/s)"),
        "mean_ttft_ms": fmetric(bench_log, "Mean TTFT \\(ms\\)") or fmetric(bench_log, "Mean TTFT (ms)"),
        "median_ttft_ms": fmetric(bench_log, "Median TTFT \\(ms\\)") or fmetric(bench_log, "Median TTFT (ms)"),
        "p99_ttft_ms": fmetric(bench_log, "P99 TTFT \\(ms\\)") or fmetric(bench_log, "P99 TTFT (ms)"),
        "mean_tpot_ms": fmetric(bench_log, "Mean TPOT \\(ms\\)") or fmetric(bench_log, "Mean TPOT (ms)"),
        "median_tpot_ms": fmetric(bench_log, "Median TPOT \\(ms\\)") or fmetric(bench_log, "Median TPOT (ms)"),
        "p99_tpot_ms": fmetric(bench_log, "P99 TPOT \\(ms\\)") or fmetric(bench_log, "P99 TPOT (ms)"),
        "json": bench_json,
    },
    "server": {
        "model_load_gib": last_float(r"Model loading took ([0-9.]+) GiB memory", server_log),
        "initial_free_memory_gib": last_float(r"Initial free memory ([0-9.]+) GiB", server_log),
        "reserved_kv_gib": last_float(r"reserved ([0-9.]+) GiB memory for KV Cache", server_log),
        "kv_tokens": max_int(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", server_log),
        "max_concurrency_for_model_len": last_float(r"Maximum concurrency for [0-9,]+ tokens per request:\s+([0-9.]+)x", server_log),
        "kv_allocator": "kv_cache_uvm" if "allocator=kv_cache_uvm" in server_log else (
            "default_cuda" if "allocator=default" in server_log else None
        ),
        "kv_cache_uvm_marker_present": "allocator=kv_cache_uvm" in server_log,
        "expert_managed_marker_present": "UVM expert-only managed prepare" in server_log,
        "running_max": max_int(r"Running:\s+([0-9]+)\s+reqs", server_log),
        "waiting_max": max_int(r"Waiting:\s+([0-9]+)\s+reqs", server_log),
        "kv_usage_pct_max": last_float(r"GPU KV cache usage:\s+([0-9.]+)%", server_log),
        "has_cuda_oom": "CUDA out of memory" in server_log or "cudaErrorMemoryAllocation" in server_log,
        "has_illegal_memory_access": "illegal memory access" in server_log.lower(),
        "has_mmu_fault": "MMU" in server_log or "FAULT_PDE" in server_log,
        "has_uvm_fatal": "UVM fatal" in server_log,
        "has_xid": "Xid" in server_log,
    },
    "memory_monitor": parse_mem_monitor(case_dir / "nvidia_smi_memory.csv"),
}

(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

case_dir="${RUN_ROOT}/expert_uvm_cuda_kv6g_s16"
mkdir -p "${case_dir}"
cleanup_port "${PORT}"
wait_for_clean_gpu "before_run"

echo "run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
echo "case=expert_uvm_cuda_kv6g_s16 port=${PORT}" | tee -a "${RUN_ROOT}/run.log"

(
  cd "${VLLM_ROOT}"
  env \
    VLLM_USE_UVM=0 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_KV_CACHE_MANAGED=0 \
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
) > "${case_dir}/server.log" 2>&1 &
server_pid=$!
echo "${server_pid}" > "${case_dir}/server.pid"

if ! wait_for_server "${PORT}" "${server_pid}"; then
  echo "server_failed" | tee "${case_dir}/status.txt"
  tail -160 "${case_dir}/server.log" >&2 || true
  stop_process_tree "${server_pid}"
  cleanup_port "${PORT}"
  wait_for_clean_gpu "after_server_failed" || true
  write_summary
  echo "summary=${RUN_ROOT}/summary.json"
  exit 0
fi

mem_pid="$(start_mem_monitor "${case_dir}/nvidia_smi_memory.csv")"

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
    --result-dir "${case_dir}" \
    --result-filename bench_result.json
) > "${case_dir}/bench.log" 2>&1
bench_rc=$?
set -e
echo "${bench_rc}" | tee "${case_dir}/status.txt"

stop_pid "${mem_pid}"
stop_process_tree "${server_pid}"
cleanup_port "${PORT}"
wait_for_clean_gpu "after_run" || true
write_summary
echo "summary=${RUN_ROOT}/summary.json"
