#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_native_cuda12_vs_cpu2_kv2_concurrency_sweep_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-${VLLM_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PORT_BASE="${PORT_BASE:-8920}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 2 4 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"

BASELINE_KV_CACHE_MEMORY_BYTES="${BASELINE_KV_CACHE_MEMORY_BYTES:-1288490189}"
UVA_KV_CACHE_MEMORY_BYTES="${UVA_KV_CACHE_MEMORY_BYTES:-2147483648}"
UVA_CPU_OFFLOAD_GB="${UVA_CPU_OFFLOAD_GB:-2}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_UVA="${RUN_UVA:-1}"
UVA_CASE_NAME="native_uva_cpu${UVA_CPU_OFFLOAD_GB//./p}_kv2g"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used
  for _ in $(seq 1 120); do
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

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local deadline=$((SECONDS + 1200))
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
}

case_kv_bytes() {
  local mode="$1"
  if [[ "${mode}" == "native_cuda_kv12g" ]]; then
    printf '%s\n' "${BASELINE_KV_CACHE_MEMORY_BYTES}"
  else
    printf '%s\n' "${UVA_KV_CACHE_MEMORY_BYTES}"
  fi
}

case_cpu_offload_gb() {
  local mode="$1"
  if [[ "${mode}" == "native_cuda_kv12g" ]]; then
    printf '0\n'
  else
    printf '%s\n' "${UVA_CPU_OFFLOAD_GB}"
  fi
}

run_case() {
  local mode="$1"
  local max_num_seqs="$2"
  local idx="$3"
  local port=$((PORT_BASE + idx))
  local kv_bytes
  local cpu_offload_gb
  kv_bytes="$(case_kv_bytes "${mode}")"
  cpu_offload_gb="$(case_cpu_offload_gb "${mode}")"

  local name="${mode}_s${max_num_seqs}"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  if [[ -f "${dir}/bench_status.txt" && -f "${dir}/bench_result.json" ]] &&
     [[ "$(cat "${dir}/bench_status.txt" 2>/dev/null)" == "0" ]]; then
    echo "=== ${name}: already completed, skipping ===" \
      | tee -a "${RUN_ROOT}/run.log"
    return 0
  fi

  rm -f "${dir}/server.log" "${dir}/bench.log" "${dir}/bench_status.txt" \
    "${dir}/bench_result.json" "${dir}/offload_manifest.jsonl"
  cleanup_port "${port}"
  wait_for_clean_gpu "${name}"

  echo "=== ${name}: max_num_seqs=${max_num_seqs} kv_bytes=${kv_bytes} cpu_offload_gb=${cpu_offload_gb} port=${port} ===" \
    | tee -a "${RUN_ROOT}/run.log"

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_CPU_OFFLOAD_BACKEND=uva \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/offload_manifest.jsonl" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
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
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${max_num_seqs}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${kv_bytes}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --cpu-offload-gb "${cpu_offload_gb}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server_failed" | tee "${dir}/bench_status.txt" | tee -a "${RUN_ROOT}/run.log"
    tail -n 160 "${dir}/server.log" >&2 || true
    stop_process_tree "${server_pid}"
    cleanup_port "${port}"
    wait_for_clean_gpu "after_failed_${name}" || true
    return 0
  fi

  local mem_pid=""
  local pcie_pid=""
  start_mem_monitor "${dir}/mem_monitor.csv"
  mem_pid=$!
  nvidia-smi dmon -s t -d 1 -o DT > "${dir}/dmon_pcie.csv" 2>&1 &
  pcie_pid=$!

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
      --port "${port}" \
      --endpoint /v1/completions \
      --ready-check-timeout-sec 600 \
      --save-result \
      --save-detailed \
      --result-dir "${dir}" \
      --result-filename bench_result.json
  ) > "${dir}/bench.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/bench_status.txt"

  stop_pid "${mem_pid}"
  stop_pid "${pcie_pid}"
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  wait_for_clean_gpu "after_${name}" || true
  return 0
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${PY}" - "${RUN_ROOT}" \
      "${BASELINE_KV_CACHE_MEMORY_BYTES}" \
      "${UVA_KV_CACHE_MEMORY_BYTES}" \
      "${UVA_CPU_OFFLOAD_GB}" <<'PY'
import json
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
baseline_kv_bytes = int(sys.argv[2])
uva_kv_bytes = int(sys.argv[3])
uva_cpu_offload_gb = float(sys.argv[4])

def parse_float_metric(text: str, name: str):
    m = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(m.group(1)) if m else None

def parse_int_metric(text: str, name: str):
    v = parse_float_metric(text, name)
    return int(v) if v is not None else None

def first_float(pattern: str, text: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None

def first_int(pattern: str, text: str):
    m = re.search(pattern, text)
    return int(m.group(1).replace(",", "")) if m else None

def max_int(pattern: str, text: str):
    values = [int(v.replace(",", "")) for v in re.findall(pattern, text)]
    return max(values) if values else None

def max_float(pattern: str, text: str):
    values = [float(v) for v in re.findall(pattern, text)]
    return max(values) if values else None

def parse_bench(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    result = {
        "successful_requests": parse_int_metric(text, "Successful requests"),
        "failed_requests": parse_int_metric(text, "Failed requests"),
        "benchmark_duration_s": parse_float_metric(text, "Benchmark duration (s)"),
        "request_throughput_req_s": parse_float_metric(text, "Request throughput (req/s)"),
        "output_tok_s": parse_float_metric(text, "Output token throughput (tok/s)"),
        "total_tok_s": parse_float_metric(text, "Total Token throughput (tok/s)"),
        "mean_ttft_ms": parse_float_metric(text, "Mean TTFT (ms)"),
        "median_ttft_ms": parse_float_metric(text, "Median TTFT (ms)"),
        "p99_ttft_ms": parse_float_metric(text, "P99 TTFT (ms)"),
        "mean_tpot_ms": parse_float_metric(text, "Mean TPOT (ms)"),
        "median_tpot_ms": parse_float_metric(text, "Median TPOT (ms)"),
        "p99_tpot_ms": parse_float_metric(text, "P99 TPOT (ms)"),
        "mean_itl_ms": parse_float_metric(text, "Mean ITL (ms)"),
        "median_itl_ms": parse_float_metric(text, "Median ITL (ms)"),
        "p99_itl_ms": parse_float_metric(text, "P99 ITL (ms)"),
    }
    json_path = path.parent / "bench_result.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            data = {}
        for key, out_key in (
            ("completed", "json_completed"),
            ("failed", "json_failed"),
            ("duration", "json_duration_s"),
            ("total_input_tokens", "json_total_input_tokens"),
            ("total_output_tokens", "json_total_output_tokens"),
            ("output_throughput", "json_output_tok_s"),
            ("mean_tpot_ms", "json_mean_tpot_ms"),
            ("p99_tpot_ms", "json_p99_tpot_ms"),
            ("mean_ttft_ms", "json_mean_ttft_ms"),
            ("p99_ttft_ms", "json_p99_ttft_ms"),
        ):
            if key in data:
                result[out_key] = data[key]
    return result

def parse_server(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "model_loading_gib": first_float(r"Model loading took ([0-9.]+) GiB memory", text),
        "initial_free_memory_gib": first_float(r"Initial free memory ([0-9.]+) GiB", text),
        "reserved_kv_gib": first_float(r"reserved ([0-9.]+) GiB memory for KV Cache", text),
        "gpu_kv_cache_tokens": first_int(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", text),
        "max_concurrency_for_4096": first_float(r"Maximum concurrency for 4,096 tokens per request:\s+([0-9.]+)x", text),
        "kv_allocator_default": "initialize_kv_cache end allocator=default" in text,
        "kv_allocator_uvm": "initialize_kv_cache end allocator=kv_cache_uvm" in text,
        "server_running_max": max_int(r"Running:\s+([0-9]+)\s+reqs", text),
        "server_waiting_max": max_int(r"Waiting:\s+([0-9]+)\s+reqs", text),
        "kv_usage_pct_max": max_float(r"GPU KV cache usage:\s+([0-9.]+)%", text),
        "has_cuda_oom": "out of memory" in text.lower() or "cudaerrormemoryallocation" in text.lower(),
        "has_illegal_memory_access": "illegal memory access" in text.lower(),
        "has_xid": "xid" in text.lower(),
        "has_uvm_fatal": "uvm fatal" in text.lower(),
        "has_mmu": "mmu" in text.lower(),
    }

def parse_manifest(path: Path):
    params = 0
    total = 0
    pinned = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") != "cpu_offload_parameter":
                continue
            params += 1
            total += int(item.get("bytes") or 0)
            if item.get("pinned") or item.get("cpu_data_pinned") or item.get("pin_memory"):
                pinned += 1
    return {
        "offload_manifest_params": params,
        "offload_manifest_pinned_params": pinned,
        "offload_manifest_bytes": total,
        "offload_manifest_gib": total / (1024 ** 3),
    }

def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)

def parse_dmon_pcie(path: Path):
    rx = []
    tx = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < 5:
                continue
            try:
                rx.append(float(parts[3]))
                tx.append(float(parts[4]))
            except ValueError:
                continue
    return {
        "samples": len(rx),
        "rx_mean_MBps": statistics.mean(rx) if rx else None,
        "rx_p95_MBps": quantile(rx, 0.95),
        "rx_max_MBps": max(rx) if rx else None,
        "tx_mean_MBps": statistics.mean(tx) if tx else None,
        "tx_max_MBps": max(tx) if tx else None,
    }

def parse_mem(path: Path):
    used = []
    free = []
    util = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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

cases = {}
ordered = []
for path in sorted(root.iterdir()):
    if not path.is_dir():
        continue
    m = re.fullmatch(r"(native_cuda_kv12g|native_uva_cpu[0-9p]+_kv2g)_s([0-9]+)", path.name)
    if not m:
        continue
    mode = m.group(1)
    max_num_seqs = int(m.group(2))
    status_path = path / "bench_status.txt"
    status = status_path.read_text(encoding="utf-8", errors="replace").strip() if status_path.exists() else "missing"
    case = {
        "mode": mode,
        "max_num_seqs": max_num_seqs,
        "bench_status": status,
        "cpu_offload_gb": 0 if mode == "native_cuda_kv12g" else uva_cpu_offload_gb,
        "kv_cache_memory_bytes": baseline_kv_bytes if mode == "native_cuda_kv12g" else uva_kv_bytes,
        "kv_cache_memory_gib": (baseline_kv_bytes if mode == "native_cuda_kv12g" else uva_kv_bytes) / (1024 ** 3),
    }
    case.update(parse_bench(path / "bench.log"))
    case.update(parse_server(path / "server.log"))
    case.update(parse_manifest(path / "offload_manifest.jsonl"))
    case["dmon_pcie"] = parse_dmon_pcie(path / "dmon_pcie.csv")
    case["mem_monitor"] = parse_mem(path / "mem_monitor.csv")
    key = f"{mode}_s{max_num_seqs}"
    cases[key] = case
    ordered.append(case)

ordered.sort(key=lambda item: (item["max_num_seqs"], item["mode"]))

pairs = []
uva_case_name = f"native_uva_cpu{str(uva_cpu_offload_gb).replace('.', 'p').rstrip('0').rstrip('p')}_kv2g"
for max_num_seqs in sorted({case["max_num_seqs"] for case in ordered}):
    base = cases.get(f"native_cuda_kv12g_s{max_num_seqs}")
    uva = cases.get(f"{uva_case_name}_s{max_num_seqs}")
    pair = {
        "max_num_seqs": max_num_seqs,
        "baseline": base,
        "uva": uva,
        uva_case_name: uva,
    }
    if base and uva:
        b_out = base.get("output_tok_s")
        u_out = uva.get("output_tok_s")
        b_tpot = base.get("mean_tpot_ms")
        u_tpot = uva.get("mean_tpot_ms")
        b_ttft = base.get("mean_ttft_ms")
        u_ttft = uva.get("mean_ttft_ms")
        pair["uva_vs_baseline_output_tok_s_pct"] = (
            (u_out - b_out) / b_out * 100.0
            if b_out not in (None, 0) and u_out is not None else None
        )
        pair["uva_vs_baseline_mean_tpot_pct"] = (
            (u_tpot - b_tpot) / b_tpot * 100.0
            if b_tpot not in (None, 0) and u_tpot is not None else None
        )
        pair["uva_vs_baseline_mean_ttft_pct"] = (
            (u_ttft - b_ttft) / b_ttft * 100.0
            if b_ttft not in (None, 0) and u_ttft is not None else None
        )
    pairs.append(pair)

summary = {
    "run_root": str(root),
    "config": {
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "dataset": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "num_prompts": 16,
        "sharegpt_output_len": 512,
        "request_rate": "inf",
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.97,
        "baseline": {
            "use_uvm": False,
            "cpu_offload_gb": 0,
            "kv_cache_memory_bytes": baseline_kv_bytes,
            "kv_cache_memory_gib": baseline_kv_bytes / (1024 ** 3),
        },
        uva_case_name: {
            "use_uvm": False,
            "cpu_offload_backend": "uva",
            "cpu_offload_gb": uva_cpu_offload_gb,
            "kv_cache_memory_bytes": uva_kv_bytes,
            "kv_cache_memory_gib": uva_kv_bytes / (1024 ** 3),
        },
    },
    "cases": cases,
    "ordered": ordered,
    "pairs": pairs,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

echo "run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
echo "baseline: native cuda KV=${BASELINE_KV_CACHE_MEMORY_BYTES} bytes, no offload" \
  | tee -a "${RUN_ROOT}/run.log"
echo "uva: native cpu_offload_gb=${UVA_CPU_OFFLOAD_GB}, cuda KV=${UVA_KV_CACHE_MEMORY_BYTES} bytes" \
  | tee -a "${RUN_ROOT}/run.log"

idx=0
for max_num_seqs in ${MAX_NUM_SEQS_LIST}; do
  if [[ "${RUN_BASELINE}" == "1" ]]; then
    idx=$((idx + 1))
    run_case "native_cuda_kv12g" "${max_num_seqs}" "${idx}" || true
    write_summary > "${RUN_ROOT}/summary.latest.json" || true
  fi

  if [[ "${RUN_UVA}" == "1" ]]; then
    idx=$((idx + 1))
    run_case "${UVA_CASE_NAME}" "${max_num_seqs}" "${idx}" || true
    write_summary > "${RUN_ROOT}/summary.latest.json" || true
  fi
done

write_summary | tee "${RUN_ROOT}/summary.json"
echo "summary: ${RUN_ROOT}/summary.json"
