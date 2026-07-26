#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_native_cpu10_concurrency_pcie_sweep_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-8870}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 4 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-10}"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used
  for _ in $(seq 1 60); do
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

stop_process_group() {
  local pid="$1"
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
  local pid="$1"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    sleep 1
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

run_case() {
  local max_num_seqs="$1"
  local idx="$2"
  local port=$((PORT_BASE + idx))
  local name="cpu10_s${max_num_seqs}"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  cleanup_port "${port}"
  wait_for_clean_gpu "${name}"

  echo "=== ${name}: max_num_seqs=${max_num_seqs} port=${port} ===" | tee -a "${RUN_ROOT}/run.log"

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
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${max_num_seqs}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --cpu-offload-gb "${CPU_OFFLOAD_GB}" \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server failed for ${name}" | tee -a "${RUN_ROOT}/run.log"
    stop_process_group "${server_pid}"
    printf '%s\n' "server_failed" > "${dir}/bench_status.txt"
    return 1
  fi

  nvidia-smi dmon -s t -d 1 -o DT > "${dir}/dmon_pcie.csv" 2>&1 &
  local dmon_pcie_pid=$!
  nvidia-smi dmon -i 0 -d 1 --gpm-metrics 20,21,160,161,255,256 -o DT \
    > "${dir}/dmon_gpm.csv" 2>&1 &
  local dmon_gpm_pid=$!

  set +e
  (
    cd "${VLLM_ROOT}"
    timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" -m vllm.entrypoints.cli.main bench serve \
      --backend openai \
      --model "${MODEL}" \
      --dataset-name sharegpt \
      --dataset-path "${DATASET}" \
      --num-prompts "${NUM_PROMPTS}" \
      --sharegpt-output-len "${OUTPUT_LEN}" \
      --seed 42 \
      --request-rate "${REQUEST_RATE}" \
      --host 127.0.0.1 \
      --port "${port}" \
      --endpoint /v1/completions \
      --ready-check-timeout-sec 600
  ) > "${dir}/bench.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/bench_status.txt"

  stop_pid "${dmon_pcie_pid}"
  stop_pid "${dmon_gpm_pid}"
  stop_process_group "${server_pid}"
  cleanup_port "${port}"
  wait_for_clean_gpu "after_${name}" || true
  return 0
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RUN_ROOT}" <<'PY'
import json
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

def parse_float_metric(text: str, name: str):
    m = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(m.group(1)) if m else None

def parse_int_metric(text: str, name: str):
    v = parse_float_metric(text, name)
    return int(v) if v is not None else None

def parse_bench(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "successful_requests": parse_int_metric(text, "Successful requests"),
        "failed_requests": parse_int_metric(text, "Failed requests"),
        "benchmark_duration_s": parse_float_metric(text, "Benchmark duration (s)"),
        "request_throughput_req_s": parse_float_metric(text, "Request throughput (req/s)"),
        "output_tok_s": parse_float_metric(text, "Output token throughput (tok/s)"),
        "peak_output_tok_s": parse_float_metric(text, "Peak output token throughput (tok/s)"),
        "peak_concurrent_requests": parse_float_metric(text, "Peak concurrent requests"),
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

def parse_server(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    run_values = [int(x) for x in re.findall(r"Running:\s+([0-9]+)\s+reqs", text)]
    wait_values = [int(x) for x in re.findall(r"Waiting:\s+([0-9]+)\s+reqs", text)]
    kv = None
    m = re.search(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", text)
    if m:
        kv = int(m.group(1).replace(",", ""))
    avail = None
    m = re.search(r"Available KV cache memory:\s+([0-9.]+)\s+GiB", text)
    if m:
        avail = float(m.group(1))
    load_gib = None
    m = re.search(r"Model loading took\s+([0-9.]+)\s+GiB memory", text)
    if m:
        load_gib = float(m.group(1))
    return {
        "server_running_max": max(run_values) if run_values else None,
        "server_waiting_max": max(wait_values) if wait_values else None,
        "server_running_samples": run_values,
        "gpu_kv_cache_tokens": kv,
        "available_kv_cache_gib": avail,
        "model_load_gib": load_gib,
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
        "manifest_params": params,
        "manifest_pinned_params": pinned,
        "manifest_bytes": total,
        "manifest_gib": total / (1024 ** 3),
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
    nz_rx = [x for x in rx if x > 0]
    return {
        "samples": len(rx),
        "rx_mean_MBps": statistics.mean(rx) if rx else None,
        "rx_nonzero_mean_MBps": statistics.mean(nz_rx) if nz_rx else None,
        "rx_median_MBps": statistics.median(rx) if rx else None,
        "rx_p95_MBps": quantile(rx, 0.95),
        "rx_max_MBps": max(rx) if rx else None,
        "rx_sum_MB_1s_samples": sum(rx),
        "tx_mean_MBps": statistics.mean(tx) if tx else None,
        "tx_max_MBps": max(tx) if tx else None,
        "tx_sum_MB_1s_samples": sum(tx),
    }

def parse_dmon_gpm(path: Path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < 20:
                continue
            try:
                rows.append({
                    "sm": float(parts[6]),
                    "mem": float(parts[7]),
                    "pcitx": float(parts[14]),
                    "pcirx": float(parts[15]),
                    "hmhit": float(parts[16]),
                    "hmmiss": float(parts[17]),
                    "pcitxraw": float(parts[18]),
                    "pcirxraw": float(parts[19]),
                })
            except (ValueError, IndexError):
                continue
    if not rows:
        return {"samples": 0}
    raw_rx = [r["pcirxraw"] for r in rows]
    raw_tx = [r["pcitxraw"] for r in rows]
    return {
        "samples": len(rows),
        "sm_mean_pct": statistics.mean(r["sm"] for r in rows),
        "sm_max_pct": max(r["sm"] for r in rows),
        "mem_mean_pct": statistics.mean(r["mem"] for r in rows),
        "mem_max_pct": max(r["mem"] for r in rows),
        "pcirx_mean_MiBps": statistics.mean(r["pcirx"] for r in rows),
        "pcirx_max_MiBps": max(r["pcirx"] for r in rows),
        "pcirxraw_delta": max(raw_rx) - min(raw_rx),
        "pcirxraw_delta_gib": (max(raw_rx) - min(raw_rx)) / (1024 ** 3),
        "pcitx_mean_MiBps": statistics.mean(r["pcitx"] for r in rows),
        "pcitx_max_MiBps": max(r["pcitx"] for r in rows),
        "pcitxraw_delta": max(raw_tx) - min(raw_tx),
        "pcitxraw_delta_gib": (max(raw_tx) - min(raw_tx)) / (1024 ** 3),
        "hmmiss_mean_pct": statistics.mean(r["hmmiss"] for r in rows),
    }

cases = {}
for path in sorted(root.glob("cpu10_s*")):
    if not path.is_dir():
        continue
    m = re.fullmatch(r"cpu10_s([0-9]+)", path.name)
    if not m:
        continue
    status_path = path / "bench_status.txt"
    status = status_path.read_text(encoding="utf-8", errors="replace").strip() if status_path.exists() else "missing"
    case = {
        "max_num_seqs": int(m.group(1)),
        "bench_status": status,
    }
    case.update(parse_bench(path / "bench.log"))
    case.update(parse_server(path / "server.log"))
    case.update(parse_manifest(path / "offload_manifest.jsonl"))
    case["dmon_pcie"] = parse_dmon_pcie(path / "dmon_pcie.csv")
    case["dmon_gpm"] = parse_dmon_gpm(path / "dmon_gpm.csv")
    cases[path.name] = case

ordered = sorted(cases.values(), key=lambda x: x["max_num_seqs"])
baseline = ordered[0] if ordered else None
for case in ordered:
    if baseline and baseline.get("output_tok_s"):
        case["delta_vs_s1_output_tok_s_pct"] = (
            (case.get("output_tok_s") - baseline.get("output_tok_s"))
            / baseline.get("output_tok_s") * 100.0
            if case.get("output_tok_s") is not None else None
        )
    if baseline and baseline.get("mean_tpot_ms"):
        case["delta_vs_s1_mean_tpot_pct"] = (
            (case.get("mean_tpot_ms") - baseline.get("mean_tpot_ms"))
            / baseline.get("mean_tpot_ms") * 100.0
            if case.get("mean_tpot_ms") is not None else None
        )

summary = {
    "config": {
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "num_prompts": 16,
        "sharegpt_output_len": 512,
        "request_rate": "inf",
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
        "gpu_memory_utilization": 0.97,
        "cpu_offload_gb": 10,
    },
    "cases": cases,
    "ordered": ordered,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

idx=0
for max_num_seqs in ${MAX_NUM_SEQS_LIST}; do
  idx=$((idx + 1))
  run_case "${max_num_seqs}" "${idx}" || true
  write_summary | tee "${RUN_ROOT}/summary.latest.json"
done

write_summary | tee "${RUN_ROOT}/summary.json"
echo "summary: ${RUN_ROOT}/summary.json"
