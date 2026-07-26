#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_remote_read_reference_cases_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-8940}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

MODES="${MODES:-managed_no_offload native_uva}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-2}"

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

run_case() {
  local mode="$1"
  local idx="$2"
  local port=$((PORT_BASE + idx))
  local dir="${RUN_ROOT}/${mode}_s${MAX_NUM_SEQS}"
  mkdir -p "${dir}"
  rm -f "${dir}/server.log" "${dir}/server.pid" \
    "${dir}/bench.log" "${dir}/bench_result.json" \
    "${dir}/bench_status.txt" "${dir}/dmon_pcie.csv" \
    "${dir}/dmon_gpm.csv" "${dir}/cpu_offload_manifest.jsonl"

  cleanup_port "${port}"
  wait_for_clean_gpu "${mode}"
  echo "=== ${mode}: max_num_seqs=${MAX_NUM_SEQS} kv=${KV_CACHE_MEMORY_BYTES} ===" |
    tee -a "${RUN_ROOT}/run.log"

  if [[ "${mode}" == "managed_no_offload" ]]; then
    (
      cd "${VLLM_ROOT}"
      env \
        VLLM_USE_UVM=0 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
        VLLM_UVM_KV_CACHE_MANAGED=1 \
        VLLM_DISABLE_OPENAI_METRICS=1 \
        VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
        VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
        "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
          --port "${port}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          --seed 0 \
          --enforce-eager \
          --disable-log-requests
    ) > "${dir}/server.log" 2>&1 &
  elif [[ "${mode}" == "native_uva" ]]; then
    (
      cd "${VLLM_ROOT}"
      env \
        VLLM_USE_UVM=0 \
        VLLM_CPU_OFFLOAD_BACKEND=uva \
        VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/cpu_offload_manifest.jsonl" \
        VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
        VLLM_UVM_KV_CACHE_MANAGED=0 \
        VLLM_DISABLE_OPENAI_METRICS=1 \
        VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
        VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
        "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
          --port "${port}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          --cpu-offload-gb "${NATIVE_CPU_OFFLOAD_GB}" \
          --seed 0 \
          --enforce-eager \
          --disable-log-requests
    ) > "${dir}/server.log" 2>&1 &
  else
    echo "unknown mode: ${mode}" >&2
    return 2
  fi

  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"
  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server failed for ${mode}" | tee -a "${RUN_ROOT}/run.log"
    stop_process_tree "${server_pid}"
    printf '%s\n' server_failed > "${dir}/bench_status.txt"
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

  stop_pid "${dmon_pcie_pid}"
  stop_pid "${dmon_gpm_pid}"
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  wait_for_clean_gpu "after_${mode}" || true
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${NUM_PROMPTS}" \
    "${OUTPUT_LEN}" "${MAX_NUM_SEQS}" "${KV_CACHE_MEMORY_BYTES}" \
    "${NATIVE_CPU_OFFLOAD_GB}" <<'PY'
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))

def metric(text, name):
    match = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(match.group(1)) if match else None

def output_hash(path):
    values = read_json(path).get("generated_texts")
    if not isinstance(values, list):
        return None
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()

def pcie(path):
    rx = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < 5:
                continue
            try:
                rx.append(float(parts[3]))
            except ValueError:
                pass
    return {
        "samples": len(rx),
        "rx_mean_MBps": statistics.mean(rx) if rx else None,
        "rx_max_MBps": max(rx) if rx else None,
    }

def manifest_bytes(path):
    total = 0
    records = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("event") == "cpu_offload_parameter":
                records += 1
                total += int(item.get("num_bytes") or item.get("bytes") or 0)
    return records, total

cases = {}
for case_dir in sorted(root.glob("*_s*")):
    if not case_dir.is_dir():
        continue
    text = (case_dir / "bench.log").read_text(
        encoding="utf-8", errors="replace"
    ) if (case_dir / "bench.log").exists() else ""
    server = (case_dir / "server.log").read_text(
        encoding="utf-8", errors="replace"
    ) if (case_dir / "server.log").exists() else ""
    manifest_records, manifest_total = manifest_bytes(
        case_dir / "cpu_offload_manifest.jsonl"
    )
    cases[case_dir.name] = {
        "status": (case_dir / "bench_status.txt").read_text().strip()
        if (case_dir / "bench_status.txt").exists() else "missing",
        "successful_requests": metric(text, "Successful requests"),
        "failed_requests": metric(text, "Failed requests"),
        "output_tok_s": metric(text, "Output token throughput (tok/s)"),
        "mean_tpot_ms": metric(text, "Mean TPOT (ms)"),
        "p99_tpot_ms": metric(text, "P99 TPOT (ms)"),
        "mean_ttft_ms": metric(text, "Mean TTFT (ms)"),
        "output_sha256": output_hash(case_dir / "bench_result.json"),
        "pcie": pcie(case_dir / "dmon_pcie.csv"),
        "uva_offload_parameter_records": manifest_records,
        "uva_offload_bytes": manifest_total,
        "has_cuda_oom": "out of memory" in server.lower(),
        "has_xid": "xid" in server.lower(),
        "has_uvm_fatal": "uvm fatal" in server.lower(),
    }

summary = {
    "run_root": str(root),
    "config": {
        "num_prompts": int(sys.argv[2]),
        "output_len": int(sys.argv[3]),
        "max_num_seqs": int(sys.argv[4]),
        "kv_cache_memory_bytes": int(sys.argv[5]),
        "native_cpu_offload_gb": float(sys.argv[6]),
    },
    "cases": cases,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

[[ -x "${VLLM_ROOT}/${VLLM_PY}" ]]
[[ -f "${DATASET}" ]]

idx=0
for mode in ${MODES}; do
  idx=$((idx + 1))
  run_case "${mode}" "${idx}" || true
  write_summary | tee "${RUN_ROOT}/summary.latest.json"
done

write_summary | tee "${RUN_ROOT}/summary.json"
echo "summary: ${RUN_ROOT}/summary.json"
