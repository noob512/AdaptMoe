#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_forward_token_uva_bench_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-9060}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-536870912}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-128}"
BENCH_SIZES="${BENCH_SIZES:-1,4,8,32,64,128}"
BENCH_PATTERNS="${BENCH_PATTERNS:-identical,diverse}"
BENCH_MODE="${BENCH_MODE:-decode}"
BENCH_WARMUPS="${BENCH_WARMUPS:-5}"
BENCH_ITERS="${BENCH_ITERS:-20}"
BENCH_BURST_ITERS="${BENCH_BURST_ITERS:-10}"
BENCH_BURST_ROUNDS="${BENCH_BURST_ROUNDS:-3}"
NATIVE_CPU_OFFLOAD_GB_LIST="${NATIVE_CPU_OFFLOAD_GB_LIST-2 6 10 12}"
ROUTING_TRACE_ENABLE="${ROUTING_TRACE_ENABLE:-0}"
LAYER_TIMING_ENABLE="${LAYER_TIMING_ENABLE:-0}"

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

run_case() {
  local label="$1"
  local port="$2"
  local offload_gb="$3"
  local dir="${RUN_ROOT}/${label}"
  mkdir -p "${dir}"
  rm -f "${dir}/layer_timing.jsonl"
  cleanup_port "${port}"
  wait_for_clean_gpu "${label}"

  local -a offload_args=()
  if [[ "${offload_gb}" != "0" ]]; then
    offload_args=(--cpu-offload-gb "${offload_gb}")
  fi

  echo "=== ${label}: cpu_offload_gb=${offload_gb} ===" | tee -a "${RUN_ROOT}/run.log"
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
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${ROUTING_TRACE_ENABLE}" \
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl" \
      VLLM_QWEN3_MOE_LAYER_TIMING_ENABLE="${LAYER_TIMING_ENABLE}" \
      VLLM_QWEN3_MOE_LAYER_TIMING_FILE="${dir}/layer_timing.jsonl" \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap \
      VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
      VLLM_FORWARD_TOKEN_BENCH_ENABLE=1 \
      VLLM_FORWARD_TOKEN_BENCH_LABEL="${label}" \
      VLLM_FORWARD_TOKEN_BENCH_MODE="${BENCH_MODE}" \
      VLLM_FORWARD_TOKEN_BENCH_SIZES="${BENCH_SIZES}" \
      VLLM_FORWARD_TOKEN_BENCH_PATTERNS="${BENCH_PATTERNS}" \
      VLLM_FORWARD_TOKEN_BENCH_WARMUPS="${BENCH_WARMUPS}" \
      VLLM_FORWARD_TOKEN_BENCH_ITERS="${BENCH_ITERS}" \
      VLLM_FORWARD_TOKEN_BENCH_BURST_ITERS="${BENCH_BURST_ITERS}" \
      VLLM_FORWARD_TOKEN_BENCH_BURST_ROUNDS="${BENCH_BURST_ROUNDS}" \
      "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len 4096 \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
        --gpu-memory-utilization 0.97 \
        "${offload_args[@]}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  local status
  if wait_for_server "${port}" "${server_pid}"; then
    status=0
  else
    status=$?
    echo "server failed for ${label}" | tee -a "${RUN_ROOT}/run.log"
  fi
  printf '%s\n' "${status}" > "${dir}/status.txt"
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  wait_for_clean_gpu "after_${label}" || true
}

run_case local_hbm "${PORT_BASE}" 0
case_index=1
for offload_gb in ${NATIVE_CPU_OFFLOAD_GB_LIST}; do
  label="native_uva${offload_gb//./p}"
  run_case "${label}" "$((PORT_BASE + case_index))" "${offload_gb}"
  case_index=$((case_index + 1))
done

(
  cd "${VLLM_ROOT}"
  "${VLLM_PY}" - "${RUN_ROOT}" "${BENCH_SIZES}" \
    "${KV_CACHE_MEMORY_BYTES}" "${BENCH_WARMUPS}" "${BENCH_ITERS}" \
    "${BENCH_BURST_ITERS}" "${BENCH_BURST_ROUNDS}" \
    "${NATIVE_CPU_OFFLOAD_GB_LIST}" "${BENCH_MODE}" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
bench_sizes = [int(value) for value in sys.argv[2].split(",")]
kv_cache_memory_bytes = int(sys.argv[3])
bench_warmups = int(sys.argv[4])
bench_iters = int(sys.argv[5])
bench_burst_iters = int(sys.argv[6])
bench_burst_rounds = int(sys.argv[7])
offload_gb_values = sys.argv[8].split()
bench_mode = sys.argv[9]
labels = ["local_hbm"] + [
    f"native_uva{value.replace('.', 'p')}" for value in offload_gb_values
]
cases = []
for label in labels:
    directory = root / label
    log_path = directory / "server.log"
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    results = []
    for line in text.splitlines():
        marker = "VLLM_FORWARD_TOKEN_BENCH_RESULT "
        if marker in line:
            results.append(json.loads(line.split(marker, 1)[1]))

    manifest_path = directory / "cpu_offload_manifest.jsonl"
    offload_records = []
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("event") == "cpu_offload_parameter":
                offload_records.append(record)

    cases.append(
        {
            "label": label,
            "status": (directory / "status.txt").read_text().strip(),
            "result_count": len(results),
            "results": results,
            "offload_bytes": sum(int(r.get("bytes", 0)) for r in offload_records),
            "offload_params": len(offload_records),
            "has_cuda_oom": "out of memory" in text.lower(),
            "has_illegal_memory_access": "illegal memory access" in text.lower(),
            "has_uvm_fatal": "uvm fatal" in text.lower(),
            "has_xid": bool(re.search(r"\\bXid\\b", text, re.IGNORECASE)),
        }
    )

by_case = {
    case["label"]: {
        (r["pattern"], r["num_tokens"]): r for r in case["results"]
    }
    for case in cases
}
comparisons = []
for label in labels[1:]:
    for key in sorted(set(by_case["local_hbm"]) & set(by_case[label])):
        local = by_case["local_hbm"][key]
        uva = by_case[label][key]
        comparisons.append(
            {
                "label": label,
                "pattern": key[0],
                "num_tokens": key[1],
                "local_median_ms": local["median_ms"],
                "uva_median_ms": uva["median_ms"],
                "uva_overhead_pct": (
                    uva["median_ms"] / local["median_ms"] - 1
                )
                * 100,
                "local_synchronized_median_ms": local[
                    "synchronized_median_ms"
                ],
                "uva_synchronized_median_ms": uva["synchronized_median_ms"],
                "uva_synchronized_overhead_pct": (
                    uva["synchronized_median_ms"]
                    / local["synchronized_median_ms"]
                    - 1
                )
                * 100,
                "local_burst_median_ms": local[
                    "burst_median_ms_per_forward"
                ],
                "uva_burst_median_ms": uva["burst_median_ms_per_forward"],
                "uva_burst_overhead_pct": (
                    uva["burst_median_ms_per_forward"]
                    / local["burst_median_ms_per_forward"]
                    - 1
                )
                * 100,
                "local_tokens_per_second": local["tokens_per_second"],
                "uva_tokens_per_second": uva["tokens_per_second"],
            }
        )

summary = {
    "config": {
        "bench_iters": bench_iters,
        "bench_burst_iters": bench_burst_iters,
        "bench_burst_rounds": bench_burst_rounds,
        "bench_sizes": bench_sizes,
        "bench_warmups": bench_warmups,
        "forward_only": True,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "mode": bench_mode,
        "uniform_decode": bench_mode == "decode",
    },
    "cases": cases,
    "comparisons": comparisons,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
) | tee "${RUN_ROOT}/summary.stdout.json"

echo "summary=${RUN_ROOT}/summary.json"
