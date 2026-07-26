#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_kv_expert_fault_attribution_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-${VLLM_ROOT}/.venv/bin/python}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
PARAM_DIR="${PARAM_DIR:-/sys/module/nvidia_uvm/parameters}"
TRACE_DIR="${TRACE_DIR:-/sys/kernel/tracing}"

NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"

mkdir -p "${RUN_ROOT}"

read_param() {
  local name="$1"
  cat "${PARAM_DIR}/${name}" 2>/dev/null || true
}

write_param() {
  local name="$1"
  local value="$2"
  if [[ -e "${PARAM_DIR}/${name}" ]]; then
    printf "%s" "${value}" | sudo -n tee "${PARAM_DIR}/${name}" >/dev/null
  fi
}

OLD_ADDR="$(read_param uvm_perf_fault_log_addresses)"
OLD_DEST="$(read_param uvm_perf_fault_log_destination)"
OLD_COUNTERS="$(read_param uvm_perf_fault_log_counters)"
OLD_DEBUG="$(read_param uvm_debug_prints)"

cleanup_all() {
  set +e
  if [[ -n "${TRACE_PGID:-}" ]]; then
    sudo kill -- "-${TRACE_PGID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TRACE_PID:-}" ]]; then
    kill "${TRACE_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -e "${TRACE_DIR}/tracing_on" ]]; then
    echo 0 | sudo -n tee "${TRACE_DIR}/tracing_on" >/dev/null 2>&1 || true
  fi
  [[ -n "${OLD_ADDR}" ]] && write_param uvm_perf_fault_log_addresses "${OLD_ADDR}" || true
  [[ -n "${OLD_DEST}" ]] && write_param uvm_perf_fault_log_destination "${OLD_DEST}" || true
  [[ -n "${OLD_COUNTERS}" ]] && write_param uvm_perf_fault_log_counters "${OLD_COUNTERS}" || true
  [[ -n "${OLD_DEBUG}" ]] && write_param uvm_debug_prints "${OLD_DEBUG}" || true
}
trap cleanup_all EXIT

cleanup_port() {
  local port="$1"
  pkill -f "vllm.entrypoints.cli.main serve.*--port ${port}" 2>/dev/null || true
  pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
}

stop_tree() {
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

wait_gpu() {
  for _ in $(seq 1 120); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
      | head -1 | tr -d " " || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  nvidia-smi >&2 || true
  return 1
}

start_trace() {
  local out="$1"
  sudo -n mount -t tracefs nodev "${TRACE_DIR}" >/dev/null 2>&1 || true
  echo 0 | sudo -n tee "${TRACE_DIR}/tracing_on" >/dev/null
  : | sudo -n tee "${TRACE_DIR}/trace" >/dev/null || true
  write_param uvm_debug_prints 0
  write_param uvm_perf_fault_log_destination 1
  write_param uvm_perf_fault_log_addresses 1
  write_param uvm_perf_fault_log_counters 1
  echo 1 | sudo -n tee "${TRACE_DIR}/tracing_on" >/dev/null
  setsid bash -c \
    'sudo -n cat /sys/kernel/tracing/trace_pipe |
     grep --line-buffered -E "Replayable fault GPU|Replayable fault stats|codex_fault_attr"' \
    > "${out}" &
  TRACE_PID=$!
  TRACE_PGID="$(ps -o pgid= -p "${TRACE_PID}" | tr -d "[:space:]")"
}

stop_trace() {
  set +e
  printf "<6>codex_fault_attr_stop_%s\n" "$(date +%s%N)" \
    | sudo -n tee /dev/kmsg >/dev/null 2>&1 || true
  sleep 1
  if [[ -n "${TRACE_PGID:-}" ]]; then
    sudo kill -- "-${TRACE_PGID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${TRACE_PID:-}" ]]; then
    kill "${TRACE_PID}" >/dev/null 2>&1 || true
    wait "${TRACE_PID}" >/dev/null 2>&1 || true
  fi
  echo 0 | sudo -n tee "${TRACE_DIR}/tracing_on" >/dev/null 2>&1 || true
  unset TRACE_PID TRACE_PGID
  set -e
}

run_case() {
  local name="$1"
  local kv_bytes="$2"
  local port="$3"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  printf "case=%s kv=%s\n" "${name}" "${kv_bytes}" | tee -a "${RUN_ROOT}/run.log"
  cleanup_port "${port}"
  wait_gpu

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_UVM_ADDRESS_LOG_ENABLE=1 \
      VLLM_UVM_ADDRESS_LOG_FILE="${dir}/address_regions.log" \
      VLLM_UVM_WEIGHT_MAP_ENABLE=1 \
      VLLM_UVM_WEIGHT_MAP_FILE="${dir}/weight_map.jsonl" \
      VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
      VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
      VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=0 \
      VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0 \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
      VLLM_UVM_KV_INIT_FAULT_PROBE=1 \
      VLLM_UVM_KV_INIT_FAULT_PROBE_FILE="${dir}/kv_init_probe.jsonl" \
      VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY=0 \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      "${PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${kv_bytes}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &

  local server_pid=$!
  echo "${server_pid}" > "${dir}/server.pid"
  local ready=0
  for _ in $(seq 1 600); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      ready=2
      break
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done

  if [[ "${ready}" != 1 ]]; then
    echo "server_failed" > "${dir}/status.txt"
    tail -120 "${dir}/server.log" >&2 || true
    stop_tree "${server_pid}"
    return 0
  fi

  printf "<6>codex_fault_attr_%s_begin_%s\n" "${name}" "$(date +%s%N)" \
    | sudo -n tee /dev/kmsg >/dev/null 2>&1 || true
  printf "1\n" | sudo -n tee /proc/driver/nvidia-uvm/latency_stats_reset \
    >/dev/null 2>&1 || true
  cat /proc/driver/nvidia-uvm/latency_stats > "${dir}/latency_before.log" \
    2>/dev/null || true

  start_trace "${dir}/fault_trace.log"
  set +e
  (
    cd "${VLLM_ROOT}"
    "${PY}" -m vllm.entrypoints.cli.main bench serve \
      --backend openai \
      --model "${MODEL}" \
      --dataset-name sharegpt \
      --dataset-path "${DATASET}" \
      --num-prompts "${NUM_PROMPTS}" \
      --sharegpt-output-len "${OUTPUT_LEN}" \
      --seed 42 \
      --temperature 0 \
      --ignore-eos \
      --request-rate inf \
      --host 127.0.0.1 \
      --port "${port}" \
      --endpoint /v1/completions \
      --ready-check-timeout-sec 600 \
      --save-result \
      --save-detailed \
      --result-dir "${dir}" \
      --result-filename bench_result.json
  ) > "${dir}/bench.log" 2>&1
  local bench_rc=$?
  set -e

  stop_trace
  echo "${bench_rc}" > "${dir}/status.txt"
  cat /proc/driver/nvidia-uvm/latency_stats > "${dir}/latency_after.log" \
    2>/dev/null || true
  stop_tree "${server_pid}"
  cleanup_port "${port}"
  wait_gpu || true
}

run_case kv2g 2147483648 8893
run_case kv6g 6442450944 8894

cleanup_all
python3 "${SCRIPT_DIR}/analyze_kv_expert_fault_attribution.py" "${RUN_ROOT}"
printf "summary=%s/summary.json\n" "${RUN_ROOT}"
