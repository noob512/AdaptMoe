#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_custom_vs_uva_2gb_concurrency_sweep_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-8910}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 2 4 8 16}"
PILOT_MAX_NUM_SEQS="${PILOT_MAX_NUM_SEQS:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1288490189}"

CUSTOM_TARGET_RELEASE_BYTES="${CUSTOM_TARGET_RELEASE_BYTES:-2147483648}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-2}"
NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_EST_BYTES="${EXPERT_EST_BYTES:-4718592}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"
CUSTOM_TOPK_GATE_IMPL="${CUSTOM_TOPK_GATE_IMPL:-fused_select}"
CUSTOM_ACCESS_MODE="${CUSTOM_ACCESS_MODE:-restore}"
CUSTOM_SKIP_ACTIVE_TOPK="${CUSTOM_SKIP_ACTIVE_TOPK:-0}"
CUSTOM_KV_MANAGED="${CUSTOM_KV_MANAGED:-1}"
UVA_KV_MANAGED="${UVA_KV_MANAGED:-0}"
SKIP_POST_INIT_WARMUP="${SKIP_POST_INIT_WARMUP:-0}"
REUSE_PREP="${REUSE_PREP:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
COLD_PLAN_SELECTION_MODE="${COLD_PLAN_SELECTION_MODE:-global_budget}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-10}"
REUSE_PILOT="${REUSE_PILOT:-0}"
RUN_CUSTOM_CASES="${RUN_CUSTOM_CASES:-1}"
RUN_UVA_CASES="${RUN_UVA_CASES:-1}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
PTE_ENABLE_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_enable"
PTE_MAX_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_max_bytes"
ORIGINAL_PTE_ENABLE="$(cat "${PTE_ENABLE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_PTE_MAX="$(cat "${PTE_MAX_PATH}" 2>/dev/null || echo 0)"

PILOT_DIR="${RUN_ROOT}/pilot_trace_s${PILOT_MAX_NUM_SEQS}"
PILOT_TRACE="${PILOT_DIR}/routing_trace.jsonl"
COLD_PLAN="${RUN_ROOT}/cold_approx2gb_plan.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}" "${PILOT_DIR}"

set_param() {
  local path="$1"
  local value="$2"
  if [[ -e "${path}" ]]; then
    printf '%s\n' "${value}" | sudo -n tee "${path}" >/dev/null
  fi
}

restore_pte_params() {
  set_param "${PTE_ENABLE_PATH}" 0 || true
  set_param "${PTE_MAX_PATH}" "${ORIGINAL_PTE_MAX}" || true
  set_param "${PTE_ENABLE_PATH}" "${ORIGINAL_PTE_ENABLE}" || true
}
trap restore_pte_params EXIT

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

reset_reserved_pool() {
  local output="$1"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True),
                 indent=2, sort_keys=True))
PY
  ) > "${output}" 2>&1 || true
}

prealloc_reserved_pool() {
  local output="$1"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RESERVED_SLOT_COUNT}" "${RESERVED_SLOT_BYTES}" <<'PY'
import json
import sys
from vllm.device_allocator.uvm import driver_expert_reserved_pool_prealloc

count = int(sys.argv[1])
size = int(sys.argv[2])
slots = [{"pool_id": 0, "slot_id": slot_id, "slot_bytes": size}
         for slot_id in range(count)]
result = driver_expert_reserved_pool_prealloc(
    slots, device=0, prefer_existing=True
)
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok") or int(result.get("success_count") or 0) < count:
    raise SystemExit(2)
PY
  ) > "${output}" 2>&1
}

require_environment() {
  [[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
    echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
    exit 2
  }
  [[ -f "${DATASET}" ]] || {
    echo "ShareGPT dataset not found: ${DATASET}" >&2
    exit 2
  }
  local built loaded
  built="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")"
  loaded="$(cat /sys/module/nvidia_uvm/srcversion)"
  if [[ "${loaded}" != "${built}" ]]; then
    if [[ "${CUSTOM_ACCESS_MODE}" == "remote_read" ]]; then
      echo "warning: loaded UVM ${loaded} != built UVM ${built}; " \
        "remote_read uses native CUDA UVM APIs" | tee -a "${RUN_ROOT}/run.log"
    else
      echo "loaded UVM ${loaded} != built UVM ${built}" >&2
      exit 2
    fi
  fi
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    set_param "${PTE_MAX_PATH}" "${RESERVED_SLOT_BYTES}"
    set_param "${PTE_ENABLE_PATH}" 1
  fi
}

run_pilot_trace() {
  local dir="${PILOT_DIR}"
  local port="${PORT_BASE}"
  rm -f "${dir}/server.log" "${dir}/bench.log" "${PILOT_TRACE}" \
    "${dir}/bench_status.txt" "${dir}/bench_result.json"
  cleanup_port "${port}"
  wait_for_clean_gpu "pilot_trace"
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    reset_reserved_pool "${dir}/reserved_pool_reset_before.json"
  fi
  echo "=== pilot trace: no-offload managed, max_num_seqs=${PILOT_MAX_NUM_SEQS} ===" \
    | tee -a "${RUN_ROOT}/run.log"

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${PILOT_TRACE}" \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP="${SKIP_POST_INIT_WARMUP}" \
      "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${PILOT_MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "pilot server failed" | tee -a "${RUN_ROOT}/run.log"
    stop_process_tree "${server_pid}"
    if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
      reset_reserved_pool "${dir}/reserved_pool_reset_after_failure.json"
    fi
    exit 2
  fi

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
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    reset_reserved_pool "${dir}/reserved_pool_reset_after.json"
  fi
  wait_for_clean_gpu "after_pilot_trace" || true
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 120 "${dir}/bench.log" >&2
    exit "${rc}"
  fi
}

build_cold_plan() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${PILOT_TRACE}" "${COLD_PLAN}" \
      "${NUM_LAYERS}" "${NUM_EXPERTS}" "${CUSTOM_TARGET_RELEASE_BYTES}" \
      "${EXPERT_EST_BYTES}" "${NUM_PROMPTS}" "${OUTPUT_LEN}" \
      "${PILOT_MAX_NUM_SEQS}" "${COLD_PLAN_SELECTION_MODE}" \
      "${COLD_EXPERTS_PER_LAYER}" <<'PY'
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

trace_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
num_layers = int(sys.argv[3])
num_experts = int(sys.argv[4])
target_bytes = int(sys.argv[5])
expert_est_bytes = int(sys.argv[6])
num_prompts = int(sys.argv[7])
output_len = int(sys.argv[8])
pilot_max_num_seqs = int(sys.argv[9])
selection_mode = sys.argv[10]
cold_experts_per_layer = int(sys.argv[11])

def layer_id_from_name(layer_name):
    if not isinstance(layer_name, str):
        return None
    parts = layer_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None

token_counts = defaultdict(int)
trace_records = 0
for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    layer_id = layer_id_from_name(record.get("layer_name"))
    if layer_id is None or not (0 <= layer_id < num_layers):
        continue
    counts = record.get("expert_token_counts") or {}
    if not isinstance(counts, dict):
        continue
    trace_records += 1
    for expert_s, count in counts.items():
        try:
            expert_id = int(expert_s)
            count_i = int(count)
        except (TypeError, ValueError):
            continue
        if 0 <= expert_id < num_experts:
            token_counts[(layer_id, expert_id)] += count_i

candidates = []
for layer_id in range(num_layers):
    for expert_id in range(num_experts):
        candidates.append((token_counts[(layer_id, expert_id)],
                           layer_id, expert_id))
candidates.sort(key=lambda item: (item[0], item[1], item[2]))

if selection_mode == "global_budget":
    target_records = math.ceil(target_bytes / expert_est_bytes)
    selected = candidates[:target_records]
elif selection_mode == "per_layer_coldest_n":
    target_records = num_layers * cold_experts_per_layer
    selected = []
    for layer_id in range(num_layers):
        layer_candidates = [
            item for item in candidates if item[1] == layer_id
        ]
        selected.extend(layer_candidates[:cold_experts_per_layer])
    selected.sort(key=lambda item: (item[0], item[1], item[2]))
else:
    raise SystemExit(f"unknown cold plan selection mode: {selection_mode}")
offload_plan = [
    {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "reason": "coldest_by_pilot_token_count",
        "pilot_token_count": count,
    }
    for count, layer_id, expert_id in selected
]
zero_hit = sum(1 for count, _, _ in selected if count == 0)
selected_token_count = sum(count for count, _, _ in selected)
observed_token_count = sum(token_counts.values())
metadata = {
    "policy": "exact_used_oracle_v1",
    "oracle_scope": (
        f"serving_bench_sharegpt_{num_prompts}x{output_len}_"
        f"temperature0_seed42_pilot_s{pilot_max_num_seqs}"
    ),
    "routing_trace": str(trace_path),
    "trace_records": trace_records,
    "target_release_bytes": target_bytes,
    "selection_mode": selection_mode,
    "cold_experts_per_layer": (
        cold_experts_per_layer
        if selection_mode == "per_layer_coldest_n" else None
    ),
    "expert_est_bytes": expert_est_bytes,
    "target_records": target_records,
    "selected_records": len(offload_plan),
    "selected_zero_hit_records": zero_hit,
    "selected_nonzero_records": len(offload_plan) - zero_hit,
    "selected_token_count": selected_token_count,
    "observed_token_count": observed_token_count,
    "selected_token_fraction": (
        selected_token_count / observed_token_count
        if observed_token_count else 0.0
    ),
    "estimated_release_bytes": len(offload_plan) * expert_est_bytes,
    "min_selected_token_count": selected[0][0] if selected else None,
    "max_selected_token_count": selected[-1][0] if selected else None,
}
plan = {
    "metadata": metadata,
    "prefetch_plan": [],
    "offload_plan": offload_plan,
    "prefetch_plan_records": 0,
    "offload_plan_records": len(offload_plan),
}
output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/plan_build.log"
}

run_custom_case() {
  local max_num_seqs="$1"
  local idx="$2"
  local port=$((PORT_BASE + 10 + idx))
  local name="custom_2gb_s${max_num_seqs}"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  rm -f "${dir}/server.log" "${dir}/bench.log" "${dir}/bench_result.json" \
    "${dir}/bench_status.txt" "${dir}/init_trace.jsonl" \
    "${dir}/cpu_offload_manifest.jsonl" "${dir}/dmon_pcie.csv" \
    "${dir}/dmon_gpm.csv"

  cleanup_port "${port}"
  wait_for_clean_gpu "${name}"
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    reset_reserved_pool "${dir}/reserved_pool_reset_before.json"
    prealloc_reserved_pool "${dir}/reserved_prealloc_before_model.json"
  fi
  echo "=== ${name}: access=${CUSTOM_ACCESS_MODE} target=${CUSTOM_TARGET_RELEASE_BYTES} gate=${CUSTOM_TOPK_GATE_IMPL} ===" \
    | tee -a "${RUN_ROOT}/run.log"

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED="${CUSTOM_KV_MANAGED}" \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
      VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0 \
      VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
      VLLM_UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE=0 \
      VLLM_UVM_RESIDENCY_TRACE_ENABLE=0 \
      VLLM_UVM_DRIVER_TAG_ENABLE=1 \
      VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE="${CUSTOM_ACCESS_MODE}" \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${COLD_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
      VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
      VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
      VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${COLD_PLAN}" \
      VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
      VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
      VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
      VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1 \
      VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE=1 \
      VLLM_UVM_EXPERT_RESERVED_POST_STAGE_M_PREALLOC_FILE="${dir}/post_stage_m_reserved_prealloc.jsonl" \
      VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=dynamic \
      VLLM_UVM_EXPERT_RESERVED_BANK_COUNT=1 \
      VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
      VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
      VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS=4 \
      VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=1 \
      VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
      VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
      VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
      VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
      VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=1 \
      VLLM_UVM_EXPERT_RESERVED_SKIP_ACTIVE_TOPK="${CUSTOM_SKIP_ACTIVE_TOPK}" \
      VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_IMPL="${CUSTOM_TOPK_GATE_IMPL}" \
      VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP="${SKIP_POST_INIT_WARMUP}" \
      "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${max_num_seqs}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server failed for ${name}" | tee -a "${RUN_ROOT}/run.log"
    stop_process_tree "${server_pid}"
    if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
      reset_reserved_pool "${dir}/reserved_pool_reset_after_failure.json"
    fi
    printf '%s\n' "server_failed" > "${dir}/bench_status.txt"
    return 0
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
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    reset_reserved_pool "${dir}/reserved_pool_reset_after.json"
  fi
  wait_for_clean_gpu "after_${name}" || true
}

run_uva_case() {
  local max_num_seqs="$1"
  local idx="$2"
  local port=$((PORT_BASE + 30 + idx))
  local name="uva_2gb_s${max_num_seqs}"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  rm -f "${dir}/server.log" "${dir}/bench.log" "${dir}/bench_result.json" \
    "${dir}/bench_status.txt" "${dir}/cpu_offload_manifest.jsonl" \
    "${dir}/dmon_pcie.csv" "${dir}/dmon_gpm.csv"

  cleanup_port "${port}"
  wait_for_clean_gpu "${name}"
  echo "=== ${name}: cpu_offload_gb=${NATIVE_CPU_OFFLOAD_GB} ===" \
    | tee -a "${RUN_ROOT}/run.log"

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_CPU_OFFLOAD_BACKEND=uva \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/cpu_offload_manifest.jsonl" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
      VLLM_UVM_KV_CACHE_MANAGED="${UVA_KV_MANAGED}" \
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
      VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP="${SKIP_POST_INIT_WARMUP}" \
      "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
        --port "${port}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${max_num_seqs}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --cpu-offload-gb "${NATIVE_CPU_OFFLOAD_GB}" \
        --seed 0 \
        --enforce-eager \
        --disable-log-requests
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "server failed for ${name}" | tee -a "${RUN_ROOT}/run.log"
    stop_process_tree "${server_pid}"
    printf '%s\n' "server_failed" > "${dir}/bench_status.txt"
    return 0
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
  wait_for_clean_gpu "after_${name}" || true
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RUN_ROOT}" "${COLD_PLAN}" \
      "${CUSTOM_ACCESS_MODE}" "${NUM_PROMPTS}" "${OUTPUT_LEN}" \
      "${KV_CACHE_MEMORY_BYTES}" "${CUSTOM_KV_MANAGED}" \
      "${UVA_KV_MANAGED}" "${SKIP_POST_INIT_WARMUP}" <<'PY'
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
custom_access_mode = sys.argv[3]
num_prompts = int(sys.argv[4])
output_len = int(sys.argv[5])
kv_cache_memory_bytes = int(sys.argv[6])
custom_kv_managed = int(sys.argv[7])
uva_kv_managed = int(sys.argv[8])
skip_post_init_warmup = int(sys.argv[9])

def read_json(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}

def read_jsonl(path: Path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records

def parse_metric(text: str, name: str):
    match = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(match.group(1)) if match else None

def parse_int_metric(text: str, name: str):
    value = parse_metric(text, name)
    return int(value) if value is not None else None

def parse_bench(case_dir: Path):
    log = case_dir / "bench.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    data = read_json(case_dir / "bench_result.json")
    generated = data.get("generated_texts")
    output_sha256 = None
    if isinstance(generated, list):
        output_sha256 = hashlib.sha256(
            "\0".join(map(str, generated)).encode()
        ).hexdigest()
    return {
        "successful_requests": parse_int_metric(text, "Successful requests") or data.get("completed"),
        "failed_requests": parse_int_metric(text, "Failed requests") if parse_int_metric(text, "Failed requests") is not None else data.get("failed"),
        "benchmark_duration_s": parse_metric(text, "Benchmark duration (s)") or data.get("duration"),
        "output_tok_s": parse_metric(text, "Output token throughput (tok/s)") or data.get("output_throughput"),
        "total_tok_s": parse_metric(text, "Total Token throughput (tok/s)") or data.get("total_token_throughput"),
        "mean_ttft_ms": parse_metric(text, "Mean TTFT (ms)") or data.get("mean_ttft_ms"),
        "p99_ttft_ms": parse_metric(text, "P99 TTFT (ms)") or data.get("p99_ttft_ms"),
        "mean_tpot_ms": parse_metric(text, "Mean TPOT (ms)") or data.get("mean_tpot_ms"),
        "p99_tpot_ms": parse_metric(text, "P99 TPOT (ms)") or data.get("p99_tpot_ms"),
        "total_input_tokens": data.get("total_input_tokens"),
        "total_output_tokens": data.get("total_output_tokens"),
        "max_concurrent_requests": data.get("max_concurrent_requests"),
        "output_sha256": output_sha256,
    }

def parse_dmon_pcie(case_dir: Path):
    path = case_dir / "dmon_pcie.csv"
    rx = []
    tx = []
    if path.exists():
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            parts = line.split()
            if not parts or parts[0].startswith("#") or len(parts) < 5:
                continue
            try:
                rx.append(float(parts[3]))
                tx.append(float(parts[4]))
            except ValueError:
                continue
    def quantile(values, q):
        if not values:
            return None
        values = sorted(values)
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        fraction = position - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    return {
        "pcie_samples": len(rx),
        "pcie_rx_mean_MBps": statistics.mean(rx) if rx else None,
        "pcie_rx_p95_MBps": quantile(rx, 0.95),
        "pcie_rx_max_MBps": max(rx) if rx else None,
        "pcie_rx_sample_sum_MB": sum(rx),
        "pcie_tx_mean_MBps": statistics.mean(tx) if tx else None,
        "pcie_tx_max_MBps": max(tx) if tx else None,
    }

def parse_server(case_dir: Path):
    text = (case_dir / "server.log").read_text(encoding="utf-8", errors="replace") if (case_dir / "server.log").exists() else ""
    kv_tokens = None
    match = re.search(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", text)
    if match:
        kv_tokens = int(match.group(1).replace(",", ""))
    running = [int(v) for v in re.findall(r"Running:\s+([0-9]+)\s+reqs", text)]
    waiting = [int(v) for v in re.findall(r"Waiting:\s+([0-9]+)\s+reqs", text)]
    kv_usage = [float(v) for v in re.findall(r"GPU KV cache usage:\s+([0-9.]+)%", text)]
    return {
        "gpu_kv_cache_tokens": kv_tokens,
        "server_running_max": max(running) if running else None,
        "server_waiting_max": max(waiting) if waiting else None,
        "kv_usage_pct_max": max(kv_usage) if kv_usage else None,
        "has_cuda_oom": "out of memory" in text.lower() or "cudaerrormemoryallocation" in text.lower(),
        "has_illegal_memory_access": "illegal memory access" in text.lower(),
        "has_xid": "xid" in text.lower(),
        "has_uvm_fatal": "uvm fatal" in text.lower(),
        "has_mmu": "mmu" in text.lower(),
    }

def custom_release(case_dir: Path):
    records = read_jsonl(case_dir / "init_trace.jsonl")
    issued = [
        item for item in records
        if str(item.get("action") or "").endswith("_issued")
    ]
    success = [
        item for item in issued
        if item.get("success") is True and int(item.get("bytes") or 0) > 0
    ]
    restore_actions = [
        item for item in records
        if any(
            marker in str(item.get("action") or "")
            for marker in ("reserved_prefetch", "slot_remap", "restore")
        )
    ]
    return {
        "custom_release_issued_records": len(issued),
        "custom_release_success_records": len(success),
        "custom_release_failed_records": len(issued) - len(success),
        "custom_release_bytes": sum(int(item.get("bytes") or 0) for item in success),
        "custom_reserved_restore_action_records": len(restore_actions),
    }

def uva_manifest(case_dir: Path):
    records = read_jsonl(case_dir / "cpu_offload_manifest.jsonl")
    params = [
        item for item in records
        if item.get("event") == "cpu_offload_parameter"
    ]
    return {
        "uva_offload_params": len(params),
        "uva_offload_bytes": sum(int(item.get("num_bytes") or item.get("bytes") or 0)
                                  for item in params),
    }

cases = []
for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir():
        continue
    match = re.match(r"(custom_2gb|uva_2gb)_s([0-9]+)$", case_dir.name)
    if not match:
        continue
    mode = "custom" if match.group(1) == "custom_2gb" else "uva"
    max_num_seqs = int(match.group(2))
    status_path = case_dir / "bench_status.txt"
    status = status_path.read_text(encoding="utf-8", errors="replace").strip() if status_path.exists() else "missing"
    item = {
        "case": case_dir.name,
        "mode": mode,
        "max_num_seqs": max_num_seqs,
        "status": status,
    }
    item.update(parse_bench(case_dir))
    item.update(parse_server(case_dir))
    item.update(parse_dmon_pcie(case_dir))
    if mode == "custom":
        item.update(custom_release(case_dir))
    else:
        item.update(uva_manifest(case_dir))
    cases.append(item)

cases.sort(key=lambda item: (item["max_num_seqs"], item["mode"]))
by_concurrency = {}
for item in cases:
    by_concurrency.setdefault(str(item["max_num_seqs"]), {})[item["mode"]] = item

comparisons = []
for max_num_seqs, modes in sorted(by_concurrency.items(), key=lambda kv: int(kv[0])):
    custom = modes.get("custom")
    uva = modes.get("uva")
    row = {"max_num_seqs": int(max_num_seqs)}
    if custom and uva and custom.get("output_tok_s") and uva.get("output_tok_s"):
        row["custom_output_tok_s"] = custom["output_tok_s"]
        row["uva_output_tok_s"] = uva["output_tok_s"]
        row["custom_vs_uva_output_pct"] = (
            custom["output_tok_s"] / uva["output_tok_s"] - 1.0
        ) * 100.0
        row["custom_mean_tpot_ms"] = custom.get("mean_tpot_ms")
        row["uva_mean_tpot_ms"] = uva.get("mean_tpot_ms")
        if custom.get("mean_tpot_ms") and uva.get("mean_tpot_ms"):
            row["custom_vs_uva_mean_tpot_pct"] = (
                custom["mean_tpot_ms"] / uva["mean_tpot_ms"] - 1.0
            ) * 100.0
        row["custom_mean_ttft_ms"] = custom.get("mean_ttft_ms")
        row["uva_mean_ttft_ms"] = uva.get("mean_ttft_ms")
        row["custom_pcie_rx_mean_MBps"] = custom.get("pcie_rx_mean_MBps")
        row["uva_pcie_rx_mean_MBps"] = uva.get("pcie_rx_mean_MBps")
        row["output_hash_match"] = bool(custom.get("output_sha256")) and (
            custom.get("output_sha256") == uva.get("output_sha256")
        )
    comparisons.append(row)

summary = {
    "run_root": str(root),
    "config": {
        "custom_access_mode": custom_access_mode,
        "num_prompts": num_prompts,
        "output_len": output_len,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "custom_kv_managed": custom_kv_managed,
        "uva_kv_managed": uva_kv_managed,
        "skip_post_init_warmup": skip_post_init_warmup,
    },
    "plan": read_json(plan_path).get("metadata", {}),
    "cases": cases,
    "comparisons": comparisons,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                   encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

main() {
  require_environment
  if [[ "${SUMMARY_ONLY}" == "1" ]]; then
    write_summary
    echo "summary=${SUMMARY}"
    return 0
  fi
  echo "run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
  echo "max_num_seqs_list=${MAX_NUM_SEQS_LIST}" | tee -a "${RUN_ROOT}/run.log"
  echo "custom_access_mode=${CUSTOM_ACCESS_MODE}" | tee -a "${RUN_ROOT}/run.log"
  echo "cold_plan_selection_mode=${COLD_PLAN_SELECTION_MODE}" | tee -a "${RUN_ROOT}/run.log"
  if [[ "${REUSE_PREP}" == "1" && -s "${PILOT_TRACE}" && -s "${COLD_PLAN}" ]]; then
    echo "reusing pilot trace and cold plan" | tee -a "${RUN_ROOT}/run.log"
  elif [[ "${REUSE_PILOT}" == "1" && -s "${PILOT_TRACE}" ]]; then
    echo "reusing pilot trace and rebuilding cold plan" | tee -a "${RUN_ROOT}/run.log"
    build_cold_plan
  else
    run_pilot_trace
    build_cold_plan
  fi

  local idx=0
  if [[ "${RUN_CUSTOM_CASES}" == "1" ]]; then
    for seqs in ${MAX_NUM_SEQS_LIST}; do
      run_custom_case "${seqs}" "${idx}"
      idx=$((idx + 1))
    done
  fi

  idx=0
  if [[ "${RUN_UVA_CASES}" == "1" ]]; then
    for seqs in ${MAX_NUM_SEQS_LIST}; do
      run_uva_case "${seqs}" "${idx}"
      idx=$((idx + 1))
    done
  fi

  write_summary
  restore_pte_params
  wait_for_clean_gpu "after_all" || true
  echo "summary=${SUMMARY}"
}

main "$@"
