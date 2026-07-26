#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_custom_zero_hit_skip_gate_concurrency_sweep_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-8890}"
TIMEOUT_SEC="${TIMEOUT_SEC:-5400}"
PILOT_MAX_NUM_SEQS="${PILOT_MAX_NUM_SEQS:-16}"

MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 4 8 16}"
NUM_PROMPTS="${NUM_PROMPTS:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
REUSE_PREP="${REUSE_PREP:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"

NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"
CUSTOM_ACCESS_MODE="${CUSTOM_ACCESS_MODE:-restore}"
CUSTOM_CASE_PREFIX="${CUSTOM_CASE_PREFIX:-custom_zero_hit_skip}"
CUSTOM_SKIP_ACTIVE_TOPK="${CUSTOM_SKIP_ACTIVE_TOPK:-1}"
CUSTOM_TOPK_GATE_IMPL="${CUSTOM_TOPK_GATE_IMPL:-none}"
MAX_ZERO_HIT_EXPERTS_PER_LAYER="${MAX_ZERO_HIT_EXPERTS_PER_LAYER:-0}"
INCLUDE_ACTIVE_EXPERTS_PER_LAYER="${INCLUDE_ACTIVE_EXPERTS_PER_LAYER:-0}"
PLAN_RESIDENCY_TRACE_ENABLE="${PLAN_RESIDENCY_TRACE_ENABLE:-0}"
PLAN_RESIDENCY_TRACE_MAX_EXPERTS="${PLAN_RESIDENCY_TRACE_MAX_EXPERTS:-0}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
PTE_ENABLE_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_enable"
PTE_MAX_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_max_bytes"
ORIGINAL_PTE_ENABLE="$(cat "${PTE_ENABLE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_PTE_MAX="$(cat "${PTE_MAX_PATH}" 2>/dev/null || echo 0)"

FIXTURE="${RUN_ROOT}/sharegpt_${NUM_PROMPTS}x${OUTPUT_LEN}_fixture.json"
CUSTOM_JSONL="${RUN_ROOT}/sharegpt_${NUM_PROMPTS}x${OUTPUT_LEN}_custom.jsonl"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PILOT_DIR="${RUN_ROOT}/pilot_trace"
PILOT_TRACE="${PILOT_DIR}/routing_trace.jsonl"
PILOT_RESULT="${PILOT_DIR}/bench_result.json"
ZERO_HIT_PLAN="${RUN_ROOT}/strict_zero_hit_all_experts_plan.json"
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
        "native CUDA-UVM remote_read does not use custom release/remap ioctls" \
        | tee -a "${RUN_ROOT}/run.log"
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

generate_fixture() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET}" "${MODEL}" "${FIXTURE}" "${CUSTOM_JSONL}" \
      "${FIXTURE_META}" "${NUM_PROMPTS}" "${OUTPUT_LEN}" <<'PY'
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import is_valid_sequence

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
custom_jsonl_path = Path(sys.argv[4])
meta_path = Path(sys.argv[5])
num_prompts = int(sys.argv[6])
output_len = int(sys.argv[7])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
data = json.loads(dataset_path.read_text(encoding="utf-8"))
data = [
    item for item in data
    if isinstance(item, dict)
    and isinstance(item.get("conversations"), list)
    and len(item["conversations"]) >= 2
]
random.seed(42)
random.shuffle(data)

requests = []
custom_lines = []
prompt_lengths = []
for item in data:
    prompt = item["conversations"][0].get("value")
    completion = item["conversations"][1].get("value")
    if not isinstance(prompt, str) or not isinstance(completion, str):
        continue
    prompt_len = len(tokenizer(prompt).input_ids)
    completion_len = len(tokenizer(completion).input_ids)
    if not is_valid_sequence(
        prompt_len,
        output_len,
        skip_min_output_len_check=True,
    ):
        continue
    request = {
        "request": len(requests),
        "prompt": prompt,
        "max_tokens": output_len,
        "prompt_tokens": prompt_len,
        "original_completion_tokens": completion_len,
    }
    requests.append(request)
    custom_lines.append(json.dumps({"prompt": prompt}, ensure_ascii=False))
    prompt_lengths.append(prompt_len)
    if len(requests) >= num_prompts:
        break

if len(requests) != num_prompts:
    raise SystemExit(f"selected {len(requests)} requests, expected {num_prompts}")

metadata = {
    "source": str(dataset_path),
    "selection": "vllm_sharegpt_sampler_compatible_seed_42",
    "seed": 42,
    "num_prompts": num_prompts,
    "output_len": output_len,
    "total_prompt_tokens": sum(prompt_lengths),
    "avg_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
    "min_prompt_tokens": min(prompt_lengths),
    "max_prompt_tokens": max(prompt_lengths),
}
fixture = {"metadata": metadata, "requests": requests}
fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
custom_jsonl_path.write_text("\n".join(custom_lines) + "\n", encoding="utf-8")
meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/fixture_build.log"
}

run_pilot_trace() {
  wait_for_clean_gpu "pilot_trace"
  reset_reserved_pool "${PILOT_DIR}/reserved_pool_reset_before.json"
  rm -f \
    "${PILOT_TRACE}" \
    "${PILOT_RESULT}" \
    "${PILOT_DIR}/bench.log" \
    "${PILOT_DIR}/bench_status.txt" \
    "${PILOT_DIR}/server.log" \
    "${PILOT_DIR}/server.pid"
  local port="${PORT_BASE}"
  cleanup_port "${port}"
  echo "=== pilot trace: serving no-offload oracle source max_num_seqs=${PILOT_MAX_NUM_SEQS} ===" | tee -a "${RUN_ROOT}/run.log"
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
  ) > "${PILOT_DIR}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${PILOT_DIR}/server.pid"

  if ! wait_for_server "${port}" "${server_pid}"; then
    echo "pilot server failed" | tee -a "${RUN_ROOT}/run.log"
    stop_process_tree "${server_pid}"
    reset_reserved_pool "${PILOT_DIR}/reserved_pool_reset_after_failure.json"
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
      --result-dir "${PILOT_DIR}" \
      --result-filename "$(basename "${PILOT_RESULT}")"
  ) > "${PILOT_DIR}/bench.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${PILOT_DIR}/bench_status.txt"
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  reset_reserved_pool "${PILOT_DIR}/reserved_pool_reset_after.json"
  wait_for_clean_gpu "after_pilot_trace" || true
  if [[ "${rc}" -ne 0 ]]; then
    tail -n 120 "${PILOT_DIR}/bench.log" >&2
    exit "${rc}"
  fi
}

build_zero_hit_plan() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${PILOT_TRACE}" "${ZERO_HIT_PLAN}" \
      "${NUM_LAYERS}" "${NUM_EXPERTS}" \
      "${MAX_ZERO_HIT_EXPERTS_PER_LAYER}" \
      "${INCLUDE_ACTIVE_EXPERTS_PER_LAYER}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

trace_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
num_layers = int(sys.argv[3])
num_experts = int(sys.argv[4])
max_zero_hit_per_layer = int(sys.argv[5])
include_active_per_layer = int(sys.argv[6])

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

active_by_layer = {layer: set() for layer in range(num_layers)}
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
    active_experts = record.get("active_experts")
    if not isinstance(active_experts, list):
        continue
    trace_records += 1
    for expert_id in active_experts:
        try:
            expert = int(expert_id)
        except (TypeError, ValueError):
            continue
        if 0 <= expert < num_experts:
            active_by_layer[layer_id].add(expert)

offload_plan = []
used_counts = {}
zero_counts = {}
selected_by_layer = {}
for layer_id in range(num_layers):
    used = sorted(active_by_layer[layer_id])
    zero_candidates = [expert_id for expert_id in range(num_experts)
                       if expert_id not in active_by_layer[layer_id]]
    zero = (zero_candidates[:max_zero_hit_per_layer]
            if max_zero_hit_per_layer > 0 else zero_candidates)
    active_selected = used[:max(include_active_per_layer, 0)]
    selected = zero + active_selected
    used_counts[layer_id] = len(used)
    zero_counts[layer_id] = len(zero_candidates)
    selected_by_layer[layer_id] = selected
    for expert_id in zero:
        offload_plan.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "reason": "strict_zero_hit_all_selected_requests",
        })
    for expert_id in active_selected:
        offload_plan.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "reason": "remote_read_active_hit_validation",
        })

metadata = {
    "policy": "exact_used_oracle_v1",
    "oracle_scope": "serving_bench_sharegpt_16x512_ignore_eos_temperature0_seed0",
    "routing_trace": str(trace_path),
    "trace_records": trace_records,
    "num_layers": num_layers,
    "num_experts": num_experts,
    "used_counts_by_layer": used_counts,
    "zero_counts_by_layer": zero_counts,
    "selected_by_layer": selected_by_layer,
    "total_zero_layer_experts": sum(
        1 for item in offload_plan
        if item["reason"] == "strict_zero_hit_all_selected_requests"
    ),
    "total_selected_layer_experts": len(offload_plan),
    "max_zero_hit_experts_per_layer": max_zero_hit_per_layer,
    "include_active_experts_per_layer": include_active_per_layer,
    "estimated_release_bytes_at_4_5mib_each": len(offload_plan) * 4718592,
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
  local port=$((PORT_BASE + idx))
  local name="${CUSTOM_CASE_PREFIX}_s${max_num_seqs}"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  rm -f \
    "${dir}/bench.log" \
    "${dir}/bench_result.json" \
    "${dir}/bench_status.txt" \
    "${dir}/dmon_gpm.csv" \
    "${dir}/dmon_pcie.csv" \
    "${dir}/init_trace.jsonl" \
    "${dir}/phase_f_route_log_after.txt" \
    "${dir}/server.log" \
    "${dir}/server.pid"

  cleanup_port "${port}"
  wait_for_clean_gpu "${name}"
  reset_reserved_pool "${dir}/reserved_pool_reset_before.json"
  if [[ "${CUSTOM_ACCESS_MODE}" == "restore" ]]; then
    prealloc_reserved_pool "${dir}/reserved_prealloc_before_model.json"
  fi

  echo "=== ${name}: max_num_seqs=${max_num_seqs} kv=${KV_CACHE_MEMORY_BYTES} skip_active_topk=1 ===" | tee -a "${RUN_ROOT}/run.log"

  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
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
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${ZERO_HIT_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE="${PLAN_RESIDENCY_TRACE_ENABLE}" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS="${PLAN_RESIDENCY_TRACE_MAX_EXPERTS}" \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
      VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
      VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
      VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${ZERO_HIT_PLAN}" \
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
    reset_reserved_pool "${dir}/reserved_pool_reset_after_failure.json"
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
  cat /proc/driver/nvidia-uvm/phase_f_route_log > "${dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  reset_reserved_pool "${dir}/reserved_pool_reset_after.json"
  wait_for_clean_gpu "after_${name}" || true
  return 0
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RUN_ROOT}" "${CUSTOM_CASE_PREFIX}" \
      "${CUSTOM_ACCESS_MODE}" "${NUM_PROMPTS}" "${OUTPUT_LEN}" \
      "${REQUEST_RATE}" "${MAX_MODEL_LEN}" "${MAX_NUM_BATCHED_TOKENS}" \
      "${GPU_MEMORY_UTILIZATION}" "${KV_CACHE_MEMORY_BYTES}" \
      "${CUSTOM_SKIP_ACTIVE_TOPK}" "${PLAN_RESIDENCY_TRACE_ENABLE}" <<'PY'
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
case_prefix = sys.argv[2]
access_mode = sys.argv[3]
num_prompts = int(sys.argv[4])
output_len = int(sys.argv[5])
request_rate = sys.argv[6]
max_model_len = int(sys.argv[7])
max_num_batched_tokens = int(sys.argv[8])
gpu_memory_utilization = float(sys.argv[9])
kv_cache_memory_bytes = int(sys.argv[10])
skip_active_topk = int(sys.argv[11])
plan_residency_trace_enable = int(sys.argv[12])

def read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))

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

def parse_float_metric(text: str, name: str):
    m = re.search(rf"{re.escape(name)}:\s+([0-9.]+)", text)
    return float(m.group(1)) if m else None

def parse_int_metric(text: str, name: str):
    value = parse_float_metric(text, name)
    return int(value) if value is not None else None

def parse_bench_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "successful_requests": parse_int_metric(text, "Successful requests"),
        "failed_requests": parse_int_metric(text, "Failed requests"),
        "benchmark_duration_s": parse_float_metric(text, "Benchmark duration (s)"),
        "total_input_tokens": parse_int_metric(text, "Total input tokens"),
        "total_generated_tokens": parse_int_metric(text, "Total generated tokens"),
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

def parse_server_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    run_values = [int(x) for x in re.findall(r"Running:\s+([0-9]+)\s+reqs", text)]
    wait_values = [int(x) for x in re.findall(r"Waiting:\s+([0-9]+)\s+reqs", text)]
    kv_tokens = None
    m = re.search(r"GPU KV cache size:\s+([0-9,]+)\s+tokens", text)
    if m:
        kv_tokens = int(m.group(1).replace(",", ""))
    available_kv_gib = None
    m = re.search(r"Available KV cache memory:\s+([0-9.]+)\s+GiB", text)
    if m:
        available_kv_gib = float(m.group(1))
    model_load_gib = None
    m = re.search(r"Model loading took\s+([0-9.]+)\s+GiB memory", text)
    if m:
        model_load_gib = float(m.group(1))
    return {
        "server_running_max": max(run_values) if run_values else None,
        "server_waiting_max": max(wait_values) if wait_values else None,
        "server_running_samples": run_values,
        "gpu_kv_cache_tokens": kv_tokens,
        "available_kv_cache_gib": available_kv_gib,
        "model_load_gib": model_load_gib,
        "has_xid": "Xid" in text,
        "has_mmu_fault": "MMU" in text or "FAULT_PDE" in text,
        "has_uvm_fatal": "UVM fatal" in text,
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
    nz_rx = [value for value in rx if value > 0]
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
            except ValueError:
                continue
    if not rows:
        return {"samples": 0}
    raw_rx = [row["pcirxraw"] for row in rows]
    raw_tx = [row["pcitxraw"] for row in rows]
    return {
        "samples": len(rows),
        "sm_mean_pct": statistics.mean(row["sm"] for row in rows),
        "sm_max_pct": max(row["sm"] for row in rows),
        "mem_mean_pct": statistics.mean(row["mem"] for row in rows),
        "mem_max_pct": max(row["mem"] for row in rows),
        "pcirx_mean_MiBps": statistics.mean(row["pcirx"] for row in rows),
        "pcirx_max_MiBps": max(row["pcirx"] for row in rows),
        "pcirxraw_delta_gib": (max(raw_rx) - min(raw_rx)) / (1024 ** 3),
        "pcitx_mean_MiBps": statistics.mean(row["pcitx"] for row in rows),
        "pcitx_max_MiBps": max(row["pcitx"] for row in rows),
        "pcitxraw_delta_gib": (max(raw_tx) - min(raw_tx)) / (1024 ** 3),
        "hmmiss_mean_pct": statistics.mean(row["hmmiss"] for row in rows),
    }

def output_hash_from_bench_json(path: Path):
    payload = read_json(path)
    generated = payload.get("generated_texts")
    if not isinstance(generated, list):
        return None
    return hashlib.sha256("\0".join(map(str, generated)).encode()).hexdigest()

def output_hash_from_pilot(path: Path):
    payload = read_json(path)
    generated_texts = payload.get("generated_texts")
    if isinstance(generated_texts, list):
        return hashlib.sha256(
            "\0".join(map(str, generated_texts)).encode()
        ).hexdigest()
    results = payload.get("request_results")
    if not isinstance(results, list):
        return None
    generated = [str(item.get("generated") or "") for item in results]
    return hashlib.sha256("\0".join(generated).encode()).hexdigest()

def init_release_summary(path: Path):
    records = read_jsonl(path)
    issued = [
        item for item in records
        if str(item.get("action") or "").endswith("_issued")
    ]
    success = [item for item in issued if item.get("success") is True]
    remote_prepare = [
        item for item in records
        if item.get("action") == "init_weight_remote_read_prepare_issued"
    ]
    reserved_runtime_actions = [
        item for item in records
        if any(
            marker in str(item.get("action") or "")
            for marker in ("reserved_prefetch", "slot_remap", "restore")
        )
    ]
    residency_by_marker = {}
    for item in records:
        if item.get("action") != "init_weight_plan_residency_snapshot":
            continue
        marker = str(item.get("marker") or "unknown")
        aggregate = residency_by_marker.setdefault(
            marker,
            {
                "records": 0,
                "query_ok_records": 0,
                "query_statuses": [],
                "queried_bytes": 0,
                "gpu_bytes": 0,
                "cpu_bytes": 0,
                "unknown_bytes": 0,
            },
        )
        aggregate["records"] += 1
        aggregate["query_ok_records"] += int(
            bool(item.get("cold_plan_query_ok"))
        )
        status = int(item.get("cold_plan_query_status") or 0)
        if status not in aggregate["query_statuses"]:
            aggregate["query_statuses"].append(status)
        for field in ("queried_bytes", "gpu_bytes", "cpu_bytes", "unknown_bytes"):
            aggregate[field] += int(item.get(f"cold_plan_{field}") or 0)
    return {
        "init_trace_records": len(records),
        "init_release_issued_records": len(issued),
        "init_release_success_records": len(success),
        "init_release_failed_records": len(issued) - len(success),
        "init_release_bytes": sum(int(item.get("bytes") or 0) for item in success),
        "remote_prepare_records": len(remote_prepare),
        "remote_prepare_bytes": sum(
            int(item.get("bytes") or 0) for item in remote_prepare
        ),
        "reserved_restore_action_records": len(reserved_runtime_actions),
        "plan_residency_by_marker": residency_by_marker,
        "envelope_regions": sum(
            len(item.get("envelope_records") or []) for item in success
        ),
    }

fixture = read_json(root / "fixture_metadata.json")
plan = read_json(root / "strict_zero_hit_all_experts_plan.json")
pilot = read_json(root / "pilot_trace" / "result.json")
if not pilot:
    pilot = read_json(root / "pilot_trace" / "bench_result.json")
pilot_hash = output_hash_from_pilot(root / "pilot_trace" / "result.json")
if not pilot_hash:
    pilot_hash = output_hash_from_pilot(root / "pilot_trace" / "bench_result.json")
pilot_tokens = int(pilot.get("total_output_tokens") or 0)
if not pilot_tokens:
    pilot_tokens = sum(
        int(item.get("output_tokens") or 0)
        for item in (pilot.get("request_results") or [])
    )

cases = {}
for case_dir in sorted(root.glob(f"{case_prefix}_s*")):
    if not case_dir.is_dir():
        continue
    m = re.fullmatch(rf"{re.escape(case_prefix)}_s([0-9]+)", case_dir.name)
    if not m:
        continue
    status_path = case_dir / "bench_status.txt"
    case = {
        "max_num_seqs": int(m.group(1)),
        "bench_status": status_path.read_text(encoding="utf-8").strip()
        if status_path.exists()
        else "missing",
        "output_sha256": output_hash_from_bench_json(case_dir / "bench_result.json"),
    }
    case.update(parse_bench_log(case_dir / "bench.log"))
    case.update(parse_server_log(case_dir / "server.log"))
    case.update(init_release_summary(case_dir / "init_trace.jsonl"))
    case["dmon_pcie"] = parse_dmon_pcie(case_dir / "dmon_pcie.csv")
    case["dmon_gpm"] = parse_dmon_gpm(case_dir / "dmon_gpm.csv")
    cases[case_dir.name] = case

ordered = sorted(cases.values(), key=lambda item: item["max_num_seqs"])
baseline = ordered[0] if ordered else None
for case in ordered:
    if baseline and baseline.get("output_tok_s") and case.get("output_tok_s"):
        case["delta_vs_s1_output_tok_s_pct"] = (
            (case["output_tok_s"] - baseline["output_tok_s"])
            / baseline["output_tok_s"]
            * 100.0
        )
    if baseline and baseline.get("mean_tpot_ms") and case.get("mean_tpot_ms"):
        case["delta_vs_s1_mean_tpot_pct"] = (
            (case["mean_tpot_ms"] - baseline["mean_tpot_ms"])
            / baseline["mean_tpot_ms"]
            * 100.0
        )
    case["matches_pilot_output_hash"] = (
        bool(case.get("output_sha256"))
        and bool(pilot_hash)
        and case["output_sha256"] == pilot_hash
    )

summary = {
    "run_root": str(root),
    "config": {
        "model": "Qwen3-30B-A3B-FP8",
        "num_prompts": num_prompts,
        "output_len": output_len,
        "request_rate": request_rate,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "skip_active_topk": skip_active_topk,
        "plan_residency_trace_enable": plan_residency_trace_enable,
        "access_mode": access_mode,
        "case_prefix": case_prefix,
    },
    "fixture": fixture,
    "pilot": {
        "completed_requests": pilot.get("completed_requests", pilot.get("completed")),
        "total_output_tokens": pilot_tokens,
        "output_sha256": pilot_hash,
    },
    "plan": plan.get("metadata"),
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

require_environment
if [[ "${SUMMARY_ONLY}" == "1" ]]; then
  write_summary | tee "${SUMMARY}"
  echo "summary: ${SUMMARY}"
  exit 0
fi
if [[ "${REUSE_PREP}" == "1" && -s "${FIXTURE}" && -s "${CUSTOM_JSONL}" && -s "${PILOT_TRACE}" && -s "${PILOT_RESULT}" && -s "${ZERO_HIT_PLAN}" ]]; then
  echo "reusing existing fixture, pilot trace, and zero-hit plan from ${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
else
  generate_fixture
  run_pilot_trace
  build_zero_hit_plan
fi

idx=0
for max_num_seqs in ${MAX_NUM_SEQS_LIST}; do
  idx=$((idx + 1))
  run_custom_case "${max_num_seqs}" "${idx}" || true
  write_summary | tee "${RUN_ROOT}/summary.latest.json"
done

write_summary | tee "${SUMMARY}"
echo "summary: ${SUMMARY}"
