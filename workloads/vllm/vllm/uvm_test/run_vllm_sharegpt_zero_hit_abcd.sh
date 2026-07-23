#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_sharegpt_zero_hit_abcd_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

NUM_REQUESTS="${NUM_REQUESTS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
SHAREGPT_SEED="${SHAREGPT_SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"
REUSE_PREP="${REUSE_PREP:-0}"
RUN_CASES="${RUN_CASES:-A,B,C,D}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
MANUAL_MODE_PATH="${PARAM_DIR}/uvm_manual_migration_fault_mode"
MANUAL_POOL_PATH="${PARAM_DIR}/uvm_manual_migration_fault_pool_mask"
MANUAL_LOG_PATH="${PARAM_DIR}/uvm_manual_migration_fault_log_enable"
ORIGINAL_MANUAL_MODE="$(cat "${MANUAL_MODE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_MANUAL_POOL="$(cat "${MANUAL_POOL_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_MANUAL_LOG="$(cat "${MANUAL_LOG_PATH}" 2>/dev/null || echo 0)"

FIXTURE="${RUN_ROOT}/sharegpt_32x64_fixture.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
EMPTY_PLAN="${RUN_ROOT}/empty_plan.json"
ZERO_HIT_PLAN="${RUN_ROOT}/strict_zero_hit_plan.json"
PILOT_DIR="${RUN_ROOT}/pilot"
PILOT_TRACE="${PILOT_DIR}/routing_trace.jsonl"
PILOT_RESULT="${PILOT_DIR}/result.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}" "${PILOT_DIR}"

set_param() {
  local path="$1"
  local value="$2"
  if [[ -e "${path}" ]]; then
    printf '%s\n' "${value}" | sudo -n tee "${path}" >/dev/null
  fi
}

restore_manual_fault_params() {
  set_param "${MANUAL_POOL_PATH}" "${ORIGINAL_MANUAL_POOL}" || true
  set_param "${MANUAL_LOG_PATH}" "${ORIGINAL_MANUAL_LOG}" || true
  set_param "${MANUAL_MODE_PATH}" "${ORIGINAL_MANUAL_MODE}" || true
}
trap restore_manual_fault_params EXIT

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

require_clean_gpu() {
  local label="$1"
  local used
  used="$(gpu_used_mib || echo 999999)"
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || [[ "${used}" -gt 1024 ]]; then
    echo "GPU is not clean before ${label}: ${used} MiB" >&2
    nvidia-smi >&2 || true
    exit 3
  fi
}

require_environment() {
  local built loaded pte_enable pte_max
  built="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")"
  loaded="$(cat /sys/module/nvidia_uvm/srcversion)"
  pte_enable="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)"
  pte_max="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes)"
  [[ "${loaded}" == "${built}" ]] || {
    echo "loaded UVM ${loaded} != built UVM ${built}" >&2
    exit 2
  }
  [[ "${pte_enable}" == "1" ]] || {
    echo "uvm_moe_slot_pte_remap_enable=${pte_enable}, expected 1" >&2
    exit 2
  }
  [[ "${pte_max}" -ge "${RESERVED_SLOT_BYTES}" ]] || {
    echo "uvm_moe_slot_pte_remap_max_bytes=${pte_max}, expected >= ${RESERVED_SLOT_BYTES}" >&2
    exit 2
  }
  [[ -f "${DATASET_PATH}" ]] || {
    echo "ShareGPT dataset not found: ${DATASET_PATH}" >&2
    exit 2
  }
  [[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
    echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
    exit 2
  }
}

reset_reserved_pool() {
  restore_manual_fault_params
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True),
                 indent=2, sort_keys=True))
PY
  ) > "${RUN_ROOT}/reserved_pool_reset_$1.json" 2>&1
}

query_reserved_pool() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_query

print(json.dumps(driver_expert_reserved_pool_query(prefer_existing=True),
                 indent=2, sort_keys=True))
PY
  ) > "$1" 2>&1 || true
}

prealloc_reserved_pool() {
  local output="$1"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RESERVED_SLOT_COUNT}" "${RESERVED_SLOT_BYTES}" <<'PY'
import json
import sys

from vllm.device_allocator.uvm import driver_expert_reserved_pool_prealloc

slot_count = int(sys.argv[1])
slot_bytes = int(sys.argv[2])
slots = [
    {"pool_id": 0, "slot_id": slot_id, "slot_bytes": slot_bytes}
    for slot_id in range(slot_count)
]
result = driver_expert_reserved_pool_prealloc(
    slots,
    device=0,
    prefer_existing=True,
)
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok") or int(result.get("success_count") or 0) < slot_count:
    raise SystemExit(2)
PY
  ) > "${output}" 2>&1
}

generate_fixture() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET_PATH}" "${MODEL}" "${FIXTURE}" \
      "${FIXTURE_META}" "${NUM_REQUESTS}" "${OUTPUT_TOKENS}" \
      "${MAX_MODEL_LEN}" "${SHAREGPT_SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
meta_path = Path(sys.argv[4])
num_requests = int(sys.argv[5])
output_tokens = int(sys.argv[6])
max_model_len = int(sys.argv[7])
seed = int(sys.argv[8])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
data = json.loads(dataset_path.read_text(encoding="utf-8"))
random.Random(seed).shuffle(data)

requests = []
prompt_lengths = []
for item in data:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        continue
    prompt = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in ("human", "user"):
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break
    if not prompt:
        continue
    token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if token_len <= 0 or token_len + output_tokens > max_model_len:
        continue
    requests.append({
        "request": len(requests),
        "prompt": prompt,
        "max_tokens": output_tokens,
        "prompt_tokens": token_len,
    })
    prompt_lengths.append(token_len)
    if len(requests) >= num_requests:
        break

if len(requests) != num_requests:
    raise SystemExit(f"selected {len(requests)} requests, expected {num_requests}")

metadata = {
    "source": "ShareGPT_V3_unfiltered_cleaned_split",
    "selection": "shuffled_valid_prompts",
    "seed": seed,
    "num_requests": num_requests,
    "output_tokens": output_tokens,
    "max_model_len": max_model_len,
    "min_prompt_tokens": min(prompt_lengths),
    "max_prompt_tokens": max(prompt_lengths),
    "avg_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
}
fixture = {"metadata": metadata, "requests": requests}
fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  )

  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${EMPTY_PLAN}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "metadata": {"policy": "exact_used_oracle_v1", "kind": "empty_control"},
    "offload_plan": [],
    "prefetch_plan": [],
    "offload_plan_records": 0,
    "prefetch_plan_records": 0,
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
PY
  )
}

run_generate() {
  local result="$1"
  local progress="$2"
  shift 2
  (
    cd "${VLLM_ROOT}"
    env "$@" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${SCRIPT_DIR}/vllm_generate_sequence.py" \
      --model "${MODEL}" \
      --request-fixture-json "${FIXTURE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --seed 0 \
      --temperature 0 \
      --output-json "${result}" \
      --progress-jsonl "${progress}"
  )
}

run_pilot() {
  require_clean_gpu pilot
  reset_reserved_pool pilot
  rm -f "${PILOT_TRACE}" "${PILOT_RESULT}" "${PILOT_DIR}/progress.jsonl"
  set +e
  run_generate "${PILOT_RESULT}" "${PILOT_DIR}/progress.jsonl" \
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
    > "${PILOT_DIR}/run.log" 2>&1
  local rc=$?
  set -e
  [[ "${rc}" -eq 0 ]] || {
    echo "pilot failed with rc=${rc}" >&2
    tail -n 100 "${PILOT_DIR}/run.log" >&2
    exit "${rc}"
  }

  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" "${SCRIPT_DIR}/build_cold_offload_plan_from_routing_trace.py" \
      --routing-trace "${PILOT_TRACE}" \
      --output "${ZERO_HIT_PLAN}" \
      --num-experts "${NUM_EXPERTS}" \
      --max-layers "${NUM_LAYERS}" \
      --max-experts-per-layer "${NUM_EXPERTS}"
  ) > "${RUN_ROOT}/plan_build.log"

  "${VLLM_ROOT}/${VLLM_PY}" - "${ZERO_HIT_PLAN}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload.setdefault("metadata", {})["policy"] = "exact_used_oracle_v1"
payload["metadata"]["oracle_scope"] = "exact_sharegpt_fixture_greedy_seed_0"
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
PY
}

run_case() {
  local name="$1"
  local plan="$2"
  local init_offload="$3"
  local runtime_enable="$4"
  local skip_active_topk="$5"
  local dir="${RUN_ROOT}/${name}"
  local result="${dir}/result.json"
  local progress="${dir}/progress.jsonl"
  local init_trace="${dir}/init_trace.jsonl"
  local prealloc_trace="${dir}/post_stage_m_reserved_prealloc.jsonl"
  local run_log="${dir}/run.log"

  mkdir -p "${dir}"
  require_clean_gpu "${name}"
  reset_reserved_pool "${name}"
  rm -f "${result}" "${progress}" "${init_trace}" "${prealloc_trace}"

  local executor=cpp_batch
  local reserved_after=0
  if [[ "${runtime_enable}" == "1" ]]; then
    executor=expert_reserved_regions_async
    reserved_after=1
    prealloc_reserved_pool "${dir}/reserved_prealloc_before_model.json"
  fi

  echo "running ${name}: plan=$(basename "${plan}") init=${init_offload} runtime=${runtime_enable} skip_topk=${skip_active_topk}"
  set +e
  run_generate "${result}" "${progress}" \
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
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE="${init_offload}" \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${plan}" \
    VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
    VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
    VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
    VLLM_UVM_INIT_WEIGHT_SYNC=1 \
    VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${init_trace}" \
    VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE="${runtime_enable}" \
    VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
    VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
    VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${plan}" \
    VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
    VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
    VLLM_UVM_WEIGHT_EXECUTOR="${executor}" \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
    VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
    VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
    VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP="${reserved_after}" \
    VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M="${reserved_after}" \
    VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE=1 \
    VLLM_UVM_EXPERT_RESERVED_POST_STAGE_M_PREALLOC_FILE="${prealloc_trace}" \
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
    VLLM_UVM_EXPERT_RESERVED_SKIP_ACTIVE_TOPK="${skip_active_topk}" \
    VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
    > "${run_log}" 2>&1
  local rc=$?
  set -e

  restore_manual_fault_params
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  cat /proc/driver/nvidia-uvm/phase_f_route_log > "${dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  query_reserved_pool "${dir}/reserved_pool_after.json"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits \
    > "${dir}/gpu_after.txt" || true

  if [[ "${rc}" -ne 0 ]]; then
    echo "${name} failed with rc=${rc}" >&2
    tail -n 120 "${run_log}" >&2
    exit "${rc}"
  fi
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" <<'PY'
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
cases = [
    "A_baseline",
    "B_empty_runtime",
    "C_zero_hit_runtime",
    "D_zero_hit_skip_topk",
]
if (root / "E_empty_skip_topk" / "result.json").exists():
    cases.append("E_empty_skip_topk")

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(item, dict):
            records.append(item)
    return records

def result_metrics(result: dict) -> dict:
    requests = result.get("request_results") or []
    seconds = [float(item.get("generate_seconds") or 0.0) for item in requests]
    tokens = sum(int(item.get("output_tokens") or 0) for item in requests)
    total = float(result.get("total_generate_seconds") or sum(seconds) or 0.0)
    outputs = [str(item.get("generated") or "") for item in requests]
    output_hash = hashlib.sha256("\0".join(outputs).encode()).hexdigest()
    return {
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "load_seconds": result.get("load_seconds"),
        "total_generate_seconds": total,
        "avg_request_seconds": sum(seconds) / len(seconds) if seconds else None,
        "p50_request_seconds": statistics.median(seconds) if seconds else None,
        "tokens": tokens,
        "tokens_per_second": tokens / total if total else None,
        "request_seconds": seconds,
        "output_sha256": output_hash,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
    }

def route_counts(case_dir: Path, progress: list[dict]) -> dict:
    begin_times = [float(item["timestamp"]) for item in progress
                   if item.get("action") == "request_begin" and item.get("timestamp")]
    end_times = [float(item["timestamp"]) for item in progress
                 if item.get("action") == "request_end" and item.get("timestamp")]
    start = min(begin_times) if begin_times else 0.0
    end = max(end_times) if end_times else float("inf")
    counts = Counter()
    route_path = case_dir / "phase_f_route_log_after.txt"
    pattern = re.compile(r"^\[([0-9.]+)\]\s+(\S+)")
    if route_path.exists():
        for line in route_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = pattern.match(line)
            if not match:
                continue
            timestamp = float(match.group(1))
            if start <= timestamp <= end:
                counts[match.group(2)] += 1
    return {
        "request_begin": start,
        "request_end": end if end != float("inf") else None,
        "manual_migration_fault": sum(value for key, value in counts.items()
                                      if "manual_migration_fault" in key),
        "reserved_prefetch": sum(value for key, value in counts.items()
                                  if "reserved_prefetch" in key),
        "moe_slot_remap": sum(value for key, value in counts.items()
                               if "moe_slot" in key or "moe_pte" in key),
        "events": dict(sorted(counts.items())),
    }

case_summaries = {}
for name in cases:
    case_dir = root / name
    result = read_json(case_dir / "result.json")
    progress = read_jsonl(case_dir / "progress.jsonl")
    init_records = read_jsonl(case_dir / "init_trace.jsonl")
    issued = [item for item in init_records
              if str(item.get("action") or "").endswith("_issued")]
    successful = [item for item in issued if item.get("success") is True]
    envelope_ids = [driver_id for item in successful
                    for driver_id in item.get("release_driver_region_ids") or []]
    run_log = (case_dir / "run.log").read_text(encoding="utf-8", errors="replace") \
        if (case_dir / "run.log").exists() else ""
    stage_match = re.search(r"Stage M module loop end issued_modules=\d+ elapsed=([0-9.]+)s", run_log)
    case_summaries[name] = {
        **result_metrics(result),
        "returncode": int((case_dir / "returncode.txt").read_text().strip()),
        "stage_m_seconds": float(stage_match.group(1)) if stage_match else 0.0,
        "init_release_records": len(successful),
        "init_release_bytes": sum(int(item.get("bytes") or 0) for item in successful),
        "envelope_regions": len(envelope_ids),
        "unique_envelope_regions": len(set(envelope_ids)),
        "failed_issued_records": len(issued) - len(successful),
        "route": route_counts(case_dir, progress),
    }

plan = read_json(root / "strict_zero_hit_plan.json")
selected = Counter(int(item["layer_id"]) for item in plan.get("offload_plan") or [])
baseline = case_summaries["A_baseline"]

def delta(lhs: str, rhs: str) -> dict:
    left = case_summaries[lhs]
    right = case_summaries[rhs]
    return {
        "total_seconds": left["total_generate_seconds"] - right["total_generate_seconds"],
        "total_ratio": left["total_generate_seconds"] / right["total_generate_seconds"],
        "avg_request_ratio": left["avg_request_seconds"] / right["avg_request_seconds"],
        "tokens_per_second_ratio": left["tokens_per_second"] / right["tokens_per_second"],
    }

fixture_bytes = (root / "sharegpt_32x64_fixture.json").read_bytes()
comparisons = {
    "B_minus_A_reserved_framework": delta("B_empty_runtime", "A_baseline"),
    "D_minus_A_zero_hit_skip_vs_baseline": delta("D_zero_hit_skip_topk", "A_baseline"),
    "C_minus_D_runtime_gate": delta("C_zero_hit_runtime", "D_zero_hit_skip_topk"),
    "C_minus_B_zero_hit_vs_empty_runtime": delta("C_zero_hit_runtime", "B_empty_runtime"),
    "C_minus_A_production_zero_hit": delta("C_zero_hit_runtime", "A_baseline"),
}
if "E_empty_skip_topk" in case_summaries:
    comparisons.update({
        "E_minus_A_slots_and_reserved_outer": delta("E_empty_skip_topk", "A_baseline"),
        "B_minus_E_empty_topk_check": delta("B_empty_runtime", "E_empty_skip_topk"),
        "D_minus_E_release_only": delta("D_zero_hit_skip_topk", "E_empty_skip_topk"),
    })

summary = {
    "run_root": str(root),
    "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
    "strict_zero_hit_plan": {
        "records": len(plan.get("offload_plan") or []),
        "layers": len(selected),
        "per_layer_histogram": dict(sorted(Counter(selected.values()).items())),
        "max_per_layer": max(selected.values(), default=0),
    },
    "cases": case_summaries,
    "comparisons": comparisons,
    "outputs_match_baseline": all(
        item["output_sha256"] == baseline["output_sha256"]
        for item in case_summaries.values()
    ),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

require_environment
require_clean_gpu start
if [[ "${REUSE_PREP}" == "0" ]]; then
  generate_fixture
else
  [[ -s "${FIXTURE}" && -s "${EMPTY_PLAN}" && -s "${ZERO_HIT_PLAN}" ]] || {
    echo "REUSE_PREP=1 requires existing fixture and plans in ${RUN_ROOT}" >&2
    exit 2
  }
fi
cat > "${RUN_ROOT}/env.txt" <<EOF
model=${MODEL}
dataset=${DATASET_PATH}
num_requests=${NUM_REQUESTS}
output_tokens=${OUTPUT_TOKENS}
max_model_len=${MAX_MODEL_LEN}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
expert_envelope_bytes=${EXPERT_ENVELOPE_BYTES}
loaded_uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)
built_uvm_srcversion=$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")
EOF

RUN_STARTED="$(date --iso-8601=seconds)"
if [[ "${REUSE_PREP}" == "0" ]]; then
  run_pilot
fi
if [[ ",${RUN_CASES}," == *,A,* ]]; then
  run_case A_baseline "${EMPTY_PLAN}" 0 0 0
fi
if [[ ",${RUN_CASES}," == *,B,* ]]; then
  run_case B_empty_runtime "${EMPTY_PLAN}" 0 1 0
fi
if [[ ",${RUN_CASES}," == *,C,* ]]; then
  run_case C_zero_hit_runtime "${ZERO_HIT_PLAN}" 1 1 0
fi
if [[ ",${RUN_CASES}," == *,D,* ]]; then
  run_case D_zero_hit_skip_topk "${ZERO_HIT_PLAN}" 1 1 1
fi
if [[ ",${RUN_CASES}," == *,E,* ]]; then
  run_case E_empty_skip_topk "${EMPTY_PLAN}" 0 1 1
fi
write_summary
journalctl -k --since "${RUN_STARTED}" --no-pager > "${RUN_ROOT}/kernel_since.log" 2>/dev/null || true
restore_manual_fault_params
reset_reserved_pool final
echo "run_root=${RUN_ROOT}"
