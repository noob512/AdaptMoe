#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_long_context_baseline_vs_cold_kv_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-21840}"
LONG_PROMPT_TOKENS="${LONG_PROMPT_TOKENS:-20000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MECH_KV_CACHE_MEMORY_BYTES="${MECH_KV_CACHE_MEMORY_BYTES:-2147483648}"
PILOT_KV_CACHE_MEMORY_BYTES="${PILOT_KV_CACHE_MEMORY_BYTES:-${MECH_KV_CACHE_MEMORY_BYTES}}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
NUM_LAYERS="${NUM_LAYERS:-48}"
PLAN_MAX_EXPERTS_PER_LAYER="${PLAN_MAX_EXPERTS_PER_LAYER:-5}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"
MECH_CASE_DIR_NAME="${MECH_CASE_DIR_NAME:-cold_offload_large_kv}"
SKIP_ACTIVE_TOPK="${SKIP_ACTIVE_TOPK:-0}"

RUN_CASES="${RUN_CASES:-baseline,pilot,mechanism}"
REUSE_PREP="${REUSE_PREP:-0}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
MANUAL_MODE_PATH="${PARAM_DIR}/uvm_manual_migration_fault_mode"
MANUAL_POOL_PATH="${PARAM_DIR}/uvm_manual_migration_fault_pool_mask"
MANUAL_LOG_PATH="${PARAM_DIR}/uvm_manual_migration_fault_log_enable"
ORIGINAL_MANUAL_MODE="$(cat "${MANUAL_MODE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_MANUAL_POOL="$(cat "${MANUAL_POOL_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_MANUAL_LOG="$(cat "${MANUAL_LOG_PATH}" 2>/dev/null || echo 0)"

FIXTURE="${RUN_ROOT}/long_prompt_fixture.json"
FIXTURE_META="${RUN_ROOT}/long_prompt_fixture_meta.json"
PILOT_DIR="${RUN_ROOT}/pilot_managed_no_offload"
BASELINE_DIR="${RUN_ROOT}/baseline_cudamalloc_auto_kv"
MECH_DIR="${RUN_ROOT}/${MECH_CASE_DIR_NAME}"
PLAN="${RUN_ROOT}/cold_offload_plan.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}" "${PILOT_DIR}" "${BASELINE_DIR}" "${MECH_DIR}"

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

snapshot_gpu() {
  local label="$1"
  {
    echo "== ${label} =="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits || true
  } > "${RUN_ROOT}/gpu_${label}.txt"
}

require_clean_gpu() {
  local label="$1"
  local used
  used="$(gpu_used_mib || echo 999999)"
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || [[ "${used}" -gt 1024 ]]; then
    echo "GPU is not clean before ${label}: ${used} MiB" >&2
    snapshot_gpu "blocked_${label}"
    exit 3
  fi
}

require_environment() {
  [[ -f "${MODEL}/config.json" ]] || {
    echo "model path not found: ${MODEL}" >&2
    exit 2
  }
  [[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
    echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
    exit 2
  }

  local built loaded pte_enable pte_max
  built="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
  loaded="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  pte_enable="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)"
  pte_max="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || echo 0)"
  {
    echo "built_uvm_srcversion=${built}"
    echo "loaded_uvm_srcversion=${loaded}"
    echo "uvm_moe_slot_pte_remap_enable=${pte_enable}"
    echo "uvm_moe_slot_pte_remap_max_bytes=${pte_max}"
  } > "${RUN_ROOT}/uvm_environment.txt"

  [[ -n "${built}" && "${loaded}" == "${built}" ]] || {
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
}

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

generate_long_fixture() {
  if [[ "${REUSE_PREP}" == "1" && -f "${FIXTURE}" ]]; then
    return
  fi
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${MODEL}" "${FIXTURE}" "${FIXTURE_META}" \
      "${LONG_PROMPT_TOKENS}" "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

model = sys.argv[1]
fixture_path = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
target_tokens = int(sys.argv[4])
output_tokens = int(sys.argv[5])
max_model_len = int(sys.argv[6])

if target_tokens + output_tokens > max_model_len:
    raise SystemExit(
        f"target prompt tokens + output tokens exceeds max_model_len: "
        f"{target_tokens}+{output_tokens}>{max_model_len}"
    )

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
unit = (
    "In a long-context systems experiment, compare scheduling pressure, "
    "KV cache residency, CUDA memory allocation, and expert routing decisions. "
)

def token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

low, high = 1, 1
while token_count(unit * high) < target_tokens:
    high *= 2
while low < high:
    mid = (low + high + 1) // 2
    if token_count(unit * mid) <= target_tokens:
        low = mid
    else:
        high = mid - 1

prompt = unit * low
while token_count(prompt + unit) <= target_tokens:
    prompt += unit
prompt_tokens = token_count(prompt)
payload = {
    "metadata": {
        "source": "run_vllm_long_context_baseline_vs_cold_kv.sh",
        "target_prompt_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "max_model_len": max_model_len,
    },
    "requests": [
        {
            "request": 0,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
        }
    ],
}
fixture_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
meta_path.write_text(json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
PY
  )
}

run_generate() {
  local result="$1"
  local progress="$2"
  shift 2
  env "$@" timeout "${TIMEOUT_SEC}s" "${VLLM_ROOT}/${VLLM_PY}" \
    "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${FIXTURE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${result}" \
    --progress-jsonl "${progress}"
}

run_generate_with_explicit_kv() {
  local result="$1"
  local progress="$2"
  local kv_bytes="$3"
  shift 3
  env "$@" timeout "${TIMEOUT_SEC}s" "${VLLM_ROOT}/${VLLM_PY}" \
    "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${FIXTURE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-memory-bytes "${kv_bytes}" \
    --output-json "${result}" \
    --progress-jsonl "${progress}"
}

run_baseline() {
  local dir="${BASELINE_DIR}"
  local result="${dir}/result.json"
  local progress="${dir}/progress.jsonl"
  local run_log="${dir}/run.log"
  require_clean_gpu "baseline"
  snapshot_gpu "before_baseline"
  set +e
  run_generate "${result}" "${progress}" \
    VLLM_USE_UVM=0 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
    VLLM_UVM_KV_CACHE_MANAGED=0 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=0 \
    > "${run_log}" 2>&1
  local rc=$?
  set -e
  echo "${rc}" > "${dir}/returncode.txt"
  snapshot_gpu "after_baseline"
}

run_pilot() {
  local dir="${PILOT_DIR}"
  local result="${dir}/result.json"
  local progress="${dir}/progress.jsonl"
  local run_log="${dir}/run.log"
  local routing_trace="${dir}/routing_trace.jsonl"
  if [[ "${REUSE_PREP}" == "1" && -f "${PLAN}" ]]; then
    return
  fi
  require_clean_gpu "pilot"
  snapshot_gpu "before_pilot"
  set +e
  run_generate_with_explicit_kv "${result}" "${progress}" "${PILOT_KV_CACHE_MEMORY_BYTES}" \
    VLLM_USE_UVM=0 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_KV_CACHE_MANAGED=1 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${routing_trace}" \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=0 \
    > "${run_log}" 2>&1
  local rc=$?
  set -e
  echo "${rc}" > "${dir}/returncode.txt"
  snapshot_gpu "after_pilot"

  if [[ "${rc}" == "0" ]]; then
    "${VLLM_ROOT}/${VLLM_PY}" "${SCRIPT_DIR}/build_cold_offload_plan_from_routing_trace.py" \
      --routing-trace "${routing_trace}" \
      --output "${PLAN}" \
      --num-experts "${NUM_EXPERTS}" \
      --max-layers "${NUM_LAYERS}" \
      --max-experts-per-layer "${PLAN_MAX_EXPERTS_PER_LAYER}" \
      > "${RUN_ROOT}/build_plan.log" 2>&1
  fi
}

run_mechanism() {
  local dir="${MECH_DIR}"
  local result="${dir}/result.json"
  local progress="${dir}/progress.jsonl"
  local run_log="${dir}/run.log"
  local init_trace="${dir}/init_trace.jsonl"
  local prealloc_trace="${dir}/post_stage_m_reserved_prealloc.jsonl"
  local prefetch_trace="${dir}/weight_prefetch_trace.jsonl"
  local kv_trace="${dir}/kv_runtime_trace.jsonl"
  local residency_trace="${dir}/residency_trace.jsonl"
  local kv_physical_trace="${dir}/kv_physical_trace.jsonl"
  local pointer_map="${dir}/kv_pointer_map.jsonl"

  if [[ ! -f "${PLAN}" ]]; then
    echo "missing cold plan: ${PLAN}" | tee "${dir}/blocked_missing_plan.txt"
    echo "2" > "${dir}/returncode.txt"
    return
  fi

  restore_manual_fault_params
  require_clean_gpu "mechanism"
  snapshot_gpu "before_mechanism"
  set +e
  run_generate_with_explicit_kv "${result}" "${progress}" "${MECH_KV_CACHE_MEMORY_BYTES}" \
    VLLM_USE_UVM=0 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_KV_CACHE_MANAGED=1 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=1 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE="${prefetch_trace}" \
    VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0 \
    VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
    VLLM_UVM_MOE_ADDRESS_MAP_ENABLE=0 \
    VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
    VLLM_UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE=0 \
    VLLM_UVM_DRIVER_TAG_ENABLE=1 \
    VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
    VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
    VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
    VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
    VLLM_UVM_INIT_WEIGHT_SYNC=1 \
    VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${init_trace}" \
    VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=1 \
    VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
    VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
    VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
    VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
    VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
    VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
    VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
    VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
    VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
    VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
    VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1 \
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
    VLLM_UVM_EXPERT_RESERVED_SKIP_ACTIVE_TOPK="${SKIP_ACTIVE_TOPK}" \
    VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=1 \
    VLLM_UVM_KV_RUNTIME_MODE=trace_only \
    VLLM_UVM_KV_RUNTIME_TRACE_FILE="${kv_trace}" \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
    VLLM_UVM_RESIDENCY_TRACE_ENABLE=1 \
    VLLM_UVM_RESIDENCY_TRACE_FILE="${residency_trace}" \
    VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE=1 \
    VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE="${pointer_map}" \
    VLLM_UVM_KV_PHYSICAL_TRACE_FILE="${kv_physical_trace}" \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=1 \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=0 \
    VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
    > "${run_log}" 2>&1
  local rc=$?
  set -e
  restore_manual_fault_params
  echo "${rc}" > "${dir}/returncode.txt"
  snapshot_gpu "after_mechanism"
}

analyze() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" "${PLAN}" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
plan_path = Path(sys.argv[3])

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out

def read_rc(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def log_tail(path: Path, limit: int = 80) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [
        line for line in lines
        if any(token in line.lower() for token in (
            "error", "exception", "traceback", "out of memory", "kv cache",
            "maximum number of tokens", "failed", "valueerror"
        ))
    ]
    return interesting[-limit:] if interesting else lines[-min(limit, len(lines)):]

def result_metrics(case_dir: Path) -> dict:
    result = read_json(case_dir / "result.json")
    requests = result.get("request_results") or []
    req_seconds = [float(r.get("generate_seconds") or 0.0) for r in requests]
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in requests)
    total = float(result.get("total_generate_seconds") or sum(req_seconds) or 0.0)
    return {
        "returncode": read_rc(case_dir / "returncode.txt"),
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "load_seconds": result.get("load_seconds"),
        "total_generate_seconds": total,
        "tokens_per_second": output_tokens / total if total else None,
        "p50_request_seconds": statistics.median(req_seconds) if req_seconds else None,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
        "error": result.get("error"),
        "log_tail": log_tail(case_dir / "run.log", 40),
    }

def release_metrics(case_dir: Path) -> dict:
    records = read_jsonl(case_dir / "init_trace.jsonl")
    entries = []
    summaries = []
    for record in records:
        action = record.get("action")
        if action == "init_weight_offload_driver_release_issued":
            entry = record.get("driver_release_entry") or {}
            nested = entry.get("entries")
            if isinstance(nested, list):
                entries.extend([item for item in nested if isinstance(item, dict)])
            elif isinstance(entry, dict):
                entries.append(entry)
        elif action == "init_weight_offload_summary":
            summaries.append(record)
    failed = [entry for entry in entries if not entry.get("ok")]
    released_bytes = sum(int(entry.get("released_bytes") or 0) for entry in entries)
    return {
        "trace_records": len(records),
        "release_entries": len(entries),
        "release_failed_entries": len(failed),
        "released_bytes": released_bytes,
        "released_mib": released_bytes / (1024 * 1024),
        "summary_records": len(summaries),
        "last_summary": summaries[-1] if summaries else None,
    }

def prefetch_metrics(case_dir: Path) -> dict:
    records = read_jsonl(case_dir / "weight_prefetch_trace.jsonl")
    restore_records = [
        record for record in records
        if "reserved" in str(record.get("action", ""))
        and ("prefetch" in str(record.get("action", "")) or "restore" in str(record.get("action", "")))
    ]
    remap_success = [
        record for record in records
        if int(record.get("successful_region_records") or 0) > 0
        or int(record.get("slot_remap_records") or 0) > 0
    ]
    return {
        "trace_records": len(records),
        "reserved_restore_like_records": len(restore_records),
        "records_with_successful_regions": len(remap_success),
        "last_record": records[-1] if records else None,
    }

def dmesg_fault_count() -> int:
    # dmesg may require privileges on some systems; use run logs as the hard evidence.
    return 0

plan = read_json(plan_path)
offload_plan = plan.get("offload_plan") if isinstance(plan.get("offload_plan"), list) else []
summary = {
    "run_root": str(root),
    "fixture": read_json(root / "long_prompt_fixture_meta.json"),
    "plan": {
        "path": str(plan_path),
        "exists": plan_path.exists(),
        "policy": (plan.get("metadata") or {}).get("policy") if isinstance(plan, dict) else None,
        "offload_plan_records": len(offload_plan),
    },
    "cases": {
        "baseline_cudamalloc_auto_kv": result_metrics(root / "baseline_cudamalloc_auto_kv"),
        "pilot_managed_no_offload": result_metrics(root / "pilot_managed_no_offload"),
        "cold_offload_large_kv": {
            **result_metrics(root / "cold_offload_large_kv"),
            "expert_release": release_metrics(root / "cold_offload_large_kv"),
            "runtime_prefetch": prefetch_metrics(root / "cold_offload_large_kv"),
        },
        "cold_offload_large_kv_skip_topk": {
            **result_metrics(root / "cold_offload_large_kv_skip_topk"),
            "expert_release": release_metrics(root / "cold_offload_large_kv_skip_topk"),
            "runtime_prefetch": prefetch_metrics(root / "cold_offload_large_kv_skip_topk"),
        },
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
max_model_len=${MAX_MODEL_LEN}
long_prompt_tokens=${LONG_PROMPT_TOKENS}
output_tokens=${OUTPUT_TOKENS}
max_num_seqs=${MAX_NUM_SEQS}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
pilot_kv_cache_memory_bytes=${PILOT_KV_CACHE_MEMORY_BYTES}
mechanism_kv_cache_memory_bytes=${MECH_KV_CACHE_MEMORY_BYTES}
num_experts=${NUM_EXPERTS}
num_layers=${NUM_LAYERS}
plan_max_experts_per_layer=${PLAN_MAX_EXPERTS_PER_LAYER}
target_roles=${TARGET_ROLES}
expert_envelope_bytes=${EXPERT_ENVELOPE_BYTES}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
mechanism_case_dir=${MECH_CASE_DIR_NAME}
skip_active_topk=${SKIP_ACTIVE_TOPK}
run_cases=${RUN_CASES}
EOF

require_environment
generate_long_fixture
snapshot_gpu "start"
if should_run_case "baseline"; then
  run_baseline
fi
if should_run_case "pilot"; then
  run_pilot
fi
if should_run_case "mechanism"; then
  run_mechanism
fi
snapshot_gpu "end"
analyze
