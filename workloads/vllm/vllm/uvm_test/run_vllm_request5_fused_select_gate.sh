#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
FIXTURE="${FIXTURE:-/tmp/vllm_request5_small_cold_skip_gate_20260724_004102/request5_fixture.json}"
PLAN="${PLAN:-/tmp/vllm_request5_small_cold_skip_gate_20260724_004102/small_cold_plan.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_request5_fused_select_gate_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
CASES="${CASES:-full_gate:triton,triton_async:triton_async,fused_select_gate:fused_select,no_kernel:none}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
PTE_ENABLE_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_enable"
PTE_MAX_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_max_bytes"
ORIGINAL_PTE_ENABLE="$(cat "${PTE_ENABLE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_PTE_MAX="$(cat "${PTE_MAX_PATH}" 2>/dev/null || echo 0)"

mkdir -p "${RUN_ROOT}"
cp "${FIXTURE}" "${RUN_ROOT}/request5_fixture.json"
cp "${PLAN}" "${RUN_ROOT}/small_cold_plan.json"
FIXTURE="${RUN_ROOT}/request5_fixture.json"
PLAN="${RUN_ROOT}/small_cold_plan.json"

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
slots = [
    {"pool_id": 0, "slot_id": slot_id, "slot_bytes": size}
    for slot_id in range(count)
]
result = driver_expert_reserved_pool_prealloc(
    slots, device=0, prefer_existing=True
)
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok") or int(result.get("success_count") or 0) < count:
    raise SystemExit(2)
PY
  ) > "${output}" 2>&1
}

run_case() {
  local name="$1"
  local gate_impl="$2"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  require_clean_gpu "${name}"
  reset_reserved_pool "${dir}/reset_before.json"
  prealloc_reserved_pool "${dir}/reserved_prealloc_before_model.json"
  echo "running ${name}: gate_impl=${gate_impl}"

  set +e
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
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=0 \
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
      VLLM_UVM_EXPERT_RESERVED_SKIP_ACTIVE_TOPK=0 \
      VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_IMPL="${gate_impl}" \
      VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE=2 \
      VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK=2 \
      VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE=1 \
      timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
        "${SCRIPT_DIR}/vllm_generate_sequence.py" \
        --model "${MODEL}" \
        --request-fixture-json "${FIXTURE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --seed 0 \
        --temperature 0 \
        --output-json "${dir}/result.json" \
        --progress-jsonl "${dir}/progress.jsonl"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  cat /proc/driver/nvidia-uvm/phase_f_route_log \
    > "${dir}/phase_f_route_log_after.txt" 2>/dev/null || true
  reset_reserved_pool "${dir}/reset_after.json"

  if [[ "${rc}" -ne 0 ]]; then
    tail -n 160 "${dir}/run.log" >&2
    exit "${rc}"
  fi
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${CASES}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
pairs = [item.split(":", 1) for item in sys.argv[2].split(",") if item]

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

def load_jsonl(path: Path) -> list[dict]:
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

cases = {}
for name, impl in pairs:
    case_dir = root / name
    result = load_json(case_dir / "result.json")
    request_results = result.get("request_results") or []
    generated = "\0".join(str(item.get("generated") or "")
                           for item in request_results)
    total = float(result.get("total_generate_seconds") or 0.0)
    tokens = sum(int(item.get("output_tokens") or 0)
                 for item in request_results)
    init_records = load_jsonl(case_dir / "init_trace.jsonl")
    issued = [item for item in init_records
              if str(item.get("action") or "").endswith("_issued")]
    successful = [item for item in issued if item.get("success") is True]
    log = (case_dir / "run.log").read_text(
        encoding="utf-8", errors="replace"
    )
    kv_tokens = None
    match = re.search(r"GPU KV cache size: ([0-9,]+) tokens", log)
    if match:
        kv_tokens = int(match.group(1).replace(",", ""))
    cases[name] = {
        "gate_impl": impl,
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "generate_seconds": total,
        "tokens_per_second": tokens / total if total else None,
        "load_seconds": result.get("load_seconds"),
        "gpu_kv_tokens": kv_tokens,
        "output_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "init_release_records": len(successful),
        "init_release_bytes": sum(int(item.get("bytes") or 0)
                                  for item in successful),
        "failed_issued_records": len(issued) - len(successful),
    }

reference = cases.get("no_kernel") or next(iter(cases.values()))
for item in cases.values():
    base = float(reference["generate_seconds"] or 0.0)
    current = float(item["generate_seconds"] or 0.0)
    item["vs_no_kernel_seconds"] = current - base
    item["vs_no_kernel_pct"] = ((current / base) - 1.0) * 100 if base else None

summary = {
    "run_root": str(root),
    "kv_cache_memory_bytes": "auto",
    "fixture": load_json(root / "request5_fixture.json").get("metadata"),
    "plan": load_json(root / "small_cold_plan.json").get("metadata"),
    "cases": cases,
    "outputs_match": len({item["output_sha256"] for item in cases.values()}) == 1,
}
(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

[[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
  echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
  exit 2
}
[[ -s "${FIXTURE}" && -s "${PLAN}" ]] || {
  echo "fixture or plan missing" >&2
  exit 2
}
[[ "$(cat /sys/module/nvidia_uvm/srcversion)" == \
   "$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")" ]] || {
  echo "loaded nvidia_uvm does not match the workspace module" >&2
  exit 2
}

set_param "${PTE_MAX_PATH}" "${RESERVED_SLOT_BYTES}"
set_param "${PTE_ENABLE_PATH}" 1
require_clean_gpu start

RUN_STARTED="$(date --iso-8601=seconds)"
IFS=',' read -r -a case_pairs <<< "${CASES}"
for pair in "${case_pairs[@]}"; do
  run_case "${pair%%:*}" "${pair#*:}"
done
write_summary
journalctl -k --since "${RUN_STARTED}" --no-pager \
  > "${RUN_ROOT}/kernel_since.log" 2>/dev/null || true
restore_pte_params
reset_reserved_pool "${RUN_ROOT}/reset_final.json"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_final.txt"
echo "run_root=${RUN_ROOT}"
