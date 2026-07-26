#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_request5_custom_vs_native_10gb_offload_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"

FIXTURE="${FIXTURE:-/tmp/vllm_request5_repeat_prep/request5_fixture.json}"
ROUTING_TRACE="${ROUTING_TRACE:-/tmp/vllm_request5_repeat_prep/pilot/routing_trace.jsonl}"
CUSTOM_PLAN="${CUSTOM_PLAN:-${RUN_ROOT}/request5_cold_approx10gb_plan.json}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-10}"

TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
CUSTOM_MAX_EXPERTS_PER_LAYER="${CUSTOM_MAX_EXPERTS_PER_LAYER:-52}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-524288}"
CUSTOM_TOPK_GATE_IMPL="${CUSTOM_TOPK_GATE_IMPL:-fused_select}"

PARAM_DIR=/sys/module/nvidia_uvm/parameters
PTE_ENABLE_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_enable"
PTE_MAX_PATH="${PARAM_DIR}/uvm_moe_slot_pte_remap_max_bytes"
ORIGINAL_PTE_ENABLE="$(cat "${PTE_ENABLE_PATH}" 2>/dev/null || echo 0)"
ORIGINAL_PTE_MAX="$(cat "${PTE_MAX_PATH}" 2>/dev/null || echo 0)"

mkdir -p "${RUN_ROOT}"
cp "${FIXTURE}" "${RUN_ROOT}/request5_fixture.json"
FIXTURE="${RUN_ROOT}/request5_fixture.json"

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

build_custom_plan() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${ROUTING_TRACE}" "${CUSTOM_PLAN}" \
      "${NUM_EXPERTS}" "${CUSTOM_MAX_EXPERTS_PER_LAYER}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

trace_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
num_experts = int(sys.argv[3])
max_experts_per_layer = int(sys.argv[4])

def layer_id_from_name(name):
    if not isinstance(name, str):
        return None
    parts = name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None

counts = defaultdict(Counter)
trace_records = 0
for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    layer_id = layer_id_from_name(record.get("layer_name"))
    if layer_id is None:
        continue
    trace_records += 1
    for expert_id in record.get("active_experts") or []:
        try:
            expert = int(expert_id)
        except (TypeError, ValueError):
            continue
        if 0 <= expert < num_experts:
            counts[layer_id][expert] += 1

offload_plan = []
selected_by_layer = {}
candidate_counts_by_layer = {}
for layer_id in range(48):
    layer_counts = counts.get(layer_id, Counter())
    cold = [
        expert_id
        for expert_id in range(num_experts)
        if int(layer_counts.get(expert_id, 0)) == 0
    ]
    selected = cold[:max_experts_per_layer]
    selected_by_layer[layer_id] = selected
    candidate_counts_by_layer[layer_id] = len(cold)
    for expert_id in selected:
        offload_plan.append(
            {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "reason": "request5_zero_hit_cold_approx10gb",
            }
        )

plan = {
    "metadata": {
        "policy": "exact_used_oracle_v1",
        "policy_note": (
            "Request-5 zero-hit cold expert plan built from routing trace; "
            "bounded per layer to target roughly 10GiB release."
        ),
        "routing_trace": str(trace_path),
        "trace_records": trace_records,
        "num_experts": num_experts,
        "max_layers": 48,
        "max_experts_per_layer": max_experts_per_layer,
        "candidate_counts_by_layer": candidate_counts_by_layer,
        "selected_by_layer": selected_by_layer,
    },
    "prefetch_plan": [],
    "offload_plan": offload_plan,
    "prefetch_plan_records": 0,
    "offload_plan_records": len(offload_plan),
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
print(json.dumps(plan["metadata"], indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/custom_plan_build.log"
}

run_custom_case() {
  local name="custom_cold_approx10gb"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  require_clean_gpu "${name}"
  reset_reserved_pool "${dir}/reset_before.json"
  prealloc_reserved_pool "${dir}/reserved_prealloc_before_model.json"
  echo "running ${name}: topk_gate_impl=${CUSTOM_TOPK_GATE_IMPL}"

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
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${CUSTOM_PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
      VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
      VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
      VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${CUSTOM_PLAN}" \
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
      VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_IMPL="${CUSTOM_TOPK_GATE_IMPL}" \
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
  return "${rc}"
}

run_native_case() {
  local name="native_cpu_offload_10gb"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  require_clean_gpu "${name}"
  echo "running ${name}: cpu_offload_gb=${NATIVE_CPU_OFFLOAD_GB}"

  set +e
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
      VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
        "${SCRIPT_DIR}/vllm_generate_sequence.py" \
        --model "${MODEL}" \
        --request-fixture-json "${FIXTURE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --cpu-offload-gb "${NATIVE_CPU_OFFLOAD_GB}" \
        --seed 0 \
        --temperature 0 \
        --output-json "${dir}/result.json" \
        --progress-jsonl "${dir}/progress.jsonl"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  return "${rc}"
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RUN_ROOT}" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])

def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

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

def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    kv_tokens = None
    kv_match = re.search(r"GPU KV cache size: ([0-9,]+) tokens", text)
    if kv_match:
        kv_tokens = int(kv_match.group(1).replace(",", ""))
    avail_kv_gib = None
    avail_match = re.search(r"Available KV cache memory: ([0-9.]+) GiB", text)
    if avail_match:
        avail_kv_gib = float(avail_match.group(1))
    return {
        "gpu_kv_tokens": kv_tokens,
        "available_kv_cache_gib": avail_kv_gib,
        "xid": "Xid" in text,
        "mmu_fault": "MMU" in text or "FAULT_PDE" in text,
        "uvm_fatal": "UVM fatal" in text,
    }

def result_summary(case: str) -> dict:
    case_dir = root / case
    result = load_json(case_dir / "result.json")
    request_results = result.get("request_results") or []
    generated = "\0".join(str(item.get("generated") or "") for item in request_results)
    total_generate = float(result.get("total_generate_seconds") or 0.0)
    tokens = sum(int(item.get("output_tokens") or 0) for item in request_results)
    summary = {
        "returncode": int((case_dir / "returncode.txt").read_text().strip())
        if (case_dir / "returncode.txt").exists()
        else None,
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "load_seconds": result.get("load_seconds"),
        "generate_seconds": total_generate or None,
        "tokens_per_second": tokens / total_generate if total_generate else None,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_last_request": result.get("gpu_memory_after_last_request"),
        "output_sha256": hashlib.sha256(generated.encode()).hexdigest()
        if generated
        else None,
    }
    summary.update(parse_log(case_dir / "run.log"))
    return summary

custom = result_summary("custom_cold_approx10gb")
native = result_summary("native_cpu_offload_10gb")

init_records = load_jsonl(root / "custom_cold_approx10gb" / "init_trace.jsonl")
issued = [
    item for item in init_records
    if str(item.get("action") or "").endswith("_issued")
]
custom_success = [item for item in issued if item.get("success") is True]
custom["init_release_records"] = len(custom_success)
custom["init_release_bytes"] = sum(int(item.get("bytes") or 0)
                                   for item in custom_success)
custom["failed_issued_records"] = len(issued) - len(custom_success)
custom["envelope_regions"] = sum(
    len(item.get("envelope_records") or [])
    for item in custom_success
)

native_manifest = load_jsonl(root / "native_cpu_offload_10gb" /
                             "cpu_offload_manifest.jsonl")
native_params = [
    item for item in native_manifest
    if item.get("event") == "cpu_offload_parameter"
]
native["cpu_offload_parameter_records"] = len(native_params)
native["cpu_offload_bytes"] = sum(int(item.get("bytes") or 0)
                                  for item in native_params)
native["cpu_offload_last_cumulative_bytes"] = max(
    [int(item.get("cumulative_offload_bytes") or 0) for item in native_params]
    or [0]
)
native["cpu_offload_first_parameters"] = [
    {
        "name": item.get("full_parameter_name"),
        "bytes": item.get("bytes"),
        "device_after": item.get("device_after"),
        "offload_backend": item.get("offload_backend"),
    }
    for item in native_params[:8]
]

plan = load_json(root / "request5_cold_approx10gb_plan.json")
fixture = load_json(root / "request5_fixture.json")

cases = {
    "custom_cold_approx10gb": custom,
    "native_cpu_offload_10gb": native,
}
hashes = {
    item.get("output_sha256")
    for item in cases.values()
    if item.get("output_sha256")
}
all_cases_have_hash = all(
    bool(item.get("output_sha256")) for item in cases.values()
)
summary = {
    "run_root": str(root),
    "kv_cache_memory_bytes": "auto",
    "fixture": fixture.get("metadata"),
    "custom_plan": plan.get("metadata"),
    "cases": cases,
    "outputs_match": all_cases_have_hash and len(hashes) == 1,
}

base = custom.get("generate_seconds")
other = native.get("generate_seconds")
if base and other:
    summary["native_vs_custom_generate_seconds"] = other - base
    summary["native_vs_custom_generate_pct"] = ((other / base) - 1.0) * 100
    summary["native_vs_custom_tokens_per_second_pct"] = (
        ((native["tokens_per_second"] / custom["tokens_per_second"]) - 1.0) * 100
        if native.get("tokens_per_second") and custom.get("tokens_per_second")
        else None
    )

(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  )
}

[[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
  echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
  exit 2
}
[[ -s "${FIXTURE}" ]] || {
  echo "fixture missing: ${FIXTURE}" >&2
  exit 2
}
[[ -s "${ROUTING_TRACE}" ]] || {
  echo "routing trace missing: ${ROUTING_TRACE}" >&2
  exit 2
}
[[ "$(cat /sys/module/nvidia_uvm/srcversion)" == \
   "$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko")" ]] || {
  echo "loaded nvidia_uvm does not match the workspace module" >&2
  exit 2
}

RUN_STARTED="$(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_before.txt"

build_custom_plan
set_param "${PTE_MAX_PATH}" "${RESERVED_SLOT_BYTES}"
set_param "${PTE_ENABLE_PATH}" 1
require_clean_gpu start

custom_rc=0
native_rc=0
run_custom_case || custom_rc=$?
restore_pte_params
reset_reserved_pool "${RUN_ROOT}/reset_after_custom.json"

run_native_case || native_rc=$?

write_summary
journalctl -k --since "${RUN_STARTED}" --no-pager \
  > "${RUN_ROOT}/kernel_since.log" 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits > "${RUN_ROOT}/gpu_final.txt"

echo "run_root=${RUN_ROOT}"
echo "custom_rc=${custom_rc}"
echo "native_rc=${native_rc}"
if [[ "${custom_rc}" -ne 0 || "${native_rc}" -ne 0 ]]; then
  exit 1
fi
