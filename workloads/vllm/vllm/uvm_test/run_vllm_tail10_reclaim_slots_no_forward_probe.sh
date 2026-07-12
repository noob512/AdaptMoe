#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_tail10_reclaim_slots_no_forward_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
TAIL_EXPERTS="${TAIL_EXPERTS:-10}"
TAIL_START="${TAIL_START:-$((NUM_EXPERTS - TAIL_EXPERTS))}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# This script intentionally does not preallocate slots before model load.  It
# first offloads tail experts, then tries to reclaim that freed residency as
# reserved slot backing.
POST_WARMUP_PREALLOC_SLOTS="${POST_WARMUP_PREALLOC_SLOTS:-80}"
POST_WARMUP_PREALLOC_ALLOW_EVICT="${POST_WARMUP_PREALLOC_ALLOW_EVICT:-0}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESET_RESERVED_POOL="${RESET_RESERVED_POOL:-1}"

SLOT_USE_LAYER_ID="${SLOT_USE_LAYER_ID:-22}"
SLOT_USE_EXPERTS="${SLOT_USE_EXPERTS:-118,119,120,121}"
SLOT_USE_SLOT_OFFSET="${SLOT_USE_SLOT_OFFSET:-0}"
SLOT_USE_STRICT_FOOTPRINT="${SLOT_USE_STRICT_FOOTPRINT:-0}"
SLOT_USE_INVALIDATE_AFTER="${SLOT_USE_INVALIDATE_AFTER:-0}"
SLOT_USE_TRACE="${RUN_ROOT}/post_offload_slot_use_trace.jsonl"

ACTIVE_LAYER_ID="${ACTIVE_LAYER_ID:-22}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-1,5,14,20,21,22,26,28,30,37,38,40,43,45,47,53,55,56,57,58,61,63,65,68,69,70,71,73,74,78,80,81,82,84,86,90,98,99,104,105,109,110,111,112,123,124,126}"

mkdir -p "${RUN_ROOT}"

PLAN="${RUN_ROOT}/tail${TAIL_EXPERTS}_plan.json"
ACTIVE_PROBE="${RUN_ROOT}/active_expert_probe.json"
RESULT="${RUN_ROOT}/warmup_only_result.json"
INIT_TRACE="${RUN_ROOT}/init_trace.jsonl"
ACTIVE_TRACE="${RUN_ROOT}/active_expert_residency_trace.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
SUMMARY="${RUN_ROOT}/summary.json"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

snapshot_gpu() {
  local label="$1"
  {
    echo "== ${label} =="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  } > "${RUN_ROOT}/gpu_${label}.txt"
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
num_experts=${NUM_EXPERTS}
max_layers=${MAX_LAYERS}
tail_start=${TAIL_START}
tail_experts=${TAIL_EXPERTS}
target_roles=${TARGET_ROLES}
post_warmup_prealloc_slots=${POST_WARMUP_PREALLOC_SLOTS}
post_warmup_prealloc_allow_evict=${POST_WARMUP_PREALLOC_ALLOW_EVICT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
slot_use_layer_id=${SLOT_USE_LAYER_ID}
slot_use_experts=${SLOT_USE_EXPERTS}
slot_use_slot_offset=${SLOT_USE_SLOT_OFFSET}
slot_use_strict_footprint=${SLOT_USE_STRICT_FOOTPRINT}
slot_use_invalidate_after=${SLOT_USE_INVALIDATE_AFTER}
active_layer_id=${ACTIVE_LAYER_ID}
active_experts=${ACTIVE_EXPERTS}
uvm_srcversion_sysfs=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)
uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)
uvm_moe_slot_pool_vidmem_budget_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes 2>/dev/null || true)
uvm_moe_slot_pte_remap_max_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || true)
reset_reserved_pool=${RESET_RESERVED_POOL}
EOF

used="$(gpu_used_mib || echo 999999)"
if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -gt 1024 ]]; then
  echo "GPU memory is not clean: ${used} MiB used. Refusing to run." | tee "${RUN_ROOT}/blocked.txt"
  snapshot_gpu "blocked"
  exit 3
fi

python3 - "${PLAN}" "${NUM_EXPERTS}" "${MAX_LAYERS}" "${TAIL_START}" "${TAIL_EXPERTS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
num_experts = int(sys.argv[2])
max_layers = int(sys.argv[3])
tail_start = int(sys.argv[4])
tail_experts = int(sys.argv[5])
offload_plan = []
for layer_id in range(max_layers):
    for expert_id in range(tail_start, min(num_experts, tail_start + tail_experts)):
        offload_plan.append({"layer_id": layer_id, "expert_id": expert_id})
payload = {
    "metadata": {
        "policy": "tail10_reclaim_slots_no_forward_probe",
        "num_experts": num_experts,
        "max_layers": max_layers,
        "tail_start": tail_start,
        "tail_experts": tail_experts,
    },
    "prefetch_plan": [],
    "offload_plan": offload_plan,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload["metadata"], sort_keys=True))
PY

python3 - "${ACTIVE_PROBE}" "${ACTIVE_LAYER_ID}" "${ACTIVE_EXPERTS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
layer_id = int(sys.argv[2])
experts = [
    int(item.strip())
    for item in sys.argv[3].replace(";", ",").split(",")
    if item.strip()
]
payload = {
    "metadata": {
        "source": "last_reserved_fault_coverage",
        "note": "Probe active expert residency without issuing forward.",
    },
    "layers": {
        str(layer_id): sorted(set(experts)),
    },
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY

snapshot_gpu "start"

if [[ "${RESET_RESERVED_POOL}" == "1" ]]; then
  (
    cd "${VLLM_ROOT}"
    ${VLLM_PY} - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

result = driver_expert_reserved_pool_reset(prefer_existing=True)
print(json.dumps(result, indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/reserved_pool_reset.json"
  snapshot_gpu "after_reserved_pool_reset"
fi

prealloc_allow_evict_flag=()
if [[ "${POST_WARMUP_PREALLOC_ALLOW_EVICT}" == "1" ]]; then
  prealloc_allow_evict_flag=(--post-warmup-prealloc-allow-evict)
fi

slot_use_strict_flag=()
if [[ "${SLOT_USE_STRICT_FOOTPRINT}" == "1" ]]; then
  slot_use_strict_flag=(--slot-use-strict-footprint)
fi

slot_use_invalidate_flag=()
if [[ "${SLOT_USE_INVALIDATE_AFTER}" == "1" ]]; then
  slot_use_invalidate_flag=(--slot-use-invalidate-after)
fi

set +e
(
  cd "${VLLM_ROOT}"
  rm -f vllm_uvm_weight_regions.jsonl
  VLLM_USE_UVM=1 \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_ENABLE=1 \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_FILE="${ACTIVE_PROBE}" \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_TRACE_FILE="${ACTIVE_TRACE}" \
VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_MAX_EXPERTS=0 \
  VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=dynamic \
  VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${POST_WARMUP_PREALLOC_SLOTS}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT="${SLOT_USE_STRICT_FOOTPRINT}" \
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=1 \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_ENABLE=1 \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_TRACE_FILE="${SLOT_USE_TRACE}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_LAYER_ID="${SLOT_USE_LAYER_ID}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_EXPERTS="${SLOT_USE_EXPERTS}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_COUNT="${POST_WARMUP_PREALLOC_SLOTS}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_STRICT_FOOTPRINT="${SLOT_USE_STRICT_FOOTPRINT}" \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=prefetch_cpu \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${TAIL_EXPERTS}" \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${INIT_TRACE}" \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_warmup_only.py" \
    --model "${MODEL}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${RESULT}" \
    --active-expert-probe-json "${ACTIVE_PROBE}" \
    --active-expert-roles "${TARGET_ROLES}" \
    --post-warmup-prealloc-slots 0 \
    --post-warmup-prealloc-slot-bytes "${RESERVED_SLOT_BYTES}" \
    "${prealloc_allow_evict_flag[@]}"
) 2>&1 | tee "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e

if [[ -f "${VLLM_ROOT}/vllm_uvm_weight_regions.jsonl" ]]; then
  cp "${VLLM_ROOT}/vllm_uvm_weight_regions.jsonl" "${RUN_ROOT}/vllm_uvm_weight_regions.jsonl"
fi

snapshot_gpu "after_run"

python3 - "${RUN_ROOT}" "${rc}" "${RESULT}" "${INIT_TRACE}" "${ACTIVE_TRACE}" "${SLOT_USE_TRACE}" "${SUMMARY}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
rc = int(sys.argv[2])
result_path = Path(sys.argv[3])
init_trace_path = Path(sys.argv[4])
active_trace_path = Path(sys.argv[5])
slot_use_trace_path = Path(sys.argv[6])
summary_path = Path(sys.argv[7])

result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}

init_issued = []
snapshots = []
if init_trace_path.exists():
    for line in init_trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("action") == "init_weight_offload_prefetch_cpu_issued":
            init_issued.append(record)
        elif record.get("action") == "init_weight_plan_residency_snapshot":
            snapshots.append(record)

active_by_marker = {}
if active_trace_path.exists():
    for line in active_trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        marker = str(record.get("marker"))
        states = Counter(
            item.get("state")
            for item in record.get("active_residency_records", [])
        )
        active_by_marker[marker] = {
            "layer_id": record.get("layer_id"),
            "expert_records": record.get("active_residency_expert_records"),
            "queried_bytes": record.get("active_residency_queried_bytes"),
            "gpu_bytes": record.get("active_residency_gpu_bytes"),
            "cpu_bytes": record.get("active_residency_cpu_bytes"),
            "unknown_bytes": record.get("active_residency_unknown_bytes"),
            "state_counts": dict(states),
            "cpu_present_experts": record.get("active_residency_cpu_present_experts"),
            "non_full_gpu_experts": record.get("active_residency_non_full_gpu_experts"),
        }

slot_use_records = []
if slot_use_trace_path.exists():
    for line in slot_use_trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            slot_use_records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
slot_use = slot_use_records[-1] if slot_use_records else {}
prefetch = slot_use.get("prefetch_result") or {}
source = slot_use.get("source_offload_result") or {}
driver = prefetch.get("driver_result") or {}
prealloc = result.get("reserved_prealloc") or {}
summary = {
    "returncode": rc,
    "result_status": result.get("mode"),
    "load_and_warmup_seconds": result.get("load_and_warmup_seconds"),
    "gpu_memory_after_warmup": result.get("gpu_memory_after_warmup"),
    "gpu_memory_before_prealloc": result.get("gpu_memory_before_prealloc"),
    "gpu_memory_after_prealloc": result.get("gpu_memory_after_prealloc"),
    "gpu_memory_after_slot_use_probe": result.get("gpu_memory_after_slot_use_probe"),
    "post_warmup_prealloc_slots": result.get("post_warmup_prealloc_slots"),
    "engine_slot_probe_records": len(slot_use_records),
    "prealloc_success_count": prealloc.get("success_count"),
    "prealloc_failure_count": prealloc.get("failure_count"),
    "prealloc_reserved_backing_bytes": prealloc.get("reserved_backing_bytes"),
    "init_issued_records": len(init_issued),
    "init_issued_bytes": sum(int(item.get("bytes", 0) or 0) for item in init_issued),
    "stage_m_layer22_snapshots": [
        {
            "marker": item.get("marker"),
            "cold_query_ok": item.get("cold_query_ok"),
            "cold_query_status": item.get("cold_query_status"),
            "cold_queried_bytes": item.get("cold_queried_bytes"),
            "cold_gpu_bytes": item.get("cold_gpu_bytes"),
            "cold_cpu_bytes": item.get("cold_cpu_bytes"),
        }
        for item in snapshots
        if item.get("layer_id") == 22
    ],
    "engine_active_residency_by_marker": active_by_marker,
    "slot_use_enabled": bool(slot_use_records),
    "slot_use_ok": slot_use.get("ok"),
    "slot_use_selected_region_count": slot_use.get("item_count"),
    "slot_use_source_successes": source.get("successes"),
    "slot_use_source_range_count": source.get("range_count"),
    "slot_use_source_executor": source.get("executor"),
    "slot_use_prefetch_ok": prefetch.get("ok"),
    "slot_use_prefetch_success_count": prefetch.get("success_count"),
    "slot_use_prefetch_failure_count": prefetch.get("failure_count"),
    "slot_use_prefetch_status": prefetch.get("status"),
    "slot_use_prefetch_elapsed_us": prefetch.get("elapsed_us"),
    "slot_use_driver_ok": driver.get("ok"),
    "slot_use_driver_success_count": driver.get("success_count"),
    "slot_use_driver_failure_count": driver.get("failure_count"),
    "slot_use_driver_requested_bytes": driver.get("requested_bytes"),
    "slot_use_driver_accepted_bytes": driver.get("accepted_bytes"),
    "slot_use_driver_reserved_backing_bytes": driver.get("reserved_backing_bytes"),
    "slot_use_driver_ordinary_resident_bytes": driver.get("ordinary_resident_bytes"),
    "slot_use_assigned_slots": prefetch.get("assigned_slots"),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${rc}"
