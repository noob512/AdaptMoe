#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_tail10_post_offload_slots_e2e16_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"

PROMPT_EXACT="${PROMPT_EXACT:-Compare CUDA streams and events in one concise paragraph. [request 0]}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
TAIL_EXPERTS="${TAIL_EXPERTS:-10}"
TAIL_START="${TAIL_START:-$((NUM_EXPERTS - TAIL_EXPERTS))}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

# Post-offload slot pool. bank_count=1 makes each layer reuse the whole pool
# after invalidating stale mappings, which is slow but simpler for a safety run.
POST_OFFLOAD_SLOT_COUNT="${POST_OFFLOAD_SLOT_COUNT:-30}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-${POST_OFFLOAD_SLOT_COUNT}}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-1}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-10}"

# Known offloaded experts active in the previous failing prefill layer.
SLOT_USE_LAYER_ID="${SLOT_USE_LAYER_ID:-22}"
SLOT_USE_EXPERTS="${SLOT_USE_EXPERTS:-123,124,126}"
SLOT_USE_MAX_EXPERTS="${SLOT_USE_MAX_EXPERTS:-3}"
ACTIVE_LAYER_ID="${ACTIVE_LAYER_ID:-22}"
ACTIVE_EXPERTS="${ACTIVE_EXPERTS:-1,5,14,20,21,22,26,28,30,37,38,40,43,45,47,53,55,56,57,58,61,63,65,68,69,70,71,73,74,78,80,81,82,84,86,90,98,99,104,105,109,110,111,112,123,124,126}"

COVERAGE_TRACE="${COVERAGE_TRACE:-1}"
ACTIVE_RESIDENCY_TRACE="${ACTIVE_RESIDENCY_TRACE:-1}"
ACTIVE_RESIDENCY_MAX_EXPERTS="${ACTIVE_RESIDENCY_MAX_EXPERTS:-0}"
PLAN_RESIDENCY_TRACE="${PLAN_RESIDENCY_TRACE:-1}"
RESET_RESERVED_POOL="${RESET_RESERVED_POOL:-1}"
ACTIVE_GATE="${ACTIVE_GATE:-1}"

mkdir -p "${RUN_ROOT}"

PLAN="${RUN_ROOT}/tail${TAIL_EXPERTS}_plan.json"
ACTIVE_PROBE="${RUN_ROOT}/active_expert_probe.json"
REQUEST_FIXTURE="${RUN_ROOT}/request_fixture.json"
RESULT="${RUN_ROOT}/sequence_result.json"
PROGRESS="${RUN_ROOT}/sequence_progress.jsonl"
INIT_TRACE="${RUN_ROOT}/init_trace.jsonl"
ACTIVE_TRACE="${RUN_ROOT}/active_expert_residency_trace.jsonl"
SLOT_USE_TRACE="${RUN_ROOT}/post_offload_slot_use_trace.jsonl"
COVERAGE_FILE="${RUN_ROOT}/coverage_trace.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
SUMMARY="${RUN_ROOT}/summary.json"
XID_LOG="${RUN_ROOT}/xid_tail.log"

BUILT_UVM_SRCVERSION="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"

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

collect_xid_tail() {
  {
    grep -hE "Xid|MMU Fault|faulted @" /var/log/kern.log /var/log/syslog 2>/dev/null || true
  } | tail -100 > "${XID_LOG}"
}

require_custom_uvm() {
  local loaded pte_enable pte_max
  loaded="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  pte_enable="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)"
  pte_max="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || echo 0)"

  if [[ -z "${BUILT_UVM_SRCVERSION}" || "${loaded}" != "${BUILT_UVM_SRCVERSION}" ]]; then
    echo "loaded nvidia_uvm srcversion ${loaded} != built ${BUILT_UVM_SRCVERSION}" | tee "${RUN_ROOT}/blocked_uvm.txt"
    return 1
  fi
  if [[ "${pte_enable}" != "1" ]]; then
    echo "uvm_moe_slot_pte_remap_enable=${pte_enable}, expected 1" | tee "${RUN_ROOT}/blocked_uvm.txt"
    return 1
  fi
  if [[ "${pte_max}" -lt "${RESERVED_SLOT_BYTES}" ]]; then
    echo "uvm_moe_slot_pte_remap_max_bytes=${pte_max}, expected >= ${RESERVED_SLOT_BYTES}" | tee "${RUN_ROOT}/blocked_uvm.txt"
    return 1
  fi
}

reset_reserved_pool() {
  if [[ "${RESET_RESERVED_POOL}" != "1" ]]; then
    return 0
  fi
  (
    cd "${VLLM_ROOT}"
    ${VLLM_PY} - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True), indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/reserved_pool_reset.json"
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
prompt_exact=${PROMPT_EXACT}
max_tokens_sequence=${MAX_TOKENS_SEQUENCE}
num_experts=${NUM_EXPERTS}
max_layers=${MAX_LAYERS}
tail_start=${TAIL_START}
tail_experts=${TAIL_EXPERTS}
post_offload_slot_count=${POST_OFFLOAD_SLOT_COUNT}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_bank_count=${RESERVED_BANK_COUNT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
slot_use_layer_id=${SLOT_USE_LAYER_ID}
slot_use_experts=${SLOT_USE_EXPERTS}
slot_use_max_experts=${SLOT_USE_MAX_EXPERTS}
active_layer_id=${ACTIVE_LAYER_ID}
active_experts=${ACTIVE_EXPERTS}
active_gate=${ACTIVE_GATE}
built_uvm_srcversion=${BUILT_UVM_SRCVERSION}
loaded_uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)
uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)
uvm_moe_slot_pte_remap_max_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || true)
uvm_moe_slot_pool_vidmem_budget_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes 2>/dev/null || true)
EOF

used="$(gpu_used_mib || echo 999999)"
if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -gt 1024 ]]; then
  echo "GPU memory is not clean: ${used} MiB used. Refusing to run." | tee "${RUN_ROOT}/blocked.txt"
  snapshot_gpu "blocked"
  exit 3
fi

require_custom_uvm

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
        "policy": "tail10_post_offload_slots_e2e16",
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

if [[ "${ACTIVE_GATE}" == "1" ]]; then
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
        "note": "Gate e2e generate on active expert residency after post-offload slot setup.",
    },
    "layers": {str(layer_id): sorted(set(experts))},
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
PY
else
  printf '{"metadata":{"source":"disabled"},"layers":{}}\n' > "${ACTIVE_PROBE}"
fi

python3 - "${REQUEST_FIXTURE}" "${PROMPT_EXACT}" "${MAX_TOKENS_SEQUENCE}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt = sys.argv[2]
tokens = [
    int(item.strip())
    for item in sys.argv[3].replace(";", ",").split(",")
    if item.strip()
]
payload = {
    "metadata": {"source": "run_vllm_tail10_post_offload_slots_e2e16.sh"},
    "requests": [
        {"request": index, "prompt": prompt, "max_tokens": max_tokens}
        for index, max_tokens in enumerate(tokens)
    ],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

collect_xid_tail
cp -f "${XID_LOG}" "${RUN_ROOT}/xid_before.log" || true
snapshot_gpu "start"
reset_reserved_pool
snapshot_gpu "after_reserved_pool_reset"

active_gate_args=()
if [[ "${ACTIVE_GATE}" == "1" ]]; then
  active_gate_args=(
    --require-active-all-gpu
    --require-active-gpu-trace-jsonl "${ACTIVE_TRACE}"
    --require-active-slot-trace-jsonl "${SLOT_USE_TRACE}"
    --require-active-marker after_post_offload_slot_use_probe
  )
fi

set +e
(
  cd "${VLLM_ROOT}"
  rm -f vllm_uvm_weight_regions.jsonl
  VLLM_USE_UVM=1 \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE=0 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE="${COVERAGE_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_FILE="${COVERAGE_FILE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY="${ACTIVE_RESIDENCY_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_MAX_EXPERTS="${ACTIVE_RESIDENCY_MAX_EXPERTS}" \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_ENABLE="${ACTIVE_GATE}" \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_FILE="${ACTIVE_PROBE}" \
  VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_TRACE_FILE="${ACTIVE_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
  VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
  VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
  VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
  VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
  VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=layer_mod_bank \
  VLLM_UVM_EXPERT_RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS=0 \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=1 \
  VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
  VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
  VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
  VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=0 \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_ENABLE=1 \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_TRACE_FILE="${SLOT_USE_TRACE}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_LAYER_ID="${SLOT_USE_LAYER_ID}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_EXPERTS="${SLOT_USE_EXPERTS}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_MAX_EXPERTS="${SLOT_USE_MAX_EXPERTS}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_STRICT_FOOTPRINT=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=prefetch_cpu \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${TAIL_EXPERTS}" \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${INIT_TRACE}" \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE="${PLAN_RESIDENCY_TRACE}" \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${RESULT}" \
    --progress-jsonl "${PROGRESS}" \
    "${active_gate_args[@]}"
) 2>&1 | tee "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e

snapshot_gpu "after_run"
collect_xid_tail
cp -f "${VLLM_ROOT}/vllm_uvm_weight_regions.jsonl" "${RUN_ROOT}/vllm_uvm_weight_regions.jsonl" 2>/dev/null || true

python3 - "${RUN_ROOT}" "${rc}" "${RESULT}" "${PROGRESS}" "${ACTIVE_TRACE}" "${SLOT_USE_TRACE}" "${COVERAGE_FILE}" "${INIT_TRACE}" "${XID_LOG}" "${SUMMARY}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
rc = int(sys.argv[2])
result_path = Path(sys.argv[3])
progress_path = Path(sys.argv[4])
active_trace_path = Path(sys.argv[5])
slot_use_trace_path = Path(sys.argv[6])
coverage_path = Path(sys.argv[7])
init_trace_path = Path(sys.argv[8])
xid_path = Path(sys.argv[9])
summary_path = Path(sys.argv[10])

def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records

result = read_json(result_path)
progress = read_jsonl(progress_path)
active_records = read_jsonl(active_trace_path)
slot_records = read_jsonl(slot_use_trace_path)
coverage = read_jsonl(coverage_path)
init_records = read_jsonl(init_trace_path)

active_by_marker = {}
for record in active_records:
    marker = str(record.get("marker"))
    states = Counter(item.get("state") for item in record.get("active_residency_records", []))
    active_by_marker[marker] = {
        "layer_id": record.get("layer_id"),
        "expert_records": record.get("active_residency_expert_records"),
        "state_counts": dict(states),
        "queried_bytes": record.get("active_residency_queried_bytes"),
        "gpu_bytes": record.get("active_residency_gpu_bytes"),
        "cpu_bytes": record.get("active_residency_cpu_bytes"),
        "unknown_bytes": record.get("active_residency_unknown_bytes"),
        "cpu_present_experts": record.get("active_residency_cpu_present_experts"),
        "non_full_gpu_experts": record.get("active_residency_non_full_gpu_experts"),
        "query_failed_experts": record.get("active_residency_query_failed_experts"),
    }

slot_use = slot_records[-1] if slot_records else {}
prefetch = slot_use.get("prefetch_result") or {}
driver = prefetch.get("driver_result") or {}
gate = result.get("pre_generate_active_gate") or {}
init_issued = [
    item for item in init_records
    if item.get("action") == "init_weight_offload_prefetch_cpu_issued"
]
coverage_bad = [item for item in coverage if not item.get("coverage_ok", False)]
xid_text = xid_path.read_text(encoding="utf-8", errors="replace") if xid_path.exists() else ""
summary = {
    "returncode": rc,
    "result_status": result.get("status"),
    "completed_requests": result.get("completed_requests"),
    "request_results": result.get("request_results"),
    "progress_actions": [item.get("action") for item in progress],
    "last_progress": progress[-1] if progress else None,
    "pre_generate_active_gate": gate,
    "active_by_marker": active_by_marker,
    "slot_use_records": len(slot_records),
    "slot_use_ok": slot_use.get("ok"),
    "slot_use_item_count": slot_use.get("item_count"),
    "slot_use_prefetch_ok": prefetch.get("ok"),
    "slot_use_prefetch_status": prefetch.get("status"),
    "slot_use_prefetch_success_count": prefetch.get("success_count"),
    "slot_use_prefetch_failure_count": prefetch.get("failure_count"),
    "slot_use_driver_ok": driver.get("ok"),
    "slot_use_driver_success_count": driver.get("success_count"),
    "slot_use_driver_failure_count": driver.get("failure_count"),
    "slot_use_driver_requested_bytes": driver.get("requested_bytes"),
    "slot_use_driver_accepted_bytes": driver.get("accepted_bytes"),
    "slot_use_driver_reserved_backing_bytes": driver.get("reserved_backing_bytes"),
    "slot_use_driver_ordinary_resident_bytes": driver.get("ordinary_resident_bytes"),
    "init_issued_records": len(init_issued),
    "init_issued_bytes": sum(int(item.get("bytes", 0) or 0) for item in init_issued),
    "coverage_records": len(coverage),
    "coverage_bad": len(coverage_bad),
    "last_coverage": coverage[-1] if coverage else None,
    "xid_tail_has_xid": "Xid" in xid_text or "MMU Fault" in xid_text,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${rc}"
