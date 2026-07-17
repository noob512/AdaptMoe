#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_tail5_static_0pct_diag_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-480}"

PROMPT_EXACT="${PROMPT_EXACT:-Compare CUDA streams and events in one concise paragraph.}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

MAX_LAYERS="${MAX_LAYERS:-48}"
TAIL_EXPERT_IDS="${TAIL_EXPERT_IDS:-123,124,125,126,127}"
BASE_EXPERT_IDS="${BASE_EXPERT_IDS:-0,1,2,3,4,5,6,7}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"

RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-128}"
RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY:-dynamic}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-8}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-5}"
RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS:-4}"
PREALLOC_RESERVED_BEFORE_RUN="${PREALLOC_RESERVED_BEFORE_RUN:-0}"
POST_STAGE_M_RESERVED_PREALLOC="${POST_STAGE_M_RESERVED_PREALLOC:-1}"
EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED:-1}"
KV_CACHE_MANAGED="${KV_CACHE_MANAGED:-1}"
TAIL_HIT_PROB="${TAIL_HIT_PROB:-0}"
WEIGHT_MAP_ENABLE="${WEIGHT_MAP_ENABLE:-1}"
MOE_ADDRESS_MAP_ENABLE="${MOE_ADDRESS_MAP_ENABLE:-1}"
ROUTING_TRACE_ENABLE="${ROUTING_TRACE_ENABLE:-1}"
PREFETCH_TRACE_ENABLE="${PREFETCH_TRACE_ENABLE:-1}"
RESERVED_COVERAGE_TRACE_ENABLE="${RESERVED_COVERAGE_TRACE_ENABLE:-0}"
RESERVED_COVERAGE_RESIDENCY_QUERY="${RESERVED_COVERAGE_RESIDENCY_QUERY:-0}"

mkdir -p "${RUN_ROOT}"

REQUEST_FIXTURE="${RUN_ROOT}/request_fixture.json"
PLAN="${RUN_ROOT}/tail5_static_plan.json"
RESULT="${RUN_ROOT}/result.json"
PROGRESS="${RUN_ROOT}/progress.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
INIT_TRACE="${RUN_ROOT}/init_trace.jsonl"
PREFETCH_TRACE="${RUN_ROOT}/prefetch_trace.jsonl"
ROUTING_TRACE="${RUN_ROOT}/routing_trace.jsonl"
MOE_ADDRESS_MAP="${RUN_ROOT}/moe_address_map.jsonl"
WEIGHT_MAP="${RUN_ROOT}/weight_map.jsonl"
POST_STAGE_M_PREALLOC_TRACE="${RUN_ROOT}/post_stage_m_reserved_prealloc.jsonl"
XID_BEFORE="${RUN_ROOT}/xid_before.log"
XID_TAIL="${RUN_ROOT}/xid_tail.log"
KERNEL_SINCE="${RUN_ROOT}/kernel_since.log"
ROUTE_LOG_AFTER="${RUN_ROOT}/phase_f_route_log_after.txt"
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

collect_xid_tail() {
  {
    grep -hE "Xid|MMU Fault|faulted @" /var/log/kern.log /var/log/syslog 2>/dev/null || true
  } | tail -160 > "${XID_TAIL}"
}

require_custom_uvm() {
  local built loaded pte_enable pte_max
  built="$(modinfo -F srcversion "${REPO_ROOT}/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
  loaded="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  pte_enable="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)"
  pte_max="$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || echo 0)"

  if [[ -z "${built}" || "${loaded}" != "${built}" ]]; then
    echo "loaded nvidia_uvm srcversion ${loaded} != built ${built}" | tee "${RUN_ROOT}/blocked_uvm.txt"
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
  (
    cd "${VLLM_ROOT}"
    ${VLLM_PY} - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True), indent=2, sort_keys=True))
PY
  ) > "${RUN_ROOT}/reserved_pool_reset.json" 2>&1 || true
}

prealloc_reserved_pool() {
  (
    cd "${VLLM_ROOT}"
    ${VLLM_PY} - <<PY
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_prealloc

slots = [
    {"pool_id": 0, "slot_id": slot_id, "slot_bytes": ${RESERVED_SLOT_BYTES}}
    for slot_id in range(${RESERVED_SLOT_COUNT})
]
result = driver_expert_reserved_pool_prealloc(slots, device=0, prefer_existing=True)
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok") or int(result.get("success_count", 0)) < ${RESERVED_SLOT_COUNT}:
    raise SystemExit(2)
PY
  ) | tee "${RUN_ROOT}/reserved_prealloc.json"
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
prompt_exact=${PROMPT_EXACT}
max_tokens_sequence=${MAX_TOKENS_SEQUENCE}
max_layers=${MAX_LAYERS}
tail_expert_ids=${TAIL_EXPERT_IDS}
base_expert_ids=${BASE_EXPERT_IDS}
target_roles=${TARGET_ROLES}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_slot_policy=${RESERVED_SLOT_POLICY}
reserved_bank_count=${RESERVED_BANK_COUNT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
reserved_experts_per_layer=${RESERVED_EXPERTS_PER_LAYER}
reserved_cleanup_delay_steps=${RESERVED_CLEANUP_DELAY_STEPS}
prealloc_reserved_before_run=${PREALLOC_RESERVED_BEFORE_RUN}
post_stage_m_reserved_prealloc=${POST_STAGE_M_RESERVED_PREALLOC}
expert_weights_managed=${EXPERT_WEIGHTS_MANAGED}
kv_cache_managed=${KV_CACHE_MANAGED}
tail_hit_prob=${TAIL_HIT_PROB}
weight_map_enable=${WEIGHT_MAP_ENABLE}
moe_address_map_enable=${MOE_ADDRESS_MAP_ENABLE}
routing_trace_enable=${ROUTING_TRACE_ENABLE}
prefetch_trace_enable=${PREFETCH_TRACE_ENABLE}
reserved_coverage_trace_enable=${RESERVED_COVERAGE_TRACE_ENABLE}
reserved_coverage_residency_query=${RESERVED_COVERAGE_RESIDENCY_QUERY}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)
uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)
uvm_moe_slot_pte_remap_max_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || true)
EOF

used="$(gpu_used_mib || echo 999999)"
if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -gt 1024 ]]; then
  echo "GPU memory is not clean: ${used} MiB used. Refusing to run." | tee "${RUN_ROOT}/blocked.txt"
  snapshot_gpu "blocked"
  exit 3
fi

require_custom_uvm

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
    "metadata": {"source": "run_vllm_tail5_static_0pct_diag.sh"},
    "requests": [
        {"request": index, "prompt": f"{prompt} [request {index}]", "max_tokens": max_tokens}
        for index, max_tokens in enumerate(tokens)
    ],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload["requests"], indent=2, sort_keys=True))
PY

python3 - "${PLAN}" "${MAX_LAYERS}" "${TAIL_EXPERT_IDS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
max_layers = int(sys.argv[2])
tail_ids = [
    int(item.strip())
    for item in sys.argv[3].replace(";", ",").split(",")
    if item.strip()
]
offload_plan = [
    {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "reason": "static_tail5_0pct_diag",
    }
    for layer_id in range(max_layers)
    for expert_id in tail_ids
]
plan = {
    "metadata": {
        "policy": "static_tail5_0pct_diag_v1",
        "max_layers": max_layers,
        "tail_expert_ids": tail_ids,
    },
    "offload_plan": offload_plan,
    "offload_plan_records": len(offload_plan),
    "prefetch_plan": list(offload_plan),
    "prefetch_plan_records": len(offload_plan),
}
path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"offload_plan_records": len(offload_plan), "output": str(path)}, indent=2, sort_keys=True))
PY

snapshot_gpu "start"
reset_reserved_pool
if [[ "${PREALLOC_RESERVED_BEFORE_RUN}" == "1" ]]; then
  prealloc_reserved_pool
  snapshot_gpu "after_reserved_prealloc"
fi

collect_xid_tail
cp -f "${XID_TAIL}" "${XID_BEFORE}" || true

START_ISO="$(date --iso-8601=seconds)"
rm -f "${RESULT}" "${PROGRESS}" "${RUN_LOG}" "${INIT_TRACE}" "${PREFETCH_TRACE}" "${ROUTING_TRACE}" "${MOE_ADDRESS_MAP}" "${WEIGHT_MAP}"

set +e
(
  cd "${VLLM_ROOT}"
  VLLM_USE_UVM=0 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
  VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_WEIGHT_MAP_ENABLE="${WEIGHT_MAP_ENABLE}" \
  VLLM_UVM_WEIGHT_MAP_FILE="${WEIGHT_MAP}" \
  VLLM_UVM_WEIGHT_MAP_INCLUDE_ALIASES=1 \
  VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
  VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
  VLLM_UVM_WEIGHT_MAP_REGISTER_EXPERT_TENSORS=1 \
  VLLM_UVM_WEIGHT_MAP_QUERY_EXPERT_SLICES=1 \
  VLLM_UVM_MOE_ADDRESS_MAP_ENABLE="${MOE_ADDRESS_MAP_ENABLE}" \
  VLLM_UVM_MOE_ADDRESS_MAP_FILE="${MOE_ADDRESS_MAP}" \
  VLLM_UVM_MOE_ADDRESS_MAP_DRIVER_PROBE=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${ROUTING_TRACE_ENABLE}" \
  VLLM_UVM_MOE_ROUTING_TRACE_FILE="${ROUTING_TRACE}" \
  VLLM_MOE_ROUTING_SIMULATION_STRATEGY=static_tail_hit \
  VLLM_MOE_ROUTING_STATIC_TAIL_EXPERT_IDS="${TAIL_EXPERT_IDS}" \
  VLLM_MOE_ROUTING_STATIC_BASE_EXPERT_IDS="${BASE_EXPERT_IDS}" \
  VLLM_MOE_ROUTING_STATIC_TAIL_HIT_PROB="${TAIL_HIT_PROB}" \
  VLLM_MOE_ROUTING_STATIC_SEED=17 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=prefetch_cpu \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=5 \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${INIT_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
  VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
  VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
  VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
  VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER=5 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE="${PREFETCH_TRACE_ENABLE}" \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE="${PREFETCH_TRACE}" \
  VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
  VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only \
  VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER=5 \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
  VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M="${POST_STAGE_M_RESERVED_PREALLOC}" \
  VLLM_UVM_EXPERT_RESERVED_POST_STAGE_M_PREALLOC_FILE="${POST_STAGE_M_PREALLOC_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY}" \
  VLLM_UVM_EXPERT_RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERT_IDS="${TAIL_EXPERT_IDS}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=1 \
  VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
  VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
  VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
  VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE="${RESERVED_COVERAGE_TRACE_ENABLE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY="${RESERVED_COVERAGE_RESIDENCY_QUERY}" \
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${RESULT}" \
    --progress-jsonl "${PROGRESS}"
) 2>&1 | tee "${RUN_LOG}"
run_rc=${PIPESTATUS[0]}
set -e

snapshot_gpu "after_run"
collect_xid_tail
journalctl -k --since "${START_ISO}" --no-pager > "${KERNEL_SINCE}" 2>/dev/null || true
cat /proc/driver/nvidia-uvm/phase_f_route_log > "${ROUTE_LOG_AFTER}" 2>/dev/null || true
reset_reserved_pool
snapshot_gpu "after_reserved_reset"

python3 - "${RUN_ROOT}" "${run_rc}" "${RESULT}" "${PROGRESS}" "${PLAN}" "${INIT_TRACE}" "${PREFETCH_TRACE}" "${ROUTING_TRACE}" "${MOE_ADDRESS_MAP}" "${WEIGHT_MAP}" "${POST_STAGE_M_PREALLOC_TRACE}" "${XID_BEFORE}" "${XID_TAIL}" "${KERNEL_SINCE}" "${ROUTE_LOG_AFTER}" "${SUMMARY}" "${TAIL_EXPERT_IDS}" <<'PY'
import json
import re
import sys
from pathlib import Path

(
    run_root,
    run_rc,
    result_path,
    progress_path,
    plan_path,
    init_trace_path,
    prefetch_trace_path,
    routing_trace_path,
    address_map_path,
    weight_map_path,
    post_stage_m_prealloc_path,
    xid_before_path,
    xid_tail_path,
    kernel_since_path,
    route_log_path,
    summary_path,
    tail_ids_raw,
) = sys.argv[1:]

tail_ids = {
    int(item.strip())
    for item in tail_ids_raw.replace(";", ",").split(",")
    if item.strip()
}

def read_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}

def read_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records

def text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

result = read_json(result_path)
plan = read_json(plan_path)
progress = read_jsonl(progress_path)
init_records = read_jsonl(init_trace_path)
prefetch_records = read_jsonl(prefetch_trace_path)
routing_records = read_jsonl(routing_trace_path)
address_records = read_jsonl(address_map_path)
weight_records = read_jsonl(weight_map_path)
post_stage_m_prealloc_records = read_jsonl(post_stage_m_prealloc_path)
xid_before = text(xid_before_path)
xid_tail = text(xid_tail_path)
kernel_since = text(kernel_since_path)
route_log = text(route_log_path)

request_begin_ts = min(
    (
        float(record.get("timestamp"))
        for record in progress
        if record.get("action") == "request_begin"
    ),
    default=None,
)
request_routing = [
    record
    for record in routing_records
    if request_begin_ts is None or float(record.get("timestamp", 0.0) or 0.0) >= request_begin_ts
]
all_tail_assignments = 0
all_topk_assignments = 0
all_active_pairs: set[tuple[int, int]] = set()
all_tail_active_pairs: set[tuple[int, int]] = set()
request_tail_assignments = 0
request_topk_assignments = 0
request_active_pairs: set[tuple[int, int]] = set()
request_tail_active_pairs: set[tuple[int, int]] = set()

def routing_layer_id(record: dict) -> int | None:
    layer_name = str(record.get("layer_name") or "")
    parts = layer_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[index + 1])
            except ValueError:
                pass
            break
    return None

for record in routing_records:
    layer_id = routing_layer_id(record)
    active_experts = record.get("active_experts")
    if isinstance(active_experts, list):
        for value in active_experts:
            try:
                expert = int(value)
            except (TypeError, ValueError):
                continue
            if layer_id is not None:
                all_active_pairs.add((layer_id, expert))
                if expert in tail_ids:
                    all_tail_active_pairs.add((layer_id, expert))
    counts = record.get("expert_token_counts")
    if isinstance(counts, dict):
        for key, value in counts.items():
            count = int(value or 0)
            all_topk_assignments += count
            try:
                expert = int(key)
            except (TypeError, ValueError):
                continue
            if expert in tail_ids:
                all_tail_assignments += count
for record in request_routing:
    layer_id = routing_layer_id(record)
    active_experts = record.get("active_experts")
    if isinstance(active_experts, list):
        for value in active_experts:
            try:
                expert = int(value)
            except (TypeError, ValueError):
                continue
            if layer_id is not None:
                request_active_pairs.add((layer_id, expert))
                if expert in tail_ids:
                    request_tail_active_pairs.add((layer_id, expert))
    counts = record.get("expert_token_counts")
    if isinstance(counts, dict):
        for key, value in counts.items():
            count = int(value or 0)
            request_topk_assignments += count
            try:
                expert = int(key)
            except (TypeError, ValueError):
                continue
            if expert in tail_ids:
                request_tail_assignments += count

fault_addrs: list[int] = []
for source in (kernel_since, xid_tail):
    for match in re.finditer(r"faulted @ 0x([0-9a-fA-F_]+)", source):
        fault_addrs.append(int(match.group(1).replace("_", ""), 16))
fault_addrs = list(dict.fromkeys(fault_addrs))

init_issued = [
    record
    for record in init_records
    if str(record.get("action") or "").endswith("_issued")
    or record.get("action") == "init_weight_offload_prefetch_cpu_issued"
]
prefetch_issued = [
    record
    for record in prefetch_records
    if "issued" in str(record.get("action") or "")
]

def output_metrics(data: dict) -> dict:
    request_results = data.get("request_results") or []
    seconds = [float(item.get("generate_seconds", 0.0) or 0.0) for item in request_results]
    tokens = sum(int(item.get("output_tokens", 0) or 0) for item in request_results)
    total = float(data.get("total_generate_seconds") or sum(seconds) or 0.0)
    return {
        "status": data.get("status"),
        "completed_requests": data.get("completed_requests"),
        "request_seconds": seconds,
        "output_tokens": tokens,
        "total_generate_seconds": total,
        "tokens_per_second": tokens / total if total else None,
        "load_seconds": data.get("load_seconds"),
        "gpu_memory_after_load": data.get("gpu_memory_after_load"),
        "gpu_memory_after_last_request": data.get("gpu_memory_after_last_request"),
    }

summary = {
    "run_root": run_root,
    "returncode": int(run_rc),
    "result": output_metrics(result),
    "plan": {
        "offload_plan_records": len(plan.get("offload_plan") or []),
        "prefetch_plan_records": len(plan.get("prefetch_plan") or []),
        "tail_expert_ids": sorted(tail_ids),
    },
    "init_offload": {
        "issued_records": len(init_issued),
        "issued_bytes": sum(int(record.get("bytes", 0) or 0) for record in init_issued),
    },
    "runtime_prefetch_trace": {
        "records": len(prefetch_records),
        "issued_records": len(prefetch_issued),
        "issued_bytes": sum(int(record.get("bytes", 0) or 0) for record in prefetch_issued),
    },
    "post_stage_m_reserved_prealloc": {
        "records": len(post_stage_m_prealloc_records),
        "last": post_stage_m_prealloc_records[-1] if post_stage_m_prealloc_records else None,
    },
    "routing": {
        "records": len(routing_records),
        "request_records": len(request_routing),
        "request_begin_timestamp": request_begin_ts,
        "all_active_pairs": len(all_active_pairs),
        "all_tail_active_pairs": [list(item) for item in sorted(all_tail_active_pairs)],
        "all_topk_assignments": all_topk_assignments,
        "all_tail_topk_assignments": all_tail_assignments,
        "request_active_pairs": len(request_active_pairs),
        "request_topk_assignments": request_topk_assignments,
        "request_tail_topk_assignments": request_tail_assignments,
        "request_tail_active_pairs": [list(item) for item in sorted(request_tail_active_pairs)],
    },
    "address_map": {
        "records": len(address_records),
    },
    "weight_map": {
        "records": len(weight_records),
    },
    "faults": {
        "fault_addresses": [f"0x{addr:x}" for addr in fault_addrs],
        "new_xid_or_mmu_fault": xid_before != xid_tail and ("Xid" in xid_tail or "MMU Fault" in xid_tail),
        "kernel_since_has_xid_or_mmu_fault": "Xid" in kernel_since or "MMU Fault" in kernel_since,
        "kernel_since_excerpt": kernel_since[-2400:],
        "xid_tail_excerpt": xid_tail[-2400:],
    },
    "route_log": {
        "bytes": len(route_log.encode("utf-8")),
        "moe_pte_range_lines": route_log.count("moe_pte_range"),
    },
}
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "run_root=${RUN_ROOT}"
exit "${run_rc}"
