#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_pre_service_real_load_offload_perf_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"

PROMPT_EXACT="${PROMPT_EXACT:-Compare CUDA streams and events in one concise paragraph.}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16,16,16,16,16,16,16,16}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
PLAN_MAX_EXPERTS_PER_LAYER="${PLAN_MAX_EXPERTS_PER_LAYER:-2}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
INIT_WEIGHT_OFFLOAD_MODE="${INIT_WEIGHT_OFFLOAD_MODE:-driver_release_cpu}"
EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE:-1}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE:-1}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-48}"
RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY:-dynamic}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-4}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-${PLAN_MAX_EXPERTS_PER_LAYER}}"
RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS:-4}"

RESET_RESERVED_POOL="${RESET_RESERVED_POOL:-1}"
PREALLOC_RESERVED_BEFORE_RUN="${PREALLOC_RESERVED_BEFORE_RUN:-0}"
POST_STAGE_M_RESERVED_PREALLOC="${POST_STAGE_M_RESERVED_PREALLOC:-1}"
CLEAN_RUNTIME_TRACE="${CLEAN_RUNTIME_TRACE:-1}"
EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED:-1}"
KV_CACHE_MANAGED="${KV_CACHE_MANAGED:-1}"

mkdir -p "${RUN_ROOT}"

PILOT_DIR="${RUN_ROOT}/pilot_trace"
MECH_DIR="${RUN_ROOT}/mechanism_perf"
mkdir -p "${PILOT_DIR}" "${MECH_DIR}"

REQUEST_FIXTURE="${RUN_ROOT}/request_fixture.json"
PILOT_TRACE="${PILOT_DIR}/routing_trace.jsonl"
PILOT_RESULT="${PILOT_DIR}/result.json"
PILOT_PROGRESS="${PILOT_DIR}/progress.jsonl"
PLAN_RAW="${RUN_ROOT}/real_load_offload_plan_raw.json"
PLAN="${RUN_ROOT}/real_load_offload_plan.json"
MECH_RESULT="${MECH_DIR}/result.json"
MECH_PROGRESS="${MECH_DIR}/progress.jsonl"
MECH_INIT_TRACE="${MECH_DIR}/init_trace.jsonl"
MECH_POST_STAGE_M_PREALLOC_TRACE="${MECH_DIR}/post_stage_m_reserved_prealloc.jsonl"
MECH_RUN_LOG="${MECH_DIR}/run.log"
MECH_XID_LOG="${MECH_DIR}/xid_tail.log"
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
  } | tail -100 > "${MECH_XID_LOG}"
}

require_custom_uvm() {
  local loaded pte_enable pte_max built
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
  ) | tee "${MECH_DIR}/reserved_prealloc.json"
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
prompt_exact=${PROMPT_EXACT}
max_tokens_sequence=${MAX_TOKENS_SEQUENCE}
num_experts=${NUM_EXPERTS}
max_layers=${MAX_LAYERS}
plan_max_experts_per_layer=${PLAN_MAX_EXPERTS_PER_LAYER}
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
    "metadata": {
        "source": "run_vllm_pre_service_real_load_offload_perf.sh",
    },
    "requests": [
        {"request": index, "prompt": f"{prompt} [request {index}]", "max_tokens": max_tokens}
        for index, max_tokens in enumerate(tokens)
    ],
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload["requests"], indent=2, sort_keys=True))
PY

snapshot_gpu "start"

set +e
(
  cd "${VLLM_ROOT}"
  VLLM_USE_UVM=0 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
  VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_FILE="${PILOT_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${PILOT_RESULT}" \
    --progress-jsonl "${PILOT_PROGRESS}"
) 2>&1 | tee "${PILOT_DIR}/run.log"
pilot_rc=${PIPESTATUS[0]}
set -e

if [[ "${pilot_rc}" -ne 0 ]]; then
  echo "pilot routing trace run failed with rc=${pilot_rc}" | tee "${RUN_ROOT}/pilot_failed.txt"
  exit "${pilot_rc}"
fi

python3 "${SCRIPT_DIR}/build_offload_plan_from_routing_trace.py" \
  --routing-trace "${PILOT_TRACE}" \
  --output "${PLAN_RAW}" \
  --max-layers "${MAX_LAYERS}" \
  --max-experts-per-layer "${PLAN_MAX_EXPERTS_PER_LAYER}" \
  | tee "${RUN_ROOT}/plan_build.log"

python3 - "${PLAN_RAW}" "${PLAN}" <<'PY'
import json
import sys
from pathlib import Path

raw_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
plan = json.loads(raw_path.read_text(encoding="utf-8"))
offload_plan = plan.get("offload_plan")
if not isinstance(offload_plan, list):
    raise SystemExit("plan has no offload_plan")
plan["prefetch_plan"] = list(offload_plan)
plan["prefetch_plan_records"] = len(plan["prefetch_plan"])
plan["metadata"] = dict(plan.get("metadata") or {})
plan["metadata"]["policy"] = "real_load_active_offload_prefetch_v1"
out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({
    "offload_plan_records": len(offload_plan),
    "prefetch_plan_records": len(plan["prefetch_plan"]),
    "output": str(out_path),
}, indent=2, sort_keys=True))
PY

snapshot_gpu "before_mechanism"
reset_reserved_pool
if [[ "${PREALLOC_RESERVED_BEFORE_RUN}" == "1" ]]; then
  prealloc_reserved_pool
  snapshot_gpu "after_reserved_prealloc"
fi

collect_xid_tail
cp -f "${MECH_XID_LOG}" "${RUN_ROOT}/xid_before_mechanism.log" || true

set +e
(
  cd "${VLLM_ROOT}"
  rm -f vllm_uvm_weight_regions.jsonl
  VLLM_USE_UVM=0 \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
  VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE="${INIT_WEIGHT_OFFLOAD_MODE}" \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${PLAN_MAX_EXPERTS_PER_LAYER}" \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${MECH_INIT_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
  VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
  VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
  VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
  VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${PLAN_MAX_EXPERTS_PER_LAYER}" \
  VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
  VLLM_UVM_EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE}" \
  VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
  VLLM_UVM_DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE}" \
  VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only \
  VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="${PLAN_MAX_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
  VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M="${POST_STAGE_M_RESERVED_PREALLOC}" \
  VLLM_UVM_EXPERT_RESERVED_POST_STAGE_M_PREALLOC_FILE="${MECH_POST_STAGE_M_PREALLOC_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY}" \
  VLLM_UVM_EXPERT_RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=1 \
  VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
  VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
  VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
  VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${MECH_RESULT}" \
    --progress-jsonl "${MECH_PROGRESS}"
) 2>&1 | tee "${MECH_RUN_LOG}"
mech_rc=${PIPESTATUS[0]}
set -e

snapshot_gpu "after_mechanism"
collect_xid_tail
cp -f "${VLLM_ROOT}/vllm_uvm_weight_regions.jsonl" "${RUN_ROOT}/vllm_uvm_weight_regions.jsonl" 2>/dev/null || true

python3 - "${PILOT_RESULT}" "${PLAN}" "${MECH_RESULT}" "${MECH_INIT_TRACE}" "${MECH_POST_STAGE_M_PREALLOC_TRACE}" "${MECH_XID_LOG}" "${SUMMARY}" <<'PY'
import json
import sys
from pathlib import Path

pilot_result_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
mech_result_path = Path(sys.argv[3])
init_trace_path = Path(sys.argv[4])
post_stage_m_prealloc_path = Path(sys.argv[5])
xid_path = Path(sys.argv[6])
summary_path = Path(sys.argv[7])

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}

def compute_metrics(result: dict) -> dict:
    requests = result.get("request_results") or []
    request_seconds = [float(item.get("generate_seconds", 0.0) or 0.0) for item in requests]
    output_tokens = sum(int(item.get("output_tokens", 0) or 0) for item in requests)
    total_generate = float(result.get("total_generate_seconds") or sum(request_seconds) or 0.0)
    return {
        "status": result.get("status"),
        "completed_requests": result.get("completed_requests"),
        "load_seconds": result.get("load_seconds"),
        "total_generate_seconds": total_generate,
        "avg_request_seconds": (sum(request_seconds) / len(request_seconds)) if request_seconds else None,
        "p50_request_seconds": (sorted(request_seconds)[len(request_seconds) // 2] if request_seconds else None),
        "request_seconds": request_seconds,
        "output_tokens": output_tokens,
        "tokens_per_second": (output_tokens / total_generate) if total_generate else None,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
        "gpu_memory_after_last_request": result.get("gpu_memory_after_last_request"),
        "pre_generate_active_gate": result.get("pre_generate_active_gate"),
    }

pilot = read_json(pilot_result_path)
mech = read_json(mech_result_path)
plan = read_json(plan_path)
init_records = []
if init_trace_path.exists():
    for line in init_trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        init_records.append(record)
init_issued = [
    item for item in init_records
    if str(item.get("action") or "").endswith("_issued")
    or item.get("action") == "init_weight_offload_prefetch_cpu_issued"
]
post_stage_m_prealloc_records = []
if post_stage_m_prealloc_path.exists():
    for line in post_stage_m_prealloc_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        post_stage_m_prealloc_records.append(record)
xid_text = xid_path.read_text(encoding="utf-8", errors="replace") if xid_path.exists() else ""

summary = {
    "pilot": compute_metrics(pilot),
    "mechanism": compute_metrics(mech),
    "plan": {
        "offload_plan_records": len(plan.get("offload_plan") or []),
        "prefetch_plan_records": len(plan.get("prefetch_plan") or []),
        "policy": (plan.get("metadata") or {}).get("policy"),
    },
    "init_offload": {
        "records": len(init_issued),
        "bytes": sum(int(item.get("bytes", 0) or 0) for item in init_issued),
    },
    "post_stage_m_reserved_prealloc": {
        "records": len(post_stage_m_prealloc_records),
        "last": post_stage_m_prealloc_records[-1] if post_stage_m_prealloc_records else None,
    },
    "xid_tail_has_xid": ("Xid" in xid_text) or ("MMU Fault" in xid_text),
    "xid_tail_excerpt": xid_text[-2000:],
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${mech_rc}"
