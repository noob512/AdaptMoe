#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
PROBE_MODE="${PROBE_MODE:-reserved}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_tail10_active_fault_${PROBE_MODE}_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"

PROMPT_EXACT="${PROMPT_EXACT:-Compare CUDA streams and events in one concise paragraph. [request 0]}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16}"
REQUEST_FIXTURE="${REQUEST_FIXTURE:-}"
REQUEST_SLEEP_SECONDS="${REQUEST_SLEEP_SECONDS:-0}"
REQUEST_SYNC_BETWEEN="${REQUEST_SYNC_BETWEEN:-0}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
TAIL_EXPERTS="${TAIL_EXPERTS:-10}"
TAIL_START="${TAIL_START:-$((NUM_EXPERTS - TAIL_EXPERTS))}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

PREALLOC_RESERVED_SLOTS="${PREALLOC_RESERVED_SLOTS:-80}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-80}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-4}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-10}"
RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE:-}"

COVERAGE_TRACE="${COVERAGE_TRACE:-1}"
ACTIVE_RESIDENCY_TRACE="${ACTIVE_RESIDENCY_TRACE:-1}"
ACTIVE_RESIDENCY_MAX_EXPERTS="${ACTIVE_RESIDENCY_MAX_EXPERTS:-0}"
PLAN_RESIDENCY_TRACE="${PLAN_RESIDENCY_TRACE:-1}"

mkdir -p "${RUN_ROOT}"

case "${PROBE_MODE}" in
  reserved)
    INIT_WEIGHT_OFFLOAD_ENABLE=1
    WEIGHT_PREFETCH_ENABLE=1
    WEIGHT_EXECUTOR="expert_reserved_regions_async"
    RESERVED_AFTER_WARMUP=1
    RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE:-0}"
    ;;
  native_uvm)
    INIT_WEIGHT_OFFLOAD_ENABLE=1
    WEIGHT_PREFETCH_ENABLE=0
    WEIGHT_EXECUTOR="cpp_batch"
    RESERVED_AFTER_WARMUP=0
    PREALLOC_RESERVED_SLOTS=0
    ;;
  reserved_no_stage_m)
    INIT_WEIGHT_OFFLOAD_ENABLE=0
    WEIGHT_PREFETCH_ENABLE=1
    WEIGHT_EXECUTOR="expert_reserved_regions_async"
    RESERVED_AFTER_WARMUP=1
    RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE:-1}"
    ;;
  *)
    echo "Unknown PROBE_MODE=${PROBE_MODE}. Use reserved, native_uvm, or reserved_no_stage_m." >&2
    exit 2
    ;;
esac

PLAN="${RUN_ROOT}/tail${TAIL_EXPERTS}_plan.json"
RESULT="${RUN_ROOT}/sequence_result.json"
PROGRESS="${RUN_ROOT}/sequence_progress.jsonl"
INIT_TRACE="${RUN_ROOT}/init_trace.jsonl"
COVERAGE_FILE="${RUN_ROOT}/coverage_trace.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
XID_LOG="${RUN_ROOT}/xid_tail.log"
WEIGHT_REGIONS_COPY="${RUN_ROOT}/vllm_uvm_weight_regions.jsonl"
REQUEST_FIXTURE_RUN="${RUN_ROOT}/request_fixture.json"

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
  } | tail -80 > "${XID_LOG}"
}

prealloc_reserved_pool() {
  if [[ "${PREALLOC_RESERVED_SLOTS}" -le 0 ]]; then
    return 0
  fi
  (
    cd "${VLLM_ROOT}"
    ${VLLM_PY} - <<PY
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_prealloc

slots = [
    {"pool_id": 0, "slot_id": slot_id, "slot_bytes": ${RESERVED_SLOT_BYTES}}
    for slot_id in range(${PREALLOC_RESERVED_SLOTS})
]
result = driver_expert_reserved_pool_prealloc(slots, device=0, prefer_existing=True)
print(json.dumps(result, indent=2, sort_keys=True))
if not result.get("ok") or int(result.get("success_count", 0)) < ${PREALLOC_RESERVED_SLOTS}:
    raise SystemExit(2)
PY
  ) | tee "${RUN_ROOT}/reserved_prealloc.json"
}

if [[ -n "${REQUEST_FIXTURE}" && -f "${REQUEST_FIXTURE}" ]]; then
  cp -f "${REQUEST_FIXTURE}" "${REQUEST_FIXTURE_RUN}"
else
  python3 - "${REQUEST_FIXTURE_RUN}" "${PROMPT_EXACT}" "${MAX_TOKENS_SEQUENCE}" <<'PY'
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
requests = []
for index, max_tokens in enumerate(tokens):
    exact_prompt = prompt if index == 0 else f"{prompt} [sequence {index}]"
    requests.append(
        {
            "request": index,
            "prompt": exact_prompt,
            "max_tokens": max_tokens,
        }
    )
payload = {
    "metadata": {
        "source": "run_vllm_tail10_active_fault_probe.sh",
        "reuse_note": "Pass this file via REQUEST_FIXTURE to reuse the exact request.",
    },
    "requests": requests,
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
probe_mode=${PROBE_MODE}
model=${MODEL}
request_fixture=${REQUEST_FIXTURE_RUN}
num_experts=${NUM_EXPERTS}
max_layers=${MAX_LAYERS}
tail_start=${TAIL_START}
tail_experts=${TAIL_EXPERTS}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_bank_count=${RESERVED_BANK_COUNT}
prealloc_reserved_slots=${PREALLOC_RESERVED_SLOTS}
coverage_trace=${COVERAGE_TRACE}
active_residency_trace=${ACTIVE_RESIDENCY_TRACE}
active_residency_max_experts=${ACTIVE_RESIDENCY_MAX_EXPERTS}
plan_residency_trace=${PLAN_RESIDENCY_TRACE}
request_sleep_seconds=${REQUEST_SLEEP_SECONDS}
request_sync_between=${REQUEST_SYNC_BETWEEN}
init_weight_offload_enable=${INIT_WEIGHT_OFFLOAD_ENABLE}
weight_prefetch_enable=${WEIGHT_PREFETCH_ENABLE}
weight_executor=${WEIGHT_EXECUTOR}
reserved_after_warmup=${RESERVED_AFTER_WARMUP}
reserved_prefetch_cpu_source=${RESERVED_PREFETCH_CPU_SOURCE}
uvm_srcversion=$(modinfo -F srcversion nvidia_uvm 2>/dev/null || true)
uvm_filename=$(modinfo -F filename nvidia_uvm 2>/dev/null || true)
uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)
uvm_moe_slot_pool_vidmem_budget_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes 2>/dev/null || true)
uvm_moe_slot_pte_remap_max_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes 2>/dev/null || true)
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
        "policy": "tail10_active_fault_probe",
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

collect_xid_tail
cp -f "${XID_LOG}" "${RUN_ROOT}/xid_before.log" || true
snapshot_gpu "start"
prealloc_reserved_pool
snapshot_gpu "after_prealloc"

set +e
(
  cd "${VLLM_ROOT}"
  VLLM_USE_UVM=1 \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE=0 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE="${COVERAGE_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_FILE="${COVERAGE_FILE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY="${ACTIVE_RESIDENCY_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_MAX_EXPERTS="${ACTIVE_RESIDENCY_MAX_EXPERTS}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE="${WEIGHT_PREFETCH_ENABLE}" \
  VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
  VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
  VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
  VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR}" \
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
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE}" \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP="${RESERVED_AFTER_WARMUP}" \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE}" \
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
    --request-fixture-json "${REQUEST_FIXTURE_RUN}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${RESULT}" \
    --progress-jsonl "${PROGRESS}" \
    --sleep-between-requests "${REQUEST_SLEEP_SECONDS}" \
    $([[ "${REQUEST_SYNC_BETWEEN}" == "1" ]] && printf '%s' "--sync-between-requests")
) 2>&1 | tee "${RUN_LOG}"
rc=${PIPESTATUS[0]}
set -e

snapshot_gpu "after_run"
collect_xid_tail
cp -f "${VLLM_ROOT}/vllm_uvm_weight_regions.jsonl" "${WEIGHT_REGIONS_COPY}" 2>/dev/null || true

python3 - "${RUN_ROOT}" "${rc}" "${RESULT}" "${PROGRESS}" "${COVERAGE_FILE}" "${INIT_TRACE}" "${XID_LOG}" "${WEIGHT_REGIONS_COPY}" "${PLAN}" "${REQUEST_FIXTURE_RUN}" <<'PY'
import json
import re
import sys
from pathlib import Path
from typing import Any

run_root = Path(sys.argv[1])
rc = int(sys.argv[2])
result_path = Path(sys.argv[3])
progress_path = Path(sys.argv[4])
coverage_path = Path(sys.argv[5])
init_trace_path = Path(sys.argv[6])
xid_path = Path(sys.argv[7])
weight_regions_path = Path(sys.argv[8])
plan_path = Path(sys.argv[9])
request_fixture_path = Path(sys.argv[10])


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return records


def classify_range(addr: int, ranges: list[tuple[int, int, dict[str, Any]]]) -> list[dict[str, Any]]:
    matches = []
    for start, end, record in ranges:
        if start <= addr < end:
            size = end - start
            matches.append(
                {
                    "start": f"0x{start:x}",
                    "end": f"0x{end:x}",
                    "size_bytes": size,
                    "name": record.get("name"),
                    "role": record.get("role"),
                    "layer_id": record.get("layer_id"),
                    "expert_id": record.get("expert_id"),
                    "is_moe_expert": record.get("is_moe_expert"),
                    "residency_region_id": record.get("residency_region_id"),
                }
            )
    matches.sort(key=lambda item: int(item["size_bytes"]))
    return matches


progress = read_jsonl(progress_path)
coverage = read_jsonl(coverage_path)
init_records = read_jsonl(init_trace_path)
result = read_json(result_path)
request_fixture = read_json(request_fixture_path)
plan = read_json(plan_path) or {}

coverage_bad = sum(not item.get("coverage_ok", False) for item in coverage)
missing_records = sum(bool(item.get("missing_reserved_experts")) for item in coverage)
failed_records = sum(bool(item.get("failed_reserved_experts")) for item in coverage)
last_coverage = coverage[-1] if coverage else None

xid_text = xid_path.read_text(encoding="utf-8", errors="replace") if xid_path.exists() else ""
xid_matches = re.findall(
    r"Xid .*?pid=(\d+).*?faulted @ 0x([0-9a-fA-F_]+).*?Fault is of type ([A-Z_]+) ([A-Z_]+)",
    xid_text,
)
latest_xid = None
fault_addr = None
if xid_matches:
    pid, raw_addr, fault_type, access_type = xid_matches[-1]
    fault_addr = int(raw_addr.replace("_", ""), 16)
    latest_xid = {
        "pid": int(pid),
        "fault_addr": f"0x{fault_addr:x}",
        "fault_type": fault_type,
        "access_type": access_type,
    }

offload_ranges: list[tuple[int, int, dict[str, Any]]] = []
for record in init_records:
    if record.get("action") != "init_weight_offload_prefetch_cpu_issued":
        continue
    ptr = record.get("ptr")
    size = record.get("bytes")
    if not isinstance(ptr, str) or size is None:
        continue
    start = int(ptr, 16)
    end = start + int(size)
    offload_ranges.append((start, end, record))

weight_ranges: list[tuple[int, int, dict[str, Any]]] = []
if weight_regions_path.exists():
    for record in read_jsonl(weight_regions_path):
        start = record.get("start_int", record.get("data_start_int"))
        end_inclusive = record.get("end_int", record.get("data_end_int"))
        if start is None or end_inclusive is None:
            continue
        weight_ranges.append((int(start), int(end_inclusive) + 1, record))

fault_offload_matches: list[dict[str, Any]] = []
fault_weight_matches: list[dict[str, Any]] = []
fault_best_weight: dict[str, Any] | None = None
fault_active_in_last_coverage = None
fault_reserved_in_last_coverage = None
fault_active_residency_record = None
if fault_addr is not None:
    fault_offload_matches = classify_range(fault_addr, offload_ranges)
    fault_weight_matches = classify_range(fault_addr, weight_ranges)
    fault_best_weight = fault_weight_matches[0] if fault_weight_matches else None
    if fault_best_weight and last_coverage:
        fault_layer = fault_best_weight.get("layer_id")
        fault_expert = fault_best_weight.get("expert_id")
        active = set(last_coverage.get("active_global_experts") or [])
        reserved = set(last_coverage.get("active_reserved_experts") or [])
        same_layer = fault_layer == last_coverage.get("layer_id")
        fault_active_in_last_coverage = bool(same_layer and fault_expert in active)
        fault_reserved_in_last_coverage = bool(same_layer and fault_expert in reserved)
        for item in last_coverage.get("active_residency_records") or []:
            if item.get("global_expert_id") == fault_expert:
                fault_active_residency_record = item
                break

plan_pairs = {
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in plan.get("offload_plan", [])
    if "layer_id" in item and "expert_id" in item
}
fault_in_offload_plan = None
if fault_best_weight:
    layer_id = fault_best_weight.get("layer_id")
    expert_id = fault_best_weight.get("expert_id")
    if isinstance(layer_id, int) and isinstance(expert_id, int):
        fault_in_offload_plan = (layer_id, expert_id) in plan_pairs

summary = {
    "returncode": rc,
    "request_fixture": request_fixture,
    "result_exists": result_path.exists(),
    "result_status": result.get("status") if isinstance(result, dict) else None,
    "completed_requests": result.get("completed_requests") if isinstance(result, dict) else None,
    "request_results": result.get("request_results") if isinstance(result, dict) else None,
    "progress_actions": [item.get("action") for item in progress],
    "last_progress": progress[-1] if progress else None,
    "coverage_records": len(coverage),
    "coverage_bad": coverage_bad,
    "missing_records": missing_records,
    "failed_records": failed_records,
    "last_coverage": last_coverage,
    "latest_xid": latest_xid,
    "fault_inside_tail_offload_ranges": bool(fault_offload_matches),
    "fault_offload_matches": fault_offload_matches,
    "fault_weight_matches": fault_weight_matches[:8],
    "fault_best_weight": fault_best_weight,
    "fault_in_offload_plan": fault_in_offload_plan,
    "fault_active_in_last_coverage": fault_active_in_last_coverage,
    "fault_reserved_in_last_coverage": fault_reserved_in_last_coverage,
    "fault_active_residency_record": fault_active_residency_record,
}
(run_root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${rc}"
