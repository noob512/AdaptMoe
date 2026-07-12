#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
PROMPT="${PROMPT:-Compare CUDA streams and events in one concise paragraph.}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16,8}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_tail10_reserved_two_request_probe_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"

NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
TAIL_EXPERTS="${TAIL_EXPERTS:-10}"
TAIL_START="${TAIL_START:-$((NUM_EXPERTS - TAIL_EXPERTS))}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

PREALLOC_RESERVED_SLOTS="${PREALLOC_RESERVED_SLOTS:-40}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-40}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-2}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-10}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
COVERAGE_TRACE="${COVERAGE_TRACE:-1}"
REQUEST_SLEEP_SECONDS="${REQUEST_SLEEP_SECONDS:-0}"
REQUEST_SYNC_BETWEEN="${REQUEST_SYNC_BETWEEN:-0}"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

snapshot_gpu() {
  local label="$1"
  {
    echo "== ${label} =="
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
  } > "${RUN_ROOT}/gpu_${label}.txt"
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

PLAN="${RUN_ROOT}/tail${TAIL_EXPERTS}_plan.json"
RESULT="${RUN_ROOT}/sequence_result.json"
PROGRESS="${RUN_ROOT}/sequence_progress.jsonl"
INIT_TRACE="${RUN_ROOT}/init_trace.jsonl"
COVERAGE_FILE="${RUN_ROOT}/coverage_trace.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
max_tokens_sequence=${MAX_TOKENS_SEQUENCE}
num_experts=${NUM_EXPERTS}
max_layers=${MAX_LAYERS}
tail_start=${TAIL_START}
tail_experts=${TAIL_EXPERTS}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_bank_count=${RESERVED_BANK_COUNT}
coverage_trace=${COVERAGE_TRACE}
request_sleep_seconds=${REQUEST_SLEEP_SECONDS}
request_sync_between=${REQUEST_SYNC_BETWEEN}
uvm_srcversion=$(modinfo -F srcversion nvidia_uvm 2>/dev/null || true)
uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 2>/dev/null || true)
uvm_moe_slot_pool_vidmem_budget_bytes=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes 2>/dev/null || true)
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
        "policy": "tail_reserved_two_request_probe",
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
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=prefetch_cpu \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${TAIL_EXPERTS}" \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${INIT_TRACE}" \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --max-tokens-sequence "${MAX_TOKENS_SEQUENCE}" \
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

python3 - "${RUN_ROOT}" "${rc}" "${RESULT}" "${PROGRESS}" "${COVERAGE_FILE}" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rc = int(sys.argv[2])
result_path = Path(sys.argv[3])
progress_path = Path(sys.argv[4])
coverage_path = Path(sys.argv[5])

progress = []
if progress_path.exists():
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            progress.append(json.loads(line))
        except json.JSONDecodeError:
            pass

coverage_records = 0
coverage_bad = 0
missing_records = 0
failed_records = 0
last_coverage = None
if coverage_path.exists():
    for line in coverage_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        coverage_records += 1
        last_coverage = record
        if not record.get("coverage_ok", False):
            coverage_bad += 1
        if record.get("missing_reserved_experts"):
            missing_records += 1
        if record.get("failed_reserved_experts"):
            failed_records += 1

result = None
if result_path.exists():
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result = None

summary = {
    "returncode": rc,
    "result_exists": result_path.exists(),
    "result_status": result.get("status") if isinstance(result, dict) else None,
    "completed_requests": result.get("completed_requests") if isinstance(result, dict) else None,
    "request_results": result.get("request_results") if isinstance(result, dict) else None,
    "progress_actions": [item.get("action") for item in progress],
    "last_progress": progress[-1] if progress else None,
    "coverage_records": coverage_records,
    "coverage_bad": coverage_bad,
    "missing_records": missing_records,
    "failed_records": failed_records,
    "last_coverage": last_coverage,
}
summary_path = run_root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

exit "${rc}"
