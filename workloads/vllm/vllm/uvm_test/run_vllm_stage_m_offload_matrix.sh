#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
PROMPT="${PROMPT:-Compare CUDA streams and events in one concise paragraph.}"
MAX_TOKENS="${MAX_TOKENS:-8}"
REPEATS="${REPEATS:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
COUNTS="${COUNTS:-2 4 6 8 10}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_offload_matrix_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
MAX_LAYERS="${MAX_LAYERS:-48}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-1073741824}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
RUNTIME_RESERVED="${RUNTIME_RESERVED:-0}"
PREALLOC_RESERVED_SLOTS="${PREALLOC_RESERVED_SLOTS:-40}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-4194304}"
RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-40}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-10}"
OFFLOAD_MODE="${OFFLOAD_MODE:-prefetch_cpu}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
REQUIRE_ROUTING_PLAN="${REQUIRE_ROUTING_PLAN:-0}"
INIT_WEIGHT_OFFLOAD_STAGE="${INIT_WEIGHT_OFFLOAD_STAGE:-pre_warmup}"

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

summarize_trace() {
  local trace="$1"
  local output="$2"
  python3 - "$trace" "$output" <<'PY'
import collections
import json
import sys
from pathlib import Path

trace_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
counts = collections.Counter()
issued_bytes = 0
if trace_path.exists():
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = record.get("action")
            if isinstance(action, str):
                counts[action] += 1
            for key in ("issued_bytes", "accepted_bytes", "bytes"):
                value = record.get(key)
                if isinstance(value, int) and action and "offload" in action:
                    issued_bytes += value
                    break
summary = {
    "trace": str(trace_path),
    "actions": dict(sorted(counts.items())),
    "offload_related_bytes_seen": issued_bytes,
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

run_generate() {
  local output_json="$1"
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${SCRIPT_DIR}/vllm_generate_once.py" \
    --model "${MODEL}" \
    --prompt "${PROMPT}" \
    --max-tokens "${MAX_TOKENS}" \
    --repeats "${REPEATS}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${output_json}"
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

used="$(gpu_used_mib || echo 999999)"
if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -gt 1024 ]]; then
  echo "GPU memory is not clean: ${used} MiB used. Refusing to run matrix." | tee "${RUN_ROOT}/blocked.txt"
  snapshot_gpu "blocked"
  exit 3
fi

snapshot_gpu "start"

echo "[0/3] optional reserved pool prealloc: ${PREALLOC_RESERVED_SLOTS} slots"
prealloc_reserved_pool
snapshot_gpu "after_prealloc"

echo "[1/3] routing warmup"
ROUTING_TRACE="${RUN_ROOT}/routing_trace.jsonl"
(
  cd "${VLLM_ROOT}"
  VLLM_USE_UVM=1 \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_FILE="${ROUTING_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
  run_generate "${RUN_ROOT}/warmup_result.json"
) 2>&1 | tee "${RUN_ROOT}/warmup.log"

echo "[2/3] Stage-M offload count matrix: ${COUNTS}"
SUMMARY_JSONL="${RUN_ROOT}/matrix_summary.jsonl"
: > "${SUMMARY_JSONL}"
for count in ${COUNTS}; do
  CASE_DIR="${RUN_ROOT}/offload_${count}"
  mkdir -p "${CASE_DIR}"
  PLAN="${CASE_DIR}/cold_${count}_plan.json"
  TRACE="${CASE_DIR}/trace.jsonl"
  INIT_TRACE="${CASE_DIR}/init_trace.jsonl"
  ROUTING_OUT="${CASE_DIR}/routing_trace.jsonl"
  RESULT="${CASE_DIR}/result.json"
  echo "== count=${count} =="
  python3 "${SCRIPT_DIR}/build_cold_offload_plan_from_routing_trace.py" \
    --routing-trace "${ROUTING_TRACE}" \
    --output "${PLAN}" \
    --num-experts "${NUM_EXPERTS}" \
    --max-layers "${MAX_LAYERS}" \
    --max-experts-per-layer "${count}" | tee "${CASE_DIR}/plan_build.log"

  set +e
  (
    cd "${VLLM_ROOT}"
    VLLM_USE_UVM=1 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
    VLLM_UVM_MOE_ROUTING_TRACE_FILE="${ROUTING_OUT}" \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=1 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE="${TRACE}" \
    VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${INIT_TRACE}" \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE="${RUNTIME_RESERVED}" \
    VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
    VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
    VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
    VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
    VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
    VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
    VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
    VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER}" \
    VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS=2 \
    VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
    VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
    VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE="${OFFLOAD_MODE}" \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE="${INIT_WEIGHT_OFFLOAD_STAGE}" \
    VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
    VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN="${REQUIRE_ROUTING_PLAN}" \
    VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${count}" \
    VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
    VLLM_UVM_INIT_WEIGHT_SYNC=1 \
    run_generate "${RESULT}"
  ) 2>&1 | tee "${CASE_DIR}/run.log"
  rc=${PIPESTATUS[0]}
  set -e

  summarize_trace "${TRACE}" "${CASE_DIR}/trace_summary.json" | tee "${CASE_DIR}/trace_summary.log" || true
  summarize_trace "${INIT_TRACE}" "${CASE_DIR}/init_trace_summary.json" | tee "${CASE_DIR}/init_trace_summary.log" || true
  snapshot_gpu "after_count_${count}"

  python3 - "$SUMMARY_JSONL" "$count" "$rc" "$RESULT" "$CASE_DIR/trace_summary.json" "$CASE_DIR/init_trace_summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
count = int(sys.argv[2])
rc = int(sys.argv[3])
result_path = Path(sys.argv[4])
trace_summary_path = Path(sys.argv[5])
init_summary_path = Path(sys.argv[6])
record = {"count": count, "returncode": rc, "ok": rc == 0}
if result_path.exists():
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        record.update({
            "generate_seconds": result.get("generate_seconds"),
            "repeat_avg_generate_seconds": result.get("repeat_avg_generate_seconds"),
            "output_tokens": result.get("output_tokens"),
            "gpu_memory_after_load": result.get("gpu_memory_after_load"),
            "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
        })
    except Exception as exc:
        record["result_error"] = str(exc)
for key, path in (("trace", trace_summary_path), ("init_trace", init_summary_path)):
    if path.exists():
        try:
            record[key] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            record[f"{key}_error"] = str(exc)
with summary_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
PY

  if [[ "${rc}" -ne 0 ]]; then
    echo "count=${count} failed or timed out with rc=${rc}; stopping matrix to avoid polluted follow-up runs." | tee -a "${RUN_ROOT}/failed.txt"
    exit "${rc}"
  fi
done

echo "[3/3] matrix complete"
cat "${SUMMARY_JSONL}"
