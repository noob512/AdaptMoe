#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${SCRIPT_DIR}/vllm"
UVM_TEST_DIR="${VLLM_ROOT}/uvm_test"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_cold8_reserved_slot_failfast_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

NUM_REQUESTS="${NUM_REQUESTS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SHAREGPT_SEED="${SHAREGPT_SEED:-42}"

NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-8}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"

RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT:-48}"
RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY:-dynamic}"
RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT:-1}"
RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES:-2097152}"
RESERVED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER:-8}"
RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS:-4}"
RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE:-1}"
EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE:-1}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-2097152}"
EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT="${EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT:-1}"
PRE_SERVICE_MANAGED="${PRE_SERVICE_MANAGED:-0}"
EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED:-1}"
KV_CACHE_MANAGED="${KV_CACHE_MANAGED:-1}"

mkdir -p "${RUN_ROOT}"

REQUEST_FIXTURE="${RUN_ROOT}/sharegpt_32x64_fixture.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PILOT_DIR="${RUN_ROOT}/pilot_trace"
MECH_DIR="${RUN_ROOT}/cold8_offload_mechanism"
PLAN="${RUN_ROOT}/cold8_offload_plan.json"
PLAN_STATS="${RUN_ROOT}/cold8_plan_stats.json"
SUMMARY="${RUN_ROOT}/summary.json"
REPORT="${RUN_ROOT}/report.md"

PILOT_TRACE="${PILOT_DIR}/routing_trace.jsonl"
PILOT_RESULT="${PILOT_DIR}/result.json"
PILOT_PROGRESS="${PILOT_DIR}/progress.jsonl"

MECH_RESULT="${MECH_DIR}/result.json"
MECH_PROGRESS="${MECH_DIR}/progress.jsonl"
MECH_INIT_TRACE="${MECH_DIR}/init_trace.jsonl"
MECH_PREFETCH_TRACE="${MECH_DIR}/prefetch_trace.jsonl"
MECH_RESIDENCY_TRACE="${MECH_DIR}/residency_trace.jsonl"
MECH_POST_STAGE_M_PREALLOC_TRACE="${MECH_DIR}/post_stage_m_reserved_prealloc.jsonl"
MECH_RUN_LOG="${MECH_DIR}/run.log"
MECH_XID_BEFORE="${MECH_DIR}/xid_before.log"
MECH_XID_AFTER="${MECH_DIR}/xid_after.log"

mkdir -p "${PILOT_DIR}" "${MECH_DIR}"

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

collect_xid_tail() {
  local out="$1"
  {
    grep -hE "Xid|MMU Fault|faulted @" /var/log/kern.log /var/log/syslog 2>/dev/null || true
  } | tail -200 > "${out}"
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

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True),
                 indent=2, sort_keys=True))
PY
  ) > "${RUN_ROOT}/reserved_pool_reset.json" 2>&1 || true
}

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
dataset_path=${DATASET_PATH}
num_requests=${NUM_REQUESTS}
output_tokens=${OUTPUT_TOKENS}
max_model_len=${MAX_MODEL_LEN}
max_num_seqs=${MAX_NUM_SEQS}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
cold_experts_per_layer=${COLD_EXPERTS_PER_LAYER}
reserved_slot_count=${RESERVED_SLOT_COUNT}
reserved_slot_policy=${RESERVED_SLOT_POLICY}
reserved_bank_count=${RESERVED_BANK_COUNT}
reserved_slot_bytes=${RESERVED_SLOT_BYTES}
expert_envelope_enable=${EXPERT_ENVELOPE_ENABLE}
expert_envelope_bytes=${EXPERT_ENVELOPE_BYTES}
expert_envelope_strict_logical_alignment=${EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT}
reserved_cleanup_delay_steps=${RESERVED_CLEANUP_DELAY_STEPS}
reserved_prefetch_cpu_source=${RESERVED_PREFETCH_CPU_SOURCE}
init_weight_offload_mode=driver_release
pre_service_managed=${PRE_SERVICE_MANAGED}
expert_weights_managed=${EXPERT_WEIGHTS_MANAGED}
kv_cache_managed=${KV_CACHE_MANAGED}
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
snapshot_gpu "start"

(
  cd "${VLLM_ROOT}"
  ${VLLM_PY} - "${DATASET_PATH}" "${MODEL}" "${REQUEST_FIXTURE}" "${FIXTURE_META}" \
    "${NUM_REQUESTS}" "${OUTPUT_TOKENS}" "${MAX_MODEL_LEN}" "${SHAREGPT_SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
meta_path = Path(sys.argv[4])
num_requests = int(sys.argv[5])
output_tokens = int(sys.argv[6])
max_model_len = int(sys.argv[7])
seed = int(sys.argv[8])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
data = json.loads(dataset_path.read_text(encoding="utf-8"))
random.Random(seed).shuffle(data)

requests = []
prompt_token_lengths = []
for item in data:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        continue
    prompt = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in ("human", "user"):
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break
    if not prompt:
        continue
    token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if token_len <= 0 or token_len + output_tokens > max_model_len:
        continue
    requests.append(
        {
            "request": len(requests),
            "prompt": prompt,
            "max_tokens": output_tokens,
            "prompt_tokens": token_len,
        }
    )
    prompt_token_lengths.append(token_len)
    if len(requests) >= num_requests:
        break

if len(requests) < num_requests:
    raise SystemExit(f"only selected {len(requests)} requests")

fixture = {
    "metadata": {
        "source": "ShareGPT_V3_unfiltered_cleaned_split",
        "selection": "shuffled_valid_prompts",
        "seed": seed,
        "num_requests": num_requests,
        "output_tokens": output_tokens,
        "max_model_len": max_model_len,
        "min_prompt_tokens": min(prompt_token_lengths),
        "max_prompt_tokens": max(prompt_token_lengths),
        "avg_prompt_tokens": sum(prompt_token_lengths) / len(prompt_token_lengths),
    },
    "requests": requests,
}
fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
meta_path.write_text(json.dumps(fixture["metadata"], indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps(fixture["metadata"], indent=2, sort_keys=True))
PY
)

set +e
(
  cd "${VLLM_ROOT}"
  VLLM_USE_UVM=0 \
  VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED="${PRE_SERVICE_MANAGED}" \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
  VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_FILE="${PILOT_TRACE}" \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${UVM_TEST_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${PILOT_RESULT}" \
    --progress-jsonl "${PILOT_PROGRESS}"
) 2>&1 | tee "${PILOT_DIR}/run.log"
pilot_rc=${PIPESTATUS[0]}
set -e

if [[ "${pilot_rc}" -ne 0 ]]; then
  echo "pilot failed rc=${pilot_rc}" | tee "${RUN_ROOT}/pilot_failed.txt"
  exit "${pilot_rc}"
fi

python3 - "${PILOT_TRACE}" "${PLAN}" "${PLAN_STATS}" \
  "${NUM_LAYERS}" "${NUM_EXPERTS}" "${COLD_EXPERTS_PER_LAYER}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

trace_path = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
stats_path = Path(sys.argv[3])
num_layers = int(sys.argv[4])
num_experts = int(sys.argv[5])
cold_per_layer = int(sys.argv[6])

def layer_id_from_name(layer_name: Any) -> int | None:
    if not isinstance(layer_name, str):
        return None
    parts = layer_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            try:
                return int(parts[i + 1])
            except ValueError:
                return None
    return None

counts: dict[int, dict[int, int]] = {
    layer: {expert: 0 for expert in range(num_experts)}
    for layer in range(num_layers)
}
total_assignments = 0
records = 0
for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if not line.strip():
        continue
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        continue
    layer = layer_id_from_name(record.get("layer_name"))
    if layer is None or layer < 0 or layer >= num_layers:
        continue
    expert_counts = record.get("expert_token_counts")
    if not isinstance(expert_counts, dict):
        continue
    records += 1
    for expert_raw, count_raw in expert_counts.items():
        try:
            expert = int(expert_raw)
            count = int(count_raw)
        except (TypeError, ValueError):
            continue
        if 0 <= expert < num_experts and count > 0:
            counts[layer][expert] += count
            total_assignments += count

offload_plan = []
selected_by_layer: dict[str, list[int]] = {}
cold_assignments = 0
for layer in range(num_layers):
    selected = [
        expert for expert, _ in sorted(
            counts[layer].items(), key=lambda item: (item[1], item[0])
        )[:cold_per_layer]
    ]
    selected_by_layer[str(layer)] = selected
    cold_assignments += sum(counts[layer][expert] for expert in selected)
    for expert in selected:
        offload_plan.append(
            {
                "layer_id": layer,
                "expert_id": expert,
                "reason": "sharegpt_token_hit_cold8",
                "token_hits": counts[layer][expert],
            }
        )

plan = {
    "metadata": {
        "policy": "sharegpt_token_hit_cold8_v1",
        "routing_trace": str(trace_path),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "cold_experts_per_layer": cold_per_layer,
        "routing_records": records,
        "total_assignments": total_assignments,
        "cold_assignments": cold_assignments,
        "cold_share": (cold_assignments / total_assignments)
        if total_assignments else 0.0,
        "selected_by_layer": selected_by_layer,
    },
    "offload_plan": offload_plan,
    "prefetch_plan": list(offload_plan),
    "offload_plan_records": len(offload_plan),
    "prefetch_plan_records": len(offload_plan),
}
stats = {
    "routing_records": records,
    "offload_plan_records": len(offload_plan),
    "prefetch_plan_records": len(offload_plan),
    "total_assignments": total_assignments,
    "cold_assignments": cold_assignments,
    "cold_share": (cold_assignments / total_assignments)
    if total_assignments else 0.0,
    "selected_by_layer": selected_by_layer,
}
plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
print(json.dumps(stats, indent=2, sort_keys=True))
PY

snapshot_gpu "before_mechanism"
reset_reserved_pool
collect_xid_tail "${MECH_XID_BEFORE}"

set +e
(
  cd "${VLLM_ROOT}"
  rm -f vllm_uvm_weight_regions.jsonl
  VLLM_USE_UVM=0 \
  VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED="${PRE_SERVICE_MANAGED}" \
  VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
  VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
  VLLM_DISABLE_OPENAI_METRICS=1 \
  VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
  VLLM_UVM_RESIDENCY_TAG_ONLY=1 \
  VLLM_UVM_RESIDENCY_TRACE_ENABLE=1 \
  VLLM_UVM_RESIDENCY_TRACE_FILE="${MECH_RESIDENCY_TRACE}" \
  VLLM_UVM_DRIVER_TAG_ENABLE=1 \
  VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=1 \
  VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE="${MECH_PREFETCH_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
  VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
  VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
  VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER}" \
  VLLM_UVM_INIT_WEIGHT_SYNC=1 \
  VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${MECH_INIT_TRACE}" \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=1 \
  VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS=0 \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
  VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
  VLLM_UVM_WEIGHT_PREFETCH_SOURCE=plan_intersection \
  VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=1 \
  VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
  VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER}" \
  VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
  VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_WEIGHT_OFFLOAD_MODE=trace_only \
  VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE="${PLAN}" \
  VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
  VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1 \
  VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE=1 \
  VLLM_UVM_EXPERT_RESERVED_POST_STAGE_M_PREALLOC_FILE="${MECH_POST_STAGE_M_PREALLOC_TRACE}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY="${RESERVED_SLOT_POLICY}" \
  VLLM_UVM_EXPERT_RESERVED_BANK_COUNT="${RESERVED_BANK_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${RESERVED_SLOT_COUNT}" \
  VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${RESERVED_SLOT_BYTES}" \
  VLLM_UVM_EXPERT_ENVELOPE_ENABLE="${EXPERT_ENVELOPE_ENABLE}" \
  VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
  VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT="${EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT}" \
  VLLM_UVM_EXPERT_2MB_ENVELOPE_ENABLE=0 \
  VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER="${RESERVED_EXPERTS_PER_LAYER}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS="${RESERVED_CLEANUP_DELAY_STEPS}" \
  VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=1 \
  VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
  VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
  VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
  VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
  VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE="${RESERVED_PREFETCH_CPU_SOURCE}" \
  timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${UVM_TEST_DIR}/vllm_generate_sequence.py" \
    --model "${MODEL}" \
    --request-fixture-json "${REQUEST_FIXTURE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --output-json "${MECH_RESULT}" \
    --progress-jsonl "${MECH_PROGRESS}"
) 2>&1 | tee "${MECH_RUN_LOG}"
mech_rc=${PIPESTATUS[0]}
set -e

snapshot_gpu "after_mechanism"
collect_xid_tail "${MECH_XID_AFTER}"

python3 - "${RUN_ROOT}" "${PILOT_RESULT}" "${PLAN}" "${PLAN_STATS}" \
  "${MECH_RESULT}" "${MECH_INIT_TRACE}" "${MECH_POST_STAGE_M_PREALLOC_TRACE}" \
  "${MECH_RESIDENCY_TRACE}" "${MECH_RUN_LOG}" "${MECH_XID_BEFORE}" \
  "${MECH_XID_AFTER}" "${SUMMARY}" "${REPORT}" "${mech_rc}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
pilot_result_path = Path(sys.argv[2])
plan_path = Path(sys.argv[3])
plan_stats_path = Path(sys.argv[4])
mech_result_path = Path(sys.argv[5])
init_trace_path = Path(sys.argv[6])
prealloc_path = Path(sys.argv[7])
residency_trace_path = Path(sys.argv[8])
run_log_path = Path(sys.argv[9])
xid_before_path = Path(sys.argv[10])
xid_after_path = Path(sys.argv[11])
summary_path = Path(sys.argv[12])
report_path = Path(sys.argv[13])
mech_rc = int(sys.argv[14])

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": repr(exc)}

def read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records

def metrics(result: dict) -> dict:
    requests = result.get("request_results") or []
    req = [float(item.get("generate_seconds", 0.0) or 0.0)
           for item in requests]
    tokens = sum(int(item.get("output_tokens", 0) or 0)
                 for item in requests)
    total = float(result.get("total_generate_seconds") or sum(req) or 0.0)
    return {
        "status": result.get("status"),
        "completed_requests": result.get("completed_requests"),
        "load_seconds": result.get("load_seconds"),
        "total_generate_seconds": total,
        "avg_request_seconds": (sum(req) / len(req)) if req else None,
        "p50_request_seconds": statistics.median(req) if req else None,
        "request_seconds": req,
        "output_tokens": tokens,
        "tokens_per_second": (tokens / total) if total else None,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
    }

def sum_plan_residency(records: list[dict], marker: str) -> dict:
    selected = [
        r for r in records
        if r.get("action") == "init_weight_plan_residency_snapshot"
        and r.get("marker") == marker
    ]
    fields = [
        "cold_plan_expert_records",
        "cold_plan_range_records",
        "cold_plan_queried_bytes",
        "cold_plan_gpu_bytes",
        "cold_plan_cpu_bytes",
        "cold_plan_unknown_bytes",
    ]
    out = {"records": len(selected)}
    for field in fields:
        out[field] = sum(int(r.get(field) or 0) for r in selected)
    queried = out["cold_plan_queried_bytes"]
    out["cold_plan_gpu_ratio"] = (
        out["cold_plan_gpu_bytes"] / queried if queried else 0.0
    )
    out["cold_plan_cpu_ratio"] = (
        out["cold_plan_cpu_bytes"] / queried if queried else 0.0
    )
    return out

pilot = read_json(pilot_result_path)
plan = read_json(plan_path)
plan_stats = read_json(plan_stats_path)
mech = read_json(mech_result_path)
init_records = read_jsonl(init_trace_path)
residency_records = read_jsonl(residency_trace_path)
prealloc_records = read_jsonl(prealloc_path)
prealloc_last = prealloc_records[-1] if prealloc_records else None
prealloc_result = (prealloc_last or {}).get("result") or {}
prealloc_query = (prealloc_last or {}).get("query_after") or {}
issued_actions = {
    "init_weight_offload_prefetch_cpu_issued",
    "init_weight_offload_offload_cpu_issued",
    "init_weight_offload_driver_release_issued",
}
issued = [r for r in init_records if r.get("action") in issued_actions]
driver_release_issued = [
    r for r in init_records
    if r.get("action") == "init_weight_offload_driver_release_issued"
]
driver_release_attempted = [
    r for r in init_records
    if r.get("mode") == "driver_release" and r.get("driver_release_entry")
]

def driver_release_entries(record: dict) -> list[dict]:
    entry = record.get("driver_release_entry") or {}
    nested = entry.get("entries")
    if isinstance(nested, list):
        return [item for item in nested if isinstance(item, dict)]
    return [entry] if isinstance(entry, dict) and entry else []

def driver_release_record_ok(record: dict) -> bool:
    entries = driver_release_entries(record)
    return bool(entries) and all(bool(entry.get("ok")) for entry in entries)

def first_driver_release_failure(records: list[dict]) -> dict:
    for record in records:
        for entry in driver_release_entries(record):
            if not bool(entry.get("ok")):
                return entry
    return {}

driver_release_issued_entries = [
    entry
    for record in driver_release_issued
    for entry in driver_release_entries(record)
]
driver_release_attempted_entries = [
    entry
    for record in driver_release_attempted
    for entry in driver_release_entries(record)
]
driver_release_failed = [
    r for r in driver_release_attempted
    if not driver_release_record_ok(r)
]
driver_release_released_bytes = sum(
    int(entry.get("released_bytes") or 0)
    for entry in driver_release_issued_entries
)
driver_release_attempted_released_bytes = sum(
    int(entry.get("released_bytes") or 0)
    for entry in driver_release_attempted_entries
)
driver_release_attempted_blocks = sum(
    int(entry.get("attempted_blocks") or 0)
    for entry in driver_release_attempted_entries
)
driver_release_success_blocks = sum(
    int(entry.get("success_blocks") or 0)
    for entry in driver_release_attempted_entries
)
driver_release_failure_blocks = sum(
    int(entry.get("failure_blocks") or 0)
    for entry in driver_release_attempted_entries
)
driver_release_first_failure = first_driver_release_failure(driver_release_failed)
driver_tag_records = [
    r for r in residency_records
    if r.get("action") == "residency_register_region"
]
driver_tag_attempts = [
    r for r in driver_tag_records
    if (r.get("driver_tag") or {}).get("attempted")
]
driver_tag_success = [
    r for r in driver_tag_records
    if (r.get("driver_tag") or {}).get("registered")
]
driver_tag_failures = [
    r for r in driver_tag_attempts
    if not (r.get("driver_tag") or {}).get("registered")
]
before_res = sum_plan_residency(init_records, "before_stage_m")
after_res = sum_plan_residency(
    init_records,
    "after_stage_m_clear_before_prealloc",
)
if after_res.get("records", 0) == 0:
    after_res = sum_plan_residency(init_records, "after_stage_m_clear")
gpu_freed = (
    before_res.get("cold_plan_gpu_bytes", 0)
    - after_res.get("cold_plan_gpu_bytes", 0)
)

xid_before = xid_before_path.read_text(encoding="utf-8", errors="replace") \
    if xid_before_path.exists() else ""
xid_after = xid_after_path.read_text(encoding="utf-8", errors="replace") \
    if xid_after_path.exists() else ""
new_xid = xid_before != xid_after and (
    "Xid" in xid_after or "MMU Fault" in xid_after or "faulted @" in xid_after
)

run_tail = ""
if run_log_path.exists():
    lines = run_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    run_tail = "\n".join(lines[-80:])

prealloc_ok = bool(prealloc_result.get("ok")) and (
    int(prealloc_result.get("success_count") or 0)
    >= int((prealloc_last or {}).get("slot_count") or 0)
) and (
    int(prealloc_query.get("backed_slots") or 0)
    >= int((prealloc_last or {}).get("slot_count") or 0)
)

failure_reason = None
if mech_rc != 0:
    if prealloc_last and not prealloc_ok:
        failure_reason = "reserved_slot_prealloc_failed"
    else:
        failure_reason = "mechanism_run_failed_before_or_during_generation"

summary = {
    "run_root": str(root),
    "mechanism_returncode": mech_rc,
    "failure_reason": failure_reason,
    "pilot": metrics(pilot),
    "mechanism": metrics(mech),
    "plan": {
        "offload_plan_records": len(plan.get("offload_plan") or []),
        "prefetch_plan_records": len(plan.get("prefetch_plan") or []),
        "policy": (plan.get("metadata") or {}).get("policy"),
        **plan_stats,
    },
    "init_offload": {
        "issued_records": len(issued),
        "issued_bytes": sum(int(r.get("bytes") or 0) for r in issued),
        "driver_release_attempted_records": len(driver_release_attempted),
        "driver_release_attempted_entries": len(
            driver_release_attempted_entries
        ),
        "driver_release_issued_records": len(driver_release_issued),
        "driver_release_issued_entries": len(driver_release_issued_entries),
        "driver_release_failed_records": len(driver_release_failed),
        "driver_release_released_bytes": driver_release_released_bytes,
        "driver_release_attempted_released_bytes": (
            driver_release_attempted_released_bytes
        ),
        "driver_release_attempted_blocks": driver_release_attempted_blocks,
        "driver_release_success_blocks": driver_release_success_blocks,
        "driver_release_failure_blocks": driver_release_failure_blocks,
        "driver_release_first_failure": driver_release_first_failure,
    },
    "residency_driver_tag": {
        "records": len(driver_tag_records),
        "attempts": len(driver_tag_attempts),
        "success": len(driver_tag_success),
        "failures": len(driver_tag_failures),
        "first_failure": (
            driver_tag_failures[0].get("driver_tag")
            if driver_tag_failures else None
        ),
    },
    "cold_plan_residency_before_stage_m": before_res,
    "cold_plan_residency_after_stage_m_clear": after_res,
    "cold_plan_gpu_bytes_freed": gpu_freed,
    "post_stage_m_reserved_prealloc": {
        "records": len(prealloc_records),
        "last": prealloc_last,
        "ok": prealloc_ok,
        "result": prealloc_result,
        "query_after": prealloc_query,
    },
    "new_xid_or_mmu_fault": new_xid,
    "run_log_tail": run_tail,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

report = f"""# Cold8 Reserved Slot Fail-Fast Experiment

Root: `{root}`

## Result

- mechanism returncode: `{mech_rc}`
- failure_reason: `{failure_reason}`
- pilot: `{summary['pilot'].get('status')}`, completed `{summary['pilot'].get('completed_requests')}`
- mechanism: `{summary['mechanism'].get('status')}`, completed `{summary['mechanism'].get('completed_requests')}`
- plan records: offload `{summary['plan'].get('offload_plan_records')}`, prefetch `{summary['plan'].get('prefetch_plan_records')}`
- cold share from pilot trace: `{fmt(summary['plan'].get('cold_share'), 6)}`

## Expert Offload Residency

- Stage-M issued records: `{summary['init_offload']['issued_records']}`
- Stage-M issued bytes: `{summary['init_offload']['issued_bytes']}`
- driver release attempted records: `{summary['init_offload']['driver_release_attempted_records']}`
- driver release attempted entries: `{summary['init_offload']['driver_release_attempted_entries']}`
- driver release issued records: `{summary['init_offload']['driver_release_issued_records']}`
- driver release issued entries: `{summary['init_offload']['driver_release_issued_entries']}`
- driver release failed records: `{summary['init_offload']['driver_release_failed_records']}`
- driver release released bytes: `{summary['init_offload']['driver_release_released_bytes']}`
- driver release raw released bytes across attempts: `{summary['init_offload']['driver_release_attempted_released_bytes']}`
- driver release blocks attempted/success/failure: `{summary['init_offload']['driver_release_attempted_blocks']}` / `{summary['init_offload']['driver_release_success_blocks']}` / `{summary['init_offload']['driver_release_failure_blocks']}`
- driver release first failure: `{summary['init_offload']['driver_release_first_failure']}`
- driver tag register attempts/success/failures: `{summary['residency_driver_tag']['attempts']}` / `{summary['residency_driver_tag']['success']}` / `{summary['residency_driver_tag']['failures']}`
- driver tag first failure: `{summary['residency_driver_tag']['first_failure']}`
- cold GPU bytes before Stage-M: `{before_res.get('cold_plan_gpu_bytes')}`
- cold GPU bytes after Stage-M clear: `{after_res.get('cold_plan_gpu_bytes')}`
- cold GPU bytes freed by residency query: `{gpu_freed}`
- cold CPU bytes after Stage-M clear: `{after_res.get('cold_plan_cpu_bytes')}`

## Reserved Slot Prealloc

- prealloc ok: `{prealloc_ok}`
- slot_count: `{(prealloc_last or {}).get('slot_count')}`
- slot_bytes: `{(prealloc_last or {}).get('slot_bytes')}`
- result ok: `{prealloc_result.get('ok')}`
- success_count: `{prealloc_result.get('success_count')}`
- failure_count: `{prealloc_result.get('failure_count')}`
- alloc_failed_count: `{prealloc_result.get('alloc_failed_count')}`
- backed_slots after query: `{prealloc_query.get('backed_slots')}`
- reserved_backing_bytes after query: `{prealloc_query.get('reserved_backing_bytes')}`
- rm_status: `{prealloc_result.get('rm_status')}`

## Performance

- output tokens: `{summary['mechanism'].get('output_tokens')}`
- total generate seconds: `{fmt(summary['mechanism'].get('total_generate_seconds'))}`
- tokens/s: `{fmt(summary['mechanism'].get('tokens_per_second'))}`
- avg request seconds: `{fmt(summary['mechanism'].get('avg_request_seconds'))}`

## Fault Check

- new Xid/MMU fault: `{new_xid}`
"""
report_path.write_text(report, encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

cat "${REPORT}"

if [[ "${mech_rc}" -ne 0 ]]; then
  exit "${mech_rc}"
fi
