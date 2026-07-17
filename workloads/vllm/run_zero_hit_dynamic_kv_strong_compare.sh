#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VLLM_ROOT="${SCRIPT_DIR}/vllm"
UVM_TEST_DIR="${VLLM_ROOT}/uvm_test"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
SOURCE_ROOT="${SOURCE_ROOT:-/tmp/vllm_sharegpt_zero_hit_512k_reserved_codex_20260713_230900}"
FIXTURE_SOURCE="${FIXTURE_SOURCE:-${SOURCE_ROOT}/sharegpt_32x64_fixture.json}"
PLAN_SOURCE="${PLAN_SOURCE:-${SOURCE_ROOT}/zero_hit_offload_plan.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_zero_hit_dynamic_kv_strong_compare_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

NUM_REQUESTS="${NUM_REQUESTS:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-2147483648}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

INITIAL_TARGET_BLOCKS="${INITIAL_TARGET_BLOCKS:-128}"
GROW_TARGET_BLOCKS="${GROW_TARGET_BLOCKS:-384}"
GROW_AFTER_COMPLETED_REQUESTS="${GROW_AFTER_COMPLETED_REQUESTS:-4}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
KV_DATA_OFFLOAD_MODE="${KV_DATA_OFFLOAD_MODE:-prefetch_cpu}"
KV_PREFETCH_BACK_MODE="${KV_PREFETCH_BACK_MODE:-prefetch}"
KV_DRIVER_TAG_ENABLE="${KV_DRIVER_TAG_ENABLE:-0}"
KV_FORCE_DRIVER_TAG="${KV_FORCE_DRIVER_TAG:-0}"
KV_POINTER_TAG_GRANULARITY="${KV_POINTER_TAG_GRANULARITY:-tensor}"
KV_PRE_SERVICE_APPLY="${KV_PRE_SERVICE_APPLY:-0}"
KV_PRE_SERVICE_GROW_TARGET_BLOCKS="${KV_PRE_SERVICE_GROW_TARGET_BLOCKS:-0}"
KV_IN_PROCESS_GROW_AFTER_REQUEST="${KV_IN_PROCESS_GROW_AFTER_REQUEST:--1}"
KV_IN_PROCESS_GROW_TARGET_BLOCKS="${KV_IN_PROCESS_GROW_TARGET_BLOCKS:-0}"
RUN_CASES="${RUN_CASES:-no_offload,zero_hit_offload}"

mkdir -p "${RUN_ROOT}"

FIXTURE="${RUN_ROOT}/sharegpt_fixture.json"
PLAN="${RUN_ROOT}/zero_hit_offload_plan.json"
cp "${FIXTURE_SOURCE}" "${FIXTURE}"
cp "${PLAN_SOURCE}" "${PLAN}"
python3 - "${FIXTURE}" "${NUM_REQUESTS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
limit = int(sys.argv[2])
payload = json.loads(path.read_text(encoding="utf-8"))
requests = payload.get("requests")
if isinstance(requests, list) and limit > 0:
    payload["requests"] = requests[:limit]
    metadata = dict(payload.get("metadata") or {})
    metadata["num_requests"] = len(payload["requests"])
    metadata["truncated_by"] = "run_zero_hit_dynamic_kv_strong_compare"
    payload["metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
PY

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

require_clean_gpu() {
  local label="$1"
  local used
  used="$(gpu_used_mib || echo 999999)"
  if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -gt 1024 ]]; then
    echo "GPU memory is not clean before ${label}: ${used} MiB used" | tee "${RUN_ROOT}/blocked_${label}.txt"
    snapshot_gpu "blocked_${label}"
    exit 3
  fi
}

write_env_summary() {
  cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
model=${MODEL}
fixture_source=${FIXTURE_SOURCE}
plan_source=${PLAN_SOURCE}
num_requests=${NUM_REQUESTS}
max_model_len=${MAX_MODEL_LEN}
max_num_seqs=${MAX_NUM_SEQS}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
kv_cache_memory_bytes=${KV_CACHE_MEMORY_BYTES}
initial_target_blocks=${INITIAL_TARGET_BLOCKS}
grow_target_blocks=${GROW_TARGET_BLOCKS}
grow_after_completed_requests=${GROW_AFTER_COMPLETED_REQUESTS}
target_roles=${TARGET_ROLES}
expert_envelope_bytes=${EXPERT_ENVELOPE_BYTES}
kv_data_offload_mode=${KV_DATA_OFFLOAD_MODE}
kv_prefetch_back_mode=${KV_PREFETCH_BACK_MODE}
kv_driver_tag_enable=${KV_DRIVER_TAG_ENABLE}
kv_force_driver_tag=${KV_FORCE_DRIVER_TAG}
kv_pointer_tag_granularity=${KV_POINTER_TAG_GRANULARITY}
kv_pre_service_apply=${KV_PRE_SERVICE_APPLY}
kv_pre_service_grow_target_blocks=${KV_PRE_SERVICE_GROW_TARGET_BLOCKS}
kv_in_process_grow_after_request=${KV_IN_PROCESS_GROW_AFTER_REQUEST}
kv_in_process_grow_target_blocks=${KV_IN_PROCESS_GROW_TARGET_BLOCKS}
run_cases=${RUN_CASES}
uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)
EOF
}

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

start_control_watcher() {
  local progress="$1"
  local control_file="$2"
  local log_file="$3"
  python3 - "${progress}" "${control_file}" "${GROW_AFTER_COMPLETED_REQUESTS}" "${GROW_TARGET_BLOCKS}" > "${log_file}" 2>&1 <<'PY' &
import json
import sys
import time
from pathlib import Path

progress = Path(sys.argv[1])
control = Path(sys.argv[2])
threshold = int(sys.argv[3])
target = int(sys.argv[4])

control.parent.mkdir(parents=True, exist_ok=True)
control.write_text("", encoding="utf-8")
seen = 0
deadline = time.time() + 900
while time.time() < deadline:
    if progress.exists():
        completed = 0
        for line in progress.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("action") == "request_end":
                completed += 1
        if completed != seen:
            print(f"completed_requests={completed}", flush=True)
            seen = completed
        if completed >= threshold:
            control.write_text(f"target_blocks={target}\n", encoding="utf-8")
            print(f"wrote target_blocks={target} after completed_requests={completed}", flush=True)
            raise SystemExit(0)
    time.sleep(0.25)
print("watcher timed out without writing target", flush=True)
raise SystemExit(2)
PY
  echo $!
}

run_case() {
  local name="$1"
  local offload="$2"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  local result="${dir}/result.json"
  local progress="${dir}/progress.jsonl"
  local run_log="${dir}/run.log"
  local kv_trace="${dir}/kv_runtime_trace.jsonl"
  local residency_trace="${dir}/residency_trace.jsonl"
  local real_residency="${dir}/real_residency_trace.jsonl"
  local kv_physical_trace="${dir}/kv_physical_trace.jsonl"
  local pointer_map="${dir}/kv_pointer_map.jsonl"
  local init_trace="${dir}/init_trace.jsonl"
  local control_file="${dir}/kv_dynamic_pool.control"
  local watcher_log="${dir}/control_watcher.log"

  require_clean_gpu "${name}"
  snapshot_gpu "before_${name}"

  local watcher_pid
  watcher_pid="$(start_control_watcher "${progress}" "${control_file}" "${watcher_log}")"

  set +e
  if [[ "${offload}" == "1" ]]; then
    (
      cd "${VLLM_ROOT}"
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_UVM_KV_RUNTIME_ENABLE=1 \
      VLLM_UVM_KV_RUNTIME_MODE=trace_only \
      VLLM_UVM_KV_RUNTIME_TRACE_FILE="${kv_trace}" \
      VLLM_UVM_KV_PHYSICAL_TRACE_FILE="${kv_physical_trace}" \
      VLLM_UVM_KV_PHYSICAL_TRACE_FSYNC=1 \
      VLLM_UVM_KV_PHYSICAL_TRACE_RESIDENCY=1 \
      VLLM_UVM_KV_PHYSICAL_TRACE_MANAGED_PROBE=1 \
	      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
	      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_TARGET_BLOCKS}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE="${control_file}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY="${KV_PRE_SERVICE_APPLY}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS="${INITIAL_TARGET_BLOCKS}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS="${KV_PRE_SERVICE_GROW_TARGET_BLOCKS}" \
	      VLLM_UVM_RESIDENCY_STAGE=stage_l6 \
	      VLLM_UVM_RESIDENCY_TRACE_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_TRACE_FILE="${residency_trace}" \
	      VLLM_UVM_RESIDENCY_TAG_ONLY=0 \
	      VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE="${KV_DATA_OFFLOAD_MODE}" \
	      VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE="${KV_PREFETCH_BACK_MODE}" \
	      VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE="${pointer_map}" \
	      VLLM_UVM_RESIDENCY_KV_FORCE_DRIVER_TAG="${KV_FORCE_DRIVER_TAG}" \
	      VLLM_UVM_RESIDENCY_KV_POINTER_TAG_GRANULARITY="${KV_POINTER_TAG_GRANULARITY}" \
	      VLLM_UVM_REAL_RESIDENCY_ENABLE=1 \
      VLLM_UVM_REAL_RESIDENCY_TRACE_FILE="${real_residency}" \
      VLLM_UVM_REAL_RESIDENCY_DEVICE=0 \
      VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL=0 \
      VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS=32 \
      VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY=0 \
      VLLM_UVM_DRIVER_TAG_ENABLE=1 \
      VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
      VLLM_UVM_DRIVER_KV_TAG_ENABLE="${KV_DRIVER_TAG_ENABLE}" \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=0 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=128 \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${init_trace}" \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS=0 \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
      VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
      timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${UVM_TEST_DIR}/vllm_generate_sequence.py" \
        --model "${MODEL}" \
        --request-fixture-json "${FIXTURE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
	        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
	        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
	        --kv-dynamic-pool-grow-after-request "${KV_IN_PROCESS_GROW_AFTER_REQUEST}" \
	        --kv-dynamic-pool-grow-target-blocks "${KV_IN_PROCESS_GROW_TARGET_BLOCKS}" \
	        --output-json "${result}" \
	        --progress-jsonl "${progress}"
    ) 2>&1 | tee "${run_log}"
  else
    (
      cd "${VLLM_ROOT}"
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=1 \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_UVM_KV_RUNTIME_ENABLE=1 \
      VLLM_UVM_KV_RUNTIME_MODE=trace_only \
      VLLM_UVM_KV_RUNTIME_TRACE_FILE="${kv_trace}" \
      VLLM_UVM_KV_PHYSICAL_TRACE_FILE="${kv_physical_trace}" \
      VLLM_UVM_KV_PHYSICAL_TRACE_FSYNC=1 \
      VLLM_UVM_KV_PHYSICAL_TRACE_RESIDENCY=1 \
      VLLM_UVM_KV_PHYSICAL_TRACE_MANAGED_PROBE=1 \
	      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
	      VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_TARGET_BLOCKS}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE="${control_file}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY="${KV_PRE_SERVICE_APPLY}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS="${INITIAL_TARGET_BLOCKS}" \
	      VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS="${KV_PRE_SERVICE_GROW_TARGET_BLOCKS}" \
	      VLLM_UVM_RESIDENCY_STAGE=stage_l6 \
	      VLLM_UVM_RESIDENCY_TRACE_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_TRACE_FILE="${residency_trace}" \
	      VLLM_UVM_RESIDENCY_TAG_ONLY=0 \
	      VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE="${KV_DATA_OFFLOAD_MODE}" \
	      VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE=1 \
	      VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE="${KV_PREFETCH_BACK_MODE}" \
	      VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE="${pointer_map}" \
	      VLLM_UVM_RESIDENCY_KV_FORCE_DRIVER_TAG="${KV_FORCE_DRIVER_TAG}" \
	      VLLM_UVM_RESIDENCY_KV_POINTER_TAG_GRANULARITY="${KV_POINTER_TAG_GRANULARITY}" \
	      VLLM_UVM_REAL_RESIDENCY_ENABLE=1 \
      VLLM_UVM_REAL_RESIDENCY_TRACE_FILE="${real_residency}" \
      VLLM_UVM_REAL_RESIDENCY_DEVICE=0 \
      VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL=0 \
      VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS=32 \
      VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY=0 \
      VLLM_UVM_DRIVER_TAG_ENABLE=1 \
      VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
      VLLM_UVM_DRIVER_KV_TAG_ENABLE="${KV_DRIVER_TAG_ENABLE}" \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
      timeout "${TIMEOUT_SEC}s" ${VLLM_PY} "${UVM_TEST_DIR}/vllm_generate_sequence.py" \
        --model "${MODEL}" \
        --request-fixture-json "${FIXTURE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
	        --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
	        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
	        --kv-dynamic-pool-grow-after-request "${KV_IN_PROCESS_GROW_AFTER_REQUEST}" \
	        --kv-dynamic-pool-grow-target-blocks "${KV_IN_PROCESS_GROW_TARGET_BLOCKS}" \
	        --output-json "${result}" \
	        --progress-jsonl "${progress}"
    ) 2>&1 | tee "${run_log}"
  fi
  local rc=${PIPESTATUS[0]}
  set -e

  if kill -0 "${watcher_pid}" 2>/dev/null; then
    kill "${watcher_pid}" 2>/dev/null || true
    wait "${watcher_pid}" 2>/dev/null || true
  else
    wait "${watcher_pid}" 2>/dev/null || true
  fi
  echo "${rc}" > "${dir}/returncode.txt"
  snapshot_gpu "after_${name}"
}

analyze() {
  python3 - "${RUN_ROOT}" "${INITIAL_TARGET_BLOCKS}" "${GROW_TARGET_BLOCKS}" "${EXPERT_ENVELOPE_BYTES}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
initial = int(sys.argv[2])
target = int(sys.argv[3])
expert_envelope_bytes = int(sys.argv[4])

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
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records

def metrics(result: dict) -> dict:
    requests = result.get("request_results") or []
    req_seconds = [float(r.get("generate_seconds") or 0.0) for r in requests]
    output_tokens = sum(int(r.get("output_tokens") or 0) for r in requests)
    total = float(result.get("total_generate_seconds") or sum(req_seconds) or 0.0)
    return {
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "output_tokens": output_tokens,
        "total_generate_seconds": total,
        "tokens_per_second": output_tokens / total if total else None,
        "p50_request_seconds": statistics.median(req_seconds) if req_seconds else None,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
        "error": result.get("error"),
    }

def plan_residency(records: list[dict], marker: str) -> dict:
    selected = [
        r for r in records
        if r.get("action") == "init_weight_plan_residency_snapshot"
        and r.get("marker") == marker
    ]
    out = {"records": len(selected)}
    for key in (
        "cold_plan_expert_records",
        "cold_plan_range_records",
        "cold_plan_queried_bytes",
        "cold_plan_gpu_bytes",
        "cold_plan_cpu_bytes",
        "cold_plan_unknown_bytes",
    ):
        out[key] = sum(int(r.get(key) or 0) for r in selected)
    return out

def summarize_case(name: str) -> dict:
    case_dir = root / name
    result = read_json(case_dir / "result.json")
    kv_records = read_jsonl(case_dir / "kv_runtime_trace.jsonl")
    residency = read_jsonl(case_dir / "residency_trace.jsonl")
    init = read_jsonl(case_dir / "init_trace.jsonl")
    real = read_jsonl(case_dir / "real_residency_trace.jsonl")
    physical_trace = read_jsonl(case_dir / "kv_physical_trace.jsonl")
    rc_text = (case_dir / "returncode.txt").read_text(encoding="utf-8").strip() if (case_dir / "returncode.txt").exists() else ""
    try:
        rc = int(rc_text)
    except ValueError:
        rc = None

    resizes = [r for r in kv_records if r.get("action") == "kv_dynamic_pool_resize"]
    grows = [r for r in resizes if r.get("direction") == "grow"]
    shrinks = [r for r in resizes if r.get("direction") == "shrink"]
    target_updates = [r for r in kv_records if r.get("action") == "kv_dynamic_pool_target_update"]
    alloc_no_free = [r for r in kv_records if r.get("action") == "allocation_no_free_blocks"]

    prefetch_issued = [
        r for r in residency if r.get("action") == "residency_kv_prefetch_back_issued"
    ]
    prefetch_fallback = [
        r for r in residency if r.get("action") == "residency_kv_prefetch_back_fallback"
    ]
    prefetch_physical = [
        r for r in prefetch_issued
        if bool((r.get("metadata") or {}).get("physical_migration_issued"))
    ]
    prefetch_logical_only = [
        r for r in prefetch_issued
        if not bool((r.get("metadata") or {}).get("physical_migration_issued"))
    ]
    offload_issued = [
        r for r in residency if r.get("action") == "residency_kv_data_offload_issued"
    ]
    offload_physical = [
        r for r in offload_issued
        if bool((r.get("metadata") or {}).get("physical_migration_issued"))
    ]
    offload_fallback = [
        r for r in residency if r.get("action") == "residency_kv_data_offload_fallback"
    ]

    before = plan_residency(init, "before_stage_m")
    after = plan_residency(init, "after_stage_m_clear_before_prealloc")
    if after["records"] == 0:
        after = plan_residency(init, "after_stage_m_clear")
    release_entries = []
    for r in init:
        if r.get("action") != "init_weight_offload_driver_release_issued":
            continue
        entry = r.get("driver_release_entry") or {}
        nested = entry.get("entries")
        if isinstance(nested, list):
            release_entries.extend([e for e in nested if isinstance(e, dict)])
        elif isinstance(entry, dict):
            release_entries.append(entry)
    release_failed = [e for e in release_entries if not e.get("ok")]
    release_bytes = sum(int(e.get("released_bytes") or 0) for e in release_entries)
    physical_begins = [
        r for r in physical_trace
        if r.get("action") == "kv_physical_migration_boundary"
        and r.get("phase") == "begin"
    ]
    physical_ends = [
        r for r in physical_trace
        if r.get("action") == "kv_physical_migration_boundary"
        and r.get("phase") == "end"
    ]
    managed_probe_failures = [
        r for r in physical_trace
        if int(r.get("managed_probe_count") or 0)
        and int(r.get("managed_probe_success_count") or 0) != int(r.get("managed_probe_count") or 0)
    ]
    residency_before = [r for r in physical_begins if isinstance(r.get("residency"), dict)]
    residency_after = [r for r in physical_ends if isinstance(r.get("residency"), dict)]
    physical_cpu_success = [
        r for r in physical_ends
        if r.get("direction") == "cpu"
        and int(r.get("accepted_ranges") or 0) == int(r.get("range_count") or 0)
        and int((r.get("residency") or {}).get("cpu_bytes") or 0) == int(r.get("bytes") or 0)
        and int((r.get("residency") or {}).get("gpu_bytes") or 0) == 0
    ]
    physical_gpu_success = [
        r for r in physical_ends
        if r.get("direction") == "gpu"
        and int(r.get("accepted_ranges") or 0) == int(r.get("range_count") or 0)
        and int((r.get("residency") or {}).get("gpu_bytes") or 0) == int(r.get("bytes") or 0)
        and int((r.get("residency") or {}).get("cpu_bytes") or 0) == 0
    ]

    real_last = real[-1] if real else {}
    real_by_pool = real_last.get("by_pool") if isinstance(real_last, dict) else {}

    expected_growth = max(target - initial, 0)
    grow_prefetched = sum(int(r.get("prefetched_blocks") or 0) for r in grows)
    physical_prefetch_blocks = len(prefetch_physical)
    hard_physical_prefetch_blocks = len(physical_gpu_success)
    passed_physical_grow = (
        rc == 0
        and metrics(result)["status"] == "ok"
        and grow_prefetched >= expected_growth
        and (
            physical_prefetch_blocks >= expected_growth
            or hard_physical_prefetch_blocks >= expected_growth
        )
        and not prefetch_fallback
        and max(len(physical_begins) - len(physical_ends), 0) == 0
    )

    return {
        "name": name,
        "returncode": rc,
        "result": metrics(result),
        "dynamic_pool": {
            "initial_target_blocks": initial,
            "grow_target_blocks": target,
            "expected_growth_blocks": expected_growth,
            "resize_records": len(resizes),
            "shrink_records": len(shrinks),
            "grow_records": len(grows),
            "target_update_records": len(target_updates),
            "last_grow": grows[-1] if grows else None,
            "logical_prefetched_blocks": grow_prefetched,
            "allocation_no_free_blocks_records": len(alloc_no_free),
            "allocation_no_free_blocks_total": sum(int(r.get("needed_blocks") or r.get("num_blocks") or 0) for r in alloc_no_free),
        },
        "kv_physical": {
            "data_offload_issued": len(offload_issued),
            "data_offload_physical_issued": len(offload_physical),
            "data_offload_fallback": len(offload_fallback),
            "prefetch_back_issued": len(prefetch_issued),
            "prefetch_back_physical_issued": physical_prefetch_blocks,
            "prefetch_back_logical_only": len(prefetch_logical_only),
            "prefetch_back_fallback": len(prefetch_fallback),
            "prefetch_back_bytes_physical": sum(int(r.get("bytes") or 0) for r in prefetch_physical),
            "prefetch_back_bytes_logical_only": sum(int(r.get("bytes") or 0) for r in prefetch_logical_only),
            "physical_trace_records": len(physical_trace),
            "physical_trace_begin_records": len(physical_begins),
            "physical_trace_end_records": len(physical_ends),
            "physical_trace_unmatched_begin_records": max(len(physical_begins) - len(physical_ends), 0),
            "physical_trace_cpu_success_records": len(physical_cpu_success),
            "physical_trace_gpu_success_records": hard_physical_prefetch_blocks,
            "physical_trace_last_record": physical_trace[-1] if physical_trace else None,
            "managed_probe_failure_records": len(managed_probe_failures),
            "first_managed_probe_failure": managed_probe_failures[0] if managed_probe_failures else None,
            "first_residency_before": residency_before[0] if residency_before else None,
            "last_residency_after": residency_after[-1] if residency_after else None,
        },
        "expert_release": {
            "entries": len(release_entries),
            "failed_entries": len(release_failed),
            "released_bytes": release_bytes,
            "before": before,
            "after": after,
            "gpu_bytes_freed_by_query": int(before.get("cold_plan_gpu_bytes") or 0) - int(after.get("cold_plan_gpu_bytes") or 0),
            "envelope_bytes": expert_envelope_bytes,
        },
        "real_residency_last": {
            "present": bool(real_last),
            "ok": real_last.get("ok") if isinstance(real_last, dict) else None,
            "trigger": real_last.get("trigger") if isinstance(real_last, dict) else None,
            "gpu_bytes": real_last.get("gpu_bytes") if isinstance(real_last, dict) else None,
            "cpu_bytes": real_last.get("cpu_bytes") if isinstance(real_last, dict) else None,
            "by_pool": real_by_pool,
        },
        "passed_physical_grow": passed_physical_grow,
    }

summary = {
    "run_root": str(root),
    "initial_target_blocks": initial,
    "grow_target_blocks": target,
    "cases": {
        "no_offload": summarize_case("no_offload"),
        "zero_hit_offload": summarize_case("zero_hit_offload"),
    },
}

summary_path = root / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def fmt(v, digits=3):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)

rows = []
for case_name in ("no_offload", "zero_hit_offload"):
    case = summary["cases"][case_name]
    rows.append(
        "| {name} | {completed} | {tps} | {grow} | {phys} | {fallback} | {release_mib} | {passed} |".format(
            name=case_name,
            completed=case["result"]["completed_requests"],
            tps=fmt(case["result"]["tokens_per_second"]),
            grow=case["dynamic_pool"]["logical_prefetched_blocks"],
            phys=max(
                case["kv_physical"]["prefetch_back_physical_issued"],
                case["kv_physical"]["physical_trace_gpu_success_records"],
            ),
            fallback=case["kv_physical"]["prefetch_back_fallback"],
            release_mib=fmt(case["expert_release"]["released_bytes"] / (1024 * 1024)),
            passed=case["passed_physical_grow"],
        )
    )

report = f"""# Zero-Hit Expert Offload Dynamic KV Strong Compare

Root: `{root}`

Configuration:

```text
initial_target_blocks = {initial}
grow_target_blocks = {target}
expected_growth_blocks = {max(target - initial, 0)}
bytes_per_block ~= 1.5 MiB for this Qwen run
```

| case | completed | tokens/s | logical grow blocks | physical prefetch blocks | prefetch fallback | expert released MiB | physical grow pass |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

Detailed JSON: `{summary_path}`
"""
(root / "report.md").write_text(report, encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
print(report)
PY
}

write_env_summary
snapshot_gpu "start"
if should_run_case "no_offload"; then
  run_case "no_offload" "0"
fi
if should_run_case "zero_hit_offload"; then
  run_case "zero_hit_offload" "1"
fi
snapshot_gpu "end"
analyze
