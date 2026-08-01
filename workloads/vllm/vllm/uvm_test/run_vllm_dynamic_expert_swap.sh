#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
DATASET="${DATASET:-/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_dynamic_expert_swap_$(date +%Y%m%d_%H%M%S)}"
KV_BYTES="${KV_BYTES:-2362232013}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-10}"
MAX_SWAPS_PER_LAYER="${MAX_SWAPS_PER_LAYER:-10}"
MIN_COUNT_DELTA="${MIN_COUNT_DELTA:-1}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-128}"
PILOT_REQUEST_COUNT="${PILOT_REQUEST_COUNT:-64}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"
RUN_CASES="${RUN_CASES:-pilot_a,control,dynamic}"
B_PHASE_NAMES="${B_PHASE_NAMES:-observe_b,after_b,repeat_b}"
B_PHASE_COUNTS="${B_PHASE_COUNTS:-64,64,64}"
B_PHASE_OFFSETS="${B_PHASE_OFFSETS:-64,64,64}"
B_PHASE_OUTPUTS="${B_PHASE_OUTPUTS:-${OUTPUT_TOKENS},${OUTPUT_TOKENS},${OUTPUT_TOKENS}}"
REMOTE_PREPARE_MODE="${REMOTE_PREPARE_MODE:-native}"
REQUIRE_COMPLETE_REMOTE_PREPARE="${REQUIRE_COMPLETE_REMOTE_PREPARE:-0}"
REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED="${REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED:-0}"
DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE:-1}"
DRIVER_TAG_REQUIRE_MANAGED="${DRIVER_TAG_REQUIRE_MANAGED:-1}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
NVIDIA_SMI_MONITOR_ENABLE="${NVIDIA_SMI_MONITOR_ENABLE:-1}"

FIXTURE="${RUN_ROOT}/sharegpt_shift_128.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PLAN="${RUN_ROOT}/initial_cold_plan.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used=999999
  for _ in $(seq 1 120); do
    used="$(gpu_used_mib || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU is not clean before ${label}: ${used} MiB" >&2
  nvidia-smi >&2 || true
  return 1
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

require_uvm_proactive_migration_disabled() {
  if [[ "${REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED}" != "1" ]]; then
    return 0
  fi

  local parameter
  for parameter in \
    uvm_perf_prefetch_enable \
    uvm_perf_access_counter_migration_enable \
    uvm_perf_thrashing_enable; do
    local path="/sys/module/nvidia_uvm/parameters/${parameter}"
    if [[ ! -r "${path}" ]] || [[ "$(<"${path}")" != "0" ]]; then
      echo "Required UVM parameter ${parameter}=0 is not active" >&2
      return 1
    fi
  done
}

prepare_fixture() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET}" "${MODEL}" "${FIXTURE}" \
      "${FIXTURE_META}" <<'PY'
import json
import sys
from pathlib import Path

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
requests = []
source_indices = []
lengths = []
for index, item in enumerate(dataset):
    conversations = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conversations, list) or not conversations:
        continue
    first = conversations[0]
    prompt = first.get("value") if isinstance(first, dict) else None
    if not isinstance(prompt, str) or not prompt:
        continue
    length = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not 138 <= length <= 178:
        continue
    requests.append(
        {
            "request": len(requests),
            "prompt": prompt,
            "prompt_tokens": length,
            "max_tokens": 128,
        }
    )
    source_indices.append(index)
    lengths.append(length)
    if len(requests) == 128:
        break

if len(requests) != 128:
    raise SystemExit(f"expected 128 requests, found {len(requests)}")

metadata = {
    "source": str(dataset_path),
    "request_count": 128,
    "group_a": {"offset": 0, "count": 64},
    "group_b": {"offset": 64, "count": 64},
    "source_indices": source_indices,
    "total_prompt_tokens": sum(lengths),
    "min_prompt_tokens": min(lengths),
    "max_prompt_tokens": max(lengths),
}
fixture_path.write_text(
    json.dumps({"metadata": metadata, "requests": requests}, indent=2) + "\n",
    encoding="utf-8",
)
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(json.dumps(metadata, indent=2))
PY
  )
}

run_case() {
  local name="$1"
  local mode="$2"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  wait_for_clean_gpu "${name}"
  rm -f "${dir}/result.json" "${dir}/progress.jsonl" \
    "${dir}/routing_trace.jsonl" "${dir}/expert_trace.jsonl" \
    "${dir}/run.log" "${dir}/returncode.txt" "${dir}/dmon_pcie.csv"

  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
    VLLM_UVM_KV_CACHE_MANAGED=0
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_RUNTIME_ENABLE=0
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
  )
  local -a phase_args=()

  if [[ "${mode}" == "pilot" ]]; then
    env_args+=(
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl"
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0
    )
    phase_args=(
      --phase-names pilot_a
      --phase-request-counts "${PILOT_REQUEST_COUNT}"
      --phase-request-offsets 0
      --phase-output-tokens "${OUTPUT_TOKENS}"
    )
  else
    env_args+=(
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
      VLLM_UVM_REMOTE_PREPARE_MODE="${REMOTE_PREPARE_MODE}"
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}"
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=128
      VLLM_UVM_INIT_WEIGHT_SYNC=1
      VLLM_UVM_INIT_WEIGHT_REQUIRE_COMPLETE_REMOTE_PREPARE="${REQUIRE_COMPLETE_REMOTE_PREPARE}"
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/expert_trace.jsonl"
      VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS=1
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
      VLLM_UVM_DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE}"
      VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED="${DRIVER_TAG_REQUIRE_MANAGED}"
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1
      VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}"
      VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1
      VLLM_UVM_REQUIRE_PROACTIVE_MIGRATION_DISABLED="${REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED}"
      VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE=1
      VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
      VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=0
      VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=1
    )
    phase_args=(
      --phase-names "${B_PHASE_NAMES}"
      --phase-request-counts "${B_PHASE_COUNTS}"
      --phase-request-offsets "${B_PHASE_OFFSETS}"
      --phase-output-tokens "${B_PHASE_OUTPUTS}"
      --reset-dynamic-expert-heat-before-phases
    )
    if [[ "${mode}" == "dynamic" ]]; then
      phase_args+=(
        --dynamic-expert-swap-after-phases 0
        --dynamic-expert-swap-max-per-layer "${MAX_SWAPS_PER_LAYER}"
        --dynamic-expert-swap-min-count-delta "${MIN_COUNT_DELTA}"
      )
    fi
  fi

  local dmon_pid=""
  if [[ "${NVIDIA_SMI_MONITOR_ENABLE}" == "1" ]]; then
    nvidia-smi dmon -s t -d 1 -o DT > "${dir}/dmon_pcie.csv" 2>&1 &
    dmon_pid=$!
  fi
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${SCRIPT_DIR}/vllm_generate_phased.py" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --output "${dir}/result.json" \
      --progress "${dir}/progress.jsonl" \
      --grow-after-phase -1 \
      --kv-cache-memory-bytes "${KV_BYTES}" \
      --max-model-len 512 \
      --max-num-seqs 64 \
      --max-num-batched-tokens 2048 \
      --gpu-memory-utilization 0.97 \
      --disable-prefix-caching \
      "${phase_args[@]}"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  stop_pid "${dmon_pid}"
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  wait_for_clean_gpu "after_${name}" || true
  return "${rc}"
}

build_plan() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" "${SCRIPT_DIR}/build_hot_cold_expert_plan.py" \
      --routing-trace "${RUN_ROOT}/pilot_a/routing_trace.jsonl" \
      --output "${PLAN}" \
      --num-layers 48 \
      --num-experts 128 \
      --cold-experts-per-layer "${COLD_EXPERTS_PER_LAYER}" \
      --expert-est-bytes 4718592
  ) > "${RUN_ROOT}/plan_build.log" 2>&1
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" "${PLAN}" \
    "${FIXTURE_META}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
plan_path = Path(sys.argv[3])
fixture_meta_path = Path(sys.argv[4])

def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records

def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)

def parse_dmon(path):
    rx, tx = [], []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            values = []
            for field in line.split():
                try:
                    values.append(float(field))
                except ValueError:
                    pass
            if len(values) >= 3:
                rx.append(values[-2])
                tx.append(values[-1])
    return {
        "pcie_rx_mean_MBps": statistics.mean(rx) if rx else None,
        "pcie_rx_p95_MBps": quantile(rx, 0.95),
        "pcie_rx_max_MBps": max(rx) if rx else None,
        "pcie_tx_mean_MBps": statistics.mean(tx) if tx else None,
    }

def case(name):
    case_dir = root / name
    data = read_json(case_dir / "result.json")
    trace = read_jsonl(case_dir / "expert_trace.jsonl")
    phases = []
    for phase in data.get("phases") or []:
        phases.append({key: value for key, value in phase.items() if key != "requests"})
    rc = None
    try:
        rc = int((case_dir / "returncode.txt").read_text().strip())
    except Exception:
        pass
    promote = [r for r in trace if r.get("action") == "dynamic_expert_swap_promote_issued"]
    demote = [r for r in trace if r.get("action") == "dynamic_expert_swap_demote_issued"]
    return {
        "returncode": rc,
        "status": data.get("status"),
        "error": data.get("error"),
        "load_seconds": data.get("load_seconds"),
        "phases": phases,
        "expert_swaps": data.get("expert_swaps") or [],
        "promote_range_records": len(promote),
        "demote_range_records": len(demote),
        "promote_bytes": sum(int(r.get("bytes") or 0) for r in promote),
        "demote_bytes": sum(int(r.get("bytes") or 0) for r in demote),
        **parse_dmon(case_dir / "dmon_pcie.csv"),
    }

summary = {
    "run_root": str(root),
    "fixture": read_json(fixture_meta_path),
    "initial_plan": read_json(plan_path).get("metadata") or {},
    "cases": {name: case(name) for name in ("pilot_a", "control", "dynamic")},
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

prepare_fixture
require_uvm_proactive_migration_disabled
if should_run_case pilot_a; then
  run_case pilot_a pilot || true
fi
if should_run_case pilot_a || [[ ! -f "${PLAN}" ]]; then
  build_plan
fi
if should_run_case control; then
  run_case control control || true
fi
if should_run_case dynamic; then
  run_case dynamic dynamic || true
fi
write_summary
echo "summary=${SUMMARY}"
