#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_online_hot_cold_kv_rebalance_$(date +%Y%m%d_%H%M%S)}"
SOURCE_FIXTURE="${SOURCE_FIXTURE:-/tmp/vllm_uva1_vs_custom_cold5_cuda_kv22_s64_20260726_01/sharegpt_64_prompt10116_fixture.json}"
DATASET="${DATASET:-/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

SMALL_KV_BYTES="${SMALL_KV_BYTES:-1288490189}"
LARGE_KV_BYTES="${LARGE_KV_BYTES:-2362232013}"
INITIAL_KV_BLOCKS="${INITIAL_KV_BLOCKS:-818}"
KV_GROW_TARGETS="${KV_GROW_TARGETS:-954,1090,1227,1363,1500}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-5}"
EXPERT_EST_BYTES="${EXPERT_EST_BYTES:-4718592}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-2400}"
RUN_CASES="${RUN_CASES:-pilot,baseline,adaptive}"
REMOTE_PREPARE_MODE="${REMOTE_PREPARE_MODE:-native}"
REQUIRE_COMPLETE_REMOTE_PREPARE="${REQUIRE_COMPLETE_REMOTE_PREPARE:-0}"
REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED="${REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED:-0}"
DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE:-1}"
DRIVER_TAG_REQUIRE_MANAGED="${DRIVER_TAG_REQUIRE_MANAGED:-1}"
EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES:-524288}"
NVIDIA_SMI_MONITOR_ENABLE="${NVIDIA_SMI_MONITOR_ENABLE:-1}"
ADAPTIVE_PHASE_NAMES="${ADAPTIVE_PHASE_NAMES:-low,high}"
ADAPTIVE_PHASE_COUNTS="${ADAPTIVE_PHASE_COUNTS:-8,64}"
ADAPTIVE_PHASE_OUTPUTS="${ADAPTIVE_PHASE_OUTPUTS:-64,256}"

FIXTURE="${RUN_ROOT}/sharegpt_64_fixture.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PLAN="${RUN_ROOT}/hot_cold_plan.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

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

should_run_case() {
  local name="$1"
  [[ ",${RUN_CASES}," == *",${name},"* ]]
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
    "${VLLM_PY}" - "${SOURCE_FIXTURE}" "${DATASET}" "${MODEL}" \
      "${FIXTURE}" "${FIXTURE_META}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
model = sys.argv[3]
fixture_path = Path(sys.argv[4])
metadata_path = Path(sys.argv[5])

if source_path.exists():
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_name = str(source_path)
else:
    source = json.loads(dataset_path.read_text(encoding="utf-8"))
    source_name = str(dataset_path)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)

requests = []
source_indices = []
prompt_lengths = []
for index, item in enumerate(source):
    conversations = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conversations, list) or not conversations:
        continue
    prompt = conversations[0].get("value") if isinstance(conversations[0], dict) else None
    if not isinstance(prompt, str) or not prompt:
        continue
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if source_path.exists() or 138 <= prompt_tokens <= 178:
        requests.append({
            "request": len(requests),
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "max_tokens": 256,
        })
        source_indices.append(index)
        prompt_lengths.append(prompt_tokens)
    if len(requests) == 64:
        break

if len(requests) != 64:
    raise SystemExit(f"expected 64 fixture requests, found {len(requests)}")

payload = {
    "metadata": {
        "source": source_name,
        "selection": "existing_fixed_fixture" if source_path.exists() else "first_64_length_matched",
        "source_indices": source_indices,
        "request_count": len(requests),
        "total_prompt_tokens": sum(prompt_lengths),
        "min_prompt_tokens": min(prompt_lengths),
        "max_prompt_tokens": max(prompt_lengths),
    },
    "requests": requests,
}
fixture_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
metadata_path.write_text(json.dumps(payload["metadata"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
PY
  )
}

run_case() {
  local name="$1"
  local kv_bytes="$2"
  local phase_names="$3"
  local phase_counts="$4"
  local phase_outputs="$5"
  local mode="$6"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"
  rm -f \
    "${dir}/result.json" \
    "${dir}/progress.jsonl" \
    "${dir}/run.log" \
    "${dir}/returncode.txt" \
    "${dir}/dmon_pcie.csv" \
    "${dir}/routing_trace.jsonl" \
    "${dir}/expert_trace.jsonl" \
    "${dir}/kv_runtime_trace.jsonl" \
    "${dir}/kv_physical_trace.jsonl" \
    "${dir}/residency_trace.jsonl"
  wait_for_clean_gpu "${name}"

  local dmon_pid=""
  if [[ "${NVIDIA_SMI_MONITOR_ENABLE}" == "1" ]]; then
    nvidia-smi dmon -s t -d 1 -o DT > "${dir}/dmon_pcie.csv" 2>&1 &
    dmon_pid=$!
  fi
  set +e
  (
    cd "${VLLM_ROOT}"
    if [[ "${mode}" == "adaptive" ]]; then
      env \
        VLLM_USE_UVM=0 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
        VLLM_UVM_KV_CACHE_MANAGED=1 \
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
        VLLM_UVM_REMOTE_PREPARE_MODE="${REMOTE_PREPARE_MODE}" \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
        VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
        VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
        VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down \
        VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=128 \
        VLLM_UVM_INIT_WEIGHT_SYNC=1 \
        VLLM_UVM_INIT_WEIGHT_REQUIRE_COMPLETE_REMOTE_PREPARE="${REQUIRE_COMPLETE_REMOTE_PREPARE}" \
        VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/expert_trace.jsonl" \
        VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS=1 \
        VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1 \
        VLLM_UVM_DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE}" \
        VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED="${DRIVER_TAG_REQUIRE_MANAGED}" \
        VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
        VLLM_UVM_EXPERT_ENVELOPE_BYTES="${EXPERT_ENVELOPE_BYTES}" \
        VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
        VLLM_UVM_REQUIRE_PROACTIVE_MIGRATION_DISABLED="${REQUIRE_UVM_PROACTIVE_MIGRATION_DISABLED}" \
        VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE=1 \
        VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1 \
        VLLM_UVM_INCREMENTAL_EXPERTS_PER_LAYER_PER_ROUND=1 \
        VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=0 \
        VLLM_UVM_KV_RUNTIME_ENABLE=1 \
        VLLM_UVM_KV_RUNTIME_MODE=trace_only \
        VLLM_UVM_KV_RUNTIME_TRACE_FILE="${dir}/kv_runtime_trace.jsonl" \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
        VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_KV_BLOCKS}" \
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
        VLLM_UVM_POST_WARMUP_REBALANCE_TRACE_FILE="${dir}/kv_physical_trace.jsonl" \
        VLLM_UVM_RESIDENCY_TRACE_ENABLE=1 \
        VLLM_UVM_RESIDENCY_TRACE_FILE="${dir}/residency_trace.jsonl" \
        VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE=1 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
        timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" "${SCRIPT_DIR}/vllm_generate_phased.py" \
          --model "${MODEL}" \
          --fixture "${FIXTURE}" \
          --output "${dir}/result.json" \
          --progress "${dir}/progress.jsonl" \
          --phase-names "${phase_names}" \
          --phase-request-counts "${phase_counts}" \
          --phase-output-tokens "${phase_outputs}" \
          --grow-after-phase 0 \
          --grow-target-blocks "${KV_GROW_TARGETS}" \
          --release-expert-wave \
          --kv-cache-memory-bytes "${kv_bytes}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    else
      local trace_enable=0
      if [[ "${mode}" == "pilot" ]]; then
        trace_enable=1
      fi
      env \
        VLLM_USE_UVM=0 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
        VLLM_UVM_KV_CACHE_MANAGED=1 \
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${trace_enable}" \
        VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl" \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_KV_RUNTIME_ENABLE=0 \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
        timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" "${SCRIPT_DIR}/vllm_generate_phased.py" \
          --model "${MODEL}" \
          --fixture "${FIXTURE}" \
          --output "${dir}/result.json" \
          --progress "${dir}/progress.jsonl" \
          --phase-names "${phase_names}" \
          --phase-request-counts "${phase_counts}" \
          --phase-output-tokens "${phase_outputs}" \
          --grow-after-phase -1 \
          --kv-cache-memory-bytes "${kv_bytes}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    fi
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
      --routing-trace "${RUN_ROOT}/pilot/routing_trace.jsonl" \
      --output "${PLAN}" \
      --num-layers 48 \
      --num-experts 128 \
      --cold-experts-per-layer "${COLD_EXPERTS_PER_LAYER}" \
      --expert-est-bytes "${EXPERT_EST_BYTES}"
  ) > "${RUN_ROOT}/plan_build.log" 2>&1
}

write_summary() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" "${PLAN}" <<'PY'
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_path = Path(sys.argv[2])
plan_path = Path(sys.argv[3])

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

def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - index) + ordered[hi] * (index - lo)

def parse_dmon(path):
    rx, tx = [], []
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        numbers = []
        for item in parts:
            try:
                numbers.append(float(item))
            except ValueError:
                pass
        if len(numbers) >= 3:
            rx.append(numbers[-2])
            tx.append(numbers[-1])
    return {
        "pcie_rx_mean_MBps": statistics.mean(rx) if rx else None,
        "pcie_rx_p95_MBps": quantile(rx, 0.95),
        "pcie_rx_max_MBps": max(rx) if rx else None,
        "pcie_tx_mean_MBps": statistics.mean(tx) if tx else None,
    }

def output_hash(data):
    token_ids = []
    for phase in data.get("phases") or []:
        for request in phase.get("requests") or []:
            token_ids.append(request.get("output_token_ids") or [])
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()

def case(name):
    case_dir = root / name
    data = read_json(case_dir / "result.json")
    log = (case_dir / "run.log").read_text(errors="replace") if (case_dir / "run.log").exists() else ""
    running = [int(value) for value in re.findall(r"Running:\s*(\d+)\s*reqs", log)]
    waiting = [int(value) for value in re.findall(r"Waiting:\s*(\d+)\s*reqs", log)]
    kv_usage = [float(value) for value in re.findall(r"GPU KV cache usage:\s*([0-9.]+)%", log)]
    fatal_markers = [
        marker for marker in (
            "OutOfMemoryError", "illegal memory access", "UVM fatal", "Xid", "MMU Fault"
        ) if marker.lower() in log.lower()
    ]
    rc = None
    try:
        rc = int((case_dir / "returncode.txt").read_text().strip())
    except Exception:
        pass
    return {
        "returncode": rc,
        "status": data.get("status"),
        "load_seconds": data.get("load_seconds"),
        "phases": data.get("phases") or [],
        "maintenance": data.get("maintenance") or [],
        "max_running": max(running) if running else None,
        "max_waiting": max(waiting) if waiting else None,
        "max_kv_usage_pct": max(kv_usage) if kv_usage else None,
        "output_hash": output_hash(data),
        "fatal_markers": fatal_markers,
        **parse_dmon(case_dir / "dmon_pcie.csv"),
    }

plan = read_json(plan_path)
adaptive = case("adaptive")
expert_records = read_jsonl(root / "adaptive" / "expert_trace.jsonl")
remote_success = [r for r in expert_records if r.get("action") == "init_weight_remote_read_prepare_issued"]
remote_failed = [r for r in expert_records if r.get("action") == "init_weight_remote_read_prepare_failed"]
kv_physical = [
    r for r in read_jsonl(root / "adaptive" / "kv_physical_trace.jsonl")
    if r.get("action") == "adaptive_kv_target_resize"
]

wave_summary = []
for wave in adaptive.get("maintenance") or []:
    result = wave.get("result") or {}
    worker_results = result.get("worker_results") or []
    physical_bytes = sum(int(item.get("range_bytes") or 0) for item in worker_results if isinstance(item, dict))
    expert_results = result.get("expert_results") or []
    released = sum(int(item.get("released_count") or 0) for item in expert_results if isinstance(item, dict))
    wave_summary.append({
        "wave": wave.get("wave"),
        "target_blocks": wave.get("target_blocks"),
        "elapsed_seconds": wave.get("elapsed_seconds"),
        "ok": result.get("ok"),
        "selected_layers": len(result.get("selected_experts_by_layer") or {}),
        "released_experts": released,
        "kv_prefetch_bytes": physical_bytes,
        "applied_target_blocks": result.get("applied_target_blocks"),
    })

baseline = case("baseline")
def phase_by_name(item, name):
    return next((phase for phase in item.get("phases") or [] if phase.get("phase") == name), {})

baseline_high = phase_by_name(baseline, "high")
adaptive_high = phase_by_name(adaptive, "high")
summary = {
    "run_root": str(root),
    "fixture": read_json(root / "fixture_metadata.json"),
    "plan": {
        "path": str(plan_path),
        "metadata": plan.get("metadata") or {},
        "cold_records": len(plan.get("offload_plan") or []),
        "hot_records": len(plan.get("hot_plan") or []),
    },
    "cases": {
        "pilot": case("pilot"),
        "baseline": baseline,
        "adaptive": adaptive,
    },
    "adaptive_transition": {
        "waves": wave_summary,
        "remote_prepare_records": len(remote_success),
        "remote_prepare_failed_records": len(remote_failed),
        "remote_prepare_bytes": sum(int(r.get("bytes") or 0) for r in remote_success),
        "kv_physical_resize_records": len(kv_physical),
        "maintenance_total_seconds": sum(float(w.get("elapsed_seconds") or 0) for w in wave_summary),
    },
    "high_phase_comparison": {
        "baseline_output_tokens_per_second": baseline_high.get("output_tokens_per_second"),
        "adaptive_output_tokens_per_second": adaptive_high.get("output_tokens_per_second"),
        "adaptive_throughput_delta_pct": (
            (adaptive_high.get("output_tokens_per_second") / baseline_high.get("output_tokens_per_second") - 1) * 100
            if baseline_high.get("output_tokens_per_second") and adaptive_high.get("output_tokens_per_second") else None
        ),
        "baseline_mean_tpot_ms": baseline_high.get("mean_tpot_ms"),
        "adaptive_mean_tpot_ms": adaptive_high.get("mean_tpot_ms"),
        "baseline_mean_ttft_ms": baseline_high.get("mean_ttft_ms"),
        "adaptive_mean_ttft_ms": adaptive_high.get("mean_ttft_ms"),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

prepare_fixture
require_uvm_proactive_migration_disabled
if should_run_case pilot; then
  run_case pilot "${SMALL_KV_BYTES}" pilot 64 256 pilot
fi
if should_run_case pilot \
  || { should_run_case adaptive && [[ -f "${RUN_ROOT}/pilot/routing_trace.jsonl" ]]; } \
  || [[ ! -f "${PLAN}" ]]; then
  build_plan
fi
if should_run_case baseline; then
  run_case baseline "${SMALL_KV_BYTES}" low,high 8,64 64,256 baseline
fi
if should_run_case adaptive; then
  run_case adaptive "${LARGE_KV_BYTES}" \
    "${ADAPTIVE_PHASE_NAMES}" "${ADAPTIVE_PHASE_COUNTS}" \
    "${ADAPTIVE_PHASE_OUTPUTS}" adaptive
fi
write_summary

echo "summary=${SUMMARY}"
