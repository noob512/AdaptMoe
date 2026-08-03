#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
DATASET="${DATASET:-/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_online_hot_cold_kv10g_scale_$(date +%Y%m%d_%H%M%S)}"

REQUEST_COUNT="${REQUEST_COUNT:-256}"
PILOT_REQUEST_COUNT="${PILOT_REQUEST_COUNT:-${REQUEST_COUNT}}"
PILOT_OUTPUT_TOKENS="${PILOT_OUTPUT_TOKENS:-${OUTPUT_TOKENS:-64}}"
PROMPT_MIN_TOKENS="${PROMPT_MIN_TOKENS:-320}"
PROMPT_MAX_TOKENS="${PROMPT_MAX_TOKENS:-360}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
SMALL_KV_BYTES="${SMALL_KV_BYTES:-1288490189}"
LARGE_KV_BYTES="${LARGE_KV_BYTES:-10737418240}"
INITIAL_KV_BLOCKS="${INITIAL_KV_BLOCKS:-818}"
FINAL_KV_BLOCKS="${FINAL_KV_BLOCKS:-6825}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-45}"
EXPERT_EST_BYTES="${EXPERT_EST_BYTES:-4718592}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-10.0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
TIMEOUT_SEC="${TIMEOUT_SEC:-3600}"
RUN_CASES="${RUN_CASES:-pilot,baseline,adaptive,static_custom,native_uva}"
RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE:-1}"
RESIDENCY_KV_METADATA_ENABLE="${RESIDENCY_KV_METADATA_ENABLE:-1}"

PHASE_NAMES="${PHASE_NAMES:-low,stage1,stage2,stage3,stage4,stage5,stage6,stage7,stage8,stage9}"
PHASE_COUNTS="${PHASE_COUNTS:-8,64,96,128,160,192,224,256,256,256}"
PHASE_OUTPUTS="${PHASE_OUTPUTS:-64,64,64,64,64,64,64,64,64,64}"
STATIC_PHASE_NAMES="${STATIC_PHASE_NAMES:-warm,final}"
STATIC_PHASE_COUNTS="${STATIC_PHASE_COUNTS:-256,256}"
STATIC_PHASE_OUTPUTS="${STATIC_PHASE_OUTPUTS:-1,64}"
BASELINE_PHASE_NAMES="${BASELINE_PHASE_NAMES:-warm,final}"
BASELINE_PHASE_COUNTS="${BASELINE_PHASE_COUNTS:-256,256}"
BASELINE_PHASE_OUTPUTS="${BASELINE_PHASE_OUTPUTS:-1,64}"
NATIVE_PHASE_NAMES="${NATIVE_PHASE_NAMES:-warm,final}"
NATIVE_PHASE_COUNTS="${NATIVE_PHASE_COUNTS:-256,256}"
NATIVE_PHASE_OUTPUTS="${NATIVE_PHASE_OUTPUTS:-1,64}"

FIXTURE="${RUN_ROOT}/sharegpt_256_prompt320_360.json"
FIXTURE_META="${RUN_ROOT}/fixture_metadata.json"
PLAN="${RUN_ROOT}/hot_cold_plan_k45.json"
TARGETS_FILE="${RUN_ROOT}/kv_grow_targets.txt"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used=999999
  for _ in $(seq 1 180); do
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

prepare_fixture() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET}" "${MODEL}" "${FIXTURE}" "${FIXTURE_META}" \
      "${REQUEST_COUNT}" "${PROMPT_MIN_TOKENS}" "${PROMPT_MAX_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])
request_count = int(sys.argv[5])
min_tokens = int(sys.argv[6])
max_tokens = int(sys.argv[7])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
source = json.loads(dataset_path.read_text(encoding="utf-8"))
requests = []
source_indices = []
prompt_lengths = []
for index, item in enumerate(source):
    conversations = item.get("conversations") if isinstance(item, dict) else None
    if not isinstance(conversations, list) or not conversations:
        continue
    first = conversations[0]
    prompt = first.get("value") if isinstance(first, dict) else None
    if not isinstance(prompt, str) or not prompt:
        continue
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    if not min_tokens <= prompt_tokens <= max_tokens:
        continue
    requests.append({
        "request": len(requests),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "max_tokens": 64,
    })
    source_indices.append(index)
    prompt_lengths.append(prompt_tokens)
    if len(requests) == request_count:
        break

if len(requests) != request_count:
    raise SystemExit(f"expected {request_count} requests, found {len(requests)}")

metadata = {
    "source": str(dataset_path),
    "selection": "first_length_matched",
    "source_indices": source_indices,
    "request_count": len(requests),
    "total_prompt_tokens": sum(prompt_lengths),
    "mean_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
    "min_prompt_tokens": min(prompt_lengths),
    "max_prompt_tokens": max(prompt_lengths),
}
fixture_path.write_text(
    json.dumps({"metadata": metadata, "requests": requests}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)
metadata_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  )
}

generate_targets() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${INITIAL_KV_BLOCKS}" "${FINAL_KV_BLOCKS}" \
    "${COLD_EXPERTS_PER_LAYER}" "${TARGETS_FILE}" <<'PY'
import sys
from pathlib import Path

initial = int(sys.argv[1])
final = int(sys.argv[2])
waves = int(sys.argv[3])
path = Path(sys.argv[4])
targets = [initial + round((final - initial) * wave / waves)
           for wave in range(1, waves + 1)]
if targets[-1] != final or any(a >= b for a, b in zip(targets, targets[1:])):
    raise SystemExit(f"invalid targets: {targets}")
path.write_text(",".join(str(value) for value in targets) + "\n", encoding="utf-8")
print(path.read_text().strip())
PY
}

run_case() {
  local name="$1"
  local mode="$2"
  local kv_bytes="$3"
  local phase_names="$4"
  local phase_counts="$5"
  local phase_outputs="$6"
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
    "${dir}/residency_trace.jsonl" \
    "${dir}/cpu_offload_manifest.jsonl"
  wait_for_clean_gpu "${name}"

  local -a env_args=(
    VLLM_USE_UVM=0
    VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0
  )
  local -a extra_args=(--grow-after-phase -1)
  local cpu_offload_gb=0

  case "${mode}" in
    pilot|baseline)
      env_args+=(
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_KV_CACHE_MANAGED=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
        VLLM_UVM_KV_RUNTIME_ENABLE=0
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
      )
      if [[ "${mode}" == "pilot" ]]; then
        env_args+=(
          VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=1
          VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl"
        )
      else
        env_args+=(VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0)
      fi
      ;;
    adaptive|static_custom)
      env_args+=(
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1
        VLLM_UVM_KV_CACHE_MANAGED=1
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup
        VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}"
        VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1
        VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down
        VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=128
        VLLM_UVM_INIT_WEIGHT_SYNC=1
        VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/expert_trace.jsonl"
        VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS=1
        VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1
        VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE=1
        VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch
        VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=0
        VLLM_UVM_KV_RUNTIME_ENABLE=1
        VLLM_UVM_KV_RUNTIME_MODE=trace_only
        VLLM_UVM_KV_RUNTIME_TRACE_FILE="${dir}/kv_runtime_trace.jsonl"
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_KV_BLOCKS}"
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch
        VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1
        VLLM_UVM_POST_WARMUP_REBALANCE_TRACE_FILE="${dir}/kv_physical_trace.jsonl"
        VLLM_UVM_RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE}"
        VLLM_UVM_RESIDENCY_TRACE_FILE="${dir}/residency_trace.jsonl"
        VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE="${RESIDENCY_KV_METADATA_ENABLE}"
      )
      if [[ "${mode}" == "adaptive" ]]; then
        env_args+=(
          VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=1
          VLLM_UVM_INCREMENTAL_EXPERTS_PER_LAYER_PER_ROUND=1
        )
        extra_args=(
          --grow-after-phase -1
          --grow-target-blocks "$(<"${TARGETS_FILE}")"
          --maintenance-waves-per-phase 5
          --release-expert-wave
        )
      else
        env_args+=(VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0)
        extra_args=(
          --grow-after-phase -1
          --pre-phase-target-blocks "${FINAL_KV_BLOCKS}"
        )
      fi
      ;;
    native_uva)
      cpu_offload_gb="${NATIVE_CPU_OFFLOAD_GB}"
      env_args+=(
        VLLM_CPU_OFFLOAD_BACKEND=uva
        VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/cpu_offload_manifest.jsonl"
        VLLM_CPU_OFFLOAD_MANIFEST_RESET=1
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0
        VLLM_UVM_KV_CACHE_MANAGED=0
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0
        VLLM_UVM_KV_RUNTIME_ENABLE=0
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0
      )
      ;;
    *)
      echo "unknown mode: ${mode}" >&2
      return 2
      ;;
  esac

  nvidia-smi dmon -s t -d 1 -o DT > "${dir}/dmon_pcie.csv" 2>&1 &
  local dmon_pid=$!
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_args[@]}" timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${SCRIPT_DIR}/vllm_generate_phased.py" \
      --model "${MODEL}" \
      --fixture "${FIXTURE}" \
      --output "${dir}/result.json" \
      --progress "${dir}/progress.jsonl" \
      --phase-names "${phase_names}" \
      --phase-request-counts "${phase_counts}" \
      --phase-output-tokens "${phase_outputs}" \
      --kv-cache-memory-bytes "${kv_bytes}" \
      --cpu-offload-gb "${cpu_offload_gb}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --disable-prefix-caching \
      "${extra_args[@]}"
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
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" "${SUMMARY}" "${PLAN}" \
    "${FIXTURE_META}" <<'PY'
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

def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)

def parse_dmon(path):
    rx, tx = [], []
    if not path.exists():
        return {}
    for line in path.read_text(errors="replace").splitlines():
        numbers = []
        for item in line.split():
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

def phase_monitor(log):
    sections = {}
    current = None
    for line in log.splitlines():
        begin = re.search(r"UVM_PHASE_BEGIN name=([^ ]+)", line)
        if begin:
            current = begin.group(1)
            sections[current] = []
        if current is not None:
            sections[current].append(line)
        end = re.search(r"UVM_PHASE_END name=([^ ]+)", line)
        if end and current == end.group(1):
            current = None
    result = {}
    for name, lines in sections.items():
        text = "\n".join(lines)
        running = [int(v) for v in re.findall(r"Running:\s*(\d+)\s*reqs", text)]
        waiting = [int(v) for v in re.findall(r"Waiting:\s*(\d+)\s*reqs", text)]
        usage = [float(v) for v in re.findall(r"GPU KV cache usage:\s*([0-9.]+)%", text)]
        result[name] = {
            "max_running": max(running) if running else None,
            "max_waiting": max(waiting) if waiting else None,
            "max_kv_usage_pct": max(usage) if usage else None,
        }
    return result

def output_hash(data):
    values = []
    for phase in data.get("phases") or []:
        values.extend(request.get("output_token_ids") or []
                      for request in phase.get("requests") or [])
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()

def summarize_maintenance(data):
    values = []
    for wave in data.get("maintenance") or []:
        result = wave.get("result") or {}
        expert_results = result.get("expert_results") or []
        worker_results = result.get("worker_results") or []
        values.append({
            "wave": wave.get("wave"),
            "after_phase": wave.get("after_phase"),
            "target_blocks": wave.get("target_blocks"),
            "elapsed_seconds": wave.get("elapsed_seconds"),
            "ok": result.get("ok"),
            "released_experts": sum(int(item.get("released_count") or 0)
                                    for item in expert_results if isinstance(item, dict)),
            "kv_prefetch_bytes": sum(int(item.get("range_bytes") or 0)
                                     for item in worker_results if isinstance(item, dict)),
        })
    return values

def case(name):
    case_dir = root / name
    data = read_json(case_dir / "result.json")
    log_path = case_dir / "run.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""
    rc = None
    try:
        rc = int((case_dir / "returncode.txt").read_text().strip())
    except Exception:
        pass
    phases = []
    monitor = phase_monitor(log)
    for phase in data.get("phases") or []:
        item = {key: value for key, value in phase.items() if key != "requests"}
        item.update(monitor.get(str(phase.get("phase")), {}))
        phases.append(item)
    fatal_markers = [
        marker for marker in (
            "OutOfMemoryError", "illegal memory access", "UVM fatal", "Xid", "MMU Fault"
        ) if marker.lower() in log.lower()
    ]
    return {
        "returncode": rc,
        "status": data.get("status"),
        "error": data.get("error"),
        "load_seconds": data.get("load_seconds"),
        "phases": phases,
        "maintenance": summarize_maintenance(data),
        "maintenance_total_seconds": sum(
            float(wave.get("elapsed_seconds") or 0)
            for wave in data.get("maintenance") or []
        ),
        "output_hash": output_hash(data),
        "fatal_markers": fatal_markers,
        **parse_dmon(case_dir / "dmon_pcie.csv"),
    }

cases = {
    name: case(name)
    for name in ("pilot", "baseline", "adaptive", "static_custom", "native_uva")
}
for name in ("adaptive", "static_custom"):
    records = read_jsonl(root / name / "expert_trace.jsonl")
    issued = [r for r in records if r.get("action") == "init_weight_remote_read_prepare_issued"]
    failed = [r for r in records if r.get("action") == "init_weight_remote_read_prepare_failed"]
    cases[name]["remote_prepare_records"] = len(issued)
    cases[name]["remote_prepare_failed_records"] = len(failed)
    cases[name]["remote_prepare_bytes"] = sum(int(r.get("bytes") or 0) for r in issued)

native_manifest = read_jsonl(root / "native_uva" / "cpu_offload_manifest.jsonl")
native_parameters = [r for r in native_manifest if r.get("event") == "cpu_offload_parameter"]
cases["native_uva"]["cpu_offload_parameter_records"] = len(native_parameters)
cases["native_uva"]["cpu_offload_bytes"] = sum(
    int(record.get("bytes") or 0) for record in native_parameters
)

summary = {
    "run_root": str(root),
    "fixture": read_json(fixture_meta_path),
    "plan": read_json(plan_path).get("metadata") or {},
    "kv_grow_targets": [
        int(value) for value in (root / "kv_grow_targets.txt").read_text().split(",")
        if value.strip()
    ],
    "cases": cases,
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

prepare_fixture
generate_targets

if should_run_case pilot; then
  run_case pilot pilot "${SMALL_KV_BYTES}" pilot \
    "${PILOT_REQUEST_COUNT}" "${PILOT_OUTPUT_TOKENS}"
fi
if should_run_case pilot \
  || { should_run_case adaptive && [[ -f "${RUN_ROOT}/pilot/routing_trace.jsonl" ]]; } \
  || [[ ! -f "${PLAN}" ]]; then
  build_plan
fi

if should_run_case baseline; then
  run_case baseline baseline "${SMALL_KV_BYTES}" \
    "${BASELINE_PHASE_NAMES}" "${BASELINE_PHASE_COUNTS}" \
    "${BASELINE_PHASE_OUTPUTS}" || true
fi
if should_run_case adaptive; then
  run_case adaptive adaptive "${LARGE_KV_BYTES}" \
    "${PHASE_NAMES}" "${PHASE_COUNTS}" "${PHASE_OUTPUTS}" || true
fi
if should_run_case static_custom; then
  run_case static_custom static_custom "${LARGE_KV_BYTES}" \
    "${STATIC_PHASE_NAMES}" "${STATIC_PHASE_COUNTS}" \
    "${STATIC_PHASE_OUTPUTS}" || true
fi
if should_run_case native_uva; then
  run_case native_uva native_uva "${LARGE_KV_BYTES}" \
    "${NATIVE_PHASE_NAMES}" "${NATIVE_PHASE_COUNTS}" \
    "${NATIVE_PHASE_OUTPUTS}" || true
fi

write_summary
echo "summary=${SUMMARY}"
