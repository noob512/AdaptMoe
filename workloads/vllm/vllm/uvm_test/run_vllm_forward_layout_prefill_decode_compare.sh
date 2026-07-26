#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_forward_layout_prefill_decode_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
PORT_BASE="${PORT_BASE:-9080}"

BENCH_SIZES="${BENCH_SIZES:-128,256,512}"
BENCH_PATTERNS="${BENCH_PATTERNS:-diverse}"
BENCH_WARMUPS="${BENCH_WARMUPS:-5}"
BENCH_ITERS="${BENCH_ITERS:-20}"
BENCH_BURST_ITERS="${BENCH_BURST_ITERS:-3}"
BENCH_BURST_ROUNDS="${BENCH_BURST_ROUNDS:-3}"
TRACE_BENCH_WARMUPS="${TRACE_BENCH_WARMUPS:-0}"
TRACE_BENCH_ITERS="${TRACE_BENCH_ITERS:-1}"
TRACE_BENCH_BURST_ITERS="${TRACE_BENCH_BURST_ITERS:-1}"
TRACE_BENCH_BURST_ROUNDS="${TRACE_BENCH_BURST_ROUNDS:-1}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-536870912}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-2}"

NUM_LAYERS="${NUM_LAYERS:-48}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-10}"
TARGET_ROLES="${TARGET_ROLES:-moe_gate_up,moe_down}"
EXPERT_EST_BYTES="${EXPERT_EST_BYTES:-4718592}"
CUSTOM_PLAN="${CUSTOM_PLAN:-${RUN_ROOT}/per_layer${COLD_EXPERTS_PER_LAYER}_cold_plan.json}"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local label="$1"
  local used
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

cleanup_port() {
  local port="$1"
  pkill -f "vllm.entrypoints.cli.main serve.*--port ${port}" 2>/dev/null || true
  pkill -f "vllm serve.*--port ${port}" 2>/dev/null || true
}

stop_process_tree() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    pkill -TERM -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
    sleep 5
  fi
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    pkill -KILL -P "${pid}" 2>/dev/null || true
    kill -9 "${pid}" 2>/dev/null || true
  fi
}

wait_for_server_or_exit() {
  local port="$1"
  local server_pid="$2"
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 2
}

require_environment() {
  [[ -x "${VLLM_ROOT}/${VLLM_PY}" ]] || {
    echo "vLLM Python not found: ${VLLM_ROOT}/${VLLM_PY}" >&2
    exit 2
  }
  [[ -d "${MODEL}" ]] || {
    echo "model path not found: ${MODEL}" >&2
    exit 2
  }
}

run_forward_case() {
  local label="$1"
  local mode="$2"
  local layout="$3"
  local port="$4"
  local trace_enable="$5"
  local layer_timing_enable="$6"
  local warmups="$7"
  local iterations="$8"
  local burst_iters="$9"
  local burst_rounds="${10}"
  local dir="${RUN_ROOT}/${label}"
  mkdir -p "${dir}"
  rm -f "${dir}/server.log" "${dir}/status.txt" \
    "${dir}/routing_trace.jsonl" "${dir}/layer_timing.jsonl" \
    "${dir}/cpu_offload_manifest.jsonl" "${dir}/init_trace.jsonl"

  cleanup_port "${port}"
  wait_for_clean_gpu "${label}"

  local -a offload_args=()
  if [[ "${layout}" == "native_uva" ]]; then
    offload_args=(--cpu-offload-gb "${NATIVE_CPU_OFFLOAD_GB}")
  fi

  echo "=== ${label}: mode=${mode} layout=${layout} ===" | tee -a "${RUN_ROOT}/run.log"
  (
    cd "${VLLM_ROOT}"
    if [[ "${layout}" == "custom_per_layer_cold" ]]; then
      env \
        VLLM_USE_UVM=0 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
        VLLM_UVM_KV_CACHE_MANAGED=0 \
        VLLM_DISABLE_OPENAI_METRICS=1 \
        VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${trace_enable}" \
        VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl" \
        VLLM_QWEN3_MOE_LAYER_TIMING_ENABLE="${layer_timing_enable}" \
        VLLM_QWEN3_MOE_LAYER_TIMING_FILE="${dir}/layer_timing.jsonl" \
        VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
        VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${CUSTOM_PLAN}" \
        VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
        VLLM_UVM_INIT_WEIGHT_TARGET_ROLES="${TARGET_ROLES}" \
        VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${NUM_EXPERTS}" \
        VLLM_UVM_INIT_WEIGHT_SYNC=1 \
        VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
        VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
        VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
        VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${CUSTOM_PLAN}" \
        VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES="${TARGET_ROLES}" \
        VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
        VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
        VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap \
        VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
        VLLM_FORWARD_TOKEN_BENCH_ENABLE=1 \
        VLLM_FORWARD_TOKEN_BENCH_LABEL="${label}" \
        VLLM_FORWARD_TOKEN_BENCH_MODE="${mode}" \
        VLLM_FORWARD_TOKEN_BENCH_SIZES="${BENCH_SIZES}" \
        VLLM_FORWARD_TOKEN_BENCH_PATTERNS="${BENCH_PATTERNS}" \
        VLLM_FORWARD_TOKEN_BENCH_WARMUPS="${warmups}" \
        VLLM_FORWARD_TOKEN_BENCH_ITERS="${iterations}" \
        VLLM_FORWARD_TOKEN_BENCH_BURST_ITERS="${burst_iters}" \
        VLLM_FORWARD_TOKEN_BENCH_BURST_ROUNDS="${burst_rounds}" \
        "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
          --port "${port}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          --seed 0 \
          --enforce-eager \
          --disable-log-requests
    else
      env \
        VLLM_USE_UVM=0 \
        VLLM_CPU_OFFLOAD_BACKEND=uva \
        VLLM_CPU_OFFLOAD_MANIFEST_FILE="${dir}/cpu_offload_manifest.jsonl" \
        VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
        VLLM_UVM_EXPERT_WEIGHTS_MANAGED=0 \
        VLLM_UVM_KV_CACHE_MANAGED=0 \
        VLLM_DISABLE_OPENAI_METRICS=1 \
        VLLM_MOE_ROUTING_SIMULATION_STRATEGY= \
        VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${trace_enable}" \
        VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl" \
        VLLM_QWEN3_MOE_LAYER_TIMING_ENABLE="${layer_timing_enable}" \
        VLLM_QWEN3_MOE_LAYER_TIMING_FILE="${dir}/layer_timing.jsonl" \
        VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
        VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
        VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
        VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
        VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap \
        VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
        VLLM_FORWARD_TOKEN_BENCH_ENABLE=1 \
        VLLM_FORWARD_TOKEN_BENCH_LABEL="${label}" \
        VLLM_FORWARD_TOKEN_BENCH_MODE="${mode}" \
        VLLM_FORWARD_TOKEN_BENCH_SIZES="${BENCH_SIZES}" \
        VLLM_FORWARD_TOKEN_BENCH_PATTERNS="${BENCH_PATTERNS}" \
        VLLM_FORWARD_TOKEN_BENCH_WARMUPS="${warmups}" \
        VLLM_FORWARD_TOKEN_BENCH_ITERS="${iterations}" \
        VLLM_FORWARD_TOKEN_BENCH_BURST_ITERS="${burst_iters}" \
        VLLM_FORWARD_TOKEN_BENCH_BURST_ROUNDS="${burst_rounds}" \
        "${VLLM_PY}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
          --port "${port}" \
          --max-model-len "${MAX_MODEL_LEN}" \
          --max-num-seqs "${MAX_NUM_SEQS}" \
          --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
          --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
          --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
          "${offload_args[@]}" \
          --seed 0 \
          --enforce-eager \
          --disable-log-requests
    fi
  ) > "${dir}/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" > "${dir}/server.pid"

  local status=0
  set +e
  wait_for_server_or_exit "${port}" "${server_pid}"
  status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "server failed for ${label}" | tee -a "${RUN_ROOT}/run.log"
  fi
  printf '%s\n' "${status}" > "${dir}/status.txt"
  stop_process_tree "${server_pid}"
  cleanup_port "${port}"
  wait_for_clean_gpu "after_${label}" || true
}

build_cold_plan_from_traces() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${CUSTOM_PLAN}" "${NUM_LAYERS}" "${NUM_EXPERTS}" \
      "${COLD_EXPERTS_PER_LAYER}" "${EXPERT_EST_BYTES}" \
      "${RUN_ROOT}/trace_decode/routing_trace.jsonl" \
      "${RUN_ROOT}/trace_prefill/routing_trace.jsonl" <<'PY'
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

output_path = Path(sys.argv[1])
num_layers = int(sys.argv[2])
num_experts = int(sys.argv[3])
cold_experts_per_layer = int(sys.argv[4])
expert_est_bytes = int(sys.argv[5])
trace_paths = [Path(value) for value in sys.argv[6:]]

def layer_id_from_name(layer_name):
    if not isinstance(layer_name, str):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value < num_layers else None

token_counts = defaultdict(int)
trace_records = 0
for trace_path in trace_paths:
    for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        layer_id = layer_id_from_name(record.get("layer_name"))
        if layer_id is None:
            continue
        counts = record.get("expert_token_counts") or {}
        if not isinstance(counts, dict):
            continue
        trace_records += 1
        for expert_s, count in counts.items():
            try:
                expert_id = int(expert_s)
                count_i = int(count)
            except (TypeError, ValueError):
                continue
            if 0 <= expert_id < num_experts:
                token_counts[(layer_id, expert_id)] += count_i

selected = []
for layer_id in range(num_layers):
    layer_candidates = [
        (token_counts[(layer_id, expert_id)], layer_id, expert_id)
        for expert_id in range(num_experts)
    ]
    layer_candidates.sort(key=lambda item: (item[0], item[2]))
    selected.extend(layer_candidates[:cold_experts_per_layer])
selected.sort(key=lambda item: (item[0], item[1], item[2]))

offload_plan = [
    {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "reason": "forward_token_per_layer_coldest",
        "pilot_token_count": count,
    }
    for count, layer_id, expert_id in selected
]
zero_hit = sum(1 for count, _, _ in selected if count == 0)
selected_token_count = sum(count for count, _, _ in selected)
observed_token_count = sum(token_counts.values())
metadata = {
    "policy": "exact_used_oracle_v1",
    "oracle_scope": "forward_token_decode_prefill_diverse",
    "routing_traces": [str(path) for path in trace_paths],
    "trace_records": trace_records,
    "selection_mode": "per_layer_coldest_n",
    "cold_experts_per_layer": cold_experts_per_layer,
    "expert_est_bytes": expert_est_bytes,
    "selected_records": len(offload_plan),
    "selected_zero_hit_records": zero_hit,
    "selected_nonzero_records": len(offload_plan) - zero_hit,
    "selected_token_count": selected_token_count,
    "observed_token_count": observed_token_count,
    "selected_token_fraction": (
        selected_token_count / observed_token_count if observed_token_count else 0.0
    ),
    "estimated_release_bytes": len(offload_plan) * expert_est_bytes,
    "min_selected_token_count": selected[0][0] if selected else None,
    "max_selected_token_count": selected[-1][0] if selected else None,
}
plan = {
    "metadata": metadata,
    "prefetch_plan": [],
    "offload_plan": offload_plan,
    "prefetch_plan_records": 0,
    "offload_plan_records": len(offload_plan),
}
output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/plan_build.log"
}

write_summary() {
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${RUN_ROOT}" "${BENCH_SIZES}" "${BENCH_PATTERNS}" \
      "${BENCH_WARMUPS}" "${BENCH_ITERS}" "${BENCH_BURST_ITERS}" \
    "${BENCH_BURST_ROUNDS}" "${NATIVE_CPU_OFFLOAD_GB}" \
    "${COLD_EXPERTS_PER_LAYER}" "${CUSTOM_PLAN}" \
    "${KV_CACHE_MEMORY_BYTES}" "${MAX_NUM_SEQS}" \
    "${MAX_NUM_BATCHED_TOKENS}" <<'PY'
import json
import math
import pathlib
import re
import statistics
import sys
from collections import defaultdict

root = pathlib.Path(sys.argv[1])
bench_sizes = [int(value) for value in sys.argv[2].split(",") if value]
bench_patterns = [value for value in sys.argv[3].split(",") if value]
warmups = int(sys.argv[4])
iterations = int(sys.argv[5])
burst_iters = int(sys.argv[6])
burst_rounds = int(sys.argv[7])
native_cpu_offload_gb = float(sys.argv[8])
cold_experts_per_layer = int(sys.argv[9])
plan_path = pathlib.Path(sys.argv[10])
kv_cache_memory_bytes = int(sys.argv[11])
max_num_seqs = int(sys.argv[12])
max_num_batched_tokens = int(sys.argv[13])

def load_jsonl(path):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def percentile(values, pct):
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac

def forward_results(case_dir):
    marker = "VLLM_FORWARD_TOKEN_BENCH_RESULT "
    results = []
    text = (case_dir / "server.log").read_text(errors="replace")
    for line in text.splitlines():
        if marker in line:
            results.append(json.loads(line.split(marker, 1)[1]))
    return results, text

def native_manifest(case_dir):
    records = [
        item for item in load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
        if item.get("event") == "cpu_offload_parameter"
    ]
    layer_bytes = defaultdict(int)
    for item in records:
        name = str(item.get("full_parameter_name", ""))
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match:
            layer_bytes[int(match.group(1))] += int(item.get("bytes", 0))
    return {
        "offload_params": len(records),
        "offload_bytes": sum(int(item.get("bytes", 0)) for item in records),
        "offload_bytes_by_layer": dict(sorted(layer_bytes.items())),
    }

def custom_trace(case_dir):
    records = [
        item for item in load_jsonl(case_dir / "init_trace.jsonl")
        if item.get("action") == "init_weight_remote_read_prepare_issued"
        and bool(item.get("success"))
    ]
    layer_bytes = defaultdict(int)
    for item in records:
        layer_name = str(item.get("layer_name", ""))
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
        if match:
            layer_bytes[int(match.group(1))] += int(item.get("bytes", 0))
    return {
        "remote_prepare_records": len(records),
        "remote_prepare_bytes": sum(int(item.get("bytes", 0)) for item in records),
        "remote_prepare_bytes_by_layer": dict(sorted(layer_bytes.items())),
    }

def layer_timing(case_dir):
    records = [
        item for item in load_jsonl(case_dir / "layer_timing.jsonl")
        if item.get("event") == "qwen3_moe_layer_timing"
    ]
    expected = warmups + iterations + burst_iters * burst_rounds
    grouped = defaultdict(list)
    for item in records:
        grouped[(int(item["num_tokens"]), int(item["layer_id"]))].append(item)
    summaries = {}
    by_tokens = defaultdict(list)
    for key, values in grouped.items():
        values.sort(key=lambda item: float(item.get("timestamp", 0.0)))
        if len(values) < expected:
            continue
        selected = values[-expected:]
        measured = selected[warmups:warmups + iterations]
        samples = [float(item["elapsed_ms"]) for item in measured]
        row = {
            "num_tokens": key[0],
            "layer_id": key[1],
            "median_ms": statistics.median(samples),
            "mean_ms": statistics.fmean(samples),
            "p95_ms": percentile(samples, 95.0),
            "sample_count": len(samples),
        }
        summaries[f"{key[0]}:{key[1]}"] = row
        by_tokens[key[0]].append(row)
    token_summaries = {}
    for num_tokens, rows in sorted(by_tokens.items()):
        rows.sort(key=lambda item: item["layer_id"])
        if len(rows) != 48:
            continue
        medians = [float(item["median_ms"]) for item in rows]
        token_summaries[str(num_tokens)] = {
            "sum_layer_median_ms": sum(medians),
            "mean_layer_median_ms": statistics.fmean(medians),
            "layers_0_9_sum_ms": sum(medians[:10]),
            "layer_10_ms": medians[10],
            "layers_11_47_sum_ms": sum(medians[11:]),
            "slowest_layer_id": int(max(rows, key=lambda item: item["median_ms"])["layer_id"]),
            "slowest_layer_median_ms": max(medians),
        }
    return summaries, token_summaries

case_specs = []
for mode in ("decode", "prefill"):
    case_specs.extend([
        (f"local_{mode}", mode, "local_hbm"),
        (f"native_uva2_{mode}", mode, "native_uva_front_layers"),
        (f"custom_per_layer{cold_experts_per_layer}_{mode}", mode, "custom_per_layer_cold"),
    ])

cases = []
layer_maps = {}
for label, mode, layout in case_specs:
    case_dir = root / label
    results, text = forward_results(case_dir)
    item = {
        "label": label,
        "mode": mode,
        "layout": layout,
        "status": (case_dir / "status.txt").read_text().strip()
        if (case_dir / "status.txt").exists() else "missing",
        "result_count": len(results),
        "results": results,
        "has_cuda_oom": "out of memory" in text.lower(),
        "has_illegal_memory_access": "illegal memory access" in text.lower(),
        "has_uvm_fatal": "uvm fatal" in text.lower(),
        "has_xid": bool(re.search(r"\\bXid\\b", text, re.IGNORECASE)),
    }
    if layout == "native_uva_front_layers":
        item.update(native_manifest(case_dir))
    elif layout == "custom_per_layer_cold":
        item.update(custom_trace(case_dir))
    else:
        item["offload_bytes"] = 0
    layer_rows, token_summary = layer_timing(case_dir)
    item["layer_summary_by_tokens"] = token_summary
    layer_maps[label] = layer_rows
    cases.append(item)

result_index = {
    (case["label"], result["pattern"], int(result["num_tokens"])): result
    for case in cases
    for result in case["results"]
}

comparisons = []
for mode in ("decode", "prefill"):
    local_label = f"local_{mode}"
    for remote_label in (
        f"native_uva2_{mode}",
        f"custom_per_layer{cold_experts_per_layer}_{mode}",
    ):
        for pattern in bench_patterns:
            for num_tokens in bench_sizes:
                local = result_index.get((local_label, pattern, num_tokens))
                remote = result_index.get((remote_label, pattern, num_tokens))
                if not local or not remote:
                    continue
                local_ms = float(local["median_ms"])
                remote_ms = float(remote["median_ms"])
                row = {
                    "mode": mode,
                    "pattern": pattern,
                    "num_tokens": num_tokens,
                    "local_label": local_label,
                    "remote_label": remote_label,
                    "remote_layout": next(
                        case["layout"] for case in cases if case["label"] == remote_label
                    ),
                    "local_median_ms": local_ms,
                    "remote_median_ms": remote_ms,
                    "remote_overhead_pct": (remote_ms / local_ms - 1.0) * 100.0,
                    "local_tokens_per_second": float(local["tokens_per_second"]),
                    "remote_tokens_per_second": float(remote["tokens_per_second"]),
                    "remote_tps_vs_local_pct": (
                        float(remote["tokens_per_second"])
                        / float(local["tokens_per_second"])
                        - 1.0
                    ) * 100.0,
                    "local_ms_per_token": float(local["ms_per_token"]),
                    "remote_ms_per_token": float(remote["ms_per_token"]),
                }
                local_layer = layer_maps.get(local_label, {})
                remote_layer = layer_maps.get(remote_label, {})
                layer_deltas = []
                for layer_id in range(48):
                    local_row = local_layer.get(f"{num_tokens}:{layer_id}")
                    remote_row = remote_layer.get(f"{num_tokens}:{layer_id}")
                    if local_row and remote_row:
                        layer_deltas.append({
                            "layer_id": layer_id,
                            "delta_ms": (
                                float(remote_row["median_ms"])
                                - float(local_row["median_ms"])
                            ),
                            "remote_median_ms": float(remote_row["median_ms"]),
                            "local_median_ms": float(local_row["median_ms"]),
                        })
                if len(layer_deltas) == 48:
                    row["sum_layer_delta_ms"] = sum(item["delta_ms"] for item in layer_deltas)
                    row["layers_0_9_delta_ms"] = sum(item["delta_ms"] for item in layer_deltas[:10])
                    row["layer_10_delta_ms"] = layer_deltas[10]["delta_ms"]
                    row["layers_11_47_delta_ms"] = sum(item["delta_ms"] for item in layer_deltas[11:])
                    row["top_delta_layers"] = sorted(
                        layer_deltas, key=lambda item: item["delta_ms"], reverse=True
                    )[:10]
                comparisons.append(row)

plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
summary = {
    "run_root": str(root),
    "config": {
        "bench_sizes": bench_sizes,
        "bench_patterns": bench_patterns,
        "bench_warmups": warmups,
        "bench_iters": iterations,
        "bench_burst_iters": burst_iters,
        "bench_burst_rounds": burst_rounds,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "native_cpu_offload_gb": native_cpu_offload_gb,
        "custom_cold_experts_per_layer": cold_experts_per_layer,
        "forward_only": True,
    },
    "plan": plan.get("metadata", {}),
    "cases": cases,
    "comparisons": comparisons,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  ) | tee "${RUN_ROOT}/summary.stdout.json"
}

main() {
  require_environment
  echo "run_root=${RUN_ROOT}" | tee -a "${RUN_ROOT}/run.log"
  echo "bench_sizes=${BENCH_SIZES}" | tee -a "${RUN_ROOT}/run.log"
  echo "patterns=${BENCH_PATTERNS}" | tee -a "${RUN_ROOT}/run.log"

  if [[ ! -s "${CUSTOM_PLAN}" ]]; then
    run_forward_case "trace_decode" "decode" "local_hbm" \
      "${PORT_BASE}" 1 0 "${TRACE_BENCH_WARMUPS}" "${TRACE_BENCH_ITERS}" \
      "${TRACE_BENCH_BURST_ITERS}" "${TRACE_BENCH_BURST_ROUNDS}"
    run_forward_case "trace_prefill" "prefill" "local_hbm" \
      "$((PORT_BASE + 1))" 1 0 "${TRACE_BENCH_WARMUPS}" \
      "${TRACE_BENCH_ITERS}" "${TRACE_BENCH_BURST_ITERS}" \
      "${TRACE_BENCH_BURST_ROUNDS}"
    build_cold_plan_from_traces
  else
    echo "reusing custom plan: ${CUSTOM_PLAN}" | tee -a "${RUN_ROOT}/run.log"
  fi

  local index=2
  for mode in decode prefill; do
    run_forward_case "local_${mode}" "${mode}" "local_hbm" \
      "$((PORT_BASE + index))" 0 1 "${BENCH_WARMUPS}" "${BENCH_ITERS}" \
      "${BENCH_BURST_ITERS}" "${BENCH_BURST_ROUNDS}"
    index=$((index + 1))
    run_forward_case "native_uva2_${mode}" "${mode}" "native_uva" \
      "$((PORT_BASE + index))" 0 1 "${BENCH_WARMUPS}" "${BENCH_ITERS}" \
      "${BENCH_BURST_ITERS}" "${BENCH_BURST_ROUNDS}"
    index=$((index + 1))
    run_forward_case "custom_per_layer${COLD_EXPERTS_PER_LAYER}_${mode}" \
      "${mode}" "custom_per_layer_cold" "$((PORT_BASE + index))" 0 1 \
      "${BENCH_WARMUPS}" "${BENCH_ITERS}" "${BENCH_BURST_ITERS}" \
      "${BENCH_BURST_ROUNDS}"
    index=$((index + 1))
  done

  write_summary
  wait_for_clean_gpu "after_all" || true
  echo "summary=${SUMMARY}"
}

main "$@"
