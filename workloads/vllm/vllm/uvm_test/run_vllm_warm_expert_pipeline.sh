#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
SOURCE_PLAN="${SOURCE_PLAN:-/tmp/vllm_dynamic_expert_swap_20260729_01/initial_cold_plan.json}"
FIXTURE="${FIXTURE:-/tmp/vllm_dynamic_expert_swap_20260729_01/sharegpt_shift_128.json}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_warm_expert_pipeline_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
CASES="${CASES:-sync,pipeline}"
WARM_LAYER_START="${WARM_LAYER_START:-8}"
WARM_LAYER_END="${WARM_LAYER_END:-15}"
REQUEST_COUNT="${REQUEST_COUNT:-8}"
REQUEST_OFFSET="${REQUEST_OFFSET:-64}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-32}"
KV_BYTES="${KV_BYTES:-1288490188}"
SLOT_BYTES="${SLOT_BYTES:-524288}"
BANK_COUNT="${BANK_COUNT:-2}"
PREFETCH_DISTANCE="${PREFETCH_DISTANCE:-1}"
SLOTS_PER_BANK="${SLOTS_PER_BANK:-9}"
SLOT_COUNT="${SLOT_COUNT:-$((BANK_COUNT * SLOTS_PER_BANK))}"
FUSED_BANK_REUSE="${FUSED_BANK_REUSE:-1}"
RESERVED_DEBUG_TRACE="${RESERVED_DEBUG_TRACE:-0}"
PLAN="${RUN_ROOT}/warm_plan.json"
SUMMARY="${RUN_ROOT}/summary.json"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

wait_for_clean_gpu() {
  local used=999999
  for _ in $(seq 1 120); do
    used="$(gpu_used_mib || echo 999999)"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le 1024 ]]; then
      return 0
    fi
    sleep 2
  done
  echo "GPU did not become idle: ${used} MiB" >&2
  return 1
}

reset_reserved_pool() {
  local output="$1"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - <<'PY'
import json
from vllm.device_allocator.uvm import driver_expert_reserved_pool_reset

print(json.dumps(driver_expert_reserved_pool_reset(prefer_existing=True),
                 indent=2, sort_keys=True))
PY
  ) > "${output}" 2>&1 || true
}

build_plan() {
  python3 - "${SOURCE_PLAN}" "${PLAN}" "${WARM_LAYER_START}" \
    "${WARM_LAYER_END}" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
start, end = int(sys.argv[3]), int(sys.argv[4])
records = list(source.get("offload_plan") or []) + list(source.get("hot_plan") or [])
warm = []
for layer_id in range(start, end + 1):
    layer = sorted(
        (item for item in records if int(item.get("layer_id", -1)) == layer_id),
        key=lambda item: (
            int(item.get("pilot_token_count", 0)),
            int(item.get("expert_id", -1)),
        ),
    )
    if not layer:
        raise SystemExit(f"source plan has no records for layer {layer_id}")
    selected = dict(layer[len(layer) // 2])
    selected["reason"] = "median_frequency_warm_pipeline"
    warm.append(selected)

payload = {
    "policy": "routing_heatmap_v1",
    "metadata": {
        "policy": "routing_heatmap_v1",
        "source_plan": str(Path(sys.argv[1])),
        "warm_layer_start": start,
        "warm_layer_end": end,
        "warm_experts": len(warm),
        "selection": "per-layer median pilot_token_count",
    },
    "offload_plan": [],
    "prefetch_plan": [],
    "hot_plan": [],
    "warm_plan": warm,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                  encoding="utf-8")
print(json.dumps(payload["metadata"], indent=2, sort_keys=True))
PY
}

run_case() {
  local name="$1"
  local background=0
  local dir="${RUN_ROOT}/${name}"
  [[ "${name}" == "pipeline" ]] && background=1
  mkdir -p "${dir}"
  wait_for_clean_gpu
  reset_reserved_pool "${dir}/reset_before.json"
  local kernel_since
  kernel_since="$(date --iso-8601=seconds)"

  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
      VLLM_UVM_KV_CACHE_MANAGED=0 \
      VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
      VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=restore \
      VLLM_UVM_DRIVER_TAG_ENABLE=1 \
      VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup \
      VLLM_UVM_INIT_WEIGHT_PLAN_FILE="${PLAN}" \
      VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN=1 \
      VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=1 \
      VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS=1 \
      VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=0 \
      VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=1 \
      VLLM_UVM_INIT_WEIGHT_TARGET_ROLES=moe_gate_up,moe_down \
      VLLM_UVM_INIT_WEIGHT_SYNC=1 \
      VLLM_UVM_INIT_WEIGHT_TRACE_FILE="${dir}/init_trace.jsonl" \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1 \
      VLLM_UVM_WEIGHT_PREFETCH_MODE=prefetch \
      VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN=0 \
      VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE="${PLAN}" \
      VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES=moe_gate_up,moe_down \
      VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async \
      VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=1 \
      VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE="${dir}/warm_trace.jsonl" \
      VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE="${RESERVED_DEBUG_TRACE}" \
      VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE_FILE="${dir}/reserved_trace.jsonl" \
      VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1 \
      VLLM_UVM_EXPERT_ENVELOPE_BYTES="${SLOT_BYTES}" \
      VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT=1 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=1 \
      VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M=1 \
      VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE=1 \
      VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=layer_mod_bank \
      VLLM_UVM_EXPERT_RESERVED_BANK_COUNT="${BANK_COUNT}" \
      VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT="${SLOT_COUNT}" \
      VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES="${SLOT_BYTES}" \
      VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERTS_PER_LAYER=1 \
      VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS=0 \
      VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE=0 \
      VLLM_UVM_EXPERT_RESERVED_INVALIDATE_ON_LAYER_DONE=0 \
      VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE=1 \
      VLLM_UVM_EXPERT_RESERVED_FUSED_BANK_REUSE="${FUSED_BANK_REUSE}" \
      VLLM_UVM_EXPERT_RESERVED_WAIT_ON_REUSE=1 \
      VLLM_UVM_EXPERT_RESERVED_ASYNC=0 \
      VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE=1 \
      VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT=1 \
      VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE=0 \
      VLLM_UVM_EXPERT_RESERVED_FAIL_FAST=1 \
      VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE=1 \
      VLLM_UVM_EXPERT_WARM_BACKGROUND_ENABLE="${background}" \
      VLLM_UVM_EXPERT_WARM_FAIL_CLOSED=1 \
      VLLM_UVM_EXPERT_WARM_PREFETCH_DISTANCE="${PREFETCH_DISTANCE}" \
      VLLM_UVM_EXPERT_WARM_PREFETCH_MAX_EXPERTS_PER_LAYER=1 \
      VLLM_UVM_EXPERT_WARM_ASYNC_MIN_TOKENS=0 \
      VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
      timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
        "${SCRIPT_DIR}/vllm_generate_phased.py" \
        --model "${MODEL}" \
        --fixture "${FIXTURE}" \
        --output "${dir}/result.json" \
        --progress "${dir}/progress.jsonl" \
        --phase-names measured \
        --phase-request-counts "${REQUEST_COUNT}" \
        --phase-request-offsets "${REQUEST_OFFSET}" \
        --phase-output-tokens "${OUTPUT_TOKENS}" \
        --grow-after-phase -1 \
        --kv-cache-memory-bytes "${KV_BYTES}" \
        --max-model-len 512 \
        --max-num-seqs "${REQUEST_COUNT}" \
        --max-num-batched-tokens 2048 \
        --gpu-memory-utilization 0.97 \
        --disable-prefix-caching
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${dir}/returncode.txt"
  journalctl -k --since "${kernel_since}" --no-pager \
    > "${dir}/kernel.log" 2>/dev/null || true
  wait_for_clean_gpu || true
}

write_summary() {
  python3 - "${RUN_ROOT}" "${SUMMARY}" "${PLAN}" <<'PY'
import json
import re
import sys
from pathlib import Path

root, output, plan_path = map(Path, sys.argv[1:4])

def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_jsonl(path):
    result = []
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result

plan = load_json(plan_path)
expected_warm_experts = sorted({
    (int(item["layer_id"]), int(item["expert_id"]))
    for item in (plan.get("warm_plan") or [])
})

def record_layer_id(record):
    if record.get("layer_id") is not None:
        return int(record["layer_id"])
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)",
                      str(record.get("layer_name") or ""))
    return int(match.group(1)) if match else -1

cases = {}
for name in ("sync", "pipeline"):
    directory = root / name
    result = load_json(directory / "result.json")
    phase = (result.get("phases") or [{}])[-1]
    warm_trace = load_jsonl(directory / "warm_trace.jsonl")
    barriers = [r for r in warm_trace
                if r.get("action") == "warm_prefetch_target_barrier"]
    init_trace = load_jsonl(directory / "init_trace.jsonl")
    warm_release = [r for r in init_trace
                    if r.get("expert_init_class") == "warm"
                    and str(r.get("action", "")).endswith("_issued")
                    and r.get("success") is True]
    kernel = (directory / "kernel.log").read_text(errors="replace") \
        if (directory / "kernel.log").exists() else ""
    run_log = (directory / "run.log").read_text(errors="replace") \
        if (directory / "run.log").exists() else ""
    worker_us = sum(float(r.get("worker_us") or 0) for r in barriers)
    wait_us = sum(float(r.get("target_wait_us") or 0) for r in barriers)
    hidden_us = sum(float(r.get("pipeline_hidden_us") or 0) for r in barriers)
    direct_overlap_ms = sum(
        float(r.get("direct_compute_ioctl_overlap_ms") or 0) for r in barriers
    )
    actual_warm_experts = sorted({
        (record_layer_id(record), int(record["global_expert_id"]))
        for record in warm_release
    })
    cases[name] = {
        "returncode": int((directory / "returncode.txt").read_text().strip())
            if (directory / "returncode.txt").exists() else None,
        "status": result.get("status"),
        "completed_requests": phase.get("completed_requests"),
        "output_tokens_per_second": phase.get("output_tokens_per_second"),
        "mean_tpot_ms": phase.get("mean_tpot_ms"),
        "p99_tpot_ms": phase.get("p99_tpot_ms"),
        "mean_ttft_ms": phase.get("mean_ttft_ms"),
        "barrier_records": len(barriers),
        "coverage_ok_records": sum(r.get("coverage_ok") is True for r in barriers),
        "coverage_failed_records": sum(r.get("coverage_ok") is not True for r in barriers),
        "logical_coverage_ok_records": sum(
            r.get("logical_coverage_ok") is True for r in barriers
        ),
        "physical_coverage_ok_records": sum(
            r.get("physical_coverage_ok") is True for r in barriers
        ),
        "expected_regions": sum(int(r.get("expected_records") or 0) for r in barriers),
        "successful_regions": sum(int(r.get("successful_records") or 0) for r in barriers),
        "expected_physical_regions": sum(
            int(r.get("expected_physical_records") or 0) for r in barriers
        ),
        "successful_physical_regions": sum(
            int(r.get("successful_physical_records") or 0) for r in barriers
        ),
        "assigned_physical_regions": sum(
            int(r.get("assigned_physical_records") or 0) for r in barriers
        ),
        "driver_requested_bytes": sum(
            int(r.get("driver_requested_bytes") or 0) for r in barriers
        ),
        "driver_accepted_bytes": sum(
            int(r.get("driver_accepted_bytes") or 0) for r in barriers
        ),
        "driver_ordinary_resident_bytes": sum(
            int(r.get("driver_ordinary_resident_bytes") or 0) for r in barriers
        ),
        "driver_footprint_contract_broken": sum(
            int(r.get("driver_footprint_contract_broken") or 0)
            for r in barriers
        ),
        "worker_total_ms": worker_us / 1000.0,
        "target_wait_total_ms": wait_us / 1000.0,
        "pipeline_hidden_total_ms": hidden_us / 1000.0,
        "pipeline_overlap_ratio": hidden_us / worker_us if worker_us else None,
        "overlap_observed_at_barrier": sum(
            r.get("overlap_observed_at_barrier") is True for r in barriers
        ),
        "direct_compute_ioctl_overlap_records": sum(
            r.get("direct_compute_ioctl_overlap_observed") is True
            for r in barriers
        ),
        "direct_compute_ioctl_overlap_total_ms": direct_overlap_ms,
        "warm_release_records": len(warm_release),
        "warm_release_bytes": sum(int(r.get("bytes") or 0) for r in warm_release),
        "expected_warm_experts": [list(pair) for pair in expected_warm_experts],
        "actual_warm_experts": [list(pair) for pair in actual_warm_experts],
        "warm_plan_exact_match": actual_warm_experts == expected_warm_experts,
        "warm_range_fault_lines": sum(
            "fault" in line.lower() and "expert_reserved" in line.lower()
            for line in kernel.splitlines()
        ),
        "fatal_lines": len(re.findall(
            r"out of memory|illegal memory|xid|uvm fatal|mmu fault",
            run_log + "\n" + kernel,
            flags=re.IGNORECASE,
        )),
    }

sync = cases["sync"].get("output_tokens_per_second")
pipeline = cases["pipeline"].get("output_tokens_per_second")
summary = {
    "run_root": str(root),
    "plan": plan,
    "cases": cases,
    "pipeline_vs_sync_throughput_pct": (
        (float(pipeline) / float(sync) - 1.0) * 100.0
        if sync and pipeline else None
    ),
}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

[[ -s "${SOURCE_PLAN}" && -s "${FIXTURE}" ]] || {
  echo "missing source plan or fixture" >&2
  exit 2
}
[[ -e /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable ]] || {
  echo "loaded nvidia_uvm lacks reserved-slot support" >&2
  exit 2
}

build_plan | tee "${RUN_ROOT}/plan_build.log"
IFS=',' read -r -a case_names <<< "${CASES}"
for name in "${case_names[@]}"; do
  run_case "${name}"
done
write_summary
reset_reserved_pool "${RUN_ROOT}/reset_final.json"
echo "run_root=${RUN_ROOT}"
