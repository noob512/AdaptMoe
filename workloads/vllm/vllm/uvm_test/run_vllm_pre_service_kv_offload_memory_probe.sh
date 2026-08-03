#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_pre_service_kv_offload_memory_probe_$(date +%Y%m%d_%H%M%S)}"
UVM_KO="${UVM_KO:-}"

KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-10737418240}"
INITIAL_KV_BLOCKS="${INITIAL_KV_BLOCKS:-450}"
KV_TARGET_BLOCKS="${KV_TARGET_BLOCKS:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.97}"
SETTLE_SECONDS="${SETTLE_SECONDS:-10}"
HOLD_SECONDS="${HOLD_SECONDS:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
RELOAD_UVM="${RELOAD_UVM:-0}"

RESULT="${RUN_ROOT}/result.json"
KV_TRACE="${RUN_ROOT}/kv_physical_trace.jsonl"
KV_RUNTIME_TRACE="${RUN_ROOT}/kv_runtime_trace.jsonl"
RUN_LOG="${RUN_ROOT}/run.log"
PARAMETERS="${RUN_ROOT}/uvm_parameters.txt"
RETURN_CODE="${RUN_ROOT}/returncode.txt"

mkdir -p "${RUN_ROOT}"

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

require_clean_gpu() {
  local used
  used="$(gpu_used_mib || echo 999999)"
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || [[ "${used}" -gt 1024 ]]; then
    echo "GPU memory is not clean: ${used} MiB used" >&2
    nvidia-smi >&2 || true
    return 1
  fi
}

reload_uvm_if_requested() {
  if [[ "${RELOAD_UVM}" != "1" ]]; then
    return 0
  fi
  require_clean_gpu
  run_sudo modprobe -r nvidia_uvm
  if [[ -n "${UVM_KO}" ]]; then
    if [[ ! -f "${UVM_KO}" ]]; then
      echo "nvidia-uvm module not found: ${UVM_KO}" >&2
      return 1
    fi
    run_sudo insmod "${UVM_KO}" \
      uvm_perf_prefetch_enable=0 \
      uvm_perf_access_counter_migration_enable=0 \
      uvm_perf_thrashing_enable=0
  else
    run_sudo modprobe nvidia_uvm \
      uvm_perf_prefetch_enable=0 \
      uvm_perf_access_counter_migration_enable=0 \
      uvm_perf_thrashing_enable=0
  fi
}

record_and_require_migration_disabled() {
  : > "${PARAMETERS}"
  local parameter
  for parameter in \
    uvm_perf_prefetch_enable \
    uvm_perf_access_counter_migration_enable \
    uvm_perf_thrashing_enable; do
    local path="/sys/module/nvidia_uvm/parameters/${parameter}"
    local value=""
    if [[ -r "${path}" ]]; then
      value="$(<"${path}")"
    fi
    printf '%s=%s\n' "${parameter}" "${value}" | tee -a "${PARAMETERS}"
    if [[ "${value}" != "0" ]]; then
      echo "Required UVM parameter ${parameter}=0 is not active" >&2
      return 1
    fi
  done
}

reload_uvm_if_requested
record_and_require_migration_disabled
require_clean_gpu

set +e
(
  cd "${VLLM_ROOT}"
  env \
    VLLM_USE_UVM=0 \
    VLLM_DISABLE_OPENAI_METRICS=1 \
    VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 \
    VLLM_UVM_KV_CACHE_MANAGED=1 \
    VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=1 \
    VLLM_UVM_KV_RUNTIME_MODE=trace_only \
    VLLM_UVM_KV_RUNTIME_TRACE_FILE="${KV_RUNTIME_TRACE}" \
    VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1 \
    VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS="${INITIAL_KV_BLOCKS}" \
    VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS="${KV_TARGET_BLOCKS}" \
    VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1 \
    VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch \
    VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1 \
    VLLM_UVM_POST_WARMUP_REBALANCE_TRACE_FILE="${KV_TRACE}" \
    VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0 \
    VLLM_UVM_RESIDENCY_TRACE_ENABLE=0 \
    VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE=0 \
    timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" \
      "${SCRIPT_DIR}/vllm_warmup_only.py" \
      --model "${MODEL}" \
      --output-json "${RESULT}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
      --settle-seconds "${SETTLE_SECONDS}" \
      --hold-seconds "${HOLD_SECONDS}"
) > "${RUN_LOG}" 2>&1
rc=$?
set -e

printf '%s\n' "${rc}" > "${RETURN_CODE}"
cat "${RUN_LOG}"

python3 - "${RESULT}" "${KV_TRACE}" "${RETURN_CODE}" "${RUN_ROOT}/summary.json" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
kv_trace_path = Path(sys.argv[2])
return_code_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
kv_records = []
if kv_trace_path.exists():
    for line in kv_trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            kv_records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

kv_record = kv_records[-1] if kv_records else {}
after = result.get("gpu_memory_after_settle") or result.get("gpu_memory_after_warmup") or {}
cuda_after = result.get("cuda_memory_after_settle") or []
cuda_worker = cuda_after[0] if cuda_after else {}
summary = {
    "returncode": int(return_code_path.read_text().strip()),
    "mode": result.get("mode"),
    "no_forward_or_generate": True,
    "gpu_memory_before": result.get("gpu_memory_before"),
    "gpu_memory_after_warmup": result.get("gpu_memory_after_warmup"),
    "gpu_memory_after_settle": result.get("gpu_memory_after_settle"),
    "gpu_free_after_kv_offload_mib": after.get("memory_free_mib"),
    "gpu_used_after_kv_offload_mib": after.get("memory_used_mib"),
    "cuda_free_after_kv_offload_mib": cuda_worker.get("free_mib"),
    "cuda_used_after_kv_offload_mib": cuda_worker.get("used_mib"),
    "cuda_memory_after_settle": cuda_after,
    "uvm_proactive_migration_parameters": result.get(
        "uvm_proactive_migration_parameters"
    ),
    "kv_offload": {
        "ok": kv_record.get("ok"),
        "offload_complete": kv_record.get("offload_complete"),
        "offload_successes": kv_record.get("offload_successes"),
        "range_count": kv_record.get("range_count"),
        "range_bytes": kv_record.get("range_bytes"),
        "requested_target_blocks": kv_record.get("requested_target_blocks"),
        "target_blocks": kv_record.get("target_blocks"),
        "elapsed_seconds": kv_record.get("elapsed_seconds"),
        "executor": kv_record.get("executor"),
    },
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "run_root=${RUN_ROOT}"
exit "${rc}"
