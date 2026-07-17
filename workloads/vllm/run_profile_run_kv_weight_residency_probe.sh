#!/usr/bin/env bash
# Test two residency claims in one offline vLLM process:
# 1) whether profile_run leaves all MoE expert weight slices GPU resident;
# 2) whether KV cache tensors allocated via torch.zeros are GPU resident.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/profile_run_kv_weight_residency_$(date -u +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
MAX_TOKENS="${MAX_TOKENS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-536870912}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"
HF_OFFLINE="${HF_OFFLINE:-1}"

WEIGHT_MAP="$RUN_ROOT/vllm_uvm_weight_regions.jsonl"
ADDRESS_LOG="$RUN_ROOT/vllm_uvm_address_regions.log"
ALLOCATOR_LOG="$RUN_ROOT/vllm_uvm_allocator.log"
PHASE_F_DEBUG_LOG="$RUN_ROOT/phase_f_user_debug.log"
KV_PROBE="$RUN_ROOT/vllm_uvm_kv_init_residency_probe.jsonl"
GEN_JSON="$RUN_ROOT/vllm_generate_once.json"
RUN_LOG="$RUN_ROOT/run.log"
SUMMARY_JSON="$RUN_ROOT/summary.json"
SUMMARY_MD="$RUN_ROOT/summary.md"

mkdir -p "$RUN_ROOT"

if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

echo "profile_run/KV residency probe"
echo "RUN_ROOT=$RUN_ROOT"
echo "MODEL=$MODEL"
echo "KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES"
echo "WEIGHT_MAP=$WEIGHT_MAP"
echo "KV_PROBE=$KV_PROBE"

run_status=0
set +e
timeout "$TIMEOUT_SEC" env \
  VLLM_USE_UVM=1 \
  VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
  VLLM_UVM_PHASE_F_FD_MODE=adopt \
  VLLM_UVM_PHASE_F_CPP_HELPER=1 \
  VLLM_UVM_PHASE_F_DEBUG_LOG="$PHASE_F_DEBUG_LOG" \
  VLLM_UVM_ADDRESS_LOG_ENABLE=1 \
  VLLM_UVM_ADDRESS_LOG_FILE="$ADDRESS_LOG" \
  VLLM_UVM_LOG_FILE="$ALLOCATOR_LOG" \
  VLLM_UVM_TRACE_MIN_BYTES=4096 \
  VLLM_UVM_WEIGHT_MAP_ENABLE=1 \
  VLLM_UVM_WEIGHT_MAP_FILE="$WEIGHT_MAP" \
  VLLM_UVM_WEIGHT_MAP_INCLUDE_ALIASES=0 \
  VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
  VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
  VLLM_UVM_WEIGHT_MAP_QUERY_EXPERT_SLICES=1 \
  VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=1 \
  VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_DEVICE=0 \
  VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ROUND_PAGES=1 \
  VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
  VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
  VLLM_UVM_KV_DEVICE_DIRECT_ENABLE=0 \
  VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 \
  VLLM_UVM_KV_INIT_FAULT_PROBE=1 \
  VLLM_UVM_KV_INIT_FAULT_PROBE_FILE="$KV_PROBE" \
  VLLM_UVM_KV_INIT_FAULT_PROBE_SYNC=1 \
  VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY=1 \
  VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY_DEVICE=0 \
  VLLM_UVM_KV_INIT_PROBE_QUERY_RESIDENCY_ROUND_PAGES=1 \
  uv run python vllm/uvm_test/vllm_generate_once.py \
    --model "$MODEL" \
    --prompt "hello" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
    --cpu-offload-gb "$CPU_OFFLOAD_GB" \
    --output-json "$GEN_JSON" \
  > "$RUN_LOG" 2>&1
run_status=$?
set -e

python3 "$SCRIPT_DIR/analyze_profile_run_kv_weight_residency.py" \
  --run-root "$RUN_ROOT" \
  --weight-map "$WEIGHT_MAP" \
  --kv-probe "$KV_PROBE" \
  --run-status "$run_status" \
  --summary-json "$SUMMARY_JSON" \
  --summary-md "$SUMMARY_MD" \
  | tee "$RUN_ROOT/analysis_stdout.json"

echo "run_status=$run_status"
echo "summary_json=$SUMMARY_JSON"
echo "summary_md=$SUMMARY_MD"
exit "$run_status"
