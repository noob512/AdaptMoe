#!/usr/bin/env bash
# Validate the target policy: initialize MoE experts on CPU, prefetch only the
# current router_topk active experts, and evict non-active experts back to CPU.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_active_experts_residency_run}"
LOG_FILE="${LOG_FILE:-log/stage_m_active_experts_residency.log}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
ACTIVE_EXPERTS_PER_LAYER="${ACTIVE_EXPERTS_PER_LAYER:-8}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-3221225472}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-1}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-127}"

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    --log-file) LOG_FILE="$2"; EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

echo "Stage M active-experts residency experiment"
echo "Run root: $RUN_ROOT"
echo "Log file: $LOG_FILE"
echo "Active experts/layer target: $ACTIVE_EXPERTS_PER_LAYER"
echo "Weight GPU budget: $WEIGHT_GPU_BYTES"
echo "KV GPU budget: $KV_GPU_BYTES"
echo "MoE registry plan: hot=$HOT_EXPERTS_PER_LAYER cold=$COLD_EXPERTS_PER_LAYER"

VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
HOT_EXPERTS_PER_LAYER="$HOT_EXPERTS_PER_LAYER" \
COLD_EXPERTS_PER_LAYER="$COLD_EXPERTS_PER_LAYER" \
WEIGHT_RUNTIME_PREFETCH_ENABLE=1 \
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$ACTIVE_EXPERTS_PER_LAYER" \
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-67108864}" \
WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="${WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER:-128}" \
WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP="${WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP:-1073741824}" \
WEIGHT_ACTIVE_EVICT_ENABLE=1 \
WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-128}" \
WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-128}" \
WEIGHT_VICTIM_ACTION=offload_cpu \
WEIGHT_GPU_BYTES="$WEIGHT_GPU_BYTES" \
REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-4096}" \
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-128}" \
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-32}" \
REAL_MAX_REGIONS_PER_POOL=0 \
INIT_WEIGHT_OFFLOAD_ENABLE=1 \
SCRATCH_DEVICE_DIRECT_ENABLE=0 \
SCRATCH_POOL_ENABLE=0 \
KV_RUNTIME_ENABLE=0 \
POOL_COORDINATOR_ENABLE=1 \
POOL_REGISTRY_ENABLE=1 \
AUTO_GAP_WATCH_ENABLE=0 \
FAULT_ADDRESS_LOG_ENABLE=0 \
"$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
  --run-root "$RUN_ROOT" \
  --log-file "$LOG_FILE" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --weight-executor "$WEIGHT_EXECUTOR" \
  --init-weight-offload-enable 1 \
  --weight-runtime-prefetch-enable 1 \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --weight-active-evict-enable 1 \
  --weight-active-evict-max-experts-per-layer "${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-128}" \
  --weight-max-victims-per-step "${WEIGHT_MAX_VICTIMS_PER_STEP:-128}" \
  "${EXTRA_ARGS[@]}"
