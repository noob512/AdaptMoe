#!/usr/bin/env bash
# Runtime prefetch debug: enable a small prefetch trace and disable active
# eviction to inspect why router_topk prefetch does or does not issue actions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_prefetch_trace_debug_run}"
LOG_FILE="${LOG_FILE:-log/stage_m_prefetch_trace_debug.log}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"  # 24 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"           # 2 GiB
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-112}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"

pkill -f "vllm bench serve" || true
pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-8192}" \
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-1024}" \
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-16}" \
"$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
  --run-root "$RUN_ROOT" \
  --log-file "$LOG_FILE" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --weight-executor driver_batch \
  --init-weight-offload-enable 1 \
  --weight-runtime-prefetch-enable 1 \
  --weight-prefetch-trace-enable 1 \
  --weight-router-topk-every-n-steps 1 \
  --scratch-device-direct-enable 0 \
  --scratch-pool-enable 0 \
  --residency-enable 1 \
  --residency-trace-enable 1 \
  --kv-runtime-enable 0 \
  --pool-coordinator-enable 1 \
  --pool-coordinator-trace-enable 0 \
  --pool-registry-enable 1 \
  --auto-gap-watch-enable 0 \
  --fault-address-log-enable 0 \
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --weight-active-evict-enable 0 \
  --weight-active-evict-max-experts-per-layer 0 \
  --weight-max-victims-per-step 0 \
  --server-max-num-seqs 1 \
  --request-rate 1
