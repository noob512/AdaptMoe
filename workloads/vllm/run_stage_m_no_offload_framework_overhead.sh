#!/usr/bin/env bash
# No-offload control: keep all MoE experts hot and do not require driver_batch.
# This measures full_impl framework overhead without init weight offload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_no_offload_framework_overhead_run}"
LOG_FILE="${LOG_FILE:-log/stage_m_no_offload_framework_overhead.log}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"

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
  --no-require-driver \
  --init-weight-offload-enable 0 \
  --weight-runtime-prefetch-enable 0 \
  --weight-prefetch-trace-enable 0 \
  --weight-router-topk-every-n-steps 4 \
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
  --hot-experts-per-layer 128 \
  --cold-experts-per-layer 0 \
  --weight-gpu-bytes 32212254720 \
  --kv-gpu-bytes 2147483648 \
  --weight-active-evict-enable 0 \
  --weight-active-evict-max-experts-per-layer 0 \
  --weight-max-victims-per-step 0 \
  --server-max-num-seqs 1 \
  --request-rate 1
