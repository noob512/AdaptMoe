#!/usr/bin/env bash
# Run Stage C attention A/B twice with identical settings:
# 1) C1 device_direct backend: cuda_malloc
# 2) C2 device_direct backend: cuda_malloc_async
# Then compare the two device-direct runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PROMPTS="${PROMPTS:-10}"
REQUEST_RATE="${REQUEST_RATE:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DEVICE_DIRECT_MAX_TOTAL_BYTES="${DEVICE_DIRECT_MAX_TOTAL_BYTES:-1048576}"
ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="${ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-${DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT_DIR="${ROOT_OUT_DIR:-/tmp/vllm_stage_c_attention_backend_ab_${RUN_ID}}"

SYNC_OUT_DIR="$ROOT_OUT_DIR/cuda_malloc"
ASYNC_OUT_DIR="$ROOT_OUT_DIR/cuda_malloc_async"
SYNC_JSON="$SYNC_OUT_DIR/vllm_stage_c_attention_p${PROMPTS}_ab_comparison.json"
ASYNC_JSON="$ASYNC_OUT_DIR/vllm_stage_c_attention_p${PROMPTS}_ab_comparison.json"
BACKEND_JSON="$ROOT_OUT_DIR/vllm_stage_c_attention_backend_p${PROMPTS}_comparison.json"

mkdir -p "$ROOT_OUT_DIR"

echo "==========================================================="
echo " Stage C Attention Backend A/B"
echo " Root output dir: $ROOT_OUT_DIR"
echo " Model: $MODEL"
echo " Prompts: $PROMPTS"
echo " Request rate: $REQUEST_RATE"
echo " Output len: $OUTPUT_LEN"
echo " Device-direct max total bytes: $DEVICE_DIRECT_MAX_TOTAL_BYTES"
echo " Async pool release threshold: ${ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-<cuda-default>}"
echo "==========================================================="

echo
echo "==========================================================="
echo " Phase 1: C1 sync backend cuda_malloc"
echo "==========================================================="
PROMPTS="$PROMPTS" \
REQUEST_RATE="$REQUEST_RATE" \
OUTPUT_LEN="$OUTPUT_LEN" \
MODEL="$MODEL" \
DEVICE_DIRECT_MAX_TOTAL_BYTES="$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
DEVICE_DIRECT_BACKEND="cuda_malloc" \
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="" \
OUT_DIR="$SYNC_OUT_DIR" \
RUN_ID="${RUN_ID}_cuda_malloc" \
./run_stage_c_attention_p20_ab.sh

echo
echo "==========================================================="
echo " Phase 2: C2 async backend cuda_malloc_async"
echo "==========================================================="
PROMPTS="$PROMPTS" \
REQUEST_RATE="$REQUEST_RATE" \
OUTPUT_LEN="$OUTPUT_LEN" \
MODEL="$MODEL" \
DEVICE_DIRECT_MAX_TOTAL_BYTES="$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
DEVICE_DIRECT_BACKEND="cuda_malloc_async" \
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="$ASYNC_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" \
OUT_DIR="$ASYNC_OUT_DIR" \
RUN_ID="${RUN_ID}_cuda_malloc_async" \
./run_stage_c_attention_p20_ab.sh

echo
echo "==========================================================="
echo " Phase 3: Compare C1 sync vs C2 async"
echo "==========================================================="
python3 "$SCRIPT_DIR/compare_stage_c_backend_ab.py" \
  --sync-json "$SYNC_JSON" \
  --async-json "$ASYNC_JSON" \
  --output-json "$BACKEND_JSON"

echo
echo "==========================================================="
echo " Done"
echo " Backend comparison: $BACKEND_JSON"
echo " C1 sync raw logs: $SYNC_OUT_DIR"
echo " C2 async raw logs: $ASYNC_OUT_DIR"
echo "==========================================================="
