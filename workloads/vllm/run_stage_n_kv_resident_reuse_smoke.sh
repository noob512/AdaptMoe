#!/usr/bin/env bash
# Stage N KV resident reuse smoke.
#
# Verifies that vLLM's KV BlockPool prefers free KV blocks that the UVM
# residency ledger already considers GPU-resident.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/stage_n_kv_resident_reuse_smoke_$(date -u +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_BLOCKS="${NUM_BLOCKS:-7}"
ALLOC_BLOCKS="${ALLOC_BLOCKS:-2}"

mkdir -p "$RUN_ROOT"

export PYTHONPATH="$SCRIPT_DIR/vllm:${PYTHONPATH:-}"
export VLLM_UVM_KV_RUNTIME_ENABLE="${VLLM_UVM_KV_RUNTIME_ENABLE:-1}"
export VLLM_UVM_KV_RUNTIME_TRACE_FILE="${VLLM_UVM_KV_RUNTIME_TRACE_FILE:-$RUN_ROOT/kv_runtime_trace.jsonl}"
export VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS="${VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS:-1}"
export VLLM_UVM_RESIDENCY_ENABLE="${VLLM_UVM_RESIDENCY_ENABLE:-1}"
export VLLM_UVM_RESIDENCY_TRACE_ENABLE="${VLLM_UVM_RESIDENCY_TRACE_ENABLE:-1}"
export VLLM_UVM_RESIDENCY_TRACE_FILE="${VLLM_UVM_RESIDENCY_TRACE_FILE:-$RUN_ROOT/pool_residency_trace.jsonl}"
export VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE="${VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE:-1}"

echo "RUN_ROOT=$RUN_ROOT"
"$PYTHON_BIN" "$SCRIPT_DIR/stage_n_kv_resident_reuse_smoke.py" \
  --run-root "$RUN_ROOT" \
  --num-blocks "$NUM_BLOCKS" \
  --alloc-blocks "$ALLOC_BLOCKS" \
  | tee "$RUN_ROOT/run.log"

grep -E '"action":"gpu_resident_free_block_select"' \
  "$VLLM_UVM_KV_RUNTIME_TRACE_FILE" \
  > "$RUN_ROOT/selection_trace.log" || true

echo "summary=$RUN_ROOT/stage_n_kv_resident_reuse_summary.json"
echo "selection_trace=$RUN_ROOT/selection_trace.log"
