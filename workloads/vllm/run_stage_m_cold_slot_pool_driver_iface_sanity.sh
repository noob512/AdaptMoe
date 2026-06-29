#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-log/stage_m_cold_slot_pool_driver_iface_sanity}"
STREAMING_BYTES_LIST="${STREAMING_BYTES_LIST:-3221225472}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-32}"
PROMPTS="${PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-4}"
REQUEST_RATE="${REQUEST_RATE:-2}"
BASE_PORT="${BASE_PORT:-8096}"
COLD_SLOT_POOL_SLOTS_PER_LAYER="${COLD_SLOT_POOL_SLOTS_PER_LAYER:-40}"
COLD_SLOT_POOL_DRIVER_MODE="${COLD_SLOT_POOL_DRIVER_MODE:-validate}"
COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED="${COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED:-0}"
COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP="${COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP:-auto}"
COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT="${COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT:-128}"
SKIP_COLD_CHECKER="${SKIP_COLD_CHECKER:-1}"

export RUN_ROOT
export STREAMING_BYTES_LIST
export MAX_NUM_SEQS_LIST
export PROMPTS
export OUTPUT_LEN
export REQUEST_RATE
export BASE_PORT
export RUN_FAULT_ONLY=0
export RUN_COLD_STREAMING=1
export RUN_GLOBAL_UVM_REFERENCE=0
export COLD_SLOT_POOL_ENABLE=1
export COLD_SLOT_POOL_TRACE_ONLY=0
export COLD_SLOT_POOL_BACKEND=driver_slot_remap
export COLD_SLOT_POOL_SLOTS_PER_LAYER
export COLD_SLOT_POOL_OVERFLOW_MODE=trace_only
export COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT
export COLD_SLOT_POOL_SYNC_MIGRATION=0
export COLD_SLOT_POOL_DRIVER_MODE
export COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED
export COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP
export SKIP_COLD_CHECKER

case "$COLD_SLOT_POOL_DRIVER_MODE" in
  slot_va_block_remap)
    export CONTROLLED_REMAP_PREFLIGHT="${CONTROLLED_REMAP_PREFLIGHT:-1}"
    export CLEANUP_GPU_DISABLE_NVIDIA_SMI="${CLEANUP_GPU_DISABLE_NVIDIA_SMI:-1}"
    export UVM_MOE_ALLOW_NVIDIA_SMI_UUID="${UVM_MOE_ALLOW_NVIDIA_SMI_UUID:-0}"
    export CLEANUP_GPU_NVIDIA_SMI_TIMEOUT="${CLEANUP_GPU_NVIDIA_SMI_TIMEOUT:-1}"
    export UVM_MOE_NVIDIA_SMI_TIMEOUT="${UVM_MOE_NVIDIA_SMI_TIMEOUT:-1}"
    ;;
esac

"$SCRIPT_DIR/run_stage_m_cold_streaming_vs_fault_only_ab.sh"

slot_pool_query_args=(
  --dump-only
  --out-json "$RUN_ROOT/moe_slot_pool_state_smoke_after_vllm.json"
)
case "$COLD_SLOT_POOL_DRIVER_MODE" in
  dry_run|validate)
    slot_pool_query_args+=(--expect-metadata-only-state)
    ;;
  copy_only)
    slot_pool_query_args+=(--expect-copy-only-state)
    ;;
  gpu_copy_only)
    slot_pool_query_args+=(--expect-gpu-copy-only-state)
    ;;
  slot_pte_remap|slot_va_block_remap)
    case "${VLLM_UVM_COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP:-${COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP:-auto}}" in
      1|true|yes|on|auto)
        slot_pool_query_args+=(--expect-metadata-only-state)
        ;;
      *)
        slot_pool_query_args+=(--expect-slot-pte-state)
        ;;
    esac
    ;;
esac

python3 "$SCRIPT_DIR/check_moe_slot_pool_state_smoke.py" "${slot_pool_query_args[@]}"

echo
echo "Slot pool driver interface reports:"
find "$RUN_ROOT" -name cold_slot_pool_report.md -print | sort
find "$RUN_ROOT" -name 'moe_slot_pool_state_smoke_after_vllm.json' -print | sort
