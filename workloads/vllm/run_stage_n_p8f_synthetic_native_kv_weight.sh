#!/usr/bin/env bash
# Stage N P8f synthetic native KV/WEIGHT proof.
#
# This bypasses vLLM and uses phase_f_smoke to allocate managed memory, register
# KV and WEIGHT tags, touch native-size blocks on GPU, and validate native
# va_range/va_block accounting plus same-pool FIFO reclaim.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8f_synthetic_native_kv_weight_$(date -u +%Y%m%d_%H%M%S)}"
SYNTHETIC_ROOT="$RUN_ROOT/synthetic"
SUMMARY="$RUN_ROOT/stage_n_p8f_synthetic_native_kv_weight_summary.txt"
SHELL_LOG="$RUN_ROOT/shell.log"
STRICT="${STRICT:-1}"

BLOCK_BYTES="${BLOCK_BYTES:-2097152}"
BLOCKS="${BLOCKS:-4}"
BUDGET_BLOCKS="${BUDGET_BLOCKS:-2}"
POOLS="${POOLS:-1 2}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-1}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-2}"

mkdir -p "$RUN_ROOT"

grep_count() {
  local regex="$1"
  local root="$2"
  local count
  set +e
  count="$(grep -R -h -E "$regex" "$root" 2>/dev/null | wc -l | tr -d ' ')"
  set -e
  printf '%s\n' "${count:-0}"
}

grep_tail() {
  local regex="$1"
  local root="$2"
  set +e
  grep -R -h -E "$regex" "$root" 2>/dev/null | tail -n 120
  set -e
}

echo "Stage N P8f synthetic native KV/WEIGHT proof started."
echo "RUN_ROOT=$RUN_ROOT"

set +e
env \
  RUN_ROOT="$SYNTHETIC_ROOT" \
  BLOCK_BYTES="$BLOCK_BYTES" \
  BLOCKS="$BLOCKS" \
  BUDGET_BLOCKS="$BUDGET_BLOCKS" \
  POOLS="$POOLS" \
  REQUIRE_RECLAIM="$REQUIRE_RECLAIM" \
  ALLOWED_POST_FAULT_SLACK_BLOCKS="$ALLOWED_POST_FAULT_SLACK_BLOCKS" \
  UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
  UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=0 \
  UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
  UVM_PHASE_N_AUTO_RECLAIM_ENABLE=1 \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE=1 \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY=1 \
  UVM_PHASE_N_AUTO_RECLAIM_EVERY=1 \
  UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY=0 \
  UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS=16 \
  UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS=0 \
  UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS=0 \
  ./run_stage_n_p8d_synthetic_block_quota.sh \
  > "$SHELL_LOG" 2>&1
synthetic_status=$?
set -e

native_attach_kv="$(grep_count 'phase_n_native_range_tag_attach.*pool=1' "$SYNTHETIC_ROOT")"
native_attach_weight="$(grep_count 'phase_n_native_range_tag_attach.*pool=2' "$SYNTHETIC_ROOT")"
native_resident_kv="$(grep_count 'phase_n_native_block_resident.*pool=1' "$SYNTHETIC_ROOT")"
native_resident_weight="$(grep_count 'phase_n_native_block_resident.*pool=2' "$SYNTHETIC_ROOT")"
native_reclaim_kv="$(grep_count 'phase_n_native_block_reclaim.*pool=1.*success=1' "$SYNTHETIC_ROOT")"
native_reclaim_weight="$(grep_count 'phase_n_native_block_reclaim.*pool=2.*success=1' "$SYNTHETIC_ROOT")"
native_fastpath_kv="$(grep_count 'phase_n_reclaim_pool native_block=1.*pool=1' "$SYNTHETIC_ROOT")"
native_fastpath_weight="$(grep_count 'phase_n_reclaim_pool native_block=1.*pool=2' "$SYNTHETIC_ROOT")"
mixed_total="$(grep_count 'phase_n_native_range_tag_mixed|phase_n_block_mixed_pool' "$SYNTHETIC_ROOT")"
split_total="$(grep_count 'phase_n_native_range_tag_split' "$SYNTHETIC_ROOT")"

{
  echo "Stage N P8f Synthetic Native KV/WEIGHT Summary"
  echo "=============================================="
  echo "run_root=$RUN_ROOT"
  echo "synthetic_root=$SYNTHETIC_ROOT"
  echo "synthetic_status=$synthetic_status"
  echo "strict=$STRICT"
  echo "block_bytes=$BLOCK_BYTES"
  echo "blocks=$BLOCKS"
  echo "budget_blocks=$BUDGET_BLOCKS"
  echo "pools=$POOLS"
  echo "shell_log=$SHELL_LOG"
  echo
  echo "Native Evidence"
  echo "---------------"
  echo "native_range_tag_attach_kv=$native_attach_kv"
  echo "native_range_tag_attach_weight=$native_attach_weight"
  echo "native_block_resident_kv=$native_resident_kv"
  echo "native_block_resident_weight=$native_resident_weight"
  echo "native_block_reclaim_success_kv=$native_reclaim_kv"
  echo "native_block_reclaim_success_weight=$native_reclaim_weight"
  echo "native_reclaim_fastpath_kv=$native_fastpath_kv"
  echo "native_reclaim_fastpath_weight=$native_fastpath_weight"
  echo "mixed_pool_records=$mixed_total"
  echo "range_split_records=$split_total"
  echo
  echo "Underlying P8d Check Summary"
  echo "----------------------------"
  if [ -f "$SYNTHETIC_ROOT/stage_n_p8d_block_quota_summary.txt" ]; then
    grep -E 'run_status=|check_status=|block_quota_pass=|block_quota_violations=|pool_kv=|pool_weights=' \
      "$SYNTHETIC_ROOT/stage_n_p8d_block_quota_summary.txt" || true
  else
    echo "missing_underlying_summary"
  fi
  echo
  echo "Recent Native Evidence"
  echo "----------------------"
  grep_tail 'phase_n_native_range_|phase_n_native_block_|phase_n_reclaim_pool native_block=1' "$SYNTHETIC_ROOT"
  echo
  echo "Recent Shell Output"
  echo "-------------------"
  tail -n 120 "$SHELL_LOG" 2>/dev/null || true
} > "$SUMMARY"

cat "$SUMMARY"

if [ "$STRICT" = "1" ]; then
  exit "$synthetic_status"
fi
exit 0
