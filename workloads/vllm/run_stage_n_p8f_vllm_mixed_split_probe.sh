#!/usr/bin/env bash
# Stage N P8f vLLM native range split/mixed probe.
#
# This run answers whether real vLLM registration creates native range splits,
# mixed-pool tags, or partial-range skips after REGISTER_REGION attaches tags
# directly to managed va_ranges.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8f_vllm_mixed_split_$(date -u +%Y%m%d_%H%M%S)}"
PROBE_ROOT="$RUN_ROOT/vllm_probe"
SUMMARY="$RUN_ROOT/stage_n_p8f_vllm_mixed_split_summary.txt"
SHELL_LOG="$RUN_ROOT/shell.log"

PROFILE="${PROFILE:-both}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
WITH_ADDRESS_LOG="${WITH_ADDRESS_LOG:-1}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-0}"
REQUIRE_BLOCK_FASTPATH="${REQUIRE_BLOCK_FASTPATH:-0}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-512}"
STRICT="${STRICT:-0}"

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
  grep -R -h -E "$regex" "$root" 2>/dev/null | tail -n 80
  set -e
}

echo "Stage N P8f vLLM mixed/split probe started."
echo "RUN_ROOT=$RUN_ROOT"
echo "PROFILE=$PROFILE"

set +e
env \
  RUN_ROOT="$PROBE_ROOT" \
  PROFILE="$PROFILE" \
  PROMPTS="$PROMPTS" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  WITH_ADDRESS_LOG="$WITH_ADDRESS_LOG" \
  REQUIRE_RECLAIM="$REQUIRE_RECLAIM" \
  REQUIRE_BLOCK_FASTPATH="$REQUIRE_BLOCK_FASTPATH" \
  ALLOWED_POST_FAULT_SLACK_BLOCKS="$ALLOWED_POST_FAULT_SLACK_BLOCKS" \
  UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
  UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
  VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
  ./run_stage_n_p8d_vllm_block_quota_probe.sh \
  > "$SHELL_LOG" 2>&1
probe_status=$?
set -e

native_attach_total="$(grep_count 'phase_n_native_range_tag_attach' "$PROBE_ROOT")"
native_attach_kv="$(grep_count 'phase_n_native_range_tag_attach.*pool=1' "$PROBE_ROOT")"
native_attach_weight="$(grep_count 'phase_n_native_range_tag_attach.*pool=2' "$PROBE_ROOT")"
native_split_total="$(grep_count 'phase_n_native_range_tag_split' "$PROBE_ROOT")"
native_split_base="$(grep_count 'phase_n_native_range_tag_split.*split=base' "$PROBE_ROOT")"
native_split_end="$(grep_count 'phase_n_native_range_tag_split.*split=end' "$PROBE_ROOT")"
native_mixed_total="$(grep_count 'phase_n_native_range_tag_mixed|phase_n_block_mixed_pool' "$PROBE_ROOT")"
native_partial_skip="$(grep_count 'phase_n_native_range_tag_skip.*reason=partial_managed_range' "$PROBE_ROOT")"
native_block_init="$(grep_count 'phase_n_native_block_init' "$PROBE_ROOT")"
native_block_resident_kv="$(grep_count 'phase_n_native_block_resident.*pool=1' "$PROBE_ROOT")"
native_block_resident_weight="$(grep_count 'phase_n_native_block_resident.*pool=2' "$PROBE_ROOT")"
allocator_kv_tags="$(grep_count 'TRACE_DRIVER_KV_TAG.*registered=1' "$PROBE_ROOT")"

{
  echo "Stage N P8f vLLM Mixed/Split Probe Summary"
  echo "=========================================="
  echo "run_root=$RUN_ROOT"
  echo "probe_root=$PROBE_ROOT"
  echo "profile=$PROFILE"
  echo "prompts=$PROMPTS"
  echo "output_len=$OUTPUT_LEN"
  echo "probe_status=$probe_status"
  echo "strict=$STRICT"
  echo "shell_log=$SHELL_LOG"
  echo
  echo "Native Range Tag Evidence"
  echo "-------------------------"
  echo "native_range_tag_attach_total=$native_attach_total"
  echo "native_range_tag_attach_kv=$native_attach_kv"
  echo "native_range_tag_attach_weight=$native_attach_weight"
  echo "native_range_tag_split_total=$native_split_total"
  echo "native_range_tag_split_base=$native_split_base"
  echo "native_range_tag_split_end=$native_split_end"
  echo "native_range_tag_mixed_total=$native_mixed_total"
  echo "native_range_tag_partial_skip=$native_partial_skip"
  echo
  echo "Native Block Evidence"
  echo "---------------------"
  echo "native_block_init_records=$native_block_init"
  echo "native_block_resident_kv=$native_block_resident_kv"
  echo "native_block_resident_weight=$native_block_resident_weight"
  echo "allocator_kv_driver_tag_success_records=$allocator_kv_tags"
  echo
  echo "Interpretation"
  echo "--------------"
  if [ "$native_mixed_total" -gt 0 ]; then
    echo "mixed_pool_observed=1"
  else
    echo "mixed_pool_observed=0"
  fi
  if [ "$native_split_total" -gt 0 ]; then
    echo "range_split_observed=1"
  else
    echo "range_split_observed=0"
  fi
  if [ "$native_partial_skip" -gt 0 ]; then
    echo "partial_managed_range_skip_observed=1"
  else
    echo "partial_managed_range_skip_observed=0"
  fi
  echo
  echo "Recent Native Range Lines"
  echo "-------------------------"
  grep_tail 'phase_n_native_range_tag_(attach|split|mixed|skip|summary)' "$PROBE_ROOT"
  echo
  echo "Recent Native Block Lines"
  echo "-------------------------"
  grep_tail 'phase_n_native_block_(init|resident|dequeue|reclaim)' "$PROBE_ROOT"
  echo
  echo "Recent Shell Output"
  echo "-------------------"
  tail -n 120 "$SHELL_LOG" 2>/dev/null || true
} > "$SUMMARY"

cat "$SUMMARY"

if [ "$STRICT" = "1" ]; then
  exit "$probe_status"
fi
exit 0
