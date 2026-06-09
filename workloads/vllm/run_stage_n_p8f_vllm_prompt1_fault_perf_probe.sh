#!/usr/bin/env bash
# Stage N P8f prompt=1 vLLM native fault/performance probe.
#
# This is a small real-vLLM run intended to verify whether KV and WEIGHT native
# block events are visible while also preserving benchmark latency/throughput
# metrics in the summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8f_vllm_prompt1_fault_perf_$(date -u +%Y%m%d_%H%M%S)}"
PROBE_ROOT="$RUN_ROOT/vllm_probe"
SUMMARY="$RUN_ROOT/stage_n_p8f_vllm_prompt1_fault_perf_summary.txt"
SHELL_LOG="$RUN_ROOT/shell.log"

PROFILE="${PROFILE:-both}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
REQUEST_RATE="${REQUEST_RATE:-1}"
WITH_ADDRESS_LOG="${WITH_ADDRESS_LOG:-1}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-0}"
REQUIRE_BLOCK_FASTPATH="${REQUIRE_BLOCK_FASTPATH:-0}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-512}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-0}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-0}"
STRICT="${STRICT:-1}"

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

echo "Stage N P8f vLLM prompt=1 fault/perf probe started."
echo "RUN_ROOT=$RUN_ROOT"
echo "PROFILE=$PROFILE"

set +e
env \
  RUN_ROOT="$PROBE_ROOT" \
  PROFILE="$PROFILE" \
  PROMPTS="$PROMPTS" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  REQUEST_RATE="$REQUEST_RATE" \
  WITH_ADDRESS_LOG="$WITH_ADDRESS_LOG" \
  REQUIRE_RECLAIM="$REQUIRE_RECLAIM" \
  REQUIRE_BLOCK_FASTPATH="$REQUIRE_BLOCK_FASTPATH" \
  ALLOWED_POST_FAULT_SLACK_BLOCKS="$ALLOWED_POST_FAULT_SLACK_BLOCKS" \
  UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
  UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY" \
  UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
  UVM_PHASE_N_AUTO_RECLAIM_ENABLE=1 \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE" \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY" \
  VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
  ./run_stage_n_p8d_vllm_block_quota_probe.sh \
  > "$SHELL_LOG" 2>&1
probe_status=$?
set -e

SMOKE_ROOT="$PROBE_ROOT/vllm_smoke"
BENCH_LOG="$SMOKE_ROOT/vllm_bench.log"
CHECK_LOG="$PROBE_ROOT/block_quota_check.txt"

native_resident_kv="$(grep_count 'phase_n_native_block_resident.*pool=1' "$PROBE_ROOT")"
native_resident_weight="$(grep_count 'phase_n_native_block_resident.*pool=2' "$PROBE_ROOT")"
native_attach_kv="$(grep_count 'phase_n_native_range_tag_attach.*pool=1' "$PROBE_ROOT")"
native_attach_weight="$(grep_count 'phase_n_native_range_tag_attach.*pool=2' "$PROBE_ROOT")"
auto_hook_kv="$(grep_count 'phase_n_block_auto_hook.*pool=1' "$PROBE_ROOT")"
auto_hook_weight="$(grep_count 'phase_n_block_auto_hook.*pool=2' "$PROBE_ROOT")"
no_tagged_region="$(grep_count 'skip=no_tagged_region' "$PROBE_ROOT")"
allocator_kv_tags="$(grep_count 'TRACE_DRIVER_KV_TAG.*registered=1' "$PROBE_ROOT")"

{
  echo "Stage N P8f vLLM Prompt=1 Fault/Perf Summary"
  echo "============================================"
  echo "run_root=$RUN_ROOT"
  echo "probe_root=$PROBE_ROOT"
  echo "smoke_root=$SMOKE_ROOT"
  echo "profile=$PROFILE"
  echo "prompts=$PROMPTS"
  echo "output_len=$OUTPUT_LEN"
  echo "request_rate=$REQUEST_RATE"
  echo "probe_status=$probe_status"
  echo "strict=$STRICT"
  echo "shell_log=$SHELL_LOG"
  echo "bench_log=$BENCH_LOG"
  echo "check_log=$CHECK_LOG"
  echo
  echo "Native KV/WEIGHT Capture Evidence"
  echo "---------------------------------"
  echo "native_range_tag_attach_kv=$native_attach_kv"
  echo "native_range_tag_attach_weight=$native_attach_weight"
  echo "native_block_resident_kv=$native_resident_kv"
  echo "native_block_resident_weight=$native_resident_weight"
  echo "auto_hook_pool_kv=$auto_hook_kv"
  echo "auto_hook_pool_weight=$auto_hook_weight"
  echo "allocator_kv_driver_tag_success_records=$allocator_kv_tags"
  echo "no_tagged_region_records=$no_tagged_region"
  if [ "$native_resident_kv" -gt 0 ] && [ "$native_resident_weight" -gt 0 ]; then
    echo "kv_and_weight_native_capture=1"
  else
    echo "kv_and_weight_native_capture=0"
  fi
  echo
  echo "Benchmark"
  echo "---------"
  grep -E 'Successful requests:|Failed requests:|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P99 TPOT|Mean ITL|P99 ITL|Benchmark duration' \
    "$BENCH_LOG" 2>/dev/null || true
  echo
  echo "Checker Output"
  echo "--------------"
  if [ -f "$CHECK_LOG" ]; then
    cat "$CHECK_LOG"
  else
    echo "missing_check_log"
  fi
  echo
  echo "Recent Native/Fault Evidence"
  echo "----------------------------"
  grep_tail 'phase_n_native_range_|phase_n_native_block_|phase_n_block_auto_hook|phase_n_block_over_budget|skip=no_tagged_region' "$PROBE_ROOT"
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
