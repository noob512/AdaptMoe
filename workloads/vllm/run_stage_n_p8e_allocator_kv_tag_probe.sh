#!/usr/bin/env bash
# Stage N P8e-a allocator-side early KV tag probe.
#
# This keeps P8e-a behind VLLM_UVM_DRIVER_KV_TAG_ENABLE, but makes the
# validation path opt in by default so the run can prove whether early KV tags
# repair KV block materialization/accounting.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8e_allocator_kv_tag_$(date -u +%Y%m%d_%H%M%S)}"
PROBE_ROOT="$RUN_ROOT/vllm_probe"
SUMMARY="$RUN_ROOT/stage_n_p8e_allocator_kv_tag_summary.txt"
SHELL_LOG="$RUN_ROOT/shell.log"
BASELINE_RUN_ROOT="${BASELINE_RUN_ROOT:-}"

PROFILE="${PROFILE:-kv}"
VLLM_UVM_DRIVER_KV_TAG_ENABLE="${VLLM_UVM_DRIVER_KV_TAG_ENABLE:-1}"
WITH_ADDRESS_LOG="${WITH_ADDRESS_LOG:-1}"
PROMPTS="${PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-0}"
REQUIRE_BLOCK_FASTPATH="${REQUIRE_BLOCK_FASTPATH:-0}"
SCRATCH_TEMP_BUDGET_BYTES="${SCRATCH_TEMP_BUDGET_BYTES:-1073741824}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-0}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-0}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-256}"

mkdir -p "$RUN_ROOT"

extract_metric() {
  local file="$1"
  local key="$2"
  awk -F= -v metric="$key" '
    $1 == metric { value = $2 }
    END {
      if (value == "") {
        print 0
      } else {
        print value + 0
      }
    }
  ' "$file" 2>/dev/null || printf '0\n'
}

extract_pool_kv_field() {
  local file="$1"
  local field="$2"
  awk -v field="$field" '
    /^pool_kv=/ {
      split($0, parts, ",")
      for (i in parts) {
        split(parts[i], kv, ":")
        gsub(/^pool_kv=/, "", kv[1])
        if (kv[1] == field) {
          print kv[2] + 0
          found = 1
        }
      }
    }
    END {
      if (!found) print 0
    }
  ' "$file" 2>/dev/null || printf '0\n'
}

echo "Stage N P8e-a allocator KV tag probe started."
echo "RUN_ROOT=$RUN_ROOT"
echo "PROBE_ROOT=$PROBE_ROOT"
echo "PROFILE=$PROFILE"
echo "VLLM_UVM_DRIVER_KV_TAG_ENABLE=$VLLM_UVM_DRIVER_KV_TAG_ENABLE"
echo "SCRATCH_TEMP_BUDGET_BYTES=$SCRATCH_TEMP_BUDGET_BYTES"
echo "UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"

set +e
env \
  RUN_ROOT="$PROBE_ROOT" \
  PROFILE="$PROFILE" \
  WITH_ADDRESS_LOG="$WITH_ADDRESS_LOG" \
  PROMPTS="$PROMPTS" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  REQUIRE_RECLAIM="$REQUIRE_RECLAIM" \
  REQUIRE_BLOCK_FASTPATH="$REQUIRE_BLOCK_FASTPATH" \
  SCRATCH_TEMP_BUDGET_BYTES="$SCRATCH_TEMP_BUDGET_BYTES" \
  UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY" \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE" \
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY" \
  ALLOWED_POST_FAULT_SLACK_BLOCKS="$ALLOWED_POST_FAULT_SLACK_BLOCKS" \
  VLLM_UVM_DRIVER_KV_TAG_ENABLE="$VLLM_UVM_DRIVER_KV_TAG_ENABLE" \
  ./run_stage_n_p8d_vllm_block_quota_probe.sh \
  > "$SHELL_LOG" 2>&1
probe_status=$?
set -e

CHECK_LOG="$PROBE_ROOT/block_quota_check.txt"
baseline_check_log=""
baseline_no_tagged=""
current_no_tagged="$(extract_metric "$CHECK_LOG" "no_tagged_region_records")"
current_kv_tag_records="$(extract_metric "$CHECK_LOG" "allocator_kv_driver_tag_records")"
current_kv_tag_success="$(extract_metric "$CHECK_LOG" "allocator_kv_driver_tag_success_records")"
current_kv_materialize="$(extract_pool_kv_field "$CHECK_LOG" "materialize")"
current_kv_resident_blocks="$(extract_pool_kv_field "$CHECK_LOG" "max_resident_blocks")"

if [ -n "$BASELINE_RUN_ROOT" ]; then
  baseline_check_log="$RUN_ROOT/baseline_block_quota_check.txt"
  python3 "$SCRIPT_DIR/check_stage_n_block_quota.py" \
    "$BASELINE_RUN_ROOT" > "$baseline_check_log" 2>&1 || true
  baseline_no_tagged="$(extract_metric "$baseline_check_log" "no_tagged_region_records")"
fi

{
  echo "Stage N P8e-a Allocator KV Tag Summary"
  echo "======================================"
  echo "run_root=$RUN_ROOT"
  echo "probe_root=$PROBE_ROOT"
  echo "profile=$PROFILE"
  echo "probe_status=$probe_status"
  echo "shell_log=$SHELL_LOG"
  echo "check_log=$CHECK_LOG"
  echo "baseline_run_root=${BASELINE_RUN_ROOT:-none}"
  echo "baseline_check_log=${baseline_check_log:-none}"
  echo
  echo "P8e-a Evidence"
  echo "--------------"
  echo "allocator_kv_driver_tag_records=$current_kv_tag_records"
  echo "allocator_kv_driver_tag_success_records=$current_kv_tag_success"
  echo "pool_kv_materialize_records=$current_kv_materialize"
  echo "pool_kv_max_resident_blocks=$current_kv_resident_blocks"
  echo "no_tagged_region_records=$current_no_tagged"
  if [ -n "$baseline_no_tagged" ]; then
    echo "baseline_no_tagged_region_records=$baseline_no_tagged"
    if [ "$current_no_tagged" -lt "$baseline_no_tagged" ]; then
      echo "no_tagged_region_decreased=1"
    else
      echo "no_tagged_region_decreased=0"
    fi
  fi
  echo
  echo "Checker Output"
  echo "--------------"
  if [ -f "$CHECK_LOG" ]; then
    cat "$CHECK_LOG"
  else
    echo "missing_check_log"
  fi
  echo
  echo "Recent Shell Output"
  echo "-------------------"
  tail -n 160 "$SHELL_LOG" 2>/dev/null || true
} > "$SUMMARY"

cat "$SUMMARY"
exit "$probe_status"
