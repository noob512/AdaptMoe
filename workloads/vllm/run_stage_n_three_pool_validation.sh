#!/usr/bin/env bash
# Stage N three-pool validation bundle.
#
# Runs:
#   1. vLLM tag-only three-pool smoke with structured scratch/pool-state summary.
#   2. Manual Phase N reclaim smoke for kv, weights, and scratch pools.
#
# The reclaim smoke is synthetic/manual. Set REQUIRE_PHASE_N_AUTO_HOOK=1 to
# require hook pool/budget traces, or REQUIRE_PHASE_N_AUTO_RECLAIM=1 to require
# executable automatic soft reclaim evidence from the replayable-fault outer hook.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_three_pool_validation_$(date -u +%Y%m%d_%H%M%S)}"
TAG_ONLY_RUN_ROOT="${TAG_ONLY_RUN_ROOT:-$RUN_ROOT/tag_only}"
RECLAIM_RUN_ROOT="${RECLAIM_RUN_ROOT:-$RUN_ROOT/phase_n_reclaim}"
SUMMARY="$RUN_ROOT/stage_n_three_pool_validation_summary.txt"
COMMAND_FILE="$RUN_ROOT/command_stage_n_three_pool_validation.sh"

RESET_COUNTERS="${RESET_COUNTERS:-1}"
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
RECLAIM_BYTES="${RECLAIM_BYTES:-1048576}"
RECLAIM_ASYNC="${RECLAIM_ASYNC:-0}"
RECLAIM_DEVICE="${RECLAIM_DEVICE:-0}"
REQUIRE_PHASE_N_AUTO_HOOK="${REQUIRE_PHASE_N_AUTO_HOOK:-0}"
REQUIRE_PHASE_N_AUTO_RECLAIM="${REQUIRE_PHASE_N_AUTO_RECLAIM:-0}"
if [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" = "1" ]; then
  REQUIRE_PHASE_N_AUTO_HOOK=1
fi

mkdir -p "$RUN_ROOT" "$RECLAIM_RUN_ROOT"

cleanup_stale_stage_n_trace_pipe_readers() {
  local pids

  pids="$(pgrep -f 'trace_pipe.*log/stage_n_' 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    # Best-effort cleanup for interrupted prior validation runs. A live run in
    # this script owns trace_pipe exclusively, so stale readers would fail fast.
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup_stale_stage_n_trace_pipe_readers

{
  printf 'cd %q\n' "$SCRIPT_DIR"
  printf 'RESET_COUNTERS=%q UVM_PHASE_F_ROUTE_LOG_DELAY_MS=%q REQUIRE_PHASE_N_AUTO_HOOK=%q REQUIRE_PHASE_N_AUTO_RECLAIM=%q RUN_ROOT=%q ./run_stage_n_three_pool_validation.sh\n' \
    "$RESET_COUNTERS" "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" "$REQUIRE_PHASE_N_AUTO_HOOK" "$REQUIRE_PHASE_N_AUTO_RECLAIM" "$RUN_ROOT"
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Stage N three-pool validation started."
echo "RUN_ROOT=$RUN_ROOT"
echo "Summary: $SUMMARY"

set +e
RESET_COUNTERS="$RESET_COUNTERS" \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
RUN_ROOT="$TAG_ONLY_RUN_ROOT" \
DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY="${DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY:-1}" \
DRIVER_POOL_STATE_SNAPSHOT_DEVICE="${DRIVER_POOL_STATE_SNAPSHOT_DEVICE:-0}" \
REQUIRE_PHASE_N_AUTO_HOOK="$REQUIRE_PHASE_N_AUTO_HOOK" \
REQUIRE_PHASE_N_AUTO_RECLAIM="$REQUIRE_PHASE_N_AUTO_RECLAIM" \
./run_stage_n_tag_only_three_pool_smoke.sh > "$RUN_ROOT/tag_only_driver.log" 2>&1
tag_only_status=$?
set -e

pool_names=(kv weights scratch)
pool_ids=(1 2 3)
reclaim_statuses=()

for index in "${!pool_ids[@]}"; do
  pool_id="${pool_ids[$index]}"
  pool_name="${pool_names[$index]}"
  pool_root="$RECLAIM_RUN_ROOT/$pool_name"
  pool_log="$RECLAIM_RUN_ROOT/${pool_name}.log"
  mkdir -p "$pool_root"
  set +e
  VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
    ./run_stage_m_phase_f_smoke.sh \
      --no-enable-tests \
      --step phase-n-reclaim \
      --pool "$pool_id" \
      --bytes "$RECLAIM_BYTES" \
      --device "$RECLAIM_DEVICE" \
      --async "$RECLAIM_ASYNC" \
      --run-root "$pool_root" \
      --log-file "$pool_log" > "$RECLAIM_RUN_ROOT/${pool_name}_driver.log" 2>&1
  reclaim_statuses+=("$?")
  set -e
done

extract_summary_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="$key" '$1 == key {value=$2} END {print value}' "$file" 2>/dev/null
}

extract_reclaim_value() {
  local pool="$1"
  local prefix="$2"
  local key="$3"
  local file="$RECLAIM_RUN_ROOT/${pool}.log"
  awk -v prefix="$prefix" -v key="$key" '
    index($0, prefix) == 1 {
      for (i = 1; i <= NF; ++i) {
        split($i, parts, "=")
        if (parts[1] == key) {
          value = parts[2]
        }
      }
    }
    END {print value}
  ' "$file" 2>/dev/null
}

parse_reclaim_logs() {
  python3 - "$RECLAIM_RUN_ROOT" <<'PY'
import os
import re
import sys

root = sys.argv[1]
pool_names = ("kv", "weights", "scratch")
line_pattern = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")

def parse_fields(line: str) -> dict[str, str]:
    return {key: value for key, value in line_pattern.findall(line)}

for pool in pool_names:
    path = os.path.join(root, f"{pool}.log")
    reclaim = {}
    post = {}
    victims = ""
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("phase_n_reclaim "):
                    reclaim = parse_fields(line)
                elif line.startswith("phase_n_reclaim_victims:"):
                    victims = line.strip().split(":", 1)[1].strip()
                elif line.startswith("phase_n_reclaim_post_state "):
                    post = parse_fields(line)
    print(
        f"phase_n_reclaim_{pool}="
        f"attempted:{reclaim.get('attempted', 'missing')},"
        f"success:{reclaim.get('success', 'missing')},"
        f"failure:{reclaim.get('failure', 'missing')},"
        f"victim_count:{reclaim.get('victim_count', 'missing')},"
        f"candidate_bytes:{reclaim.get('candidate_bytes', 'missing')},"
        f"reclaimed:{reclaim.get('reclaimed', 'missing')},"
        f"post_resident_gpu:{post.get('resident_gpu', 'missing')},"
        f"post_candidate_bytes:{post.get('candidate_bytes', 'missing')},"
        f"post_selected_id:{post.get('selected_id', 'missing')},"
        f"post_selected_bytes:{post.get('selected_bytes', 'missing')},"
        f"post_victim_count:{post.get('victim_count', 'missing')},"
        f"victims:{victims or 'missing'}"
    )
PY
}

tag_summary="$TAG_ONLY_RUN_ROOT/stage_n_acceptance_summary.txt"
scratch_driver_tag_records="$(extract_summary_value scratch_driver_tag_records "$tag_summary")"
driver_pool_state_records="$(extract_summary_value driver_pool_state_records "$tag_summary")"
driver_pool_state_failures="$(extract_summary_value driver_pool_state_failures "$tag_summary")"
driver_pool_state_success_records="$(extract_summary_value driver_pool_state_success_records "$tag_summary")"
driver_pool_state_all_pools_present="$(extract_summary_value driver_pool_state_all_pools_present "$tag_summary")"
driver_pool_state_all_budgets_positive="$(extract_summary_value driver_pool_state_all_budgets_positive "$tag_summary")"
driver_pool_state_all_required_fields_present="$(extract_summary_value driver_pool_state_all_required_fields_present "$tag_summary")"
allocator_driver_pool_state_records="$(extract_summary_value allocator_driver_pool_state_records "$tag_summary")"
allocator_driver_pool_state_success_records="$(extract_summary_value allocator_driver_pool_state_success_records "$tag_summary")"
allocator_driver_pool_state_all_pools_present="$(extract_summary_value allocator_driver_pool_state_all_pools_present "$tag_summary")"
allocator_driver_pool_state_all_status_ok="$(extract_summary_value allocator_driver_pool_state_all_status_ok "$tag_summary")"
allocator_driver_pool_state_all_budgets_positive="$(extract_summary_value allocator_driver_pool_state_all_budgets_positive "$tag_summary")"
allocator_driver_pool_state_all_required_fields_present="$(extract_summary_value allocator_driver_pool_state_all_required_fields_present "$tag_summary")"
phase_n_auto_hook_records="$(extract_summary_value phase_n_auto_hook_records "$tag_summary")"
phase_n_auto_hook_would_reclaim_records="$(extract_summary_value phase_n_auto_hook_would_reclaim_records "$tag_summary")"
phase_n_auto_hook_locked_context_records="$(extract_summary_value phase_n_auto_hook_locked_context_records "$tag_summary")"
phase_n_auto_hook_outer_records="$(extract_summary_value phase_n_auto_hook_outer_records "$tag_summary")"
phase_n_auto_hook_pool_records="$(extract_summary_value phase_n_auto_hook_pool_records "$tag_summary")"
phase_n_auto_hook_no_region_records="$(extract_summary_value phase_n_auto_hook_no_region_records "$tag_summary")"
phase_n_auto_hook_execute_records="$(extract_summary_value phase_n_auto_hook_execute_records "$tag_summary")"
phase_n_auto_hook_success_records="$(extract_summary_value phase_n_auto_hook_success_records "$tag_summary")"
phase_n_auto_hook_reclaimed_records="$(extract_summary_value phase_n_auto_hook_reclaimed_records "$tag_summary")"
uvm_phase_n_auto_reclaim_every="$(extract_summary_value uvm_phase_n_auto_reclaim_every "$tag_summary")"
uvm_phase_n_auto_reclaim_scan_every="$(extract_summary_value uvm_phase_n_auto_reclaim_scan_every "$tag_summary")"
uvm_phase_n_auto_reclaim_cooldown_epochs="$(extract_summary_value uvm_phase_n_auto_reclaim_cooldown_epochs "$tag_summary")"
failed_requests="$(extract_summary_value failed_requests "$tag_summary")"
tag_verdict="$(grep -E '^PASS:' "$tag_summary" 2>/dev/null | tail -n 1 || true)"

kv_reclaim_status="${reclaim_statuses[0]:-99}"
weights_reclaim_status="${reclaim_statuses[1]:-99}"
scratch_reclaim_status="${reclaim_statuses[2]:-99}"
kv_reclaim_victim_count="$(extract_reclaim_value kv 'phase_n_reclaim ' victim_count)"
weights_reclaim_victim_count="$(extract_reclaim_value weights 'phase_n_reclaim ' victim_count)"
scratch_reclaim_victim_count="$(extract_reclaim_value scratch 'phase_n_reclaim ' victim_count)"
kv_reclaim_candidate_bytes="$(extract_reclaim_value kv 'phase_n_reclaim ' candidate_bytes)"
weights_reclaim_candidate_bytes="$(extract_reclaim_value weights 'phase_n_reclaim ' candidate_bytes)"
scratch_reclaim_candidate_bytes="$(extract_reclaim_value scratch 'phase_n_reclaim ' candidate_bytes)"

{
  echo "Stage N Three-Pool Validation Summary"
  echo "====================================="
  echo "run_root=$RUN_ROOT"
  echo "command_file=$COMMAND_FILE"
  echo "tag_only_run_root=$TAG_ONLY_RUN_ROOT"
  echo "tag_only_summary=$tag_summary"
  echo "tag_only_status=$tag_only_status"
  echo "phase_n_reclaim_run_root=$RECLAIM_RUN_ROOT"
  echo "phase_n_reclaim_kv_status=$kv_reclaim_status"
  echo "phase_n_reclaim_weights_status=$weights_reclaim_status"
  echo "phase_n_reclaim_scratch_status=$scratch_reclaim_status"
  echo "phase_n_reclaim_kv_victim_count=${kv_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_weights_victim_count=${weights_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_scratch_victim_count=${scratch_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_kv_candidate_bytes=${kv_reclaim_candidate_bytes:-0}"
  echo "phase_n_reclaim_weights_candidate_bytes=${weights_reclaim_candidate_bytes:-0}"
  echo "phase_n_reclaim_scratch_candidate_bytes=${scratch_reclaim_candidate_bytes:-0}"
  echo "require_phase_n_auto_hook=$REQUIRE_PHASE_N_AUTO_HOOK"
  echo "require_phase_n_auto_reclaim=$REQUIRE_PHASE_N_AUTO_RECLAIM"
  echo "phase_n_auto_hook_records=${phase_n_auto_hook_records:-0}"
  echo "phase_n_auto_hook_would_reclaim_records=${phase_n_auto_hook_would_reclaim_records:-0}"
  echo "phase_n_auto_hook_locked_context_records=${phase_n_auto_hook_locked_context_records:-0}"
  echo "phase_n_auto_hook_outer_records=${phase_n_auto_hook_outer_records:-0}"
  echo "phase_n_auto_hook_pool_records=${phase_n_auto_hook_pool_records:-0}"
  echo "phase_n_auto_hook_no_region_records=${phase_n_auto_hook_no_region_records:-0}"
  echo "phase_n_auto_hook_execute_records=${phase_n_auto_hook_execute_records:-0}"
  echo "phase_n_auto_hook_success_records=${phase_n_auto_hook_success_records:-0}"
  echo "phase_n_auto_hook_reclaimed_records=${phase_n_auto_hook_reclaimed_records:-0}"
  echo "uvm_phase_n_auto_reclaim_every=${uvm_phase_n_auto_reclaim_every:-unknown}"
  echo "uvm_phase_n_auto_reclaim_scan_every=${uvm_phase_n_auto_reclaim_scan_every:-unknown}"
  echo "uvm_phase_n_auto_reclaim_cooldown_epochs=${uvm_phase_n_auto_reclaim_cooldown_epochs:-unknown}"
  echo
  echo "Tag-Only Required Evidence"
  echo "--------------------------"
  echo "scratch_driver_tag_records=${scratch_driver_tag_records:-0}"
  echo "driver_pool_state_records=${driver_pool_state_records:-0}"
  echo "driver_pool_state_failures=${driver_pool_state_failures:-0}"
  echo "driver_pool_state_success_records=${driver_pool_state_success_records:-0}"
  echo "driver_pool_state_all_pools_present=${driver_pool_state_all_pools_present:-0}"
  echo "driver_pool_state_all_budgets_positive=${driver_pool_state_all_budgets_positive:-0}"
  echo "driver_pool_state_all_required_fields_present=${driver_pool_state_all_required_fields_present:-0}"
  echo "allocator_driver_pool_state_records=${allocator_driver_pool_state_records:-0}"
  echo "allocator_driver_pool_state_success_records=${allocator_driver_pool_state_success_records:-0}"
  echo "allocator_driver_pool_state_all_pools_present=${allocator_driver_pool_state_all_pools_present:-0}"
  echo "allocator_driver_pool_state_all_status_ok=${allocator_driver_pool_state_all_status_ok:-0}"
  echo "allocator_driver_pool_state_all_budgets_positive=${allocator_driver_pool_state_all_budgets_positive:-0}"
  echo "allocator_driver_pool_state_all_required_fields_present=${allocator_driver_pool_state_all_required_fields_present:-0}"
  echo "failed_requests=${failed_requests:-unknown}"
  echo "tag_only_verdict=${tag_verdict:-missing}"
  echo
  echo "Auto Hook Evidence"
  echo "------------------"
  echo "require_phase_n_auto_hook=$REQUIRE_PHASE_N_AUTO_HOOK"
  echo "require_phase_n_auto_reclaim=$REQUIRE_PHASE_N_AUTO_RECLAIM"
  echo "phase_n_auto_hook_records=${phase_n_auto_hook_records:-0}"
  echo "phase_n_auto_hook_would_reclaim_records=${phase_n_auto_hook_would_reclaim_records:-0}"
  echo "phase_n_auto_hook_locked_context_records=${phase_n_auto_hook_locked_context_records:-0}"
  echo "phase_n_auto_hook_outer_records=${phase_n_auto_hook_outer_records:-0}"
  echo "phase_n_auto_hook_pool_records=${phase_n_auto_hook_pool_records:-0}"
  echo "phase_n_auto_hook_no_region_records=${phase_n_auto_hook_no_region_records:-0}"
  echo "phase_n_auto_hook_execute_records=${phase_n_auto_hook_execute_records:-0}"
  echo "phase_n_auto_hook_success_records=${phase_n_auto_hook_success_records:-0}"
  echo "phase_n_auto_hook_reclaimed_records=${phase_n_auto_hook_reclaimed_records:-0}"
  echo "uvm_phase_n_auto_reclaim_every=${uvm_phase_n_auto_reclaim_every:-unknown}"
  echo "uvm_phase_n_auto_reclaim_scan_every=${uvm_phase_n_auto_reclaim_scan_every:-unknown}"
  echo "uvm_phase_n_auto_reclaim_cooldown_epochs=${uvm_phase_n_auto_reclaim_cooldown_epochs:-unknown}"
  echo
  if [ -s "$tag_summary" ]; then
	    awk '
	      /^Scratch\/Temp Driver Trace Summary$/ {show=1}
	      /^Driver Pool State Summary$/ {show=1}
	      /^Allocator Driver Pool State Snapshot Summary$/ {show=1}
	      /^Residency Driver Budget\/Tag Evidence$/ {show=0}
	      show {print}
	    ' "$tag_summary"
  else
    echo "missing tag-only summary: $tag_summary"
  fi
  echo
  echo "Manual Phase-N Reclaim Evidence"
  echo "-------------------------------"
  echo "phase_n_reclaim_kv_victim_count=${kv_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_weights_victim_count=${weights_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_scratch_victim_count=${scratch_reclaim_victim_count:-0}"
  echo "phase_n_reclaim_kv_candidate_bytes=${kv_reclaim_candidate_bytes:-0}"
  echo "phase_n_reclaim_weights_candidate_bytes=${weights_reclaim_candidate_bytes:-0}"
  echo "phase_n_reclaim_scratch_candidate_bytes=${scratch_reclaim_candidate_bytes:-0}"
  parse_reclaim_logs
  echo
  echo "Manual Phase-N Reclaim Log Paths"
  echo "--------------------------------"
  for pool in "${pool_names[@]}"; do
    echo "$pool=$RECLAIM_RUN_ROOT/${pool}.log"
  done
  echo
  echo "Verdict"
  echo "-------"
  if [ "$tag_only_status" -eq 0 ] &&
     [ "${scratch_driver_tag_records:-0}" -gt 0 ] &&
     [ "${driver_pool_state_records:-0}" -gt 0 ] &&
     [ "${driver_pool_state_failures:-1}" -eq 0 ] &&
     [ "${driver_pool_state_success_records:-0}" -gt 0 ] &&
     [ "${driver_pool_state_all_pools_present:-0}" -eq 1 ] &&
     [ "${driver_pool_state_all_budgets_positive:-0}" -eq 1 ] &&
     [ "${driver_pool_state_all_required_fields_present:-0}" -eq 1 ] &&
     [ "${allocator_driver_pool_state_records:-0}" -gt 0 ] &&
     [ "${allocator_driver_pool_state_success_records:-0}" -gt 0 ] &&
     [ "${allocator_driver_pool_state_all_pools_present:-0}" -eq 1 ] &&
     [ "${allocator_driver_pool_state_all_status_ok:-0}" -eq 1 ] &&
     [ "${allocator_driver_pool_state_all_budgets_positive:-0}" -eq 1 ] &&
     [ "${allocator_driver_pool_state_all_required_fields_present:-0}" -eq 1 ] &&
     [ "${failed_requests:-1}" = "0" ] &&
     { [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "${phase_n_auto_hook_records:-0}" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "${phase_n_auto_hook_pool_records:-0}" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_would_reclaim_records:-0}" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_execute_records:-0}" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_success_records:-0}" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_reclaimed_records:-0}" -gt 0 ]; } &&
     [ "$kv_reclaim_status" -eq 0 ] &&
     [ "$weights_reclaim_status" -eq 0 ] &&
     [ "$scratch_reclaim_status" -eq 0 ] &&
     [ "${kv_reclaim_victim_count:-0}" -gt 0 ] &&
     [ "${weights_reclaim_victim_count:-0}" -gt 0 ] &&
     [ "${scratch_reclaim_victim_count:-0}" -gt 0 ]; then
    echo "PASS: tag-only vLLM evidence plus automatic hook/reclaim and manual phase-n-reclaim validation succeeded for kv, weights, and scratch."
  else
    echo "WARN: Stage N three-pool validation did not satisfy all checks."
    [ "$tag_only_status" -eq 0 ] || echo "- tag-only smoke exited with status $tag_only_status"
    [ "${scratch_driver_tag_records:-0}" -gt 0 ] || echo "- scratch_driver_tag_records is not > 0"
    [ "${driver_pool_state_records:-0}" -gt 0 ] || echo "- driver_pool_state_records is not > 0"
    [ "${driver_pool_state_failures:-1}" -eq 0 ] || echo "- driver_pool_state_failures=${driver_pool_state_failures:-unknown}"
    [ "${driver_pool_state_success_records:-0}" -gt 0 ] || echo "- no successful residency_driver_pool_state snapshot contained kv, weights, and scratch"
    [ "${driver_pool_state_all_pools_present:-0}" -eq 1 ] || echo "- residency_driver_pool_state is missing kv, weights, or scratch"
    [ "${driver_pool_state_all_budgets_positive:-0}" -eq 1 ] || echo "- residency_driver_pool_state did not report positive budgets for kv, weights, and scratch"
    [ "${driver_pool_state_all_required_fields_present:-0}" -eq 1 ] || echo "- residency_driver_pool_state is missing required budget/resident/candidate/victim fields"
    [ "${allocator_driver_pool_state_records:-0}" -gt 0 ] || echo "- allocator_driver_pool_state_records is not > 0"
    [ "${allocator_driver_pool_state_success_records:-0}" -gt 0 ] || echo "- no allocator pool-state snapshot returned status=1"
    [ "${allocator_driver_pool_state_all_pools_present:-0}" -eq 1 ] || echo "- allocator pool-state snapshot is missing kv, weights, or scratch"
    [ "${allocator_driver_pool_state_all_status_ok:-0}" -eq 1 ] || echo "- latest allocator pool-state snapshot did not return status=1 for kv, weights, and scratch"
    [ "${allocator_driver_pool_state_all_budgets_positive:-0}" -eq 1 ] || echo "- allocator pool-state snapshot did not report positive budgets for kv, weights, and scratch"
    [ "${allocator_driver_pool_state_all_required_fields_present:-0}" -eq 1 ] || echo "- allocator pool-state snapshot is missing required budget/resident/candidate/victim fields"
    [ "${failed_requests:-1}" = "0" ] || echo "- failed_requests=${failed_requests:-unknown}"
    [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "${phase_n_auto_hook_records:-0}" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook records"
    [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "${phase_n_auto_hook_pool_records:-0}" -gt 0 ] || echo "- required phase_n_auto_reclaim_hook did not enter pool/budget branch"
    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_would_reclaim_records:-0}" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook would_reclaim=1 records"
    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_execute_records:-0}" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook execute=1 records"
    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_success_records:-0}" -gt 0 ] || echo "- missing successful automatic reclaim records"
    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "${phase_n_auto_hook_reclaimed_records:-0}" -gt 0 ] || echo "- automatic reclaim did not report reclaimed bytes"
    [ "$kv_reclaim_status" -eq 0 ] || echo "- kv phase-n-reclaim status=$kv_reclaim_status"
    [ "$weights_reclaim_status" -eq 0 ] || echo "- weights phase-n-reclaim status=$weights_reclaim_status"
    [ "$scratch_reclaim_status" -eq 0 ] || echo "- scratch phase-n-reclaim status=$scratch_reclaim_status"
    [ "${kv_reclaim_victim_count:-0}" -gt 0 ] || echo "- kv phase-n-reclaim victim_count=${kv_reclaim_victim_count:-0}"
    [ "${weights_reclaim_victim_count:-0}" -gt 0 ] || echo "- weights phase-n-reclaim victim_count=${weights_reclaim_victim_count:-0}"
    [ "${scratch_reclaim_victim_count:-0}" -gt 0 ] || echo "- scratch phase-n-reclaim victim_count=${scratch_reclaim_victim_count:-0}"
  fi
} > "$SUMMARY"

cat "$SUMMARY"

if grep -q '^PASS:' "$SUMMARY"; then
  exit 0
fi
exit 1
