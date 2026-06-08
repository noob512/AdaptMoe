#!/usr/bin/env bash
# Stage N P8d synthetic block-quota proof.
#
# This does not start vLLM. It runs phase_f_smoke in-process with managed
# memory, registers one tagged range per pool, touches blocks on GPU with
# cudaMemset, then checks P8c/P8d route-log evidence.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/stage_n_p8d_synthetic_block_quota_$(date -u +%Y%m%d_%H%M%S)}"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEVICE="${DEVICE:-0}"
BLOCK_BYTES="${BLOCK_BYTES:-2097152}"
BLOCKS="${BLOCKS:-4}"
BUDGET_BLOCKS="${BUDGET_BLOCKS:-2}"
POOLS="${POOLS:-1 2}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
WRITE_UVM_PARAMS="${WRITE_UVM_PARAMS:-1}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-1}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-1}"

UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-1}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-8}"
UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"
UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-0}"

RUN_LOG="$RUN_ROOT/run.log"
KERNEL_LOG="$RUN_ROOT/kernel_since_start.log"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/stage_n_p8d_block_quota_summary.txt"
CHECK_LOG="$RUN_ROOT/block_quota_check.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_p8d_synthetic.txt"

mkdir -p "$RUN_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

pool_name() {
  case "$1" in
    1) echo "kv" ;;
    2) echo "weights" ;;
    3) echo "scratch" ;;
    *) echo "unknown" ;;
  esac
}

set_param() {
  local key="$1"
  local value="$2"
  local path="$PARAM_DIR/$key"
  local actual

  [ -e "$path" ] || die "Missing UVM module parameter: $path"
  actual="$(cat "$path" 2>/dev/null || true)"
  if [ "$actual" = "$value" ]; then
    return 0
  fi

  if [ "$WRITE_UVM_PARAMS" = "1" ]; then
    sudo test -w "$path" || die "$key expected $value, got $actual, and $path is not writable. Reload nvidia_uvm with RESET_COUNTERS=1 or pass the value at insmod time."
    echo "$value" | sudo tee "$path" >/dev/null
  fi
  actual="$(cat "$path" 2>/dev/null || true)"
  [ "$actual" = "$value" ] || die "$key expected $value, got $actual"
}

reload_uvm_if_requested() {
  if [ "$RESET_COUNTERS" != "1" ]; then
    return 0
  fi

  [ -f "$UVM_KO_PATH" ] || die "nvidia-uvm.ko not found: $UVM_KO_PATH"
  if lsmod | grep -q '^nvidia_uvm'; then
    sudo rmmod nvidia_uvm || die "Failed to rmmod nvidia_uvm. Stop CUDA/vLLM processes or run RESET_COUNTERS=0."
  fi

  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs=1 \
    uvm_enable_builtin_tests=1 \
    uvm_phase_f_route_log_delay_ms="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
    uvm_phase_n_block_quota_enable="$UVM_PHASE_N_BLOCK_QUOTA_ENABLE" \
    uvm_phase_n_block_quota_trace_only="$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY" \
    uvm_phase_n_auto_reclaim_enable="$UVM_PHASE_N_AUTO_RECLAIM_ENABLE" \
    uvm_phase_n_auto_reclaim_execute="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE" \
    uvm_phase_n_auto_reclaim_every="$UVM_PHASE_N_AUTO_RECLAIM_EVERY" \
    uvm_phase_n_auto_reclaim_scan_every="$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY" \
    uvm_phase_n_auto_reclaim_max_victims="$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS" \
    uvm_phase_n_auto_reclaim_cooldown_epochs="$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS" \
    uvm_phase_n_auto_reclaim_failure_backoff_epochs="$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS" \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0 \
    || die "Failed to insmod $UVM_KO_PATH"
}

configure_uvm_params() {
  set_param uvm_phase_f_route_log_delay_ms "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  set_param uvm_phase_n_block_quota_enable "$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  set_param uvm_phase_n_block_quota_trace_only "$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  set_param uvm_phase_n_auto_reclaim_enable "$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  set_param uvm_phase_n_auto_reclaim_execute "$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  set_param uvm_phase_n_auto_reclaim_every "$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  set_param uvm_phase_n_auto_reclaim_scan_every "$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  set_param uvm_phase_n_auto_reclaim_max_victims "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  set_param uvm_phase_n_auto_reclaim_cooldown_epochs "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  set_param uvm_phase_n_auto_reclaim_failure_backoff_epochs "$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS"
}

write_kernel_log_since_start() {
  local since_human
  since_human="$(date -u -d "@$RUN_START_EPOCH" '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || true)"
  if command -v journalctl >/dev/null 2>&1 && [ -n "$since_human" ]; then
    journalctl -k --no-pager --since "$since_human" > "$KERNEL_LOG" 2>/dev/null || true
  else
    dmesg -T > "$KERNEL_LOG" 2>/dev/null || true
  fi
}

write_route_log_snapshot() {
  local proc_path="/proc/driver/nvidia-uvm/phase_f_route_log"
  if [ -r "$proc_path" ]; then
    cat "$proc_path" > "$ROUTE_LOG" 2>/dev/null || true
  else
    : > "$ROUTE_LOG"
  fi
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "UVM_KO_PATH=$UVM_KO_PATH"
  echo "DEVICE=$DEVICE"
  echo "BLOCK_BYTES=$BLOCK_BYTES"
  echo "BLOCKS=$BLOCKS"
  echo "BUDGET_BLOCKS=$BUDGET_BLOCKS"
  echo "POOLS=$POOLS"
  echo "RESET_COUNTERS=$RESET_COUNTERS"
  echo "WRITE_UVM_PARAMS=$WRITE_UVM_PARAMS"
  echo "ALLOWED_POST_FAULT_SLACK_BLOCKS=$ALLOWED_POST_FAULT_SLACK_BLOCKS"
  echo "REQUIRE_RECLAIM=$REQUIRE_RECLAIM"
  env | sort | grep -E '^UVM_PHASE_(F_ROUTE|N_BLOCK|N_AUTO)' || true
} > "$ENV_FILE"

echo "Stage N P8d synthetic block-quota proof started."
echo "RUN_ROOT=$RUN_ROOT"

reload_uvm_if_requested
configure_uvm_params
sudo dmesg -C >/dev/null 2>&1 || true
RUN_START_EPOCH="$(date +%s)"

make -C "$UVM_TEST_DIR" phase_f_smoke > "$RUN_ROOT/build.log" 2>&1

status=0
for pool in $POOLS; do
  name="$(pool_name "$pool")"
  log="$RUN_ROOT/phase_f_smoke_${name}.log"
  echo "Running synthetic pool=$pool ($name)" | tee -a "$RUN_LOG"
  set +e
  (
    cd "$UVM_TEST_DIR"
    LD_LIBRARY_PATH="$UVM_TEST_DIR:${LD_LIBRARY_PATH:-}" \
      ./phase_f_smoke \
        --step phase-n-block-quota \
        --device "$DEVICE" \
        --pool "$pool" \
        --bytes "$BLOCK_BYTES" \
        --blocks "$BLOCKS" \
        --budget-blocks "$BUDGET_BLOCKS"
  ) > "$log" 2>&1
  rc=$?
  set -e
  echo "pool=$pool name=$name status=$rc log=$log" | tee -a "$RUN_LOG"
  if [ "$rc" -ne 0 ]; then
    status=$rc
  fi
done

write_kernel_log_since_start
write_route_log_snapshot

check_args=(
  "$RUN_ROOT"
  --require-materialize
  --require-over-budget
  --require-auto-hook
  --require-block-fastpath
  --allowed-post-fault-slack-blocks "$ALLOWED_POST_FAULT_SLACK_BLOCKS"
  --fail-on-violation
)
if [ "$REQUIRE_RECLAIM" = "1" ]; then
  check_args+=(--require-reclaim)
fi
for pool in $POOLS; do
  case "$pool" in
    1) check_args+=(--require-pool kv) ;;
    2) check_args+=(--require-pool weights) ;;
    3) check_args+=(--require-pool scratch) ;;
  esac
done

set +e
python3 "$SCRIPT_DIR/check_stage_n_block_quota.py" "${check_args[@]}" > "$CHECK_LOG" 2>&1
check_status=$?
set -e

{
  echo "Stage N P8d Synthetic Block Quota Summary"
  echo "========================================="
  echo "run_root=$RUN_ROOT"
  echo "run_status=$status"
  echo "check_status=$check_status"
  echo "kernel_log=$KERNEL_LOG"
  echo "route_log=$ROUTE_LOG"
  echo "check_log=$CHECK_LOG"
  echo
  cat "$CHECK_LOG"
  echo
  echo "Recent Block Evidence"
  echo "---------------------"
  grep -R -E 'phase_n_block_|phase_n_get_pool_state block_quota=1|phase_n_reclaim_pool block_quota=1' \
    "$ROUTE_LOG" "$KERNEL_LOG" "$RUN_ROOT" 2>/dev/null | tail -n 160 || true
} > "$SUMMARY"

cat "$SUMMARY"

if [ "$status" -ne 0 ]; then
  exit "$status"
fi
exit "$check_status"
