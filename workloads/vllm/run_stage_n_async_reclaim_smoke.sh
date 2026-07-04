#!/usr/bin/env bash
# Stage N M4 async reclaim smoke.
#
# This validates the budget-change -> pending reclaim request -> safe-point
# automatic reclaim loop. Unlike the M3 smoke, this script does not call the
# manual RECLAIM_POOL ioctl after the budget shift.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/stage_n_async_reclaim_smoke_$(date -u +%Y%m%d_%H%M%S)}"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEVICE="${DEVICE:-0}"
BLOCK_BYTES="${BLOCK_BYTES:-2097152}"
BLOCKS="${BLOCKS:-2}"
BUDGET_BLOCKS="${BUDGET_BLOCKS:-1}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
WRITE_UVM_PARAMS="${WRITE_UVM_PARAMS:-1}"

UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-1}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-0}"
UVM_PHASE_N_NATIVE_BLOCK_ENABLE="${UVM_PHASE_N_NATIVE_BLOCK_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_ENABLE="${UVM_PHASE_N_ADMISSION_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_TRACE_ONLY="${UVM_PHASE_N_ADMISSION_TRACE_ONLY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_ASYNC="${UVM_PHASE_N_AUTO_RECLAIM_ASYNC:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-8}"
UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"

RUN_LOG="$RUN_ROOT/run.log"
SMOKE_LOG="$RUN_ROOT/phase_f_smoke_async_reclaim.log"
KERNEL_LOG="$RUN_ROOT/kernel_since_start.log"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/stage_n_async_reclaim_summary.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_async_reclaim.txt"

mkdir -p "$RUN_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
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
    sudo test -w "$path" || die "$key expected $value, got $actual, and $path is not writable"
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
    uvm_phase_n_native_block_enable="$UVM_PHASE_N_NATIVE_BLOCK_ENABLE" \
    uvm_phase_n_admission_enable="$UVM_PHASE_N_ADMISSION_ENABLE" \
    uvm_phase_n_admission_trace_only="$UVM_PHASE_N_ADMISSION_TRACE_ONLY" \
    uvm_phase_n_auto_reclaim_enable="$UVM_PHASE_N_AUTO_RECLAIM_ENABLE" \
    uvm_phase_n_auto_reclaim_execute="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE" \
    uvm_phase_n_auto_reclaim_async="$UVM_PHASE_N_AUTO_RECLAIM_ASYNC" \
    uvm_phase_n_auto_reclaim_every="$UVM_PHASE_N_AUTO_RECLAIM_EVERY" \
    uvm_phase_n_auto_reclaim_scan_every="$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY" \
    uvm_phase_n_auto_reclaim_max_victims="$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS" \
    uvm_phase_n_auto_reclaim_cooldown_epochs="$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS" \
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
  set_param uvm_phase_n_native_block_enable "$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  set_param uvm_phase_n_admission_enable "$UVM_PHASE_N_ADMISSION_ENABLE"
  set_param uvm_phase_n_admission_trace_only "$UVM_PHASE_N_ADMISSION_TRACE_ONLY"
  set_param uvm_phase_n_auto_reclaim_enable "$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  set_param uvm_phase_n_auto_reclaim_execute "$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  set_param uvm_phase_n_auto_reclaim_async "$UVM_PHASE_N_AUTO_RECLAIM_ASYNC"
  set_param uvm_phase_n_auto_reclaim_every "$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  set_param uvm_phase_n_auto_reclaim_scan_every "$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  set_param uvm_phase_n_auto_reclaim_max_victims "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  set_param uvm_phase_n_auto_reclaim_cooldown_epochs "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
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

extract_field() {
  local key="$1"
  local line="$2"
  sed -n "s/.*${key}=\\([0-9][0-9]*\\).*/\\1/p" <<<"$line"
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "UVM_KO_PATH=$UVM_KO_PATH"
  echo "DEVICE=$DEVICE"
  echo "BLOCK_BYTES=$BLOCK_BYTES"
  echo "BLOCKS=$BLOCKS"
  echo "BUDGET_BLOCKS=$BUDGET_BLOCKS"
  env | sort | grep -E '^UVM_PHASE_(F_ROUTE|N_BLOCK|N_NATIVE|N_ADMISSION|N_AUTO)' || true
} > "$ENV_FILE"

reload_uvm_if_requested
configure_uvm_params
sudo dmesg -C >/dev/null 2>&1 || true
RUN_START_EPOCH="$(date +%s)"

make -C "$UVM_TEST_DIR" phase_f_smoke > "$RUN_ROOT/build.log" 2>&1

echo "Running async reclaim smoke" | tee -a "$RUN_LOG"
set +e
(
  cd "$UVM_TEST_DIR"
  LD_LIBRARY_PATH="$UVM_TEST_DIR:${LD_LIBRARY_PATH:-}" \
    ./phase_f_smoke \
      --step phase-n-dynamic-budget-auto \
      --device "$DEVICE" \
      --bytes "$BLOCK_BYTES" \
      --blocks "$BLOCKS" \
      --budget-blocks "$BUDGET_BLOCKS"
) > "$SMOKE_LOG" 2>&1
rc=$?
set -e
echo "phase_f_smoke status=$rc log=$SMOKE_LOG" | tee -a "$RUN_LOG"

write_kernel_log_since_start
write_route_log_snapshot

status="$rc"
shift_line="$(grep 'phase_n_dynamic_budget_auto_shift ' "$SMOKE_LOG" | tail -n 1 || true)"
after_line="$(grep 'phase_n_dynamic_budget_auto_after ' "$SMOKE_LOG" | tail -n 1 || true)"
async_line="$(grep 'phase_n_async_reclaim .*pool=2 ' "$ROUTE_LOG" | tail -n 1 || true)"

shift_expert_over="$(extract_field expert_over "$shift_line")"
after_expert_over="$(extract_field expert_over "$after_line")"
after_kv_resident="$(extract_field kv_resident "$after_line")"
async_reclaimed="$(extract_field reclaimed "$async_line")"
expected_kv_resident=$((BLOCK_BYTES * BLOCKS))

if [ "$status" -eq 0 ]; then
  [ -n "$shift_line" ] || { echo "missing phase_n_dynamic_budget_auto_shift" | tee -a "$RUN_LOG"; status=1; }
  [ -n "$after_line" ] || { echo "missing phase_n_dynamic_budget_auto_after" | tee -a "$RUN_LOG"; status=1; }
  [ -n "$async_line" ] || { echo "missing phase_n_async_reclaim pool=2" | tee -a "$RUN_LOG"; status=1; }
fi

if [ "$status" -eq 0 ]; then
  [ "${shift_expert_over:-0}" -gt 0 ] || { echo "expected expert_over > 0 after auto budget shift" | tee -a "$RUN_LOG"; status=1; }
  [ "${async_reclaimed:-0}" -gt 0 ] || { echo "expected async reclaimed bytes > 0" | tee -a "$RUN_LOG"; status=1; }
  [ "${after_expert_over:-1}" -eq 0 ] || { echo "expected expert_over == 0 after async reclaim" | tee -a "$RUN_LOG"; status=1; }
  [ "${after_kv_resident:-0}" -ge "$expected_kv_resident" ] || {
    echo "expected kv_resident >= $expected_kv_resident after KV admission" | tee -a "$RUN_LOG"
    status=1
  }
fi

{
  echo "Stage N Async Reclaim Smoke Summary"
  echo "==================================="
  echo "run_root=$RUN_ROOT"
  echo "status=$status"
  echo "smoke_log=$SMOKE_LOG"
  echo "kernel_log=$KERNEL_LOG"
  echo "route_log=$ROUTE_LOG"
  echo
  echo "Key Output"
  echo "----------"
  grep -E 'phase_n_dynamic_budget_(initial|before|auto_shift|auto_after)|manual_reclaim=0' "$SMOKE_LOG" || true
  echo
  echo "Kernel Route Evidence"
  echo "---------------------"
  grep -E 'phase_n_set_dynamic_budgets|phase_n_async_reclaim|phase_n_admission|phase_n_native_block_reclaim' \
    "$ROUTE_LOG" "$KERNEL_LOG" 2>/dev/null | tail -n 240 || true
} > "$SUMMARY"

cat "$SUMMARY"
exit "$status"
