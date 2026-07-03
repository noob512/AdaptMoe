#!/usr/bin/env bash
# Stage N admission smoke.
#
# This runs phase_f_smoke for the KV and WEIGHT pools with 1GB budgets.
# Defaults touch 513 x 2MB blocks, so enforce mode should deny the 513th block
# before UVM allocates a new GPU chunk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/stage_n_admission_smoke_$(date -u +%Y%m%d_%H%M%S)}"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEVICE="${DEVICE:-0}"
BLOCK_BYTES="${BLOCK_BYTES:-2097152}"
BUDGET_BLOCKS="${BUDGET_BLOCKS:-512}"
BLOCKS="${BLOCKS:-513}"
POOLS="${POOLS:-1 2}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
WRITE_UVM_PARAMS="${WRITE_UVM_PARAMS:-1}"
EXPECT_DENY="${EXPECT_DENY:-1}"

UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-1}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-0}"
UVM_PHASE_N_NATIVE_BLOCK_ENABLE="${UVM_PHASE_N_NATIVE_BLOCK_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_ENABLE="${UVM_PHASE_N_ADMISSION_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_TRACE_ONLY="${UVM_PHASE_N_ADMISSION_TRACE_ONLY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-0}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-0}"

RUN_LOG="$RUN_ROOT/run.log"
KERNEL_LOG="$RUN_ROOT/kernel_since_start.log"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/stage_n_admission_summary.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_admission.txt"

mkdir -p "$RUN_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

pool_name() {
  case "$1" in
    1) echo "kv" ;;
    2) echo "weights" ;;
    *) echo "pool_$1" ;;
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
  echo "BUDGET_BLOCKS=$BUDGET_BLOCKS"
  echo "BLOCKS=$BLOCKS"
  echo "POOLS=$POOLS"
  echo "EXPECT_DENY=$EXPECT_DENY"
  env | sort | grep -E '^UVM_PHASE_(F_ROUTE|N_BLOCK|N_NATIVE|N_ADMISSION|N_AUTO)' || true
} > "$ENV_FILE"

reload_uvm_if_requested
configure_uvm_params
sudo dmesg -C >/dev/null 2>&1 || true
RUN_START_EPOCH="$(date +%s)"

make -C "$UVM_TEST_DIR" phase_f_smoke > "$RUN_ROOT/build.log" 2>&1

status=0
for pool in $POOLS; do
  name="$(pool_name "$pool")"
  log="$RUN_ROOT/phase_f_smoke_${name}.log"
  echo "Running admission pool=$pool ($name)" | tee -a "$RUN_LOG"
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

  if [ "$EXPECT_DENY" = "1" ]; then
    if [ "$rc" -eq 0 ]; then
      echo "expected deny for pool=$pool but command succeeded" | tee -a "$RUN_LOG"
      status=1
    fi
  elif [ "$rc" -ne 0 ]; then
    status=$rc
  fi
done

write_kernel_log_since_start
write_route_log_snapshot

{
  echo "Stage N Admission Smoke Summary"
  echo "==============================="
  echo "run_root=$RUN_ROOT"
  echo "status=$status"
  echo "kernel_log=$KERNEL_LOG"
  echo "route_log=$ROUTE_LOG"
  echo
  echo "Admission Evidence"
  echo "------------------"
  grep -R -E 'phase_n_admission|cudaMemset block=.*failed|phase_n_block_quota_touch' \
    "$ROUTE_LOG" "$KERNEL_LOG" "$RUN_ROOT" 2>/dev/null | tail -n 200 || true
} > "$SUMMARY"

cat "$SUMMARY"
exit "$status"
