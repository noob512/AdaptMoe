#!/usr/bin/env bash
# Validate kernel-level manual migration fault control.
#
# The probe allocates managed memory, explicitly prefetches it to CPU, then
# launches a GPU kernel that reads the page.
#   mode=0: normal UVM service, kernel should succeed.
#   mode=1: observe/log only, kernel should still succeed.
#   mode=2: log and cancel before UVM migration, kernel should fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"
ROUTE_PROC="/proc/driver/nvidia-uvm/phase_f_route_log"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/manual_migration_fault_control_$(date -u +%Y%m%d_%H%M%S)}"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
RESET_UVM="${RESET_UVM:-1}"
DEVICE="${DEVICE:-0}"
BYTES="${BYTES:-4194304}"
POOL_MASK="${POOL_MASK:-0}"

PROBE="$UVM_TEST_DIR/manual_migration_fault_probe"
SUMMARY="$RUN_ROOT/summary.txt"

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

  [ -e "$path" ] || die "Missing UVM module parameter: $path. Rebuild and reload nvidia_uvm first, or run with RESET_UVM=1."
  actual="$(cat "$path" 2>/dev/null || true)"
  if [ "$actual" = "$value" ]; then
    return 0
  fi

  echo "$value" | sudo tee "$path" >/dev/null
  actual="$(cat "$path" 2>/dev/null || true)"
  [ "$actual" = "$value" ] || die "$key expected $value, got $actual"
}

route_line_count() {
  if [ -r "$ROUTE_PROC" ]; then
    wc -l < "$ROUTE_PROC"
  else
    echo 0
  fi
}

capture_new_route_lines() {
  local before="$1"
  local out="$2"
  local start=$((before + 1))

  if [ -r "$ROUTE_PROC" ]; then
    sed -n "${start},\$p" "$ROUTE_PROC" > "$out" 2>/dev/null || true
  else
    : > "$out"
  fi
}

restore_mode() {
  if [ -e "$PARAM_DIR/uvm_manual_migration_fault_mode" ]; then
    echo 0 | sudo tee "$PARAM_DIR/uvm_manual_migration_fault_mode" >/dev/null 2>&1 || true
  fi
}
trap restore_mode EXIT

build_probe() {
  make -C "$UVM_TEST_DIR" manual_migration_fault_probe
}

reload_uvm_if_requested() {
  if [ "$RESET_UVM" != "1" ]; then
    return 0
  fi

  [ -f "$UVM_KO_PATH" ] || die "nvidia-uvm.ko not found: $UVM_KO_PATH"
  if lsmod | grep -q '^nvidia_uvm'; then
    sudo rmmod nvidia_uvm || die "Failed to rmmod nvidia_uvm. Stop CUDA/vLLM processes or run RESET_UVM=0 with a preloaded module."
  fi

  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs=1 \
    uvm_enable_builtin_tests=1 \
    uvm_manual_migration_fault_mode=0 \
    uvm_manual_migration_fault_pool_mask="$POOL_MASK" \
    uvm_manual_migration_fault_log_enable=1 \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    || die "Failed to insmod $UVM_KO_PATH"
}

run_case() {
  local name="$1"
  local mode="$2"
  local expect="$3"
  local before
  local rc
  local out="$RUN_ROOT/${name}.out"
  local route="$RUN_ROOT/${name}.route.log"

  set_param uvm_manual_migration_fault_pool_mask "$POOL_MASK"
  set_param uvm_manual_migration_fault_log_enable 1
  set_param uvm_manual_migration_fault_mode "$mode"

  before="$(route_line_count)"

  set +e
  "$PROBE" "--expect-${expect}" --device "$DEVICE" --bytes "$BYTES" > "$out" 2>&1
  rc=$?
  set -e

  capture_new_route_lines "$before" "$route"

  if [ "$rc" -ne 0 ]; then
    cat "$out" >&2
    cat "$route" >&2
    die "$name failed with rc=$rc"
  fi

  case "$name" in
    normal)
      if grep -q "manual_migration_fault" "$route"; then
        cat "$route" >&2
        die "normal mode unexpectedly logged manual_migration_fault"
      fi
      ;;
    observe)
      grep -q "manual_migration_fault mode=1 action=observe" "$route" \
        || die "observe mode did not log action=observe"
      grep -q "kernel_success" "$out" \
        || die "observe mode did not preserve successful kernel execution"
      ;;
    fail_fast)
      grep -q "manual_migration_fault mode=2 action=block" "$route" \
        || die "fail_fast mode did not log action=block"
      grep -q "expected_failure_observed=1" "$out" \
        || die "fail_fast mode did not observe expected CUDA failure"
      ;;
  esac
}

main() {
  build_probe
  reload_uvm_if_requested

  set_param uvm_manual_migration_fault_pool_mask "$POOL_MASK"
  set_param uvm_manual_migration_fault_log_enable 1

  run_case normal 0 success
  run_case observe 1 success
  run_case fail_fast 2 failure

  {
    echo "manual_migration_fault_control: PASS"
    echo "run_root=$RUN_ROOT"
    echo "normal_out=$RUN_ROOT/normal.out"
    echo "observe_out=$RUN_ROOT/observe.out"
    echo "observe_route=$RUN_ROOT/observe.route.log"
    echo "fail_fast_out=$RUN_ROOT/fail_fast.out"
    echo "fail_fast_route=$RUN_ROOT/fail_fast.route.log"
  } | tee "$SUMMARY"
}

main "$@"
