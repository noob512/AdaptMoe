#!/usr/bin/env bash
# Minimal Phase F driver executor smoke test.
#
# This script intentionally does not start vLLM. It validates the Phase F
# test ioctl on a tiny cudaMallocManaged allocation so driver failures can be
# isolated before running a full model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_f_smoke_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
BYTES="${BYTES:-1048576}"
DEVICE="${DEVICE:-0}"
ASYNC="${ASYNC:-0}"
ENABLE_BUILTIN_TESTS=1
FORCE_RELOAD_UVM=0

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_f_smoke.sh [options]

Options:
  --run-root <path>
  --log-file <path>
  --bytes <n>             managed allocation size, default 1048576
  --device <n>            CUDA device, default 0
  --async <0|1>           use async migrate flag for prefetch, default 0
  --no-enable-tests       do not try to enable uvm_enable_builtin_tests
  --force-reload-uvm      last-resort: reload nvidia_uvm with builtin tests
  --reload-uvm            deprecated alias for --force-reload-uvm
  -h, --help

Checks:
  1. Build uvm_allocator.so and phase_f_smoke
  2. Verify /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests
  3. Run Phase F ping ioctl with count=0
  4. Allocate a small cudaMallocManaged buffer
  5. Run QUERY_RESIDENCY
  6. Run PREFETCH_GPU
  7. Run ADVISE_CPU
  8. Run PREFETCH_CPU

Example:
  ./run_stage_m_phase_f_smoke.sh \
    --log-file log/stage_m_phase_f_smoke.log \
    --no-enable-tests
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --async) ASYNC="$2"; shift 2 ;;
    --no-enable-tests) ENABLE_BUILTIN_TESTS=0; shift ;;
    --force-reload-uvm) FORCE_RELOAD_UVM=1; shift ;;
    --reload-uvm) FORCE_RELOAD_UVM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_f_smoke.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage M Phase F smoke test started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

PHASE_F_DEBUG_LOG="$RUN_ROOT/phase_f_user_debug.log"
JOURNAL_BEFORE="$RUN_ROOT/journal_before.log"
JOURNAL_AFTER="$RUN_ROOT/journal_after.log"
DMESG_AFTER="$RUN_ROOT/dmesg_after.log"

log_section() {
  echo "==========================================================="
  echo " $*"
  echo "==========================================================="
}

run_step() {
  local name="$1"
  shift
  log_section "step: $name"
  date -u '+time_utc=%F %T'
  set +e
  "$@"
  local rc=$?
  set -e
  echo "step_result name=$name rc=$rc"
  date -u '+time_utc=%F %T'
  journalctl -k --since '-2 minutes' --no-pager | tail -120 || true
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: step '$name' failed; dumping Phase F user debug log tail." >&2
    tail -200 "$PHASE_F_DEBUG_LOG" 2>/dev/null || true
  fi
  return "$rc"
}

log_section "Stage M Phase F Smoke"
echo "run_root: $RUN_ROOT"
echo "bytes: $BYTES"
echo "device: $DEVICE"
echo "async: $ASYNC"
echo "enable_builtin_tests: $ENABLE_BUILTIN_TESTS"
echo "force_reload_uvm: $FORCE_RELOAD_UVM"
echo "uvm_ko_path: $UVM_KO_PATH"
echo "phase_f_user_debug_log: $PHASE_F_DEBUG_LOG"
uname -a

journalctl -k --since '-10 minutes' --no-pager > "$JOURNAL_BEFORE" || true

log_section "Build smoke binary"
make -B -C "$UVM_TEST_DIR" uvm phase_f_smoke

if [ "$ENABLE_BUILTIN_TESTS" = "1" ] && [ "$FORCE_RELOAD_UVM" != "1" ]; then
  log_section "Enable builtin tests through writable sysfs parameter"
  if [ -w /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
    echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests >/dev/null
  else
    echo "uvm_enable_builtin_tests is not writable in the currently loaded module."
    echo "Build and load the updated module once, then re-run without --force-reload-uvm."
  fi
fi

if [ "$FORCE_RELOAD_UVM" = "1" ]; then
  log_section "Force reload nvidia_uvm with builtin tests"
  echo "WARNING: forcing UVM reload is known to be unstable on this test system."
  if lsmod | awk '{print $1}' | grep -qx nvidia_uvm; then
    sudo rmmod nvidia_uvm
  fi
  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs=1 \
    uvm_enable_builtin_tests=1 \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0
fi

log_section "UVM module state"
lsmod | grep -E 'nvidia(_uvm)?' || true
if [ -r /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
  echo -n "uvm_enable_builtin_tests="
  cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests
else
  echo "uvm_enable_builtin_tests parameter is not readable"
fi

tests_enabled="$(cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests 2>/dev/null || echo missing)"
if [ "$tests_enabled" != "1" ]; then
  echo "ERROR: Phase F test ioctl requires uvm_enable_builtin_tests=1, got '$tests_enabled'." >&2
  echo "Preferred fix: rebuild/load the updated UVM module once so the parameter is writable." >&2
  echo "Last resort: re-run with --force-reload-uvm, but that path has caused crashes." >&2
  exit 3
fi

export VLLM_UVM_PHASE_F_DEBUG_LOG="$PHASE_F_DEBUG_LOG"

SMOKE="$UVM_TEST_DIR/phase_f_smoke"
run_step "ping" "$SMOKE" --step ping --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
run_step "query" "$SMOKE" --step query --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
run_step "prefetch-gpu" "$SMOKE" --step prefetch-gpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
run_step "advise-cpu" "$SMOKE" --step advise-cpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
run_step "prefetch-cpu" "$SMOKE" --step prefetch-cpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
run_step "all" "$SMOKE" --step all --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"

log_section "Final diagnostics"
journalctl -k --since '-10 minutes' --no-pager > "$JOURNAL_AFTER" || true
dmesg -T > "$DMESG_AFTER" 2>/dev/null || true
echo "phase_f_user_debug_log=$PHASE_F_DEBUG_LOG"
tail -200 "$PHASE_F_DEBUG_LOG" 2>/dev/null || true
echo "journal_before=$JOURNAL_BEFORE"
echo "journal_after=$JOURNAL_AFTER"
echo "dmesg_after=$DMESG_AFTER"
echo "PASS: Phase F smoke test completed."
