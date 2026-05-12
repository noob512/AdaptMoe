#!/usr/bin/env bash
# Validate the minimal Phase F test ioctl before exercising the large batch ABI.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_f_small_ping_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
BYTES="${BYTES:-4096}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_f_small_ping_check.sh [options]

Options:
  --run-root <path>
  --log-file <path>
  --bytes <n>       forwarded to phase_f_smoke, default 4096
  -h, --help

This script does not reload nvidia_uvm and does not allocate managed memory.
It runs:
  1. ping-open   open/close /dev/nvidia-uvm
  2. ping-init   open + UVM_INITIALIZE_IOCTL
  3. ping-small  open + initialize + small UVM_TEST_PHASE_F_PING ioctl
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_f_small_ping_check.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

exec > >(tee "$LOG_FILE") 2>&1

echo "Stage M Phase F small ping check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

echo "==========================================================="
echo " Phase F Small Ping Check"
echo "==========================================================="
echo "run_root: $RUN_ROOT"
echo "bytes: $BYTES"
uname -a

if [ ! -r /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
  echo "ERROR: nvidia_uvm is not loaded or uvm_enable_builtin_tests is unavailable." >&2
  exit 3
fi

tests_enabled="$(cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests)"
echo "uvm_enable_builtin_tests=$tests_enabled"
if [ "$tests_enabled" != "1" ]; then
  echo "ERROR: Phase F test ioctl requires uvm_enable_builtin_tests=1." >&2
  echo "Load the modified nvidia_uvm module once with builtin tests enabled, then re-run." >&2
  exit 4
fi

"$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
  --run-root "$RUN_ROOT/open" \
  --log-file "$RUN_ROOT/ping_open.log" \
  --no-enable-tests \
  --step ping-open \
  --bytes "$BYTES"

"$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
  --run-root "$RUN_ROOT/init" \
  --log-file "$RUN_ROOT/ping_init.log" \
  --no-enable-tests \
  --step ping-init \
  --bytes "$BYTES"

"$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
  --run-root "$RUN_ROOT/small" \
  --log-file "$RUN_ROOT/ping_small.log" \
  --no-enable-tests \
  --step ping-small \
  --bytes "$BYTES"

echo "==========================================================="
echo " Result"
echo "==========================================================="
echo "PASS: small Phase F ping ioctl completed."
echo "Logs:"
echo "  $RUN_ROOT/ping_open.log"
echo "  $RUN_ROOT/ping_init.log"
echo "  $RUN_ROOT/ping_small.log"
