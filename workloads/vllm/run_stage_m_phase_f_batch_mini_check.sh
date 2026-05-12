#!/usr/bin/env bash
# Validate Phase F batch-shaped ioctl with a tiny stack-routed parameter struct.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_f_batch_mini_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
BYTES="${BYTES:-4096}"
RUN_COUNT1=1

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_f_batch_mini_check.sh [options]

Options:
  --run-root <path>
  --log-file <path>
  --bytes <n>       managed allocation size for batch-mini1, default 4096
  --count0-only     run only batch-mini0, no CUDA allocation
  -h, --help

This script does not reload nvidia_uvm.
It runs:
  1. ping-small   minimal ioctl sanity check
  2. batch-mini0  small batch-shaped ioctl with count=0
  3. batch-mini1  small batch-shaped ioctl with count=1, no query/migrate
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --count0-only) RUN_COUNT1=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_f_batch_mini_check.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

exec > >(tee "$LOG_FILE") 2>&1

echo "Stage M Phase F batch-mini check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

echo "==========================================================="
echo " Phase F Batch Mini Check"
echo "==========================================================="
echo "run_root: $RUN_ROOT"
echo "bytes: $BYTES"
echo "run_count1: $RUN_COUNT1"
uname -a

if [ ! -r /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
  echo "ERROR: nvidia_uvm is not loaded or uvm_enable_builtin_tests is unavailable." >&2
  exit 3
fi

tests_enabled="$(cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests)"
echo "uvm_enable_builtin_tests=$tests_enabled"
if [ "$tests_enabled" != "1" ]; then
  echo "ERROR: Phase F test ioctl requires uvm_enable_builtin_tests=1." >&2
  exit 4
fi

"$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
  --run-root "$RUN_ROOT/small" \
  --log-file "$RUN_ROOT/ping_small.log" \
  --no-enable-tests \
  --step ping-small \
  --bytes "$BYTES"

"$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
  --run-root "$RUN_ROOT/batch-mini0" \
  --log-file "$RUN_ROOT/batch_mini0.log" \
  --no-enable-tests \
  --step batch-mini0 \
  --bytes "$BYTES"

if [ "$RUN_COUNT1" = "1" ]; then
  "$SCRIPT_DIR/run_stage_m_phase_f_smoke.sh" \
    --run-root "$RUN_ROOT/batch-mini1" \
    --log-file "$RUN_ROOT/batch_mini1.log" \
    --no-enable-tests \
    --step batch-mini1 \
    --bytes "$BYTES"
fi

echo "==========================================================="
echo " Result"
echo "==========================================================="
echo "PASS: Phase F batch-mini checks completed."
echo "Logs:"
echo "  $RUN_ROOT/ping_small.log"
echo "  $RUN_ROOT/batch_mini0.log"
if [ "$RUN_COUNT1" = "1" ]; then
  echo "  $RUN_ROOT/batch_mini1.log"
fi
