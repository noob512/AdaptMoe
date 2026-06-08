#!/usr/bin/env bash
# Stage N P8d block-quota experiment bundle.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8d_block_quota_all_$(date -u +%Y%m%d_%H%M%S)}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_VLLM="${RUN_VLLM:-1}"
PROFILE="${PROFILE:-kv}"
SUMMARY="$RUN_ROOT/stage_n_p8d_block_quota_all_summary.txt"

mkdir -p "$RUN_ROOT"

synthetic_status=0
vllm_status=0

if [ "$RUN_SYNTHETIC" = "1" ]; then
  echo "Running P8d synthetic block-quota proof..."
  set +e
  RUN_ROOT="$RUN_ROOT/synthetic" ./run_stage_n_p8d_synthetic_block_quota.sh
  synthetic_status=$?
  set -e
fi

if [ "$RUN_VLLM" = "1" ]; then
  echo "Running P8d vLLM block-quota probe..."
  set +e
  RUN_ROOT="$RUN_ROOT/vllm_${PROFILE}" PROFILE="$PROFILE" ./run_stage_n_p8d_vllm_block_quota_probe.sh
  vllm_status=$?
  set -e
fi

{
  echo "Stage N P8d Block Quota Experiment Bundle"
  echo "========================================="
  echo "run_root=$RUN_ROOT"
  echo "run_synthetic=$RUN_SYNTHETIC"
  echo "run_vllm=$RUN_VLLM"
  echo "profile=$PROFILE"
  echo "synthetic_status=$synthetic_status"
  echo "vllm_status=$vllm_status"
  echo
  for path in \
    "$RUN_ROOT/synthetic/stage_n_p8d_block_quota_summary.txt" \
    "$RUN_ROOT/vllm_${PROFILE}/stage_n_p8d_vllm_block_quota_summary.txt"; do
    if [ -f "$path" ]; then
      echo "===== $path ====="
      grep -E 'run_status=|smoke_status=|check_status=|block_quota_pass=|block_quota_violations=|pool_kv=|pool_weights=' "$path" || true
      echo
    fi
  done
} > "$SUMMARY"

cat "$SUMMARY"

if [ "$synthetic_status" -ne 0 ]; then
  exit "$synthetic_status"
fi
exit "$vllm_status"
