#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/nvidia-uvm-gpu}
RUN_ROOT=${RUN_ROOT:-/tmp/uvm_128expert_cold8_selective_restore_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-0}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/workloads/vllm:${REPO_ROOT}/workloads/vllm/vllm:${PYTHONPATH:-}" \
python3 workloads/vllm/check_128expert_cold8_selective_restore.py \
  --device "${DEVICE}" \
  --out-json "${RUN_ROOT}/128expert_cold8_selective_restore_result.json" \
  "$@"

echo "128-expert cold8 selective-restore result: ${RUN_ROOT}/128expert_cold8_selective_restore_result.json"
