#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/nvidia-uvm-gpu}
RUN_ROOT=${RUN_ROOT:-/tmp/uvm_512k_reserved_restore_phase_s3_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-0}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/workloads/vllm:${REPO_ROOT}/workloads/vllm/vllm:${PYTHONPATH:-}" \
python3 workloads/vllm/check_512k_reserved_restore_phase_s3.py \
  --device "${DEVICE}" \
  --out-json "${RUN_ROOT}/phase_s3_512k_reserved_restore_result.json" \
  "$@"

echo "Phase S3 result: ${RUN_ROOT}/phase_s3_512k_reserved_restore_result.json"
