#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/nvidia-uvm-gpu}
RUN_ROOT=${RUN_ROOT:-/tmp/uvm_512k_subrange_release_phase_s2_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-0}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/workloads/vllm:${REPO_ROOT}/workloads/vllm/vllm:${PYTHONPATH:-}" \
python3 workloads/vllm/check_512k_subrange_release_phase_s2.py \
  --device "${DEVICE}" \
  --out-json "${RUN_ROOT}/phase_s2_512k_subrange_release_result.json" \
  "$@"

echo "Phase S2 result: ${RUN_ROOT}/phase_s2_512k_subrange_release_result.json"
