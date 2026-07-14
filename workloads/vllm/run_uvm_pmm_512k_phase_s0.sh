#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/nvidia-uvm-gpu}
RUN_ROOT=${RUN_ROOT:-/tmp/uvm_pmm_512k_phase_s0_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-0}
SLOT_COUNT=${SLOT_COUNT:-48}
SLOT_BYTES=${SLOT_BYTES:-524288}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/workloads/vllm:${REPO_ROOT}/workloads/vllm/vllm:${PYTHONPATH:-}" \
python3 workloads/vllm/check_uvm_pmm_512k_phase_s0.py \
  --device "${DEVICE}" \
  --slot-count "${SLOT_COUNT}" \
  --slot-bytes "${SLOT_BYTES}" \
  --output "${RUN_ROOT}/phase_s0_512k_result.json" \
  "$@"

echo "Phase S0 result: ${RUN_ROOT}/phase_s0_512k_result.json"
