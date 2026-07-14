#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ubuntu/nvidia-uvm-gpu}
RUN_ROOT=${RUN_ROOT:-/tmp/uvm_reserved_512k_64k_phase_s1_$(date +%Y%m%d_%H%M%S)}
DEVICE=${DEVICE:-0}
SLOT_COUNT=${SLOT_COUNT:-48}
SLOT_BYTES=${SLOT_BYTES:-524288}
PHYSICAL_CHUNK_BYTES=${PHYSICAL_CHUNK_BYTES:-65536}

mkdir -p "${RUN_ROOT}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}/workloads/vllm:${REPO_ROOT}/workloads/vllm/vllm:${PYTHONPATH:-}" \
python3 workloads/vllm/check_uvm_reserved_512k_64k_phase_s1.py \
  --device "${DEVICE}" \
  --slot-count "${SLOT_COUNT}" \
  --slot-bytes "${SLOT_BYTES}" \
  --physical-chunk-bytes "${PHYSICAL_CHUNK_BYTES}" \
  --output "${RUN_ROOT}/phase_s1_512k_64k_result.json" \
  "$@"

echo "Phase S1 result: ${RUN_ROOT}/phase_s1_512k_64k_result.json"
