#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/uvm_2mb_offload_release_slot_restore_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "${RUN_ROOT}"

cd "${REPO_ROOT}"
python3 workloads/vllm/check_2mb_offload_release_slot_restore.py \
  --out-json "${RUN_ROOT}/result.json" \
  2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}"
