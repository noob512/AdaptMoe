#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="${SCRIPT_DIR}/vllm"

RUN_ROOT="${RUN_ROOT:-/tmp/uvm_kv_block_batch_migration_smoke_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-uv run python}"

NUM_BLOCKS="${NUM_BLOCKS:-128}"
LAYERS="${LAYERS:-48}"
RANGE_BYTES="${RANGE_BYTES:-32768}"
MIGRATE_START="${MIGRATE_START:-64}"
MIGRATE_COUNT="${MIGRATE_COUNT:-32}"
MIGRATION_MODE="${MIGRATION_MODE:-block}"
DEVICE="${DEVICE:-0}"
PROGRESS_EVERY_BLOCKS="${PROGRESS_EVERY_BLOCKS:-128}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"

mkdir -p "${RUN_ROOT}"

cat > "${RUN_ROOT}/env.txt" <<EOF
run_root=${RUN_ROOT}
num_blocks=${NUM_BLOCKS}
layers=${LAYERS}
range_bytes=${RANGE_BYTES}
bytes_per_block=$((LAYERS * RANGE_BYTES))
migrate_start=${MIGRATE_START}
migrate_count=${MIGRATE_COUNT}
migration_mode=${MIGRATION_MODE}
device=${DEVICE}
progress_every_blocks=${PROGRESS_EVERY_BLOCKS}
timeout_sec=${TIMEOUT_SEC}
uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)
EOF

{
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
} > "${RUN_ROOT}/gpu_before.txt"

set +e
(
  cd "${VLLM_ROOT}"
  if [[ "${TIMEOUT_SEC}" -gt 0 ]]; then
    VLLM_UVM_RESIDENCY_TAG_ONLY=0 \
    timeout --foreground --kill-after=30s "${TIMEOUT_SEC}s" \
      ${VLLM_PY} "${SCRIPT_DIR}/check_kv_block_batch_migration_smoke.py" \
        --num-blocks "${NUM_BLOCKS}" \
        --layers "${LAYERS}" \
        --range-bytes "${RANGE_BYTES}" \
        --migrate-start "${MIGRATE_START}" \
        --migrate-count "${MIGRATE_COUNT}" \
        --mode "${MIGRATION_MODE}" \
        --device "${DEVICE}" \
        --progress-every-blocks "${PROGRESS_EVERY_BLOCKS}" \
        --out-json "${RUN_ROOT}/kv_block_batch_migration_result.json"
  else
    VLLM_UVM_RESIDENCY_TAG_ONLY=0 \
    ${VLLM_PY} "${SCRIPT_DIR}/check_kv_block_batch_migration_smoke.py" \
      --num-blocks "${NUM_BLOCKS}" \
      --layers "${LAYERS}" \
      --range-bytes "${RANGE_BYTES}" \
      --migrate-start "${MIGRATE_START}" \
      --migrate-count "${MIGRATE_COUNT}" \
      --mode "${MIGRATION_MODE}" \
      --device "${DEVICE}" \
      --progress-every-blocks "${PROGRESS_EVERY_BLOCKS}" \
      --out-json "${RUN_ROOT}/kv_block_batch_migration_result.json"
  fi
) 2>&1 | tee "${RUN_ROOT}/run.log"
rc=${PIPESTATUS[0]}
set -e

{
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits || true
  nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
} > "${RUN_ROOT}/gpu_after.txt"

echo "${rc}" > "${RUN_ROOT}/returncode.txt"
echo "KV block batch migration smoke result: ${RUN_ROOT}/kv_block_batch_migration_result.json"
exit "${rc}"
