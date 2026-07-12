#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/expert_offload_release_smoke_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-0}"
EXPERT_COUNT="${EXPERT_COUNT:-8}"
EXPERT_BYTES="${EXPERT_BYTES:-64M}"
PROBE_BYTES="${PROBE_BYTES:-256M}"
TARGET_FREE_BEFORE_PROBE="${TARGET_FREE_BEFORE_PROBE:-128M}"
PRESSURE_CHUNK_BYTES="${PRESSURE_CHUNK_BYTES:-256M}"
SKIP_BEFORE_PROBE="${SKIP_BEFORE_PROBE:-0}"
KEEP_AFTER_PROBE="${KEEP_AFTER_PROBE:-0}"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "device=${DEVICE}"
  echo "expert_count=${EXPERT_COUNT}"
  echo "expert_bytes=${EXPERT_BYTES}"
  echo "probe_bytes=${PROBE_BYTES}"
  echo "target_free_before_probe=${TARGET_FREE_BEFORE_PROBE}"
  echo "pressure_chunk_bytes=${PRESSURE_CHUNK_BYTES}"
  echo "skip_before_probe=${SKIP_BEFORE_PROBE}"
  echo "keep_after_probe=${KEEP_AFTER_PROBE}"
  if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
    echo "uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
  fi
} | tee "${RUN_ROOT}/env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" || true
fi

make -C "${SCRIPT_DIR}" expert_offload_release_smoke 2>&1 | tee "${RUN_ROOT}/build.log"

args=(
  "${SCRIPT_DIR}/expert_offload_release_smoke"
  --device "${DEVICE}"
  --expert-count "${EXPERT_COUNT}"
  --expert-bytes "${EXPERT_BYTES}"
  --probe-bytes "${PROBE_BYTES}"
  --target-free-before-probe "${TARGET_FREE_BEFORE_PROBE}"
  --pressure-chunk-bytes "${PRESSURE_CHUNK_BYTES}"
)

if [[ "${KEEP_AFTER_PROBE}" == "1" ]]; then
  args+=(--keep-after-probe)
fi
if [[ "${SKIP_BEFORE_PROBE}" == "1" ]]; then
  args+=(--skip-before-probe)
fi

set +e
"${args[@]}" 2>&1 | tee "${RUN_ROOT}/run.log"
status=${PIPESTATUS[0]}
set -e

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_after.txt" || true
fi
dmesg -T | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log" || true

{
  echo "{"
  echo "  \"run_root\": \"${RUN_ROOT}\","
  echo "  \"status\": ${status},"
  echo "  \"device\": ${DEVICE},"
  echo "  \"expert_count\": ${EXPERT_COUNT},"
  echo "  \"expert_bytes\": \"${EXPERT_BYTES}\","
  echo "  \"probe_bytes\": \"${PROBE_BYTES}\","
  echo "  \"target_free_before_probe\": \"${TARGET_FREE_BEFORE_PROBE}\""
  echo "}"
} > "${RUN_ROOT}/summary.json"

echo "run_root=${RUN_ROOT}"
echo "status=${status}"
exit "${status}"
