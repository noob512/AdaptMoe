#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/expert_reclaim_slot_smoke_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-0}"
EXPERT_COUNT="${EXPERT_COUNT:-160}"
OFFLOAD_COUNT="${OFFLOAD_COUNT:-16}"
EXPERT_BYTES="${EXPERT_BYTES:-4M}"
SLOT_COUNT="${SLOT_COUNT:-8}"
SLOT_BYTES="${SLOT_BYTES:-4M}"
TARGET_FREE_AFTER_EXPERTS="${TARGET_FREE_AFTER_EXPERTS:-0}"
PRESSURE_CHUNK_BYTES="${PRESSURE_CHUNK_BYTES:-256M}"
REQUIRE_FULL_AFTER="${REQUIRE_FULL_AFTER:-0}"
CONTROL_BEFORE_OFFLOAD="${CONTROL_BEFORE_OFFLOAD:-1}"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "device=${DEVICE}"
  echo "expert_count=${EXPERT_COUNT}"
  echo "offload_count=${OFFLOAD_COUNT}"
  echo "expert_bytes=${EXPERT_BYTES}"
  echo "slot_count=${SLOT_COUNT}"
  echo "slot_bytes=${SLOT_BYTES}"
  echo "target_free_after_experts=${TARGET_FREE_AFTER_EXPERTS}"
  echo "pressure_chunk_bytes=${PRESSURE_CHUNK_BYTES}"
  echo "require_full_after=${REQUIRE_FULL_AFTER}"
  echo "control_before_offload=${CONTROL_BEFORE_OFFLOAD}"
  if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
    echo "uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
  fi
  if [[ -r /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable ]]; then
    echo "uvm_moe_slot_pte_remap_enable=$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)"
  fi
} | tee "${RUN_ROOT}/env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" || true
fi

make -C "${SCRIPT_DIR}" expert_reclaim_slot_smoke 2>&1 | tee "${RUN_ROOT}/build.log"

args=(
  "${SCRIPT_DIR}/expert_reclaim_slot_smoke"
  --device "${DEVICE}"
  --expert-count "${EXPERT_COUNT}"
  --offload-count "${OFFLOAD_COUNT}"
  --expert-bytes "${EXPERT_BYTES}"
  --slot-count "${SLOT_COUNT}"
  --slot-bytes "${SLOT_BYTES}"
  --target-free-after-experts "${TARGET_FREE_AFTER_EXPERTS}"
  --pressure-chunk-bytes "${PRESSURE_CHUNK_BYTES}"
)

if [[ "${REQUIRE_FULL_AFTER}" == "1" ]]; then
  args+=(--require-full-after)
fi
if [[ "${CONTROL_BEFORE_OFFLOAD}" != "1" ]]; then
  args+=(--no-control-before-offload)
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
  echo "  \"offload_count\": ${OFFLOAD_COUNT},"
  echo "  \"expert_bytes\": \"${EXPERT_BYTES}\","
  echo "  \"slot_count\": ${SLOT_COUNT},"
  echo "  \"slot_bytes\": \"${SLOT_BYTES}\","
  echo "  \"target_free_after_experts\": \"${TARGET_FREE_AFTER_EXPERTS}\""
  echo "}"
} > "${RUN_ROOT}/summary.json"

echo "run_root=${RUN_ROOT}"
echo "status=${status}"
exit "${status}"
