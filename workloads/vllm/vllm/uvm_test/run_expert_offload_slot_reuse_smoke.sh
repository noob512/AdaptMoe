#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/expert_offload_slot_reuse_smoke_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-0}"
EXPERT_COUNT="${EXPERT_COUNT:-8}"
EXPERT_BYTES="${EXPERT_BYTES:-64M}"
SLOT_COUNT="${SLOT_COUNT:-48}"
SLOT_BYTES="${SLOT_BYTES:-4M}"
POOL_ID="${POOL_ID:-0}"
TARGET_FREE_BEFORE_SLOTS="${TARGET_FREE_BEFORE_SLOTS:-128M}"
PRESSURE_CHUNK_BYTES="${PRESSURE_CHUNK_BYTES:-256M}"
ALLOW_EVICT="${ALLOW_EVICT:-0}"
SKIP_BEFORE_PREALLOC="${SKIP_BEFORE_PREALLOC:-0}"
USE_DRIVER_RELEASE="${USE_DRIVER_RELEASE:-0}"
REGION_ID_BASE="${REGION_ID_BASE:-0x0ff1000000000000}"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "device=${DEVICE}"
  echo "expert_count=${EXPERT_COUNT}"
  echo "expert_bytes=${EXPERT_BYTES}"
  echo "slot_count=${SLOT_COUNT}"
  echo "slot_bytes=${SLOT_BYTES}"
  echo "pool_id=${POOL_ID}"
  echo "target_free_before_slots=${TARGET_FREE_BEFORE_SLOTS}"
  echo "pressure_chunk_bytes=${PRESSURE_CHUNK_BYTES}"
  echo "allow_evict=${ALLOW_EVICT}"
  echo "skip_before_prealloc=${SKIP_BEFORE_PREALLOC}"
  echo "use_driver_release=${USE_DRIVER_RELEASE}"
  echo "region_id_base=${REGION_ID_BASE}"
  if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
    echo "uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
  fi
} | tee "${RUN_ROOT}/env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" || true
fi

make -C "${SCRIPT_DIR}" expert_offload_slot_reuse_smoke \
  2>&1 | tee "${RUN_ROOT}/build.log"

args=(
  "${SCRIPT_DIR}/expert_offload_slot_reuse_smoke"
  --device "${DEVICE}"
  --expert-count "${EXPERT_COUNT}"
  --expert-bytes "${EXPERT_BYTES}"
  --slot-count "${SLOT_COUNT}"
  --slot-bytes "${SLOT_BYTES}"
  --pool-id "${POOL_ID}"
  --target-free-before-slots "${TARGET_FREE_BEFORE_SLOTS}"
  --pressure-chunk-bytes "${PRESSURE_CHUNK_BYTES}"
)

if [[ "${ALLOW_EVICT}" == "1" ]]; then
  args+=(--allow-evict)
fi
if [[ "${SKIP_BEFORE_PREALLOC}" == "1" ]]; then
  args+=(--skip-before-prealloc)
fi
if [[ "${USE_DRIVER_RELEASE}" == "1" ]]; then
  args+=(--use-driver-release --region-id-base "${REGION_ID_BASE}")
fi

set +e
"${args[@]}" 2>&1 | tee "${RUN_ROOT}/run.log"
status=${PIPESTATUS[0]}
set -e

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_after.txt" || true
fi
if command -v sudo >/dev/null 2>&1; then
  sudo -n dmesg -T 2>/dev/null | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log" || \
    dmesg -T 2>/dev/null | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log" || true
else
  dmesg -T 2>/dev/null | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log" || true
fi

python3 - "$RUN_ROOT" "$status" <<'PY'
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
status = int(sys.argv[2])
log = (run_root / "run.log").read_text(errors="replace")

def parse_line(prefix, label=None):
    for line in log.splitlines():
        if not line.startswith(prefix):
            continue
        if label is not None and f"label={label}" not in line:
            continue
        pairs = dict(re.findall(r"([A-Za-z0-9_]+)=([^\s]+)", line))
        return pairs
    return {}

summary = {
    "run_root": str(run_root),
    "status": status,
    "before_prealloc": parse_line("reserved_pool_prealloc", "before_offload"),
    "after_prealloc": parse_line("reserved_pool_prealloc", "after_offload"),
    "after_gpu_residency": parse_line("residency_summary", "after_expert_gpu_prefetch"),
    "after_cpu_residency": parse_line("residency_summary", "after_expert_cpu_offload"),
    "va_stability": parse_line("expert_va_stability"),
    "final": parse_line("expert_offload_slot_reuse_smoke_final"),
}
(run_root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
PY

echo "run_root=${RUN_ROOT}"
echo "status=${status}"
exit "${status}"
