#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/expert_reserved_double_slot_event.$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"
LAYERS="${LAYERS:-20}"
OFFLOADED_EXPERTS="${OFFLOADED_EXPERTS:-3}"
BANK_COUNT="${BANK_COUNT:-2}"
SLOT_BYTES="${SLOT_BYTES:-4194304}"
SPIN_ITERS="${SPIN_ITERS:-2000}"

usage() {
  cat <<EOF
Usage: $0 [extra expert_reserved_double_slot_event_probe args...]

Default experiment:
  layers=${LAYERS}
  offloaded_experts=${OFFLOADED_EXPERTS}
  bank_count=${BANK_COUNT}
  slots_per_bank=offloaded_experts * 2
  total_slots=bank_count * slots_per_bank
  gate_up=3MiB, down=1.5MiB, slot=4MiB

Environment overrides:
  DEVICE=${DEVICE}
  LAYERS=${LAYERS}
  OFFLOADED_EXPERTS=${OFFLOADED_EXPERTS}
  BANK_COUNT=${BANK_COUNT}
  SLOT_BYTES=${SLOT_BYTES}
  SPIN_ITERS=${SPIN_ITERS}
  LOG_DIR=${LOG_DIR}
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

echo "log_dir=${LOG_DIR}"
echo "building expert_reserved_double_slot_event_probe"
make expert_reserved_double_slot_event_probe

check_param() {
  local path="$1"
  local want="$2"
  if [[ ! -e "${path}" ]]; then
    echo "missing ${path}; custom nvidia_uvm module is probably not loaded" >&2
    return 1
  fi
  local got
  got="$(cat "${path}")"
  echo "${path}=${got}"
  if [[ "${got}" != "${want}" ]]; then
    echo "warning: expected ${path}=${want}, got ${got}" >&2
  fi
}

check_param /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable 1
check_param /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes "${SLOT_BYTES}"
check_param /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes 67108864 || true

nvidia-smi | tee "${LOG_DIR}/nvidia_smi_before.txt"

set +e
./expert_reserved_double_slot_event_probe \
  --device "${DEVICE}" \
  --layers "${LAYERS}" \
  --offloaded-experts "${OFFLOADED_EXPERTS}" \
  --bank-count "${BANK_COUNT}" \
  --slot-bytes "${SLOT_BYTES}" \
  --spin-iters "${SPIN_ITERS}" \
  --quiet-layers \
  "$@" 2>&1 | tee "${LOG_DIR}/probe.log"
rc=${PIPESTATUS[0]}
set -e

nvidia-smi | tee "${LOG_DIR}/nvidia_smi_after.txt" || true

echo "logs written to ${LOG_DIR}"
exit "${rc}"
