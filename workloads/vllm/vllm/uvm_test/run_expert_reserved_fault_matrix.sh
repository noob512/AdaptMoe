#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/expert_reserved_fault_matrix.$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"
BYTES="${BYTES:-3145728}"
BASE_OFFSET="${BASE_OFFSET:-1048576}"
GUARD_BYTES="${GUARD_BYTES:-1048576}"
SLOT_BYTES="${SLOT_BYTES:-4194304}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
INCLUDE_DESTRUCTIVE=0

usage() {
  cat <<EOF
Usage: $0 [--include-destructive]

Runs minimal reserved-slot fault probes:

  alignment_guard
    Tests non-2MiB-aligned 3MiB region install/invalidate does not corrupt
    adjacent pages in the same managed allocation.

  cleanup_reaccess
    Tests same managed VA can fault/re-read correctly after slot invalidate.
    The runner also checks 2MiB aligned and 1.5MiB unaligned variants, because
    this is the current strongest reproducer.

  unrelated_write_after_cleanup
    Tests cleanup/invalidate does not break a different managed allocation's
    later GPU write path. This is closest to the observed VIRT_WRITE fault
    outside the reserved expert region.

  no_invalidate_unrelated_write
    Control for the previous case. If this passes but cleanup case fails,
    cleanup/unmap is the likely trigger.

  big_pte_split_cleanup
    Primes native GPU PTEs first, then tests reserved 3MiB remap/cleanup.
    This targets 2M/big-PTE split and TLB invalidation bugs.

  cross_stream_cleanup --sync-mode=device
    Safe control showing device-wide sync before invalidate is sufficient in a
    minimal two-stream program.

Destructive cases, only with --include-destructive:

  readonly_write
    Intentionally writes into readonly slot-backed PTEs. A CUDA illegal access
    or Xid is expected and may require reboot.

  cross_stream_cleanup --sync-mode=current
    Launches a side-stream reader and only syncs current stream before
    invalidate. This may reproduce an early-cleanup race and may require reboot.

Environment overrides:
  DEVICE=${DEVICE}
  BYTES=${BYTES}
  BASE_OFFSET=${BASE_OFFSET}
  GUARD_BYTES=${GUARD_BYTES}
  SLOT_BYTES=${SLOT_BYTES}
  TIMEOUT_SEC=${TIMEOUT_SEC}
  LOG_DIR=${LOG_DIR}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-destructive)
      INCLUDE_DESTRUCTIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
cd "${SCRIPT_DIR}"

echo "log_dir=${LOG_DIR}"
echo "building expert_reserved_fault_probe"
make expert_reserved_fault_probe

check_param() {
  local path="$1"
  local want="$2"
  local got
  if [[ ! -e "${path}" ]]; then
    echo "missing ${path}; custom nvidia_uvm module is probably not loaded" >&2
    return 1
  fi
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

run_case() {
  local name="$1"
  shift
  local log="${LOG_DIR}/${name}.log"

  echo
  echo "=== ${name} ==="
  set +e
  timeout "${TIMEOUT_SEC}s" ./expert_reserved_fault_probe \
    --device "${DEVICE}" \
    --bytes "${BYTES}" \
    --base-offset "${BASE_OFFSET}" \
    --guard-bytes "${GUARD_BYTES}" \
    --slot-bytes "${SLOT_BYTES}" \
    "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -eq 0 ]]; then
    echo "RESULT ${name}: PASS"
  elif [[ ${rc} -eq 77 ]]; then
    echo "RESULT ${name}: SKIP_DESTRUCTIVE_GUARD"
  elif [[ ${rc} -eq 124 ]]; then
    echo "RESULT ${name}: TIMEOUT"
  else
    echo "RESULT ${name}: FAIL rc=${rc}"
  fi
  return 0
}

run_case alignment_guard --case alignment_guard
run_case cleanup_reaccess --case cleanup_reaccess
run_case cleanup_reaccess_2m_aligned \
  --case cleanup_reaccess \
  --bytes 2097152 \
  --base-offset 0 \
  --guard-bytes 1048576 \
  --slot-bytes 2097152
run_case cleanup_reaccess_1_5m_unaligned \
  --case cleanup_reaccess \
  --bytes 1572864 \
  --base-offset 1572864 \
  --guard-bytes 1048576 \
  --slot-bytes "${SLOT_BYTES}"
run_case unrelated_write_after_cleanup --case unrelated_write_after_cleanup
run_case no_invalidate_unrelated_write --case no_invalidate_unrelated_write
run_case big_pte_split_cleanup --case big_pte_split_cleanup
run_case cross_stream_cleanup_device --case cross_stream_cleanup --sync-mode device

if [[ "${INCLUDE_DESTRUCTIVE}" -eq 1 ]]; then
  echo
  echo "Running destructive probes. These may trigger Xid and require reboot."
  run_case readonly_write_destructive --case readonly_write --destructive
  run_case cross_stream_cleanup_current_destructive --case cross_stream_cleanup --sync-mode current --destructive
else
  echo
  echo "Skipping destructive probes. Use --include-destructive after saving state/rebooting:"
  echo "  ${0} --include-destructive"
fi

nvidia-smi | tee "${LOG_DIR}/nvidia_smi_after.txt" || true
echo
echo "logs written to ${LOG_DIR}"
