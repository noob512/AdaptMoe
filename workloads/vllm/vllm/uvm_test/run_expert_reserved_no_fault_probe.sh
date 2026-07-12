#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-/tmp/expert_reserved_no_fault_probe_$(date +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-0}"
BYTES="${BYTES:-4194304}"
SLOT_BYTES="${SLOT_BYTES:-4194304}"
TIMEOUT_SEC="${TIMEOUT_SEC:-90}"
FAULT_LOG="${FAULT_LOG:-1}"

mkdir -p "${RUN_ROOT}"

{
  echo "run_root=${RUN_ROOT}"
  echo "date=$(date --iso-8601=seconds)"
  echo "device=${DEVICE}"
  echo "bytes=${BYTES}"
  echo "slot_bytes=${SLOT_BYTES}"
  if [[ -r /sys/module/nvidia_uvm/srcversion ]]; then
    echo "uvm_srcversion=$(cat /sys/module/nvidia_uvm/srcversion)"
  fi
  for param in \
    uvm_moe_slot_pte_remap_enable \
    uvm_moe_slot_pte_remap_max_bytes \
    uvm_moe_slot_pool_vidmem_budget_bytes \
    uvm_perf_fault_log_counters \
    uvm_perf_fault_log_addresses \
    uvm_perf_fault_log_destination; do
    if [[ -r "/sys/module/nvidia_uvm/parameters/${param}" ]]; then
      echo "${param}=$(cat "/sys/module/nvidia_uvm/parameters/${param}")"
    fi
  done
} | tee "${RUN_ROOT}/env.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_before.txt" || true
fi

make -C "${SCRIPT_DIR}" expert_reserved_no_fault_probe 2>&1 | tee "${RUN_ROOT}/build.log"

read_param() {
  local name="$1"
  cat "/sys/module/nvidia_uvm/parameters/${name}" 2>/dev/null || true
}

write_param() {
  local name="$1"
  local value="$2"
  if [[ -e "/sys/module/nvidia_uvm/parameters/${name}" ]]; then
    printf '%s' "${value}" | sudo tee "/sys/module/nvidia_uvm/parameters/${name}" >/dev/null
  fi
}

OLD_DEBUG_PRINTS="$(read_param uvm_debug_prints)"
OLD_FAULT_COUNTERS="$(read_param uvm_perf_fault_log_counters)"
OLD_FAULT_DESTINATION="$(read_param uvm_perf_fault_log_destination)"

restore_params() {
  if [[ -n "${OLD_DEBUG_PRINTS}" ]]; then
    write_param uvm_debug_prints "${OLD_DEBUG_PRINTS}"
  fi
  if [[ -n "${OLD_FAULT_COUNTERS}" ]]; then
    write_param uvm_perf_fault_log_counters "${OLD_FAULT_COUNTERS}"
  fi
  if [[ -n "${OLD_FAULT_DESTINATION}" ]]; then
    write_param uvm_perf_fault_log_destination "${OLD_FAULT_DESTINATION}"
  fi
}
trap restore_params EXIT

if [[ "${FAULT_LOG}" == "1" ]]; then
  write_param uvm_debug_prints 1
  write_param uvm_perf_fault_log_counters 1
  write_param uvm_perf_fault_log_destination 0
fi

run_mode() {
  local mode="$1"
  local log="${RUN_ROOT}/${mode}.log"
  local dmesg_before="${RUN_ROOT}/${mode}_dmesg_before.log"
  local dmesg_after="${RUN_ROOT}/${mode}_dmesg_after.log"
  local dmesg_delta="${RUN_ROOT}/${mode}_dmesg_delta.log"
  local status_file="${RUN_ROOT}/${mode}.status"
  local marker="codex_uvm_reserved_no_fault_${mode}_$(date +%s%N)"

  sudo dmesg -T > "${dmesg_before}" || true
  printf '<6>%s_begin\n' "${marker}" | sudo tee /dev/kmsg >/dev/null || true
  set +e
  timeout "${TIMEOUT_SEC}s" "${SCRIPT_DIR}/expert_reserved_no_fault_probe" \
    --device "${DEVICE}" \
    --bytes "${BYTES}" \
    --slot-bytes "${SLOT_BYTES}" \
    --mode "${mode}" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '<6>%s_end\n' "${marker}" | sudo tee /dev/kmsg >/dev/null || true
  sudo dmesg -T > "${dmesg_after}" || true
  python3 - "$dmesg_after" "$dmesg_delta" "${marker}" <<'PY'
import sys
after, out, marker = sys.argv[1:4]
try:
    lines = open(after, "r", errors="replace").read().splitlines()
except OSError:
    lines = []
begin = next((i for i, line in enumerate(lines) if f"{marker}_begin" in line), None)
end = next((i for i, line in enumerate(lines) if f"{marker}_end" in line and (begin is None or i > begin)), None)
if begin is not None and end is not None:
    delta = lines[begin + 1:end]
else:
    delta = []
with open(out, "w") as f:
    for line in delta:
        f.write(line + "\n")
PY
  echo "${rc}" > "${status_file}"
  echo "mode=${mode} status=${rc}"
  echo "mode=${mode} replayable_fault_stats:"
  grep -F "Replayable fault stats" "${dmesg_delta}" || true
  return 0
}

run_mode native
native_status="$(cat "${RUN_ROOT}/native.status")"
run_mode reserved
reserved_status="$(cat "${RUN_ROOT}/reserved.status")"
status=0
if [[ ${native_status} -ne 0 || ${reserved_status} -ne 0 ]]; then
  status=1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "${RUN_ROOT}/nvidia_smi_after.txt" || true
fi
{ dmesg -T 2>&1 || true; } | tail -n 200 > "${RUN_ROOT}/dmesg_tail.log"

{
  echo "{"
  echo "  \"run_root\": \"${RUN_ROOT}\","
  echo "  \"status\": ${status},"
  echo "  \"native_status\": ${native_status},"
  echo "  \"reserved_status\": ${reserved_status},"
  echo "  \"device\": ${DEVICE},"
  echo "  \"bytes\": ${BYTES},"
  echo "  \"slot_bytes\": ${SLOT_BYTES}"
  echo "}"
} > "${RUN_ROOT}/summary.json"

echo "run_root=${RUN_ROOT}"
echo "status=${status}"
exit "${status}"
