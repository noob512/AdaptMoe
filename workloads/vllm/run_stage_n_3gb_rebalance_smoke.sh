#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$ROOT/workloads/vllm/vllm/uvm_test"
BIN="$TEST_DIR/phase_n_3gb_rebalance_smoke"
PARAM_DIR="${PARAM_DIR:-/sys/module/nvidia_uvm/parameters}"
ROUTE_PROC="${ROUTE_PROC:-/proc/driver/nvidia-uvm/phase_f_route_log}"

DEVICE="${DEVICE:-0}"
REGION_MIB="${REGION_MIB:-512}"
EXPERT_REGIONS="${EXPERT_REGIONS:-5}"
KV_REGIONS="${KV_REGIONS:-2}"
ASYNC="${ASYNC:-0}"
LOW_REBALANCE="${LOW_REBALANCE:-0}"
RUN_ROOT="${RUN_ROOT:-$ROOT/workloads/vllm/log/stage_n_3gb_rebalance_smoke_$(date +%Y%m%d_%H%M%S)}"
RESTORE_UVM_PARAMS="${RESTORE_UVM_PARAMS:-1}"
PHASE_F_DEBUG_LOG="${VLLM_UVM_PHASE_F_DEBUG_LOG:-$RUN_ROOT/phase_f_user_debug.log}"

REQUIRED_UVM_PARAMS=(
  uvm_phase_n_block_quota_enable
  uvm_phase_n_native_block_enable
)

declare -A ORIGINAL_UVM_PARAMS=()

mkdir -p "$RUN_ROOT"
export VLLM_UVM_PHASE_F_DEBUG_LOG="$PHASE_F_DEBUG_LOG"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

read_param() {
  local name="$1"
  local path="$PARAM_DIR/$name"
  [ -e "$path" ] || die "Missing UVM module parameter: $path"
  cat "$path"
}

write_param() {
  local name="$1"
  local value="$2"
  local path="$PARAM_DIR/$name"

  [ -e "$path" ] || die "Missing UVM module parameter: $path"
  if [ "$(cat "$path" 2>/dev/null || true)" = "$value" ]; then
    return 0
  fi
  echo "$value" | sudo tee "$path" >/dev/null
  [ "$(cat "$path" 2>/dev/null || true)" = "$value" ] \
    || die "Failed to set $name=$value"
}

snapshot_uvm_params() {
  local out="$1"

  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "loaded_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
    echo "built_srcversion=$(modinfo -F srcversion "$ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
    for path in "$PARAM_DIR"/uvm_phase_n_* "$PARAM_DIR"/uvm_manual_migration_fault_*; do
      [ -e "$path" ] || continue
      printf '%s=%s\n' "$(basename "$path")" "$(cat "$path" 2>/dev/null || true)"
    done
  } > "$out"
}

configure_uvm_params() {
  local name

  for name in "${REQUIRED_UVM_PARAMS[@]}"; do
    ORIGINAL_UVM_PARAMS["$name"]="$(read_param "$name")"
    write_param "$name" 1
  done
}

restore_uvm_params() {
  local name

  [ "$RESTORE_UVM_PARAMS" = "1" ] || return 0
  for name in "${REQUIRED_UVM_PARAMS[@]}"; do
    if [ -n "${ORIGINAL_UVM_PARAMS[$name]+set}" ]; then
      echo "${ORIGINAL_UVM_PARAMS[$name]}" | sudo tee "$PARAM_DIR/$name" >/dev/null 2>&1 || true
    fi
  done
  snapshot_uvm_params "$RUN_ROOT/uvm_params_restored.txt" || true
}

route_line_count() {
  if [ -r "$ROUTE_PROC" ]; then
    wc -l < "$ROUTE_PROC"
  else
    echo 0
  fi
}

capture_new_route_lines() {
  local before="$1"
  local out="$2"
  local start=$((before + 1))

  if [ -r "$ROUTE_PROC" ]; then
    sed -n "${start},\$p" "$ROUTE_PROC" > "$out" 2>/dev/null || true
  else
    : > "$out"
  fi
}

trap restore_uvm_params EXIT

echo "RUN_ROOT=$RUN_ROOT"
echo "phase_f_user_debug_log=$PHASE_F_DEBUG_LOG"
snapshot_uvm_params "$RUN_ROOT/uvm_params_before.txt"
configure_uvm_params
snapshot_uvm_params "$RUN_ROOT/uvm_params_configured.txt"
echo "building phase_n_3gb_rebalance_smoke"
make -C "$TEST_DIR" phase_n_3gb_rebalance_smoke
route_before="$(route_line_count)"

{
  echo "=== srcversion ==="
  echo "loaded: $(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  echo "built:  $(modinfo -F srcversion "$ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
  echo
  echo "=== UVM params configured ==="
  cat "$RUN_ROOT/uvm_params_configured.txt"
  echo
  echo "=== nvidia-smi before ==="
  nvidia-smi || true
  echo
  echo "=== phase_n_3gb_rebalance_smoke ==="
  "$BIN" \
    --device "$DEVICE" \
    --region-mib "$REGION_MIB" \
    --expert-regions "$EXPERT_REGIONS" \
    --kv-regions "$KV_REGIONS" \
    --async "$ASYNC" \
    --low-rebalance "$LOW_REBALANCE"
  echo
  echo "=== nvidia-smi after ==="
  nvidia-smi || true
} 2>&1 | tee "$RUN_ROOT/smoke.log"
capture_new_route_lines "$route_before" "$RUN_ROOT/phase_f_route_log_delta.log"

if grep -q 'phase_n_3gb_final .*total_ok=1' "$RUN_ROOT/smoke.log"; then
  echo "PASS stage_n_3gb_rebalance_smoke"
  exit 0
fi

echo "FAIL stage_n_3gb_rebalance_smoke"
exit 1
