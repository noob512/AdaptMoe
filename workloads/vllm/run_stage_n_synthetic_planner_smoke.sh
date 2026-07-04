#!/usr/bin/env bash
# Stage N M6 synthetic planner smoke.
#
# This runs a small non-vLLM planner loop that simulates:
#   1. expert-heavy phase: small active KV, large expert hot budget;
#   2. KV-heavy long-context phase: active KV budget is protected, expert budget shrinks;
#   3. expert-heavy recovery: KV protection is removed, expert budget grows back.
#
# It verifies the mechanism-level FluxMoE-style behavior without measuring
# vLLM throughput or using compression/decompression.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/log/stage_n_synthetic_planner_smoke_$(date -u +%Y%m%d_%H%M%S)}"
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"
DEVICE="${DEVICE:-0}"
BLOCK_BYTES="${BLOCK_BYTES:-2097152}"
LOGICAL_BLOCKS="${LOGICAL_BLOCKS:-4}"
MANAGED_BLOCKS="${MANAGED_BLOCKS:-4}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
WRITE_UVM_PARAMS="${WRITE_UVM_PARAMS:-1}"

UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-1}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-0}"
UVM_PHASE_N_NATIVE_BLOCK_ENABLE="${UVM_PHASE_N_NATIVE_BLOCK_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_ENABLE="${UVM_PHASE_N_ADMISSION_ENABLE:-1}"
UVM_PHASE_N_ADMISSION_TRACE_ONLY="${UVM_PHASE_N_ADMISSION_TRACE_ONLY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_ASYNC="${UVM_PHASE_N_AUTO_RECLAIM_ASYNC:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-8}"
UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"

RUN_LOG="$RUN_ROOT/run.log"
SMOKE_LOG="$RUN_ROOT/phase_f_smoke_synthetic_planner.log"
KERNEL_LOG="$RUN_ROOT/kernel_since_start.log"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/stage_n_synthetic_planner_summary.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_synthetic_planner.txt"

mkdir -p "$RUN_ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

set_param() {
  local key="$1"
  local value="$2"
  local path="$PARAM_DIR/$key"
  local actual

  [ -e "$path" ] || die "Missing UVM module parameter: $path"
  actual="$(cat "$path" 2>/dev/null || true)"
  if [ "$actual" = "$value" ]; then
    return 0
  fi

  if [ "$WRITE_UVM_PARAMS" = "1" ]; then
    sudo test -w "$path" || die "$key expected $value, got $actual, and $path is not writable"
    echo "$value" | sudo tee "$path" >/dev/null
  fi

  actual="$(cat "$path" 2>/dev/null || true)"
  [ "$actual" = "$value" ] || die "$key expected $value, got $actual"
}

reload_uvm_if_requested() {
  if [ "$RESET_COUNTERS" != "1" ]; then
    return 0
  fi

  [ -f "$UVM_KO_PATH" ] || die "nvidia-uvm.ko not found: $UVM_KO_PATH"
  if lsmod | grep -q '^nvidia_uvm'; then
    sudo rmmod nvidia_uvm || die "Failed to rmmod nvidia_uvm. Stop CUDA/vLLM processes or run RESET_COUNTERS=0."
  fi

  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs=1 \
    uvm_enable_builtin_tests=1 \
    uvm_phase_f_route_log_delay_ms="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
    uvm_phase_n_block_quota_enable="$UVM_PHASE_N_BLOCK_QUOTA_ENABLE" \
    uvm_phase_n_block_quota_trace_only="$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY" \
    uvm_phase_n_native_block_enable="$UVM_PHASE_N_NATIVE_BLOCK_ENABLE" \
    uvm_phase_n_admission_enable="$UVM_PHASE_N_ADMISSION_ENABLE" \
    uvm_phase_n_admission_trace_only="$UVM_PHASE_N_ADMISSION_TRACE_ONLY" \
    uvm_phase_n_auto_reclaim_enable="$UVM_PHASE_N_AUTO_RECLAIM_ENABLE" \
    uvm_phase_n_auto_reclaim_execute="$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE" \
    uvm_phase_n_auto_reclaim_async="$UVM_PHASE_N_AUTO_RECLAIM_ASYNC" \
    uvm_phase_n_auto_reclaim_every="$UVM_PHASE_N_AUTO_RECLAIM_EVERY" \
    uvm_phase_n_auto_reclaim_scan_every="$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY" \
    uvm_phase_n_auto_reclaim_max_victims="$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS" \
    uvm_phase_n_auto_reclaim_cooldown_epochs="$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS" \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0 \
    || die "Failed to insmod $UVM_KO_PATH"
}

configure_uvm_params() {
  set_param uvm_phase_f_route_log_delay_ms "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  set_param uvm_phase_n_block_quota_enable "$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  set_param uvm_phase_n_block_quota_trace_only "$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  set_param uvm_phase_n_native_block_enable "$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  set_param uvm_phase_n_admission_enable "$UVM_PHASE_N_ADMISSION_ENABLE"
  set_param uvm_phase_n_admission_trace_only "$UVM_PHASE_N_ADMISSION_TRACE_ONLY"
  set_param uvm_phase_n_auto_reclaim_enable "$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  set_param uvm_phase_n_auto_reclaim_execute "$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  set_param uvm_phase_n_auto_reclaim_async "$UVM_PHASE_N_AUTO_RECLAIM_ASYNC"
  set_param uvm_phase_n_auto_reclaim_every "$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  set_param uvm_phase_n_auto_reclaim_scan_every "$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  set_param uvm_phase_n_auto_reclaim_max_victims "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  set_param uvm_phase_n_auto_reclaim_cooldown_epochs "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
}

write_kernel_log_since_start() {
  local since_human
  since_human="$(date -u -d "@$RUN_START_EPOCH" '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || true)"
  if command -v journalctl >/dev/null 2>&1 && [ -n "$since_human" ]; then
    journalctl -k --no-pager --since "$since_human" > "$KERNEL_LOG" 2>/dev/null || true
  else
    dmesg -T > "$KERNEL_LOG" 2>/dev/null || true
  fi
}

write_route_log_snapshot() {
  local proc_path="/proc/driver/nvidia-uvm/phase_f_route_log"
  if [ -r "$proc_path" ]; then
    cat "$proc_path" > "$ROUTE_LOG" 2>/dev/null || true
  else
    : > "$ROUTE_LOG"
  fi
}

extract_field() {
  local key="$1"
  local line="$2"
  sed -n "s/.*${key}=\\([0-9][0-9]*\\).*/\\1/p" <<<"$line"
}

count_matches() {
  local pattern="$1"
  local file="$2"
  grep -E "$pattern" "$file" 2>/dev/null | wc -l
}

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "UVM_KO_PATH=$UVM_KO_PATH"
  echo "DEVICE=$DEVICE"
  echo "BLOCK_BYTES=$BLOCK_BYTES"
  echo "LOGICAL_BLOCKS=$LOGICAL_BLOCKS"
  echo "MANAGED_BLOCKS=$MANAGED_BLOCKS"
  env | sort | grep -E '^UVM_PHASE_(F_ROUTE|N_BLOCK|N_NATIVE|N_ADMISSION|N_AUTO)' || true
} > "$ENV_FILE"

reload_uvm_if_requested
configure_uvm_params
sudo dmesg -C >/dev/null 2>&1 || true
RUN_START_EPOCH="$(date +%s)"

make -C "$UVM_TEST_DIR" phase_f_smoke > "$RUN_ROOT/build.log" 2>&1

echo "Running synthetic planner smoke" | tee -a "$RUN_LOG"
set +e
(
  cd "$UVM_TEST_DIR"
  LD_LIBRARY_PATH="$UVM_TEST_DIR:${LD_LIBRARY_PATH:-}" \
    ./phase_f_smoke \
      --step phase-n-synthetic-planner \
      --device "$DEVICE" \
      --bytes "$BLOCK_BYTES" \
      --blocks "$LOGICAL_BLOCKS" \
      --budget-blocks "$MANAGED_BLOCKS"
) > "$SMOKE_LOG" 2>&1
rc=$?
set -e
echo "phase_f_smoke status=$rc log=$SMOKE_LOG" | tee -a "$RUN_LOG"

write_kernel_log_since_start
write_route_log_snapshot

status="$rc"
kv_heavy_line="$(grep 'phase_n_synthetic_planner_step name=kv_heavy_long_context ' "$SMOKE_LOG" | tail -n 1 || true)"
recovery_line="$(grep 'phase_n_synthetic_planner_step name=expert_heavy_recovery ' "$SMOKE_LOG" | tail -n 1 || true)"
final_line="$(grep 'phase_n_synthetic_planner_final ' "$SMOKE_LOG" | tail -n 1 || true)"
expert_reclaim_records="$(count_matches 'phase_n_async_reclaim .*pool=2 .*reclaimed=[1-9]' "$ROUTE_LOG")"
kv_reclaim_records="$(count_matches 'phase_n_async_reclaim .*pool=1 .*reclaimed=[1-9]' "$ROUTE_LOG")"
protect_records="$(count_matches 'phase_n_protect_active_kv .*pending_mask=0x[1-9a-fA-F]' "$ROUTE_LOG")"

kv_phase_ok="$(extract_field kv_phase_ok "$final_line")"
final_kv_ok="$(extract_field final_kv_ok "$final_line")"
final_expert_ok="$(extract_field final_expert_ok "$final_line")"
total_ok="$(extract_field total_ok "$final_line")"
kv_heavy_kv_resident="$(extract_field kv_resident "$kv_heavy_line")"
recovery_expert_resident="$(extract_field expert_resident "$recovery_line")"

if [ "$status" -eq 0 ]; then
  [ -n "$kv_heavy_line" ] || { echo "missing kv_heavy planner step" | tee -a "$RUN_LOG"; status=1; }
  [ -n "$recovery_line" ] || { echo "missing expert_heavy recovery planner step" | tee -a "$RUN_LOG"; status=1; }
  [ -n "$final_line" ] || { echo "missing synthetic planner final line" | tee -a "$RUN_LOG"; status=1; }
fi

if [ "$status" -eq 0 ]; then
  [ "${protect_records:-0}" -gt 0 ] || { echo "expected phase_n_protect_active_kv with pending expert reclaim" | tee -a "$RUN_LOG"; status=1; }
  [ "${expert_reclaim_records:-0}" -gt 0 ] || { echo "expected expert async reclaim records" | tee -a "$RUN_LOG"; status=1; }
  [ "${kv_reclaim_records:-0}" -gt 0 ] || { echo "expected KV async reclaim records after KV budget decreases" | tee -a "$RUN_LOG"; status=1; }
  [ "${kv_phase_ok:-0}" -eq 1 ] || { echo "expected kv_phase_ok=1" | tee -a "$RUN_LOG"; status=1; }
  [ "${final_kv_ok:-0}" -eq 1 ] || { echo "expected final_kv_ok=1" | tee -a "$RUN_LOG"; status=1; }
  [ "${final_expert_ok:-0}" -eq 1 ] || { echo "expected final_expert_ok=1" | tee -a "$RUN_LOG"; status=1; }
  [ "${total_ok:-0}" -eq 1 ] || { echo "expected total_ok=1" | tee -a "$RUN_LOG"; status=1; }
  [ "${kv_heavy_kv_resident:-0}" -gt 0 ] || { echo "expected KV resident bytes in KV-heavy phase" | tee -a "$RUN_LOG"; status=1; }
  [ "${recovery_expert_resident:-0}" -gt 0 ] || { echo "expected expert resident bytes in recovery phase" | tee -a "$RUN_LOG"; status=1; }
fi

{
  echo "Stage N Synthetic Planner Smoke Summary"
  echo "======================================="
  echo "run_root=$RUN_ROOT"
  echo "status=$status"
  echo "smoke_log=$SMOKE_LOG"
  echo "kernel_log=$KERNEL_LOG"
  echo "route_log=$ROUTE_LOG"
  echo
  echo "Planner Output"
  echo "--------------"
  grep -E 'phase_n_synthetic_planner_(alloc|signal|step|final)' "$SMOKE_LOG" || true
  echo
  echo "Kernel Route Evidence"
  echo "---------------------"
  grep -E 'phase_n_set_dynamic_budgets|phase_n_protect_active_kv|phase_n_async_reclaim|phase_n_native_block_reclaim|phase_n_admission' \
    "$ROUTE_LOG" "$KERNEL_LOG" 2>/dev/null | tail -n 320 || true
  echo
  echo "Mechanism Counts"
  echo "----------------"
  echo "protect_records=$protect_records"
  echo "expert_reclaim_records=$expert_reclaim_records"
  echo "kv_reclaim_records=$kv_reclaim_records"
} > "$SUMMARY"

cat "$SUMMARY"
exit "$status"
