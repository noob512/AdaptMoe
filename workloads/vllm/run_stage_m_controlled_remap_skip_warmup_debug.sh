#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-log/current_controlled_remap_skip_warmup_debug}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PLAN_PATH="${PLAN_PATH:-log/controlled_remap_v2_tiny_vllm_e2e/minimal_train_static_hotset_k1.json}"
BASE_PORT="${BASE_PORT:-8120}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
BUILT_SRCVERSION="$(modinfo -F srcversion "$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
EXPECTED_SRCVERSION="${EXPECTED_SRCVERSION:-$BUILT_SRCVERSION}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

check_no_d_state_processes() {
  local d_state
  d_state="$(ps -eLo pid,tid,ppid,stat,wchan:40,comm,args | awk '$4 ~ /D/ {print}' | head -50)"
  if [ -n "$d_state" ]; then
    echo "D-state threads detected:"
    echo "$d_state"
    return 1
  fi
}

check_module_state() {
  local loaded
  loaded="$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
  [ -n "$loaded" ] || die "nvidia_uvm is not loaded"
  echo "loaded_srcversion=$loaded"
  if [ -n "$EXPECTED_SRCVERSION" ] && [ "$loaded" != "$EXPECTED_SRCVERSION" ]; then
    die "loaded srcversion $loaded != expected $EXPECTED_SRCVERSION"
  fi

  local name value
  for name in \
    uvm_moe_slot_pte_remap_enable \
    uvm_moe_slot_pte_remap_max_bytes \
    uvm_moe_slot_pool_vidmem_budget_bytes \
    uvm_enable_builtin_tests; do
    value="$(cat "/sys/module/nvidia_uvm/parameters/$name" 2>/dev/null || true)"
    echo "$name=$value"
  done

  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" = "1" ] \
    || die "uvm_moe_slot_pte_remap_enable must be 1"
  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes)" -ge 4194304 ] \
    || die "uvm_moe_slot_pte_remap_max_bytes must be >= 4194304"
}

stop_gpu_monitor_services() {
  local service
  for service in qemu-guest-agent node-exporter gpu-exporter wazuh-agent vmagent; do
    if systemctl list-unit-files "$service.service" >/dev/null 2>&1 \
      || systemctl list-units --all "$service.service" >/dev/null 2>&1; then
      sudo systemctl stop "$service" 2>/dev/null || true
    fi
  done
}

run_prefight_smoke() {
  [ "$RUN_PREFLIGHT" = "1" ] || return 0
  mkdir -p "$RUN_ROOT/preflight"
  python3 "$SCRIPT_DIR/check_moe_slot_remap_ioctl_smoke.py" \
    --mode slot_va_block_remap \
    --count 2 \
    --source-kind hmm \
    --same-slot-ranges \
    --cross-va-block \
    --verify-gpu-read \
    --out-json "$RUN_ROOT/preflight/mode6_same_slot_cross_block_hmm.json"
}

summarize_case() {
  local case_dir="$RUN_ROOT/cold_streaming_hot1_stream268435456_kv7516192768_s8"
  echo
  echo "===== controlled remap debug summary ====="
  echo "run_root=$RUN_ROOT"
  echo "case_dir=$case_dir"

  if [ -f "$case_dir/vllm_server.log" ]; then
    echo
    echo "===== boundary markers ====="
    rg -n \
      "UVM controlled remap debug|UVM controlled remap boundary|Stage M init-time|controlled remap slot PTEs cleared|Skipping post-init|compile_or_warm_up_model|Started engine|Uvicorn|Application startup|ERROR|Traceback" \
      "$case_dir/vllm_server.log" | tail -160 || true
  fi

  if [ -f "$case_dir/vllm_uvm_cold_slot_pool_stage_m.jsonl" ]; then
    echo
    echo "===== slot trace counters ====="
    python3 - "$case_dir/vllm_uvm_cold_slot_pool_stage_m.jsonl" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]

def after(row):
    return ((row.get("driver_slot_remap_reset_after_remap_result") or {}).get("after") or {})

print(f"records={len(rows)}")
print(f"backend_ok={sum(1 for row in rows if row.get('backend_status') == 'ok')}")
print(f"remap_failures={sum(int(row.get('driver_slot_remap_failure_count', 0) or 0) for row in rows)}")
print(f"copy_failed={sum(int(row.get('driver_slot_remap_copy_failed_count', 0) or 0) for row in rows)}")
print(f"reset_ok={sum(1 for row in rows if (row.get('driver_slot_remap_reset_after_remap_result') or {}).get('ok') is True)}")
print(f"max_after_active_slots={max([int(after(row).get('active_slots', 0) or 0) for row in rows] or [0])}")
print(f"max_after_residual_va_blocks={max([int(after(row).get('pte_residual_va_block_count', 0) or 0) for row in rows] or [0])}")
print(f"max_after_residual_pages={max([int(after(row).get('pte_residual_page_count', 0) or 0) for row in rows] or [0])}")
PY
  fi

  echo
  echo "===== D-state threads ====="
  ps -eLo pid,tid,ppid,stat,wchan:40,comm,args \
    | awk '$4 ~ /D/ || /VLLM|vllm|nvidia-smi|qemu-guest-agent/ {print}' \
    | head -120 || true
}

main() {
  mkdir -p "$RUN_ROOT"
  stop_gpu_monitor_services
  check_no_d_state_processes
  check_module_state
  run_prefight_smoke

  set +e
  RUN_ROOT="$RUN_ROOT" \
  MODEL="$MODEL" \
  PLAN_PATH="$PLAN_PATH" \
  HOT_K=1 \
  KV_BYTES=7516192768 \
  STREAMING_BYTES_LIST=268435456 \
  MAX_NUM_SEQS_LIST=8 \
  MAX_MODEL_LEN=1024 \
  PROMPTS=2 \
  OUTPUT_LEN=2 \
  REQUEST_RATE=2 \
  BASE_PORT="$BASE_PORT" \
  GPU_MEMORY_UTILIZATION=0.95 \
  STARTUP_TIMEOUT="$STARTUP_TIMEOUT" \
  RUN_FAULT_ONLY=0 \
  RUN_COLD_STREAMING=1 \
  RUN_GLOBAL_UVM_REFERENCE=0 \
  NO_CLEANUP=1 \
  COLD_SLOT_POOL_ENABLE=1 \
  COLD_SLOT_POOL_TRACE_ONLY=0 \
  COLD_SLOT_POOL_BACKEND=driver_slot_remap \
  COLD_SLOT_POOL_SLOTS_PER_LAYER=4 \
  COLD_SLOT_POOL_OVERFLOW_MODE=trace_only \
  COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT=16 \
  COLD_SLOT_POOL_SYNC_MIGRATION=0 \
  COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap \
  COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED=1 \
  COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP=auto \
  SKIP_COLD_CHECKER=0 \
  SKIP_SLOT_POOL_CHECKER=0 \
  INIT_COLD_OFFLOAD_MODE=offload_cpu \
  HOT_RESIDENCY_MIN_RATIO=0.80 \
  UVM_MOE_SLOT_PTE_REMAP_ENABLE=1 \
  UVM_MOE_SLOT_PTE_REMAP_MAX_BYTES=4194304 \
  UVM_MOE_SLOT_POOL_VIDMEM_BUDGET_BYTES=1073741824 \
  VLLM_UVM_ENABLE_BUILTIN_TESTS=1 \
  VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 \
  CONTROLLED_REMAP_PREFLIGHT=1 \
  CLEANUP_GPU_DISABLE_NVIDIA_SMI=1 \
  UVM_MOE_ALLOW_NVIDIA_SMI_UUID=0 \
  CLEANUP_GPU_NVIDIA_SMI_TIMEOUT=1 \
  UVM_MOE_NVIDIA_SMI_TIMEOUT=1 \
  EXTRA_SERVER_ENV="VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1" \
  "$SCRIPT_DIR/run_stage_m_cold_slot_pool_driver_iface_sanity.sh" \
    2>&1 | tee "$RUN_ROOT/run.log"
  status=${PIPESTATUS[0]}
  set -e

  summarize_case | tee "$RUN_ROOT/debug_summary.log"
  exit "$status"
}

main "$@"
