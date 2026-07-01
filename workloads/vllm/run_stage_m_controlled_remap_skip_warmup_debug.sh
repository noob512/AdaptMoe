#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-log/current_controlled_remap_skip_warmup_debug}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PLAN_PATH="${PLAN_PATH:-log/controlled_remap_v2_tiny_vllm_e2e/minimal_train_static_hotset_k1.json}"
PLAN_LAYERS="${PLAN_LAYERS:-48}"
PLAN_EXPERTS="${PLAN_EXPERTS:-128}"
PLAN_HOT_EXPERTS_PER_LAYER="${PLAN_HOT_EXPERTS_PER_LAYER:-1}"
PLAN_COLD_EXPERTS_PER_LAYER="${PLAN_COLD_EXPERTS_PER_LAYER:-1}"
BASE_PORT="${BASE_PORT:-8120}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
BUILT_SRCVERSION="$(modinfo -F srcversion "$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
EXPECTED_SRCVERSION="${EXPECTED_SRCVERSION:-$BUILT_SRCVERSION}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
EXPECTED_FAULT_BATCH_COUNT="${EXPECTED_FAULT_BATCH_COUNT:-32}"
EXPECTED_FAULT_MAX_BATCHES_PER_SERVICE="${EXPECTED_FAULT_MAX_BATCHES_PER_SERVICE:-4}"
EXPECTED_FAULT_MMAP_LOCK_YIELD_INTERVAL="${EXPECTED_FAULT_MMAP_LOCK_YIELD_INTERVAL:-4}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

plan_file_usable() {
  local plan_path="$1"
  python3 - "$plan_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    plan = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

metadata = plan.get("metadata") if isinstance(plan, dict) else None
if not isinstance(metadata, dict):
    raise SystemExit(1)
if metadata.get("policy") != "train_test_static_hotset_v1":
    raise SystemExit(1)
if not isinstance(plan.get("offload_plan"), list):
    raise SystemExit(1)
if not isinstance(plan.get("prefetch_plan"), list):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

ensure_minimal_plan() {
  local plan_path="$1"
  local plan_dir

  if [ -s "$plan_path" ] && plan_file_usable "$plan_path"; then
    return 0
  fi

  plan_dir="$(dirname "$plan_path")"
  mkdir -p "$plan_dir"

  python3 - "$plan_path" "$PLAN_LAYERS" "$PLAN_EXPERTS" "$PLAN_HOT_EXPERTS_PER_LAYER" "$PLAN_COLD_EXPERTS_PER_LAYER" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
num_layers = int(sys.argv[2])
num_experts = int(sys.argv[3])
hot_per_layer = max(0, min(int(sys.argv[4]), num_experts))
cold_per_layer = max(0, min(int(sys.argv[5]), num_experts))
prefetch_plan = []
offload_plan = []

for layer_id in range(num_layers):
    for expert_id in range(hot_per_layer):
        prefetch_plan.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "score": 0,
            "reason": "minimal_hot",
        })
    for expert_id in range(hot_per_layer, hot_per_layer + cold_per_layer):
        if expert_id >= num_experts:
            break
        offload_plan.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "score": 0,
            "reason": "minimal_cold",
        })

plan = {
    "metadata": {
        "policy": "train_test_static_hotset_v1",
        "generated_by": "run_stage_m_controlled_remap_skip_warmup_debug.sh",
        "num_layers": num_layers,
        "num_experts": num_experts,
        "hot_experts_per_layer": hot_per_layer,
        "cold_experts_per_layer": cold_per_layer,
        "target_roles": ["moe_gate_up", "moe_down"],
    },
    "prefetch_plan": prefetch_plan,
    "offload_plan": offload_plan,
    "prefetch_plan_records": len(prefetch_plan),
    "offload_plan_records": len(offload_plan),
}

path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
PY

  if ! plan_file_usable "$plan_path"; then
    die "failed to create usable plan file: $plan_path"
  fi
}

check_no_d_state_processes() {
  local d_state
  d_state="$(timeout 5s ps -eLo pid,tid,ppid,stat,wchan:40,comm,args \
    | awk '$4 ~ /D/ && $3 != 2 {print}' | head -50 || true)"
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
    uvm_perf_fault_batch_count \
    uvm_perf_fault_max_batches_per_service \
    uvm_perf_fault_mmap_lock_yield_interval \
    uvm_enable_builtin_tests; do
    value="$(cat "/sys/module/nvidia_uvm/parameters/$name" 2>/dev/null || true)"
    echo "$name=$value"
  done

  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_enable)" = "1" ] \
    || die "uvm_moe_slot_pte_remap_enable must be 1"
  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_moe_slot_pte_remap_max_bytes)" -ge 4194304 ] \
    || die "uvm_moe_slot_pte_remap_max_bytes must be >= 4194304"
  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_perf_fault_batch_count 2>/dev/null || echo 0)" -le "$EXPECTED_FAULT_BATCH_COUNT" ] \
    || die "uvm_perf_fault_batch_count must be <= $EXPECTED_FAULT_BATCH_COUNT"
  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_perf_fault_max_batches_per_service 2>/dev/null || echo 0)" -le "$EXPECTED_FAULT_MAX_BATCHES_PER_SERVICE" ] \
    || die "uvm_perf_fault_max_batches_per_service must be <= $EXPECTED_FAULT_MAX_BATCHES_PER_SERVICE"
  [ "$(cat /sys/module/nvidia_uvm/parameters/uvm_perf_fault_mmap_lock_yield_interval 2>/dev/null || echo 0)" -le "$EXPECTED_FAULT_MMAP_LOCK_YIELD_INTERVAL" ] \
    || die "uvm_perf_fault_mmap_lock_yield_interval must be <= $EXPECTED_FAULT_MMAP_LOCK_YIELD_INTERVAL"
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

run_preflight_smoke() {
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

  if [ -f "$case_dir/vllm_uvm_init_weight_offload_stage_m.jsonl" ]; then
    echo
    echo "===== init weight plan residency ====="
    python3 - "$case_dir/vllm_uvm_init_weight_offload_stage_m.jsonl" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
snapshots = [
    row
    for row in rows
    if row.get("action") == "init_weight_plan_residency_snapshot"
]
by_marker = defaultdict(list)
for row in snapshots:
    by_marker[str(row.get("marker") or "")].append(row)

print(f"snapshots={len(snapshots)}")
for marker in ("after_load_model", "before_stage_m", "after_stage_m_clear"):
    marker_rows = by_marker.get(marker, [])
    print(f"marker={marker} layers={len(marker_rows)}")
    for plan in ("hot_plan", "cold_plan"):
        queried = sum(int(row.get(f"{plan}_queried_bytes") or 0) for row in marker_rows)
        gpu = sum(int(row.get(f"{plan}_gpu_bytes") or 0) for row in marker_rows)
        cpu = sum(int(row.get(f"{plan}_cpu_bytes") or 0) for row in marker_rows)
        unknown = sum(int(row.get(f"{plan}_unknown_bytes") or 0) for row in marker_rows)
        managed = sum(int(row.get(f"{plan}_managed_records") or 0) for row in marker_rows)
        non_managed = sum(int(row.get(f"{plan}_non_managed_records") or 0) for row in marker_rows)
        ratio = (gpu / queried) if queried else 0.0
        cpu_ratio = (cpu / queried) if queried else 0.0
        unknown_ratio = (unknown / queried) if queried else 0.0
        print(
            f"  {plan}: managed_records={managed} non_managed_records={non_managed} "
            f"queried={queried} gpu={gpu} cpu={cpu} unknown={unknown} "
            f"gpu_ratio={ratio:.4f} cpu_ratio={cpu_ratio:.4f} unknown_ratio={unknown_ratio:.4f}"
        )
PY
  fi

  echo
  echo "===== D-state threads ====="
  timeout 5s ps -eLo pid,tid,ppid,stat,wchan:40,comm,args \
    | awk '$4 ~ /D/ || /VLLM|vllm|nvidia-smi|qemu-guest-agent/ {print}' \
    | head -120 || true
}

main() {
  mkdir -p "$RUN_ROOT"
  stop_gpu_monitor_services
  check_no_d_state_processes
  check_module_state
  ensure_minimal_plan "$PLAN_PATH"
  PLAN_PATH="$(abs_path "$PLAN_PATH")"
  run_preflight_smoke

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
  EXTRA_SERVER_ENV="VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 VLLM_UVM_CONTROLLED_REMAP_EARLY_STRUCTURED_OUTPUT_MANAGER=1 VLLM_UVM_CONTROLLED_REMAP_SYNC_AFTER_CLEAR=1 VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE=1" \
  "$SCRIPT_DIR/run_stage_m_cold_slot_pool_driver_iface_sanity.sh" \
    2>&1 | tee "$RUN_ROOT/run.log"
  status=${PIPESTATUS[0]}
  set -e

  summarize_case | tee "$RUN_ROOT/debug_summary.log"
  exit "$status"
}

main "$@"
