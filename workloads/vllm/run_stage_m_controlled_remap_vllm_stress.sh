#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-log/stage_m_controlled_remap_vllm_stress}"
PLAN_PATH="${PLAN_PATH:-log/controlled_remap_v2_tiny_vllm_e2e/minimal_train_static_hotset_k1.json}"
PLAN_LAYERS="${PLAN_LAYERS:-48}"
PLAN_EXPERTS="${PLAN_EXPERTS:-128}"
PLAN_HOT_EXPERTS_PER_LAYER="${PLAN_HOT_EXPERTS_PER_LAYER:-1}"
PLAN_COLD_EXPERTS_PER_LAYER="${PLAN_COLD_EXPERTS_PER_LAYER:-1}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
BASE_PORT="${BASE_PORT:-8110}"
REPEATS="${REPEATS:-2}"
NORMAL_VIDMEM_BUDGET_BYTES="${NORMAL_VIDMEM_BUDGET_BYTES:-1073741824}"
LOW_VIDMEM_BUDGET_BYTES="${LOW_VIDMEM_BUDGET_BYTES:-268435456}"
TINY_EVICT_BUDGET_BYTES="${TINY_EVICT_BUDGET_BYTES:-65536}"
STREAMING_BYTES="${STREAMING_BYTES:-268435456}"
KV_BYTES="${KV_BYTES:-7516192768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-2400}"
HOT_RESIDENCY_MIN_RATIO="${HOT_RESIDENCY_MIN_RATIO:-0.80}"
SLOTS_PER_LAYER="${SLOTS_PER_LAYER:-4}"
TRACE_ASSIGNMENTS_LIMIT="${TRACE_ASSIGNMENTS_LIMIT:-16}"
CHECK_MODULE_UNLOAD="${CHECK_MODULE_UNLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_D_STATE_PROCESSES="${ALLOW_D_STATE_PROCESSES:-0}"
RUN_CONTROLLED_SMOKE_PREFLIGHT="${RUN_CONTROLLED_SMOKE_PREFLIGHT:-1}"
RUN_NATIVE_SPLIT_PREFLIGHT="${RUN_NATIVE_SPLIT_PREFLIGHT:-auto}"
RUN_ACTIVE_EVICT_PREFLIGHT="${RUN_ACTIVE_EVICT_PREFLIGHT:-auto}"
RUN_RELEASE_PREFLIGHT="${RUN_RELEASE_PREFLIGHT:-1}"
STOP_GPU_MONITOR_SERVICES="${STOP_GPU_MONITOR_SERVICES:-1}"
GPU_MONITOR_SERVICES="${GPU_MONITOR_SERVICES:-qemu-guest-agent node-exporter gpu-exporter wazuh-agent}"
CONTROLLED_REMAP_ENABLE_BUILTIN_TESTS="${CONTROLLED_REMAP_ENABLE_BUILTIN_TESTS:-1}"
CONTROLLED_REMAP_REQUIRED_MAX_BYTES="${CONTROLLED_REMAP_REQUIRED_MAX_BYTES:-4194304}"
CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_PLACEMENT="${CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_PLACEMENT:-auto}"
CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_REMAP="${CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_REMAP:-auto}"

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

check_no_d_state_processes() {
  if [ "$ALLOW_D_STATE_PROCESSES" = "1" ] || [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local d_state
  d_state="$(ps -eLo pid,tid,ppid,stat,comm,wchan:32,args | awk '$4 ~ /D/ && $7 !~ /^\[/ {print}' | head -50)"
  if [ -n "$d_state" ]; then
    echo "D-state user threads detected:"
    echo "$d_state"
    return 1
  fi
}

stop_gpu_monitor_services() {
  if [ "$STOP_GPU_MONITOR_SERVICES" != "1" ] || [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local service
  for service in $GPU_MONITOR_SERVICES; do
    if systemctl list-unit-files "$service.service" >/dev/null 2>&1 \
      || systemctl list-units --all "$service.service" >/dev/null 2>&1; then
      sudo systemctl stop "$service" 2>/dev/null || true
    fi
  done
}

check_no_gpu_monitor_processes() {
  if [ "$ALLOW_D_STATE_PROCESSES" = "1" ] || [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local monitors
  monitors="$(ps -eo pid,ppid,stat,comm,wchan:32,args | awk '
    $4 ~ /nvidia-smi|wazuh-logcollec|node_exporter|nvidia_gpu_expo/ { print }
  ' | head -50)"
  if [ -n "$monitors" ]; then
    echo "GPU monitor processes still present after service stop:"
    echo "$monitors"
    return 1
  fi
}

query_slot_pool() {
  local out_json="$1"
  python3 "$SCRIPT_DIR/check_moe_slot_pool_state_smoke.py" \
    --dump-only \
    --expect-slot-pte-state \
    --max-entries 128 \
    --out-json "$out_json"
}

check_slot_pool_counters() {
  local query_json="$1"
  python3 - "$query_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
query = data.get("query", {})
checks = data.get("checks", {})
errors = []

if not data.get("passed"):
    failed = sorted(name for name, ok in checks.items() if not ok)
    errors.append(f"slot pool checker failed: {failed}")

for key in ("copy_failed_count", "pte_failure_count", "pte_dangerous_op_reject_count"):
    value = int(query.get(key, 0) or 0)
    if value != 0:
        errors.append(f"{key}={value}, expected 0")

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(1)

print(
    "slot_pool_ok",
    f"active_slots={query.get('active_slots')}",
    f"backed_slots={query.get('backed_slots')}",
    f"pte_remap_count={query.get('pte_remap_count')}",
    f"pte_clear_count={query.get('pte_clear_count')}",
    f"pte_split_clear_count={query.get('pte_split_clear_count')}",
    f"pte_release_clear_count={query.get('pte_release_clear_count')}",
    f"pte_evict_clear_count={query.get('pte_evict_clear_count')}",
    f"vidmem_evict_count={query.get('vidmem_evict_count')}",
)
PY
}

run_controlled_smoke() {
  local label="$1"
  shift
  local out_json="$RUN_ROOT/preflight/${label}.json"
  mkdir -p "$RUN_ROOT/preflight"

  echo "---- controlled remap preflight: $label ----"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py $* --out-json $out_json"
    return 0
  fi

  python3 "$SCRIPT_DIR/check_moe_slot_remap_ioctl_smoke.py" "$@" --out-json "$out_json" \
    2>&1 | tee "$RUN_ROOT/preflight/${label}.log"
}

uvm_builtin_tests_enabled() {
  local param="/sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests"
  [ -r "$param" ] && [ "$(cat "$param" 2>/dev/null || echo 0)" = "1" ]
}

read_uvm_module_param() {
  local name="$1"
  local path="/sys/module/nvidia_uvm/parameters/$name"

  [ -r "$path" ] || die "Missing nvidia_uvm module parameter: $name. Load the patched module first."
  cat "$path" 2>/dev/null || die "Failed to read nvidia_uvm module parameter: $name"
}

ensure_controlled_remap_module_state() {
  local pte_enable pte_max vidmem_budget

  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  pte_enable="$(read_uvm_module_param uvm_moe_slot_pte_remap_enable)"
  [ "$pte_enable" = "1" ] \
    || die "uvm_moe_slot_pte_remap_enable must be 1 for controlled remap stress, got $pte_enable"

  pte_max="$(read_uvm_module_param uvm_moe_slot_pte_remap_max_bytes)"
  [ "$pte_max" -ge "$CONTROLLED_REMAP_REQUIRED_MAX_BYTES" ] \
    || die "uvm_moe_slot_pte_remap_max_bytes=$pte_max is smaller than required $CONTROLLED_REMAP_REQUIRED_MAX_BYTES"

  vidmem_budget="$(read_uvm_module_param uvm_moe_slot_pool_vidmem_budget_bytes)"
  [ "$vidmem_budget" -gt 0 ] \
    || die "uvm_moe_slot_pool_vidmem_budget_bytes must be > 0 for controlled remap stress, got $vidmem_budget"
}

plan_file_usable() {
  local plan_path="$1"
  python3 - "$plan_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

if not isinstance(data, dict):
    raise SystemExit(1)

prefetch = data.get("prefetch_plan")
offload = data.get("offload_plan")
if not isinstance(prefetch, list) or not isinstance(offload, list):
    raise SystemExit(1)
if len(prefetch) == 0 or len(offload) == 0:
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
        "generated_by": "run_stage_m_controlled_remap_vllm_stress.sh",
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

run_controlled_smoke_preflight() {
  if [ "$RUN_CONTROLLED_SMOKE_PREFLIGHT" != "1" ]; then
    return 0
  fi

  echo "==========================================================="
  echo "controlled_remap_vllm_stress preflight"
  echo "==========================================================="
  check_no_d_state_processes

  run_controlled_smoke mode6_multi \
    --mode slot_va_block_remap \
    --count 2 \
    --validate-managed \
    --verify-gpu-read

  run_controlled_smoke mode6_native_migrate_clear \
    --mode slot_va_block_remap \
    --count 1 \
    --validate-managed \
    --verify-gpu-read \
    --clear-via-native-migrate

  case "$RUN_RELEASE_PREFLIGHT" in
    1)
      run_controlled_smoke mode6_release_clear \
        --mode slot_va_block_remap \
        --count 1 \
        --validate-managed \
        --verify-gpu-read \
        --clear-via-release
      ;;
    0)
      echo "Skipping release preflight: RUN_RELEASE_PREFLIGHT=0"
      ;;
    *)
      die "RUN_RELEASE_PREFLIGHT must be 0 or 1"
      ;;
  esac

  case "$RUN_ACTIVE_EVICT_PREFLIGHT" in
    1)
      run_controlled_smoke mode6_active_evict_clear \
        --mode slot_va_block_remap \
        --count 1 \
        --validate-managed \
        --verify-gpu-read \
        --evict-active-via-gpu-copy \
        --evict-active-temp-budget "$TINY_EVICT_BUDGET_BYTES"
      ;;
    0)
      echo "Skipping active eviction preflight: RUN_ACTIVE_EVICT_PREFLIGHT=0"
      ;;
    auto)
      run_controlled_smoke mode6_active_evict_clear \
        --mode slot_va_block_remap \
        --count 1 \
        --validate-managed \
        --verify-gpu-read \
        --evict-active-via-gpu-copy \
        --evict-active-temp-budget "$TINY_EVICT_BUDGET_BYTES"
      ;;
    *)
      die "RUN_ACTIVE_EVICT_PREFLIGHT must be 0, 1, or auto"
      ;;
  esac

  case "$RUN_NATIVE_SPLIT_PREFLIGHT" in
    1)
      ;;
    0)
      echo "Skipping native split preflight: RUN_NATIVE_SPLIT_PREFLIGHT=0"
      return 0
      ;;
    auto)
      if ! uvm_builtin_tests_enabled; then
        echo "Skipping native split preflight: uvm_enable_builtin_tests is not 1"
        return 0
      fi
      ;;
    *)
      die "RUN_NATIVE_SPLIT_PREFLIGHT must be 0, 1, or auto"
      ;;
  esac

  run_controlled_smoke mode6_native_split_clear \
    --mode slot_va_block_remap \
    --count 1 \
    --validate-managed \
    --verify-gpu-read \
    --clear-via-native-split
}

run_case() {
  local repeat="$1"
  local label="$2"
  local prompts="$3"
  local output_len="$4"
  local max_num_seqs="$5"
  local vidmem_budget="$6"
  local port="$7"
  local case_root="$RUN_ROOT/repeat_${repeat}/${label}"

  mkdir -p "$case_root"

  echo "==========================================================="
  echo "controlled_remap_vllm_stress repeat=$repeat case=$label"
  echo "prompts=$prompts output_len=$output_len max_num_seqs=$max_num_seqs vidmem_budget=$vidmem_budget port=$port"
  echo "case_root=$case_root"
  echo "==========================================================="

  check_no_d_state_processes

  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] would run vLLM controlled remap case"
    return 0
  fi

  RUN_ROOT="$case_root/vllm" \
  MODEL="$MODEL" \
  PLAN_PATH="$PLAN_PATH" \
  HOT_K=1 \
  KV_BYTES="$KV_BYTES" \
  STREAMING_BYTES_LIST="$STREAMING_BYTES" \
  MAX_NUM_SEQS_LIST="$max_num_seqs" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  PROMPTS="$prompts" \
  OUTPUT_LEN="$output_len" \
  REQUEST_RATE="$prompts" \
  BASE_PORT="$port" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  STARTUP_TIMEOUT="$STARTUP_TIMEOUT" \
  RUN_FAULT_ONLY=0 \
  RUN_COLD_STREAMING=1 \
  RUN_GLOBAL_UVM_REFERENCE=0 \
  NO_CLEANUP=1 \
  COLD_SLOT_POOL_ENABLE=1 \
  COLD_SLOT_POOL_TRACE_ONLY=0 \
  COLD_SLOT_POOL_BACKEND=driver_slot_remap \
  COLD_SLOT_POOL_SLOTS_PER_LAYER="$SLOTS_PER_LAYER" \
  COLD_SLOT_POOL_OVERFLOW_MODE=trace_only \
  COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT="$TRACE_ASSIGNMENTS_LIMIT" \
  COLD_SLOT_POOL_SYNC_MIGRATION=0 \
  COLD_SLOT_POOL_DRIVER_MODE=slot_va_block_remap \
  COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED=1 \
  COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP="$CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_REMAP" \
  UVM_INIT_WEIGHT_CLEAR_SLOT_PTES_AFTER_PLACEMENT="$CONTROLLED_REMAP_CLEAR_SLOT_PTES_AFTER_PLACEMENT" \
  SKIP_COLD_CHECKER=0 \
  SKIP_SLOT_POOL_CHECKER=0 \
  INIT_COLD_OFFLOAD_MODE=offload_cpu \
  HOT_RESIDENCY_MIN_RATIO="$HOT_RESIDENCY_MIN_RATIO" \
  UVM_MOE_SLOT_PTE_REMAP_ENABLE=1 \
  UVM_MOE_SLOT_PTE_REMAP_MAX_BYTES=4194304 \
  UVM_MOE_SLOT_POOL_VIDMEM_BUDGET_BYTES="$vidmem_budget" \
  VLLM_UVM_ENABLE_BUILTIN_TESTS="$CONTROLLED_REMAP_ENABLE_BUILTIN_TESTS" \
  VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 \
  EXTRA_SERVER_ENV="VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0 ${EXTRA_SERVER_ENV:-}" \
  CONTROLLED_REMAP_PREFLIGHT=1 \
  CLEANUP_GPU_DISABLE_NVIDIA_SMI=1 \
  UVM_MOE_ALLOW_NVIDIA_SMI_UUID=0 \
  CLEANUP_GPU_NVIDIA_SMI_TIMEOUT=1 \
  UVM_MOE_NVIDIA_SMI_TIMEOUT=1 \
  "$SCRIPT_DIR/run_stage_m_cold_slot_pool_driver_iface_sanity.sh" \
    2>&1 | tee "$case_root/run.log"

  check_no_d_state_processes

  local query_json="$case_root/slot_pool_after_case.json"
  query_slot_pool "$query_json" 2>&1 | tee "$case_root/slot_pool_after_case.log"
  check_slot_pool_counters "$query_json" 2>&1 | tee "$case_root/counter_check.log"

  if [ "$CHECK_MODULE_UNLOAD" = "1" ]; then
    sudo rmmod nvidia_uvm
    echo "module_unload_ok" | tee "$case_root/module_unload.log"
  fi
}

main() {
  [ "$REPEATS" -ge 1 ] || die "REPEATS must be >= 1"
  ensure_minimal_plan "$PLAN_PATH"
  PLAN_PATH="$(abs_path "$PLAN_PATH")"
  mkdir -p "$RUN_ROOT"
  stop_gpu_monitor_services
  check_no_d_state_processes
  check_no_gpu_monitor_processes
  ensure_controlled_remap_module_state
  run_controlled_smoke_preflight

  local repeat port
  for repeat in $(seq 1 "$REPEATS"); do
    port=$((BASE_PORT + (repeat - 1) * 10))
    run_case "$repeat" "p2_o2_s8_budget_normal" 2 2 8 "$NORMAL_VIDMEM_BUDGET_BYTES" "$port"
    run_case "$repeat" "p8_o2_s8_budget_normal" 8 2 8 "$NORMAL_VIDMEM_BUDGET_BYTES" "$((port + 1))"
    run_case "$repeat" "p8_o2_s16_budget_normal" 8 2 16 "$NORMAL_VIDMEM_BUDGET_BYTES" "$((port + 2))"
    run_case "$repeat" "p2_o2_s8_budget_low" 2 2 8 "$LOW_VIDMEM_BUDGET_BYTES" "$((port + 3))"
  done

  echo "controlled_remap_vllm_stress PASS"
  echo "run_root=$RUN_ROOT"
}

main "$@"
