#!/usr/bin/env bash
# Phase U0 baseline/fix runner for unified HBM residency work.
#
# Default run covers the Phase U0 smoke gates that are quick enough for
# iteration:
#   1. built/loaded nvidia_uvm srcversion and parameter snapshot
#   2. Phase-N 3GiB rebalance smoke
#   3. real vLLM KV dynamic target 2 -> 4 -> 1
#   4. manual migration fault observe/fail-fast smoke
#
# Set RUN_LOWTRACE=1 to also rerun the four Qwen3 LOWTRACE baseline cases from
# workloads/vllm/vllm/uvm_test/LOWTRACE_FOURWAY_EXPERIMENT.md.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_DIR="${PARAM_DIR:-/sys/module/nvidia_uvm/parameters}"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_unified_hbm_phase0_$(date -u +%Y%m%d_%H%M%S)}"
RUN_STAGE_N="${RUN_STAGE_N:-1}"
RUN_KV_DYNAMIC="${RUN_KV_DYNAMIC:-1}"
RUN_MANUAL_FAULT="${RUN_MANUAL_FAULT:-1}"
RUN_LOWTRACE="${RUN_LOWTRACE:-0}"

KV_DYNAMIC_TIMEOUT="${KV_DYNAMIC_TIMEOUT:-900}"
MANUAL_RESET_UVM="${MANUAL_RESET_UVM:-1}"
LOWTRACE_TIMEOUT_SEC="${LOWTRACE_TIMEOUT_SEC:-1200}"
LOWTRACE_CONFIGURE_UVM_PARAMS="${LOWTRACE_CONFIGURE_UVM_PARAMS:-1}"
LOWTRACE_RESTORE_UVM_PARAMS="${LOWTRACE_RESTORE_UVM_PARAMS:-1}"
LOWTRACE_PTE_REMAP_MAX_BYTES="${LOWTRACE_PTE_REMAP_MAX_BYTES:-4194304}"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
PROMPT_EXACT="${PROMPT_EXACT:-Compare CUDA streams and events in one concise paragraph.}"
MAX_TOKENS_SEQUENCE="${MAX_TOKENS_SEQUENCE:-16,16,16,16,16,16,16,16}"

STATUS_TSV="$RUN_ROOT/status.tsv"
LOWTRACE_PARAM_NAMES=(
  uvm_moe_slot_pte_remap_enable
  uvm_moe_slot_pte_remap_max_bytes
)
declare -A LOWTRACE_ORIGINAL_PARAMS=()

mkdir -p "$RUN_ROOT"
: > "$STATUS_TSV"

log() {
  printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

snapshot_env() {
  {
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "run_root=$RUN_ROOT"
    echo "repo_root=$REPO_ROOT"
    echo "loaded_srcversion=$(cat /sys/module/nvidia_uvm/srcversion 2>/dev/null || true)"
    echo "built_srcversion=$(modinfo -F srcversion "$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko" 2>/dev/null || true)"
    echo "uname=$(uname -a)"
    echo
    echo "=== nvidia-smi ==="
    nvidia-smi || true
    echo
    echo "=== selected nvidia_uvm params ==="
    for path in \
      "$PARAM_DIR"/uvm_phase_n_* \
      "$PARAM_DIR"/uvm_manual_migration_fault_* \
      "$PARAM_DIR"/uvm_moe_slot_pte_remap_*; do
      [ -e "$path" ] || continue
      printf '%s=%s\n' "$(basename "$path")" "$(cat "$path" 2>/dev/null || true)"
    done
  } > "$RUN_ROOT/phase0_environment.txt"
}

run_case_shell() {
  local label="$1"
  local command="$2"
  local case_dir="$RUN_ROOT/$label"
  local rc

  mkdir -p "$case_dir"
  printf '%s\n' "$command" > "$case_dir/command.sh"
  log "RUN $label"
  set +e
  (
    cd "$REPO_ROOT" || exit 1
    bash -lc "$command"
  ) 2>&1 | tee "$case_dir/run.log"
  rc=${PIPESTATUS[0]}
  set -u
  printf '%s\n' "$rc" > "$case_dir/status.txt"
  printf '%s\t%s\t%s\n' "$label" "$rc" "$case_dir" >> "$STATUS_TSV"
  if [ "$rc" -eq 0 ]; then
    log "PASS $label"
  else
    log "FAIL $label rc=$rc"
  fi
  return 0
}

read_param_optional() {
  local name="$1"
  local path="$PARAM_DIR/$name"

  [ -e "$path" ] || return 1
  cat "$path"
}

write_param_required() {
  local name="$1"
  local value="$2"
  local path="$PARAM_DIR/$name"

  if [ ! -e "$path" ]; then
    echo "ERROR: missing UVM module parameter $path" >&2
    return 1
  fi
  if [ "$(cat "$path" 2>/dev/null || true)" = "$value" ]; then
    return 0
  fi
  echo "$value" | sudo tee "$path" >/dev/null
  [ "$(cat "$path" 2>/dev/null || true)" = "$value" ]
}

configure_lowtrace_uvm_params() {
  [ "$LOWTRACE_CONFIGURE_UVM_PARAMS" = "1" ] || return 0

  LOWTRACE_ORIGINAL_PARAMS[uvm_moe_slot_pte_remap_enable]="$(read_param_optional uvm_moe_slot_pte_remap_enable || true)"
  LOWTRACE_ORIGINAL_PARAMS[uvm_moe_slot_pte_remap_max_bytes]="$(read_param_optional uvm_moe_slot_pte_remap_max_bytes || true)"
  write_param_required uvm_moe_slot_pte_remap_enable 1 || return 1
  write_param_required uvm_moe_slot_pte_remap_max_bytes "$LOWTRACE_PTE_REMAP_MAX_BYTES" || return 1
}

restore_lowtrace_uvm_params() {
  local name
  local value

  [ "$LOWTRACE_RESTORE_UVM_PARAMS" = "1" ] || return 0
  for name in "${LOWTRACE_PARAM_NAMES[@]}"; do
    value="${LOWTRACE_ORIGINAL_PARAMS[$name]-}"
    if [ -n "$value" ] && [ -e "$PARAM_DIR/$name" ]; then
      echo "$value" | sudo tee "$PARAM_DIR/$name" >/dev/null 2>&1 || true
    fi
  done
}

trap restore_lowtrace_uvm_params EXIT

write_lowtrace_fixture() {
  local fixture="$1"

  python3 - "$fixture" "$PROMPT_EXACT" "$MAX_TOKENS_SEQUENCE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
prompt = sys.argv[2]
tokens = [
    int(item.strip())
    for item in sys.argv[3].replace(";", ",").split(",")
    if item.strip()
]
payload = {
    "metadata": {"source": "phase0_lowtrace_fourway"},
    "requests": [
        {"request": i, "prompt": f"{prompt} [request {i}]", "max_tokens": max_tokens}
        for i, max_tokens in enumerate(tokens)
    ],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY
}

run_lowtrace_fourway() {
  local lowtrace_root="$RUN_ROOT/lowtrace_fourway"
  local fixture="$lowtrace_root/request_fixture.json"

  mkdir -p "$lowtrace_root"
  configure_lowtrace_uvm_params || {
    printf '%s\t%s\t%s\n' "lowtrace_uvm_param_config" "1" "$lowtrace_root" >> "$STATUS_TSV"
    return 0
  }
  write_lowtrace_fixture "$fixture"

  run_case_shell \
    "lowtrace_baseline_expertkv_no_offload" \
    "cd workloads/vllm/vllm && mkdir -p '$lowtrace_root/baseline_expertkv_no_offload_lowtrace_warm' && VLLM_USE_UVM=0 VLLM_UVM_PRE_SERVICE_ALLOCATIONS_MANAGED=0 VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 VLLM_UVM_KV_CACHE_MANAGED=1 VLLM_DISABLE_OPENAI_METRICS=1 VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 timeout '${LOWTRACE_TIMEOUT_SEC}s' uv run python uvm_test/vllm_generate_sequence.py --model '$MODEL' --request-fixture-json '$fixture' --kv-cache-memory-bytes 1073741824 --gpu-memory-utilization 0.90 --output-json '$lowtrace_root/baseline_expertkv_no_offload_lowtrace_warm/result.json' --progress-jsonl '$lowtrace_root/baseline_expertkv_no_offload_lowtrace_warm/progress.jsonl'"

  run_case_shell \
    "lowtrace_dynamic_real_load_slot" \
    "RUN_ROOT='$lowtrace_root/dynamic_real_load_lowtrace_96slots' MODEL='$MODEL' PROMPT_EXACT='$PROMPT_EXACT' MAX_TOKENS_SEQUENCE='$MAX_TOKENS_SEQUENCE' PLAN_MAX_EXPERTS_PER_LAYER=2 RESERVED_SLOT_COUNT=96 RESERVED_BANK_COUNT=1 RESERVED_SLOT_BYTES=4194304 RESERVED_EXPERTS_PER_LAYER=2 RESERVED_CLEANUP_DELAY_STEPS=4 PREALLOC_RESERVED_BEFORE_RUN=0 POST_STAGE_M_RESERVED_PREALLOC=1 PRE_SERVICE_MANAGED=0 EXPERT_WEIGHTS_MANAGED=1 KV_CACHE_MANAGED=1 KV_CACHE_MEMORY_BYTES=1073741824 GPU_MEMORY_UTILIZATION=0.90 TIMEOUT_SEC='$LOWTRACE_TIMEOUT_SEC' bash workloads/vllm/vllm/uvm_test/run_vllm_pre_service_real_load_offload_perf.sh"

  run_case_shell \
    "lowtrace_static_tail5_0pct" \
    "RUN_ROOT='$lowtrace_root/static_tail5_0pct_lowtrace_96slots' MODEL='$MODEL' PROMPT_EXACT='$PROMPT_EXACT' MAX_TOKENS_SEQUENCE='$MAX_TOKENS_SEQUENCE' TAIL_EXPERT_IDS=123,124,125,126,127 BASE_EXPERT_IDS=0,1,2,3,4,5,6,7 TAIL_HIT_PROB=0 RESERVED_SLOT_COUNT=96 RESERVED_BANK_COUNT=1 RESERVED_SLOT_BYTES=4194304 RESERVED_EXPERTS_PER_LAYER=5 RESERVED_CLEANUP_DELAY_STEPS=4 PREALLOC_RESERVED_BEFORE_RUN=0 POST_STAGE_M_RESERVED_PREALLOC=1 PRE_SERVICE_MANAGED=0 EXPERT_WEIGHTS_MANAGED=1 KV_CACHE_MANAGED=1 WEIGHT_MAP_ENABLE=0 MOE_ADDRESS_MAP_ENABLE=0 ROUTING_TRACE_ENABLE=0 PREFETCH_TRACE_ENABLE=0 RESERVED_COVERAGE_TRACE_ENABLE=0 RESERVED_COVERAGE_RESIDENCY_QUERY=0 KV_CACHE_MEMORY_BYTES=1073741824 GPU_MEMORY_UTILIZATION=0.90 TIMEOUT_SEC='$LOWTRACE_TIMEOUT_SEC' bash workloads/vllm/vllm/uvm_test/run_vllm_tail5_static_0pct_diag.sh"

  run_case_shell \
    "lowtrace_static_tail5_2pct" \
    "RUN_ROOT='$lowtrace_root/static_tail5_2pct_lowtrace_96slots' MODEL='$MODEL' PROMPT_EXACT='$PROMPT_EXACT' MAX_TOKENS_SEQUENCE='$MAX_TOKENS_SEQUENCE' TAIL_EXPERT_IDS=123,124,125,126,127 BASE_EXPERT_IDS=0,1,2,3,4,5,6,7 TAIL_HIT_PROB=0.02 RESERVED_SLOT_COUNT=96 RESERVED_BANK_COUNT=1 RESERVED_SLOT_BYTES=4194304 RESERVED_EXPERTS_PER_LAYER=5 RESERVED_CLEANUP_DELAY_STEPS=4 PREALLOC_RESERVED_BEFORE_RUN=0 POST_STAGE_M_RESERVED_PREALLOC=1 PRE_SERVICE_MANAGED=0 EXPERT_WEIGHTS_MANAGED=1 KV_CACHE_MANAGED=1 WEIGHT_MAP_ENABLE=0 MOE_ADDRESS_MAP_ENABLE=0 ROUTING_TRACE_ENABLE=0 PREFETCH_TRACE_ENABLE=0 RESERVED_COVERAGE_TRACE_ENABLE=0 RESERVED_COVERAGE_RESIDENCY_QUERY=0 KV_CACHE_MEMORY_BYTES=1073741824 GPU_MEMORY_UTILIZATION=0.90 TIMEOUT_SEC='$LOWTRACE_TIMEOUT_SEC' bash workloads/vllm/vllm/uvm_test/run_vllm_tail5_static_0pct_diag.sh"

  restore_lowtrace_uvm_params
}

write_summary() {
  python3 - "$RUN_ROOT" "$STATUS_TSV" "$RUN_LOWTRACE" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
status_path = Path(sys.argv[2])
run_lowtrace = sys.argv[3] == "1"

cases = []
if status_path.is_file():
    for line in status_path.read_text(encoding="utf-8").splitlines():
        label, rc, case_dir = line.split("\t", 2)
        cases.append({"label": label, "returncode": int(rc), "dir": case_dir})

def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def find_case(label: str) -> dict:
    return next((case for case in cases if case["label"] == label), {})

stage_n_log = text(root / "stage_n_3gb_rebalance" / "smoke.log")
stage_n_final = ""
for line in stage_n_log.splitlines():
    if line.startswith("phase_n_3gb_final "):
        stage_n_final = line

stage_n_metrics = {}
if stage_n_final:
    for key, value in re.findall(r"(\w+)=([^\s]+)", stage_n_final):
        stage_n_metrics[key] = value

kv_summary_path = root / "kv_dynamic_pool_phase2_prefetch_real" / "kv_dynamic_pool_phase2_real_summary.json"
try:
    kv_summary = json.loads(kv_summary_path.read_text(encoding="utf-8"))
except Exception:
    kv_summary = {}

manual_summary = text(root / "manual_migration_fault_control" / "summary.txt")

summary = {
    "run_root": str(root),
    "environment": str(root / "phase0_environment.txt"),
    "lowtrace_enabled": run_lowtrace,
    "passed": all(case["returncode"] == 0 for case in cases),
    "cases": cases,
    "stage_n": {
        "passed": find_case("stage_n_3gb_rebalance").get("returncode") == 0,
        "final_line": stage_n_final,
        "metrics": stage_n_metrics,
    },
    "kv_dynamic": {
        "passed": bool(kv_summary.get("passed")),
        "summary_json": str(kv_summary_path),
        "physical_offload_records": kv_summary.get("physical_offload_records"),
        "physical_prefetch_records": kv_summary.get("physical_prefetch_records"),
        "ledger_estimated_kv_gpu_bytes": kv_summary.get("ledger_estimated_kv_gpu_bytes"),
        "real_residency_observation": kv_summary.get("real_residency_observation"),
    },
    "manual_fault": {
        "passed": "manual_migration_fault_control: PASS" in manual_summary,
        "summary": str(root / "manual_migration_fault_control" / "summary.txt"),
    },
}

summary_json = root / "phase0_summary.json"
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# Unified HBM Residency Phase U0",
    "",
    f"- run_root: `{root}`",
    f"- passed: `{summary['passed']}`",
    f"- lowtrace_enabled: `{run_lowtrace}`",
    f"- environment: `{summary['environment']}`",
    "",
    "| case | rc | dir |",
    "|---|---:|---|",
]
for case in cases:
    lines.append(f"| {case['label']} | {case['returncode']} | `{case['dir']}` |")

lines.extend([
    "",
    "## Key Results",
    "",
    f"- Stage-N final: `{stage_n_final or 'missing'}`",
    f"- KV dynamic passed: `{summary['kv_dynamic']['passed']}`",
    f"- KV dynamic ledger bytes: `{summary['kv_dynamic']['ledger_estimated_kv_gpu_bytes']}`",
    f"- KV physical offload/prefetch records: `{summary['kv_dynamic']['physical_offload_records']}` / `{summary['kv_dynamic']['physical_prefetch_records']}`",
    f"- manual fault smoke passed: `{summary['manual_fault']['passed']}`",
])
summary_md = root / "phase0_summary.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md)
PY
}

snapshot_env
log "RUN_ROOT=$RUN_ROOT"
log "Environment snapshot: $RUN_ROOT/phase0_environment.txt"

if [ "$RUN_STAGE_N" = "1" ]; then
  run_case_shell \
    "stage_n_3gb_rebalance" \
    "RUN_ROOT='$RUN_ROOT/stage_n_3gb_rebalance' RESTORE_UVM_PARAMS=1 bash workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh"
fi

if [ "$RUN_KV_DYNAMIC" = "1" ]; then
  run_case_shell \
    "kv_dynamic_pool_phase2_prefetch_real" \
    "python3 workloads/vllm/check_kv_dynamic_pool_phase2_prefetch_real.py --run-dir '$RUN_ROOT/kv_dynamic_pool_phase2_prefetch_real' --timeout-seconds '$KV_DYNAMIC_TIMEOUT'"
fi

if [ "$RUN_MANUAL_FAULT" = "1" ]; then
  run_case_shell \
    "manual_migration_fault_control" \
    "RUN_ROOT='$RUN_ROOT/manual_migration_fault_control' RESET_UVM='$MANUAL_RESET_UVM' bash workloads/vllm/run_manual_migration_fault_control_smoke.sh"
fi

if [ "$RUN_LOWTRACE" = "1" ]; then
  run_lowtrace_fourway
else
  printf 'lowtrace_fourway\tskipped\t%s\n' "$RUN_ROOT/lowtrace_fourway" > "$RUN_ROOT/lowtrace_skipped.tsv"
  log "SKIP lowtrace_fourway (set RUN_LOWTRACE=1 to run Qwen3 four-way baseline)"
fi

write_summary

if awk -F '\t' '{ if ($2 != "0") bad=1 } END { exit bad ? 1 : 0 }' "$STATUS_TSV"; then
  log "PASS phase0: $RUN_ROOT/phase0_summary.md"
  exit 0
fi

log "FAIL phase0: $RUN_ROOT/phase0_summary.md"
exit 1
