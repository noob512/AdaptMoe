#!/usr/bin/env bash
set -euo pipefail

# Compare two matched Stage-M expert residency modes:
#   A. fault_only:
#      KV and non-expert fixed weights are native/device-direct; MoE experts are
#      UVM managed; hotK experts are prefetched to GPU; non-hot experts are
#      offloaded to CPU at init; active cold experts are served by ordinary UVM
#      replayable faults.
#   B. cold_streaming:
#      Same initial layout, but active cold experts are explicitly prefetched
#      into an application-level streaming pool after router topk is known.
#
# Optional C. global_uvm_reference:
#      Original-style global UVM page-fault behavior without fixed native weights
#      or hot/cold expert residency. This is a useful reference, but it is not
#      the fairness baseline for isolating cold streaming.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_cold_streaming_vs_fault_only_ab}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
TRACE_PATH="${TRACE_PATH:-log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl}"
PLAN_PATH="${PLAN_PATH:-}"
HOT_K="${HOT_K:-64}"
KV_BYTES="${KV_BYTES:-7516192768}"
STREAMING_BYTES_LIST="${STREAMING_BYTES_LIST:-1073741824 2147483648 3221225472}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-16 32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PROMPTS="${PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
REQUEST_RATE="${REQUEST_RATE:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
BASE_PORT="${BASE_PORT:-8060}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
STREAMING_TTL_STEPS="${STREAMING_TTL_STEPS:-32}"
STREAMING_MAX_BYTES_PER_STEP="${STREAMING_MAX_BYTES_PER_STEP:-}"
STREAMING_MAX_EXPERTS_PER_LAYER="${STREAMING_MAX_EXPERTS_PER_LAYER:-0}"
STREAMING_SYNC_PREFETCH="${STREAMING_SYNC_PREFETCH:-0}"
STREAMING_SYNC_EVICT="${STREAMING_SYNC_EVICT:-1}"
STREAMING_REAL_QUERY_ENABLE="${STREAMING_REAL_QUERY_ENABLE:-0}"
EXTRA_SERVER_ENV="${EXTRA_SERVER_ENV:-}"
SKIP_COLD_CHECKER="${SKIP_COLD_CHECKER:-0}"
COLD_SLOT_POOL_ENABLE="${COLD_SLOT_POOL_ENABLE:-0}"
COLD_SLOT_POOL_TRACE_ONLY="${COLD_SLOT_POOL_TRACE_ONLY:-1}"
COLD_SLOT_POOL_BACKEND="${COLD_SLOT_POOL_BACKEND:-trace_only}"
COLD_SLOT_POOL_SLOTS_PER_LAYER="${COLD_SLOT_POOL_SLOTS_PER_LAYER:-40}"
COLD_SLOT_POOL_OVERFLOW_MODE="${COLD_SLOT_POOL_OVERFLOW_MODE:-trace_only}"
COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT="${COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT:-128}"
COLD_SLOT_POOL_SYNC_MIGRATION="${COLD_SLOT_POOL_SYNC_MIGRATION:-0}"
COLD_SLOT_POOL_DRIVER_MODE="${COLD_SLOT_POOL_DRIVER_MODE:-dry_run}"
COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED="${COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED:-0}"
COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP="${COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP:-auto}"
KV_DEVICE_DIRECT_ENABLE="${KV_DEVICE_DIRECT_ENABLE:-}"
KV_DEVICE_DIRECT_MIN_BYTES="${KV_DEVICE_DIRECT_MIN_BYTES:-0}"
KV_DEVICE_DIRECT_MAX_BYTES="${KV_DEVICE_DIRECT_MAX_BYTES:-0}"
KV_DEVICE_DIRECT_MAX_TOTAL_BYTES="${KV_DEVICE_DIRECT_MAX_TOTAL_BYTES:-0}"
FIXED_WEIGHT_DEVICE_DIRECT_ENABLE="${FIXED_WEIGHT_DEVICE_DIRECT_ENABLE:-}"
SKIP_SLOT_POOL_CHECKER="${SKIP_SLOT_POOL_CHECKER:-0}"
INIT_COLD_OFFLOAD_MODE="${INIT_COLD_OFFLOAD_MODE:-offload_cpu}"
INIT_COLD_OFFLOAD_SYNC="${INIT_COLD_OFFLOAD_SYNC:-1}"
HOT_RESIDENCY_MIN_RATIO="${HOT_RESIDENCY_MIN_RATIO:-0.90}"
FAULT_ADDRESS_LOG="${FAULT_ADDRESS_LOG:-0}"
RUN_FAULT_ONLY="${RUN_FAULT_ONLY:-1}"
RUN_COLD_STREAMING="${RUN_COLD_STREAMING:-1}"
RUN_GLOBAL_UVM_REFERENCE="${RUN_GLOBAL_UVM_REFERENCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
NO_CLEANUP="${NO_CLEANUP:-0}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"
COMMANDS_LOG="$RUN_ROOT/commands.log"
: > "$COMMANDS_LOG"

if [ -z "$KV_DEVICE_DIRECT_ENABLE" ]; then
  case "$COLD_SLOT_POOL_DRIVER_MODE" in
    slot_pte_remap|slot_va_block_remap)
      # Controlled remap targets read-only MoE weights. Keep KV cache on the
      # managed path by default so vLLM E2E does not mix slot PTE validation
      # with large native cudaMalloc/RM allocation pressure.
      KV_DEVICE_DIRECT_ENABLE=0
      ;;
    *)
      KV_DEVICE_DIRECT_ENABLE=1
      ;;
  esac
fi

if [ -z "$FIXED_WEIGHT_DEVICE_DIRECT_ENABLE" ]; then
  case "$COLD_SLOT_POOL_DRIVER_MODE" in
    slot_pte_remap|slot_va_block_remap)
      FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=0
      ;;
    *)
      FIXED_WEIGHT_DEVICE_DIRECT_ENABLE=1
      ;;
  esac
fi

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

append_command() {
  printf 'command:' >> "$COMMANDS_LOG"
  printf ' %q' "$@" >> "$COMMANDS_LOG"
  printf '\n' >> "$COMMANDS_LOG"
}

run_logged() {
  local log_file="$1"
  shift
  append_command "$@"
  if [ "$DRY_RUN" = "1" ]; then
    printf '[dry-run]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  mkdir -p "$(dirname "$log_file")"
  set +e
  "$@" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

run_cmd() {
  append_command "$@"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

echo "==========================================================="
echo "Stage M Cold Streaming vs UVM Fault-only A/B"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL hotK=$HOT_K kv_bytes=$KV_BYTES"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "streaming_bytes=$STREAMING_BYTES_LIST fault_address_log=$FAULT_ADDRESS_LOG"
echo "==========================================================="

if [ -z "$PLAN_PATH" ]; then
  if [ ! -s "$TRACE_PATH" ]; then
    echo "ERROR: TRACE_PATH not found or empty: $TRACE_PATH" >&2
    exit 2
  fi
  plan_dir="$RUN_ROOT/plan"
  run_cmd python3 "$SCRIPT_DIR/analyze_moe_hotset_train_test.py" \
    --trace "$TRACE_PATH" \
    --out-dir "$plan_dir" \
    --k-values "$HOT_K" \
    --split-fraction 0.5 \
    --score-mode weighted \
    --expert-bytes 4718592 \
    --write-plans
  PLAN_PATH="$(abs_path "$plan_dir/plans/train_static_hotset_k${HOT_K}.json")"
else
  PLAN_PATH="$(abs_path "$PLAN_PATH")"
fi

if [ "$DRY_RUN" != "1" ] && [ ! -s "$PLAN_PATH" ]; then
  echo "ERROR: hot plan not found: $PLAN_PATH" >&2
  exit 2
fi

make_common_extra_env() {
  local driver_tag_enable="$1"
  local env_string
  env_string="HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
VLLM_UVM_DRIVER_TAG_ENABLE=$driver_tag_enable \
VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=$driver_tag_enable \
VLLM_UVM_DRIVER_POOL_BUDGET_MODE=trace_only \
VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
VLLM_UVM_WEIGHT_MAP_QUERY_EXPERT_SLICES=1 \
VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=0"
  if [ -n "$EXTRA_SERVER_ENV" ]; then
    env_string="$env_string $EXTRA_SERVER_ENV"
  fi
  printf '%s' "$env_string"
}

classify_faults_if_requested() {
  local case_dir="$1"
  if [ "$FAULT_ADDRESS_LOG" != "1" ]; then
    return 0
  fi
  if [ ! -s "$case_dir/uvm_fault_addrs.log" ]; then
    return 0
  fi
  mkdir -p "$case_dir/fault_address_analysis"
  run_cmd python3 "$SCRIPT_DIR/analyze_uvm_fault_addresses.py" \
    --address-log "$case_dir/vllm_uvm_address_regions.log" \
    --fault-log "$case_dir/uvm_fault_addrs.log" \
    --stats-log "$case_dir/uvm_fault_stats.log" \
    --dedup-mode global-page \
    --summary-json "$case_dir/fault_address_analysis/summary.json" \
    --regions-csv "$case_dir/fault_address_analysis/regions.csv" \
    --faults-csv "$case_dir/fault_address_analysis/faults.csv" \
    --top-n 20 > "$case_dir/fault_address_analysis/report.txt" || true
}

run_matched_case() {
  local mode_name="$1"
  local streaming_bytes="$2"
  local max_num_seqs="$3"
  local port="$4"
  local case_dir="$RUN_ROOT/${mode_name}_hot${HOT_K}_stream${streaming_bytes}_kv${KV_BYTES}_s${max_num_seqs}"
  mkdir -p "$case_dir"

  local fixed_trace="$case_dir/vllm_uvm_fixed_weight_device_direct.jsonl"
  local weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"
  local init_trace="$case_dir/vllm_uvm_init_weight_offload_stage_m.jsonl"
  local cold_trace="$case_dir/vllm_uvm_cold_streaming_stage_m.jsonl"
  local slot_pool_trace="$case_dir/vllm_uvm_cold_slot_pool_stage_m.jsonl"
  local residency_trace="$case_dir/vllm_uvm_pool_residency_stage_m.jsonl"
  local extra_env
  extra_env="$(make_common_extra_env 1)"

  local cold_streaming_enable=0
  if [ "$mode_name" = "cold_streaming" ]; then
    cold_streaming_enable=1
  fi
  local max_bytes_per_step="$streaming_bytes"
  if [ -n "$STREAMING_MAX_BYTES_PER_STEP" ]; then
    max_bytes_per_step="$STREAMING_MAX_BYTES_PER_STEP"
  fi

  local -a address_args=()
  if [ "$FAULT_ADDRESS_LOG" = "1" ]; then
    address_args+=(--with-address-log --address-trace-log "$case_dir/uvm_fault_addrs.log")
  fi
  local -a cleanup_args=()
  if [ "$NO_CLEANUP" = "1" ]; then
    cleanup_args+=(--no-cleanup)
  fi

  local -a cmd=(
    "$SCRIPT_DIR/run_kv_fault_ratio.sh"
    --model "$MODEL"
    --mode trace
    --port "$port"
    --allocator-log "$case_dir/vllm_uvm_allocator.log"
    --trace-log "$case_dir/uvm_fault_stats.log"
    --address-log "$case_dir/vllm_uvm_address_regions.log"
    --server-log "$case_dir/vllm_server.log"
    --bench-log "$case_dir/vllm_bench.log"
    "${address_args[@]}"
    "${cleanup_args[@]}"
    --uvm-policy-enable 1
    --uvm-policy-mode trace_only
    --uvm-device-direct-enable 0
    --uvm-kv-device-direct-enable "$KV_DEVICE_DIRECT_ENABLE"
    --uvm-kv-device-direct-min-bytes "$KV_DEVICE_DIRECT_MIN_BYTES"
    --uvm-kv-device-direct-max-bytes "$KV_DEVICE_DIRECT_MAX_BYTES"
    --uvm-kv-device-direct-max-total-bytes "$KV_DEVICE_DIRECT_MAX_TOTAL_BYTES"
    --uvm-kv-range-config-enable 0
    --uvm-fixed-weight-device-direct-enable "$FIXED_WEIGHT_DEVICE_DIRECT_ENABLE"
    --uvm-fixed-weight-device-direct-min-bytes 0
    --uvm-fixed-weight-device-direct-max-bytes 0
    --uvm-fixed-weight-device-direct-max-total-bytes 0
    --uvm-fixed-weight-device-direct-trace-file "$fixed_trace"
    --uvm-residency-enable 1
    --uvm-residency-stage stage_m_cold_streaming
    --uvm-residency-mode trace_only
    --uvm-residency-trace-enable 1
    --uvm-residency-trace-file "$residency_trace"
    --uvm-pool-registry-enable 1
    --uvm-pool-coordinator-enable 0
    --uvm-scratch-pool-enable 0
    --uvm-kv-runtime-enable 0
    --uvm-weight-map-enable 1
    --uvm-weight-map-file "$weight_map"
    --uvm-weight-prefetch-enable 0
    --uvm-weight-offload-enable 0
    --uvm-weight-batch-executor-enable 1
    --uvm-weight-executor "$WEIGHT_EXECUTOR"
    --uvm-cold-streaming-enable "$cold_streaming_enable"
    --uvm-cold-streaming-budget-bytes "$streaming_bytes"
    --uvm-cold-streaming-ttl-steps "$STREAMING_TTL_STEPS"
    --uvm-cold-streaming-max-bytes-per-step "$max_bytes_per_step"
    --uvm-cold-streaming-max-experts-per-layer "$STREAMING_MAX_EXPERTS_PER_LAYER"
    --uvm-cold-streaming-evict-policy ttl_lru
    --uvm-cold-streaming-trace-file "$cold_trace"
    --uvm-cold-streaming-device 0
    --uvm-cold-streaming-sync-prefetch "$STREAMING_SYNC_PREFETCH"
    --uvm-cold-streaming-sync-evict "$STREAMING_SYNC_EVICT"
    --uvm-cold-streaming-real-query-enable "$STREAMING_REAL_QUERY_ENABLE"
    --uvm-cold-slot-pool-enable "$COLD_SLOT_POOL_ENABLE"
    --uvm-cold-slot-pool-trace-only "$COLD_SLOT_POOL_TRACE_ONLY"
    --uvm-cold-slot-pool-backend "$COLD_SLOT_POOL_BACKEND"
    --uvm-cold-slot-pool-slots-per-layer "$COLD_SLOT_POOL_SLOTS_PER_LAYER"
    --uvm-cold-slot-pool-trace-file "$slot_pool_trace"
    --uvm-cold-slot-pool-overflow-mode "$COLD_SLOT_POOL_OVERFLOW_MODE"
    --uvm-cold-slot-pool-trace-assignments-limit "$COLD_SLOT_POOL_TRACE_ASSIGNMENTS_LIMIT"
    --uvm-cold-slot-pool-sync-migration "$COLD_SLOT_POOL_SYNC_MIGRATION"
    --uvm-cold-slot-pool-driver-mode "$COLD_SLOT_POOL_DRIVER_MODE"
    --uvm-cold-slot-pool-driver-validate-managed "$COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED"
    --uvm-cold-slot-pool-clear-slot-ptes-after-remap "$COLD_SLOT_POOL_CLEAR_SLOT_PTES_AFTER_REMAP"
    --uvm-init-weight-offload-enable 1
    --uvm-init-weight-offload-mode "$INIT_COLD_OFFLOAD_MODE"
    --uvm-init-weight-plan-file "$PLAN_PATH"
    --uvm-init-weight-target-roles moe_gate_up,moe_down
    --uvm-init-weight-sync "$INIT_COLD_OFFLOAD_SYNC"
    --uvm-init-weight-hot-prefetch-enable 1
    --uvm-init-weight-hot-prefetch-sync 1
    --uvm-init-weight-hot-prefetch-max-bytes 0
    --uvm-init-weight-hot-prefetch-device 0
    --uvm-init-weight-hot-prefetch-real-query-enable 1
    --uvm-hot-residency-stage-trace-enable 1
    --uvm-hot-residency-stage-max-num-tokens 32
    --uvm-hot-residency-stage-device 0
    --uvm-init-weight-require-routing-plan 1
    --uvm-init-weight-trace-file "$init_trace"
    --server-max-num-seqs "$max_num_seqs"
    --max-model-len "$MAX_MODEL_LEN"
    --server-kv-cache-memory-bytes "$KV_BYTES"
    --server-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --server-extra-env "$extra_env"
    --prompts "$PROMPTS"
    --request-rate "$REQUEST_RATE"
    --output-len "$OUTPUT_LEN"
  )

  echo "---- $mode_name streaming=$streaming_bytes max_num_seqs=$max_num_seqs port=$port ----"
  run_logged "$case_dir/run.log" "${cmd[@]}"

  local -a static_checker=(
    python3 "$SCRIPT_DIR/check_stage_m_static_hot_residency.py"
    --case-dir "$case_dir"
    --hot-k "$HOT_K"
    --kv-bytes "$KV_BYTES"
    --min-hot-gpu-ratio "$HOT_RESIDENCY_MIN_RATIO"
    --expect-cold-offload
  )
  run_cmd "${static_checker[@]}"

  if [ "$mode_name" = "cold_streaming" ] \
    && [ "$SKIP_COLD_CHECKER" != "1" ] \
    && [ "$COLD_SLOT_POOL_ENABLE" != "1" ]; then
    local -a cold_checker=(
      python3 "$SCRIPT_DIR/check_stage_m_cold_streaming.py"
      --case-dir "$case_dir"
      --trace-file "$cold_trace"
      --streaming-budget-bytes "$streaming_bytes"
      --expect-driver-tags
    )
    run_cmd "${cold_checker[@]}"
  fi

  if [ "$mode_name" = "cold_streaming" ] && [ "$COLD_SLOT_POOL_ENABLE" = "1" ] && [ "$SKIP_SLOT_POOL_CHECKER" != "1" ]; then
    local -a slot_pool_checker=(
      python3 "$SCRIPT_DIR/check_stage_m_cold_slot_pool.py"
      --case-dir "$case_dir"
      --trace-file "$slot_pool_trace"
      --slots-per-layer "$COLD_SLOT_POOL_SLOTS_PER_LAYER"
    )
    run_cmd "${slot_pool_checker[@]}"
  fi

  classify_faults_if_requested "$case_dir"
}

run_global_reference_case() {
  local max_num_seqs="$1"
  local port="$2"
  local case_dir="$RUN_ROOT/global_uvm_reference_kv${KV_BYTES}_s${max_num_seqs}"
  mkdir -p "$case_dir"

  local weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"
  local extra_env
  extra_env="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1"

  local -a address_args=()
  if [ "$FAULT_ADDRESS_LOG" = "1" ]; then
    address_args+=(--with-address-log --address-trace-log "$case_dir/uvm_fault_addrs.log")
  fi
  local -a cleanup_args=()
  if [ "$NO_CLEANUP" = "1" ]; then
    cleanup_args+=(--no-cleanup)
  fi

  local -a cmd=(
    "$SCRIPT_DIR/run_kv_fault_ratio.sh"
    --model "$MODEL"
    --mode trace
    --port "$port"
    --allocator-log "$case_dir/vllm_uvm_allocator.log"
    --trace-log "$case_dir/uvm_fault_stats.log"
    --address-log "$case_dir/vllm_uvm_address_regions.log"
    --server-log "$case_dir/vllm_server.log"
    --bench-log "$case_dir/vllm_bench.log"
    "${address_args[@]}"
    "${cleanup_args[@]}"
    --uvm-policy-enable 1
    --uvm-policy-mode trace_only
    --uvm-device-direct-enable 0
    --uvm-kv-device-direct-enable 0
    --uvm-kv-range-config-enable 1
    --uvm-fixed-weight-device-direct-enable 0
    --uvm-residency-enable 0
    --uvm-kv-runtime-enable 0
    --uvm-weight-map-enable 1
    --uvm-weight-map-file "$weight_map"
    --uvm-weight-prefetch-enable 0
    --uvm-weight-offload-enable 0
    --uvm-weight-batch-executor-enable 0
    --uvm-cold-streaming-enable 0
    --uvm-init-weight-offload-enable 0
    --uvm-init-weight-hot-prefetch-enable 0
    --server-max-num-seqs "$max_num_seqs"
    --max-model-len "$MAX_MODEL_LEN"
    --server-kv-cache-memory-bytes "$KV_BYTES"
    --server-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --server-extra-env "$extra_env"
    --prompts "$PROMPTS"
    --request-rate "$REQUEST_RATE"
    --output-len "$OUTPUT_LEN"
  )

  echo "---- global_uvm_reference max_num_seqs=$max_num_seqs port=$port ----"
  run_logged "$case_dir/run.log" "${cmd[@]}"
  classify_faults_if_requested "$case_dir"
}

case_index=0
for max_num_seqs in $MAX_NUM_SEQS_LIST; do
  if [ "$RUN_GLOBAL_UVM_REFERENCE" = "1" ]; then
    run_global_reference_case "$max_num_seqs" "$((BASE_PORT + case_index))"
    case_index=$((case_index + 1))
  fi
  for streaming_bytes in $STREAMING_BYTES_LIST; do
    if [ "$RUN_FAULT_ONLY" = "1" ]; then
      run_matched_case fault_only "$streaming_bytes" "$max_num_seqs" "$((BASE_PORT + case_index))"
      case_index=$((case_index + 1))
    fi
    if [ "$RUN_COLD_STREAMING" = "1" ]; then
      run_matched_case cold_streaming "$streaming_bytes" "$max_num_seqs" "$((BASE_PORT + case_index))"
      case_index=$((case_index + 1))
    fi
  done
done

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

python3 - "$RUN_ROOT" <<'PY'
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

GIB = 1024 ** 3
root = Path(sys.argv[1])

def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(text: str, pattern: str):
    match = re.search(pattern, text)
    return float(match.group(1).replace(",", "")) if match else None

def first_int(text: str, pattern: str):
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else None

def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def bench(path: Path) -> dict:
    text = read(path)
    return {
        "successful_requests": first_int(text, r"Successful requests:\s+([0-9,]+)"),
        "failed_requests": first_int(text, r"Failed requests:\s+([0-9,]+)"),
        "output_tok_s": first_float(text, r"Output token throughput \(tok/s\):\s+([0-9,.]+)"),
        "total_tok_s": first_float(text, r"Total Token throughput \(tok/s\):\s+([0-9,.]+)"),
        "mean_ttft_ms": first_float(text, r"Mean TTFT \(ms\):\s+([0-9,.]+)"),
        "p99_ttft_ms": first_float(text, r"P99 TTFT \(ms\):\s+([0-9,.]+)"),
        "mean_tpot_ms": first_float(text, r"Mean TPOT \(ms\):\s+([0-9,.]+)"),
        "p99_tpot_ms": first_float(text, r"P99 TPOT \(ms\):\s+([0-9,.]+)"),
        "benchmark_exit_code": first_int(text, r"Benchmark phase 'main' exit code:\s+([0-9]+)"),
    }

def run_stats(path: Path) -> dict:
    text = read(path)
    return {
        "delta_faults": first_int(text, r"delta_faults=([0-9]+)"),
        "delta_after_dedup": first_int(text, r"delta_after_dedup=([0-9]+)"),
        "delta_kv_after_dedup": first_int(text, r"delta_kv_after_dedup=([0-9]+)"),
        "delta_kv_after_dedup_ratio_pct": first_float(text, r"delta_kv_after_dedup_ratio=([0-9.]+)%"),
    }

def fault_summary(path: Path) -> dict:
    data = load_json(path)
    classified = data.get("classification", {})
    if not classified:
        classified = data
    counts = classified.get("counts", {}) if isinstance(classified, dict) else {}
    total = sum(int(v or 0) for v in counts.values()) if counts else None
    weight = int(counts.get("weight", 0) or 0) if counts else None
    kv = int(counts.get("kv_cache", 0) or 0) if counts else None
    unknown = int(counts.get("unknown", 0) or 0) if counts else None
    return {
        "fault_unique_total": total,
        "fault_unique_weight": weight,
        "fault_unique_kv": kv,
        "fault_unique_unknown": unknown,
        "fault_unique_weight_pct": (100.0 * weight / total) if total and weight is not None else None,
        "fault_unique_kv_pct": (100.0 * kv / total) if total and kv is not None else None,
        "fault_unique_unknown_pct": (100.0 * unknown / total) if total and unknown is not None else None,
    }

rows = []
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    name = case_dir.name
    mode = None
    streaming_bytes = None
    kv_bytes = None
    max_num_seqs = None
    matched = re.match(r"(fault_only|cold_streaming)_hot(\d+)_stream(\d+)_kv(\d+)_s(\d+)$", name)
    if matched:
        mode = matched.group(1)
        hot_k = int(matched.group(2))
        streaming_bytes = int(matched.group(3))
        kv_bytes = int(matched.group(4))
        max_num_seqs = int(matched.group(5))
    else:
        ref = re.match(r"(global_uvm_reference)_kv(\d+)_s(\d+)$", name)
        if not ref:
            continue
        mode = ref.group(1)
        hot_k = None
        kv_bytes = int(ref.group(2))
        max_num_seqs = int(ref.group(3))

    static = load_json(case_dir / "static_hot_residency_summary.json")
    cold = load_json(case_dir / "cold_streaming_summary.json")
    static_hot = static.get("hot_prefetch", {}) if static else {}
    cstats = cold.get("cold_streaming", {}) if cold else {}
    rows.append({
        "case": name,
        "mode": mode,
        "hot_k": hot_k,
        "streaming_bytes": streaming_bytes,
        "streaming_gib": streaming_bytes / GIB if streaming_bytes else 0,
        "kv_bytes": kv_bytes,
        "max_num_seqs": max_num_seqs,
        "static_passed": static.get("passed") if static else None,
        "cold_passed": cold.get("passed") if cold else None,
        "hot_gpu_ratio": static_hot.get("hot_real_gpu_ratio"),
        "init_cold_offload_gib": (
            (static_hot.get("init_weight_offload_issued_bytes") or 0) / GIB
            if static_hot else None
        ),
        **bench(case_dir / "vllm_bench.log"),
        **run_stats(case_dir / "run.log"),
        "cold_miss_experts": cstats.get("cold_miss_experts"),
        "cold_miss_gib": cstats.get("cold_miss_gib"),
        "streaming_prefetch_gib": cstats.get("streaming_prefetch_issued_gib"),
        "streaming_eviction_count": cstats.get("streaming_eviction_count"),
        "streaming_eviction_gib": cstats.get("streaming_eviction_gib"),
        "max_streaming_live_gib": cstats.get("max_streaming_live_gib"),
        "hot_evicted_by_cold": cstats.get("hot_evicted_by_cold_count"),
        **fault_summary(case_dir / "fault_address_analysis" / "summary.json"),
    })

csv_path = root / "cold_streaming_vs_fault_only_summary.csv"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

lines = [
    "# Cold Streaming vs Fault-only Summary",
    "",
    "| mode | streaming GiB | max seqs | static | cold | tok/s | TPOT ms | after-dedup faults | hot ratio | cold offload GiB | stream prefetch GiB | evict GiB | hot evicted |",
    "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {mode} | {streaming_gib} | {max_num_seqs} | {static_passed} | {cold_passed} | "
        "{output_tok_s} | {mean_tpot_ms} | {delta_after_dedup} | {hot_gpu_ratio} | "
        "{init_cold_offload_gib} | {streaming_prefetch_gib} | {streaming_eviction_gib} | "
        "{hot_evicted_by_cold} |".format(**{k: fmt(v) for k, v in row.items()})
    )

lines += [
    "",
    "## Matched A/B Interpretation",
    "",
    "- Compare `cold_streaming` only against `fault_only` rows with the same `streaming GiB`, `KV bytes`, and `max seqs`.",
    "- `global_uvm_reference` is a historical reference, not the fairness baseline, because its KV/fixed-weight/hotset layout differs.",
    "- Treat cold streaming as better only if TPOT drops, output throughput rises, after-dedup faults drop, and `hot_evicted_by_cold` stays zero.",
]
md_path = root / "cold_streaming_vs_fault_only_summary.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {csv_path}")
print(f"wrote {md_path}")
PY

echo "Summary: $RUN_ROOT/cold_streaming_vs_fault_only_summary.md"
