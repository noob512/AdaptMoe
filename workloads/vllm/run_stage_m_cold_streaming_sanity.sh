#!/usr/bin/env bash
set -euo pipefail

# Stage M short-term sanity:
#   KV native cudaMalloc + fixed non-expert weights native cudaMalloc +
#   MoE expert weights managed UVM + init hotK prefetch +
#   application-level cold expert streaming buffer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_cold_streaming_sanity}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
TRACE_PATH="${TRACE_PATH:-log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl}"
PLAN_PATH="${PLAN_PATH:-}"
HOT_K="${HOT_K:-64}"
KV_BYTES="${KV_BYTES:-7516192768}"
STREAMING_BYTES="${STREAMING_BYTES:-1073741824}"
STREAMING_TTL_STEPS="${STREAMING_TTL_STEPS:-32}"
STREAMING_MAX_BYTES_PER_STEP="${STREAMING_MAX_BYTES_PER_STEP:-$STREAMING_BYTES}"
STREAMING_MAX_EXPERTS_PER_LAYER="${STREAMING_MAX_EXPERTS_PER_LAYER:-0}"
STREAMING_EVICT_POLICY="${STREAMING_EVICT_POLICY:-ttl_lru}"
STREAMING_SYNC_PREFETCH="${STREAMING_SYNC_PREFETCH:-0}"
STREAMING_SYNC_EVICT="${STREAMING_SYNC_EVICT:-1}"
STREAMING_REAL_QUERY_ENABLE="${STREAMING_REAL_QUERY_ENABLE:-0}"
EXPECT_EVICTIONS="${EXPECT_EVICTIONS:-0}"
EXPECT_DRIVER_TAGS="${EXPECT_DRIVER_TAGS:-0}"
DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE:-0}"
INIT_COLD_OFFLOAD_ENABLE="${INIT_COLD_OFFLOAD_ENABLE:-0}"
INIT_COLD_OFFLOAD_MODE="${INIT_COLD_OFFLOAD_MODE:-offload_cpu}"
INIT_COLD_OFFLOAD_SYNC="${INIT_COLD_OFFLOAD_SYNC:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PROMPTS="${PROMPTS:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-8}"
PORT="${PORT:-8020}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
EXPERT_BYTES="${EXPERT_BYTES:-4718592}"
HOTSET_SCORE_MODE="${HOTSET_SCORE_MODE:-weighted}"
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.5}"
HOT_RESIDENCY_MIN_RATIO="${HOT_RESIDENCY_MIN_RATIO:-0.90}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  RUN_ROOT=log/stage_m_cold_streaming_sanity \
  TRACE_PATH=log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl \
  STREAMING_BYTES=1073741824 HOT_K=64 KV_BYTES=7516192768 \
  workloads/vllm/run_stage_m_cold_streaming_sanity.sh

Environment:
  HOT_K, KV_BYTES, STREAMING_BYTES, STREAMING_TTL_STEPS,
  STREAMING_MAX_BYTES_PER_STEP, STREAMING_MAX_EXPERTS_PER_LAYER,
  STREAMING_EVICT_POLICY, MAX_NUM_SEQS, PROMPTS, OUTPUT_LEN, REQUEST_RATE.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --trace-path) TRACE_PATH="$2"; shift 2 ;;
    --plan-path) PLAN_PATH="$2"; shift 2 ;;
    --hot-k) HOT_K="$2"; shift 2 ;;
    --kv-bytes) KV_BYTES="$2"; shift 2 ;;
    --streaming-bytes) STREAMING_BYTES="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"
COMMANDS_LOG="$RUN_ROOT/commands.log"
touch "$COMMANDS_LOG"

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

run_cmd() {
  append_command "$@"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

echo "==========================================================="
echo "Stage M Cold Expert Streaming Sanity"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL"
echo "trace=$TRACE_PATH"
echo "hotK=$HOT_K kv_bytes=$KV_BYTES streaming_bytes=$STREAMING_BYTES"
echo "max_num_seqs=$MAX_NUM_SEQS prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
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
    --split-fraction "$TRAIN_SPLIT_FRACTION" \
    --score-mode "$HOTSET_SCORE_MODE" \
    --expert-bytes "$EXPERT_BYTES" \
    --write-plans
  PLAN_PATH="$(abs_path "$plan_dir/plans/train_static_hotset_k${HOT_K}.json")"
else
  PLAN_PATH="$(abs_path "$PLAN_PATH")"
fi

if [ "$DRY_RUN" != "1" ] && [ ! -s "$PLAN_PATH" ]; then
  echo "ERROR: hot plan not found: $PLAN_PATH" >&2
  exit 2
fi

case_dir="$RUN_ROOT/cold_stream_hot${HOT_K}_stream${STREAMING_BYTES}_kv${KV_BYTES}_s${MAX_NUM_SEQS}"
mkdir -p "$case_dir"

fixed_trace="$case_dir/vllm_uvm_fixed_weight_device_direct.jsonl"
weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"
init_trace="$case_dir/vllm_uvm_init_weight_offload_stage_m.jsonl"
cold_trace="$case_dir/vllm_uvm_cold_streaming_stage_m.jsonl"
residency_trace="$case_dir/vllm_uvm_pool_residency_stage_m.jsonl"

extra_env="HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
VLLM_UVM_DRIVER_TAG_ENABLE=$DRIVER_TAG_ENABLE \
VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=$DRIVER_TAG_ENABLE \
VLLM_UVM_DRIVER_POOL_BUDGET_MODE=trace_only \
VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=1 \
VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=1 \
VLLM_UVM_WEIGHT_MAP_QUERY_EXPERT_SLICES=1 \
VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=0"

run_cmd "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
  --model "$MODEL" \
  --mode trace \
  --port "$PORT" \
  --allocator-log "$case_dir/vllm_uvm_allocator.log" \
  --trace-log "$case_dir/uvm_fault_stats.log" \
  --address-log "$case_dir/vllm_uvm_address_regions.log" \
  --server-log "$case_dir/vllm_server.log" \
  --bench-log "$case_dir/vllm_bench.log" \
  --uvm-policy-enable 1 \
  --uvm-policy-mode trace_only \
  --uvm-device-direct-enable 0 \
  --uvm-kv-device-direct-enable 1 \
  --uvm-kv-device-direct-min-bytes 0 \
  --uvm-kv-device-direct-max-bytes 0 \
  --uvm-kv-device-direct-max-total-bytes 0 \
  --uvm-kv-range-config-enable 0 \
  --uvm-fixed-weight-device-direct-enable 1 \
  --uvm-fixed-weight-device-direct-min-bytes 0 \
  --uvm-fixed-weight-device-direct-max-bytes 0 \
  --uvm-fixed-weight-device-direct-max-total-bytes 0 \
  --uvm-fixed-weight-device-direct-trace-file "$fixed_trace" \
  --uvm-residency-enable 1 \
  --uvm-residency-stage stage_m_cold_streaming \
  --uvm-residency-mode trace_only \
  --uvm-residency-trace-enable 1 \
  --uvm-residency-trace-file "$residency_trace" \
  --uvm-pool-registry-enable 1 \
  --uvm-pool-coordinator-enable 0 \
  --uvm-scratch-pool-enable 0 \
  --uvm-kv-runtime-enable 0 \
  --uvm-weight-map-enable 1 \
  --uvm-weight-map-file "$weight_map" \
  --uvm-weight-prefetch-enable 0 \
  --uvm-weight-offload-enable 0 \
  --uvm-weight-batch-executor-enable 1 \
  --uvm-weight-executor "$WEIGHT_EXECUTOR" \
  --uvm-cold-streaming-enable 1 \
  --uvm-cold-streaming-budget-bytes "$STREAMING_BYTES" \
  --uvm-cold-streaming-ttl-steps "$STREAMING_TTL_STEPS" \
  --uvm-cold-streaming-max-bytes-per-step "$STREAMING_MAX_BYTES_PER_STEP" \
  --uvm-cold-streaming-max-experts-per-layer "$STREAMING_MAX_EXPERTS_PER_LAYER" \
  --uvm-cold-streaming-evict-policy "$STREAMING_EVICT_POLICY" \
  --uvm-cold-streaming-trace-file "$cold_trace" \
  --uvm-cold-streaming-device 0 \
  --uvm-cold-streaming-sync-prefetch "$STREAMING_SYNC_PREFETCH" \
  --uvm-cold-streaming-sync-evict "$STREAMING_SYNC_EVICT" \
  --uvm-cold-streaming-real-query-enable "$STREAMING_REAL_QUERY_ENABLE" \
  --uvm-init-weight-offload-enable "$INIT_COLD_OFFLOAD_ENABLE" \
  --uvm-init-weight-offload-mode "$INIT_COLD_OFFLOAD_MODE" \
  --uvm-init-weight-plan-file "$PLAN_PATH" \
  --uvm-init-weight-target-roles moe_gate_up,moe_down \
  --uvm-init-weight-sync "$INIT_COLD_OFFLOAD_SYNC" \
  --uvm-init-weight-hot-prefetch-enable 1 \
  --uvm-init-weight-hot-prefetch-sync 1 \
  --uvm-init-weight-hot-prefetch-max-bytes 0 \
  --uvm-init-weight-hot-prefetch-device 0 \
  --uvm-init-weight-hot-prefetch-real-query-enable 1 \
  --uvm-hot-residency-stage-trace-enable 1 \
  --uvm-hot-residency-stage-max-num-tokens 32 \
  --uvm-hot-residency-stage-device 0 \
  --uvm-init-weight-require-routing-plan 1 \
  --uvm-init-weight-trace-file "$init_trace" \
  --server-max-num-seqs "$MAX_NUM_SEQS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --server-kv-cache-memory-bytes "$KV_BYTES" \
  --server-gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --server-extra-env "$extra_env" \
  --prompts "$PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --output-len "$OUTPUT_LEN"

static_checker=(python3 "$SCRIPT_DIR/check_stage_m_static_hot_residency.py" \
  --case-dir "$case_dir" \
  --hot-k "$HOT_K" \
  --kv-bytes "$KV_BYTES" \
  --min-hot-gpu-ratio "$HOT_RESIDENCY_MIN_RATIO")
if [ "$INIT_COLD_OFFLOAD_ENABLE" = "1" ]; then
  static_checker+=(--expect-cold-offload)
fi
run_cmd "${static_checker[@]}"

cold_checker=(python3 "$SCRIPT_DIR/check_stage_m_cold_streaming.py" \
  --case-dir "$case_dir" \
  --trace-file "$cold_trace" \
  --streaming-budget-bytes "$STREAMING_BYTES")
if [ "$EXPECT_EVICTIONS" = "1" ]; then
  cold_checker+=(--expect-evictions)
fi
if [ "$EXPECT_DRIVER_TAGS" = "1" ]; then
  cold_checker+=(--expect-driver-tags)
fi
run_cmd "${cold_checker[@]}"

echo "Static report: $case_dir/static_hot_residency_report.md"
echo "Cold streaming report: $case_dir/cold_streaming_report.md"
