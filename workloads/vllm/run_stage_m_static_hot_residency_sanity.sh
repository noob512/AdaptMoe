#!/usr/bin/env bash
# One-case sanity for static hot expert pre-residency:
#   KV native cudaMalloc + fixed non-expert weights native cudaMalloc +
#   MoE expert weights managed UVM + init hotK prefetch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_static_hot_residency_sanity_$(date -u +%Y%m%d_%H%M%S)}"
TRACE_PATH="${TRACE_PATH:-$REPO_ROOT/log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl}"
PLAN_PATH="${PLAN_PATH:-}"

HOT_K="${HOT_K:-64}"
KV_BYTES="${KV_BYTES:-7516192768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
REQUEST_RATE="${REQUEST_RATE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
PORT="${PORT:-8017}"
EXPERT_BYTES="${EXPERT_BYTES:-4718592}"
HOTSET_SCORE_MODE="${HOTSET_SCORE_MODE:-weighted}"
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.5}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
HOT_RESIDENCY_MIN_RATIO="${HOT_RESIDENCY_MIN_RATIO:-0.90}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'USAGE'
Usage:
  workloads/vllm/run_stage_m_static_hot_residency_sanity.sh

Environment knobs:
  RUN_ROOT=log/stage_m_static_hot_residency_sanity_first
  TRACE_PATH=log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl
  HOT_K=64
  KV_BYTES=7516192768
  MAX_NUM_SEQS=16
  PROMPTS=4 OUTPUT_LEN=8 REQUEST_RATE=4

The script generates a train/test static hotset plan when PLAN_PATH is unset,
starts one vLLM case, and writes static_hot_residency_report.md/json.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --trace-path) TRACE_PATH="$2"; shift 2 ;;
    --plan-path) PLAN_PATH="$2"; shift 2 ;;
    --hot-k) HOT_K="$2"; shift 2 ;;
    --kv-bytes) KV_BYTES="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

abs_path() {
  local path="$1"
  case "$path" in
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "$path" ;;
  esac
}

RUN_ROOT="$(abs_path "$RUN_ROOT")"
TRACE_PATH="$(abs_path "$TRACE_PATH")"
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(cd "$RUN_ROOT" && pwd)"

COMMANDS_LOG="$RUN_ROOT/commands.log"
: > "$COMMANDS_LOG"

append_command() {
  printf 'command:' | tee -a "$COMMANDS_LOG"
  printf ' %q' "$@" | tee -a "$COMMANDS_LOG"
  printf '\n' | tee -a "$COMMANDS_LOG"
}

run_cmd() {
  append_command "$@"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

echo "==========================================================="
echo "Stage M Static Hot Expert Residency Sanity"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL"
echo "trace=$TRACE_PATH"
echo "hotK=$HOT_K kv_bytes=$KV_BYTES max_num_seqs=$MAX_NUM_SEQS"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "==========================================================="

if [ -z "$PLAN_PATH" ]; then
  if [ ! -s "$TRACE_PATH" ]; then
    echo "ERROR: TRACE_PATH not found or empty: $TRACE_PATH" >&2
    echo "Set TRACE_PATH to an existing MoE routing trace, or generate one first." >&2
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

case_dir="$RUN_ROOT/static_hot_k${HOT_K}_kv${KV_BYTES}_s${MAX_NUM_SEQS}"
mkdir -p "$case_dir"

fixed_trace="$case_dir/vllm_uvm_fixed_weight_device_direct.jsonl"
weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"
init_trace="$case_dir/vllm_uvm_init_weight_offload_stage_m.jsonl"

extra_env="HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
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
  --uvm-residency-trace-enable 1 \
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
  --uvm-init-weight-offload-enable 0 \
  --uvm-init-weight-offload-mode trace_only \
  --uvm-init-weight-plan-file "$PLAN_PATH" \
  --uvm-init-weight-target-roles moe_gate_up,moe_down \
  --uvm-init-weight-sync 1 \
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

run_cmd python3 "$SCRIPT_DIR/check_stage_m_static_hot_residency.py" \
  --case-dir "$case_dir" \
  --hot-k "$HOT_K" \
  --kv-bytes "$KV_BYTES" \
  --min-hot-gpu-ratio "$HOT_RESIDENCY_MIN_RATIO"

echo "Report: $case_dir/static_hot_residency_report.md"
echo "Summary: $case_dir/static_hot_residency_summary.json"
