#!/usr/bin/env bash
# Feasibility experiments for a balanced KV + resident hot expert plan.
#
# Defaults are intentionally conservative:
#   - outputs go to /tmp, not repo log/
#   - only Stage 0/1 run by default
#   - runtime KV/residency/live ablations are opt-in

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-/tmp/stage_m_resident_hotset_feasibility_$(date -u +%Y%m%d_%H%M%S)}"

PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-32}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-32}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-0}"
K_VALUES="${K_VALUES:-32 48 64 72 80 96}"
EXPERT_BYTES="${EXPERT_BYTES:-4718592}"
TRAIN_SPLIT_FRACTION="${TRAIN_SPLIT_FRACTION:-0.5}"
HOTSET_SCORE_MODE="${HOTSET_SCORE_MODE:-weighted}"

RUN_TRACE="${RUN_TRACE:-1}"
RUN_TRAIN_TEST="${RUN_TRAIN_TEST:-1}"
RUN_KV_DRYRUN="${RUN_KV_DRYRUN:-0}"
RUN_HOT_RESIDENCY="${RUN_HOT_RESIDENCY:-0}"
RUN_LIVE_ABLATION="${RUN_LIVE_ABLATION:-0}"
DRY_RUN="${DRY_RUN:-0}"

# 5/7/9 GiB.
KV_BYTES_LIST="${KV_BYTES_LIST:-5368709120 7516192768 9663676416}"
HOT_K_LIST="${HOT_K_LIST:-48 64 72 80}"
STREAMING_BYTES_LIST="${STREAMING_BYTES_LIST:-1073741824 2147483648 3221225472}"
LIVE_MAX_NUM_SEQS_LIST="${LIVE_MAX_NUM_SEQS_LIST:-16 32 64}"

# A 32GB card usually needs a few GiB for runtime/workspace/fragmentation.
GLOBAL_GPU_BYTES="${GLOBAL_GPU_BYTES:-30064771072}"       # 28 GiB estimated budget.
FIXED_RUNTIME_HEADROOM_BYTES="${FIXED_RUNTIME_HEADROOM_BYTES:-4294967296}" # 4 GiB.
STAGE_L_SCRATCH_GPU_BYTES="${STAGE_L_SCRATCH_GPU_BYTES:-1073741824}"
SAFETY_MARGIN_BYTES="${SAFETY_MARGIN_BYTES:-2147483648}"

WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
INIT_COLD_OFFLOAD_MAX_BYTES="${INIT_COLD_OFFLOAD_MAX_BYTES:-0}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-0}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-268435456}"
WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="${WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP:-64}"
WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="${WEIGHT_ROUTER_TOPK_EVERY_N_STEPS:-1}"
WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-16}"
WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-16}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"

usage() {
  cat <<'USAGE'
Usage:
  workloads/vllm/run_stage_m_resident_hotset_feasibility.sh [options]

Environment variables are also supported.

Default stages:
  RUN_TRACE=1
  RUN_TRAIN_TEST=1
  RUN_KV_DRYRUN=0
  RUN_HOT_RESIDENCY=0
  RUN_LIVE_ABLATION=0

Useful knobs:
  RUN_ROOT=/tmp/stage_m_resident_hotset_feasibility_first
  PROMPTS=32 OUTPUT_LEN=64 REQUEST_RATE=32
  MAX_NUM_SEQS_LIST="32 64"
  MAX_NUM_BATCHED_TOKENS_LIST="0"
  K_VALUES="32 48 64 72 80 96"
  KV_BYTES_LIST="5368709120 7516192768 9663676416"
  HOT_K_LIST="48 64 72 80"
  STREAMING_BYTES_LIST="1073741824 2147483648 3221225472"
  LIVE_MAX_NUM_SEQS_LIST="16 32 64"
  DRY_RUN=1

Stages:
  0 trace:
    Collect MoE routing traces under pure/global UVM, no prefetch/offload.

  1 train/test:
    Build per-layer top-K from the first trace half and evaluate on the second
    half. Also emits Stage-M-compatible train_static_hotset_k*.json plans.

  2 KV dry-run:
    Start real vLLM full_impl cases with fixed KV budgets and no runtime weight
    prefetch. This checks startup, KV blocks, OOM/cudaMalloc failures, and basic
    performance.

  3 hot residency:
    Init-prefetch train top-K hot experts, query real residency after init and
    at first decode marker. This tests whether hot pages stay resident.

  4 live ablation:
    Run hotK + streaming-budget matrix with runtime prefetch/hit tracing.

Examples:
  # Cheap first pass: trace + train/test only.
  RUN_ROOT=/tmp/hotset_feas_first \
  workloads/vllm/run_stage_m_resident_hotset_feasibility.sh

  # After coverage looks promising, run a small runtime smoke.
  RUN_ROOT=/tmp/hotset_feas_runtime \
  RUN_TRACE=0 RUN_TRAIN_TEST=0 RUN_KV_DRYRUN=1 RUN_HOT_RESIDENCY=1 \
  TRACE_PATH=/tmp/hotset_feas_first/stage0_trace/s32_bt0/vllm_uvm_moe_routing_trace.jsonl \
  workloads/vllm/run_stage_m_resident_hotset_feasibility.sh
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --max-num-seqs-list) MAX_NUM_SEQS_LIST="$2"; shift 2 ;;
    --max-num-batched-tokens-list) MAX_NUM_BATCHED_TOKENS_LIST="$2"; shift 2 ;;
    --k-values) K_VALUES="$2"; shift 2 ;;
    --kv-bytes-list) KV_BYTES_LIST="$2"; shift 2 ;;
    --hot-k-list) HOT_K_LIST="$2"; shift 2 ;;
    --streaming-bytes-list) STREAMING_BYTES_LIST="$2"; shift 2 ;;
    --live-max-num-seqs-list) LIVE_MAX_NUM_SEQS_LIST="$2"; shift 2 ;;
    --run-trace) RUN_TRACE="$2"; shift 2 ;;
    --run-train-test) RUN_TRAIN_TEST="$2"; shift 2 ;;
    --run-kv-dryrun) RUN_KV_DRYRUN="$2"; shift 2 ;;
    --run-hot-residency) RUN_HOT_RESIDENCY="$2"; shift 2 ;;
    --run-live-ablation) RUN_LIVE_ABLATION="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
COMMAND_LOG="$RUN_ROOT/commands.log"
SUMMARY_CSV="$RUN_ROOT/experiment_summary.csv"
: > "$COMMAND_LOG"

run_cmd() {
  printf 'command:' | tee -a "$COMMAND_LOG"
  printf ' %q' "$@" | tee -a "$COMMAND_LOG"
  printf '\n' | tee -a "$COMMAND_LOG"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

batch_token_args() {
  local value="$1"
  if [ "$value" != "0" ] && [ -n "$value" ]; then
    printf '%s\0%s\0' --server-max-num-batched-tokens "$value"
  fi
}

trace_path_for() {
  local seqs="$1"
  local batched_tokens="$2"
  printf '%s/stage0_trace/s%s_bt%s/vllm_uvm_moe_routing_trace.jsonl' \
    "$RUN_ROOT" "$seqs" "$batched_tokens"
}

plan_path_for() {
  local seqs="$1"
  local batched_tokens="$2"
  local k="$3"
  printf '%s/stage1_train_test/s%s_bt%s/plans/train_static_hotset_k%s.json' \
    "$RUN_ROOT" "$seqs" "$batched_tokens" "$k"
}

hot_bytes_for_k() {
  local k="$1"
  awk -v k="$k" -v expert="$EXPERT_BYTES" 'BEGIN { printf "%.0f", k * 48 * expert }'
}

weight_budget_for() {
  local k="$1"
  local streaming_bytes="$2"
  local hot_bytes
  hot_bytes="$(hot_bytes_for_k "$k")"
  awk -v hot="$hot_bytes" -v streaming="$streaming_bytes" 'BEGIN { printf "%.0f", hot + streaming }'
}

echo "==========================================================="
echo "Stage M Resident Hotset Feasibility"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS_LIST"
echo "K=$K_VALUES hotK=$HOT_K_LIST"
echo "KV bytes=$KV_BYTES_LIST streaming bytes=$STREAMING_BYTES_LIST"
echo "stages trace=$RUN_TRACE train_test=$RUN_TRAIN_TEST kv_dryrun=$RUN_KV_DRYRUN hot_residency=$RUN_HOT_RESIDENCY live=$RUN_LIVE_ABLATION dry_run=$DRY_RUN"
echo "==========================================================="

if [ "$RUN_TRACE" = "1" ]; then
  for seqs in $MAX_NUM_SEQS_LIST; do
    for batched_tokens in $MAX_NUM_BATCHED_TOKENS_LIST; do
      case_dir="$RUN_ROOT/stage0_trace/s${seqs}_bt${batched_tokens}"
      mkdir -p "$case_dir"
      extra=()
      while IFS= read -r -d '' item; do
        extra+=("$item")
      done < <(batch_token_args "$batched_tokens")
      run_cmd "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
        --model "$MODEL" \
        --mode trace \
        --with-address-log \
        --allocator-log "$case_dir/vllm_uvm_allocator.log" \
        --trace-log "$case_dir/uvm_fault_stats.log" \
        --address-trace-log "$case_dir/uvm_fault_addrs.log" \
        --address-log "$case_dir/vllm_uvm_address_regions.log" \
        --server-log "$case_dir/vllm_server.log" \
        --bench-log "$case_dir/vllm_bench.log" \
        --uvm-policy-enable 0 \
        --uvm-device-direct-enable 0 \
        --uvm-residency-enable 0 \
        --uvm-kv-runtime-enable 0 \
        --uvm-weight-prefetch-enable 0 \
        --uvm-weight-offload-enable 0 \
        --uvm-weight-batch-executor-enable 0 \
        --uvm-pool-coordinator-enable 0 \
        --uvm-pool-registry-enable 0 \
        --uvm-scratch-pool-enable 0 \
        --uvm-moe-routing-trace-enable 1 \
        --uvm-moe-routing-trace-file "$case_dir/vllm_uvm_moe_routing_trace.jsonl" \
        --server-max-num-seqs "$seqs" \
        "${extra[@]}" \
        --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
        --prompts "$PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --output-len "$OUTPUT_LEN"
    done
  done
fi

if [ "$RUN_TRAIN_TEST" = "1" ]; then
  k_csv="$(echo "$K_VALUES" | tr ' ' ',')"
  for seqs in $MAX_NUM_SEQS_LIST; do
    for batched_tokens in $MAX_NUM_BATCHED_TOKENS_LIST; do
      trace="${TRACE_PATH:-$(trace_path_for "$seqs" "$batched_tokens")}"
      out_dir="$RUN_ROOT/stage1_train_test/s${seqs}_bt${batched_tokens}"
      if [ ! -s "$trace" ] && [ "$DRY_RUN" != "1" ]; then
        echo "skip_train_test_missing_trace=$trace" | tee -a "$COMMAND_LOG"
        continue
      fi
      run_cmd python3 "$SCRIPT_DIR/analyze_moe_hotset_train_test.py" \
        --trace "$trace" \
        --out-dir "$out_dir" \
        --k-values "$k_csv" \
        --split-fraction "$TRAIN_SPLIT_FRACTION" \
        --score-mode "$HOTSET_SCORE_MODE" \
        --expert-bytes "$EXPERT_BYTES" \
        --write-plans
    done
  done
fi

run_full_impl_case() {
  local case_dir="$1"
  local plan_json="$2"
  local seqs="$3"
  local kv_bytes="$4"
  local weight_bytes="$5"
  local init_hot="$6"
  local runtime_prefetch="$7"
  local hit_trace="$8"
  local streaming_bytes="$9"
  local hot_k="${10}"
  local prefetch_source="plan_intersection"
  if [ "$runtime_prefetch" = "1" ]; then
    prefetch_source="router_topk"
  fi

  mkdir -p "$case_dir"
  run_cmd "$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" \
    --run-root "$case_dir" \
    --log-file "$case_dir/run.log" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --repeat 1 \
    --skip-trace-control \
    --server-max-num-seqs "$seqs" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --global-gpu-bytes "$GLOBAL_GPU_BYTES" \
    --weight-gpu-bytes "$weight_bytes" \
    --kv-gpu-bytes "$kv_bytes" \
    --stage-l-scratch-gpu-bytes "$STAGE_L_SCRATCH_GPU_BYTES" \
    --safety-margin-bytes "$SAFETY_MARGIN_BYTES" \
    --weight-plan-file "$plan_json" \
    --weight-executor "$WEIGHT_EXECUTOR" \
    --init-weight-max-bytes "$INIT_COLD_OFFLOAD_MAX_BYTES" \
    --init-weight-max-experts-per-layer 0 \
    --init-weight-hot-prefetch-enable "$init_hot" \
    --init-weight-hot-prefetch-sync 1 \
    --init-weight-hot-prefetch-max-bytes 0 \
    --init-weight-hot-prefetch-device 0 \
    --init-weight-hot-prefetch-real-query-enable "$init_hot" \
    --hot-residency-stage-trace-enable "$init_hot" \
    --hot-residency-stage-max-num-tokens "$seqs" \
    --hot-residency-stage-device 0 \
    --weight-runtime-prefetch-enable "$runtime_prefetch" \
    --weight-prefetch-trace-enable 1 \
    --weight-prefetch-source "$prefetch_source" \
    --weight-prefetch-max-experts-per-layer "$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" \
    --weight-prefetch-max-bytes-per-step "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" \
    --weight-router-topk-every-n-steps "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS" \
    --weight-prefetch-skip-resident 1 \
    --weight-prefetch-expert-cache-enable 1 \
    --weight-prefetch-region-cooldown-steps 8 \
    --weight-prefetch-hit-trace-enable "$hit_trace" \
    --weight-prefetch-hit-driver-query-enable "$hit_trace" \
    --weight-prefetch-hit-max-regions-per-step "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" \
    --weight-active-evict-enable 1 \
    --weight-active-evict-max-experts-per-layer "$WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER" \
    --weight-max-victims-per-step "$WEIGHT_MAX_VICTIMS_PER_STEP" \
    --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
    --scratch-device-direct-enable 0 \
    --scratch-pool-enable 0 \
    --residency-enable 1 \
    --residency-trace-enable 1 \
    --kv-runtime-enable 0 \
    --pool-coordinator-enable 1 \
    --pool-coordinator-trace-enable 1 \
    --pool-registry-enable 1 \
    --auto-gap-watch-enable 0 \
    --fault-address-log-enable 1

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$case_dir" "$seqs" "$kv_bytes" "$hot_k" "$streaming_bytes" "$weight_bytes" \
    "$init_hot" "$runtime_prefetch" "$hit_trace" "$plan_json" >> "$SUMMARY_CSV"
}

if [ "$RUN_KV_DRYRUN" = "1" ] || [ "$RUN_HOT_RESIDENCY" = "1" ] || [ "$RUN_LIVE_ABLATION" = "1" ]; then
  echo "case_dir,max_num_seqs,kv_bytes,hot_k,streaming_bytes,weight_bytes,init_hot,runtime_prefetch,hit_trace,plan" > "$SUMMARY_CSV"
fi

if [ "$RUN_KV_DRYRUN" = "1" ]; then
  # Fixed KV budget + fixed hot plan, but runtime expert prefetch disabled.
  # This isolates startup/KV capacity/OOM behavior before live streaming.
  for seqs in $MAX_NUM_SEQS_LIST; do
    batched_tokens=0
    for kv_bytes in $KV_BYTES_LIST; do
      for hot_k in $HOT_K_LIST; do
        plan_json="$(plan_path_for "$seqs" "$batched_tokens" "$hot_k")"
        if [ ! -s "$plan_json" ] && [ "$DRY_RUN" != "1" ]; then
          echo "skip_kv_dryrun_missing_plan=$plan_json" | tee -a "$COMMAND_LOG"
          continue
        fi
        streaming_bytes="$(echo "$STREAMING_BYTES_LIST" | awk '{print $1}')"
        weight_bytes="$(weight_budget_for "$hot_k" "$streaming_bytes")"
        case_dir="$RUN_ROOT/stage2_kv_dryrun/s${seqs}_kv${kv_bytes}_hot${hot_k}_stream${streaming_bytes}"
        run_full_impl_case "$case_dir" "$plan_json" "$seqs" "$kv_bytes" "$weight_bytes" 0 0 0 "$streaming_bytes" "$hot_k"
      done
    done
  done
fi

if [ "$RUN_HOT_RESIDENCY" = "1" ]; then
  # Init-prefetch the train top-K hot plan, then query real residency.
  for seqs in $MAX_NUM_SEQS_LIST; do
    batched_tokens=0
    kv_bytes="$(echo "$KV_BYTES_LIST" | awk '{print $1}')"
    streaming_bytes="$(echo "$STREAMING_BYTES_LIST" | awk '{print $1}')"
    for hot_k in $HOT_K_LIST; do
      plan_json="$(plan_path_for "$seqs" "$batched_tokens" "$hot_k")"
      if [ ! -s "$plan_json" ] && [ "$DRY_RUN" != "1" ]; then
        echo "skip_hot_residency_missing_plan=$plan_json" | tee -a "$COMMAND_LOG"
        continue
      fi
      weight_bytes="$(weight_budget_for "$hot_k" "$streaming_bytes")"
      case_dir="$RUN_ROOT/stage3_hot_residency/s${seqs}_kv${kv_bytes}_hot${hot_k}_stream${streaming_bytes}"
      run_full_impl_case "$case_dir" "$plan_json" "$seqs" "$kv_bytes" "$weight_bytes" 1 0 0 "$streaming_bytes" "$hot_k"
    done
  done
fi

if [ "$RUN_LIVE_ABLATION" = "1" ]; then
  for seqs in $LIVE_MAX_NUM_SEQS_LIST; do
    batched_tokens=0
    for kv_bytes in $KV_BYTES_LIST; do
      for hot_k in $HOT_K_LIST; do
        plan_json="$(plan_path_for "$seqs" "$batched_tokens" "$hot_k")"
        # If trace/plans were only generated for s32, allow reusing s32 plan.
        if [ ! -s "$plan_json" ]; then
          plan_json="$(plan_path_for 32 "$batched_tokens" "$hot_k")"
        fi
        if [ ! -s "$plan_json" ] && [ "$DRY_RUN" != "1" ]; then
          echo "skip_live_missing_plan=$plan_json" | tee -a "$COMMAND_LOG"
          continue
        fi
        for streaming_bytes in $STREAMING_BYTES_LIST; do
          weight_bytes="$(weight_budget_for "$hot_k" "$streaming_bytes")"
          case_dir="$RUN_ROOT/stage4_live_ablation/s${seqs}_kv${kv_bytes}_hot${hot_k}_stream${streaming_bytes}"
          run_full_impl_case "$case_dir" "$plan_json" "$seqs" "$kv_bytes" "$weight_bytes" 1 1 1 "$streaming_bytes" "$hot_k"
        done
      done
    done
  done
fi

echo "==========================================================="
echo "Done."
echo "run_root=$RUN_ROOT"
echo "commands=$COMMAND_LOG"
if [ -s "$SUMMARY_CSV" ]; then
  echo "summary_csv=$SUMMARY_CSV"
fi
echo "==========================================================="
