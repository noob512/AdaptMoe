#!/usr/bin/env bash
# Check Phase F: weight prefetch/offload actions can use the UVM driver batch
# executor, while policy decisions remain in user space.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_f_check_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
ROUTING_TRACE="${ROUTING_TRACE:-}"

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-8}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-8}"
WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_PLAN_FILE="${WEIGHT_PLAN_FILE:-}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-10737418240}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_MAX_NUM_BATCHED_TOKENS="${SERVER_MAX_NUM_BATCHED_TOKENS:-}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-auto}"
REQUIRE_DRIVER=0
SCRATCH_DEVICE_DIRECT_ENABLE="${SCRATCH_DEVICE_DIRECT_ENABLE:-1}"
SCRATCH_POOL_ENABLE="${SCRATCH_POOL_ENABLE:-1}"
RESIDENCY_ENABLE="${RESIDENCY_ENABLE:-1}"
RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE:-1}"
KV_RUNTIME_ENABLE="${KV_RUNTIME_ENABLE:-1}"
WEIGHT_RUNTIME_PREFETCH_ENABLE="${WEIGHT_RUNTIME_PREFETCH_ENABLE:-1}"
WEIGHT_PREFETCH_TRACE_ENABLE="${WEIGHT_PREFETCH_TRACE_ENABLE:-1}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-33554432}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-2}"
WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="${WEIGHT_ROUTER_TOPK_EVERY_N_STEPS:-1}"
WEIGHT_PREFETCH_SKIP_RESIDENT="${WEIGHT_PREFETCH_SKIP_RESIDENT:-1}"
WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
WEIGHT_PREFETCH_SOURCE="${WEIGHT_PREFETCH_SOURCE:-router_topk}"
ORACLE_EXPERT_TRACE_FILE="${ORACLE_EXPERT_TRACE_FILE:-}"
ORACLE_PREFETCH_LOOKAHEAD_STEPS="${ORACLE_PREFETCH_LOOKAHEAD_STEPS:-1}"
WEIGHT_PREFETCH_HIT_TRACE_ENABLE="${WEIGHT_PREFETCH_HIT_TRACE_ENABLE:-0}"
WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="${WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE:-0}"
WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="${WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP:-64}"
INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE:-1}"
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-}"
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-0}"
INIT_WEIGHT_HOT_PREFETCH_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_ENABLE:-0}"
INIT_WEIGHT_HOT_PREFETCH_SYNC="${INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE:-0}"
HOT_RESIDENCY_STAGE_TRACE_ENABLE="${HOT_RESIDENCY_STAGE_TRACE_ENABLE:-0}"
HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="${HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS:-1}"
HOT_RESIDENCY_STAGE_DEVICE="${HOT_RESIDENCY_STAGE_DEVICE:--1}"
POOL_COORDINATOR_ENABLE="${POOL_COORDINATOR_ENABLE:-1}"
POOL_COORDINATOR_TRACE_ENABLE="${POOL_COORDINATOR_TRACE_ENABLE:-1}"
POOL_REGISTRY_ENABLE="${POOL_REGISTRY_ENABLE:-1}"
AUTO_GAP_WATCH_ENABLE="${AUTO_GAP_WATCH_ENABLE:-1}"
FAULT_ADDRESS_LOG_ENABLE="${FAULT_ADDRESS_LOG_ENABLE:-1}"
WEIGHT_ACTIVE_EVICT_ENABLE="${WEIGHT_ACTIVE_EVICT_ENABLE:-0}"
WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-0}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"
WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-16}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_f_check.sh [options]

Purpose:
  Verify Phase F implementation:
    - the UVM allocator shared library exports driver batch APIs
    - Stage M init cold weight offload can request driver_batch/auto
    - runtime router_topk weight prefetch can request driver_batch/auto
    - unsupported running kernels fall back explicitly in auto mode
    - runtime cold offload remains disabled

Options:
  --routing-trace <path>                 reuse an existing routing trace
  --run-root <path>
  --log-file <path>
  --model <name>
  --prompts <n>                          default: 1
  --output-len <n>                       default: 64
  --request-rate <f>                     default: 1
  --hot-experts-per-layer <n>            default: 8
  --cold-experts-per-layer <n>           default: 8
  --weight-plan-file <path>              optional external prefetch/offload plan
  --weight-gpu-bytes <n>                 default: 10GiB
  --kv-gpu-bytes <n>                     default: 2GiB
  --server-max-num-seqs <n>              default: 1
  --server-max-num-batched-tokens <n>    optional vLLM server limit
  --server-gpu-memory-utilization <f>    default: 0.70
  --weight-executor <mode>               auto|driver_batch|cpp_batch (default: auto)
  --require-driver                       fail unless executor=driver_batch is observed
  --scratch-device-direct-enable <0|1>   default: 1
  --scratch-pool-enable <0|1>            default: 1
  --residency-enable <0|1>               default: 1
  --residency-trace-enable <0|1>         default: 1
  --kv-runtime-enable <0|1>              default: 1
  --weight-runtime-prefetch-enable <0|1> default: 1
  --weight-prefetch-trace-enable <0|1>   default: 1
  --weight-prefetch-max-bytes-per-step <n> default: 33554432
  --weight-prefetch-max-experts-per-layer <n> default: 2, 0 means unlimited
  --weight-router-topk-every-n-steps <n> default: 1
  --weight-prefetch-skip-resident <0|1>  default: 1
  --weight-prefetch-expert-cache-enable <0|1> default: 1
  --weight-prefetch-region-cooldown-steps <n> default: 8
  --weight-prefetch-source <source>      router_topk|plan_intersection|oracle_replay
  --oracle-expert-trace-file <path>      routing trace JSONL for oracle_replay
  --oracle-prefetch-lookahead-steps <n>  per-layer replay lookahead
  --weight-prefetch-hit-trace-enable <0|1> default: 0
  --weight-prefetch-hit-driver-query-enable <0|1> default: 0
  --weight-prefetch-hit-max-regions-per-step <n> default: 64
  --init-weight-offload-enable <0|1>     default: 1
  --init-weight-max-bytes <n>            default: weight-gpu-bytes
  --init-weight-max-experts-per-layer <n> default: 0 means unlimited
  --init-weight-hot-prefetch-enable <0|1> default: 0
  --init-weight-hot-prefetch-sync <0|1>  default: 1
  --init-weight-hot-prefetch-max-bytes <n> default: 0 means unlimited
  --init-weight-hot-prefetch-device <id> default: -1/current
  --init-weight-hot-prefetch-real-query-enable <0|1> default: 0
  --hot-residency-stage-trace-enable <0|1> default: 0
  --hot-residency-stage-max-num-tokens <n> default: 1
  --hot-residency-stage-device <id> default: -1/current
  --pool-coordinator-enable <0|1>        default: 1
  --pool-coordinator-trace-enable <0|1>  default: 1
  --pool-registry-enable <0|1>           default: 1
  --auto-gap-watch-enable <0|1>          default: 1
  --fault-address-log-enable <0|1>       default: 1
  --weight-active-evict-enable <0|1>     default: 0
  --weight-active-evict-max-experts-per-layer <n>
                                           default: 0
  --weight-evict-protect-recent-steps <n> default: 8
  --weight-max-victims-per-step <n>      default: 16
  -h, --help

Example:
  ./run_stage_m_phase_f_check.sh \
    --log-file log/stage_m_phase_f_check.log
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --routing-trace) ROUTING_TRACE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --hot-experts-per-layer) HOT_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --cold-experts-per-layer) COLD_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-plan-file) WEIGHT_PLAN_FILE="$2"; shift 2 ;;
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;;
    --server-max-num-batched-tokens) SERVER_MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --weight-executor) WEIGHT_EXECUTOR="$2"; shift 2 ;;
    --require-driver) REQUIRE_DRIVER=1; shift ;;
    --scratch-device-direct-enable) SCRATCH_DEVICE_DIRECT_ENABLE="$2"; shift 2 ;;
    --scratch-pool-enable) SCRATCH_POOL_ENABLE="$2"; shift 2 ;;
    --residency-enable) RESIDENCY_ENABLE="$2"; shift 2 ;;
    --residency-trace-enable) RESIDENCY_TRACE_ENABLE="$2"; shift 2 ;;
    --kv-runtime-enable) KV_RUNTIME_ENABLE="$2"; shift 2 ;;
    --weight-runtime-prefetch-enable) WEIGHT_RUNTIME_PREFETCH_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-trace-enable) WEIGHT_PREFETCH_TRACE_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-max-bytes-per-step) WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="$2"; shift 2 ;;
    --weight-prefetch-max-experts-per-layer) WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-router-topk-every-n-steps) WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="$2"; shift 2 ;;
    --weight-prefetch-skip-resident) WEIGHT_PREFETCH_SKIP_RESIDENT="$2"; shift 2 ;;
    --weight-prefetch-expert-cache-enable) WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-region-cooldown-steps) WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="$2"; shift 2 ;;
    --weight-prefetch-source) WEIGHT_PREFETCH_SOURCE="$2"; shift 2 ;;
    --oracle-expert-trace-file) ORACLE_EXPERT_TRACE_FILE="$2"; shift 2 ;;
    --oracle-prefetch-lookahead-steps) ORACLE_PREFETCH_LOOKAHEAD_STEPS="$2"; shift 2 ;;
    --weight-prefetch-hit-trace-enable) WEIGHT_PREFETCH_HIT_TRACE_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-hit-driver-query-enable) WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-hit-max-regions-per-step) WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="$2"; shift 2 ;;
    --init-weight-offload-enable) INIT_WEIGHT_OFFLOAD_ENABLE="$2"; shift 2 ;;
    --init-weight-max-bytes) INIT_WEIGHT_MAX_BYTES="$2"; shift 2 ;;
    --init-weight-max-experts-per-layer) INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --init-weight-hot-prefetch-enable) INIT_WEIGHT_HOT_PREFETCH_ENABLE="$2"; shift 2 ;;
    --init-weight-hot-prefetch-sync) INIT_WEIGHT_HOT_PREFETCH_SYNC="$2"; shift 2 ;;
    --init-weight-hot-prefetch-max-bytes) INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="$2"; shift 2 ;;
    --init-weight-hot-prefetch-device) INIT_WEIGHT_HOT_PREFETCH_DEVICE="$2"; shift 2 ;;
    --init-weight-hot-prefetch-real-query-enable) INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="$2"; shift 2 ;;
    --hot-residency-stage-trace-enable) HOT_RESIDENCY_STAGE_TRACE_ENABLE="$2"; shift 2 ;;
    --hot-residency-stage-max-num-tokens) HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="$2"; shift 2 ;;
    --hot-residency-stage-device) HOT_RESIDENCY_STAGE_DEVICE="$2"; shift 2 ;;
    --pool-coordinator-enable) POOL_COORDINATOR_ENABLE="$2"; shift 2 ;;
    --pool-coordinator-trace-enable) POOL_COORDINATOR_TRACE_ENABLE="$2"; shift 2 ;;
    --pool-registry-enable) POOL_REGISTRY_ENABLE="$2"; shift 2 ;;
    --auto-gap-watch-enable) AUTO_GAP_WATCH_ENABLE="$2"; shift 2 ;;
    --fault-address-log-enable) FAULT_ADDRESS_LOG_ENABLE="$2"; shift 2 ;;
    --weight-active-evict-enable) WEIGHT_ACTIVE_EVICT_ENABLE="$2"; shift 2 ;;
    --weight-active-evict-max-experts-per-layer) WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-evict-protect-recent-steps) WEIGHT_EVICT_PROTECT_RECENT_STEPS="$2"; shift 2 ;;
    --weight-max-victims-per-step) WEIGHT_MAX_VICTIMS_PER_STEP="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for item in \
  SCRATCH_DEVICE_DIRECT_ENABLE \
  SCRATCH_POOL_ENABLE \
  RESIDENCY_ENABLE \
  RESIDENCY_TRACE_ENABLE \
  KV_RUNTIME_ENABLE \
  WEIGHT_RUNTIME_PREFETCH_ENABLE \
  WEIGHT_PREFETCH_TRACE_ENABLE \
  WEIGHT_PREFETCH_SKIP_RESIDENT \
  WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE \
  WEIGHT_PREFETCH_HIT_TRACE_ENABLE \
  WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE \
  INIT_WEIGHT_OFFLOAD_ENABLE \
  INIT_WEIGHT_HOT_PREFETCH_ENABLE \
  INIT_WEIGHT_HOT_PREFETCH_SYNC \
  INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE \
  HOT_RESIDENCY_STAGE_TRACE_ENABLE \
  POOL_COORDINATOR_ENABLE \
  POOL_COORDINATOR_TRACE_ENABLE \
  POOL_REGISTRY_ENABLE \
  AUTO_GAP_WATCH_ENABLE \
  WEIGHT_ACTIVE_EVICT_ENABLE \
  FAULT_ADDRESS_LOG_ENABLE; do
  if ! [[ "${!item}" =~ ^[01]$ ]]; then
    echo "$item must be 0 or 1" >&2
    exit 2
  fi
done
if [ "$REQUIRE_DRIVER" = "1" ] \
  && [ "$INIT_WEIGHT_OFFLOAD_ENABLE" = "0" ] \
  && [ "$WEIGHT_RUNTIME_PREFETCH_ENABLE" = "0" ]; then
  echo "--require-driver needs init offload or runtime prefetch enabled" >&2
  exit 2
fi
if ! [[ "$WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_MAX_VICTIMS_PER_STEP" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_MAX_VICTIMS_PER_STEP must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_ROUTER_TOPK_EVERY_N_STEPS must be a non-negative integer" >&2
  exit 2
fi
if [ -n "$SERVER_MAX_NUM_BATCHED_TOKENS" ] \
  && ! [[ "$SERVER_MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "SERVER_MAX_NUM_BATCHED_TOKENS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_PREFETCH_MAX_BYTES_PER_STEP must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_SOURCE" =~ ^(router_topk|plan_intersection|oracle_replay)$ ]]; then
  echo "WEIGHT_PREFETCH_SOURCE must be router_topk, plan_intersection, or oracle_replay" >&2
  exit 2
fi
if ! [[ "$ORACLE_PREFETCH_LOOKAHEAD_STEPS" =~ ^[0-9]+$ ]]; then
  echo "ORACLE_PREFETCH_LOOKAHEAD_STEPS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP must be a non-negative integer" >&2
  exit 2
fi
if [ -z "$INIT_WEIGHT_MAX_BYTES" ]; then
  INIT_WEIGHT_MAX_BYTES="$WEIGHT_GPU_BYTES"
fi
if ! [[ "$INIT_WEIGHT_MAX_BYTES" =~ ^[0-9]+$ ]]; then
  echo "INIT_WEIGHT_MAX_BYTES must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]]; then
  echo "INIT_WEIGHT_MAX_EXPERTS_PER_LAYER must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES" =~ ^[0-9]+$ ]]; then
  echo "INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$INIT_WEIGHT_HOT_PREFETCH_DEVICE" =~ ^-?[0-9]+$ ]]; then
  echo "INIT_WEIGHT_HOT_PREFETCH_DEVICE must be an integer" >&2
  exit 2
fi
if [ "$WEIGHT_PREFETCH_SOURCE" = "oracle_replay" ] && [ -z "$ORACLE_EXPERT_TRACE_FILE" ]; then
  echo "ORACLE_EXPERT_TRACE_FILE is required when WEIGHT_PREFETCH_SOURCE=oracle_replay" >&2
  exit 2
fi
if ! [[ "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" =~ ^[0-9]+$ ]]; then
  echo "WEIGHT_EVICT_PROTECT_RECENT_STEPS must be a non-negative integer" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_f_check.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage M Phase F check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

PROFILE_DIR="$RUN_ROOT/profile"
CHECK_DIR="$RUN_ROOT/routing_phase_f_full_impl"
PLAN_JSON="$RUN_ROOT/routing_phase_f_weight_plan.json"
PLAN_REPORT="$RUN_ROOT/routing_phase_f_weight_plan_report.md"
CHECK_JSON="$RUN_ROOT/phase_f_check.json"
CHECK_MD="$RUN_ROOT/phase_f_check.md"
PHASE_F_DEBUG_LOG="$RUN_ROOT/phase_f_user_debug.log"
mkdir -p "$PROFILE_DIR" "$CHECK_DIR"
export VLLM_UVM_PHASE_F_DEBUG_LOG="$PHASE_F_DEBUG_LOG"
SERVER_BATCH_TOKEN_ARGS=()
if [ -n "$SERVER_MAX_NUM_BATCHED_TOKENS" ]; then
  SERVER_BATCH_TOKEN_ARGS=(
    --server-max-num-batched-tokens "$SERVER_MAX_NUM_BATCHED_TOKENS"
  )
fi

echo "==========================================================="
echo " Stage M Phase F Check"
echo " run_root: $RUN_ROOT"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " server_max_num_seqs: $SERVER_MAX_NUM_SEQS"
echo " server_max_num_batched_tokens: ${SERVER_MAX_NUM_BATCHED_TOKENS:-<unset>}"
echo " routing_trace: ${ROUTING_TRACE:-<generate>}"
echo " weight_plan_file: ${WEIGHT_PLAN_FILE:-<generate>}"
echo " weight_executor: $WEIGHT_EXECUTOR"
echo " require_driver: $REQUIRE_DRIVER"
echo " scratch_device_direct_enable: $SCRATCH_DEVICE_DIRECT_ENABLE"
echo " scratch_pool_enable: $SCRATCH_POOL_ENABLE"
echo " residency_enable: $RESIDENCY_ENABLE"
echo " residency_trace_enable: $RESIDENCY_TRACE_ENABLE"
echo " kv_runtime_enable: $KV_RUNTIME_ENABLE"
echo " weight_runtime_prefetch_enable: $WEIGHT_RUNTIME_PREFETCH_ENABLE"
echo " weight_prefetch_trace_enable: $WEIGHT_PREFETCH_TRACE_ENABLE"
echo " weight_prefetch_max_bytes_per_step: $WEIGHT_PREFETCH_MAX_BYTES_PER_STEP"
echo " weight_prefetch_max_experts_per_layer: $WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER"
echo " weight_router_topk_every_n_steps: $WEIGHT_ROUTER_TOPK_EVERY_N_STEPS"
echo " weight_prefetch_skip_resident: $WEIGHT_PREFETCH_SKIP_RESIDENT"
echo " weight_prefetch_expert_cache_enable: $WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE"
echo " weight_prefetch_region_cooldown_steps: $WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS"
echo " weight_prefetch_source: $WEIGHT_PREFETCH_SOURCE"
echo " oracle_expert_trace_file: ${ORACLE_EXPERT_TRACE_FILE:-<none>}"
echo " oracle_prefetch_lookahead_steps: $ORACLE_PREFETCH_LOOKAHEAD_STEPS"
echo " weight_prefetch_hit_trace_enable: $WEIGHT_PREFETCH_HIT_TRACE_ENABLE"
echo " weight_prefetch_hit_driver_query_enable: $WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE"
echo " weight_prefetch_hit_max_regions_per_step: $WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP"
echo " init_weight_offload_enable: $INIT_WEIGHT_OFFLOAD_ENABLE"
echo " init_weight_max_bytes: $INIT_WEIGHT_MAX_BYTES"
echo " init_weight_max_experts_per_layer: $INIT_WEIGHT_MAX_EXPERTS_PER_LAYER"
echo " init_weight_hot_prefetch_enable: $INIT_WEIGHT_HOT_PREFETCH_ENABLE"
echo " init_weight_hot_prefetch_sync: $INIT_WEIGHT_HOT_PREFETCH_SYNC"
echo " init_weight_hot_prefetch_max_bytes: $INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES"
echo " init_weight_hot_prefetch_device: $INIT_WEIGHT_HOT_PREFETCH_DEVICE"
echo " init_weight_hot_prefetch_real_query_enable: $INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE"
echo " hot_residency_stage_trace_enable: $HOT_RESIDENCY_STAGE_TRACE_ENABLE"
echo " hot_residency_stage_max_num_tokens: $HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS"
echo " hot_residency_stage_device: $HOT_RESIDENCY_STAGE_DEVICE"
echo " pool_coordinator_enable: $POOL_COORDINATOR_ENABLE"
echo " pool_coordinator_trace_enable: $POOL_COORDINATOR_TRACE_ENABLE"
echo " pool_registry_enable: $POOL_REGISTRY_ENABLE"
echo " auto_gap_watch_enable: $AUTO_GAP_WATCH_ENABLE"
echo " fault_address_log_enable: $FAULT_ADDRESS_LOG_ENABLE"
echo " weight_active_evict_enable: $WEIGHT_ACTIVE_EVICT_ENABLE"
echo " weight_active_evict_max_experts_per_layer: $WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER"
echo " weight_evict_protect_recent_steps: $WEIGHT_EVICT_PROTECT_RECENT_STEPS"
echo " weight_max_victims_per_step: $WEIGHT_MAX_VICTIMS_PER_STEP"
echo " phase_f_user_debug_log: $PHASE_F_DEBUG_LOG"
echo "==========================================================="

echo "==========================================================="
echo " [1/5] Build UVM allocator shared library with Phase F API"
echo "==========================================================="
make -B -C "$SCRIPT_DIR/vllm/uvm_test" uvm

python3 - "$SCRIPT_DIR" <<'PY'
import ctypes
import sys
from pathlib import Path

script_dir = Path(sys.argv[1])
lib_path = script_dir / "vllm" / "uvm_test" / "uvm_allocator.so"
lib = ctypes.CDLL(str(lib_path))
missing = [
    name for name in (
        "uvm_batch_prefetch",
        "uvm_batch_advise",
        "uvm_driver_batch_prefetch_cpu",
        "uvm_driver_batch_prefetch_gpu",
        "uvm_driver_batch_advise_cpu",
        "uvm_driver_batch_query_residency",
    )
    if not hasattr(lib, name)
]
if missing:
    raise SystemExit(f"missing phase_f symbols: {missing}")
print(f"phase_f_symbols_ok={lib_path}")
PY

if [ -z "$ROUTING_TRACE" ]; then
  echo "==========================================================="
  echo " [2/5] Generate routing trace under pure UVM"
  echo "==========================================================="
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --with-address-log \
    --allocator-log "$PROFILE_DIR/vllm_uvm_allocator_trace_profile.log" \
    --trace-log "$PROFILE_DIR/uvm_kv_fault_stats_profile.log" \
    --address-trace-log "$PROFILE_DIR/uvm_kv_fault_addrs_profile.log" \
    --address-log "$PROFILE_DIR/vllm_uvm_address_regions_profile.log" \
    --server-log "$PROFILE_DIR/vllm_server_profile.log" \
    --bench-log "$PROFILE_DIR/vllm_bench_profile.log" \
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
    --uvm-moe-routing-trace-file "$PROFILE_DIR/vllm_uvm_moe_routing_trace.jsonl" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    "${SERVER_BATCH_TOKEN_ARGS[@]}" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN"
  ROUTING_TRACE="$PROFILE_DIR/vllm_uvm_moe_routing_trace.jsonl"
else
  echo "==========================================================="
  echo " [2/5] Reuse routing trace"
  echo "==========================================================="
  echo "routing_trace=$ROUTING_TRACE"
fi

echo "==========================================================="
echo " [3/5] Build routing-aware weight plan"
echo "==========================================================="
if [ -n "$WEIGHT_PLAN_FILE" ]; then
  cp "$WEIGHT_PLAN_FILE" "$PLAN_JSON"
  echo "external_weight_plan=$WEIGHT_PLAN_FILE"
else
  python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
    --routing-trace "$ROUTING_TRACE" \
    --output-plan "$PLAN_JSON" \
    --report-md "$PLAN_REPORT" \
    --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
    --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
    --fallback-layers "$WEIGHT_PLAN_LAYERS" \
    --fallback-experts "$WEIGHT_PLAN_EXPERTS" \
    --min-routing-records 1
fi

echo "==========================================================="
echo " [4/5] Run full_impl with Phase F executor mode"
echo "==========================================================="
WEIGHT_BATCH_EXECUTOR_ENABLE=1 \
"$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" \
  --run-root "$CHECK_DIR" \
  --log-file "$CHECK_DIR/run.log" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --repeat 1 \
  --skip-trace-control \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  "${SERVER_BATCH_TOKEN_ARGS[@]}" \
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
  --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" \
  --weight-plan-file "$PLAN_JSON" \
  --weight-executor "$WEIGHT_EXECUTOR" \
  --uvm-enable-builtin-tests 1 \
  --no-reset-counters \
  --weight-budget-cpu-offload-enable 0 \
  --scratch-device-direct-enable "$SCRATCH_DEVICE_DIRECT_ENABLE" \
  --scratch-pool-enable "$SCRATCH_POOL_ENABLE" \
  --residency-enable "$RESIDENCY_ENABLE" \
  --residency-trace-enable "$RESIDENCY_TRACE_ENABLE" \
  --kv-runtime-enable "$KV_RUNTIME_ENABLE" \
  --weight-runtime-prefetch-enable "$WEIGHT_RUNTIME_PREFETCH_ENABLE" \
  --weight-prefetch-trace-enable "$WEIGHT_PREFETCH_TRACE_ENABLE" \
  --weight-prefetch-max-bytes-per-step "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" \
  --weight-prefetch-max-experts-per-layer "$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" \
  --weight-router-topk-every-n-steps "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS" \
  --weight-prefetch-skip-resident "$WEIGHT_PREFETCH_SKIP_RESIDENT" \
  --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
  --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
  --weight-prefetch-source "$WEIGHT_PREFETCH_SOURCE" \
  --oracle-expert-trace-file "$ORACLE_EXPERT_TRACE_FILE" \
  --oracle-prefetch-lookahead-steps "$ORACLE_PREFETCH_LOOKAHEAD_STEPS" \
  --weight-prefetch-hit-trace-enable "$WEIGHT_PREFETCH_HIT_TRACE_ENABLE" \
  --weight-prefetch-hit-driver-query-enable "$WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE" \
  --weight-prefetch-hit-max-regions-per-step "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" \
  --init-weight-offload-enable "$INIT_WEIGHT_OFFLOAD_ENABLE" \
  --init-weight-hot-prefetch-enable "$INIT_WEIGHT_HOT_PREFETCH_ENABLE" \
  --init-weight-hot-prefetch-sync "$INIT_WEIGHT_HOT_PREFETCH_SYNC" \
  --init-weight-hot-prefetch-max-bytes "$INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES" \
  --init-weight-hot-prefetch-device "$INIT_WEIGHT_HOT_PREFETCH_DEVICE" \
  --init-weight-hot-prefetch-real-query-enable "$INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE" \
  --hot-residency-stage-trace-enable "$HOT_RESIDENCY_STAGE_TRACE_ENABLE" \
  --hot-residency-stage-max-num-tokens "$HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS" \
  --hot-residency-stage-device "$HOT_RESIDENCY_STAGE_DEVICE" \
  --pool-coordinator-enable "$POOL_COORDINATOR_ENABLE" \
  --pool-coordinator-trace-enable "$POOL_COORDINATOR_TRACE_ENABLE" \
  --pool-registry-enable "$POOL_REGISTRY_ENABLE" \
  --auto-gap-watch-enable "$AUTO_GAP_WATCH_ENABLE" \
  --weight-active-evict-enable "$WEIGHT_ACTIVE_EVICT_ENABLE" \
  --weight-active-evict-max-experts-per-layer "$WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER" \
  --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
  --weight-max-victims-per-step "$WEIGHT_MAX_VICTIMS_PER_STEP" \
  --fault-address-log-enable "$FAULT_ADDRESS_LOG_ENABLE"

echo "==========================================================="
echo " [5/5] Validate Phase F traces"
echo "==========================================================="
python3 - "$CHECK_DIR" "$CHECK_JSON" "$CHECK_MD" "$REQUIRE_DRIVER" "$WEIGHT_EXECUTOR" "$INIT_WEIGHT_OFFLOAD_ENABLE" "$WEIGHT_RUNTIME_PREFETCH_ENABLE" "$WEIGHT_PREFETCH_TRACE_ENABLE" "$WEIGHT_PREFETCH_SOURCE" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

check_dir = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
require_driver = sys.argv[4] == "1"
requested_executor = sys.argv[5]
init_weight_offload_enabled = sys.argv[6] == "1"
runtime_prefetch_enabled = sys.argv[7] == "1"
weight_prefetch_trace_enabled = sys.argv[8] == "1"
requested_prefetch_source = sys.argv[9]
run_dir = check_dir / "runs" / "full_impl_enforce_rep1"
prefetch_trace = run_dir / "vllm_uvm_weight_prefetch_full_impl_enforce_rep1.jsonl"
init_trace = run_dir / "vllm_uvm_init_weight_offload_full_impl_enforce_rep1.jsonl"
summary_csv = check_dir / "summary.csv"

def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

prefetch_records = read_jsonl(prefetch_trace)
init_records = read_jsonl(init_trace)

prefetch_actions = Counter(str(r.get("action") or "") for r in prefetch_records)
prefetch_sources = Counter(
    str(r.get("prefetch_source"))
    for r in prefetch_records
    if r.get("prefetch_source") is not None
)
prefetch_executors = Counter(
    str(r.get("executor")) for r in prefetch_records if r.get("executor")
)
init_actions = Counter(str(r.get("action") or "") for r in init_records)
init_executors = Counter(
    str(r.get("executor")) for r in init_records if r.get("executor")
)
offload_actions = Counter(
    action for action in prefetch_actions if action.startswith("offload_")
)

summary_row = {}
if summary_csv.exists():
    with summary_csv.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                summary_row = row
                break

def as_int(value):
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None

failed_requests = as_int(summary_row.get("failed_requests"))
exit_code = as_int(summary_row.get("exit_code"))
runtime_offload_issued = as_int(summary_row.get("weight_offload_prefetch_cpu_issued"))

driver_records = init_executors.get("driver_batch", 0) + prefetch_executors.get("driver_batch", 0)
fallback_records = (
    init_executors.get("cpp_batch_fallback", 0)
    + prefetch_executors.get("cpp_batch_fallback", 0)
)
cpp_records = init_executors.get("cpp_batch", 0) + prefetch_executors.get("cpp_batch", 0)
accepted_executors = {"driver_batch", "cpp_batch_fallback"}
if requested_executor == "cpp_batch":
    accepted_executors.add("cpp_batch")
init_batch_records = sum(init_executors.get(name, 0) for name in accepted_executors)
runtime_batch_records = sum(prefetch_executors.get(name, 0) for name in accepted_executors)
runtime_summary_records = prefetch_actions.get("step_summary", 0)
runtime_candidate_records = sum(
    int(r.get("candidate_region_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_candidate_expert_records = sum(
    int(r.get("candidate_expert_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_expert_resident_skip_records = sum(
    int(r.get("expert_resident_skip_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_resident_skip_records = sum(
    int(r.get("resident_skip_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_cooldown_skip_records = sum(
    int(r.get("cooldown_skip_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_oracle_replay_records = sum(
    int(r.get("oracle_replay_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_oracle_match_records = sum(
    int(r.get("oracle_match_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_oracle_mismatch_records = sum(
    int(r.get("oracle_mismatch_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_oracle_missing_records = sum(
    int(r.get("oracle_trace_missing_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_access_probe_records = sum(
    int(r.get("prefetch_access_probe_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_before_access_candidates = sum(
    int(r.get("prefetch_before_access_candidate_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_hit_before_access_estimated = sum(
    int(r.get("prefetch_hit_before_access_estimated_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_miss_before_access_estimated = sum(
    int(r.get("prefetch_miss_before_access_estimated_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_same_step_access = sum(
    int(r.get("prefetch_same_step_access_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_access_no_prior_prefetch = sum(
    int(r.get("prefetch_access_no_prior_prefetch_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_real_query_records = sum(
    int(r.get("prefetch_hit_before_access_real_query_records") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_real_query_ranges = sum(
    int(r.get("prefetch_hit_before_access_real_query_ranges") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_real_queried_bytes = sum(
    int(r.get("prefetch_hit_before_access_real_queried_bytes") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_real_gpu_bytes = sum(
    int(r.get("prefetch_hit_before_access_real_gpu_bytes") or 0)
    for r in prefetch_records
    if r.get("action") == "step_summary"
)
runtime_prefetch_hit_estimated_rate = (
    runtime_prefetch_hit_before_access_estimated
    / runtime_prefetch_before_access_candidates
    if runtime_prefetch_before_access_candidates
    else None
)
runtime_prefetch_real_gpu_ratio = (
    runtime_prefetch_real_gpu_bytes / runtime_prefetch_real_queried_bytes
    if runtime_prefetch_real_queried_bytes
    else None
)
runtime_filtered_noop_records = (
    runtime_resident_skip_records + runtime_expert_resident_skip_records
)
init_batch_sizes = [
    int(r.get("batch_size", 0))
    for r in init_records
    if r.get("executor") in accepted_executors
]
runtime_batch_sizes = [
    int(r.get("batch_size", 0))
    for r in prefetch_records
    if r.get("executor") in accepted_executors
]
driver_results = [
    r.get(key)
    for records in (init_records, prefetch_records)
    for r in records
    for key in ("driver_result", "driver_advise_result", "driver_prefetch_result")
    if r.get(key) not in (None, "")
]
driver_success_records = sum(
    1
    for records in (init_records, prefetch_records)
    for r in records
    if r.get("executor") == "driver_batch" and r.get("success") is True
)
driver_positive_results = sum(
    1
    for value in driver_results
    if isinstance(value, (int, float)) and value > 0
)

checks = {
    "prefetch_trace_present": (
        prefetch_trace.exists()
        if runtime_prefetch_enabled and weight_prefetch_trace_enabled
        else True
    ),
    "init_trace_present": init_trace.exists() if init_weight_offload_enabled else True,
    "init_phase_f_executor_seen": (
        init_batch_records > 0 if init_weight_offload_enabled else init_batch_records == 0
    ),
    "runtime_phase_f_executor_seen": (
        runtime_batch_records > 0 or runtime_filtered_noop_records > 0
        if runtime_prefetch_enabled and weight_prefetch_trace_enabled
        else True
    ),
    "init_batch_groups_are_multi_range": (
        max(init_batch_sizes or [0]) > 1 if init_weight_offload_enabled else True
    ),
    "runtime_batch_groups_are_multi_range": (
        max(runtime_batch_sizes or [0]) > 1 or runtime_filtered_noop_records > 0
        if runtime_prefetch_enabled and weight_prefetch_trace_enabled
        else True
    ),
    "driver_batch_seen_or_fallback_recorded": driver_records > 0 or fallback_records > 0,
    "driver_action_success_seen": (
        driver_success_records > 0 or driver_positive_results > 0
    ),
    "require_driver_satisfied": (
        driver_records > 0
        and fallback_records == 0
        and (driver_success_records > 0 or driver_positive_results > 0)
    ) if require_driver else True,
    "requested_prefetch_source_present": (
        prefetch_sources.get(requested_prefetch_source, 0) > 0
        if runtime_prefetch_enabled and weight_prefetch_trace_enabled
        else True
    ),
    "oracle_replay_matchable": (
        runtime_oracle_replay_records > 0
        and runtime_oracle_missing_records == 0
        if runtime_prefetch_enabled
        and weight_prefetch_trace_enabled
        and requested_prefetch_source == "oracle_replay"
        else True
    ),
    "runtime_cold_offload_disabled": runtime_offload_issued == 0,
    "no_runtime_offload_actions": sum(offload_actions.values()) == 0,
    "benchmark_exit_zero": exit_code == 0,
    "benchmark_failed_requests_zero": failed_requests == 0,
}
passed = all(checks.values())
payload = {
    "passed": passed,
    "checks": checks,
    "init_actions": dict(init_actions),
    "init_executors": dict(init_executors),
    "prefetch_actions": dict(prefetch_actions),
    "prefetch_sources": dict(prefetch_sources),
    "prefetch_executors": dict(prefetch_executors),
    "driver_records": driver_records,
    "fallback_records": fallback_records,
    "cpp_records": cpp_records,
    "driver_success_records": driver_success_records,
    "driver_positive_results": driver_positive_results,
    "driver_results": driver_results[:20],
    "requested_executor": requested_executor,
    "requested_prefetch_source": requested_prefetch_source,
    "require_driver": require_driver,
    "init_weight_offload_enabled": init_weight_offload_enabled,
    "runtime_prefetch_enabled": runtime_prefetch_enabled,
    "max_init_batch_size": max(init_batch_sizes or [0]),
    "max_runtime_batch_size": max(runtime_batch_sizes or [0]),
    "runtime_summary_records": runtime_summary_records,
    "runtime_candidate_expert_records": runtime_candidate_expert_records,
    "runtime_expert_resident_skip_records": runtime_expert_resident_skip_records,
    "runtime_filtered_noop_records": runtime_filtered_noop_records,
    "runtime_candidate_records": runtime_candidate_records,
    "runtime_resident_skip_records": runtime_resident_skip_records,
    "runtime_cooldown_skip_records": runtime_cooldown_skip_records,
    "runtime_oracle_replay_records": runtime_oracle_replay_records,
    "runtime_oracle_match_records": runtime_oracle_match_records,
    "runtime_oracle_mismatch_records": runtime_oracle_mismatch_records,
    "runtime_oracle_missing_records": runtime_oracle_missing_records,
    "runtime_prefetch_access_probe_records": runtime_prefetch_access_probe_records,
    "runtime_prefetch_before_access_candidates": runtime_prefetch_before_access_candidates,
    "runtime_prefetch_hit_before_access_estimated": runtime_prefetch_hit_before_access_estimated,
    "runtime_prefetch_miss_before_access_estimated": runtime_prefetch_miss_before_access_estimated,
    "runtime_prefetch_hit_before_access_estimated_rate": runtime_prefetch_hit_estimated_rate,
    "runtime_prefetch_same_step_access": runtime_prefetch_same_step_access,
    "runtime_prefetch_access_no_prior_prefetch": runtime_prefetch_access_no_prior_prefetch,
    "runtime_prefetch_hit_before_access_real_query_records": runtime_prefetch_real_query_records,
    "runtime_prefetch_hit_before_access_real_query_ranges": runtime_prefetch_real_query_ranges,
    "runtime_prefetch_hit_before_access_real_queried_bytes": runtime_prefetch_real_queried_bytes,
    "runtime_prefetch_hit_before_access_real_gpu_bytes": runtime_prefetch_real_gpu_bytes,
    "runtime_prefetch_hit_before_access_real_gpu_ratio": runtime_prefetch_real_gpu_ratio,
    "failed_requests": failed_requests,
    "exit_code": exit_code,
    "runtime_offload_issued": runtime_offload_issued,
    "prefetch_trace": str(prefetch_trace),
    "init_trace": str(init_trace),
    "summary_csv": str(summary_csv),
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

lines = ["# Stage M Phase F Check", "", f"- passed: {passed}", ""]
lines.append("## Checks")
for key, value in checks.items():
    lines.append(f"- {key}: {value}")
lines.extend([
    "",
    "## Evidence",
    f"- init_executors: {dict(init_executors)}",
    f"- prefetch_executors: {dict(prefetch_executors)}",
    f"- requested_executor: {requested_executor}",
    f"- requested_prefetch_source: {requested_prefetch_source}",
    f"- driver_records: {driver_records}",
    f"- fallback_records: {fallback_records}",
    f"- driver_success_records: {driver_success_records}",
    f"- driver_positive_results: {driver_positive_results}",
    f"- sample_driver_results: {driver_results[:8]}",
    f"- max_init_batch_size: {payload['max_init_batch_size']}",
    f"- max_runtime_batch_size: {payload['max_runtime_batch_size']}",
    f"- runtime_summary_records: {runtime_summary_records}",
    f"- runtime_candidate_expert_records: {runtime_candidate_expert_records}",
    f"- runtime_expert_resident_skip_records: {runtime_expert_resident_skip_records}",
    f"- runtime_filtered_noop_records: {runtime_filtered_noop_records}",
    f"- runtime_candidate_records: {runtime_candidate_records}",
    f"- runtime_resident_skip_records: {runtime_resident_skip_records}",
    f"- runtime_cooldown_skip_records: {runtime_cooldown_skip_records}",
    f"- runtime_oracle_replay_records: {runtime_oracle_replay_records}",
    f"- runtime_oracle_match_records: {runtime_oracle_match_records}",
    f"- runtime_oracle_mismatch_records: {runtime_oracle_mismatch_records}",
    f"- runtime_oracle_missing_records: {runtime_oracle_missing_records}",
    f"- runtime_prefetch_access_probe_records: {runtime_prefetch_access_probe_records}",
    f"- runtime_prefetch_before_access_candidates: {runtime_prefetch_before_access_candidates}",
    f"- runtime_prefetch_hit_before_access_estimated: {runtime_prefetch_hit_before_access_estimated}",
    f"- runtime_prefetch_miss_before_access_estimated: {runtime_prefetch_miss_before_access_estimated}",
    f"- runtime_prefetch_hit_before_access_estimated_rate: {runtime_prefetch_hit_estimated_rate}",
    f"- runtime_prefetch_same_step_access: {runtime_prefetch_same_step_access}",
    f"- runtime_prefetch_access_no_prior_prefetch: {runtime_prefetch_access_no_prior_prefetch}",
    f"- runtime_prefetch_hit_before_access_real_query_records: {runtime_prefetch_real_query_records}",
    f"- runtime_prefetch_hit_before_access_real_query_ranges: {runtime_prefetch_real_query_ranges}",
    f"- runtime_prefetch_hit_before_access_real_queried_bytes: {runtime_prefetch_real_queried_bytes}",
    f"- runtime_prefetch_hit_before_access_real_gpu_bytes: {runtime_prefetch_real_gpu_bytes}",
    f"- runtime_prefetch_hit_before_access_real_gpu_ratio: {runtime_prefetch_real_gpu_ratio}",
    f"- prefetch_sources: {dict(prefetch_sources)}",
    f"- runtime_offload_issued: {runtime_offload_issued}",
])
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY

cat "$CHECK_MD"

echo "Stage M Phase F check finished."
echo "run_root: $RUN_ROOT"
echo "check_json: $CHECK_JSON"
echo "check_md: $CHECK_MD"
