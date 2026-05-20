#!/usr/bin/env bash
# A/B comparison: pure UVM baseline vs full UVM pool implementation.
#
# This runner is intentionally different from
# run_stage_l_with_scratch_direct_diagnosis.sh. The diagnosis runner keeps many
# Stage L paths in trace_only mode. This runner uses enforce/prefetch modes for
# the treatment so KV, weight, and scratch limits can actually affect placement.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REPEAT="${REPEAT:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_l_full_impl_uvm_ab_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""

RUN_TRACE_CONTROL="${RUN_TRACE_CONTROL:-1}"
FAIL_FAST="${FAIL_FAST:-0}"

SCRATCH_PROFILE="${SCRATCH_PROFILE:-16m}"
SCRATCH_TARGET_PHASES="${SCRATCH_TARGET_PHASES:-enabled:init,enabled:warmup,enabled:dummy_run,enabled:prefill_step,enabled:model_forward,enabled:attention,enabled:moe}"
SCRATCH_TARGET_CLASS="${SCRATCH_TARGET_CLASS:-gap_hot_runtime_scratch}"
SCRATCH_TARGET_GAP="${SCRATCH_TARGET_GAP:-2}"
SCRATCH_MIN_BYTES="${SCRATCH_MIN_BYTES:-4096}"
SCRATCH_BACKEND="${SCRATCH_BACKEND:-cuda_malloc_async}"
SCRATCH_POOL_RELEASE_THRESHOLD="${SCRATCH_POOL_RELEASE_THRESHOLD:-1048576}"
SCRATCH_MAX_TOTAL_BYTES=""
SCRATCH_MAX_BYTES=""

SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-4}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.75}"

GLOBAL_GPU_BYTES="${GLOBAL_GPU_BYTES:-25769803776}"       # 24 GiB
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-6442450944}"        # 6 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-1073741824}"                # 1 GiB
STAGE_L_SCRATCH_GPU_BYTES="${STAGE_L_SCRATCH_GPU_BYTES:-1073741824}"
SAFETY_MARGIN_BYTES="${SAFETY_MARGIN_BYTES:-2147483648}" # 2 GiB

KV_RUNTIME_BUDGET_BLOCKS="${KV_RUNTIME_BUDGET_BLOCKS:-512}"
KV_RUNTIME_CANDIDATE_LIMIT="${KV_RUNTIME_CANDIDATE_LIMIT:-32}"
KV_MAX_OFFLOAD_BLOCKS="${KV_MAX_OFFLOAD_BLOCKS:-16}"

WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_PLAN_FILE="${WEIGHT_PLAN_FILE:-}"
WEIGHT_PLAN_IS_EXTERNAL=0
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-33554432}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-2}"
WEIGHT_PREFETCH_TRACE_ENABLE="${WEIGHT_PREFETCH_TRACE_ENABLE:-1}"
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
WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP="${WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP:-268435456}"
WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="${WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER:-8}"
WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-16}"
WEIGHT_VICTIM_ACTION="${WEIGHT_VICTIM_ACTION:-offload_cpu}"
WEIGHT_ACTIVE_EVICT_ENABLE="${WEIGHT_ACTIVE_EVICT_ENABLE:-0}"
WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-0}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"
WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="${WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE:-0}"
INIT_KV_MIN_BLOCKS="${INIT_KV_MIN_BLOCKS:-128}"
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-8589934592}"
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-16}"
INIT_WEIGHT_HOT_PREFETCH_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_ENABLE:-0}"
INIT_WEIGHT_HOT_PREFETCH_SYNC="${INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE:-0}"
HOT_RESIDENCY_STAGE_TRACE_ENABLE="${HOT_RESIDENCY_STAGE_TRACE_ENABLE:-0}"
HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="${HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS:-1}"
HOT_RESIDENCY_STAGE_DEVICE="${HOT_RESIDENCY_STAGE_DEVICE:--1}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"

COORDINATOR_GLOBAL_BYTES_PER_STEP="${COORDINATOR_GLOBAL_BYTES_PER_STEP:-536870912}"
COORDINATOR_KV_BYTES_PER_STEP="${COORDINATOR_KV_BYTES_PER_STEP:-268435456}"
COORDINATOR_WEIGHT_BYTES_PER_STEP="${COORDINATOR_WEIGHT_BYTES_PER_STEP:-268435456}"
COORDINATOR_SCRATCH_BYTES_PER_STEP="${COORDINATOR_SCRATCH_BYTES_PER_STEP:-16777216}"
COORDINATOR_PRIORITY="${COORDINATOR_PRIORITY:-kv,scratch,weights}"
POOL_COORDINATOR_TRACE_ENABLE="${POOL_COORDINATOR_TRACE_ENABLE:-1}"

SCRATCH_DEVICE_DIRECT_ENABLE="${SCRATCH_DEVICE_DIRECT_ENABLE:-1}"
SCRATCH_POOL_ENABLE="${SCRATCH_POOL_ENABLE:-1}"
RESIDENCY_ENABLE="${RESIDENCY_ENABLE:-1}"
RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE:-1}"
KV_RUNTIME_ENABLE="${KV_RUNTIME_ENABLE:-1}"
WEIGHT_RUNTIME_PREFETCH_ENABLE="${WEIGHT_RUNTIME_PREFETCH_ENABLE:-1}"
INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE:-1}"
POOL_COORDINATOR_ENABLE="${POOL_COORDINATOR_ENABLE:-1}"
POOL_REGISTRY_ENABLE="${POOL_REGISTRY_ENABLE:-1}"
AUTO_GAP_WATCH_ENABLE="${AUTO_GAP_WATCH_ENABLE:-1}"
FAULT_ADDRESS_LOG_ENABLE="${FAULT_ADDRESS_LOG_ENABLE:-1}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_l_full_impl_uvm_ab.sh [options]

Purpose:
  Compare a pure UVM baseline against the full implementation where Stage C
  scratch device-direct, Stage L KV residency/offload, Stage J KV runtime
  budget, Stage I/L weight prefetch/offload, and Stage K coordinator are active.

Main variants:
  pure_uvm_baseline
    Policy/residency/runtime/direct/coordinator are disabled. This is the
    closest "no optimization" UVM baseline in this runner.

  uvm_trace_control
    Optional instrumentation control. It enables policy/gap-watch trace but
    does not perform direct allocation or Stage L actions. This helps separate
    instrumentation overhead from optimization behavior.

  full_impl_enforce
    Full treatment. Uses scratch device-direct, residency enforce, KV runtime
    enforce, KV prefetch_cpu/prefetch, weight prefetch/offload, scratch pool
    enforce, pool registry, and coordinator enforce.

Common options:
  --run-root <path>
  --log-file <path>
  --model <name>                         default: Qwen/Qwen3-30B-A3B-FP8
  --prompts <n>                          default: 4
  --output-len <n>                       default: 256
  --request-rate <f>                     default: 2
  --repeat <n>                           default: 1
  --fail-fast                            stop after the first failed variant
  --skip-trace-control                   only run baseline and full treatment

Server pressure controls:
  --server-max-num-seqs <n>              default: 4
  --server-gpu-memory-utilization <f>    default: 0.75

Scratch direct controls:
  --scratch-profile 4m|16m|custom        default: 16m
  --scratch-total-bytes <n>              required with custom
  --scratch-max-bytes <n>                required with custom
  --scratch-backend cuda_malloc_async|cuda_malloc
  --scratch-target-phases <csv>          default includes warmup/dummy/prefill/attention/moe

Stage L pool budgets:
  --global-gpu-bytes <n>                 default: 24GiB
  --weight-gpu-bytes <n>                 default: 6GiB
  --kv-gpu-bytes <n>                     default: 1GiB
  --stage-l-scratch-gpu-bytes <n>        default: 1GiB
  --safety-margin-bytes <n>              default: 2GiB

Runtime enforcement controls:
  --kv-runtime-budget-blocks <n>         default: 512
  --kv-max-offload-blocks <n>            default: 16
  --weight-plan-layers <n>               default: 48
  --weight-plan-experts <n>              default: 128
  --weight-plan-file <path>              optional external prefetch/offload plan
  --weight-executor <mode>               cpp_batch|driver_batch|auto (default: cpp_batch)
  --weight-prefetch-trace-enable <0|1>   write runtime weight prefetch JSONL
  --weight-router-topk-every-n-steps <n> run router_topk prefetch every n MoE calls
  --weight-prefetch-skip-resident <0|1>  skip already-GPU regions before coordinator
  --weight-prefetch-expert-cache-enable <0|1>
                                            skip roles for fully GPU-resident experts
  --weight-prefetch-region-cooldown-steps <n>
                                         avoid repeated prefetch for same region
  --weight-prefetch-source <source>      router_topk|plan_intersection|oracle_replay
  --oracle-expert-trace-file <path>      routing trace JSONL for oracle_replay
  --oracle-prefetch-lookahead-steps <n>  per-layer replay lookahead
  --uvm-enable-builtin-tests <0|1>       keep UVM test ioctl enabled across
                                         counter-reset reloads (default: 0)
  --no-reset-counters                    do not reload nvidia_uvm between runs
  --init-kv-min-blocks <n>               default: 128
  --init-weight-max-bytes <n>            default: 8GiB
  --init-weight-max-experts-per-layer <n> default: 16
  --weight-max-victims-per-step <n>      default: 16
  --weight-active-evict-enable <0|1>     evict non-active experts each step
  --weight-active-evict-max-experts-per-layer <n>
                                         cap non-active expert scan/offload
  --weight-evict-protect-recent-steps <n>
                                         protect recently active regions
  --weight-budget-cpu-offload-enable <0|1>
                                         default: 0
  --scratch-device-direct-enable <0|1>   full_impl scratch device-direct action
  --scratch-pool-enable <0|1>            full_impl scratch pool enforcement
  --residency-enable <0|1>               full_impl Stage L residency policy
  --residency-trace-enable <0|1>         write Stage L residency JSONL
  --kv-runtime-enable <0|1>              full_impl Stage J runtime KV policy
  --weight-runtime-prefetch-enable <0|1> full_impl runtime weight prefetch
  --init-weight-offload-enable <0|1>     full_impl init cold-weight offload
  --pool-coordinator-enable <0|1>        full_impl Stage K coordinator
  --pool-coordinator-trace-enable <0|1>  write Stage K coordinator JSONL
  --pool-registry-enable <0|1>           full_impl pool registry
  --auto-gap-watch-enable <0|1>          auto gap-watch probe/main flow
  --fault-address-log-enable <0|1>       kernel per-fault address trace

Outputs:
  <run-root>/summary.csv
  <run-root>/summary.md
  <run-root>/runs/<variant>_rep<n>/*

Recommended first run:
  ./run_stage_l_full_impl_uvm_ab.sh \
    --log-file log/stage_l_full_impl_uvm_ab.log
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --skip-trace-control) RUN_TRACE_CONTROL=0; shift ;;
    --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --scratch-profile) SCRATCH_PROFILE="$2"; shift 2 ;;
    --scratch-total-bytes) SCRATCH_MAX_TOTAL_BYTES="$2"; SCRATCH_PROFILE="custom"; shift 2 ;;
    --scratch-max-bytes) SCRATCH_MAX_BYTES="$2"; SCRATCH_PROFILE="custom"; shift 2 ;;
    --scratch-backend) SCRATCH_BACKEND="$2"; shift 2 ;;
    --scratch-target-phases) SCRATCH_TARGET_PHASES="$2"; shift 2 ;;
    --global-gpu-bytes) GLOBAL_GPU_BYTES="$2"; shift 2 ;;
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --stage-l-scratch-gpu-bytes) STAGE_L_SCRATCH_GPU_BYTES="$2"; shift 2 ;;
    --safety-margin-bytes) SAFETY_MARGIN_BYTES="$2"; shift 2 ;;
    --kv-runtime-budget-blocks) KV_RUNTIME_BUDGET_BLOCKS="$2"; shift 2 ;;
    --kv-max-offload-blocks) KV_MAX_OFFLOAD_BLOCKS="$2"; shift 2 ;;
    --weight-plan-layers) WEIGHT_PLAN_LAYERS="$2"; shift 2 ;;
    --weight-plan-experts) WEIGHT_PLAN_EXPERTS="$2"; shift 2 ;;
    --weight-plan-file) WEIGHT_PLAN_FILE="$2"; shift 2 ;;
    --weight-executor) WEIGHT_EXECUTOR="$2"; shift 2 ;;
    --weight-prefetch-max-bytes-per-step) WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="$2"; shift 2 ;;
    --weight-prefetch-max-experts-per-layer) WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-prefetch-trace-enable) WEIGHT_PREFETCH_TRACE_ENABLE="$2"; shift 2 ;;
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
    --uvm-enable-builtin-tests) UVM_ENABLE_BUILTIN_TESTS="$2"; shift 2 ;;
    --no-reset-counters) RESET_COUNTERS=0; shift ;;
    --init-kv-min-blocks) INIT_KV_MIN_BLOCKS="$2"; shift 2 ;;
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
    --weight-max-victims-per-step) WEIGHT_MAX_VICTIMS_PER_STEP="$2"; shift 2 ;;
    --weight-victim-action) WEIGHT_VICTIM_ACTION="$2"; shift 2 ;;
    --weight-active-evict-enable) WEIGHT_ACTIVE_EVICT_ENABLE="$2"; shift 2 ;;
    --weight-active-evict-max-experts-per-layer) WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-evict-protect-recent-steps) WEIGHT_EVICT_PROTECT_RECENT_STEPS="$2"; shift 2 ;;
    --weight-budget-cpu-offload-enable) WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="$2"; shift 2 ;;
    --scratch-device-direct-enable) SCRATCH_DEVICE_DIRECT_ENABLE="$2"; shift 2 ;;
    --scratch-pool-enable) SCRATCH_POOL_ENABLE="$2"; shift 2 ;;
    --residency-enable) RESIDENCY_ENABLE="$2"; shift 2 ;;
    --residency-trace-enable) RESIDENCY_TRACE_ENABLE="$2"; shift 2 ;;
    --kv-runtime-enable) KV_RUNTIME_ENABLE="$2"; shift 2 ;;
    --weight-runtime-prefetch-enable) WEIGHT_RUNTIME_PREFETCH_ENABLE="$2"; shift 2 ;;
    --init-weight-offload-enable) INIT_WEIGHT_OFFLOAD_ENABLE="$2"; shift 2 ;;
    --pool-coordinator-enable) POOL_COORDINATOR_ENABLE="$2"; shift 2 ;;
    --pool-coordinator-trace-enable) POOL_COORDINATOR_TRACE_ENABLE="$2"; shift 2 ;;
    --pool-registry-enable) POOL_REGISTRY_ENABLE="$2"; shift 2 ;;
    --auto-gap-watch-enable) AUTO_GAP_WATCH_ENABLE="$2"; shift 2 ;;
    --fault-address-log-enable) FAULT_ADDRESS_LOG_ENABLE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SCRATCH_PROFILE" in
  4m)
    SCRATCH_MAX_TOTAL_BYTES="${SCRATCH_MAX_TOTAL_BYTES:-4194304}"
    SCRATCH_MAX_BYTES="${SCRATCH_MAX_BYTES:-4194304}"
    ;;
  16m)
    SCRATCH_MAX_TOTAL_BYTES="${SCRATCH_MAX_TOTAL_BYTES:-16777216}"
    SCRATCH_MAX_BYTES="${SCRATCH_MAX_BYTES:-8388608}"
    ;;
  custom)
    if [ -z "$SCRATCH_MAX_TOTAL_BYTES" ] || [ -z "$SCRATCH_MAX_BYTES" ]; then
      echo "--scratch-profile custom requires --scratch-total-bytes and --scratch-max-bytes" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown --scratch-profile: $SCRATCH_PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

if ! [[ "$WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE" =~ ^[01]$ ]]; then
  echo "--weight-budget-cpu-offload-enable must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$WEIGHT_ACTIVE_EVICT_ENABLE" =~ ^[01]$ ]]; then
  echo "--weight-active-evict-enable must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]]; then
  echo "--weight-active-evict-max-experts-per-layer must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS" =~ ^[0-9]+$ ]]; then
  echo "--weight-router-topk-every-n-steps must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" =~ ^[0-9]+$ ]]; then
  echo "--weight-prefetch-region-cooldown-steps must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" =~ ^[0-9]+$ ]]; then
  echo "--weight-prefetch-hit-max-regions-per-step must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" =~ ^[0-9]+$ ]]; then
  echo "--weight-evict-protect-recent-steps must be a non-negative integer" >&2
  exit 2
fi
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-cpp_batch}"
if ! [[ "$WEIGHT_EXECUTOR" =~ ^(cpp_batch|driver_batch|auto)$ ]]; then
  echo "--weight-executor must be cpp_batch, driver_batch, or auto" >&2
  exit 2
fi
UVM_ENABLE_BUILTIN_TESTS="${UVM_ENABLE_BUILTIN_TESTS:-0}"
if ! [[ "$UVM_ENABLE_BUILTIN_TESTS" =~ ^[01]$ ]]; then
  echo "--uvm-enable-builtin-tests must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$RESET_COUNTERS" =~ ^[01]$ ]]; then
  echo "RESET_COUNTERS must be 0 or 1" >&2
  exit 2
fi
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
  FAULT_ADDRESS_LOG_ENABLE; do
  if ! [[ "${!item}" =~ ^[01]$ ]]; then
    echo "$item must be 0 or 1" >&2
    exit 2
  fi
done
if [ "$AUTO_GAP_WATCH_ENABLE" = "1" ] && [ "$FAULT_ADDRESS_LOG_ENABLE" = "0" ]; then
  echo "--auto-gap-watch-enable=1 requires --fault-address-log-enable=1" >&2
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

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_l_full_impl_uvm_ab.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage L full implementation vs pure UVM A/B started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

echo "==========================================================="
echo " Stage L Full Implementation vs Pure UVM A/B"
echo " run_root: $RUN_ROOT"
echo " log_file: $LOG_FILE"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " repeat: $REPEAT"
echo " run_trace_control: $RUN_TRACE_CONTROL"
echo " scratch_profile: $SCRATCH_PROFILE"
echo " scratch_target_phases: $SCRATCH_TARGET_PHASES"
echo " scratch_total_bytes: $SCRATCH_MAX_TOTAL_BYTES"
echo " scratch_max_bytes: $SCRATCH_MAX_BYTES"
echo " scratch_backend: $SCRATCH_BACKEND"
echo " server_max_num_seqs: $SERVER_MAX_NUM_SEQS"
echo " server_gpu_memory_utilization: $SERVER_GPU_MEMORY_UTILIZATION"
echo " global_gpu_bytes: $GLOBAL_GPU_BYTES"
echo " weight_gpu_bytes: $WEIGHT_GPU_BYTES"
echo " kv_gpu_bytes: $KV_GPU_BYTES"
echo " stage_l_scratch_gpu_bytes: $STAGE_L_SCRATCH_GPU_BYTES"
echo " safety_margin_bytes: $SAFETY_MARGIN_BYTES"
echo " kv_runtime_budget_blocks: $KV_RUNTIME_BUDGET_BLOCKS"
echo " kv_max_offload_blocks: $KV_MAX_OFFLOAD_BLOCKS"
echo " weight_plan_layers: $WEIGHT_PLAN_LAYERS"
echo " weight_plan_experts: $WEIGHT_PLAN_EXPERTS"
echo " weight_plan_file: ${WEIGHT_PLAN_FILE:-<generated_broad_cold_plan>}"
echo " weight_max_victims_per_step: $WEIGHT_MAX_VICTIMS_PER_STEP"
echo " weight_victim_action: $WEIGHT_VICTIM_ACTION"
echo " weight_active_evict_enable: $WEIGHT_ACTIVE_EVICT_ENABLE"
echo " weight_active_evict_max_experts_per_layer: $WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER"
echo " weight_prefetch_trace_enable: $WEIGHT_PREFETCH_TRACE_ENABLE"
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
echo " weight_evict_protect_recent_steps: $WEIGHT_EVICT_PROTECT_RECENT_STEPS"
echo " weight_budget_cpu_offload_enable: $WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE"
echo " weight_batch_executor_enable: ${WEIGHT_BATCH_EXECUTOR_ENABLE:-1}"
echo " weight_executor: $WEIGHT_EXECUTOR"
echo " uvm_enable_builtin_tests: $UVM_ENABLE_BUILTIN_TESTS"
echo " reset_counters: $RESET_COUNTERS"
echo " init_kv_min_blocks: $INIT_KV_MIN_BLOCKS"
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
echo " scratch_device_direct_enable: $SCRATCH_DEVICE_DIRECT_ENABLE"
echo " scratch_pool_enable: $SCRATCH_POOL_ENABLE"
echo " residency_enable: $RESIDENCY_ENABLE"
echo " residency_trace_enable: $RESIDENCY_TRACE_ENABLE"
echo " kv_runtime_enable: $KV_RUNTIME_ENABLE"
echo " weight_runtime_prefetch_enable: $WEIGHT_RUNTIME_PREFETCH_ENABLE"
echo " init_weight_offload_enable: $INIT_WEIGHT_OFFLOAD_ENABLE"
echo " pool_coordinator_enable: $POOL_COORDINATOR_ENABLE"
echo " pool_coordinator_trace_enable: $POOL_COORDINATOR_TRACE_ENABLE"
echo " pool_registry_enable: $POOL_REGISTRY_ENABLE"
echo " auto_gap_watch_enable: $AUTO_GAP_WATCH_ENABLE"
echo " fault_address_log_enable: $FAULT_ADDRESS_LOG_ENABLE"
echo "==========================================================="

write_weight_plan() {
  local path="$1"
  python3 - "$path" "$WEIGHT_PLAN_LAYERS" "$WEIGHT_PLAN_EXPERTS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
layers = max(int(sys.argv[2]), 0)
experts = max(int(sys.argv[3]), 0)
offload_plan = [
    {
        "layer_id": layer_id,
        "expert_id": expert_id,
        "reason": "full_impl_ab_broad_cold_plan",
    }
    for layer_id in range(layers)
    for expert_id in range(experts)
]
plan = {
    "prefetch_plan": [],
    "offload_plan": offload_plan,
    "prefetch_plan_records": 0,
    "offload_plan_records": len(offload_plan),
}
path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
PY
}

base_args() {
  local run_dir="$1"
  local label="$2"
  printf '%s\0' \
    --model "$MODEL" \
    --mode trace \
    --allocator-log "$run_dir/vllm_uvm_allocator_trace_${label}.log" \
    --trace-log "$run_dir/uvm_kv_fault_stats_${label}.log" \
    --address-log "$run_dir/vllm_uvm_address_regions_${label}.log" \
    --server-log "$run_dir/vllm_server_${label}.log" \
    --bench-log "$run_dir/vllm_bench_${label}.log" \
    --uvm-trace-min-bytes 1048576 \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --uvm-gap-watch-name "$label" \
    --uvm-gap-watch-all-classes 1 \
    --uvm-gap-watch-min-bytes 4096 \
    --gap-watch-metrics-summary-json "$run_dir/gap_watch_metrics_${label}.json" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN"
  if [ "$FAULT_ADDRESS_LOG_ENABLE" = "1" ]; then
    printf '%s\0' \
      --with-address-log \
      --address-trace-log "$run_dir/uvm_kv_fault_addrs_${label}.log"
  fi
  if [ "$AUTO_GAP_WATCH_ENABLE" = "1" ]; then
    printf '%s\0' \
      --auto-gap-watch-enable 1 \
      --auto-gap-watch-probe-prompts 1 \
      --auto-gap-watch-target-gap "$SCRATCH_TARGET_GAP" \
      --auto-gap-watch-target-class-override "$SCRATCH_TARGET_CLASS" \
      --auto-gap-watch-summary-json "$run_dir/auto_gap_watch_probe_${label}.json" \
      --auto-gap-watch-post-main-summary-json "$run_dir/auto_gap_watch_post_main_${label}.json"
  else
    printf '%s\0' --auto-gap-watch-enable 0
  fi
  if [ "$RESET_COUNTERS" = "0" ]; then
    printf '%s\0' --no-reset-counters
  fi
  if [ "$UVM_ENABLE_BUILTIN_TESTS" = "1" ]; then
    printf '%s\0' --uvm-enable-builtin-tests "$UVM_ENABLE_BUILTIN_TESTS"
  fi
}

run_one() {
  local variant="$1"
  local rep="$2"
  local label="${variant}_rep${rep}"
  local run_dir="$RUN_ROOT/runs/$label"
  mkdir -p "$run_dir"

  echo "==========================================================="
  echo " Running $variant rep $rep/$REPEAT"
  echo " run_dir: $run_dir"
  echo "==========================================================="

  local args=()
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(base_args "$run_dir" "$label")

  case "$variant" in
    pure_uvm_baseline)
      args+=(
        --uvm-policy-enable 0
        --uvm-device-direct-enable 0
        --auto-gap-watch-policy-action-override observe
        --uvm-residency-enable 0
        --uvm-kv-runtime-enable 0
        --uvm-weight-prefetch-enable 0
        --uvm-weight-offload-enable 0
        --uvm-pool-coordinator-enable 0
        --uvm-pool-registry-enable 0
        --uvm-scratch-pool-enable 0
      )
      ;;
    uvm_trace_control)
      args+=(
        --uvm-policy-enable 1
        --uvm-device-direct-enable 0
        --uvm-device-direct-min-bytes "$SCRATCH_MIN_BYTES"
        --uvm-device-direct-max-bytes "$SCRATCH_MAX_BYTES"
        --uvm-device-direct-max-total-bytes "$SCRATCH_MAX_TOTAL_BYTES"
        --uvm-device-direct-backend "$SCRATCH_BACKEND"
        --uvm-device-direct-pool-release-threshold "$SCRATCH_POOL_RELEASE_THRESHOLD"
        --uvm-device-direct-target-phases "$SCRATCH_TARGET_PHASES"
        --auto-gap-watch-policy-action-override device_direct_trace
        --uvm-residency-enable 0
        --uvm-kv-runtime-enable 0
        --uvm-weight-prefetch-enable 0
        --uvm-weight-offload-enable 0
        --uvm-pool-coordinator-enable 0
        --uvm-pool-registry-enable 0
        --uvm-scratch-pool-enable 0
      )
      ;;
    full_impl_enforce)
      local plan_json="$run_dir/weight_broad_cold_plan.json"
      local require_routing_plan=0
      local gap_watch_action="device_direct"
      if [ "$SCRATCH_DEVICE_DIRECT_ENABLE" = "0" ]; then
        gap_watch_action="observe"
      fi
      if [ -n "$WEIGHT_PLAN_FILE" ]; then
        cp "$WEIGHT_PLAN_FILE" "$plan_json"
        require_routing_plan=1
      else
        write_weight_plan "$plan_json"
      fi
      args+=(
        --uvm-policy-enable 1
        --uvm-device-direct-enable "$SCRATCH_DEVICE_DIRECT_ENABLE"
        --uvm-device-direct-min-bytes "$SCRATCH_MIN_BYTES"
        --uvm-device-direct-max-bytes "$SCRATCH_MAX_BYTES"
        --uvm-device-direct-max-total-bytes "$SCRATCH_MAX_TOTAL_BYTES"
        --uvm-device-direct-backend "$SCRATCH_BACKEND"
        --uvm-device-direct-pool-release-threshold "$SCRATCH_POOL_RELEASE_THRESHOLD"
        --uvm-device-direct-target-phases "$SCRATCH_TARGET_PHASES"
        --auto-gap-watch-policy-action-override "$gap_watch_action"
        --uvm-residency-enable "$RESIDENCY_ENABLE"
        --uvm-residency-stage stage_l6
        --uvm-residency-mode enforce
        --uvm-residency-trace-enable "$RESIDENCY_TRACE_ENABLE"
        --uvm-residency-trace-file "$run_dir/vllm_uvm_pool_residency_${label}.jsonl"
        --uvm-residency-global-gpu-bytes "$GLOBAL_GPU_BYTES"
        --uvm-residency-weight-gpu-bytes "$WEIGHT_GPU_BYTES"
        --uvm-residency-kv-gpu-bytes "$KV_GPU_BYTES"
        --uvm-residency-scratch-gpu-bytes "$STAGE_L_SCRATCH_GPU_BYTES"
        --uvm-residency-safety-margin-bytes "$SAFETY_MARGIN_BYTES"
        --uvm-residency-kv-metadata-enable 1
        --uvm-residency-kv-data-offload-enable 1
        --uvm-residency-kv-data-offload-mode prefetch_cpu
        --uvm-residency-kv-prefetch-back-enable 1
        --uvm-residency-kv-prefetch-back-mode prefetch
        --uvm-residency-kv-max-offload-blocks-per-step "$KV_MAX_OFFLOAD_BLOCKS"
        --uvm-residency-kv-pointer-map-file "$run_dir/vllm_uvm_kv_pointer_map_${label}.jsonl"
        --uvm-kv-runtime-enable "$KV_RUNTIME_ENABLE"
        --uvm-kv-runtime-mode enforce
        --uvm-kv-runtime-budget-blocks "$KV_RUNTIME_BUDGET_BLOCKS"
        --uvm-kv-runtime-trace-file "$run_dir/vllm_uvm_kv_runtime_${label}.jsonl"
        --uvm-kv-runtime-candidate-limit "$KV_RUNTIME_CANDIDATE_LIMIT"
        --uvm-kv-runtime-prefix-evict-enable 0
        --uvm-weight-prefetch-enable "$WEIGHT_RUNTIME_PREFETCH_ENABLE"
        --uvm-weight-prefetch-mode prefetch
        --uvm-weight-prefetch-trace-enable "$WEIGHT_PREFETCH_TRACE_ENABLE"
        --uvm-weight-router-topk-every-n-steps "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS"
        --uvm-weight-prefetch-skip-resident "$WEIGHT_PREFETCH_SKIP_RESIDENT"
        --uvm-weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE"
        --uvm-weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS"
        --uvm-weight-prefetch-trace-file "$run_dir/vllm_uvm_weight_prefetch_${label}.jsonl"
        --uvm-weight-prefetch-plan-file "$plan_json"
        --uvm-weight-prefetch-require-plan 0
        --uvm-weight-prefetch-source "$WEIGHT_PREFETCH_SOURCE"
        --uvm-oracle-expert-trace-file "$ORACLE_EXPERT_TRACE_FILE"
        --uvm-oracle-prefetch-lookahead-steps "$ORACLE_PREFETCH_LOOKAHEAD_STEPS"
        --uvm-weight-prefetch-hit-trace-enable "$WEIGHT_PREFETCH_HIT_TRACE_ENABLE"
        --uvm-weight-prefetch-hit-driver-query-enable "$WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE"
        --uvm-weight-prefetch-hit-max-regions-per-step "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP"
        --uvm-weight-batch-executor-enable "${WEIGHT_BATCH_EXECUTOR_ENABLE:-1}"
        --uvm-weight-executor "$WEIGHT_EXECUTOR"
        --uvm-weight-prefetch-max-bytes-per-step "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP"
        --uvm-weight-prefetch-max-experts-per-layer "$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER"
        --uvm-weight-prefetch-target-roles moe_gate_up,moe_down
        --uvm-weight-offload-enable 0
        --uvm-weight-offload-mode prefetch_cpu
        --uvm-weight-offload-plan-file "$plan_json"
        --uvm-weight-offload-max-bytes-per-step "$WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP"
        --uvm-weight-offload-max-experts-per-layer "$WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER"
        --uvm-weight-offload-target-roles moe_gate_up,moe_down
        --uvm-init-kv-sizing-enable 1
        --uvm-init-kv-sizing-policy clamp
        --uvm-init-kv-max-bytes "$KV_GPU_BYTES"
        --uvm-init-kv-min-blocks "$INIT_KV_MIN_BLOCKS"
        --uvm-init-kv-trace-file "$run_dir/vllm_uvm_init_kv_sizing_${label}.jsonl"
        --uvm-init-weight-offload-enable "$INIT_WEIGHT_OFFLOAD_ENABLE"
        --uvm-init-weight-offload-mode offload_cpu
        --uvm-init-weight-plan-file "$plan_json"
        --uvm-init-weight-target-roles moe_gate_up,moe_down
        --uvm-init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES"
        --uvm-init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER"
        --uvm-init-weight-sync 1
        --uvm-init-weight-hot-prefetch-enable "$INIT_WEIGHT_HOT_PREFETCH_ENABLE"
        --uvm-init-weight-hot-prefetch-sync "$INIT_WEIGHT_HOT_PREFETCH_SYNC"
        --uvm-init-weight-hot-prefetch-max-bytes "$INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES"
        --uvm-init-weight-hot-prefetch-device "$INIT_WEIGHT_HOT_PREFETCH_DEVICE"
        --uvm-init-weight-hot-prefetch-real-query-enable "$INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE"
        --uvm-hot-residency-stage-trace-enable "$HOT_RESIDENCY_STAGE_TRACE_ENABLE"
        --uvm-hot-residency-stage-max-num-tokens "$HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS"
        --uvm-hot-residency-stage-device "$HOT_RESIDENCY_STAGE_DEVICE"
        --uvm-init-weight-require-routing-plan "$require_routing_plan"
        --uvm-init-weight-trace-file "$run_dir/vllm_uvm_init_weight_offload_${label}.jsonl"
        --uvm-residency-weight-initial-state gpu
        --uvm-residency-weight-enforce-hot-set 1
        --uvm-residency-weight-sync-offload 1
        --uvm-residency-weight-max-victims-per-step "$WEIGHT_MAX_VICTIMS_PER_STEP"
        --uvm-residency-weight-victim-policy cold_plan_lru
        --uvm-residency-weight-victim-action "$WEIGHT_VICTIM_ACTION"
        --uvm-residency-weight-active-evict-enable "$WEIGHT_ACTIVE_EVICT_ENABLE"
        --uvm-residency-weight-active-evict-max-experts-per-layer "$WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER"
        --uvm-residency-weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS"
        --uvm-weight-budget-cpu-offload-enable "$WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE"
        --uvm-pool-registry-enable "$POOL_REGISTRY_ENABLE"
        --uvm-scratch-pool-enable "$SCRATCH_POOL_ENABLE"
        --uvm-scratch-pool-budget-bytes "$SCRATCH_MAX_TOTAL_BYTES"
        --uvm-scratch-pool-mode enforce
        --uvm-scratch-pool-target-phases "$SCRATCH_TARGET_PHASES"
        --uvm-pool-coordinator-enable "$POOL_COORDINATOR_ENABLE"
        --uvm-pool-coordinator-mode enforce
        --uvm-pool-coordinator-trace-enable "$POOL_COORDINATOR_TRACE_ENABLE"
        --uvm-pool-coordinator-trace-file "$run_dir/vllm_uvm_pool_coordinator_${label}.jsonl"
        --uvm-pool-coordinator-global-bytes-per-step "$COORDINATOR_GLOBAL_BYTES_PER_STEP"
        --uvm-pool-coordinator-kv-bytes-per-step "$COORDINATOR_KV_BYTES_PER_STEP"
        --uvm-pool-coordinator-weight-bytes-per-step "$COORDINATOR_WEIGHT_BYTES_PER_STEP"
        --uvm-pool-coordinator-scratch-bytes-per-step "$COORDINATOR_SCRATCH_BYTES_PER_STEP"
        --uvm-pool-coordinator-priority "$COORDINATOR_PRIORITY"
      )
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 2
      ;;
  esac

  local start_ts end_ts status
  start_ts="$(date +%s)"
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" "${args[@]}"
  status=$?
  end_ts="$(date +%s)"
  echo "$status" > "$run_dir/exit_code.txt"
  echo "$((end_ts - start_ts))" > "$run_dir/wall_seconds.txt"
  echo "Variant $variant rep $rep exit code: $status"
  return "$status"
}

run_variant() {
  local variant="$1"
  local rep status any_status=0
  for rep in $(seq 1 "$REPEAT"); do
    run_one "$variant" "$rep"
    status=$?
    if [ "$status" -ne 0 ]; then
      any_status="$status"
      if [ "$FAIL_FAST" = "1" ]; then
        return "$status"
      fi
    fi
  done
  return "$any_status"
}

overall_status=0
run_variant pure_uvm_baseline || overall_status=$?
if [ "$overall_status" -ne 0 ] && [ "$FAIL_FAST" = "1" ]; then
  echo "Fail-fast after pure_uvm_baseline"
else
  if [ "$RUN_TRACE_CONTROL" = "1" ]; then
    run_variant uvm_trace_control || overall_status=$?
  fi
  if [ "$overall_status" -ne 0 ] && [ "$FAIL_FAST" = "1" ]; then
    echo "Fail-fast after uvm_trace_control"
  else
    run_variant full_impl_enforce || overall_status=$?
  fi
fi

echo "==========================================================="
echo " Building A/B summary"
echo "==========================================================="
python3 - "$RUN_ROOT" "$LOG_FILE" <<'PY'
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
main_log = Path(sys.argv[2])
runs_dir = root / "runs"

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def parse_bench(path: Path) -> dict[str, float | int | None]:
    text = read_text(path)
    fields = {
        "successful_requests": r"Successful requests:\s+([0-9.]+)",
        "failed_requests": r"Failed requests:\s+([0-9.]+)",
        "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
    }
    out: dict[str, float | int | None] = {}
    for key, pattern in fields.items():
      matches = re.findall(pattern, text)
      out[key] = float(matches[-1]) if matches else None
    return out

def parse_faults(path: Path) -> dict[str, int | None]:
    text = read_text(path)
    out = {}
    for key in (
        "delta_faults",
        "delta_duplicates",
        "delta_after_dedup",
        "delta_kv_faults",
        "delta_kv_duplicates",
        "delta_kv_after_dedup",
    ):
        matches = re.findall(rf"{key}=([0-9]+)", text)
        out[key] = int(matches[-1]) if matches else None
    return out

def parse_allocator_summary(path: Path) -> dict[str, int | None]:
    text = read_text(path)
    fields = {
        "allocator_weight_budget_cpu_offload_allocations": r"Weight budget CPU offload allocations:\s+([0-9]+)",
        "allocator_weight_budget_cpu_offload_bytes": r"Weight budget CPU offload bytes:\s+([0-9]+)",
        "allocator_weight_budget_cpu_offload_failures": r"Weight budget CPU offload failures:\s+([0-9]+)",
        "allocator_weight_budget_over_allocations": r"Weight budget over allocations:\s+([0-9]+)",
        "allocator_weight_budget_reject_allocations": r"Weight budget reject allocations:\s+([0-9]+)",
        "allocator_weight_live_bytes": r"Weight live bytes:\s+([0-9]+)",
        "allocator_weight_peak_live_bytes": r"Weight peak live bytes:\s+([0-9]+)",
    }
    out = {}
    for key, pattern in fields.items():
        matches = re.findall(pattern, text)
        out[key] = max((int(value) for value in matches), default=None)
    return out

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_jsonl(path: Path) -> list[dict]:
    records = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return records

def count_actions(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        action = record.get("action")
        if isinstance(action, str):
            counts[action] = counts.get(action, 0) + 1
    return counts

rows = []
for run_dir in sorted(runs_dir.glob("*_rep*")):
    label = run_dir.name
    variant = re.sub(r"_rep[0-9]+$", "", label)
    rep_match = re.search(r"_rep([0-9]+)$", label)
    rep = int(rep_match.group(1)) if rep_match else 0
    exit_code_text = read_text(run_dir / "exit_code.txt").strip()
    exit_code = int(exit_code_text) if exit_code_text.isdigit() else None
    bench = parse_bench(run_dir / f"vllm_bench_{label}.log")
    allocator = parse_allocator_summary(
        run_dir / f"vllm_uvm_allocator_trace_{label}.log"
    )
    faults = parse_faults(root.parent / "no_such_file")
    log_text = read_text(Path(sys.argv[0]))
    del log_text
    # Delta fault lines are printed in the top-level log. Parse the per-run
    # allocator/server/bench files plus top-level log fallback below.
    top_log = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in [])
    del top_log
    metrics = read_json(run_dir / f"gap_watch_metrics_{label}.json")
    probe = read_json(run_dir / f"auto_gap_watch_probe_{label}.json")
    post = read_json(run_dir / f"auto_gap_watch_post_main_{label}.json")
    residency_path = next(run_dir.glob("vllm_uvm_pool_residency_*.jsonl"), None)
    runtime_path = next(run_dir.glob("vllm_uvm_kv_runtime_*.jsonl"), None)
    weight_path = next(run_dir.glob("vllm_uvm_weight_prefetch_*.jsonl"), None)
    init_kv_path = next(run_dir.glob("vllm_uvm_init_kv_sizing_*.jsonl"), None)
    init_weight_path = next(run_dir.glob("vllm_uvm_init_weight_offload_*.jsonl"), None)
    coord_path = next(run_dir.glob("vllm_uvm_pool_coordinator_*.jsonl"), None)
    residency_counts = count_actions(read_jsonl(residency_path)) if residency_path else {}
    residency_records = read_jsonl(residency_path) if residency_path else []
    runtime_counts = count_actions(read_jsonl(runtime_path)) if runtime_path else {}
    weight_records = read_jsonl(weight_path) if weight_path else []
    weight_counts = count_actions(weight_records)
    weight_step_summaries = [
        item for item in weight_records if item.get("action") == "step_summary"
    ]
    init_kv_records = read_jsonl(init_kv_path) if init_kv_path else []
    init_weight_records = read_jsonl(init_weight_path) if init_weight_path else []
    init_weight_counts = count_actions(init_weight_records)
    coord_counts = count_actions(read_jsonl(coord_path)) if coord_path else {}
    final_residency_summary = {}
    for record in residency_records:
        if record.get("action") == "residency_summary":
            final_residency_summary = record
    init_kv_record = init_kv_records[-1] if init_kv_records else {}
    init_kv_workers = init_kv_record.get("workers") if isinstance(init_kv_record, dict) else []
    init_kv_worker0 = (
        init_kv_workers[0]
        if isinstance(init_kv_workers, list)
        and init_kv_workers
        and isinstance(init_kv_workers[0], dict)
        else {}
    )
    server_log = read_text(run_dir / f"vllm_server_{label}.log")
    bench_log = read_text(run_dir / f"vllm_bench_{label}.log")
    oom = "out of memory" in server_log.lower() or "out of memory" in bench_log.lower()
    connection_refused = "ConnectionRefusedError" in bench_log

    row = {
        "variant": variant,
        "rep": rep,
        "exit_code": exit_code,
        "oom": oom,
        "connection_refused": connection_refused,
        **bench,
        **allocator,
        "probe_gap_size_mib": (probe.get("selected_gap") or {}).get("size_mib"),
        "post_gap_size_mib": (post.get("selected_gap") or {}).get("size_mib"),
        "post_unknown_faults": post.get("total_unknown_faults"),
        "post_gap_faults": (post.get("selected_gap") or {}).get("faults"),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "device_direct_eligible_records": metrics.get("device_direct_eligible_records"),
        "device_direct_budget_reject_records": metrics.get("device_direct_budget_reject_records"),
        "device_direct_peak_live_bytes": metrics.get("device_direct_peak_live_bytes_observed"),
        "scratch_pool_enabled": metrics.get("scratch_pool_enabled"),
        "scratch_pool_mode": metrics.get("scratch_pool_mode"),
        "scratch_pool_trace_records": metrics.get("scratch_pool_trace_records"),
        "scratch_pool_eligible_records": metrics.get("scratch_pool_eligible_records"),
        "scratch_pool_device_direct_records": metrics.get("scratch_pool_device_direct_records"),
        "scratch_pool_budget_over_records": metrics.get("scratch_pool_budget_over_records"),
        "scratch_pool_budget_reject_records": metrics.get("scratch_pool_budget_reject_records"),
        "scratch_pool_live_bytes": metrics.get("scratch_pool_live_bytes"),
        "scratch_pool_peak_live_bytes": metrics.get("scratch_pool_peak_live_bytes"),
        "kv_live_bytes": metrics.get("kv_live_bytes"),
        "kv_peak_live_bytes": metrics.get("kv_peak_live_bytes_observed"),
        "weight_live_bytes": metrics.get("weight_live_bytes"),
        "weight_peak_live_bytes": metrics.get("weight_peak_live_bytes_observed"),
        "residency_weight_estimated_gpu_bytes": final_residency_summary.get("weights_estimated_gpu_resident_bytes"),
        "residency_weight_budget_bytes": final_residency_summary.get("weight_gpu_budget_bytes"),
        "residency_global_estimated_gpu_bytes": final_residency_summary.get("global_estimated_gpu_resident_bytes"),
        "residency_kv_offload_issued": residency_counts.get("residency_kv_data_offload_issued", 0),
        "residency_kv_prefetch_issued": residency_counts.get("residency_kv_prefetch_back_issued", 0),
        "residency_weight_victim_offload_issued": residency_counts.get("residency_weight_victim_offload_issued", 0),
        "weight_prefetch_issued": weight_counts.get("prefetch_issued", 0),
        "weight_offload_prefetch_cpu_issued": weight_counts.get("offload_prefetch_cpu_issued", 0),
        "weight_prefetch_candidate_experts": sum(int(item.get("candidate_expert_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_expert_resident_skip": sum(int(item.get("expert_resident_skip_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_expert_expanded": sum(int(item.get("expert_expanded_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_candidate_regions": sum(int(item.get("candidate_region_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_resident_skip": sum(int(item.get("resident_skip_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_cooldown_skip": sum(int(item.get("cooldown_skip_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_cpu_unknown_regions": sum(int(item.get("cpu_unknown_region_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_active_region_id_builds": sum(int(item.get("active_region_ids_build_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_cold_candidate_register_calls": sum(int(item.get("cold_candidate_register_calls") or 0) for item in weight_step_summaries),
        "weight_prefetch_reaccess_after_evict": sum(int(item.get("reaccess_after_evict_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_access_probe_records": sum(int(item.get("prefetch_access_probe_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_access_active_experts": sum(int(item.get("prefetch_access_active_expert_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_access_regions": sum(int(item.get("prefetch_access_region_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_before_access_candidates": sum(int(item.get("prefetch_before_access_candidate_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_estimated": sum(int(item.get("prefetch_hit_before_access_estimated_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_miss_before_access_estimated": sum(int(item.get("prefetch_miss_before_access_estimated_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_same_step_access": sum(int(item.get("prefetch_same_step_access_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_access_no_prior_prefetch": sum(int(item.get("prefetch_access_no_prior_prefetch_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_step_lag_sum": sum(int(item.get("prefetch_hit_step_lag_sum") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_step_lag_max": max([int(item.get("prefetch_hit_step_lag_max") or 0) for item in weight_step_summaries] or [0]),
        "weight_prefetch_hit_before_access_real_query_records": sum(int(item.get("prefetch_hit_before_access_real_query_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_real_query_ranges": sum(int(item.get("prefetch_hit_before_access_real_query_ranges") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_real_queried_bytes": sum(int(item.get("prefetch_hit_before_access_real_queried_bytes") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_real_gpu_bytes": sum(int(item.get("prefetch_hit_before_access_real_gpu_bytes") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_real_cpu_bytes": sum(int(item.get("prefetch_hit_before_access_real_cpu_bytes") or 0) for item in weight_step_summaries),
        "weight_prefetch_hit_before_access_real_unknown_bytes": sum(int(item.get("prefetch_hit_before_access_real_unknown_bytes") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_replay_records": sum(int(item.get("oracle_replay_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_match_records": sum(int(item.get("oracle_match_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_mismatch_records": sum(int(item.get("oracle_mismatch_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_trace_missing_records": sum(int(item.get("oracle_trace_missing_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_future_missing_records": sum(int(item.get("oracle_future_missing_records") or 0) for item in weight_step_summaries),
        "weight_prefetch_oracle_candidate_experts": sum(int(item.get("oracle_candidate_expert_records") or 0) for item in weight_step_summaries),
        "init_kv_sizing_records": len(init_kv_records),
        "init_kv_applied_worker0": init_kv_worker0.get("applied"),
        "init_kv_original_bytes_worker0": init_kv_worker0.get("original_bytes"),
        "init_kv_final_bytes_worker0": init_kv_worker0.get("final_bytes"),
        "init_kv_original_blocks_worker0": init_kv_worker0.get("original_num_blocks"),
        "init_kv_final_blocks_worker0": init_kv_worker0.get("final_num_blocks"),
        "init_weight_offload_prefetch_cpu_issued": init_weight_counts.get("init_weight_offload_prefetch_cpu_issued", 0),
        "init_weight_offload_offload_cpu_issued": init_weight_counts.get("init_weight_offload_offload_cpu_issued", 0),
        "init_weight_offload_cpu_issued_total": (
            init_weight_counts.get("init_weight_offload_prefetch_cpu_issued", 0)
            + init_weight_counts.get("init_weight_offload_offload_cpu_issued", 0)
        ),
        "init_weight_hot_prefetch_issued": init_weight_counts.get("init_weight_hot_prefetch_issued", 0),
        "init_weight_hot_prefetch_skipped": init_weight_counts.get("init_weight_hot_prefetch_skipped", 0),
        "init_weight_hot_prefetch_budget_reject": init_weight_counts.get("init_weight_hot_prefetch_budget_reject", 0),
        "init_weight_hot_prefetch_summary_issued_records": sum(int(item.get("hot_prefetch_issued_records") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_summary_issued_bytes": sum(int(item.get("hot_prefetch_issued_bytes") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_query_records": sum(int(item.get("hot_prefetch_real_query_records") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_query_ranges": sum(int(item.get("hot_prefetch_real_query_ranges") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_queried_bytes": sum(int(item.get("hot_prefetch_real_queried_bytes") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_gpu_bytes": sum(int(item.get("hot_prefetch_real_gpu_bytes") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_cpu_bytes": sum(int(item.get("hot_prefetch_real_cpu_bytes") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_prefetch_real_unknown_bytes": sum(int(item.get("hot_prefetch_real_unknown_bytes") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_hot_registered_records": sum(int(item.get("hot_registered_records") or 0) for item in init_weight_records if item.get("action") == "init_weight_offload_summary"),
        "init_weight_offload_advise_cpu_issued": init_weight_counts.get("init_weight_offload_advise_cpu_issued", 0),
        "init_weight_offload_budget_reject": init_weight_counts.get("init_weight_offload_budget_reject", 0),
        "coordinator_records": sum(coord_counts.values()),
    }
    before_access_candidates = row["weight_prefetch_before_access_candidates"]
    if before_access_candidates:
        row["weight_prefetch_hit_before_access_estimated_rate"] = (
            row["weight_prefetch_hit_before_access_estimated"]
            / before_access_candidates
        )
        row["weight_prefetch_hit_step_lag_avg"] = (
            row["weight_prefetch_hit_step_lag_sum"]
            / before_access_candidates
        )
    real_queried_bytes = row["weight_prefetch_hit_before_access_real_queried_bytes"]
    if real_queried_bytes:
        row["weight_prefetch_hit_before_access_real_gpu_ratio"] = (
            row["weight_prefetch_hit_before_access_real_gpu_bytes"]
            / real_queried_bytes
        )
    init_hot_real_queried_bytes = row["init_weight_hot_prefetch_real_queried_bytes"]
    if init_hot_real_queried_bytes:
        row["init_weight_hot_prefetch_real_gpu_ratio"] = (
            row["init_weight_hot_prefetch_real_gpu_bytes"]
            / init_hot_real_queried_bytes
        )
    rows.append(row)

# Parse fault deltas from the top-level shell log. Because run_kv_fault_ratio
# prints fault deltas to stdout, they live in this main log, not in per-run
# files. Attach them in run order.
text = read_text(main_log)
blocks = re.split(r"=+\n Running ", text)
delta_by_label: dict[str, dict[str, int]] = {}
for block in blocks:
    first = block.splitlines()[0] if block.splitlines() else ""
    m = re.search(r"([A-Za-z0-9_]+_rep[0-9]+)", first)
    if not m:
        m = re.search(r"run_dir: .*/runs/([A-Za-z0-9_]+_rep[0-9]+)", block)
    if not m:
        continue
    label = m.group(1)
    values = {}
    for key in (
        "delta_faults",
        "delta_after_dedup",
        "delta_kv_faults",
        "delta_kv_after_dedup",
    ):
        matches = re.findall(rf"{key}=([0-9]+)", block)
        if matches:
            values[key] = int(matches[-1])
    delta_by_label[label] = values

for row in rows:
    label = f"{row['variant']}_rep{row['rep']}"
    row.update(delta_by_label.get(label, {}))

csv_path = root / "summary.csv"
fieldnames = sorted({key for row in rows for key in row})
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def median(values):
    vals = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.median(vals) if vals else None

by_variant: dict[str, list[dict]] = {}
for row in rows:
    by_variant.setdefault(str(row["variant"]), []).append(row)

summary = {}
for variant, items in by_variant.items():
    summary[variant] = {
        "runs": len(items),
        "successful_runs": sum(1 for item in items if item.get("exit_code") == 0),
        "oom_runs": sum(1 for item in items if item.get("oom")),
        "output_tok_s_median": median([item.get("output_tok_s") for item in items]),
        "mean_tpot_ms_median": median([item.get("mean_tpot_ms") for item in items]),
        "mean_ttft_ms_median": median([item.get("mean_ttft_ms") for item in items]),
        "delta_after_dedup_median": median([item.get("delta_after_dedup") for item in items]),
        "post_unknown_faults_median": median([item.get("post_unknown_faults") for item in items]),
        "post_gap_faults_median": median([item.get("post_gap_faults") for item in items]),
        "device_direct_actual_records_median": median([item.get("device_direct_actual_records") for item in items]),
        "scratch_pool_eligible_records_median": median([item.get("scratch_pool_eligible_records") for item in items]),
        "scratch_pool_device_direct_records_median": median([item.get("scratch_pool_device_direct_records") for item in items]),
        "scratch_pool_budget_reject_records_median": median([item.get("scratch_pool_budget_reject_records") for item in items]),
        "scratch_pool_peak_live_bytes_median": median([item.get("scratch_pool_peak_live_bytes") for item in items]),
        "kv_peak_live_bytes_median": median([item.get("kv_peak_live_bytes") for item in items]),
        "weight_peak_live_bytes_median": median([item.get("weight_peak_live_bytes") for item in items]),
        "residency_weight_estimated_gpu_bytes_median": median([item.get("residency_weight_estimated_gpu_bytes") for item in items]),
        "residency_weight_budget_bytes_median": median([item.get("residency_weight_budget_bytes") for item in items]),
        "residency_weight_victim_offload_issued_median": median([item.get("residency_weight_victim_offload_issued") for item in items]),
        "allocator_weight_budget_cpu_offload_allocations_median": median([item.get("allocator_weight_budget_cpu_offload_allocations") for item in items]),
        "allocator_weight_budget_cpu_offload_bytes_median": median([item.get("allocator_weight_budget_cpu_offload_bytes") for item in items]),
        "allocator_weight_budget_cpu_offload_failures_median": median([item.get("allocator_weight_budget_cpu_offload_failures") for item in items]),
        "init_kv_final_bytes_worker0_median": median([item.get("init_kv_final_bytes_worker0") for item in items]),
        "init_weight_offload_prefetch_cpu_issued_median": median([item.get("init_weight_offload_prefetch_cpu_issued") for item in items]),
        "init_weight_offload_offload_cpu_issued_median": median([item.get("init_weight_offload_offload_cpu_issued") for item in items]),
        "init_weight_offload_cpu_issued_total_median": median([item.get("init_weight_offload_cpu_issued_total") for item in items]),
        "init_weight_hot_prefetch_issued_median": median([item.get("init_weight_hot_prefetch_issued") for item in items]),
        "init_weight_hot_prefetch_summary_issued_records_median": median([item.get("init_weight_hot_prefetch_summary_issued_records") for item in items]),
        "init_weight_hot_prefetch_summary_issued_bytes_median": median([item.get("init_weight_hot_prefetch_summary_issued_bytes") for item in items]),
        "init_weight_hot_prefetch_real_query_records_median": median([item.get("init_weight_hot_prefetch_real_query_records") for item in items]),
        "init_weight_hot_prefetch_real_query_ranges_median": median([item.get("init_weight_hot_prefetch_real_query_ranges") for item in items]),
        "init_weight_hot_prefetch_real_queried_bytes_median": median([item.get("init_weight_hot_prefetch_real_queried_bytes") for item in items]),
        "init_weight_hot_prefetch_real_gpu_bytes_median": median([item.get("init_weight_hot_prefetch_real_gpu_bytes") for item in items]),
        "init_weight_hot_prefetch_real_gpu_ratio_median": median([item.get("init_weight_hot_prefetch_real_gpu_ratio") for item in items]),
        "init_weight_hot_registered_records_median": median([item.get("init_weight_hot_registered_records") for item in items]),
        "weight_prefetch_issued_median": median([item.get("weight_prefetch_issued") for item in items]),
        "weight_prefetch_candidate_regions_median": median([item.get("weight_prefetch_candidate_regions") for item in items]),
        "weight_prefetch_access_probe_records_median": median([item.get("weight_prefetch_access_probe_records") for item in items]),
        "weight_prefetch_before_access_candidates_median": median([item.get("weight_prefetch_before_access_candidates") for item in items]),
        "weight_prefetch_hit_before_access_estimated_median": median([item.get("weight_prefetch_hit_before_access_estimated") for item in items]),
        "weight_prefetch_miss_before_access_estimated_median": median([item.get("weight_prefetch_miss_before_access_estimated") for item in items]),
        "weight_prefetch_hit_before_access_estimated_rate_median": median([item.get("weight_prefetch_hit_before_access_estimated_rate") for item in items]),
        "weight_prefetch_same_step_access_median": median([item.get("weight_prefetch_same_step_access") for item in items]),
        "weight_prefetch_access_no_prior_prefetch_median": median([item.get("weight_prefetch_access_no_prior_prefetch") for item in items]),
        "weight_prefetch_hit_step_lag_avg_median": median([item.get("weight_prefetch_hit_step_lag_avg") for item in items]),
        "weight_prefetch_hit_before_access_real_query_records_median": median([item.get("weight_prefetch_hit_before_access_real_query_records") for item in items]),
        "weight_prefetch_hit_before_access_real_query_ranges_median": median([item.get("weight_prefetch_hit_before_access_real_query_ranges") for item in items]),
        "weight_prefetch_hit_before_access_real_queried_bytes_median": median([item.get("weight_prefetch_hit_before_access_real_queried_bytes") for item in items]),
        "weight_prefetch_hit_before_access_real_gpu_bytes_median": median([item.get("weight_prefetch_hit_before_access_real_gpu_bytes") for item in items]),
        "weight_prefetch_hit_before_access_real_gpu_ratio_median": median([item.get("weight_prefetch_hit_before_access_real_gpu_ratio") for item in items]),
        "weight_prefetch_oracle_replay_records_median": median([item.get("weight_prefetch_oracle_replay_records") for item in items]),
        "weight_prefetch_oracle_match_records_median": median([item.get("weight_prefetch_oracle_match_records") for item in items]),
        "weight_prefetch_oracle_mismatch_records_median": median([item.get("weight_prefetch_oracle_mismatch_records") for item in items]),
        "weight_prefetch_oracle_trace_missing_records_median": median([item.get("weight_prefetch_oracle_trace_missing_records") for item in items]),
        "weight_prefetch_oracle_future_missing_records_median": median([item.get("weight_prefetch_oracle_future_missing_records") for item in items]),
        "weight_prefetch_oracle_candidate_experts_median": median([item.get("weight_prefetch_oracle_candidate_experts") for item in items]),
    }

def pct(new, old):
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)) or old == 0:
        return None
    return (new - old) / old * 100.0

baseline = summary.get("pure_uvm_baseline", {})
full = summary.get("full_impl_enforce", {})
comparison = {
    "output_tok_s_pct_vs_pure_uvm": pct(full.get("output_tok_s_median"), baseline.get("output_tok_s_median")),
    "mean_tpot_ms_pct_vs_pure_uvm": pct(full.get("mean_tpot_ms_median"), baseline.get("mean_tpot_ms_median")),
    "delta_after_dedup_pct_vs_pure_uvm": pct(full.get("delta_after_dedup_median"), baseline.get("delta_after_dedup_median")),
    "post_unknown_faults_pct_vs_pure_uvm": pct(full.get("post_unknown_faults_median"), baseline.get("post_unknown_faults_median")),
    "post_gap_faults_pct_vs_pure_uvm": pct(full.get("post_gap_faults_median"), baseline.get("post_gap_faults_median")),
}

md_path = root / "summary.md"
with md_path.open("w", encoding="utf-8") as handle:
    handle.write("# Stage L Full Implementation vs Pure UVM A/B\n\n")
    handle.write(f"- run_root: `{root}`\n")
    handle.write(f"- summary_csv: `{csv_path}`\n\n")
    handle.write("## Variant Medians\n\n")
    for variant, data in summary.items():
        handle.write(f"### {variant}\n\n")
        for key, value in data.items():
            handle.write(f"- {key}: {value}\n")
        handle.write("\n")
    handle.write("## Full Implementation vs Pure UVM\n\n")
    for key, value in comparison.items():
        handle.write(f"- {key}: {value}\n")
    handle.write("\n")
    handle.write("Negative percentages for fault and latency metrics are improvements. ")
    handle.write("Positive percentages for throughput metrics are improvements.\n")

print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
print(json.dumps({"summary": summary, "comparison": comparison}, indent=2, sort_keys=True))
PY

echo "==========================================================="
echo " Stage L Full Implementation vs Pure UVM A/B finished"
echo " run_root: $RUN_ROOT"
echo " summary_csv: $RUN_ROOT/summary.csv"
echo " summary_md: $RUN_ROOT/summary.md"
echo " overall_status: $overall_status"
echo "==========================================================="

exit "$overall_status"
