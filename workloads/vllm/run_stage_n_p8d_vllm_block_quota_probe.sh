#!/usr/bin/env bash
# Stage N P8d vLLM block-quota probe.
#
# This wraps the existing tag-only three-pool vLLM smoke, but validates the new
# P8c/P8d block-quota traces instead of the older region-scan auto-hook traces.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p8d_vllm_block_quota_$(date -u +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-kv}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
REQUIRE_RECLAIM="${REQUIRE_RECLAIM:-1}"
REQUIRE_BLOCK_FASTPATH="${REQUIRE_BLOCK_FASTPATH:-1}"
ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-2}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
PORT="${PORT:-8000}"

SCRATCH_TEMP_BUDGET_BYTES="${SCRATCH_TEMP_BUDGET_BYTES:-16777216}"
DRIVER_POOL_BUDGET_MODE="${DRIVER_POOL_BUDGET_MODE:-soft}"
VLLM_UVM_DRIVER_KV_TAG_ENABLE="${VLLM_UVM_DRIVER_KV_TAG_ENABLE:-0}"

case "$PROFILE" in
  kv)
    KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-536870912}"
    WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-21474836480}"
    UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-1}"
    ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-4}"
    ;;
  weight)
    KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-10737418240}"
    WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-8589934592}"
    UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-0}"
    ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-8}"
    ;;
  both)
    KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-536870912}"
    WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-8589934592}"
    UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-0}"
    ALLOWED_POST_FAULT_SLACK_BLOCKS="${ALLOWED_POST_FAULT_SLACK_BLOCKS:-8}"
    ;;
  *)
    echo "ERROR: PROFILE must be kv, weight, or both" >&2
    exit 2
    ;;
esac

UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-1}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-1}"
UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-1}"
UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-0}"
UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-128}"
UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"
UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-0}"

SMOKE_ROOT="$RUN_ROOT/vllm_smoke"
SUMMARY="$RUN_ROOT/stage_n_p8d_vllm_block_quota_summary.txt"
CHECK_LOG="$RUN_ROOT/block_quota_check.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_p8d_vllm.txt"
COMMAND_FILE="$RUN_ROOT/command_stage_n_p8d_vllm.sh"
mkdir -p "$RUN_ROOT"

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "SMOKE_ROOT=$SMOKE_ROOT"
  echo "PROFILE=$PROFILE"
  echo "MODEL=$MODEL"
  echo "PROMPTS=$PROMPTS"
  echo "REQUEST_RATE=$REQUEST_RATE"
  echo "OUTPUT_LEN=$OUTPUT_LEN"
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  echo "SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS"
  echo "SERVER_GPU_MEMORY_UTILIZATION=$SERVER_GPU_MEMORY_UTILIZATION"
  echo "STARTUP_TIMEOUT=$STARTUP_TIMEOUT"
  echo "PORT=$PORT"
  echo "KV_BUDGET_BYTES=$KV_BUDGET_BYTES"
  echo "WEIGHT_BUDGET_BYTES=$WEIGHT_BUDGET_BYTES"
  echo "SCRATCH_TEMP_BUDGET_BYTES=$SCRATCH_TEMP_BUDGET_BYTES"
  echo "DRIVER_POOL_BUDGET_MODE=$DRIVER_POOL_BUDGET_MODE"
  echo "VLLM_UVM_DRIVER_KV_TAG_ENABLE=$VLLM_UVM_DRIVER_KV_TAG_ENABLE"
  echo "RESET_COUNTERS=$RESET_COUNTERS"
  echo "REQUIRE_RECLAIM=$REQUIRE_RECLAIM"
  echo "REQUIRE_BLOCK_FASTPATH=$REQUIRE_BLOCK_FASTPATH"
  echo "ALLOWED_POST_FAULT_SLACK_BLOCKS=$ALLOWED_POST_FAULT_SLACK_BLOCKS"
  env | sort | grep -E '^UVM_PHASE_(F_ROUTE|N_BLOCK|N_AUTO)' || true
} > "$ENV_FILE"

cmd=(
  env
  "RUN_ROOT=$SMOKE_ROOT"
  "MODEL=$MODEL"
  "PROMPTS=$PROMPTS"
  "REQUEST_RATE=$REQUEST_RATE"
  "OUTPUT_LEN=$OUTPUT_LEN"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  "SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS"
  "SERVER_GPU_MEMORY_UTILIZATION=$SERVER_GPU_MEMORY_UTILIZATION"
  "STARTUP_TIMEOUT=$STARTUP_TIMEOUT"
  "PORT=$PORT"
  "RESET_COUNTERS=$RESET_COUNTERS"
  "KV_BUDGET_BYTES=$KV_BUDGET_BYTES"
  "WEIGHT_BUDGET_BYTES=$WEIGHT_BUDGET_BYTES"
  "SCRATCH_TEMP_BUDGET_BYTES=$SCRATCH_TEMP_BUDGET_BYTES"
  "DRIVER_POOL_BUDGET_MODE=$DRIVER_POOL_BUDGET_MODE"
  "VLLM_UVM_DRIVER_KV_TAG_ENABLE=$VLLM_UVM_DRIVER_KV_TAG_ENABLE"
  "UVM_PHASE_F_ROUTE_LOG_DELAY_MS=$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  "UVM_PHASE_N_BLOCK_QUOTA_ENABLE=$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  "UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  "UVM_PHASE_N_AUTO_RECLAIM_ENABLE=$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY"
  "UVM_PHASE_N_AUTO_RECLAIM_EVERY=$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  "UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY=$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  "UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS=$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  "UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS=$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  "UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS=$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS"
  "REQUIRE_PHASE_N_AUTO_HOOK=0"
  "REQUIRE_PHASE_N_AUTO_RECLAIM=0"
  "REQUIRE_STAGE_N_HARD_PARTITION=0"
  ./run_stage_n_tag_only_three_pool_smoke.sh
)

{
  printf 'cd %q\n' "$SCRIPT_DIR"
  printf '%q ' "${cmd[@]}"
  echo
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Stage N P8d vLLM block-quota probe started."
echo "RUN_ROOT=$RUN_ROOT"
echo "PROFILE=$PROFILE"

set +e
"${cmd[@]}"
smoke_status=$?
set -e

check_args=(
  "$SMOKE_ROOT"
  --require-materialize
  --require-auto-hook
  --allowed-post-fault-slack-blocks "$ALLOWED_POST_FAULT_SLACK_BLOCKS"
  --check-failures
  --fail-on-violation
)
if [ "$REQUIRE_BLOCK_FASTPATH" = "1" ]; then
  check_args+=(--require-block-fastpath)
fi
if [ "$VLLM_UVM_DRIVER_KV_TAG_ENABLE" = "1" ]; then
  check_args+=(
    --require-allocator-kv-tag
    --require-kv-tag-before-materialize
  )
fi
if [ "$REQUIRE_RECLAIM" = "1" ]; then
  check_args+=(--require-over-budget --require-reclaim)
fi
case "$PROFILE" in
  kv) check_args+=(--require-pool kv) ;;
  weight) check_args+=(--require-pool weights) ;;
  both) check_args+=(--require-pool kv --require-pool weights) ;;
esac

set +e
python3 "$SCRIPT_DIR/check_stage_n_block_quota.py" "${check_args[@]}" > "$CHECK_LOG" 2>&1
check_status=$?
set -e

{
  echo "Stage N P8d vLLM Block Quota Summary"
  echo "====================================="
  echo "run_root=$RUN_ROOT"
  echo "smoke_root=$SMOKE_ROOT"
  echo "profile=$PROFILE"
  echo "smoke_status=$smoke_status"
  echo "check_status=$check_status"
  echo "env_file=$ENV_FILE"
  echo "command_file=$COMMAND_FILE"
  echo "check_log=$CHECK_LOG"
  echo
  cat "$CHECK_LOG"
  echo
  echo "Smoke Summary"
  echo "-------------"
  if [ -f "$SMOKE_ROOT/stage_n_acceptance_summary.txt" ]; then
    grep -E 'run_status=|successful_requests=|failed_requests=|driver_pool_budget_mode=|driver_kv_tag_enable=|kv_driver_tag_records=|kv_driver_untag_records=|uvm_phase_n_block_quota|phase_f_route_log_snapshot=' \
      "$SMOKE_ROOT/stage_n_acceptance_summary.txt" || true
  fi
  echo
  echo "Recent Block Evidence"
  echo "---------------------"
  grep -R -E 'phase_n_block_|phase_n_get_pool_state block_quota=1|phase_n_reclaim_pool block_quota=1' \
    "$SMOKE_ROOT" 2>/dev/null | tail -n 180 || true
} > "$SUMMARY"

cat "$SUMMARY"

if [ "$smoke_status" -ne 0 ]; then
  exit "$smoke_status"
fi
exit "$check_status"
