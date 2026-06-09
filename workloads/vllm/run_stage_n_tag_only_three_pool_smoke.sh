#!/usr/bin/env bash
# Stage N vLLM tag-only three-pool smoke.
#
# This wraps run_kv_fault_ratio.sh with a fixed, low-risk configuration so the
# long argument list cannot be corrupted by shell copy/paste. It writes the full
# command output and post-run acceptance checks into RUN_ROOT.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
PORT="${PORT:-8000}"

KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-}"
WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-}"
SCRATCH_TEMP_BUDGET_BYTES="${SCRATCH_TEMP_BUDGET_BYTES:-}"
DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY="${DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY:-1}"
DRIVER_POOL_STATE_SNAPSHOT_DEVICE="${DRIVER_POOL_STATE_SNAPSHOT_DEVICE:-0}"

RESET_COUNTERS="${RESET_COUNTERS:-0}"
MODE="${MODE:-trace}"
WITH_ADDRESS_LOG="${WITH_ADDRESS_LOG:-1}"
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
UVM_PHASE_N_BLOCK_QUOTA_ENABLE="${UVM_PHASE_N_BLOCK_QUOTA_ENABLE:-0}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-1}"
UVM_PHASE_N_NATIVE_BLOCK_ENABLE="${UVM_PHASE_N_NATIVE_BLOCK_ENABLE:-0}"
REQUIRE_PHASE_N_AUTO_HOOK="${REQUIRE_PHASE_N_AUTO_HOOK:-0}"
REQUIRE_PHASE_N_AUTO_RECLAIM="${REQUIRE_PHASE_N_AUTO_RECLAIM:-0}"
REQUIRE_STAGE_N_HARD_PARTITION="${REQUIRE_STAGE_N_HARD_PARTITION:-0}"
HARD_PARTITION_TOLERANCE_BYTES="${HARD_PARTITION_TOLERANCE_BYTES:-0}"
DRIVER_POOL_BUDGET_MODE="${DRIVER_POOL_BUDGET_MODE:-trace_only}"
VLLM_UVM_DRIVER_KV_TAG_ENABLE="${VLLM_UVM_DRIVER_KV_TAG_ENABLE:-0}"
VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE="${VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE:-0}"
VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP="${VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP:-0}"
UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-}"
if [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ]; then
  REQUIRE_PHASE_N_AUTO_RECLAIM=1
  REQUIRE_PHASE_N_AUTO_HOOK=1
  DRIVER_POOL_BUDGET_MODE=enforce
  VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE=1
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-1}"
else
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY:-0}"
fi
if [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" = "1" ]; then
  REQUIRE_PHASE_N_AUTO_HOOK=1
  DRIVER_POOL_BUDGET_MODE="${DRIVER_POOL_BUDGET_MODE:-soft}"
  [ "$DRIVER_POOL_BUDGET_MODE" = "trace_only" ] && DRIVER_POOL_BUDGET_MODE=soft
  UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-1}"
  if [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ]; then
    UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-1}"
    UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-0}"
  else
    UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-64}"
    UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-256}"
  fi
elif [ "$REQUIRE_PHASE_N_AUTO_HOOK" = "1" ]; then
  UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-1}"
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-0}"
  UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-64}"
  UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-256}"
else
  UVM_PHASE_N_AUTO_RECLAIM_ENABLE="${UVM_PHASE_N_AUTO_RECLAIM_ENABLE:-0}"
  UVM_PHASE_N_AUTO_RECLAIM_EXECUTE="${UVM_PHASE_N_AUTO_RECLAIM_EXECUTE:-0}"
  UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-64}"
  UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-256}"
fi
if [ -z "$SCRATCH_TEMP_BUDGET_BYTES" ]; then
  if [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" = "1" ]; then
    SCRATCH_TEMP_BUDGET_BYTES=16777216
  else
    SCRATCH_TEMP_BUDGET_BYTES=104857600
  fi
fi
if [ -z "$KV_BUDGET_BYTES" ]; then
  KV_BUDGET_BYTES=10737418240
fi
if [ -z "$WEIGHT_BUDGET_BYTES" ]; then
  if [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" = "1" ]; then
    WEIGHT_BUDGET_BYTES=21474836480
  else
    WEIGHT_BUDGET_BYTES=10737418240
  fi
fi
if [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" = "1" ]; then
  if [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ]; then
    UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-128}"
    UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-0}"
  else
    UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-1}"
    UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-4096}"
  fi
else
  UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-2}"
  UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-256}"
fi
if [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ]; then
  UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-0}"
else
  UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-256}"
fi

RUN_ROOT="${RUN_ROOT:-log/stage_n_tag_only_three_pool_smoke_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_ROOT"

TRACE_LOG="$RUN_ROOT/uvm_fault_stats.log"
ADDRESS_TRACE_LOG="$RUN_ROOT/uvm_fault_addrs.log"
ADDRESS_LOG="$RUN_ROOT/vllm_uvm_address_regions.log"
ALLOCATOR_LOG="$RUN_ROOT/vllm_uvm_allocator_trace.log"
SERVER_LOG="$RUN_ROOT/vllm_server.log"
BENCH_LOG="$RUN_ROOT/vllm_bench.log"
RESIDENCY_TRACE="$RUN_ROOT/vllm_uvm_pool_residency_stage_n.jsonl"
KV_RUNTIME_TRACE="$RUN_ROOT/vllm_uvm_kv_runtime_stage_n.jsonl"
KV_POINTER_MAP="$RUN_ROOT/vllm_uvm_kv_pointer_map_stage_n.jsonl"
RUN_LOG="$RUN_ROOT/run.log"
KERNEL_LOG="$RUN_ROOT/kernel_since_start.log"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/stage_n_acceptance_summary.txt"
ENV_FILE="$RUN_ROOT/env_stage_n_tag_only.txt"
COMMAND_FILE="$RUN_ROOT/command_stage_n_tag_only.sh"

export VLLM_UVM_PHASE_F_IOCTL_MODE=policy
export VLLM_UVM_RESIDENCY_TAG_ONLY=1
export VLLM_UVM_DRIVER_TAG_ENABLE=1
export VLLM_UVM_DRIVER_KV_TAG_ENABLE
export VLLM_UVM_DRIVER_SCRATCH_TAG_ENABLE=1
export VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1
export VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=1
export VLLM_UVM_DRIVER_POOL_BUDGET_MODE="$DRIVER_POOL_BUDGET_MODE"
export VLLM_UVM_DRIVER_POOL_BUDGETS="kv=${KV_BUDGET_BYTES},weight=${WEIGHT_BUDGET_BYTES},temp=${SCRATCH_TEMP_BUDGET_BYTES}"
export VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES="$SCRATCH_TEMP_BUDGET_BYTES"
export VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE
export VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP
export VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY="$DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY"
export VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_DEVICE="$DRIVER_POOL_STATE_SNAPSHOT_DEVICE"
export VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE=1
export VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_EVERY="${DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_EVERY:-1}"
export VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_MAX="${DRIVER_POOL_STATE_SNAPSHOT_ON_SCRATCH_FREE_MAX:-4}"
export VLLM_UVM_RESIDENCY_STAGE=stage_l6
export VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE="$KV_POINTER_MAP"
export VLLM_UVM_RESIDENCY_KV_POINTER_TAG_ENABLE=1
export VLLM_UVM_RESIDENCY_KV_POINTER_TAG_GRANULARITY=tensor

assert_fast_phase_f_route_log() {
  local param="/sys/module/nvidia_uvm/parameters/uvm_phase_f_route_log_delay_ms"
  local loaded_delay=""

  if [ ! -r "$param" ]; then
    return 0
  fi

  loaded_delay="$(cat "$param" 2>/dev/null || true)"
  if [ "$loaded_delay" = "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" ]; then
    return 0
  fi

  cat >&2 <<EOF
ERROR: loaded nvidia_uvm has uvm_phase_f_route_log_delay_ms=$loaded_delay,
but this smoke expects $UVM_PHASE_F_ROUTE_LOG_DELAY_MS.

Stage N tag-only emits many driver region/state ioctls. A non-zero route-log
delay adds that sleep to every logged ioctl and can make startup look hung.

Reload nvidia_uvm with:
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS=$UVM_PHASE_F_ROUTE_LOG_DELAY_MS RESET_COUNTERS=1 RUN_ROOT='$RUN_ROOT' ./run_stage_n_tag_only_three_pool_smoke.sh

or manually reload nvidia_uvm with uvm_phase_f_route_log_delay_ms=$UVM_PHASE_F_ROUTE_LOG_DELAY_MS.
EOF
  exit 2
}

if [ "$RESET_COUNTERS" != "1" ]; then
  assert_fast_phase_f_route_log
fi

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "MODEL=$MODEL"
  echo "PROMPTS=$PROMPTS"
  echo "REQUEST_RATE=$REQUEST_RATE"
  echo "OUTPUT_LEN=$OUTPUT_LEN"
  echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  echo "SERVER_MAX_NUM_SEQS=$SERVER_MAX_NUM_SEQS"
  echo "SERVER_GPU_MEMORY_UTILIZATION=$SERVER_GPU_MEMORY_UTILIZATION"
  echo "STARTUP_TIMEOUT=$STARTUP_TIMEOUT"
  echo "PORT=$PORT"
  echo "UVM_PHASE_F_ROUTE_LOG_DELAY_MS=$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  echo "KV_BUDGET_BYTES=$KV_BUDGET_BYTES"
  echo "WEIGHT_BUDGET_BYTES=$WEIGHT_BUDGET_BYTES"
  echo "SCRATCH_TEMP_BUDGET_BYTES=$SCRATCH_TEMP_BUDGET_BYTES"
  echo "DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY=$DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY"
  echo "DRIVER_POOL_STATE_SNAPSHOT_DEVICE=$DRIVER_POOL_STATE_SNAPSHOT_DEVICE"
  echo "REQUIRE_PHASE_N_AUTO_HOOK=$REQUIRE_PHASE_N_AUTO_HOOK"
  echo "REQUIRE_PHASE_N_AUTO_RECLAIM=$REQUIRE_PHASE_N_AUTO_RECLAIM"
  echo "REQUIRE_STAGE_N_HARD_PARTITION=$REQUIRE_STAGE_N_HARD_PARTITION"
  echo "HARD_PARTITION_TOLERANCE_BYTES=$HARD_PARTITION_TOLERANCE_BYTES"
  echo "DRIVER_POOL_BUDGET_MODE=$DRIVER_POOL_BUDGET_MODE"
  echo "UVM_PHASE_N_BLOCK_QUOTA_ENABLE=$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  echo "UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  echo "UVM_PHASE_N_NATIVE_BLOCK_ENABLE=$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  echo "UVM_PHASE_N_AUTO_RECLAIM_ENABLE=$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  echo "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  echo "UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY"
  echo "UVM_PHASE_N_AUTO_RECLAIM_EVERY=$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  echo "UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY=$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  echo "UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS=$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  echo "UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS=$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  echo "UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS=$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS"
  echo "VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE=$VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE"
  echo "VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP=$VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP"
  env | sort | grep -E '^VLLM_UVM_(PHASE_F_IOCTL_MODE|RESIDENCY|DRIVER)=' || true
  env | sort | grep -E '^VLLM_UVM_(RESIDENCY_|DRIVER_)' || true
} > "$ENV_FILE"

cmd=(
  ./run_kv_fault_ratio.sh
  --model "$MODEL"
  --mode "$MODE"
  --startup-timeout "$STARTUP_TIMEOUT"
  --trace-log "$TRACE_LOG"
  --address-trace-log "$ADDRESS_TRACE_LOG"
  --address-log "$ADDRESS_LOG"
  --allocator-log "$ALLOCATOR_LOG"
  --server-log "$SERVER_LOG"
  --bench-log "$BENCH_LOG"
  --port "$PORT"
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS"
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION"
  --max-model-len "$MAX_MODEL_LEN"
  --prompts "$PROMPTS"
  --request-rate "$REQUEST_RATE"
  --output-len "$OUTPUT_LEN"
  --uvm-policy-enable 1
  --uvm-policy-mode trace_only
  --uvm-policy-runtime-enable 0
  --uvm-policy-warmup-advise-gpu 0
  --uvm-weight-map-enable 1
  --uvm-weight-prefetch-enable 0
  --uvm-weight-prefetch-mode trace_only
  --uvm-weight-prefetch-trace-enable 0
  --uvm-weight-prefetch-hit-trace-enable 0
  --uvm-weight-prefetch-hit-driver-query-enable 0
  --uvm-weight-batch-executor-enable 0
  --uvm-weight-offload-enable 0
  --uvm-weight-offload-mode trace_only
  --uvm-weight-budget-cpu-offload-enable 0
  --uvm-init-weight-offload-enable 0
  --uvm-init-weight-offload-mode trace_only
  --uvm-init-weight-hot-prefetch-enable 0
  --uvm-init-weight-hot-prefetch-real-query-enable 0
  --uvm-kv-runtime-enable 1
  --uvm-kv-runtime-mode trace_only
  --uvm-kv-runtime-trace-file "$KV_RUNTIME_TRACE"
  --uvm-kv-runtime-prefix-evict-enable 0
  --uvm-kv-active-protect-enable 0
  --uvm-pool-coordinator-enable 0
  --uvm-pool-coordinator-trace-enable 0
  --uvm-residency-enable 1
  --uvm-residency-stage stage_l6
  --uvm-residency-mode trace_only
  --uvm-residency-trace-enable 1
  --uvm-residency-trace-file "$RESIDENCY_TRACE"
  --uvm-residency-kv-metadata-enable 1
  --uvm-residency-kv-data-offload-enable 0
  --uvm-residency-kv-data-offload-mode trace_only
  --uvm-residency-kv-prefetch-back-enable 0
  --uvm-residency-kv-prefetch-back-mode trace_only
  --uvm-residency-kv-pointer-map-file "$KV_POINTER_MAP"
  --uvm-residency-kv-pointer-tag-enable 1
  --uvm-residency-kv-pointer-tag-granularity tensor
  --uvm-driver-kv-tag-enable "$VLLM_UVM_DRIVER_KV_TAG_ENABLE"
  --uvm-driver-scratch-tag-enable 1
  --uvm-driver-temp-buffer-bytes "$SCRATCH_TEMP_BUDGET_BYTES"
  --uvm-driver-scratch-budget-enforce "$VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE"
  --uvm-driver-scratch-budget-enforce-warmup "$VLLM_UVM_DRIVER_SCRATCH_BUDGET_ENFORCE_WARMUP"
  --uvm-phase-f-route-log-delay-ms "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  --uvm-phase-n-block-quota-enable "$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  --uvm-phase-n-block-quota-trace-only "$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  --uvm-phase-n-native-block-enable "$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  --uvm-phase-n-auto-reclaim-enable "$UVM_PHASE_N_AUTO_RECLAIM_ENABLE"
  --uvm-phase-n-auto-reclaim-execute "$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE"
  --uvm-phase-n-auto-reclaim-execute-on-ready "$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY"
  --uvm-phase-n-auto-reclaim-every "$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  --uvm-phase-n-auto-reclaim-scan-every "$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  --uvm-phase-n-auto-reclaim-max-victims "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS"
  --uvm-phase-n-auto-reclaim-cooldown-epochs "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  --uvm-phase-n-auto-reclaim-failure-backoff-epochs "$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS"
  --uvm-residency-weight-enforce-hot-set 0
  --uvm-residency-weight-active-evict-enable 0
  --uvm-pool-registry-enable 1
  --uvm-scratch-pool-enable 0
)

if [ "$WITH_ADDRESS_LOG" = "1" ]; then
  cmd+=(--with-address-log)
fi
if [ "$RESET_COUNTERS" = "1" ]; then
  cmd+=(--reset-counters)
else
  cmd+=(--no-reset-counters)
fi

{
  printf 'cd %q\n' "$SCRIPT_DIR"
  printf '%q ' "${cmd[@]}"
  echo
} > "$COMMAND_FILE"
chmod +x "$COMMAND_FILE"

echo "Stage N tag-only three-pool smoke started."
echo "RUN_ROOT=$RUN_ROOT"
echo "Full output: $RUN_LOG"
echo "Acceptance summary: $SUMMARY"

RUN_START_EPOCH="$(date +%s)"

set +e
"${cmd[@]}" > "$RUN_LOG" 2>&1
status=$?
set -e

grep_tail() {
  local pattern="$1"
  shift
  grep -R -E "$pattern" "$@" 2>/dev/null | tail -n 120 || true
}

count_matches() {
  local pattern="$1"
  shift
  (grep -R -E "$pattern" "$@" 2>/dev/null || true) | wc -l | tr -d ' '
}

write_kernel_log_since_start() {
  local since_human

  since_human="$(date -u -d "@$RUN_START_EPOCH" '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || true)"
  if command -v journalctl >/dev/null 2>&1 && [ -n "$since_human" ]; then
    journalctl -k --no-pager --since "$since_human" > "$KERNEL_LOG" 2>/dev/null || true
  else
    dmesg -T > "$KERNEL_LOG" 2>/dev/null || true
  fi
}

write_phase_f_route_log_snapshot() {
  local proc_path="/proc/driver/nvidia-uvm/phase_f_route_log"

  if [ -r "$proc_path" ]; then
    cat "$proc_path" > "$ROUTE_LOG" 2>/dev/null || true
  else
    : > "$ROUTE_LOG"
  fi
}

write_kernel_log_since_start
write_phase_f_route_log_snapshot

write_residency_parse_summary() {
  local path="$1"
  if [ ! -s "$path" ]; then
    return 0
  fi
  python3 - "$path" <<'PY'
import collections
import json
import sys

path = sys.argv[1]
actions = collections.Counter()
pools = collections.Counter()
register = collections.Counter()
requests = collections.Counter()
budget_records = []
migration_like = 0
source_counts = collections.Counter()

for line in open(path, encoding="utf-8"):
    try:
        record = json.loads(line)
    except Exception:
        continue
    action = record.get("action")
    actions[action] += 1
    pool = record.get("pool")
    if pool:
        pools[pool] += 1
    if action == "residency_driver_pool_budget_config":
        budget_records.extend(record.get("records") or [])
    if action == "residency_register_region":
        driver_tag = record.get("driver_tag") or {}
        metadata = record.get("metadata") or {}
        register[(pool, driver_tag.get("reason"), bool(record.get("driver_registered")))] += 1
        source_counts[(pool, metadata.get("source"), bool(record.get("driver_registered")))] += 1
    if action == "residency_request":
        requests[record.get("reason")] += 1
    if action and any(
        token in action
        for token in (
            "prefetch_cpu",
            "prefetch_gpu",
            "safe_offload",
            "offload_cpu",
            "residency_kv_data_offload_issued",
        )
    ):
        migration_like += 1

print("Structured Residency Parse")
print("--------------------------")
print("actions=" + ",".join(f"{key}:{value}" for key, value in actions.most_common()))
print("pools=" + ",".join(f"{key}:{value}" for key, value in pools.most_common()))
print("request_reasons=" + ",".join(f"{key}:{value}" for key, value in requests.most_common()))
print("migration_like_records=%d" % migration_like)
print("driver_registered_true=%d" % sum(v for (_pool, _reason, registered), v in register.items() if registered))
print("driver_registered_false=%d" % sum(v for (_pool, _reason, registered), v in register.items() if not registered))
print("register_reasons=" + ",".join(
    f"{pool or 'none'}:{reason or 'none'}:{registered}:{value}"
    for (pool, reason, registered), value in register.most_common()
))
print("register_sources=" + ",".join(
    f"{pool or 'none'}:{source or 'none'}:{registered}:{value}"
    for (pool, source, registered), value in source_counts.most_common()
))
print("budget_records=" + ",".join(
    f"{record.get('pool')}:{record.get('budget_bytes')}:{record.get('mode')}:{record.get('status')}"
    for record in budget_records
))
PY
}

write_scratch_summary() {
  local path="$1"
  local budget_bytes="$2"
  if [ ! -s "$path" ]; then
    cat <<EOF
Scratch/Temp Driver Trace Summary
---------------------------------
scratch_driver_tag_records=0
scratch_driver_untag_records=0
scratch_driver_reject_records=0
scratch_driver_registered_records=0
scratch_driver_live_peak_bytes=0
scratch_driver_live_final_bytes=0
scratch_driver_max_alloc_bytes=0
scratch_driver_budget_bytes=$budget_bytes
scratch_driver_over_budget_records=0
scratch_driver_over_budget_peak_bytes=0
scratch_driver_over_100mib=0
scratch_driver_phase_counts=
scratch_driver_class_counts=
EOF
    return 0
  fi
  python3 - "$path" "$budget_bytes" <<'PY'
import collections
import re
import sys

path = sys.argv[1]
budget = int(sys.argv[2])
kv_pattern = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
live_by_region = {}
tag_records = 0
untag_records = 0
reject_records = 0
registered_records = 0
live_bytes = 0
live_peak_bytes = 0
max_alloc_bytes = 0
over_budget_records = 0
over_budget_peak_bytes = 0
phase_counts = collections.Counter()
class_counts = collections.Counter()

def fields(line: str) -> dict[str, str]:
    return {key: value for key, value in kv_pattern.findall(line)}

with open(path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if "TRACE_DRIVER_SCRATCH_TAG" in line:
            record = fields(line)
            tag_records += 1
            try:
                size = int(record.get("size_bytes", "0"))
            except ValueError:
                size = 0
            region_id = record.get("region_id") or f"missing:{tag_records}"
            if record.get("registered") == "1":
                registered_records += 1
            phase_counts[record.get("phase", "unknown")] += 1
            class_counts[record.get("predicted_class", "unknown")] += 1
            max_alloc_bytes = max(max_alloc_bytes, size)
            if region_id not in live_by_region:
                live_by_region[region_id] = size
                live_bytes += size
            else:
                live_bytes += max(size - live_by_region[region_id], 0)
                live_by_region[region_id] = size
            live_peak_bytes = max(live_peak_bytes, live_bytes)
            if budget > 0 and live_bytes > budget:
                over_budget_records += 1
                over_budget_peak_bytes = max(
                    over_budget_peak_bytes,
                    live_bytes - budget,
                )
        elif "TRACE_DRIVER_SCRATCH_UNTAG" in line:
            record = fields(line)
            untag_records += 1
            region_id = record.get("region_id")
            size = live_by_region.pop(region_id, 0) if region_id else 0
            if not size:
                try:
                    size = int(record.get("size_bytes", "0"))
                except ValueError:
                    size = 0
            live_bytes = max(live_bytes - size, 0)
        elif "TRACE_DRIVER_SCRATCH_REJECT" in line:
            reject_records += 1

print("Scratch/Temp Driver Trace Summary")
print("---------------------------------")
print(f"scratch_driver_tag_records={tag_records}")
print(f"scratch_driver_untag_records={untag_records}")
print(f"scratch_driver_reject_records={reject_records}")
print(f"scratch_driver_registered_records={registered_records}")
print(f"scratch_driver_live_peak_bytes={live_peak_bytes}")
print(f"scratch_driver_live_final_bytes={live_bytes}")
print(f"scratch_driver_max_alloc_bytes={max_alloc_bytes}")
print(f"scratch_driver_budget_bytes={budget}")
print(f"scratch_driver_over_budget_records={over_budget_records}")
print(f"scratch_driver_over_budget_peak_bytes={over_budget_peak_bytes}")
print(f"scratch_driver_over_100mib={1 if live_peak_bytes > 100 * 1024 * 1024 else 0}")
print("scratch_driver_phase_counts=" + ",".join(
    f"{key}:{value}" for key, value in phase_counts.most_common()
))
print("scratch_driver_class_counts=" + ",".join(
    f"{key}:{value}" for key, value in class_counts.most_common()
))
PY
}

write_driver_pool_state_summary() {
  local path="$1"
  if [ ! -s "$path" ]; then
    cat <<'EOF'
Driver Pool State Summary
-------------------------
driver_pool_state_records=0
driver_pool_state_failures=0
driver_pool_state_success_records=0
driver_pool_state_pools=
driver_pool_state_all_pools_present=0
EOF
    return 0
  fi
  python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
records = []
latest = {}
seen = set()
failures = 0
success_records = 0
positive_budget_pools = set()
required_fields = (
    "budget_bytes",
    "resident_gpu_bytes",
    "resident_cpu_bytes",
    "evictable_bytes",
    "protected_bytes",
    "victim_candidate_bytes",
    "victim_candidate_regions",
    "selected_victim_region_id",
    "selected_victim_bytes",
    "victim_count",
)
required_field_pools = set()

with open(path, encoding="utf-8") as handle:
    for line in handle:
        try:
            record = json.loads(line)
        except Exception:
            continue
        if record.get("action") != "residency_driver_pool_state":
            continue
        records.append(record)
        failures += int(record.get("failures") or 0)
        states = record.get("states") or {}
        if all(isinstance(states.get(pool), dict) for pool in ("kv", "weights", "scratch")):
            success_records += 1
        for pool, state in states.items():
            if isinstance(state, dict):
                latest[pool] = state
                seen.add(pool)
                if int(state.get("budget_bytes") or 0) > 0:
                    positive_budget_pools.add(pool)
                if all(field in state for field in required_fields):
                    required_field_pools.add(pool)

print("Driver Pool State Summary")
print("-------------------------")
print(f"driver_pool_state_records={len(records)}")
print(f"driver_pool_state_failures={failures}")
print(f"driver_pool_state_success_records={success_records}")
print("driver_pool_state_pools=" + ",".join(sorted(seen)))
print(
    "driver_pool_state_all_pools_present="
    + ("1" if all(pool in latest for pool in ("kv", "weights", "scratch")) else "0")
)
print(
    "driver_pool_state_all_budgets_positive="
    + ("1" if all(pool in positive_budget_pools for pool in ("kv", "weights", "scratch")) else "0")
)
print(
    "driver_pool_state_all_required_fields_present="
    + ("1" if all(pool in required_field_pools for pool in ("kv", "weights", "scratch")) else "0")
)
for pool in ("kv", "weights", "scratch"):
    state = latest.get(pool)
    if not state:
        print(f"pool_state_{pool}=missing")
        continue
    victims = state.get("victims") or []
    victim_text = ",".join(
        f"{int(victim.get('region_id') or 0)}:{int(victim.get('bytes') or 0)}"
        for victim in victims
    )
    print(
        f"pool_state_{pool}="
        f"budget_bytes:{int(state.get('budget_bytes') or 0)},"
        f"resident_gpu_bytes:{int(state.get('resident_gpu_bytes') or 0)},"
        f"resident_cpu_bytes:{int(state.get('resident_cpu_bytes') or 0)},"
        f"evictable_bytes:{int(state.get('evictable_bytes') or 0)},"
        f"protected_bytes:{int(state.get('protected_bytes') or 0)},"
        f"candidate_bytes:{int(state.get('victim_candidate_bytes') or 0)},"
        f"candidate_regions:{int(state.get('victim_candidate_regions') or 0)},"
        f"selected_victim_region_id:{int(state.get('selected_victim_region_id') or 0)},"
        f"selected_victim_bytes:{int(state.get('selected_victim_bytes') or 0)},"
        f"victim_count:{int(state.get('victim_count') or 0)},"
        f"metadata_only:{int(state.get('metadata_only') or 0)},"
        f"victims:{victim_text or 'none'}"
    )
PY
}

write_allocator_pool_state_summary() {
  local path="$1"
  if [ ! -s "$path" ]; then
    cat <<'EOF'
Allocator Driver Pool State Snapshot Summary
--------------------------------------------
allocator_driver_pool_state_records=0
allocator_driver_pool_state_success_records=0
allocator_driver_pool_state_pools=
allocator_driver_pool_state_all_pools_present=0
allocator_driver_pool_state_all_status_ok=0
EOF
    return 0
  fi
  python3 - "$path" <<'PY'
import re
import sys

path = sys.argv[1]
kv_pattern = re.compile(r"([A-Za-z0-9_]+)=([^ ]+)")
records = []
latest = {}
success_events = set()
positive_budget_pools = set()
required_fields = (
    "status",
    "budget_bytes",
    "resident_gpu_bytes",
    "resident_cpu_bytes",
    "evictable_bytes",
    "protected_bytes",
    "candidate_bytes",
    "candidate_regions",
    "selected_victim_region_id",
    "selected_victim_bytes",
    "victim_count",
)
required_field_pools = set()

def fields(line: str) -> dict[str, str]:
    return {key: value for key, value in kv_pattern.findall(line)}

with open(path, encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if "TRACE_DRIVER_POOL_STATE_SNAPSHOT" not in line:
            continue
        record = fields(line)
        records.append(record)
        pool = record.get("pool")
        if pool:
            latest[pool] = record
            if record.get("status") == "1":
                success_events.add(record.get("event_id", "unknown"))
            try:
                if int(record.get("budget_bytes", "0")) > 0:
                    positive_budget_pools.add(pool)
            except ValueError:
                pass
            if all(field in record for field in required_fields):
                required_field_pools.add(pool)

print("Allocator Driver Pool State Snapshot Summary")
print("--------------------------------------------")
print(f"allocator_driver_pool_state_records={len(records)}")
print(f"allocator_driver_pool_state_success_records={sum(1 for record in records if record.get('status') == '1')}")
print("allocator_driver_pool_state_pools=" + ",".join(sorted(latest)))
print(
    "allocator_driver_pool_state_all_pools_present="
    + ("1" if all(pool in latest for pool in ("kv", "weights", "scratch")) else "0")
)
print(
    "allocator_driver_pool_state_all_status_ok="
    + ("1" if all((latest.get(pool) or {}).get("status") == "1" for pool in ("kv", "weights", "scratch")) else "0")
)
print(
    "allocator_driver_pool_state_all_budgets_positive="
    + ("1" if all(pool in positive_budget_pools for pool in ("kv", "weights", "scratch")) else "0")
)
print(
    "allocator_driver_pool_state_all_required_fields_present="
    + ("1" if all(pool in required_field_pools for pool in ("kv", "weights", "scratch")) else "0")
)
for pool in ("kv", "weights", "scratch"):
    state = latest.get(pool)
    if not state:
        print(f"allocator_pool_state_{pool}=missing")
        continue
    print(
        f"allocator_pool_state_{pool}="
        f"status:{state.get('status', 'missing')},"
        f"budget_bytes:{state.get('budget_bytes', '0')},"
        f"resident_gpu_bytes:{state.get('resident_gpu_bytes', '0')},"
        f"resident_cpu_bytes:{state.get('resident_cpu_bytes', '0')},"
        f"evictable_bytes:{state.get('evictable_bytes', '0')},"
        f"protected_bytes:{state.get('protected_bytes', '0')},"
        f"candidate_bytes:{state.get('candidate_bytes', '0')},"
        f"candidate_regions:{state.get('candidate_regions', '0')},"
        f"selected_victim_region_id:{state.get('selected_victim_region_id', '0')},"
        f"selected_victim_bytes:{state.get('selected_victim_bytes', '0')},"
        f"victim_count:{state.get('victim_count', '0')},"
        f"metadata_only:{state.get('metadata_only', '0')},"
        f"scratch_region_id:{state.get('scratch_region_id', '0')}"
    )
PY
}

residency_records=0
driver_budget_records=0
tag_only_records=0
driver_register_records=0
driver_registered_true_records=0
driver_state_records=0
driver_pool_state_records=0
driver_pool_state_failures=0
driver_pool_state_success_records=0
driver_pool_state_all_pools_present=0
driver_pool_state_all_budgets_positive=0
driver_pool_state_all_required_fields_present=0
allocator_driver_pool_state_records=0
allocator_driver_pool_state_success_records=0
allocator_driver_pool_state_all_pools_present=0
allocator_driver_pool_state_all_status_ok=0
allocator_driver_pool_state_all_budgets_positive=0
allocator_driver_pool_state_all_required_fields_present=0
app_migration_records=0
scratch_driver_tag_records=0
scratch_driver_untag_records=0
scratch_driver_reject_records=0
scratch_driver_live_peak_bytes=0
scratch_driver_max_alloc_bytes=0
scratch_driver_over_budget_records=0
phase_n_auto_hook_records=0
phase_n_auto_hook_would_reclaim_records=0
phase_n_auto_hook_locked_context_records=0
phase_n_auto_hook_outer_records=0
phase_n_auto_hook_pool_records=0
phase_n_auto_hook_no_region_records=0
phase_n_auto_hook_execute_records=0
phase_n_auto_hook_success_records=0
phase_n_auto_hook_reclaimed_records=0
hard_partition_pass=0
hard_partition_all_driver_pools_under_budget=0
hard_partition_all_transient_pools_under_budget=0
hard_partition_sufficient_reclaim_to_budget=0
hard_partition_sufficient_deny_or_block=0
hard_partition_max_driver_over_budget_bytes=0
hard_partition_auto_reclaimed_bytes=0
hard_partition_scratch_over_budget_records=0
hard_partition_scratch_budget_reject_records=0
hard_partition_scratch_driver_reject_records=0
hard_partition_failure_reasons=not_checked
successful_requests=0
failed_requests=0

if [ -s "$RESIDENCY_TRACE" ]; then
  residency_records="$(wc -l < "$RESIDENCY_TRACE" | tr -d ' ')"
  driver_budget_records="$(count_matches 'residency_driver_pool_budget_config' "$RESIDENCY_TRACE")"
  tag_only_records="$(count_matches 'tag_only_no_application_residency_decision' "$RESIDENCY_TRACE")"
  driver_register_records="$(count_matches 'driver_region|driver_tag|residency_register_region' "$RESIDENCY_TRACE")"
  driver_registered_true_records="$(count_matches '"driver_registered":[[:space:]]*true' "$RESIDENCY_TRACE")"
  driver_state_records="$(count_matches 'driver_state|driver_region_state' "$RESIDENCY_TRACE")"
  driver_pool_state_records="$(count_matches 'residency_driver_pool_state' "$RESIDENCY_TRACE")"
  driver_pool_parse="$(write_driver_pool_state_summary "$RESIDENCY_TRACE")"
  driver_pool_state_failures="$(
    printf '%s\n' "$driver_pool_parse" |
      awk -F= '$1=="driver_pool_state_failures"{print $2}' |
      tail -n 1
  )"
  driver_pool_state_success_records="$(
    printf '%s\n' "$driver_pool_parse" |
      awk -F= '$1=="driver_pool_state_success_records"{print $2}' |
      tail -n 1
  )"
  driver_pool_state_all_pools_present="$(
    printf '%s\n' "$driver_pool_parse" |
      awk -F= '$1=="driver_pool_state_all_pools_present"{print $2}' |
      tail -n 1
  )"
  driver_pool_state_all_budgets_positive="$(
    printf '%s\n' "$driver_pool_parse" |
      awk -F= '$1=="driver_pool_state_all_budgets_positive"{print $2}' |
      tail -n 1
  )"
  driver_pool_state_all_required_fields_present="$(
    printf '%s\n' "$driver_pool_parse" |
      awk -F= '$1=="driver_pool_state_all_required_fields_present"{print $2}' |
      tail -n 1
  )"
  app_migration_records="$(count_matches 'prefetch_cpu|prefetch_gpu|safe_offload|offload_cpu|residency_kv_data_offload_issued' "$RESIDENCY_TRACE")"
fi

if [ -s "$ALLOCATOR_LOG" ]; then
  kv_driver_tag_records="$(count_matches 'TRACE_DRIVER_KV_TAG' "$ALLOCATOR_LOG")"
  kv_driver_untag_records="$(count_matches 'TRACE_DRIVER_KV_UNTAG' "$ALLOCATOR_LOG")"
  scratch_driver_tag_records="$(count_matches 'TRACE_DRIVER_SCRATCH_TAG' "$ALLOCATOR_LOG")"
  scratch_driver_untag_records="$(count_matches 'TRACE_DRIVER_SCRATCH_UNTAG' "$ALLOCATOR_LOG")"
  scratch_driver_reject_records="$(count_matches 'TRACE_DRIVER_SCRATCH_REJECT' "$ALLOCATOR_LOG")"
  allocator_driver_pool_state_records="$(count_matches 'TRACE_DRIVER_POOL_STATE_SNAPSHOT' "$ALLOCATOR_LOG")"
  scratch_parse="$(write_scratch_summary "$ALLOCATOR_LOG" "$SCRATCH_TEMP_BUDGET_BYTES")"
  scratch_driver_reject_records="$(
    printf '%s\n' "$scratch_parse" |
      awk -F= '$1=="scratch_driver_reject_records"{print $2}' |
      tail -n 1
  )"
  scratch_driver_live_peak_bytes="$(
    printf '%s\n' "$scratch_parse" |
      awk -F= '$1=="scratch_driver_live_peak_bytes"{print $2}' |
      tail -n 1
  )"
  scratch_driver_max_alloc_bytes="$(
    printf '%s\n' "$scratch_parse" |
      awk -F= '$1=="scratch_driver_max_alloc_bytes"{print $2}' |
      tail -n 1
  )"
  scratch_driver_over_budget_records="$(
    printf '%s\n' "$scratch_parse" |
      awk -F= '$1=="scratch_driver_over_budget_records"{print $2}' |
      tail -n 1
  )"
  allocator_pool_parse="$(write_allocator_pool_state_summary "$ALLOCATOR_LOG")"
  allocator_driver_pool_state_success_records="$(
    printf '%s\n' "$allocator_pool_parse" |
      awk -F= '$1=="allocator_driver_pool_state_success_records"{print $2}' |
      tail -n 1
  )"
  allocator_driver_pool_state_all_pools_present="$(
    printf '%s\n' "$allocator_pool_parse" |
      awk -F= '$1=="allocator_driver_pool_state_all_pools_present"{print $2}' |
      tail -n 1
  )"
  allocator_driver_pool_state_all_status_ok="$(
    printf '%s\n' "$allocator_pool_parse" |
      awk -F= '$1=="allocator_driver_pool_state_all_status_ok"{print $2}' |
      tail -n 1
  )"
  allocator_driver_pool_state_all_budgets_positive="$(
    printf '%s\n' "$allocator_pool_parse" |
      awk -F= '$1=="allocator_driver_pool_state_all_budgets_positive"{print $2}' |
      tail -n 1
  )"
  allocator_driver_pool_state_all_required_fields_present="$(
    printf '%s\n' "$allocator_pool_parse" |
      awk -F= '$1=="allocator_driver_pool_state_all_required_fields_present"{print $2}' |
      tail -n 1
  )"
fi
if [ -s "$KERNEL_LOG" ]; then
  phase_n_auto_hook_records="$(count_matches 'phase_n_auto_reclaim_hook' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_would_reclaim_records="$(count_matches 'phase_n_auto_reclaim_hook.*would_reclaim=1' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_locked_context_records="$(count_matches 'phase_n_auto_reclaim_hook.*skip=locked_context' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_outer_records="$(count_matches 'phase_n_auto_reclaim_hook.*phase=outer' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_pool_records="$(count_matches 'phase_n_auto_reclaim_hook.*pool=' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_no_region_records="$(count_matches 'phase_n_auto_reclaim_hook.*skip=no_region' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_execute_records="$(count_matches 'phase_n_auto_reclaim_hook.*[[:space:]]execute=1([[:space:]]|$)' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_success_records="$(count_matches 'phase_n_auto_reclaim_hook.*[[:space:]]execute=1([[:space:]]|$).*success=[1-9]' "$KERNEL_LOG" "$ROUTE_LOG")"
  phase_n_auto_hook_reclaimed_records="$(count_matches 'phase_n_auto_reclaim_hook.*reclaimed=[1-9]' "$KERNEL_LOG" "$ROUTE_LOG")"
elif [ -s "$ROUTE_LOG" ]; then
  phase_n_auto_hook_records="$(count_matches 'phase_n_auto_reclaim_hook' "$ROUTE_LOG")"
  phase_n_auto_hook_would_reclaim_records="$(count_matches 'phase_n_auto_reclaim_hook.*would_reclaim=1' "$ROUTE_LOG")"
  phase_n_auto_hook_locked_context_records="$(count_matches 'phase_n_auto_reclaim_hook.*skip=locked_context' "$ROUTE_LOG")"
  phase_n_auto_hook_outer_records="$(count_matches 'phase_n_auto_reclaim_hook.*phase=outer' "$ROUTE_LOG")"
  phase_n_auto_hook_pool_records="$(count_matches 'phase_n_auto_reclaim_hook.*pool=' "$ROUTE_LOG")"
  phase_n_auto_hook_no_region_records="$(count_matches 'phase_n_auto_reclaim_hook.*skip=no_region' "$ROUTE_LOG")"
  phase_n_auto_hook_execute_records="$(count_matches 'phase_n_auto_reclaim_hook.*[[:space:]]execute=1([[:space:]]|$)' "$ROUTE_LOG")"
  phase_n_auto_hook_success_records="$(count_matches 'phase_n_auto_reclaim_hook.*[[:space:]]execute=1([[:space:]]|$).*success=[1-9]' "$ROUTE_LOG")"
  phase_n_auto_hook_reclaimed_records="$(count_matches 'phase_n_auto_reclaim_hook.*reclaimed=[1-9]' "$ROUTE_LOG")"
fi
scratch_driver_live_peak_bytes="${scratch_driver_live_peak_bytes:-0}"
scratch_driver_reject_records="${scratch_driver_reject_records:-0}"
kv_driver_tag_records="${kv_driver_tag_records:-0}"
kv_driver_untag_records="${kv_driver_untag_records:-0}"
scratch_driver_max_alloc_bytes="${scratch_driver_max_alloc_bytes:-0}"
scratch_driver_over_budget_records="${scratch_driver_over_budget_records:-0}"
phase_n_auto_hook_records="${phase_n_auto_hook_records:-0}"
phase_n_auto_hook_would_reclaim_records="${phase_n_auto_hook_would_reclaim_records:-0}"
phase_n_auto_hook_locked_context_records="${phase_n_auto_hook_locked_context_records:-0}"
phase_n_auto_hook_outer_records="${phase_n_auto_hook_outer_records:-0}"
phase_n_auto_hook_pool_records="${phase_n_auto_hook_pool_records:-0}"
phase_n_auto_hook_no_region_records="${phase_n_auto_hook_no_region_records:-0}"
phase_n_auto_hook_execute_records="${phase_n_auto_hook_execute_records:-0}"
phase_n_auto_hook_success_records="${phase_n_auto_hook_success_records:-0}"
phase_n_auto_hook_reclaimed_records="${phase_n_auto_hook_reclaimed_records:-0}"
hard_partition_parse="$(
  python3 "$SCRIPT_DIR/check_stage_n_hard_partition.py" \
    "$RUN_ROOT" \
    --tolerance-bytes "$HARD_PARTITION_TOLERANCE_BYTES" 2>/dev/null || true
)"
hard_partition_pass="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_pass"{print $2}' |
    tail -n 1
)"
hard_partition_all_driver_pools_under_budget="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_all_driver_pools_under_budget"{print $2}' |
    tail -n 1
)"
hard_partition_all_transient_pools_under_budget="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_all_transient_pools_under_budget"{print $2}' |
    tail -n 1
)"
hard_partition_sufficient_reclaim_to_budget="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_sufficient_reclaim_to_budget"{print $2}' |
    tail -n 1
)"
hard_partition_sufficient_deny_or_block="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_sufficient_deny_or_block"{print $2}' |
    tail -n 1
)"
hard_partition_max_driver_over_budget_bytes="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_max_driver_over_budget_bytes"{print $2}' |
    tail -n 1
)"
hard_partition_auto_reclaimed_bytes="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_auto_reclaimed_bytes"{print $2}' |
    tail -n 1
)"
hard_partition_scratch_over_budget_records="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_scratch_over_budget_records"{print $2}' |
    tail -n 1
)"
hard_partition_scratch_budget_reject_records="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_scratch_budget_reject_records"{print $2}' |
    tail -n 1
)"
hard_partition_scratch_driver_reject_records="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_scratch_driver_reject_records"{print $2}' |
    tail -n 1
)"
hard_partition_failure_reasons="$(
  printf '%s\n' "$hard_partition_parse" |
    awk -F= '$1=="hard_partition_failure_reasons"{print $2}' |
    tail -n 1
)"
hard_partition_pass="${hard_partition_pass:-0}"
hard_partition_all_driver_pools_under_budget="${hard_partition_all_driver_pools_under_budget:-0}"
hard_partition_all_transient_pools_under_budget="${hard_partition_all_transient_pools_under_budget:-0}"
hard_partition_sufficient_reclaim_to_budget="${hard_partition_sufficient_reclaim_to_budget:-0}"
hard_partition_sufficient_deny_or_block="${hard_partition_sufficient_deny_or_block:-0}"
hard_partition_max_driver_over_budget_bytes="${hard_partition_max_driver_over_budget_bytes:-0}"
hard_partition_auto_reclaimed_bytes="${hard_partition_auto_reclaimed_bytes:-0}"
hard_partition_scratch_over_budget_records="${hard_partition_scratch_over_budget_records:-0}"
hard_partition_scratch_budget_reject_records="${hard_partition_scratch_budget_reject_records:-0}"
hard_partition_scratch_driver_reject_records="${hard_partition_scratch_driver_reject_records:-0}"
hard_partition_failure_reasons="${hard_partition_failure_reasons:-not_checked}"
driver_pool_state_failures="${driver_pool_state_failures:-0}"
driver_pool_state_success_records="${driver_pool_state_success_records:-0}"
driver_pool_state_all_pools_present="${driver_pool_state_all_pools_present:-0}"
driver_pool_state_all_budgets_positive="${driver_pool_state_all_budgets_positive:-0}"
driver_pool_state_all_required_fields_present="${driver_pool_state_all_required_fields_present:-0}"
allocator_driver_pool_state_success_records="${allocator_driver_pool_state_success_records:-0}"
allocator_driver_pool_state_all_pools_present="${allocator_driver_pool_state_all_pools_present:-0}"
allocator_driver_pool_state_all_status_ok="${allocator_driver_pool_state_all_status_ok:-0}"
allocator_driver_pool_state_all_budgets_positive="${allocator_driver_pool_state_all_budgets_positive:-0}"
allocator_driver_pool_state_all_required_fields_present="${allocator_driver_pool_state_all_required_fields_present:-0}"

if [ -s "$BENCH_LOG" ]; then
  successful_requests="$(grep -E 'Successful requests:' "$BENCH_LOG" | tail -n 1 | awk '{print $3}' || echo 0)"
  failed_requests="$(grep -E 'Failed requests:' "$BENCH_LOG" | tail -n 1 | awk '{print $3}' || echo 0)"
fi
successful_requests="${successful_requests:-0}"
failed_requests="${failed_requests:-0}"

{
  echo "Stage N Tag-Only Three-Pool Smoke Summary"
  echo "========================================="
  echo "run_status=$status"
  echo "run_root=$RUN_ROOT"
  echo "run_log=$RUN_LOG"
  echo "env_file=$ENV_FILE"
  echo "command_file=$COMMAND_FILE"
  echo
  echo "Artifacts"
  echo "---------"
  for path in \
    "$RESIDENCY_TRACE" \
    "$KV_RUNTIME_TRACE" \
    "$KV_POINTER_MAP" \
    "$BENCH_LOG" \
    "$SERVER_LOG" \
    "$ADDRESS_LOG" \
    "$ALLOCATOR_LOG" \
    "$KERNEL_LOG" \
    "$ROUTE_LOG" \
    "$TRACE_LOG" \
    "$ADDRESS_TRACE_LOG"; do
    if [ -e "$path" ]; then
      printf "%s %s bytes\n" "$path" "$(wc -c < "$path" | tr -d ' ')"
    else
      printf "%s MISSING\n" "$path"
    fi
  done
  echo
  echo "Benchmark"
  echo "---------"
  grep -E 'Successful requests:|Failed requests:|Output token throughput|Mean TTFT|Mean TPOT|Benchmark duration' "$BENCH_LOG" 2>/dev/null || true
  echo
  echo "Acceptance Counters"
  echo "-------------------"
  echo "residency_records=$residency_records"
  echo "driver_budget_records=$driver_budget_records"
  echo "tag_only_records=$tag_only_records"
  echo "driver_register_records=$driver_register_records"
  echo "driver_registered_true_records=$driver_registered_true_records"
  echo "driver_state_records=$driver_state_records"
  echo "driver_pool_state_records=$driver_pool_state_records"
  echo "driver_pool_state_failures=$driver_pool_state_failures"
  echo "driver_pool_state_success_records=$driver_pool_state_success_records"
  echo "driver_pool_state_all_pools_present=$driver_pool_state_all_pools_present"
  echo "driver_pool_state_all_budgets_positive=$driver_pool_state_all_budgets_positive"
  echo "driver_pool_state_all_required_fields_present=$driver_pool_state_all_required_fields_present"
  echo "allocator_driver_pool_state_records=$allocator_driver_pool_state_records"
  echo "allocator_driver_pool_state_success_records=$allocator_driver_pool_state_success_records"
  echo "allocator_driver_pool_state_all_pools_present=$allocator_driver_pool_state_all_pools_present"
  echo "allocator_driver_pool_state_all_status_ok=$allocator_driver_pool_state_all_status_ok"
  echo "allocator_driver_pool_state_all_budgets_positive=$allocator_driver_pool_state_all_budgets_positive"
  echo "allocator_driver_pool_state_all_required_fields_present=$allocator_driver_pool_state_all_required_fields_present"
  echo "scratch_driver_tag_records=$scratch_driver_tag_records"
  echo "scratch_driver_untag_records=$scratch_driver_untag_records"
  echo "scratch_driver_reject_records=$scratch_driver_reject_records"
  echo "kv_driver_tag_records=$kv_driver_tag_records"
  echo "kv_driver_untag_records=$kv_driver_untag_records"
  echo "scratch_driver_live_peak_bytes=$scratch_driver_live_peak_bytes"
  echo "scratch_driver_max_alloc_bytes=$scratch_driver_max_alloc_bytes"
  echo "scratch_driver_over_budget_records=$scratch_driver_over_budget_records"
  echo "phase_n_auto_hook_records=$phase_n_auto_hook_records"
  echo "phase_n_auto_hook_would_reclaim_records=$phase_n_auto_hook_would_reclaim_records"
  echo "phase_n_auto_hook_locked_context_records=$phase_n_auto_hook_locked_context_records"
  echo "phase_n_auto_hook_outer_records=$phase_n_auto_hook_outer_records"
  echo "phase_n_auto_hook_pool_records=$phase_n_auto_hook_pool_records"
  echo "phase_n_auto_hook_no_region_records=$phase_n_auto_hook_no_region_records"
  echo "phase_n_auto_hook_execute_records=$phase_n_auto_hook_execute_records"
  echo "phase_n_auto_hook_success_records=$phase_n_auto_hook_success_records"
  echo "phase_n_auto_hook_reclaimed_records=$phase_n_auto_hook_reclaimed_records"
  echo "require_phase_n_auto_hook=$REQUIRE_PHASE_N_AUTO_HOOK"
  echo "require_phase_n_auto_reclaim=$REQUIRE_PHASE_N_AUTO_RECLAIM"
  echo "phase_n_auto_reclaim_execute_on_ready=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY"
  echo "require_stage_n_hard_partition=$REQUIRE_STAGE_N_HARD_PARTITION"
  echo "hard_partition_tolerance_bytes=$HARD_PARTITION_TOLERANCE_BYTES"
  echo "hard_partition_pass=$hard_partition_pass"
  echo "hard_partition_all_driver_pools_under_budget=$hard_partition_all_driver_pools_under_budget"
  echo "hard_partition_all_transient_pools_under_budget=$hard_partition_all_transient_pools_under_budget"
  echo "hard_partition_sufficient_reclaim_to_budget=$hard_partition_sufficient_reclaim_to_budget"
  echo "hard_partition_sufficient_deny_or_block=$hard_partition_sufficient_deny_or_block"
  echo "hard_partition_max_driver_over_budget_bytes=$hard_partition_max_driver_over_budget_bytes"
  echo "hard_partition_auto_reclaimed_bytes=$hard_partition_auto_reclaimed_bytes"
  echo "hard_partition_scratch_over_budget_records=$hard_partition_scratch_over_budget_records"
  echo "hard_partition_scratch_budget_reject_records=$hard_partition_scratch_budget_reject_records"
  echo "hard_partition_scratch_driver_reject_records=$hard_partition_scratch_driver_reject_records"
  echo "driver_pool_budget_mode=$DRIVER_POOL_BUDGET_MODE"
  echo "driver_kv_tag_enable=$VLLM_UVM_DRIVER_KV_TAG_ENABLE"
  echo "uvm_phase_n_auto_reclaim_every=$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  echo "uvm_phase_n_auto_reclaim_scan_every=$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  echo "uvm_phase_n_auto_reclaim_cooldown_epochs=$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  echo "uvm_phase_n_block_quota_enable=$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  echo "uvm_phase_n_block_quota_trace_only=$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  echo "uvm_phase_n_native_block_enable=$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  echo "app_migration_records=$app_migration_records"
  echo "successful_requests=$successful_requests"
  echo "failed_requests=$failed_requests"
  echo
  write_residency_parse_summary "$RESIDENCY_TRACE"
  echo
  write_scratch_summary "$ALLOCATOR_LOG" "$SCRATCH_TEMP_BUDGET_BYTES"
  echo
  write_driver_pool_state_summary "$RESIDENCY_TRACE"
  echo
  write_allocator_pool_state_summary "$ALLOCATOR_LOG"
  echo
  printf '%s\n' "$hard_partition_parse"
  echo
  echo "Residency Driver Budget/Tag Evidence"
  echo "------------------------------------"
  grep_tail 'residency_driver_pool_budget_config|residency_driver_pool_state|residency_register_region|tag_only_no_application_residency_decision|driver_region|driver_tag|driver_state' "$RESIDENCY_TRACE"
  echo
  echo "Kernel Driver Evidence"
  echo "----------------------"
  grep_tail 'phase_f_register_region|phase_n_update_region_state|phase_n_set_pool_budget|phase_n_get_pool_state|phase_n_reclaim_pool|phase_n_auto_reclaim_hook|phase_n_block_|TRACE_DRIVER_KV_TAG|TRACE_DRIVER_KV_UNTAG|TRACE_DRIVER_SCRATCH_TAG|TRACE_DRIVER_SCRATCH_UNTAG|TRACE_DRIVER_SCRATCH_REJECT|TRACE_DRIVER_POOL_STATE_SNAPSHOT' "$RUN_ROOT" "$KERNEL_LOG" "$ROUTE_LOG" /var/log/kern.log
  echo
  echo "Auto Hook Evidence"
  echo "------------------"
  echo "phase_n_auto_hook_records=$phase_n_auto_hook_records"
  echo "phase_n_auto_hook_would_reclaim_records=$phase_n_auto_hook_would_reclaim_records"
  echo "phase_n_auto_hook_locked_context_records=$phase_n_auto_hook_locked_context_records"
  echo "phase_n_auto_hook_outer_records=$phase_n_auto_hook_outer_records"
  echo "phase_n_auto_hook_pool_records=$phase_n_auto_hook_pool_records"
  echo "phase_n_auto_hook_no_region_records=$phase_n_auto_hook_no_region_records"
  echo "phase_n_auto_hook_execute_records=$phase_n_auto_hook_execute_records"
  echo "phase_n_auto_hook_success_records=$phase_n_auto_hook_success_records"
  echo "phase_n_auto_hook_reclaimed_records=$phase_n_auto_hook_reclaimed_records"
  echo "require_phase_n_auto_hook=$REQUIRE_PHASE_N_AUTO_HOOK"
  echo "require_phase_n_auto_reclaim=$REQUIRE_PHASE_N_AUTO_RECLAIM"
  echo "phase_n_auto_reclaim_execute_on_ready=$UVM_PHASE_N_AUTO_RECLAIM_EXECUTE_ON_READY"
  echo "uvm_phase_n_auto_reclaim_every=$UVM_PHASE_N_AUTO_RECLAIM_EVERY"
  echo "uvm_phase_n_auto_reclaim_scan_every=$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY"
  echo "uvm_phase_n_auto_reclaim_cooldown_epochs=$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
  echo "uvm_phase_n_block_quota_enable=$UVM_PHASE_N_BLOCK_QUOTA_ENABLE"
  echo "uvm_phase_n_block_quota_trace_only=$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY"
  echo "uvm_phase_n_native_block_enable=$UVM_PHASE_N_NATIVE_BLOCK_ENABLE"
  echo "phase_f_route_log_snapshot=$ROUTE_LOG"
  grep_tail 'phase_n_auto_reclaim_hook|phase_n_block_' "$ROUTE_LOG" "$KERNEL_LOG" "$RUN_ROOT"
  echo
  echo "Application Migration Evidence"
  echo "------------------------------"
  grep_tail 'prefetch_cpu|prefetch_gpu|safe_offload|offload_cpu|residency_kv_data_offload_issued' "$RESIDENCY_TRACE" "$SERVER_LOG"
  echo
  echo "Fault Delta"
  echo "-----------"
  grep -E 'delta_faults=|delta_kv_faults=|delta_kv_ratio=' "$RUN_LOG" 2>/dev/null || true
  echo
  echo "Verdict"
  echo "-------"
  if { [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$status" -eq 0 ]; } &&
     [ -s "$RESIDENCY_TRACE" ] &&
     [ "$driver_budget_records" -gt 0 ] &&
     [ "$driver_registered_true_records" -gt 0 ] &&
     [ "$driver_pool_state_records" -gt 0 ] &&
     [ "$driver_pool_state_failures" -eq 0 ] &&
     [ "$driver_pool_state_success_records" -gt 0 ] &&
     [ "$driver_pool_state_all_pools_present" -eq 1 ] &&
     [ "$driver_pool_state_all_budgets_positive" -eq 1 ] &&
     [ "$driver_pool_state_all_required_fields_present" -eq 1 ] &&
     [ "$allocator_driver_pool_state_records" -gt 0 ] &&
     [ "$allocator_driver_pool_state_success_records" -gt 0 ] &&
     [ "$allocator_driver_pool_state_all_pools_present" -eq 1 ] &&
     [ "$allocator_driver_pool_state_all_status_ok" -eq 1 ] &&
     [ "$allocator_driver_pool_state_all_budgets_positive" -eq 1 ] &&
     [ "$allocator_driver_pool_state_all_required_fields_present" -eq 1 ] &&
     { [ "$scratch_driver_tag_records" -gt 0 ] || { [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] && [ "$scratch_driver_reject_records" -gt 0 ]; }; } &&
     [ "$tag_only_records" -gt 0 ] &&
     { [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "${failed_requests:-0}" = "0" ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "$phase_n_auto_hook_records" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "$phase_n_auto_hook_pool_records" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_would_reclaim_records" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_execute_records" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_success_records" -gt 0 ]; } &&
     { [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_reclaimed_records" -gt 0 ]; } &&
     { [ "$REQUIRE_STAGE_N_HARD_PARTITION" != "1" ] || [ "$hard_partition_pass" -eq 1 ]; } &&
     [ "$app_migration_records" -eq 0 ]; then
    if [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ]; then
      echo "PASS: Stage N hard partition proof produced per-pool budget evidence and either reclaim-to-budget or deny/block/reject evidence for any over-budget pool."
    else
      echo "PASS: Stage N tag-only smoke produced driver-backed KV/weight evidence, scratch driver tags, pool-state snapshots, auto hook pool evidence, tag-only decisions, and zero failed requests."
    fi
  else
    echo "WARN: Stage N smoke did not satisfy all acceptance checks."
    [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$status" -eq 0 ] || echo "- run_kv_fault_ratio.sh exited with status $status"
    [ -s "$RESIDENCY_TRACE" ] || echo "- missing or empty residency trace: $RESIDENCY_TRACE"
    [ "$driver_budget_records" -gt 0 ] || echo "- missing residency_driver_pool_budget_config records"
    [ "$driver_registered_true_records" -gt 0 ] || echo "- no region reached driver_registered=true"
    [ "$driver_pool_state_records" -gt 0 ] || echo "- missing residency_driver_pool_state records"
    [ "$driver_pool_state_failures" -eq 0 ] || echo "- driver_pool_state_failures=$driver_pool_state_failures"
    [ "$driver_pool_state_success_records" -gt 0 ] || echo "- no successful residency_driver_pool_state snapshot contained kv, weights, and scratch"
    [ "$driver_pool_state_all_pools_present" -eq 1 ] || echo "- latest residency_driver_pool_state is missing kv, weights, or scratch"
    [ "$driver_pool_state_all_budgets_positive" -eq 1 ] || echo "- residency_driver_pool_state did not report positive budgets for kv, weights, and scratch"
    [ "$driver_pool_state_all_required_fields_present" -eq 1 ] || echo "- residency_driver_pool_state is missing required budget/resident/candidate/victim fields"
    [ "$allocator_driver_pool_state_records" -gt 0 ] || echo "- missing allocator TRACE_DRIVER_POOL_STATE_SNAPSHOT records"
    [ "$allocator_driver_pool_state_success_records" -gt 0 ] || echo "- no allocator TRACE_DRIVER_POOL_STATE_SNAPSHOT record returned status=1"
    [ "$allocator_driver_pool_state_all_pools_present" -eq 1 ] || echo "- allocator pool-state snapshots are missing kv, weights, or scratch"
    [ "$allocator_driver_pool_state_all_status_ok" -eq 1 ] || echo "- latest allocator pool-state snapshot did not return status=1 for kv, weights, and scratch"
    [ "$allocator_driver_pool_state_all_budgets_positive" -eq 1 ] || echo "- allocator pool-state snapshot did not report positive budgets for kv, weights, and scratch"
    [ "$allocator_driver_pool_state_all_required_fields_present" -eq 1 ] || echo "- allocator pool-state snapshot is missing required budget/resident/candidate/victim fields"
    [ "$scratch_driver_tag_records" -gt 0 ] || { [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] && [ "$scratch_driver_reject_records" -gt 0 ]; } || echo "- missing TRACE_DRIVER_SCRATCH_TAG records"
    [ "$tag_only_records" -gt 0 ] || echo "- missing tag_only_no_application_residency_decision records"
	    [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "${failed_requests:-0}" = "0" ] || echo "- benchmark failed_requests=$failed_requests"
	    [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "$phase_n_auto_hook_records" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook records"
	    [ "$REQUIRE_PHASE_N_AUTO_HOOK" != "1" ] || [ "$phase_n_auto_hook_pool_records" -gt 0 ] || echo "- required phase_n_auto_reclaim_hook did not enter pool/budget branch"
	    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_would_reclaim_records" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook would_reclaim=1 records"
	    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_execute_records" -gt 0 ] || echo "- missing required phase_n_auto_reclaim_hook execute=1 records"
	    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_success_records" -gt 0 ] || echo "- missing successful automatic reclaim records"
	    [ "$REQUIRE_PHASE_N_AUTO_RECLAIM" != "1" ] || [ "$REQUIRE_STAGE_N_HARD_PARTITION" = "1" ] || [ "$phase_n_auto_hook_reclaimed_records" -gt 0 ] || echo "- automatic reclaim did not report reclaimed bytes"
	    [ "$REQUIRE_STAGE_N_HARD_PARTITION" != "1" ] || [ "$hard_partition_pass" -eq 1 ] || echo "- hard partition proof failed: $hard_partition_failure_reasons"
	    [ "$app_migration_records" -eq 0 ] || echo "- found application migration/offload records"
  fi
} > "$SUMMARY"

cat "$SUMMARY"
echo
echo "Stage N tag-only three-pool smoke finished."
echo "RUN_ROOT=$RUN_ROOT"
echo "summary=$SUMMARY"
echo "run_log=$RUN_LOG"

exit "$status"
