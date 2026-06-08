#!/usr/bin/env bash
# Probe whether vLLM KV cache torch.zeros initialization produces visible UVM
# replayable faults, and classify those fault addresses against KV/weight ranges.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_kv_init_fault_probe_$(date -u +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PORT="${PORT:-8000}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-8}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
SCRATCH_TEMP_BUDGET_BYTES="${SCRATCH_TEMP_BUDGET_BYTES:-1073741824}"
KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-536870912}"
WEIGHT_BUDGET_BYTES="${WEIGHT_BUDGET_BYTES:-21474836480}"
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY="${UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY:-1}"
UVM_KV_INIT_FAULT_PROBE_SYNC="${UVM_KV_INIT_FAULT_PROBE_SYNC:-1}"
KEEP_ON_EXIT="${KEEP_ON_EXIT:-0}"

TRACE_LOG="$RUN_ROOT/uvm_fault_stats.log"
ADDRESS_TRACE_LOG="$RUN_ROOT/uvm_fault_addrs_with_kv_init_markers.log"
ADDRESS_LOG="$RUN_ROOT/vllm_uvm_address_regions.log"
ALLOCATOR_LOG="$RUN_ROOT/vllm_uvm_allocator_trace.log"
SERVER_LOG="$RUN_ROOT/vllm_server.log"
BENCH_LOG="$RUN_ROOT/vllm_bench.log"
RESIDENCY_TRACE="$RUN_ROOT/vllm_uvm_pool_residency_stage_n.jsonl"
KV_RUNTIME_TRACE="$RUN_ROOT/vllm_uvm_kv_runtime_stage_n.jsonl"
KV_POINTER_MAP="$RUN_ROOT/vllm_uvm_kv_pointer_map_stage_n.jsonl"
KV_INIT_PROBE_LOG="$RUN_ROOT/vllm_uvm_kv_init_fault_probe.jsonl"
ROUTE_LOG="$RUN_ROOT/phase_f_route_log_snapshot.log"
SUMMARY="$RUN_ROOT/kv_init_fault_probe_summary.txt"
SUMMARY_JSON="$RUN_ROOT/kv_init_fault_probe_summary.json"
FAULTS_CSV="$RUN_ROOT/kv_init_fault_classification.csv"
SHELL_LOG="$RUN_ROOT/shell.log"

mkdir -p "$RUN_ROOT"

KEEP_ARGS=()
if [ "$KEEP_ON_EXIT" = "1" ]; then
  KEEP_ARGS=(--keep)
fi

echo "Stage N KV init fault probe started."
echo "RUN_ROOT=$RUN_ROOT"
echo "MODEL=$MODEL"
echo "PROMPTS=$PROMPTS"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "UVM_KV_INIT_FAULT_PROBE_SYNC=$UVM_KV_INIT_FAULT_PROBE_SYNC"
echo "SHELL_LOG=$SHELL_LOG"

run_status=0
set +e
env \
  VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
  VLLM_UVM_RESIDENCY_TAG_ONLY=0 \
  VLLM_UVM_DRIVER_TAG_ENABLE=1 \
  VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=1 \
  VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE=1 \
  VLLM_UVM_DRIVER_POOL_BUDGET_MODE=soft \
  VLLM_UVM_DRIVER_POOL_BUDGETS="kv=${KV_BUDGET_BYTES},weight=${WEIGHT_BUDGET_BYTES},temp=${SCRATCH_TEMP_BUDGET_BYTES}" \
  VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES="$SCRATCH_TEMP_BUDGET_BYTES" \
  VLLM_UVM_RESIDENCY_STAGE=stage_l6 \
  VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
  VLLM_UVM_KV_INIT_FAULT_PROBE=1 \
  VLLM_UVM_KV_INIT_FAULT_PROBE_FILE="$KV_INIT_PROBE_LOG" \
  VLLM_UVM_KV_INIT_FAULT_PROBE_SYNC="$UVM_KV_INIT_FAULT_PROBE_SYNC" \
  VLLM_UVM_CAPTURE_STARTUP_FAULTS=1 \
  ./run_kv_fault_ratio.sh \
    --model "$MODEL" \
    --mode trace \
    --startup-timeout "$STARTUP_TIMEOUT" \
    --trace-log "$TRACE_LOG" \
    --address-trace-log "$ADDRESS_TRACE_LOG" \
    --address-log "$ADDRESS_LOG" \
    --allocator-log "$ALLOCATOR_LOG" \
    --server-log "$SERVER_LOG" \
    --bench-log "$BENCH_LOG" \
    --port "$PORT" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --uvm-policy-enable 1 \
    --uvm-policy-mode trace_only \
    --uvm-policy-runtime-enable 0 \
    --uvm-policy-warmup-advise-gpu 0 \
    --uvm-weight-map-enable 1 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-offload-enable 0 \
    --uvm-init-weight-offload-enable 0 \
    --uvm-kv-runtime-enable 1 \
    --uvm-kv-runtime-mode trace_only \
    --uvm-kv-runtime-budget-blocks 1 \
    --uvm-kv-runtime-trace-file "$KV_RUNTIME_TRACE" \
    --uvm-kv-runtime-prefix-evict-enable 0 \
    --uvm-residency-enable 1 \
    --uvm-residency-stage stage_l6 \
    --uvm-residency-mode trace_only \
    --uvm-residency-trace-enable 1 \
    --uvm-residency-trace-file "$RESIDENCY_TRACE" \
    --uvm-residency-kv-metadata-enable 1 \
    --uvm-residency-kv-data-offload-enable 0 \
    --uvm-residency-kv-pointer-map-file "$KV_POINTER_MAP" \
    --uvm-residency-kv-pointer-tag-enable 1 \
    --uvm-residency-kv-pointer-tag-granularity tensor \
    --uvm-driver-kv-tag-enable 1 \
    --uvm-kv-init-fault-probe 1 \
    --uvm-kv-init-fault-probe-file "$KV_INIT_PROBE_LOG" \
    --uvm-kv-init-fault-probe-sync "$UVM_KV_INIT_FAULT_PROBE_SYNC" \
    --uvm-driver-scratch-tag-enable 1 \
    --uvm-driver-temp-buffer-bytes "$SCRATCH_TEMP_BUDGET_BYTES" \
    --uvm-driver-scratch-budget-enforce 0 \
    --uvm-phase-f-route-log-delay-ms 0 \
    --uvm-phase-n-block-quota-enable 1 \
    --uvm-phase-n-block-quota-trace-only "$UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY" \
    --uvm-phase-n-auto-reclaim-enable 1 \
    --uvm-phase-n-auto-reclaim-execute 0 \
    --uvm-phase-n-auto-reclaim-execute-on-ready 0 \
    --uvm-phase-n-auto-reclaim-every 1 \
    --uvm-phase-n-auto-reclaim-scan-every 0 \
    --uvm-phase-n-auto-reclaim-max-victims 128 \
    --uvm-phase-n-auto-reclaim-cooldown-epochs 0 \
    --uvm-phase-n-auto-reclaim-failure-backoff-epochs 0 \
    --uvm-pool-registry-enable 1 \
    --uvm-scratch-pool-enable 0 \
    --capture-startup-faults 1 \
    --with-address-log \
    --reset-counters \
    "${KEEP_ARGS[@]}" \
  > "$SHELL_LOG" 2>&1
run_status=$?
set -e

if [ -r /proc/driver/nvidia-uvm/phase_f_route_log ]; then
  cat /proc/driver/nvidia-uvm/phase_f_route_log > "$ROUTE_LOG" || true
fi

python3 "$SCRIPT_DIR/analyze_kv_init_fault_probe.py" \
  --allocator-log "$ALLOCATOR_LOG" \
  --address-log "$ADDRESS_LOG" \
  --fault-log "$ADDRESS_TRACE_LOG" \
  --kv-init-probe-log "$KV_INIT_PROBE_LOG" \
  --route-log "$ROUTE_LOG" \
  --summary-out "$SUMMARY" \
  --json-out "$SUMMARY_JSON" \
  --faults-csv "$FAULTS_CSV" \
  || true

{
  echo
  echo "Run status: $run_status"
  echo "Run root: $RUN_ROOT"
  echo "Summary: $SUMMARY"
  echo "Summary JSON: $SUMMARY_JSON"
  echo "Fault classification CSV: $FAULTS_CSV"
  echo "Address/fault log with markers: $ADDRESS_TRACE_LOG"
  echo "KV init probe JSONL: $KV_INIT_PROBE_LOG"
  echo "Allocator log: $ALLOCATOR_LOG"
  echo "Shell log: $SHELL_LOG"
} | tee -a "$SUMMARY"

cat "$SUMMARY"
exit "$run_status"
