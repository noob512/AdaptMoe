#!/usr/bin/env bash
# Stage M smoke: test whether weight can be held near 10GiB while KV budget
# is increased, using the existing Stage L full-implementation A/B runner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REPEAT="${REPEAT:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_weight10g_kv2g_ab_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""

WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-10737418240}"       # 10 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"                # 2 GiB
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-10737418240}"  # 10 GiB
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-16}"
WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-16}"
WEIGHT_VICTIM_ACTION="${WEIGHT_VICTIM_ACTION:-offload_cpu}"
WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="${WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE:-0}"

SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-2}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"

SKIP_TRACE_CONTROL=1

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_weight10g_kv2g_ab.sh [options]

Purpose:
  Run the Stage L full-implementation vs pure-UVM A/B test with a tighter
  weight target (~10GiB) and a larger KV budget (default 2GiB).

Default smoke setup:
  weight_gpu_bytes = 10737418240  # 10 GiB
  kv_gpu_bytes     = 2147483648   # 2 GiB
  init_weight_max  = 10737418240  # 10 GiB
  init_weight_max_experts_per_layer = 16
  weight_max_victims_per_step = 16
  weight_victim_action = offload_cpu
  weight_budget_cpu_offload_enable = 0
  prompts          = 2
  output_len       = 128
  request_rate     = 2
  repeat           = 1
  server_max_num_seqs = 2
  server_gpu_memory_utilization = 0.70

This wrapper also enables the Stage M init-weight offload budget path by
exporting:
  VLLM_UVM_WEIGHT_BUDGET_BYTES=10737418240
  VLLM_UVM_WEIGHT_BUDGET_MODE=enforce

Options:
  --run-root <path>
  --log-file <path>
  --model <name>
  --prompts <int>
  --output-len <int>
  --request-rate <float>
  --repeat <int>
  --weight-gpu-bytes <n>
  --kv-gpu-bytes <n>
  --init-weight-max-bytes <n>
  --init-weight-max-experts-per-layer <n>
  --weight-max-victims-per-step <n>
  --weight-victim-action <advise_cpu|prefetch_cpu|offload_cpu>
  --weight-budget-cpu-offload-enable <0|1>
  --server-max-num-seqs <n>
  --server-gpu-memory-utilization <f>
  --include-trace-control
  --skip-trace-control
  -h, --help

Example:
  ./run_stage_m_weight10g_kv2g_ab.sh \
    --log-file log/stage_m_weight10g_kv2g_ab.log
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
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --init-weight-max-bytes) INIT_WEIGHT_MAX_BYTES="$2"; shift 2 ;;
    --init-weight-max-experts-per-layer) INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-max-victims-per-step) WEIGHT_MAX_VICTIMS_PER_STEP="$2"; shift 2 ;;
    --weight-victim-action) WEIGHT_VICTIM_ACTION="$2"; shift 2 ;;
    --weight-budget-cpu-offload-enable) WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="$2"; shift 2 ;;
    --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --include-trace-control) SKIP_TRACE_CONTROL=0; shift ;;
    --skip-trace-control) SKIP_TRACE_CONTROL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_weight10g_kv2g_ab.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

if ! [[ "$WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE" =~ ^[01]$ ]]; then
  echo "--weight-budget-cpu-offload-enable must be 0 or 1" >&2
  exit 2
fi

echo "Stage M weight-10GiB / KV-2GiB A/B evaluation started."
echo "Full shell output: $LOG_FILE"
echo "Run root: $RUN_ROOT"
echo "Follow progress with: tail -f '$LOG_FILE'"

INNER_ARGS=(
  --run-root "$RUN_ROOT"
  --log-file "$LOG_FILE"
  --model "$MODEL"
  --prompts "$PROMPTS"
  --output-len "$OUTPUT_LEN"
  --request-rate "$REQUEST_RATE"
  --repeat "$REPEAT"
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS"
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION"
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES"
  --kv-gpu-bytes "$KV_GPU_BYTES"
  --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES"
  --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER"
  --weight-max-victims-per-step "$WEIGHT_MAX_VICTIMS_PER_STEP"
  --weight-budget-cpu-offload-enable "$WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE"
)

if [ "$SKIP_TRACE_CONTROL" = "1" ]; then
  INNER_ARGS+=(--skip-trace-control)
fi

VLLM_UVM_WEIGHT_BUDGET_BYTES="$WEIGHT_GPU_BYTES" \
VLLM_UVM_WEIGHT_BUDGET_MODE=enforce \
VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="$WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE" \
WEIGHT_VICTIM_ACTION="$WEIGHT_VICTIM_ACTION" \
  "$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" "${INNER_ARGS[@]}"
