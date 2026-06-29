#!/usr/bin/env bash
# Capture and evaluate vLLM semantic weight address maps.
#
# This is a pre-Phase-0 sanity script: it does not offload or prefetch weights.
# It verifies that vLLM can emit a precise fixed/expert weight address map and,
# when fault logging is enabled, attributes UVM faults back to semantic weights.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-/tmp/vllm_weight_address_capture_eval}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PORT="${PORT:-8000}"
PROMPTS="${PROMPTS:-32}"
REQUEST_RATE="${REQUEST_RATE:-8}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MODE="${MODE:-trace}"
DEDUP_MODE="${DEDUP_MODE:-global-page}"

RUN_BENCH="${RUN_BENCH:-1}"
WITH_FAULT_LOG="${WITH_FAULT_LOG:-1}"
CAPTURE_STARTUP_FAULTS="${CAPTURE_STARTUP_FAULTS:-0}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"

INCLUDE_ALIASES="${INCLUDE_ALIASES:-1}"
PROBE_MANAGED="${PROBE_MANAGED:-1}"
REGISTER_EXPERT_TENSORS="${REGISTER_EXPERT_TENSORS:-1}"
QUERY_RESIDENCY="${QUERY_RESIDENCY:-1}"
QUERY_RESIDENCY_DEVICE="${QUERY_RESIDENCY_DEVICE:-0}"
QUERY_RESIDENCY_ROUND_PAGES="${QUERY_RESIDENCY_ROUND_PAGES:-1}"
INCLUDE_EXPERT_SLICES="${INCLUDE_EXPERT_SLICES:-1}"
QUERY_EXPERT_SLICES="${QUERY_EXPERT_SLICES:-1}"
HF_OFFLINE="${HF_OFFLINE:-1}"

# Optional: register the captured regions into the residency ledger and query
# real UVM residency after load_model. Keep this off by default because it is
# more intrusive than pure address capture.
ENABLE_RESIDENCY_TAGS="${ENABLE_RESIDENCY_TAGS:-0}"
REAL_RESIDENCY_SNAPSHOT_ENABLE="${REAL_RESIDENCY_SNAPSHOT_ENABLE:-0}"
DRIVER_TAG_ENABLE="${DRIVER_TAG_ENABLE:-0}"
DRIVER_TAG_REQUIRE_MANAGED="${DRIVER_TAG_REQUIRE_MANAGED:-1}"
REAL_RESIDENCY_MAX_REGIONS_PER_POOL="${REAL_RESIDENCY_MAX_REGIONS_PER_POOL:-0}"

SERVER_EXTRA_ENV_USER="${SERVER_EXTRA_ENV:-}"

usage() {
  cat <<'USAGE'
Usage:
  RUN_ROOT=/tmp/weight_capture PROMPTS=32 MAX_NUM_SEQS=8 \
    workloads/vllm/run_stage_m_weight_address_capture_eval.sh

Environment:
  RUN_ROOT                         Output directory, default /tmp/vllm_weight_address_capture_eval
  MODEL                            vLLM model, default Qwen/Qwen3-30B-A3B-FP8
  PROMPTS OUTPUT_LEN REQUEST_RATE  Benchmark shape
  MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  RUN_BENCH=0                      Start server and capture addresses without benchmark
  WITH_FAULT_LOG=0                 Disable per-fault address logging
  INCLUDE_ALIASES=1                Include tied/duplicate named tensors in JSONL map
  PROBE_MANAGED=1                  Probe whether each range is cudaMallocManaged
  QUERY_RESIDENCY=1                Query per-tensor UVM residency bytes
  QUERY_RESIDENCY_ROUND_PAGES=1    Query page-rounded ranges for small tensors
  INCLUDE_EXPERT_SLICES=1          Emit logical per-expert slices for fused MoE tensors
  QUERY_EXPERT_SLICES=1            Query residency for logical expert slices too
  HF_OFFLINE=1                     Export HF/Transformers offline mode for server and bench
  ENABLE_RESIDENCY_TAGS=1          Register captured weights in residency ledger
  REAL_RESIDENCY_SNAPSHOT_ENABLE=1 Query real UVM residency after weight map registration
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$RUN_ROOT"
if [[ "$HF_OFFLINE" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi
case_dir="$RUN_ROOT/default"
analysis_dir="$case_dir/weight_address_analysis"
mkdir -p "$case_dir" "$analysis_dir"

address_log="$case_dir/vllm_uvm_address_regions.log"
weight_map="$case_dir/vllm_uvm_weight_regions.jsonl"
allocator_log="$case_dir/vllm_uvm_allocator.log"
stats_log="$case_dir/uvm_fault_stats.log"
fault_addrs="$case_dir/uvm_fault_addrs.log"
server_log="$case_dir/vllm_server.log"
bench_log="$case_dir/vllm_bench.log"
residency_trace="$case_dir/vllm_uvm_pool_residency.jsonl"
real_residency_trace="$case_dir/vllm_uvm_real_residency.jsonl"
commands_log="$RUN_ROOT/commands.log"

server_extra_env="VLLM_UVM_WEIGHT_MAP_INCLUDE_ALIASES=$INCLUDE_ALIASES"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_PROBE_MANAGED=$PROBE_MANAGED"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_REGISTER_EXPERT_TENSORS=$REGISTER_EXPERT_TENSORS"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ENABLE=$QUERY_RESIDENCY"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_DEVICE=$QUERY_RESIDENCY_DEVICE"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_QUERY_RESIDENCY_ROUND_PAGES=$QUERY_RESIDENCY_ROUND_PAGES"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_INCLUDE_EXPERT_SLICES=$INCLUDE_EXPERT_SLICES"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_QUERY_EXPERT_SLICES=$QUERY_EXPERT_SLICES"
server_extra_env+=" VLLM_UVM_WEIGHT_MAP_REAL_RESIDENCY_SNAPSHOT_ENABLE=$REAL_RESIDENCY_SNAPSHOT_ENABLE"
if [[ "$HF_OFFLINE" == "1" ]]; then
  server_extra_env+=" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
fi
if [[ "$ENABLE_RESIDENCY_TAGS" == "1" ]]; then
  server_extra_env+=" VLLM_UVM_RESIDENCY_TAG_ONLY=1"
  server_extra_env+=" VLLM_UVM_DRIVER_TAG_ENABLE=$DRIVER_TAG_ENABLE"
  server_extra_env+=" VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED=$DRIVER_TAG_REQUIRE_MANAGED"
fi
if [[ -n "$SERVER_EXTRA_ENV_USER" ]]; then
  server_extra_env+=" $SERVER_EXTRA_ENV_USER"
fi

run_cmd=(
  "$SCRIPT_DIR/run_kv_fault_ratio.sh"
  --model "$MODEL"
  --port "$PORT"
  --mode "$MODE"
  --allocator-log "$allocator_log"
  --trace-log "$stats_log"
  --address-log "$address_log"
  --server-log "$server_log"
  --bench-log "$bench_log"
  --uvm-policy-enable 0
  --uvm-device-direct-enable 0
  --uvm-residency-enable "$ENABLE_RESIDENCY_TAGS"
  --uvm-residency-stage "weight_address_capture"
  --uvm-residency-mode "trace_only"
  --uvm-residency-trace-file "$residency_trace"
  --uvm-real-residency-enable "$REAL_RESIDENCY_SNAPSHOT_ENABLE"
  --uvm-real-residency-trace-file "$real_residency_trace"
  --uvm-real-residency-max-regions-per-pool "$REAL_RESIDENCY_MAX_REGIONS_PER_POOL"
  --uvm-kv-runtime-enable 0
  --uvm-weight-prefetch-enable 0
  --uvm-weight-offload-enable 0
  --uvm-weight-batch-executor-enable 0
  --uvm-pool-coordinator-enable 0
  --uvm-pool-registry-enable 0
  --uvm-scratch-pool-enable 0
  --uvm-weight-map-enable 1
  --uvm-weight-map-file "$weight_map"
  --server-max-num-seqs "$MAX_NUM_SEQS"
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION"
  --prompts "$PROMPTS"
  --request-rate "$REQUEST_RATE"
  --output-len "$OUTPUT_LEN"
  --capture-startup-faults "$CAPTURE_STARTUP_FAULTS"
  --server-extra-env "$server_extra_env"
)

if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]]; then
  run_cmd+=(--server-max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  run_cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ "$WITH_FAULT_LOG" == "1" ]]; then
  run_cmd+=(--address-trace-log "$fault_addrs")
fi
if [[ "$RUN_BENCH" != "1" ]]; then
  run_cmd+=(--no-bench)
fi
if [[ "$RESET_COUNTERS" == "1" ]]; then
  run_cmd+=(--reset-counters)
else
  run_cmd+=(--no-reset-counters)
fi

{
  echo "==========================================================="
  echo "Stage M Weight Address Capture Eval"
  echo "run_root=$RUN_ROOT"
  echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
  echo "max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-default}"
  echo "include_aliases=$INCLUDE_ALIASES probe_managed=$PROBE_MANAGED"
  echo "query_residency=$QUERY_RESIDENCY query_residency_device=$QUERY_RESIDENCY_DEVICE query_round_pages=$QUERY_RESIDENCY_ROUND_PAGES"
  echo "include_expert_slices=$INCLUDE_EXPERT_SLICES query_expert_slices=$QUERY_EXPERT_SLICES hf_offline=$HF_OFFLINE"
  echo "enable_residency_tags=$ENABLE_RESIDENCY_TAGS real_residency_snapshot=$REAL_RESIDENCY_SNAPSHOT_ENABLE"
  echo "==========================================================="
  printf 'command:'
  printf ' %q' "${run_cmd[@]}"
  echo
} | tee -a "$commands_log"

"${run_cmd[@]}"

analysis_cmd=(
  python3 "$SCRIPT_DIR/analyze_vllm_weight_address_map.py"
  --weight-map "$weight_map"
  --stats-log "$stats_log"
  --dedup-mode "$DEDUP_MODE"
  --out-dir "$analysis_dir"
)
if [[ "$WITH_FAULT_LOG" == "1" && -f "$fault_addrs" ]]; then
  analysis_cmd+=(--fault-log "$fault_addrs")
fi

{
  printf 'analysis_command:'
  printf ' %q' "${analysis_cmd[@]}"
  echo
} | tee -a "$commands_log"

"${analysis_cmd[@]}"

echo
echo "Weight address report: $analysis_dir/weight_address_map_report.md"
echo "Weight summary JSON:   $analysis_dir/weight_address_map_summary.json"
echo "Weight map JSONL:      $weight_map"
echo "Address log:           $address_log"
if [[ "$WITH_FAULT_LOG" == "1" ]]; then
  echo "Fault address log:     $fault_addrs"
fi
