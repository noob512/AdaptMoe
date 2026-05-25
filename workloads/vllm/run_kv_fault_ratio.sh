#!/usr/bin/env bash
# Single-process KV fault ratio runner for vLLM + UVM.
#
# Flow (same server process):
# 1) start ONE vLLM server process
# 2) wait until a fresh kv_cache range appears in vLLM address log
# 3) configure nvidia_uvm KV range params immediately
# 4) run benchmark against that SAME server process
# 5) print fault stats and cleanup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_DIR="/sys/module/nvidia_uvm/parameters"

MODEL="Qwen/Qwen3-30B-A3B-FP8"
PORT=8000
PROMPTS=5
REQUEST_RATE=5
OUTPUT_LEN=512
SEED=42
STARTUP_TIMEOUT=600
CHECK_INTERVAL=2
SERVER_MAX_NUM_SEQS="${VLLM_SERVER_MAX_NUM_SEQS:-8}"
SERVER_GPU_MEMORY_UTILIZATION="${VLLM_SERVER_GPU_MEMORY_UTILIZATION:-0.85}"
SERVER_MAX_MODEL_LEN="${VLLM_SERVER_MAX_MODEL_LEN:-}"
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"

DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
ADDRESS_LOG="/tmp/vllm_uvm_address_regions.log"
SERVER_LOG="$SCRIPT_DIR/vllm_server_uvm_single_process.log"
BENCH_LOG="$SCRIPT_DIR/results/uvm_kv_bench_single_process.log"
TRACE_LOG=""  # auto-generated in validate_inputs if not set via --trace-log
ADDRESS_TRACE_LOG=""  # auto-generated when --with-address-log and not set via --address-trace-log
ALLOCATOR_TRACE_LOG=""  # optional UVM allocator trace log
MODE="dmesg" # dmesg | trace
TRACE_DIR=""
UVM_TRACE_MIN_BYTES="${UVM_TRACE_MIN_BYTES:-1048576}"
UVM_POLICY_ENABLE="${VLLM_UVM_POLICY_ENABLE:-1}"
UVM_POLICY_MODE="${VLLM_UVM_POLICY_MODE:-trace_only}"
UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES="${VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES:-1048576}"
UVM_POLICY_WARMUP_ADVISE_GPU="${VLLM_UVM_POLICY_WARMUP_ADVISE_GPU:-0}"
UVM_POLICY_RUNTIME_ENABLE="${VLLM_UVM_POLICY_RUNTIME_ENABLE:-0}"
UVM_UNKNOWN_DETAIL_ENABLE="${VLLM_UVM_UNKNOWN_DETAIL_ENABLE:-0}"
UVM_UNKNOWN_DETAIL_MIN_BYTES="${VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES:-0}"
UVM_GAP_WATCH_ENABLE="${VLLM_UVM_GAP_WATCH_ENABLE:-0}"
UVM_GAP_WATCH_NAME="${VLLM_UVM_GAP_WATCH_NAME:-gap_watch}"
UVM_GAP_WATCH_START="${VLLM_UVM_GAP_WATCH_START:-}"
UVM_GAP_WATCH_END="${VLLM_UVM_GAP_WATCH_END:-}"
UVM_GAP_WATCH_ALL_CLASSES="${VLLM_UVM_GAP_WATCH_ALL_CLASSES:-1}"
UVM_GAP_WATCH_MIN_BYTES="${VLLM_UVM_GAP_WATCH_MIN_BYTES:-0}"
UVM_GAP_WATCH_TARGET_CLASS="${VLLM_UVM_GAP_WATCH_TARGET_CLASS:-any}"
UVM_GAP_WATCH_POLICY_ACTION="${VLLM_UVM_GAP_WATCH_POLICY_ACTION:-observe}"
UVM_GAP_WATCH_CONTROL_FILE="${VLLM_UVM_GAP_WATCH_CONTROL_FILE:-}"
UVM_GAP_WATCH_REFRESH_MS="${VLLM_UVM_GAP_WATCH_REFRESH_MS:-250}"
UVM_DEVICE_DIRECT_ENABLE="${VLLM_UVM_DEVICE_DIRECT_ENABLE:-0}"
UVM_DEVICE_DIRECT_MIN_BYTES="${VLLM_UVM_DEVICE_DIRECT_MIN_BYTES:-4096}"
UVM_DEVICE_DIRECT_MAX_BYTES="${VLLM_UVM_DEVICE_DIRECT_MAX_BYTES:-1048576}"
UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES="${VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES:-268435456}"
UVM_DEVICE_DIRECT_BACKEND="${VLLM_UVM_DEVICE_DIRECT_BACKEND:-cuda_malloc}"
UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="${VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-}"
UVM_DEVICE_DIRECT_TARGET_PHASES="${VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES:-enabled:attention,enabled:moe,enabled:model_forward}"
UVM_KV_BUDGET_BYTES="${VLLM_UVM_KV_BUDGET_BYTES:-0}"
UVM_KV_BUDGET_MODE="${VLLM_UVM_KV_BUDGET_MODE:-trace_only}"
UVM_KV_RUNTIME_ENABLE="${VLLM_UVM_KV_RUNTIME_ENABLE:-0}"
UVM_KV_RUNTIME_MODE="${VLLM_UVM_KV_RUNTIME_MODE:-trace_only}"
UVM_KV_RUNTIME_BUDGET_BYTES="${VLLM_UVM_KV_RUNTIME_BUDGET_BYTES:-0}"
UVM_KV_RUNTIME_BUDGET_BLOCKS="${VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS:-0}"
UVM_KV_RUNTIME_TRACE_FILE="${VLLM_UVM_KV_RUNTIME_TRACE_FILE:-}"
UVM_KV_RUNTIME_EVICTION_POLICY="${VLLM_UVM_KV_RUNTIME_EVICTION_POLICY:-lru_prefix_cache}"
UVM_KV_RUNTIME_CANDIDATE_LIMIT="${VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT:-16}"
UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE="${VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE:-0}"
UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS="${VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS:-0}"
UVM_KV_ACTIVE_PROTECT_ENABLE="${VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE:-0}"
UVM_KV_ACTIVE_PROTECT_MODE="${VLLM_UVM_KV_ACTIVE_PROTECT_MODE:-trace_only}"
UVM_KV_ACTIVE_PROTECT_DEVICE="${VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE:-0}"
UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS="${VLLM_UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS:-0}"
UVM_WEIGHT_BUDGET_BYTES="${VLLM_UVM_WEIGHT_BUDGET_BYTES:-0}"
UVM_WEIGHT_BUDGET_MODE="${VLLM_UVM_WEIGHT_BUDGET_MODE:-trace_only}"
UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="${VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE:-0}"
UVM_WEIGHT_MAP_ENABLE="${VLLM_UVM_WEIGHT_MAP_ENABLE:-1}"
UVM_WEIGHT_MAP_FILE="${VLLM_UVM_WEIGHT_MAP_FILE:-}"
UVM_MOE_ROUTING_TRACE_ENABLE="${VLLM_UVM_MOE_ROUTING_TRACE_ENABLE:-0}"
UVM_MOE_ROUTING_TRACE_FILE="${VLLM_UVM_MOE_ROUTING_TRACE_FILE:-}"
UVM_WEIGHT_PREFETCH_ENABLE="${VLLM_UVM_WEIGHT_PREFETCH_ENABLE:-0}"
UVM_WEIGHT_PREFETCH_MODE="${VLLM_UVM_WEIGHT_PREFETCH_MODE:-trace_only}"
UVM_WEIGHT_PREFETCH_TRACE_ENABLE="${VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE:-1}"
UVM_WEIGHT_PREFETCH_TRACE_FILE="${VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE:-}"
UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-67108864}"
UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-2}"
UVM_WEIGHT_PREFETCH_SKIP_RESIDENT="${VLLM_UVM_WEIGHT_PREFETCH_SKIP_RESIDENT:-1}"
UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${VLLM_UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${VLLM_UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="${VLLM_UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS:-1}"
UVM_WEIGHT_PREFETCH_TARGET_ROLES="${VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES:-moe_gate_up,moe_down}"
UVM_WEIGHT_PREFETCH_DEVICE="${VLLM_UVM_WEIGHT_PREFETCH_DEVICE:--1}"
UVM_WEIGHT_PREFETCH_PLAN_FILE="${VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE:-}"
UVM_WEIGHT_PREFETCH_REQUIRE_PLAN="${VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN:-0}"
UVM_WEIGHT_PREFETCH_SOURCE="${VLLM_UVM_WEIGHT_PREFETCH_SOURCE:-router_topk}"
UVM_ORACLE_EXPERT_TRACE_FILE="${VLLM_UVM_ORACLE_EXPERT_TRACE_FILE:-}"
UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS="${VLLM_UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS:-1}"
UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE="${VLLM_UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE:-0}"
UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="${VLLM_UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE:-0}"
UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="${VLLM_UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP:-64}"
UVM_WEIGHT_BATCH_EXECUTOR_ENABLE="${VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE:-1}"
UVM_WEIGHT_EXECUTOR="${VLLM_UVM_WEIGHT_EXECUTOR:-cpp_batch}"
UVM_WEIGHT_OFFLOAD_ENABLE="${VLLM_UVM_WEIGHT_OFFLOAD_ENABLE:-0}"
UVM_WEIGHT_OFFLOAD_MODE="${VLLM_UVM_WEIGHT_OFFLOAD_MODE:-trace_only}"
UVM_WEIGHT_OFFLOAD_PLAN_FILE="${VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE:-}"
UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP="${VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP:-67108864}"
UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="${VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER:-1}"
UVM_WEIGHT_OFFLOAD_TARGET_ROLES="${VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES:-moe_gate_up,moe_down}"
UVM_INIT_KV_SIZING_ENABLE="${VLLM_UVM_INIT_KV_SIZING_ENABLE:-0}"
UVM_INIT_KV_SIZING_POLICY="${VLLM_UVM_INIT_KV_SIZING_POLICY:-report_only}"
UVM_INIT_KV_MAX_BYTES="${VLLM_UVM_INIT_KV_MAX_BYTES:-0}"
UVM_INIT_KV_MAX_BLOCKS="${VLLM_UVM_INIT_KV_MAX_BLOCKS:-0}"
UVM_INIT_KV_MIN_BLOCKS="${VLLM_UVM_INIT_KV_MIN_BLOCKS:-0}"
UVM_INIT_KV_TRACE_FILE="${VLLM_UVM_INIT_KV_TRACE_FILE:-}"
UVM_INIT_WEIGHT_OFFLOAD_ENABLE="${VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE:-0}"
UVM_INIT_WEIGHT_OFFLOAD_MODE="${VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE:-trace_only}"
UVM_INIT_WEIGHT_PLAN_FILE="${VLLM_UVM_INIT_WEIGHT_PLAN_FILE:-}"
UVM_INIT_WEIGHT_TARGET_ROLES="${VLLM_UVM_INIT_WEIGHT_TARGET_ROLES:-moe_gate_up,moe_down}"
UVM_INIT_WEIGHT_MAX_BYTES="${VLLM_UVM_INIT_WEIGHT_MAX_BYTES:-0}"
UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-0}"
UVM_INIT_WEIGHT_SYNC="${VLLM_UVM_INIT_WEIGHT_SYNC:-1}"
UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE="${VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE:-0}"
UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC="${VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE="${VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="${VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE:-0}"
UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE="${VLLM_UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE:-0}"
UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="${VLLM_UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS:-1}"
UVM_HOT_RESIDENCY_STAGE_DEVICE="${VLLM_UVM_HOT_RESIDENCY_STAGE_DEVICE:--1}"
UVM_INIT_WEIGHT_TRACE_FILE="${VLLM_UVM_INIT_WEIGHT_TRACE_FILE:-}"
UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN="${VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN:-0}"
UVM_MATCHED_CPU_OFFLOAD_ENABLE="${VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE:-0}"
UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE="${VLLM_UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE:-}"
UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE="${VLLM_UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE:-}"
UVM_MATCHED_CPU_OFFLOAD_MODE="${VLLM_UVM_MATCHED_CPU_OFFLOAD_MODE:-offload_cpu}"
UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="${VLLM_UVM_MATCHED_CPU_OFFLOAD_EXECUTOR:-auto}"
UVM_MATCHED_CPU_OFFLOAD_SYNC="${VLLM_UVM_MATCHED_CPU_OFFLOAD_SYNC:-1}"
UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES="${VLLM_UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES:-0}"
UVM_POOL_COORDINATOR_ENABLE="${VLLM_UVM_POOL_COORDINATOR_ENABLE:-0}"
UVM_POOL_COORDINATOR_MODE="${VLLM_UVM_POOL_COORDINATOR_MODE:-trace_only}"
UVM_POOL_COORDINATOR_TRACE_ENABLE="${VLLM_UVM_POOL_COORDINATOR_TRACE_ENABLE:-1}"
UVM_POOL_COORDINATOR_TRACE_FILE="${VLLM_UVM_POOL_COORDINATOR_TRACE_FILE:-}"
UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP="${VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP:-0}"
UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP="${VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP:-0}"
UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP="${VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP:-0}"
UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP="${VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP:-0}"
UVM_POOL_COORDINATOR_PRIORITY="${VLLM_UVM_POOL_COORDINATOR_PRIORITY:-kv,weights,scratch}"
UVM_RESIDENCY_ENABLE="${VLLM_UVM_RESIDENCY_ENABLE:-0}"
UVM_RESIDENCY_STAGE="${VLLM_UVM_RESIDENCY_STAGE:-stage_l0}"
UVM_RESIDENCY_MODE="${VLLM_UVM_RESIDENCY_MODE:-trace_only}"
UVM_RESIDENCY_TRACE_ENABLE="${VLLM_UVM_RESIDENCY_TRACE_ENABLE:-1}"
UVM_RESIDENCY_TRACE_FILE="${VLLM_UVM_RESIDENCY_TRACE_FILE:-}"
UVM_RESIDENCY_GLOBAL_GPU_BYTES="${VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES:-0}"
UVM_RESIDENCY_WEIGHT_GPU_BYTES="${VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES:-0}"
UVM_RESIDENCY_KV_GPU_BYTES="${VLLM_UVM_RESIDENCY_KV_GPU_BYTES:-0}"
UVM_RESIDENCY_SCRATCH_GPU_BYTES="${VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES:-0}"
UVM_RESIDENCY_SAFETY_MARGIN_BYTES="${VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES:-0}"
UVM_RESIDENCY_SCRATCH_RESERVED_BYTES="${VLLM_UVM_RESIDENCY_SCRATCH_RESERVED_BYTES:-0}"
UVM_RESIDENCY_SCRATCH_MARGIN_BYTES="${VLLM_UVM_RESIDENCY_SCRATCH_MARGIN_BYTES:-0}"
UVM_RESIDENCY_KV_SIZING_ENABLE="${VLLM_UVM_RESIDENCY_KV_SIZING_ENABLE:-0}"
UVM_RESIDENCY_KV_SIZING_POLICY="${VLLM_UVM_RESIDENCY_KV_SIZING_POLICY:-report_only}"
UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS="${VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS:-0}"
UVM_RESIDENCY_KV_METADATA_ENABLE="${VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE:-0}"
UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE="${VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE:-0}"
UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE="${VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE:-trace_only}"
UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE="${VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE:-0}"
UVM_RESIDENCY_KV_PREFETCH_BACK_MODE="${VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE:-trace_only}"
UVM_RESIDENCY_KV_PREFETCH_DEVICE="${VLLM_UVM_RESIDENCY_KV_PREFETCH_DEVICE:-0}"
UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP="${VLLM_UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP:-0}"
UVM_RESIDENCY_KV_POINTER_MAP_FILE="${VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE:-}"
UVM_RESIDENCY_PRIORITY="${VLLM_UVM_RESIDENCY_PRIORITY:-kv,scratch,weights}"
UVM_RESIDENCY_WEIGHT_INITIAL_STATE="${VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE:-cpu}"
UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET="${VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET:-0}"
UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD="${VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD:-1}"
UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP="${VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP:-0}"
UVM_RESIDENCY_WEIGHT_VICTIM_POLICY="${VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY:-cold_plan_lru}"
UVM_RESIDENCY_WEIGHT_VICTIM_ACTION="${VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION:-offload_cpu}"
UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE="${VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE:-0}"
UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-0}"
UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS="${VLLM_UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"
UVM_REAL_RESIDENCY_ENABLE="${VLLM_UVM_REAL_RESIDENCY_ENABLE:-0}"
UVM_REAL_RESIDENCY_TRACE_FILE="${VLLM_UVM_REAL_RESIDENCY_TRACE_FILE:-}"
UVM_REAL_RESIDENCY_DEVICE="${VLLM_UVM_REAL_RESIDENCY_DEVICE:-0}"
UVM_REAL_RESIDENCY_ON_REGISTER_EVERY="${VLLM_UVM_REAL_RESIDENCY_ON_REGISTER_EVERY:-0}"
UVM_REAL_RESIDENCY_ON_ACTION_EVERY="${VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY:-0}"
UVM_REAL_RESIDENCY_MAX_SNAPSHOTS="${VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS:-0}"
UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL="${VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL:-0}"
UVM_POOL_REGISTRY_ENABLE="${VLLM_UVM_POOL_REGISTRY_ENABLE:-0}"
UVM_SCRATCH_POOL_ENABLE="${VLLM_UVM_SCRATCH_POOL_ENABLE:-0}"
UVM_SCRATCH_POOL_BUDGET_BYTES="${VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES:-1048576}"
UVM_SCRATCH_POOL_MODE="${VLLM_UVM_SCRATCH_POOL_MODE:-trace_only}"
UVM_SCRATCH_POOL_TARGET_PHASES="${VLLM_UVM_SCRATCH_POOL_TARGET_PHASES:-enabled:attention}"
UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE="${VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE:-0}"
UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES="${VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES:-67108864}"
UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET="${VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET:-8}"
GAP_WATCH_METRICS_SUMMARY_JSON=""

AUTO_GAP_WATCH_ENABLE=0
AUTO_GAP_WATCH_PROBE_PROMPTS=1
AUTO_GAP_WATCH_TARGET_GAP=2
AUTO_GAP_WATCH_FALLBACK_TO_HOTTEST=1
AUTO_GAP_WATCH_SUMMARY_JSON=""
AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON=""
AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE=""
AUTO_GAP_WATCH_TARGET_CLASS_OVERRIDE=""

ENABLE_FAULT_ADDRESS_LOG=0
KEEP_ON_EXIT=0
NO_CLEANUP=0
NO_BENCH=0
RESET_COUNTERS=1
UVM_KO_PATH="$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
UVM_ENABLE_DEBUG_PROCFS=1
UVM_ENABLE_BUILTIN_TESTS="${VLLM_UVM_ENABLE_BUILTIN_TESTS:-0}"

KV_START=""
KV_END=""
SERVER_PID=""
TRACE_PID=""
DMESG_PID=""
RUN_STATS_LOG=""
RUN_ADDRESS_LOG=""
TRACE_READER_ERROR_LOG=""
CUSTOM_BENCH_CMD=0
AUTO_GAP_WATCH_MAIN_START_LINE=1

BENCH_CMD=()

# Stats line markers:
# - legacy English: "Replayable fault stats GPU" + key=value fields
# - newer format may include Chinese labels (e.g. "本批次总缺页实例数=")
STATS_LINE_PATTERN='Replayable fault stats GPU|batch_faults=|本批次总缺页实例数='

usage() {
  cat <<'USAGE'
Usage:
  run_kv_fault_ratio.sh [options] [-- <custom bench command...>]

Options:
  --model <name>             vLLM model (default: Qwen/Qwen3-30B-A3B-FP8)
  --port <int>               vLLM server port (default: 8000)
  --prompts <int>            Benchmark prompts (default: 50)
  --request-rate <float>     Benchmark request-rate (default: 5)
  --output-len <int>         sharegpt output len (default: 512)
  --seed <int>               Benchmark seed (default: 42)
  --dataset-path <path>      ShareGPT dataset path
  --address-log <path>       vLLM address log path
  --server-log <path>        Server log file path
  --bench-log <path>         Benchmark output log path
  --mode <dmesg|trace>       Fault stat destination mode (default: dmesg)
  --trace-log <path>         Replayable fault stats capture file
  --address-trace-log <path> Per-fault address capture file (implies --with-address-log)
  --allocator-log <path>     UVM allocator trace file for alloc/free/phase events
  --uvm-trace-min-bytes <n>  Min allocation size to record in allocator trace (default: 1048576)
  --uvm-policy-enable <0|1>  Enable allocator policy classification logs (default: 1)
  --uvm-policy-mode <mode>   UVM policy mode: trace_only|prefetch|warmup_prefetch (default: trace_only)
  --uvm-policy-warmup-prefetch-min-bytes <n>
                             Min warmup allocation size for policy prefetch (default: 1048576)
  --uvm-policy-warmup-advise-gpu <0|1>
                             Also set PreferredLocation=GPU for warmup policy prefetch (default: 0)
  --uvm-policy-runtime-enable <0|1>
                             Enable Stage M in-process policy runtime facade.
  --uvm-unknown-detail-enable <0|1>
                             Emit TRACE_UNKNOWN_DETAIL for unknown_managed allocations (default: 0)
  --uvm-unknown-detail-min-bytes <n>
                             Min unknown_managed allocation size to detail-trace (default: 0)
  --uvm-gap-watch-enable <0|1>
                             Enable allocator-side watch for a specific address range (default: 0)
  --uvm-gap-watch-name <name>
                             Human-readable label for the watched range (default: gap_watch)
  --uvm-gap-watch-start <hex>
                             Watched range start address, e.g. 0x7cd2f5180000
  --uvm-gap-watch-end <hex>
                             Watched range end address, e.g. 0x7cd2f5ffffff
  --uvm-gap-watch-all-classes <0|1>
                             If 1, log all allocations overlapping the watch range; if 0, only unknown_managed (default: 1)
  --uvm-gap-watch-min-bytes <n>
                             Min allocation size for watch-range tracing (default: 0)
  --uvm-gap-watch-target-class <name>
                             Optional target class for watch policy: any|runtime_scratch|runtime_workspace|warmup_workspace|weight_persistent|kv_persistent
  --uvm-gap-watch-policy-action <mode>
                             Watch policy action: observe|prefetch|advise_prefetch|device_direct_trace|device_direct
                             device_direct changes backend only when --uvm-device-direct-enable=1 (default: observe)
  --uvm-gap-watch-control-file <path>
                             Optional runtime-updatable control file for same-process gap watch
  --uvm-gap-watch-refresh-ms <n>
                             Control-file polling interval in allocator (default: 250)
  --uvm-device-direct-enable <0|1>
                             Stage C device-direct execution gate; default 0 keeps trace-only behavior
  --uvm-device-direct-min-bytes <n>
                             Min allocation size to mark device-direct eligible (default: 4096)
  --uvm-device-direct-max-bytes <n>
                             Max allocation size to mark device-direct eligible (default: 1048576)
  --uvm-device-direct-max-total-bytes <n>
                             Max live bytes allowed for Stage C device-direct backend;
                             0 means unlimited (default: 268435456)
  --uvm-device-direct-backend <backend>
                             Stage C/C2 device-direct backend:
                             cuda_malloc|cuda_malloc_async (default: cuda_malloc)
  --uvm-device-direct-pool-release-threshold <n>
                             Configure cuda_malloc_async default CUDA mempool
                             release threshold. Empty/unset keeps CUDA default;
                             0 is allowed and means aggressive release.
  --uvm-device-direct-target-phases <csv>
                             Comma-separated phase prefixes allowed for Stage B eligibility
                             (default: enabled:attention,enabled:moe,enabled:model_forward)
  --uvm-kv-budget-bytes <n>
                             Stage D KV cache logical budget in bytes.
                             0 means unlimited telemetry-only budget (default: 0)
  --uvm-kv-budget-mode <mode>
                             Stage D KV budget mode: trace_only|enforce.
                             enforce is allocator-side soft signaling; block-manager
                             eviction/swap is intentionally not done here (default: trace_only)
  --uvm-kv-runtime-enable <0|1>
                             Stage J runtime KV pressure policy gate. This is
                             block-manager-side, not allocator-side eviction.
  --uvm-kv-runtime-mode <mode>
                             Stage J mode: trace_only|enforce. trace_only emits
                             would-evict/would-deny records; enforce denies new
                             block admission over runtime budget.
  --uvm-kv-runtime-budget-bytes <n>
                             Stage J runtime KV block budget in bytes. Converted
                             to blocks using KV config bytes-per-block.
  --uvm-kv-runtime-budget-blocks <n>
                             Stage J runtime KV block budget override. 0 means
                             derive from bytes or disable budget pressure.
  --uvm-kv-runtime-trace-file <path>
                             Stage J JSONL trace file.
  --uvm-kv-runtime-eviction-policy <name>
                             Stage J policy name: lru_prefix_cache|scheduler_aware.
                             Both are safety-gated to free/ref_cnt==0 candidates
                             in this prototype.
  --uvm-kv-runtime-candidate-limit <n>
                             Max free-queue candidates emitted per pressure event.
  --uvm-kv-runtime-prefix-evict-enable <0|1>
                             Enable Stage J executor that clears prefix-cache
                             metadata only from free/ref_cnt==0 KV blocks.
  --uvm-kv-runtime-prefix-evict-max-blocks <n>
                             Max prefix-cache free blocks to evict per pressure
                             event. 0 means use pressure_blocks.
  --uvm-kv-active-protect-enable <0|1>
                             Enable Stage M active KV protection.
  --uvm-kv-active-protect-mode <mode>
                             Active KV protection mode:
                             trace_only|advise|prefetch|advise_prefetch.
  --uvm-kv-active-protect-device <int>
                             GPU device id for active KV prefetch/advise.
  --uvm-kv-active-protect-max-blocks <n>
                             Max active KV blocks to protect per request; 0 means all.
  --uvm-weight-budget-bytes <n>
                             Stage E model weights logical budget in bytes.
                             0 means unlimited telemetry-only budget (default: 0)
  --uvm-weight-budget-mode <mode>
                             Stage E weight budget mode: trace_only|enforce.
                             enforce is allocator-side soft signaling; real
                             weight offload/eviction is intentionally future work
                             (default: trace_only)
  --uvm-weight-map-enable <0|1>
                             Emit Stage E weight tensor semantic address map JSONL
                             sidecar (default: 1)
  --uvm-weight-map-file <path>
                             Stage E weight tensor semantic address map JSONL path
  --uvm-moe-routing-trace-enable <0|1>
                             Emit Stage E MoE expert routing aggregate JSONL
                             during benchmark/runtime phases (default: 0)
  --uvm-moe-routing-trace-file <path>
                             Stage E MoE routing aggregate JSONL path
  --uvm-weight-prefetch-enable <0|1>
                             Stage I expert weight prefetch gate. Default 0.
  --uvm-weight-prefetch-mode <mode>
                             Stage I mode: trace_only|prefetch. prefetch issues
                             cudaMemPrefetchAsync on active expert weight slices.
  --uvm-weight-prefetch-trace-enable <0|1>
                             Stage I prefetch JSONL trace gate.
  --uvm-weight-prefetch-trace-file <path>
                             Stage I expert prefetch JSONL trace path
  --uvm-weight-prefetch-max-bytes-per-step <n>
                             Max expert weight bytes prefetched per MoE layer call
  --uvm-weight-prefetch-max-experts-per-layer <n>
                             Max active experts considered per MoE layer call
  --uvm-weight-prefetch-skip-resident <0|1>
                             Skip coordinator/driver calls for already-GPU regions
  --uvm-weight-prefetch-expert-cache-enable <0|1>
                             Skip all roles for experts cached as fully GPU-resident
  --uvm-weight-prefetch-region-cooldown-steps <n>
                             Skip repeated prefetch of same region for N MoE steps
  --uvm-weight-router-topk-every-n-steps <n>
                             Run router_topk prefetch/offload only every n MoE calls
  --uvm-weight-prefetch-target-roles <csv>
                             Expert weight roles to prefetch (default: moe_gate_up,moe_down)
  --uvm-weight-prefetch-device <n>
                             Target CUDA device for prefetch. -1 means current device.
  --uvm-weight-prefetch-plan-file <path>
                             Optional Stage H plan JSON. When present, Stage I
                             prefetch only acts on hot experts in prefetch_plan.
  --uvm-weight-prefetch-require-plan <0|1>
                             Require a Stage H prefetch plan before issuing
                             active-expert prefetch actions (default: 0).
  --uvm-weight-prefetch-source <source>
                             router_topk|plan_intersection|oracle_replay.
  --uvm-oracle-expert-trace-file <path>
                             Routing trace JSONL used when source=oracle_replay.
  --uvm-oracle-prefetch-lookahead-steps <n>
                             Per-layer replay lookahead for oracle_replay.
  --uvm-weight-executor <mode>
                             Stage M/F weight action executor:
                             cpp_batch|driver_batch|auto (default: cpp_batch).
  --uvm-weight-offload-enable <0|1>
                             Stage I optional cold expert offload/advise gate.
  --uvm-weight-offload-mode <mode>
                             Stage I cold expert mode:
                             trace_only|advise_cpu|prefetch_cpu.
  --uvm-weight-offload-plan-file <path>
                             Stage H plan JSON containing offload_plan.
  --uvm-weight-offload-max-bytes-per-step <n>
                             Max cold expert weight bytes offloaded per MoE layer call.
  --uvm-weight-offload-max-experts-per-layer <n>
                             Max cold experts processed per MoE layer call.
  --uvm-weight-offload-target-roles <csv>
                             Expert weight roles eligible for cold offload.
  --uvm-pool-coordinator-enable <0|1>
                             Stage K global action coordinator gate.
                             Coordinates high-level safe-point actions only.
  --uvm-pool-coordinator-mode <mode>
                             Stage K mode: trace_only|enforce. trace_only
                             records would-deny but does not skip actions.
  --uvm-pool-coordinator-trace-enable <0|1>
                             Stage K coordinator JSONL trace gate.
  --uvm-pool-coordinator-trace-file <path>
                             Stage K unified coordinator JSONL trace path.
  --uvm-pool-coordinator-global-bytes-per-step <n>
                             Global action budget per coordinator scope.
                             0 means unlimited.
  --uvm-pool-coordinator-weight-bytes-per-step <n>
                             Stage I weights action budget per scope.
  --uvm-pool-coordinator-kv-bytes-per-step <n>
                             Stage J KV action budget per scope.
  --uvm-pool-coordinator-scratch-bytes-per-step <n>
                             Stage G scratch action budget placeholder.
  --uvm-pool-coordinator-priority <csv>
                             Pool priority label for reports
                             (default: kv,weights,scratch)
  --uvm-residency-enable <0|1>
                             Stage L0 estimated GPU residency ledger gate.
  --uvm-residency-stage <label>
                             Stage label written into residency JSONL records.
                             (default: stage_l0)
  --uvm-residency-mode <mode>
                             Stage L0 mode: trace_only|enforce. L0 only
                             records accounting decisions; no real reclaim.
  --uvm-residency-trace-file <path>
                             Stage L0 residency JSONL trace path.
  --uvm-residency-global-gpu-bytes <n>
                             Global estimated GPU residency budget.
                             0 means unlimited.
  --uvm-residency-weight-gpu-bytes <n>
                             Weights estimated GPU residency budget.
  --uvm-residency-kv-gpu-bytes <n>
                             KV estimated GPU residency budget.
  --uvm-residency-scratch-gpu-bytes <n>
                             Scratch estimated GPU residency budget.
  --uvm-residency-safety-margin-bytes <n>
                             Safety margin subtracted from global budget.
  --uvm-residency-scratch-reserved-bytes <n>
                             Stage L3 scratch reserve reported alongside
                             Stage G observed scratch peak.
  --uvm-residency-scratch-margin-bytes <n>
                             Stage L3 extra scratch margin for reports.
  --uvm-residency-kv-sizing-enable <0|1>
                             Stage L4 KV cache sizing report gate.
  --uvm-residency-kv-sizing-policy <name>
                             Stage L4 KV sizing policy:
                             report_only|clamp|override.
  --uvm-residency-kv-sizing-max-blocks <n>
                             Stage L4 optional max block guard for override.
                             0 means no extra guard.
  --uvm-residency-kv-metadata-enable <0|1>
                             Stage L5 KV block metadata-only residency gate.
  --uvm-residency-kv-data-offload-enable <0|1>
                             Stage L6 safe KV block data offload gate.
  --uvm-residency-kv-data-offload-mode <mode>
                             Stage L6 offload mode: trace_only|prefetch_cpu.
  --uvm-residency-kv-prefetch-back-enable <0|1>
                             Stage L6 prefetch-back-before-reuse gate.
  --uvm-residency-kv-prefetch-back-mode <mode>
                             Stage L6 prefetch-back mode: trace_only|prefetch.
  --uvm-residency-kv-prefetch-device <n>
                             Stage L6 target CUDA device for prefetch-back.
  --uvm-residency-kv-max-offload-blocks-per-step <n>
                             Stage L6 max safe KV offload blocks per pressure
                             event. 0 derives from runtime pressure.
  --uvm-residency-kv-pointer-map-file <path>
                             Stage L6 raw KV block pointer map sidecar.
  --uvm-residency-priority <csv>
                             Stage L priority label for reports
                             (default: kv,scratch,weights)
  --uvm-residency-weight-initial-state <state>
                             Initial expert weight residency state:
                             unknown|cpu|gpu (default: cpu)
  --uvm-residency-weight-enforce-hot-set <0|1>
                             Stage L2 weights hot/cold residency enforce gate.
  --uvm-residency-weight-sync-offload <0|1>
                             If 1, L2 victim offload releases estimated bytes
                             immediately after action success.
  --uvm-residency-weight-max-victims-per-step <n>
                             Max L2 weight victims per hot prefetch request.
                             0 means unlimited.
  --uvm-residency-weight-victim-policy <name>
                             L2 weight victim policy label.
  --uvm-residency-weight-victim-action <mode>
                             L2 victim action: advise_cpu|prefetch_cpu|offload_cpu.
  --uvm-residency-weight-evict-protect-recent-steps <n>
                             Protect recently active weight regions from eviction.
  --uvm-matched-cpu-offload-enable <0|1>
                             Apply UVM CPU placement to the exact parameter set
                             recorded by vLLM native CPU offload manifest.
  --uvm-matched-cpu-offload-manifest-file <path>
                             JSONL manifest generated by VLLM_CPU_OFFLOAD_MANIFEST_FILE.
  --uvm-matched-cpu-offload-trace-file <path>
                             JSONL trace for matched UVM CPU placement.
  --uvm-matched-cpu-offload-mode <mode>
                             trace_only|advise_cpu|prefetch_cpu|offload_cpu|prefetch_gpu.
  --uvm-matched-cpu-offload-executor <mode>
                             cpp_batch|driver_batch|auto.
  --uvm-matched-cpu-offload-sync <0|1>
                             Synchronize after matched UVM placement.
  --uvm-matched-cpu-offload-max-bytes <n>
                             Optional cap; 0 means no cap beyond manifest.
  --uvm-pool-registry-enable <0|1>
                             Stage F unified pool registry telemetry.
                             Records kv_cache/weights/runtime_scratch objects
                             without changing allocation policy (default: 0)
  --uvm-scratch-pool-enable <0|1>
                             Stage G runtime scratch pool admission control.
                             Budgeted device-direct fast path with managed
                             fallback, no eviction (default: 0)
  --uvm-scratch-pool-budget-bytes <n>
                             Stage G scratch pool device-direct live byte budget.
                             0 means unlimited (default: 1048576)
  --uvm-scratch-pool-mode <mode>
                             Stage G scratch pool mode: trace_only|enforce.
                             trace_only records over-budget but allows fast path;
                             enforce falls back managed when over budget
                             (default: trace_only)
  --uvm-scratch-pool-target-phases <csv>
                             Comma-separated phase prefixes allowed for Stage G
                             scratch pool admission (default: enabled:attention)
  --uvm-scratch-pool-reuse-cache-enable <0|1>
                             Phase 7 exact-size reuse cache for scratch
                             cuda_malloc_async device-direct blocks (default: 0)
  --uvm-scratch-pool-reuse-cache-max-bytes <n>
                             Max bytes retained by the scratch reuse cache
                             (default: 67108864)
  --uvm-scratch-pool-reuse-cache-max-entries-per-bucket <n>
                             Per-size-bucket retained entry cap (default: 8)
  --gap-watch-metrics-summary-json <path>
                             Optional post-run JSON summary proving whether gap-watch policy hit and succeeded
  --auto-gap-watch-enable <0|1>
                             Same-process probe -> discover -> main workflow for dynamic gap watch (default: 0)
  --auto-gap-watch-probe-prompts <n>
                             Probe benchmark prompts before gap discovery (default: 1)
  --auto-gap-watch-target-gap <n>
                             Preferred gap index to watch after probe (default: 2)
  --auto-gap-watch-fallback-to-hottest <0|1>
                             If target gap has no faults, fall back to the hottest unknown gap (default: 1)
  --auto-gap-watch-summary-json <path>
                             Optional summary JSON emitted by the auto gap discovery step
  --auto-gap-watch-post-main-summary-json <path>
                             Optional summary JSON emitted after main phase for gap consistency comparison
  --auto-gap-watch-policy-action-override <mode>
                             Override discovered policy action for A/B runs:
                             observe|prefetch|advise_prefetch|device_direct_trace|device_direct
  --auto-gap-watch-target-class-override <name>
                             Override discovered target class for A/B runs
  --startup-timeout <sec>    Server/KV wait timeout (default: 600)
  --check-interval <sec>     Poll interval (default: 2)
  --server-max-num-seqs <n>  vLLM server --max-num-seqs (default: 8)
  --server-gpu-memory-utilization <f>
                             vLLM server --gpu-memory-utilization (default: 0.85)
  --max-model-len <n>        vLLM server --max-model-len
  --with-address-log         Also enable per-fault address logs
  --no-cleanup               Skip workloads/cleanup_gpu.py before run
  --no-bench                 Configure params only, do not run benchmark
  --reset-counters           Reset total_* counters by reloading nvidia_uvm (default: on)
  --no-reset-counters        Do not reload nvidia_uvm before run
  --uvm-ko-path <path>       nvidia-uvm.ko path for --reset-counters
  --uvm-enable-debug-procfs <0|1>  Set at insmod time (default: 1)
  --uvm-enable-builtin-tests <0|1>
                             Set uvm_enable_builtin_tests at insmod time.
                             Required for Phase F test ioctl driver executor.
  --uvm-phase-f-route-log-delay-ms <n>
                             Set uvm_phase_f_route_log_delay_ms at insmod time (default: 0)
  --keep                     Keep UVM params on exit (no rollback)
  -h, --help                 Show help

Custom bench command:
  If provided after '--', it runs against the same server process after KV params are configured.

Examples:
  # One-click same-process run (recommended)
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --trace-log /tmp/uvm_kv_fault_trace.log

  # Split stats and per-fault address logs into two files
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs.log

  # Split stats/address logs and also capture allocator trace
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs.log \
    --allocator-log /tmp/vllm_uvm_allocator_trace.log \
    --uvm-trace-min-bytes 1048576

  # Trace unknown_managed and directly watch gap#2
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs.log \
    --allocator-log /tmp/vllm_uvm_allocator_trace.log \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --uvm-gap-watch-enable 1 \
    --uvm-gap-watch-name gap2 \
    --uvm-gap-watch-start 0x7cd2f5180000 \
    --uvm-gap-watch-end 0x7cd2f5ffffff

  # Same-process auto workflow: probe once, discover current gap#2, then continue with watch enabled
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace --with-address-log \
    --trace-log /tmp/uvm_kv_fault_stats_auto.log \
    --address-trace-log /tmp/uvm_kv_fault_addrs_auto.log \
    --allocator-log /tmp/vllm_uvm_allocator_trace_auto.log \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --auto-gap-watch-enable 1 \
    --auto-gap-watch-probe-prompts 1 \
    --prompts 20

  # Custom bench command
  ./workloads/vllm/run_kv_fault_ratio.sh --mode trace -- \
    uv run vllm bench serve --model Qwen/Qwen3-30B-A3B-FP8 \
      --dataset-name sharegpt --dataset-path workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
      --num-prompts 100 --sharegpt-output-len 512 --seed 42 --request-rate 5 --port 8000
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_file() {
  local path="$1"
  [ -e "$path" ] || die "Missing required path: $path"
}

need_writable_param() {
  local key="$1"
  sudo test -w "$PARAM_DIR/$key" || die "Parameter is not writable: $PARAM_DIR/$key"
}

write_param() {
  local key="$1"
  local value="$2"
  echo "$value" | sudo tee "$PARAM_DIR/$key" >/dev/null
}

is_hex_u64() {
  [[ "$1" =~ ^0x[0-9a-fA-F]+$ ]]
}

extract_kv_range_from_log() {
  local line
  line="$(grep -E 'kv_cache:(contiguous_range|span_range),all_layers,0x[0-9a-fA-F]+,0x[0-9a-fA-F]+' "$ADDRESS_LOG" | tail -n 1 || true)"
  [ -n "$line" ] || return 1

  local kind name start end
  IFS=',' read -r kind name start end _ <<<"$line"
  is_hex_u64 "$start" || return 1
  is_hex_u64 "$end" || return 1

  KV_START="$start"
  KV_END="$end"
  return 0
}

# 启动 vLLM 服务器进程
start_server() {
    local allocator_env=""
    local policy_env=""

    # 1. 环境准备：创建日志文件夹
    # dirname 获取文件的父目录，mkdir -p 确保路径存在且不重复报错
    mkdir -p "$(dirname "$SERVER_LOG")"
    mkdir -p "$(dirname "$ADDRESS_LOG")"
    if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
      mkdir -p "$(dirname "$ALLOCATOR_TRACE_LOG")"
    fi

    # 2. 初始化日志文件
    # ': >' 是 Bash 的一种高效清空文件内容的方法（不启动外部进程）
    # 确保每次运行压测时，之前的旧日志不会干扰本次统计
    : > "$SERVER_LOG"
    : > "$ADDRESS_LOG"
    if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
      : > "$ALLOCATOR_TRACE_LOG"
      allocator_env="VLLM_UVM_LOG_FILE='$ALLOCATOR_TRACE_LOG' VLLM_UVM_TRACE_MIN_BYTES='$UVM_TRACE_MIN_BYTES'"
    fi
    local uvm_stage_e_log_dir="$SCRIPT_DIR"
    if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
      uvm_stage_e_log_dir="$(dirname "$ALLOCATOR_TRACE_LOG")"
    fi
    if [ -z "$UVM_WEIGHT_MAP_FILE" ]; then
      UVM_WEIGHT_MAP_FILE="$uvm_stage_e_log_dir/vllm_uvm_weight_regions.jsonl"
    fi
    if [ -z "$UVM_MOE_ROUTING_TRACE_FILE" ]; then
      UVM_MOE_ROUTING_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_moe_routing_trace.jsonl"
    fi
    if [ -z "$UVM_WEIGHT_PREFETCH_TRACE_FILE" ]; then
      UVM_WEIGHT_PREFETCH_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_weight_prefetch_stage_i.jsonl"
    fi
    if [ -z "$UVM_INIT_KV_TRACE_FILE" ]; then
      UVM_INIT_KV_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_init_kv_sizing_stage_m.jsonl"
    fi
    if [ -z "$UVM_INIT_WEIGHT_PLAN_FILE" ]; then
      UVM_INIT_WEIGHT_PLAN_FILE="$UVM_WEIGHT_OFFLOAD_PLAN_FILE"
    fi
    if [ -z "$UVM_INIT_WEIGHT_TRACE_FILE" ]; then
      UVM_INIT_WEIGHT_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_init_weight_offload_stage_m.jsonl"
    fi
    if [ -z "$UVM_KV_RUNTIME_TRACE_FILE" ]; then
      UVM_KV_RUNTIME_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_kv_runtime_stage_j.jsonl"
    fi
    if [ -z "$UVM_POOL_COORDINATOR_TRACE_FILE" ]; then
      UVM_POOL_COORDINATOR_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_pool_coordinator_stage_k.jsonl"
    fi
    if [ -z "$UVM_RESIDENCY_TRACE_FILE" ]; then
      UVM_RESIDENCY_TRACE_FILE="$uvm_stage_e_log_dir/vllm_uvm_pool_residency_stage_l.jsonl"
    fi
    policy_env="VLLM_UVM_POLICY_ENABLE='$UVM_POLICY_ENABLE' \
                VLLM_UVM_POLICY_MODE='$UVM_POLICY_MODE' \
                VLLM_UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES='$UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES' \
                VLLM_UVM_POLICY_WARMUP_ADVISE_GPU='$UVM_POLICY_WARMUP_ADVISE_GPU' \
                VLLM_UVM_POLICY_RUNTIME_ENABLE='$UVM_POLICY_RUNTIME_ENABLE' \
                VLLM_UVM_UNKNOWN_DETAIL_ENABLE='$UVM_UNKNOWN_DETAIL_ENABLE' \
                VLLM_UVM_UNKNOWN_DETAIL_MIN_BYTES='$UVM_UNKNOWN_DETAIL_MIN_BYTES' \
                VLLM_UVM_GAP_WATCH_ENABLE='$UVM_GAP_WATCH_ENABLE' \
                VLLM_UVM_GAP_WATCH_NAME='$UVM_GAP_WATCH_NAME' \
                VLLM_UVM_GAP_WATCH_START='$UVM_GAP_WATCH_START' \
                VLLM_UVM_GAP_WATCH_END='$UVM_GAP_WATCH_END' \
                VLLM_UVM_GAP_WATCH_ALL_CLASSES='$UVM_GAP_WATCH_ALL_CLASSES' \
                VLLM_UVM_GAP_WATCH_MIN_BYTES='$UVM_GAP_WATCH_MIN_BYTES' \
                VLLM_UVM_GAP_WATCH_TARGET_CLASS='$UVM_GAP_WATCH_TARGET_CLASS' \
                VLLM_UVM_GAP_WATCH_POLICY_ACTION='$UVM_GAP_WATCH_POLICY_ACTION' \
                VLLM_UVM_GAP_WATCH_CONTROL_FILE='$UVM_GAP_WATCH_CONTROL_FILE' \
                VLLM_UVM_GAP_WATCH_REFRESH_MS='$UVM_GAP_WATCH_REFRESH_MS' \
                VLLM_UVM_DEVICE_DIRECT_ENABLE='$UVM_DEVICE_DIRECT_ENABLE' \
                VLLM_UVM_DEVICE_DIRECT_MIN_BYTES='$UVM_DEVICE_DIRECT_MIN_BYTES' \
                VLLM_UVM_DEVICE_DIRECT_MAX_BYTES='$UVM_DEVICE_DIRECT_MAX_BYTES' \
                VLLM_UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES='$UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES' \
                VLLM_UVM_DEVICE_DIRECT_BACKEND='$UVM_DEVICE_DIRECT_BACKEND' \
                VLLM_UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD='$UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD' \
                VLLM_UVM_DEVICE_DIRECT_TARGET_PHASES='$UVM_DEVICE_DIRECT_TARGET_PHASES' \
                VLLM_UVM_KV_BUDGET_BYTES='$UVM_KV_BUDGET_BYTES' \
                VLLM_UVM_KV_BUDGET_MODE='$UVM_KV_BUDGET_MODE' \
                VLLM_UVM_KV_RUNTIME_ENABLE='$UVM_KV_RUNTIME_ENABLE' \
                VLLM_UVM_KV_RUNTIME_MODE='$UVM_KV_RUNTIME_MODE' \
                VLLM_UVM_KV_RUNTIME_BUDGET_BYTES='$UVM_KV_RUNTIME_BUDGET_BYTES' \
                VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS='$UVM_KV_RUNTIME_BUDGET_BLOCKS' \
                VLLM_UVM_KV_RUNTIME_TRACE_FILE='$UVM_KV_RUNTIME_TRACE_FILE' \
                VLLM_UVM_KV_RUNTIME_EVICTION_POLICY='$UVM_KV_RUNTIME_EVICTION_POLICY' \
                VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT='$UVM_KV_RUNTIME_CANDIDATE_LIMIT' \
                VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE='$UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE' \
                VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS='$UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS' \
                VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE='$UVM_KV_ACTIVE_PROTECT_ENABLE' \
                VLLM_UVM_KV_ACTIVE_PROTECT_MODE='$UVM_KV_ACTIVE_PROTECT_MODE' \
                VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE='$UVM_KV_ACTIVE_PROTECT_DEVICE' \
                VLLM_UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS='$UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS' \
                VLLM_UVM_WEIGHT_BUDGET_BYTES='$UVM_WEIGHT_BUDGET_BYTES' \
                VLLM_UVM_WEIGHT_BUDGET_MODE='$UVM_WEIGHT_BUDGET_MODE' \
                VLLM_UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE='$UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE' \
                VLLM_UVM_WEIGHT_MAP_ENABLE='$UVM_WEIGHT_MAP_ENABLE' \
                VLLM_UVM_WEIGHT_MAP_FILE='$UVM_WEIGHT_MAP_FILE' \
                VLLM_UVM_MOE_ROUTING_TRACE_ENABLE='$UVM_MOE_ROUTING_TRACE_ENABLE' \
                VLLM_UVM_MOE_ROUTING_TRACE_FILE='$UVM_MOE_ROUTING_TRACE_FILE' \
                VLLM_UVM_WEIGHT_PREFETCH_ENABLE='$UVM_WEIGHT_PREFETCH_ENABLE' \
                VLLM_UVM_WEIGHT_PREFETCH_MODE='$UVM_WEIGHT_PREFETCH_MODE' \
                VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE='$UVM_WEIGHT_PREFETCH_TRACE_ENABLE' \
                VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE='$UVM_WEIGHT_PREFETCH_TRACE_FILE' \
                VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP='$UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP' \
                VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER='$UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER' \
                VLLM_UVM_WEIGHT_PREFETCH_SKIP_RESIDENT='$UVM_WEIGHT_PREFETCH_SKIP_RESIDENT' \
                VLLM_UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE='$UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE' \
                VLLM_UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS='$UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS' \
                VLLM_UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS='$UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS' \
                VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES='$UVM_WEIGHT_PREFETCH_TARGET_ROLES' \
                VLLM_UVM_WEIGHT_PREFETCH_DEVICE='$UVM_WEIGHT_PREFETCH_DEVICE' \
                VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE='$UVM_WEIGHT_PREFETCH_PLAN_FILE' \
                VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN='$UVM_WEIGHT_PREFETCH_REQUIRE_PLAN' \
                VLLM_UVM_WEIGHT_PREFETCH_SOURCE='$UVM_WEIGHT_PREFETCH_SOURCE' \
                VLLM_UVM_ORACLE_EXPERT_TRACE_FILE='$UVM_ORACLE_EXPERT_TRACE_FILE' \
                VLLM_UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS='$UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS' \
                VLLM_UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE='$UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE' \
                VLLM_UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE='$UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE' \
                VLLM_UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP='$UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP' \
                VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE='$UVM_WEIGHT_BATCH_EXECUTOR_ENABLE' \
                VLLM_UVM_WEIGHT_EXECUTOR='$UVM_WEIGHT_EXECUTOR' \
                VLLM_UVM_WEIGHT_OFFLOAD_ENABLE='$UVM_WEIGHT_OFFLOAD_ENABLE' \
                VLLM_UVM_WEIGHT_OFFLOAD_MODE='$UVM_WEIGHT_OFFLOAD_MODE' \
                VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE='$UVM_WEIGHT_OFFLOAD_PLAN_FILE' \
                VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP='$UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP' \
                VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER='$UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER' \
                VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES='$UVM_WEIGHT_OFFLOAD_TARGET_ROLES' \
                VLLM_UVM_INIT_KV_SIZING_ENABLE='$UVM_INIT_KV_SIZING_ENABLE' \
                VLLM_UVM_INIT_KV_SIZING_POLICY='$UVM_INIT_KV_SIZING_POLICY' \
                VLLM_UVM_INIT_KV_MAX_BYTES='$UVM_INIT_KV_MAX_BYTES' \
                VLLM_UVM_INIT_KV_MAX_BLOCKS='$UVM_INIT_KV_MAX_BLOCKS' \
                VLLM_UVM_INIT_KV_MIN_BLOCKS='$UVM_INIT_KV_MIN_BLOCKS' \
                VLLM_UVM_INIT_KV_TRACE_FILE='$UVM_INIT_KV_TRACE_FILE' \
                VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE='$UVM_INIT_WEIGHT_OFFLOAD_ENABLE' \
                VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE='$UVM_INIT_WEIGHT_OFFLOAD_MODE' \
                VLLM_UVM_INIT_WEIGHT_PLAN_FILE='$UVM_INIT_WEIGHT_PLAN_FILE' \
                VLLM_UVM_INIT_WEIGHT_TARGET_ROLES='$UVM_INIT_WEIGHT_TARGET_ROLES' \
                VLLM_UVM_INIT_WEIGHT_MAX_BYTES='$UVM_INIT_WEIGHT_MAX_BYTES' \
                VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER='$UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER' \
                VLLM_UVM_INIT_WEIGHT_SYNC='$UVM_INIT_WEIGHT_SYNC' \
                VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE='$UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE' \
                VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC='$UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC' \
                VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES='$UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES' \
                VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE='$UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE' \
                VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE='$UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE' \
                VLLM_UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE='$UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE' \
                VLLM_UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS='$UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS' \
                VLLM_UVM_HOT_RESIDENCY_STAGE_DEVICE='$UVM_HOT_RESIDENCY_STAGE_DEVICE' \
                VLLM_UVM_INIT_WEIGHT_TRACE_FILE='$UVM_INIT_WEIGHT_TRACE_FILE' \
                VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN='$UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE='$UVM_MATCHED_CPU_OFFLOAD_ENABLE' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE='$UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE='$UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_MODE='$UVM_MATCHED_CPU_OFFLOAD_MODE' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_EXECUTOR='$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_SYNC='$UVM_MATCHED_CPU_OFFLOAD_SYNC' \
                VLLM_UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES='$UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES' \
                VLLM_UVM_POOL_COORDINATOR_ENABLE='$UVM_POOL_COORDINATOR_ENABLE' \
                VLLM_UVM_POOL_COORDINATOR_MODE='$UVM_POOL_COORDINATOR_MODE' \
                VLLM_UVM_POOL_COORDINATOR_TRACE_ENABLE='$UVM_POOL_COORDINATOR_TRACE_ENABLE' \
                VLLM_UVM_POOL_COORDINATOR_TRACE_FILE='$UVM_POOL_COORDINATOR_TRACE_FILE' \
                VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP='$UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP' \
                VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP='$UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP' \
                VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP='$UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP' \
                VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP='$UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP' \
                VLLM_UVM_POOL_COORDINATOR_PRIORITY='$UVM_POOL_COORDINATOR_PRIORITY' \
                VLLM_UVM_RESIDENCY_ENABLE='$UVM_RESIDENCY_ENABLE' \
                VLLM_UVM_RESIDENCY_STAGE='$UVM_RESIDENCY_STAGE' \
                VLLM_UVM_RESIDENCY_MODE='$UVM_RESIDENCY_MODE' \
                VLLM_UVM_RESIDENCY_TRACE_ENABLE='$UVM_RESIDENCY_TRACE_ENABLE' \
                VLLM_UVM_RESIDENCY_TRACE_FILE='$UVM_RESIDENCY_TRACE_FILE' \
                VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES='$UVM_RESIDENCY_GLOBAL_GPU_BYTES' \
                VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES='$UVM_RESIDENCY_WEIGHT_GPU_BYTES' \
                VLLM_UVM_RESIDENCY_KV_GPU_BYTES='$UVM_RESIDENCY_KV_GPU_BYTES' \
                VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES='$UVM_RESIDENCY_SCRATCH_GPU_BYTES' \
                VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES='$UVM_RESIDENCY_SAFETY_MARGIN_BYTES' \
                VLLM_UVM_RESIDENCY_SCRATCH_RESERVED_BYTES='$UVM_RESIDENCY_SCRATCH_RESERVED_BYTES' \
                VLLM_UVM_RESIDENCY_SCRATCH_MARGIN_BYTES='$UVM_RESIDENCY_SCRATCH_MARGIN_BYTES' \
                VLLM_UVM_RESIDENCY_KV_SIZING_ENABLE='$UVM_RESIDENCY_KV_SIZING_ENABLE' \
                VLLM_UVM_RESIDENCY_KV_SIZING_POLICY='$UVM_RESIDENCY_KV_SIZING_POLICY' \
                VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS='$UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS' \
                VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE='$UVM_RESIDENCY_KV_METADATA_ENABLE' \
                VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE='$UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE' \
                VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE='$UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE' \
                VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE='$UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE' \
                VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE='$UVM_RESIDENCY_KV_PREFETCH_BACK_MODE' \
                VLLM_UVM_RESIDENCY_KV_PREFETCH_DEVICE='$UVM_RESIDENCY_KV_PREFETCH_DEVICE' \
                VLLM_UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP='$UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP' \
                VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE='$UVM_RESIDENCY_KV_POINTER_MAP_FILE' \
                VLLM_UVM_RESIDENCY_PRIORITY='$UVM_RESIDENCY_PRIORITY' \
                VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE='$UVM_RESIDENCY_WEIGHT_INITIAL_STATE' \
                VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET='$UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET' \
                VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD='$UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD' \
                VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP='$UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP' \
                VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY='$UVM_RESIDENCY_WEIGHT_VICTIM_POLICY' \
                VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION='$UVM_RESIDENCY_WEIGHT_VICTIM_ACTION' \
                VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE='$UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE' \
                VLLM_UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER='$UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER' \
                VLLM_UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS='$UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS' \
                VLLM_UVM_REAL_RESIDENCY_ENABLE='$UVM_REAL_RESIDENCY_ENABLE' \
                VLLM_UVM_REAL_RESIDENCY_TRACE_FILE='$UVM_REAL_RESIDENCY_TRACE_FILE' \
                VLLM_UVM_REAL_RESIDENCY_DEVICE='$UVM_REAL_RESIDENCY_DEVICE' \
                VLLM_UVM_REAL_RESIDENCY_ON_REGISTER_EVERY='$UVM_REAL_RESIDENCY_ON_REGISTER_EVERY' \
                VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY='$UVM_REAL_RESIDENCY_ON_ACTION_EVERY' \
                VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS='$UVM_REAL_RESIDENCY_MAX_SNAPSHOTS' \
                VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL='$UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL' \
                VLLM_UVM_POOL_REGISTRY_ENABLE='$UVM_POOL_REGISTRY_ENABLE' \
                VLLM_UVM_SCRATCH_POOL_ENABLE='$UVM_SCRATCH_POOL_ENABLE' \
                VLLM_UVM_SCRATCH_POOL_BUDGET_BYTES='$UVM_SCRATCH_POOL_BUDGET_BYTES' \
                VLLM_UVM_SCRATCH_POOL_MODE='$UVM_SCRATCH_POOL_MODE' \
                VLLM_UVM_SCRATCH_POOL_TARGET_PHASES='$UVM_SCRATCH_POOL_TARGET_PHASES' \
                VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE='$UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE' \
                VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES='$UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES' \
                VLLM_UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET='$UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET'"

    # 3. 构造服务器启动指令
    local max_model_len_arg=""
    if [ -n "$SERVER_MAX_MODEL_LEN" ]; then
      max_model_len_arg="--max-model-len '$SERVER_MAX_MODEL_LEN'"
    fi
    local server_cmd
    # 指令解析：
    # - cd '$SCRIPT_DIR': 切换到脚本目录，确保相对路径正确
    # - VLLM_USE_UVM=1: 强制 vLLM 使用统一虚拟内存 (UVM) 模式
    # - VLLM_UVM_ADDRESS_LOG_ENABLE=1: 开启 vLLM 的内存地址探测插件
    # - VLLM_UVM_ADDRESS_LOG_FILE: 指定 vLLM 将内存布局信息写到哪个文件
    # - uv run vllm serve: 使用 uv 工具启动 vLLM 推理引擎
    # - --enforce-eager: 强制使用 Eager 模式（通常为了避免 CUDA Graph 干扰显存地址分析）
    # - --max-num-seqs: 限制 warmup/运行时最大并发序列数，避免启动阶段 dummy sampler OOM
    # - --gpu-memory-utilization: 给 sampler/logits/runtime workspace 留出显存余量
    # - > '$SERVER_LOG' 2>&1: 将标准输出和错误信息全部重定向到服务器日志
    server_cmd="cd '$SCRIPT_DIR' && \
                VLLM_USE_UVM=1 \
                VLLM_UVM_ADDRESS_LOG_ENABLE=1 \
                VLLM_UVM_ADDRESS_LOG_FILE='$ADDRESS_LOG' \
                $allocator_env \
                $policy_env \
                uv run vllm serve '$MODEL' \
                --enforce-eager \
                --max-num-seqs '$SERVER_MAX_NUM_SEQS' \
                $max_model_len_arg \
                --gpu-memory-utilization '$SERVER_GPU_MEMORY_UTILIZATION' \
                --port '$PORT' > '$SERVER_LOG' 2>&1"

    # 4. 后台执行并脱离终端控制
    # - setsid: 创建一个新的会话（Session）。
    #   这样做的好处是即便当前脚本所在的终端关闭，vLLM 进程组也不会被系统杀死。
    # - bash -lc: 使用 login shell 模式执行命令，确保加载了用户环境（如 PATH, CUDA 路径等）。
    # - &: 将整个进程放入后台运行。
    setsid bash -lc "$server_cmd" &
    
    # 5. 获取并检查进程状态
    # $! 是 Bash 中最近一个后台运行进程的 PID (进程 ID)
    SERVER_PID=$!
    
    # 给服务器一点点启动时间（实例化 Python 解释器的时间）
    sleep 1
    
    # 6. 进程存活检查
    # kill -0 并不发送杀死信号，而是检查该 PID 是否存在且用户是否有权限操作。
    # 如果进程启动失败（例如显存不足或端口冲突），则调用 die 函数终止整个脚本。
    kill -0 "$SERVER_PID" 2>/dev/null || die "Failed to start vLLM server process"
}

stop_server_if_needed() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    # Kill whole process group to avoid child leftovers
    kill -TERM -- "-$SERVER_PID" >/dev/null 2>&1 || true
    sleep 2
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -- "-$SERVER_PID" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

wait_for_server_and_kv_range() {
  local start_ts now elapsed
  start_ts="$(date +%s)"

  while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "===== server log tail =====" >&2
      tail -n 80 "$SERVER_LOG" >&2 || true
      die "Server exited before ready"
    fi

    if extract_kv_range_from_log; then
      if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        return 0
      fi
    fi

    now="$(date +%s)"
    elapsed=$(( now - start_ts ))
    if [ "$elapsed" -ge "$STARTUP_TIMEOUT" ]; then
      echo "===== server log tail =====" >&2
      tail -n 120 "$SERVER_LOG" >&2 || true
      echo "===== address log tail =====" >&2
      tail -n 60 "$ADDRESS_LOG" >&2 || true
      die "Timeout waiting for server readiness and KV range (>${STARTUP_TIMEOUT}s)"
    fi

    sleep "$CHECK_INTERVAL"
  done
}

start_trace_capture_if_needed() {
  [ "$MODE" = "trace" ] || return 0

  resolve_trace_dir
  need_file "$TRACE_DIR/trace_pipe"

  mkdir -p "$(dirname "$TRACE_LOG")"
  RUN_STATS_LOG="$TRACE_LOG"
  : > "$RUN_STATS_LOG"

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    mkdir -p "$(dirname "$ADDRESS_TRACE_LOG")"
    RUN_ADDRESS_LOG="$ADDRESS_TRACE_LOG"
    : > "$RUN_ADDRESS_LOG"
  else
    RUN_ADDRESS_LOG=""
  fi
  TRACE_READER_ERROR_LOG="/tmp/uvm_trace_reader_$$.log"
  : > "$TRACE_READER_ERROR_LOG"

  sudo sh -c ": > '$TRACE_DIR/trace'"
  echo 1 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null

  if sudo fuser "$TRACE_DIR/trace_pipe" >/dev/null 2>&1; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
    die "trace_pipe is already busy. Another reader is consuming $TRACE_DIR/trace_pipe. Stop that reader first, or rerun with --mode dmesg."
  fi

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    setsid bash -c '
      trace_dir="$1"
      stats="$2"
      addrs="$3"
      err="$4"
      sudo cat "$trace_dir/trace_pipe" 2>"$err" | awk -v stats="$stats" -v addrs="$addrs" '"'"'
        /Replayable fault stats GPU|batch_faults=|本批次总缺页实例数=/ { print >> stats; fflush(stats); next }
        /Replayable fault GPU|精确原始地址=/ { print >> addrs; fflush(addrs); next }
      '"'"'
    ' _ "$TRACE_DIR" "$RUN_STATS_LOG" "$RUN_ADDRESS_LOG" "$TRACE_READER_ERROR_LOG" &
  else
    setsid bash -c '
      trace_dir="$1"
      stats="$2"
      pattern="$3"
      err="$4"
      sudo cat "$trace_dir/trace_pipe" 2>"$err" | grep -E --line-buffered "$pattern" > "$stats"
    ' _ "$TRACE_DIR" "$RUN_STATS_LOG" "$STATS_LINE_PATTERN" "$TRACE_READER_ERROR_LOG" &
  fi

  TRACE_PID=$!
  sleep 1
  if ! kill -0 "$TRACE_PID" 2>/dev/null; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
    if [ -s "$TRACE_READER_ERROR_LOG" ]; then
      echo "===== trace reader stderr =====" >&2
      tail -n 20 "$TRACE_READER_ERROR_LOG" >&2 || true
    fi
    die "Failed to start trace capture"
  fi
}

start_dmesg_capture_if_needed() {
  [ "$MODE" = "dmesg" ] || return 0

  local dmesg_follow_arg="--follow"
  if dmesg --help 2>&1 | grep -q -- "--follow-new"; then
    dmesg_follow_arg="--follow-new"
  fi

  mkdir -p "$(dirname "$TRACE_LOG")"
  RUN_STATS_LOG="$TRACE_LOG"
  : > "$RUN_STATS_LOG"

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    mkdir -p "$(dirname "$ADDRESS_TRACE_LOG")"
    RUN_ADDRESS_LOG="$ADDRESS_TRACE_LOG"
    : > "$RUN_ADDRESS_LOG"
  else
    RUN_ADDRESS_LOG=""
  fi

  if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ]; then
    sudo dmesg "$dmesg_follow_arg" --time-format iso | awk -v stats="$RUN_STATS_LOG" -v addrs="$RUN_ADDRESS_LOG" '
      /Replayable fault stats GPU|batch_faults=|本批次总缺页实例数=/ { print >> stats; fflush(stats); next }
      /Replayable fault GPU|精确原始地址=/ { print >> addrs; fflush(addrs); next }
    ' &
  else
    sudo dmesg "$dmesg_follow_arg" --time-format iso | grep -E --line-buffered "$STATS_LINE_PATTERN" > "$RUN_STATS_LOG" &
  fi

  DMESG_PID=$!
  sleep 1
  kill -0 "$DMESG_PID" 2>/dev/null || die "Failed to start dmesg capture"
}

reload_uvm_module_for_counter_reset() {
  need_file "$UVM_KO_PATH"
  [ "$UVM_ENABLE_DEBUG_PROCFS" = "0" ] || [ "$UVM_ENABLE_DEBUG_PROCFS" = "1" ] \
    || die "--uvm-enable-debug-procfs must be 0 or 1"
  [ "$UVM_ENABLE_BUILTIN_TESTS" = "0" ] || [ "$UVM_ENABLE_BUILTIN_TESTS" = "1" ] \
    || die "--uvm-enable-builtin-tests must be 0 or 1"

  echo "Resetting UVM total counters by reloading nvidia_uvm..."

  if lsmod | grep -q '^nvidia_uvm'; then
    sudo rmmod nvidia_uvm || die "Failed to rmmod nvidia_uvm (module may still be in use)"
  fi

  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs="$UVM_ENABLE_DEBUG_PROCFS" \
    uvm_enable_builtin_tests="$UVM_ENABLE_BUILTIN_TESTS" \
    uvm_phase_f_route_log_delay_ms="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0 \
    || die "Failed to insmod $UVM_KO_PATH"

  if [ -r "$PARAM_DIR/uvm_enable_debug_procfs" ]; then
    local debug_val
    debug_val="$(cat "$PARAM_DIR/uvm_enable_debug_procfs" 2>/dev/null || true)"
    [ "$debug_val" = "$UVM_ENABLE_DEBUG_PROCFS" ] \
      || die "uvm_enable_debug_procfs expected $UVM_ENABLE_DEBUG_PROCFS, got $debug_val"
  fi
  if [ -r "$PARAM_DIR/uvm_enable_builtin_tests" ]; then
    local tests_val
    tests_val="$(cat "$PARAM_DIR/uvm_enable_builtin_tests" 2>/dev/null || true)"
    [ "$tests_val" = "$UVM_ENABLE_BUILTIN_TESTS" ] \
      || die "uvm_enable_builtin_tests expected $UVM_ENABLE_BUILTIN_TESTS, got $tests_val"
  fi
  if [ -r "$PARAM_DIR/uvm_phase_f_route_log_delay_ms" ]; then
    local route_delay_val
    route_delay_val="$(cat "$PARAM_DIR/uvm_phase_f_route_log_delay_ms" 2>/dev/null || true)"
    [ "$route_delay_val" = "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" ] \
      || die "uvm_phase_f_route_log_delay_ms expected $UVM_PHASE_F_ROUTE_LOG_DELAY_MS, got $route_delay_val"
  fi

  # Optional: clear old kernel ring lines so dmesg mode only shows this run.
  if [ "$MODE" = "dmesg" ]; then
    sudo dmesg -C >/dev/null 2>&1 || true
  fi
}

stop_trace_capture_if_needed() {
  [ "$MODE" = "trace" ] || return 0

  if [ -n "$TRACE_PID" ] && kill -0 "$TRACE_PID" 2>/dev/null; then
    # TRACE_PID is a setsid-launched process-group leader. Kill the whole group
    # so the left side of the trace_pipe pipeline (`sudo cat`) cannot linger.
    kill -TERM -- "-$TRACE_PID" >/dev/null 2>&1 || kill "$TRACE_PID" >/dev/null 2>&1 || true
    # Avoid hanging forever on wait if child teardown is abnormal.
    for _ in $(seq 1 20); do
      kill -0 "$TRACE_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL -- "-$TRACE_PID" >/dev/null 2>&1 || kill -KILL "$TRACE_PID" >/dev/null 2>&1 || true
    wait "$TRACE_PID" >/dev/null 2>&1 || true
  fi
  TRACE_PID=""
  if [ -n "$TRACE_DIR" ] && [ -e "$TRACE_DIR/tracing_on" ]; then
    echo 0 | sudo tee "$TRACE_DIR/tracing_on" >/dev/null || true
  fi
  if [ -n "$TRACE_READER_ERROR_LOG" ] && [ -e "$TRACE_READER_ERROR_LOG" ]; then
    rm -f "$TRACE_READER_ERROR_LOG" >/dev/null 2>&1 || true
  fi
  TRACE_READER_ERROR_LOG=""
}

stop_dmesg_capture_if_needed() {
  [ "$MODE" = "dmesg" ] || return 0

  if [ -n "$DMESG_PID" ] && kill -0 "$DMESG_PID" 2>/dev/null; then
    kill "$DMESG_PID" >/dev/null 2>&1 || true
    # Avoid hanging forever on wait if child teardown is abnormal.
    for _ in $(seq 1 20); do
      kill -0 "$DMESG_PID" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "$DMESG_PID" >/dev/null 2>&1 || true
    wait "$DMESG_PID" >/dev/null 2>&1 || true
  fi
  DMESG_PID=""
}

resolve_trace_dir() {
  if [ -n "$TRACE_DIR" ] && [ -e "$TRACE_DIR/trace_pipe" ]; then
    return 0
  fi

  # Prefer tracefs mountpoint on modern kernels.
  if [ -e /sys/kernel/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/tracing"
    return 0
  fi

  # Fallback to debugfs-mounted tracing path.
  if [ -e /sys/kernel/debug/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/debug/tracing"
    return 0
  fi

  # Try mounting tracefs first.
  sudo mount -t tracefs nodev /sys/kernel/tracing >/dev/null 2>&1 || true
  if [ -e /sys/kernel/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/tracing"
    return 0
  fi

  # Last fallback: mount debugfs then use tracing under it.
  sudo mount -t debugfs none /sys/kernel/debug >/dev/null 2>&1 || true
  if [ -e /sys/kernel/debug/tracing/trace_pipe ]; then
    TRACE_DIR="/sys/kernel/debug/tracing"
    return 0
  fi

  die "Unable to locate trace_pipe in /sys/kernel/tracing or /sys/kernel/debug/tracing"
}

configure_uvm_params() {
  local destination=0
  [ "$MODE" = "trace" ] && destination=1

  write_param uvm_perf_fault_kv_start "$KV_START"
  write_param uvm_perf_fault_kv_end "$KV_END"
  write_param uvm_perf_fault_kv_range_enable 1
  write_param uvm_perf_fault_log_destination "$destination"
  write_param uvm_perf_fault_log_counters 1
  write_param uvm_perf_fault_log_addresses "$ENABLE_FAULT_ADDRESS_LOG"
}

cleanup() {
  stop_dmesg_capture_if_needed || true
  stop_trace_capture_if_needed || true

  if [ "$KEEP_ON_EXIT" -eq 0 ]; then
    write_param uvm_perf_fault_log_counters 0 || true
    write_param uvm_perf_fault_kv_range_enable 0 || true
    write_param uvm_perf_fault_log_addresses 0 || true
    write_param uvm_perf_fault_log_destination 0 || true
  fi

  stop_server_if_needed || true
}

# 解析命令行传入的参数
parse_args() {
  # $# 代表当前传入参数的总个数。
  # 当参数个数大于 0 时，持续循环处理。每次处理完后会通过 shift 踢除已处理的参数。
  while [ $# -gt 0 ]; do
    # $1 代表当前参数列表中的第一个参数（即当前正在检查的选项名）
    case "$1" in
      # ==========================================
      # 键值对类型参数 (Key-Value): 选项后面跟一个具体的值
      # 处理方式: 将 $2 (紧跟在选项后的值) 赋给对应变量，
      # 然后使用 'shift 2' 将这两个参数从参数列表中移除，让下一个选项成为新的 $1。
      # ==========================================
      --model) MODEL="$2"; shift 2 ;;                       # 设置模型名称
      --port) PORT="$2"; shift 2 ;;                         # 设置服务端口
      --prompts) PROMPTS="$2"; shift 2 ;;                   # 设置测试使用的 prompt 数量
      --request-rate) REQUEST_RATE="$2"; shift 2 ;;         # 设置请求发送速率 (QPS)
      --output-len) OUTPUT_LEN="$2"; shift 2 ;;             # 设置生成的输出 token 长度
      --seed) SEED="$2"; shift 2 ;;                         # 设置随机种子
      --dataset-path) DATASET_PATH="$2"; shift 2 ;;         # 设置数据集的文件路径
      --address-log) ADDRESS_LOG="$2"; shift 2 ;;           # 设置 vLLM 地址日志的输出路径
      --server-log) SERVER_LOG="$2"; shift 2 ;;             # 设置服务端运行日志的路径
      --bench-log) BENCH_LOG="$2"; shift 2 ;;               # 设置压测结果日志的路径
      --mode) MODE="$2"; shift 2 ;;                         # 设置统计模式 (dmesg 还是 trace)
      --trace-log) TRACE_LOG="$2"; shift 2 ;;               # 设置 trace 日志的输出路径
      --address-trace-log) ADDRESS_TRACE_LOG="$2"; ENABLE_FAULT_ADDRESS_LOG=1; shift 2 ;; # 设置地址日志路径（并自动开启地址记录）
      --allocator-log) ALLOCATOR_TRACE_LOG="$2"; shift 2 ;; # 设置 allocator trace 日志路径
      --uvm-trace-min-bytes) UVM_TRACE_MIN_BYTES="$2"; shift 2 ;; # 设置 allocator trace 最小阈值
      --uvm-policy-enable) UVM_POLICY_ENABLE="$2"; shift 2 ;; # 是否开启 allocator policy 分类日志
      --uvm-policy-mode) UVM_POLICY_MODE="$2"; shift 2 ;; # allocator policy 模式
      --uvm-policy-warmup-prefetch-min-bytes) UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES="$2"; shift 2 ;; # warmup prefetch 最小阈值
      --uvm-policy-warmup-advise-gpu) UVM_POLICY_WARMUP_ADVISE_GPU="$2"; shift 2 ;; # warmup prefetch 时是否设置 GPU preferred location
      --uvm-policy-runtime-enable) UVM_POLICY_RUNTIME_ENABLE="$2"; shift 2 ;; # Stage M in-process policy runtime
      --uvm-unknown-detail-enable) UVM_UNKNOWN_DETAIL_ENABLE="$2"; shift 2 ;; # 是否记录 unknown_managed 细粒度日志
      --uvm-unknown-detail-min-bytes) UVM_UNKNOWN_DETAIL_MIN_BYTES="$2"; shift 2 ;; # unknown_managed 细粒度日志的最小尺寸
      --uvm-gap-watch-enable) UVM_GAP_WATCH_ENABLE="$2"; shift 2 ;; # 是否启用 gap 地址段观察
      --uvm-gap-watch-name) UVM_GAP_WATCH_NAME="$2"; shift 2 ;; # 被观察 gap 的标签名
      --uvm-gap-watch-start) UVM_GAP_WATCH_START="$2"; shift 2 ;; # 被观察 gap 的起始地址
      --uvm-gap-watch-end) UVM_GAP_WATCH_END="$2"; shift 2 ;; # 被观察 gap 的结束地址
      --uvm-gap-watch-all-classes) UVM_GAP_WATCH_ALL_CLASSES="$2"; shift 2 ;; # gap watch 是否捕获所有类别
      --uvm-gap-watch-min-bytes) UVM_GAP_WATCH_MIN_BYTES="$2"; shift 2 ;; # gap watch 最小尺寸
      --uvm-gap-watch-target-class) UVM_GAP_WATCH_TARGET_CLASS="$2"; shift 2 ;; # gap watch 目标类别
      --uvm-gap-watch-policy-action) UVM_GAP_WATCH_POLICY_ACTION="$2"; shift 2 ;; # gap watch 命中后的策略动作
      --uvm-gap-watch-control-file) UVM_GAP_WATCH_CONTROL_FILE="$2"; shift 2 ;; # allocator 运行时热更新的控制文件
      --uvm-gap-watch-refresh-ms) UVM_GAP_WATCH_REFRESH_MS="$2"; shift 2 ;; # allocator 轮询控制文件的间隔
      --uvm-device-direct-enable) UVM_DEVICE_DIRECT_ENABLE="$2"; shift 2 ;; # device_direct 阶段 C 默认关闭的真实执行门控
      --uvm-device-direct-min-bytes) UVM_DEVICE_DIRECT_MIN_BYTES="$2"; shift 2 ;; # device_direct 候选最小尺寸
      --uvm-device-direct-max-bytes) UVM_DEVICE_DIRECT_MAX_BYTES="$2"; shift 2 ;; # device_direct 候选最大尺寸
      --uvm-device-direct-max-total-bytes) UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES="$2"; shift 2 ;; # device_direct C1 总 live bytes 预算
      --uvm-device-direct-backend) UVM_DEVICE_DIRECT_BACKEND="$2"; shift 2 ;; # device_direct C2 后端 cuda_malloc/cuda_malloc_async
      --uvm-device-direct-pool-release-threshold) UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="$2"; shift 2 ;; # cuda_malloc_async 默认 mempool release threshold
      --uvm-device-direct-target-phases) UVM_DEVICE_DIRECT_TARGET_PHASES="$2"; shift 2 ;; # device_direct 候选 phase 前缀 allowlist
      --uvm-kv-budget-bytes) UVM_KV_BUDGET_BYTES="$2"; shift 2 ;; # Stage D KV cache 独立预算，0 表示不限额
      --uvm-kv-budget-mode) UVM_KV_BUDGET_MODE="$2"; shift 2 ;; # Stage D KV budget 模式 trace_only/enforce
      --uvm-kv-runtime-enable) UVM_KV_RUNTIME_ENABLE="$2"; shift 2 ;; # Stage J runtime KV pressure policy 开关
      --uvm-kv-runtime-mode) UVM_KV_RUNTIME_MODE="$2"; shift 2 ;; # Stage J mode trace_only/enforce
      --uvm-kv-runtime-budget-bytes) UVM_KV_RUNTIME_BUDGET_BYTES="$2"; shift 2 ;; # Stage J runtime KV bytes budget
      --uvm-kv-runtime-budget-blocks) UVM_KV_RUNTIME_BUDGET_BLOCKS="$2"; shift 2 ;; # Stage J runtime KV block budget
      --uvm-kv-runtime-trace-file) UVM_KV_RUNTIME_TRACE_FILE="$2"; shift 2 ;; # Stage J runtime JSONL trace
      --uvm-kv-runtime-eviction-policy) UVM_KV_RUNTIME_EVICTION_POLICY="$2"; shift 2 ;; # Stage J runtime policy
      --uvm-kv-runtime-candidate-limit) UVM_KV_RUNTIME_CANDIDATE_LIMIT="$2"; shift 2 ;; # Stage J candidate trace cap
      --uvm-kv-runtime-prefix-evict-enable) UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE="$2"; shift 2 ;; # Stage J prefix-cache executor gate
      --uvm-kv-runtime-prefix-evict-max-blocks) UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS="$2"; shift 2 ;; # Stage J prefix-cache max evict blocks
      --uvm-kv-active-protect-enable) UVM_KV_ACTIVE_PROTECT_ENABLE="$2"; shift 2 ;; # Stage M active KV protection gate
      --uvm-kv-active-protect-mode) UVM_KV_ACTIVE_PROTECT_MODE="$2"; shift 2 ;; # Stage M active KV protection action
      --uvm-kv-active-protect-device) UVM_KV_ACTIVE_PROTECT_DEVICE="$2"; shift 2 ;; # Stage M active KV protection device
      --uvm-kv-active-protect-max-blocks) UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS="$2"; shift 2 ;; # Stage M active KV block cap
      --uvm-weight-budget-bytes) UVM_WEIGHT_BUDGET_BYTES="$2"; shift 2 ;; # Stage E weights 独立预算，0 表示不限额
      --uvm-weight-budget-mode) UVM_WEIGHT_BUDGET_MODE="$2"; shift 2 ;; # Stage E weight budget 模式 trace_only/enforce
      --uvm-weight-budget-cpu-offload-enable) UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE="$2"; shift 2 ;; # Stage G allocator-level CPU offload gate
      --uvm-weight-map-enable) UVM_WEIGHT_MAP_ENABLE="$2"; shift 2 ;; # Stage E weight semantic map 开关
      --uvm-weight-map-file) UVM_WEIGHT_MAP_FILE="$2"; shift 2 ;; # Stage E weight semantic map JSONL
      --uvm-moe-routing-trace-enable) UVM_MOE_ROUTING_TRACE_ENABLE="$2"; shift 2 ;; # Stage E MoE routing trace 开关
      --uvm-moe-routing-trace-file) UVM_MOE_ROUTING_TRACE_FILE="$2"; shift 2 ;; # Stage E MoE routing trace JSONL
      --uvm-weight-prefetch-enable) UVM_WEIGHT_PREFETCH_ENABLE="$2"; shift 2 ;; # Stage I expert weight prefetch 开关
      --uvm-weight-prefetch-mode) UVM_WEIGHT_PREFETCH_MODE="$2"; shift 2 ;; # Stage I mode trace_only/prefetch
      --uvm-weight-prefetch-trace-enable) UVM_WEIGHT_PREFETCH_TRACE_ENABLE="$2"; shift 2 ;; # Stage I prefetch trace gate
      --uvm-weight-prefetch-trace-file) UVM_WEIGHT_PREFETCH_TRACE_FILE="$2"; shift 2 ;; # Stage I prefetch trace JSONL
      --uvm-weight-prefetch-max-bytes-per-step) UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="$2"; shift 2 ;; # Stage I 每层调用 prefetch 字节预算
      --uvm-weight-prefetch-max-experts-per-layer) UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;; # Stage I 每层最多 active experts
      --uvm-weight-prefetch-skip-resident) UVM_WEIGHT_PREFETCH_SKIP_RESIDENT="$2"; shift 2 ;; # Stage I 跳过已 resident region
      --uvm-weight-prefetch-expert-cache-enable) UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="$2"; shift 2 ;; # Stage I 跳过全 resident expert
      --uvm-weight-prefetch-region-cooldown-steps) UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="$2"; shift 2 ;; # Stage I region prefetch cooldown
      --uvm-weight-router-topk-every-n-steps) UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="$2"; shift 2 ;; # Stage I router_topk 低频触发
      --uvm-weight-prefetch-target-roles) UVM_WEIGHT_PREFETCH_TARGET_ROLES="$2"; shift 2 ;; # Stage I prefetch roles
      --uvm-weight-prefetch-device) UVM_WEIGHT_PREFETCH_DEVICE="$2"; shift 2 ;; # Stage I target device
      --uvm-weight-prefetch-plan-file) UVM_WEIGHT_PREFETCH_PLAN_FILE="$2"; shift 2 ;; # Stage I Stage H prefetch plan JSON
      --uvm-weight-prefetch-require-plan) UVM_WEIGHT_PREFETCH_REQUIRE_PLAN="$2"; shift 2 ;; # Stage I 是否要求 prefetch_plan 命中
      --uvm-weight-prefetch-source) UVM_WEIGHT_PREFETCH_SOURCE="$2"; shift 2 ;; # Stage I source router_topk/plan_intersection/oracle_replay
      --uvm-oracle-expert-trace-file) UVM_ORACLE_EXPERT_TRACE_FILE="$2"; shift 2 ;; # Stage M oracle replay routing trace
      --uvm-oracle-prefetch-lookahead-steps) UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS="$2"; shift 2 ;; # Stage M oracle replay lookahead
      --uvm-weight-prefetch-hit-trace-enable) UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE="$2"; shift 2 ;; # Stage M prefetch hit-before-access probe
      --uvm-weight-prefetch-hit-driver-query-enable) UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="$2"; shift 2 ;; # Stage M real residency query before access
      --uvm-weight-prefetch-hit-max-regions-per-step) UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="$2"; shift 2 ;; # Stage M real query sample limit
      --uvm-weight-batch-executor-enable) UVM_WEIGHT_BATCH_EXECUTOR_ENABLE="$2"; shift 2 ;; # Stage M/E C++ batch executor gate
      --uvm-weight-executor) UVM_WEIGHT_EXECUTOR="$2"; shift 2 ;; # Stage M/F executor cpp_batch/driver_batch/auto
      --uvm-weight-offload-enable) UVM_WEIGHT_OFFLOAD_ENABLE="$2"; shift 2 ;; # Stage I cold expert offload/advise 开关
      --uvm-weight-offload-mode) UVM_WEIGHT_OFFLOAD_MODE="$2"; shift 2 ;; # Stage I cold expert mode trace_only/advise_cpu/prefetch_cpu
      --uvm-weight-offload-plan-file) UVM_WEIGHT_OFFLOAD_PLAN_FILE="$2"; shift 2 ;; # Stage I Stage H offload plan JSON
      --uvm-weight-offload-max-bytes-per-step) UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP="$2"; shift 2 ;; # Stage I 每层 cold offload 字节预算
      --uvm-weight-offload-max-experts-per-layer) UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;; # Stage I 每层最多 cold experts
      --uvm-weight-offload-target-roles) UVM_WEIGHT_OFFLOAD_TARGET_ROLES="$2"; shift 2 ;; # Stage I cold offload roles
      --uvm-init-kv-sizing-enable) UVM_INIT_KV_SIZING_ENABLE="$2"; shift 2 ;; # Stage M init-time KV sizing gate
      --uvm-init-kv-sizing-policy) UVM_INIT_KV_SIZING_POLICY="$2"; shift 2 ;; # Stage M KV sizing policy report_only/clamp/override
      --uvm-init-kv-max-bytes) UVM_INIT_KV_MAX_BYTES="$2"; shift 2 ;; # Stage M KV max bytes before cache allocation
      --uvm-init-kv-max-blocks) UVM_INIT_KV_MAX_BLOCKS="$2"; shift 2 ;; # Stage M KV max blocks before cache allocation
      --uvm-init-kv-min-blocks) UVM_INIT_KV_MIN_BLOCKS="$2"; shift 2 ;; # Stage M KV min block guard
      --uvm-init-kv-trace-file) UVM_INIT_KV_TRACE_FILE="$2"; shift 2 ;; # Stage M KV sizing JSONL trace
      --uvm-init-weight-offload-enable) UVM_INIT_WEIGHT_OFFLOAD_ENABLE="$2"; shift 2 ;; # Stage M cold expert pre-offload gate
      --uvm-init-weight-offload-mode) UVM_INIT_WEIGHT_OFFLOAD_MODE="$2"; shift 2 ;; # Stage M trace_only/advise_cpu/prefetch_cpu/offload_cpu
      --uvm-init-weight-plan-file) UVM_INIT_WEIGHT_PLAN_FILE="$2"; shift 2 ;; # Stage M offload_plan JSON
      --uvm-init-weight-target-roles) UVM_INIT_WEIGHT_TARGET_ROLES="$2"; shift 2 ;; # Stage M expert weight roles
      --uvm-init-weight-max-bytes) UVM_INIT_WEIGHT_MAX_BYTES="$2"; shift 2 ;; # Stage M total pre-offload byte cap
      --uvm-init-weight-max-experts-per-layer) UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;; # Stage M per-layer cold expert cap
      --uvm-init-weight-sync) UVM_INIT_WEIGHT_SYNC="$2"; shift 2 ;; # Stage M synchronize after pre-offload
      --uvm-init-weight-hot-prefetch-enable) UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE="$2"; shift 2 ;; # Stage M init hot expert GPU prefetch gate
      --uvm-init-weight-hot-prefetch-sync) UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC="$2"; shift 2 ;; # Stage M synchronize after init hot GPU prefetch
      --uvm-init-weight-hot-prefetch-max-bytes) UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="$2"; shift 2 ;; # Stage M total init hot GPU prefetch byte cap
      --uvm-init-weight-hot-prefetch-device) UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE="$2"; shift 2 ;; # Stage M init hot GPU prefetch target device
      --uvm-init-weight-hot-prefetch-real-query-enable) UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="$2"; shift 2 ;; # Query real GPU residency after init hot prefetch
      --uvm-hot-residency-stage-trace-enable) UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE="$2"; shift 2 ;; # Query hot plan residency at stage markers
      --uvm-hot-residency-stage-max-num-tokens) UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="$2"; shift 2 ;; # Decode marker threshold
      --uvm-hot-residency-stage-device) UVM_HOT_RESIDENCY_STAGE_DEVICE="$2"; shift 2 ;; # Stage marker query device
      --uvm-init-weight-trace-file) UVM_INIT_WEIGHT_TRACE_FILE="$2"; shift 2 ;; # Stage M weight JSONL trace
      --uvm-init-weight-require-routing-plan) UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN="$2"; shift 2 ;; # Stage M require routing_heatmap_v1 plan
      --uvm-matched-cpu-offload-enable) UVM_MATCHED_CPU_OFFLOAD_ENABLE="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-manifest-file) UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-trace-file) UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-mode) UVM_MATCHED_CPU_OFFLOAD_MODE="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-executor) UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-sync) UVM_MATCHED_CPU_OFFLOAD_SYNC="$2"; shift 2 ;;
      --uvm-matched-cpu-offload-max-bytes) UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES="$2"; shift 2 ;;
      --uvm-pool-coordinator-enable) UVM_POOL_COORDINATOR_ENABLE="$2"; shift 2 ;; # Stage K global coordinator gate
      --uvm-pool-coordinator-mode) UVM_POOL_COORDINATOR_MODE="$2"; shift 2 ;; # Stage K mode trace_only/enforce
      --uvm-pool-coordinator-trace-enable) UVM_POOL_COORDINATOR_TRACE_ENABLE="$2"; shift 2 ;; # Stage K coordinator trace gate
      --uvm-pool-coordinator-trace-file) UVM_POOL_COORDINATOR_TRACE_FILE="$2"; shift 2 ;; # Stage K JSONL trace
      --uvm-pool-coordinator-global-bytes-per-step) UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP="$2"; shift 2 ;; # Stage K global action budget
      --uvm-pool-coordinator-weight-bytes-per-step) UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP="$2"; shift 2 ;; # Stage K weights action budget
      --uvm-pool-coordinator-kv-bytes-per-step) UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP="$2"; shift 2 ;; # Stage K KV action budget
      --uvm-pool-coordinator-scratch-bytes-per-step) UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP="$2"; shift 2 ;; # Stage K scratch action budget
      --uvm-pool-coordinator-priority) UVM_POOL_COORDINATOR_PRIORITY="$2"; shift 2 ;; # Stage K report priority label
      --uvm-residency-enable) UVM_RESIDENCY_ENABLE="$2"; shift 2 ;; # Stage L0 estimated residency ledger
      --uvm-residency-stage) UVM_RESIDENCY_STAGE="$2"; shift 2 ;; # Stage L JSONL label
      --uvm-residency-mode) UVM_RESIDENCY_MODE="$2"; shift 2 ;; # Stage L0 mode trace_only/enforce
      --uvm-residency-trace-enable) UVM_RESIDENCY_TRACE_ENABLE="$2"; shift 2 ;; # Stage L0 JSONL trace gate
      --uvm-residency-trace-file) UVM_RESIDENCY_TRACE_FILE="$2"; shift 2 ;; # Stage L0 JSONL trace
      --uvm-residency-global-gpu-bytes) UVM_RESIDENCY_GLOBAL_GPU_BYTES="$2"; shift 2 ;; # Stage L0 global estimated GPU budget
      --uvm-residency-weight-gpu-bytes) UVM_RESIDENCY_WEIGHT_GPU_BYTES="$2"; shift 2 ;; # Stage L0 weights estimated GPU budget
      --uvm-residency-kv-gpu-bytes) UVM_RESIDENCY_KV_GPU_BYTES="$2"; shift 2 ;; # Stage L0 KV estimated GPU budget
      --uvm-residency-scratch-gpu-bytes) UVM_RESIDENCY_SCRATCH_GPU_BYTES="$2"; shift 2 ;; # Stage L0 scratch estimated GPU budget
      --uvm-residency-safety-margin-bytes) UVM_RESIDENCY_SAFETY_MARGIN_BYTES="$2"; shift 2 ;; # Stage L0 global safety margin
      --uvm-residency-scratch-reserved-bytes) UVM_RESIDENCY_SCRATCH_RESERVED_BYTES="$2"; shift 2 ;; # Stage L3 scratch reserve report
      --uvm-residency-scratch-margin-bytes) UVM_RESIDENCY_SCRATCH_MARGIN_BYTES="$2"; shift 2 ;; # Stage L3 scratch reserve margin
      --uvm-residency-kv-sizing-enable) UVM_RESIDENCY_KV_SIZING_ENABLE="$2"; shift 2 ;; # Stage L4 KV sizing report gate
      --uvm-residency-kv-sizing-policy) UVM_RESIDENCY_KV_SIZING_POLICY="$2"; shift 2 ;; # Stage L4 KV sizing policy label
      --uvm-residency-kv-sizing-max-blocks) UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS="$2"; shift 2 ;; # Stage L4 override max block guard
      --uvm-residency-kv-metadata-enable) UVM_RESIDENCY_KV_METADATA_ENABLE="$2"; shift 2 ;; # Stage L5 KV metadata-only residency
      --uvm-residency-kv-data-offload-enable) UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE="$2"; shift 2 ;; # Stage L6 safe KV offload gate
      --uvm-residency-kv-data-offload-mode) UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE="$2"; shift 2 ;; # Stage L6 offload mode trace_only/prefetch_cpu
      --uvm-residency-kv-prefetch-back-enable) UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE="$2"; shift 2 ;; # Stage L6 prefetch-back gate
      --uvm-residency-kv-prefetch-back-mode) UVM_RESIDENCY_KV_PREFETCH_BACK_MODE="$2"; shift 2 ;; # Stage L6 prefetch-back mode trace_only/prefetch
      --uvm-residency-kv-prefetch-device) UVM_RESIDENCY_KV_PREFETCH_DEVICE="$2"; shift 2 ;; # Stage L6 prefetch-back target device
      --uvm-residency-kv-max-offload-blocks-per-step) UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP="$2"; shift 2 ;; # Stage L6 max safe offload blocks
      --uvm-residency-kv-pointer-map-file) UVM_RESIDENCY_KV_POINTER_MAP_FILE="$2"; shift 2 ;; # Stage L6 raw pointer map sidecar
      --uvm-residency-priority) UVM_RESIDENCY_PRIORITY="$2"; shift 2 ;; # Stage L0 report priority label
      --uvm-residency-weight-initial-state) UVM_RESIDENCY_WEIGHT_INITIAL_STATE="$2"; shift 2 ;; # Stage L2 initial expert state
      --uvm-residency-weight-enforce-hot-set) UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET="$2"; shift 2 ;; # Stage L2 weights enforce gate
      --uvm-residency-weight-sync-offload) UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD="$2"; shift 2 ;; # Stage L2 immediate release on successful victim action
      --uvm-residency-weight-max-victims-per-step) UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP="$2"; shift 2 ;; # Stage L2 victim cap
      --uvm-residency-weight-victim-policy) UVM_RESIDENCY_WEIGHT_VICTIM_POLICY="$2"; shift 2 ;; # Stage L2 victim policy label
      --uvm-residency-weight-victim-action) UVM_RESIDENCY_WEIGHT_VICTIM_ACTION="$2"; shift 2 ;; # Stage L2 victim action
      --uvm-residency-weight-active-evict-enable) UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE="$2"; shift 2 ;; # Stage L2 active-set eviction gate
      --uvm-residency-weight-active-evict-max-experts-per-layer) UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="$2"; shift 2 ;; # Stage L2 active-set eviction cap
      --uvm-residency-weight-evict-protect-recent-steps) UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS="$2"; shift 2 ;; # Stage L2 recent active protection
      --uvm-pool-registry-enable) UVM_POOL_REGISTRY_ENABLE="$2"; shift 2 ;; # Stage F unified pool registry telemetry
      --uvm-scratch-pool-enable) UVM_SCRATCH_POOL_ENABLE="$2"; shift 2 ;; # Stage G scratch pool admission control
      --uvm-scratch-pool-budget-bytes) UVM_SCRATCH_POOL_BUDGET_BYTES="$2"; shift 2 ;; # Stage G scratch pool device-direct live bytes budget
      --uvm-scratch-pool-mode) UVM_SCRATCH_POOL_MODE="$2"; shift 2 ;; # Stage G scratch pool mode trace_only/enforce
      --uvm-scratch-pool-target-phases) UVM_SCRATCH_POOL_TARGET_PHASES="$2"; shift 2 ;; # Stage G scratch pool phase allowlist
      --uvm-scratch-pool-reuse-cache-enable) UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE="$2"; shift 2 ;; # Phase 7 scratch size-bucket reuse cache
      --uvm-scratch-pool-reuse-cache-max-bytes) UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES="$2"; shift 2 ;; # Phase 7 cache retained bytes cap
      --uvm-scratch-pool-reuse-cache-max-entries-per-bucket) UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET="$2"; shift 2 ;; # Phase 7 cache bucket entry cap
      --gap-watch-metrics-summary-json) GAP_WATCH_METRICS_SUMMARY_JSON="$2"; shift 2 ;; # gap watch 命中/动作证明摘要
      --auto-gap-watch-enable) AUTO_GAP_WATCH_ENABLE="$2"; shift 2 ;; # 同进程 probe -> discover -> main 自动流程
      --auto-gap-watch-probe-prompts) AUTO_GAP_WATCH_PROBE_PROMPTS="$2"; shift 2 ;; # 自动流程 probe 阶段的 prompt 数
      --auto-gap-watch-target-gap) AUTO_GAP_WATCH_TARGET_GAP="$2"; shift 2 ;; # 自动流程优先关注的 gap index
      --auto-gap-watch-fallback-to-hottest) AUTO_GAP_WATCH_FALLBACK_TO_HOTTEST="$2"; shift 2 ;; # target gap 不热时是否回退到最热点
      --auto-gap-watch-summary-json) AUTO_GAP_WATCH_SUMMARY_JSON="$2"; shift 2 ;; # 自动流程 discovery 摘要输出
      --auto-gap-watch-post-main-summary-json) AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON="$2"; shift 2 ;; # main 阶段结束后的 gap 摘要输出
      --auto-gap-watch-policy-action-override) AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE="$2"; shift 2 ;; # 自动流程强制覆盖 gap-watch 动作，用于 observe/prefetch A/B
      --auto-gap-watch-target-class-override) AUTO_GAP_WATCH_TARGET_CLASS_OVERRIDE="$2"; shift 2 ;; # 自动流程强制覆盖目标类别
      --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;   # 设置等待服务器启动的超时时间
      --check-interval) CHECK_INTERVAL="$2"; shift 2 ;;     # 设置轮询检查的间隔时间
      --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;; # 设置 vLLM server 最大并发序列数
      --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;; # 设置 vLLM server 显存利用率
      --max-model-len) SERVER_MAX_MODEL_LEN="$2"; shift 2 ;; # 设置 vLLM server max model len

      # ==========================================
      # 布尔开关类型参数 (Boolean Flags): 只需要出现该选项即可，不需要值
      # 处理方式: 将对应的标志变量设为 1 (True)，
      # 然后使用 'shift' (等同于 shift 1) 仅移除当前这个选项本身。
      # ==========================================
      --with-address-log) ENABLE_FAULT_ADDRESS_LOG=1; shift ;; # 开启具体缺页地址记录
      --no-cleanup) NO_CLEANUP=1; shift ;;                     # 禁用运行前的环境清理
      --no-bench) NO_BENCH=1; shift ;;                         # 仅配置环境，不实际执行压测
      --reset-counters) RESET_COUNTERS=1; shift ;;             # 运行前重载 nvidia_uvm 清零累计计数
      --no-reset-counters) RESET_COUNTERS=0; shift ;;          # 不重载模块，保留累计计数
      --uvm-ko-path) UVM_KO_PATH="$2"; shift 2 ;;              # 指定 nvidia-uvm.ko 的绝对路径
      --uvm-enable-debug-procfs) UVM_ENABLE_DEBUG_PROCFS="$2"; shift 2 ;; # insmod 时的 debug procfs 开关
      --uvm-enable-builtin-tests) UVM_ENABLE_BUILTIN_TESTS="$2"; shift 2 ;; # insmod 时开启 builtin tests
      --uvm-phase-f-route-log-delay-ms) UVM_PHASE_F_ROUTE_LOG_DELAY_MS="$2"; shift 2 ;; # insmod 时的 Phase F route log 人工延迟
      --keep) KEEP_ON_EXIT=1; shift ;;                         # 退出时不重置内核状态

      # ==========================================
      # 帮助与特殊标识
      # ==========================================
      -h|--help) 
        usage;    # 调用外部定义的 usage 函数打印帮助信息
        exit 0 ;; # 正常退出脚本
      
      --) 
        # '--' 是 Linux 命令行的标准约定，表示“选项解析到此结束”。
        # 后面出现的所有内容，即使以 '-' 开头，也会被当作普通参数/命令对待。
        shift # 移除 '--' 这个符号本身
        
        # "$@" 代表当前列表中剩下的所有参数。
        # 将它们打包存入 BENCH_CMD 数组中，作为用户自定义的实际压测命令。
        BENCH_CMD=("$@") 
        CUSTOM_BENCH_CMD=1
        
        break ;; # 结束 while 循环，停止解析后续参数
        
      *) 
        # 默认匹配 (类似 switch-case 的 default)
        # 如果输入了脚本不认识的选项，调用 die 函数报错并退出。
        die "Unknown argument: $1 (use --help)" ;; 
    esac
  done
}

# 验证输入参数、检查环境依赖并构建默认压测命令
validate_inputs() {
    # 1. 验证统计模式：必须是 dmesg 或 trace 其中之一
    # [ A ] || [ B ] || die 表示：如果 A 不成立且 B 也不成立，则触发 die 函数报错
    [ "$MODE" = "dmesg" ] || [ "$MODE" = "trace" ] || die "--mode must be dmesg or trace"

    # 2. 检查系统工具依赖：确保系统已安装 curl（用于检查 vLLM 服务器是否存活）
    # command -v 用于寻找命令的路径，如果找不到会返回非零状态码
    command -v curl >/dev/null 2>&1 || die "curl is required but not found"

    # 3. 验证 NVIDIA UVM 内核参数接口：
    # 依次检查定义的 sysfs 文件是否存在。如果这些文件缺失，说明 nvidia_uvm 驱动版本不匹配
    # 或者没有打上支持 KV Cache 统计的内核补丁。
    need_file "$PARAM_DIR/uvm_perf_fault_log_counters"
    need_file "$PARAM_DIR/uvm_perf_fault_log_destination"
  need_file "$PARAM_DIR/uvm_perf_fault_log_addresses"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_range_enable"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_start"
  need_file "$PARAM_DIR/uvm_perf_fault_kv_end"
  need_writable_param "uvm_perf_fault_log_counters"
  need_writable_param "uvm_perf_fault_log_destination"
  need_writable_param "uvm_perf_fault_log_addresses"
  need_writable_param "uvm_perf_fault_kv_range_enable"
  need_writable_param "uvm_perf_fault_kv_start"
  need_writable_param "uvm_perf_fault_kv_end"

    # 4. 验证数据集文件：确保指定的 ShareGPT 等测试数据集路径有效
    need_file "$DATASET_PATH"

    # 4.1 验证 allocator trace 阈值
    [[ "$UVM_TRACE_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-trace-min-bytes must be a non-negative integer"
    [[ "$UVM_POLICY_ENABLE" =~ ^[01]$ ]] || die "--uvm-policy-enable must be 0 or 1"
    [[ "$UVM_POLICY_MODE" =~ ^(trace_only|prefetch|warmup_prefetch)$ ]] || die "--uvm-policy-mode must be trace_only, prefetch, or warmup_prefetch"
    [[ "$UVM_POLICY_WARMUP_PREFETCH_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-policy-warmup-prefetch-min-bytes must be a non-negative integer"
    [[ "$UVM_POLICY_WARMUP_ADVISE_GPU" =~ ^[01]$ ]] || die "--uvm-policy-warmup-advise-gpu must be 0 or 1"
    [[ "$UVM_POLICY_RUNTIME_ENABLE" =~ ^[01]$ ]] || die "--uvm-policy-runtime-enable must be 0 or 1"
    [[ "$UVM_UNKNOWN_DETAIL_ENABLE" =~ ^[01]$ ]] || die "--uvm-unknown-detail-enable must be 0 or 1"
    [[ "$UVM_UNKNOWN_DETAIL_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-unknown-detail-min-bytes must be a non-negative integer"
    [[ "$UVM_GAP_WATCH_ENABLE" =~ ^[01]$ ]] || die "--uvm-gap-watch-enable must be 0 or 1"
    [[ "$UVM_GAP_WATCH_ALL_CLASSES" =~ ^[01]$ ]] || die "--uvm-gap-watch-all-classes must be 0 or 1"
    [[ "$UVM_GAP_WATCH_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-gap-watch-min-bytes must be a non-negative integer"
    [[ "$UVM_GAP_WATCH_REFRESH_MS" =~ ^[0-9]+$ ]] || die "--uvm-gap-watch-refresh-ms must be a non-negative integer"
    [[ "$UVM_GAP_WATCH_POLICY_ACTION" =~ ^(observe|prefetch|advise_prefetch|device_direct_trace|device_direct|managed_default|managed_prefetch_gpu|managed_advise_prefetch_gpu)$ ]] || die "--uvm-gap-watch-policy-action must be observe, prefetch, advise_prefetch, device_direct_trace, device_direct, managed_default, managed_prefetch_gpu, or managed_advise_prefetch_gpu"
    [[ "$UVM_DEVICE_DIRECT_ENABLE" =~ ^[01]$ ]] || die "--uvm-device-direct-enable must be 0 or 1"
    [[ "$UVM_DEVICE_DIRECT_MIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-device-direct-min-bytes must be a non-negative integer"
    [[ "$UVM_DEVICE_DIRECT_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-device-direct-max-bytes must be a non-negative integer"
    [[ "$UVM_DEVICE_DIRECT_MAX_TOTAL_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-device-direct-max-total-bytes must be a non-negative integer"
    [[ "$UVM_DEVICE_DIRECT_BACKEND" =~ ^(cuda_malloc|cuda_malloc_async)$ ]] || die "--uvm-device-direct-backend must be cuda_malloc or cuda_malloc_async"
    if [ -n "$UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" ]; then
      [[ "$UVM_DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" =~ ^[0-9]+$ ]] || die "--uvm-device-direct-pool-release-threshold must be a non-negative integer"
    fi
    [ -n "$UVM_DEVICE_DIRECT_TARGET_PHASES" ] || die "--uvm-device-direct-target-phases must not be empty"
    [[ "$UVM_KV_BUDGET_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-kv-budget-bytes must be a non-negative integer"
    [[ "$UVM_KV_BUDGET_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-kv-budget-mode must be trace_only or enforce"
    [[ "$UVM_KV_RUNTIME_ENABLE" =~ ^[01]$ ]] || die "--uvm-kv-runtime-enable must be 0 or 1"
    [[ "$UVM_KV_RUNTIME_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-kv-runtime-mode must be trace_only or enforce"
    [[ "$UVM_KV_RUNTIME_BUDGET_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-kv-runtime-budget-bytes must be a non-negative integer"
    [[ "$UVM_KV_RUNTIME_BUDGET_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-kv-runtime-budget-blocks must be a non-negative integer"
    [[ "$UVM_KV_RUNTIME_EVICTION_POLICY" =~ ^(lru_prefix_cache|scheduler_aware)$ ]] || die "--uvm-kv-runtime-eviction-policy must be lru_prefix_cache or scheduler_aware"
    [[ "$UVM_KV_RUNTIME_CANDIDATE_LIMIT" =~ ^[0-9]+$ ]] || die "--uvm-kv-runtime-candidate-limit must be a non-negative integer"
    [[ "$UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE" =~ ^[01]$ ]] || die "--uvm-kv-runtime-prefix-evict-enable must be 0 or 1"
    [[ "$UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-kv-runtime-prefix-evict-max-blocks must be a non-negative integer"
    [[ "$UVM_KV_ACTIVE_PROTECT_ENABLE" =~ ^[01]$ ]] || die "--uvm-kv-active-protect-enable must be 0 or 1"
    [[ "$UVM_KV_ACTIVE_PROTECT_MODE" =~ ^(trace_only|advise|prefetch|advise_prefetch)$ ]] || die "--uvm-kv-active-protect-mode must be trace_only, advise, prefetch, or advise_prefetch"
    [[ "$UVM_KV_ACTIVE_PROTECT_DEVICE" =~ ^-?[0-9]+$ ]] || die "--uvm-kv-active-protect-device must be an integer"
    [[ "$UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-kv-active-protect-max-blocks must be a non-negative integer"
    [[ "$UVM_WEIGHT_BUDGET_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-weight-budget-bytes must be a non-negative integer"
    [[ "$UVM_WEIGHT_BUDGET_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-weight-budget-mode must be trace_only or enforce"
    [[ "$UVM_WEIGHT_BUDGET_CPU_OFFLOAD_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-budget-cpu-offload-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_MAP_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-map-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_MODE" =~ ^(trace_only|prefetch)$ ]] || die "--uvm-weight-prefetch-mode must be trace_only or prefetch"
    [[ "$UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-weight-prefetch-max-bytes-per-step must be a non-negative integer"
    [[ "$UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]] || die "--uvm-weight-prefetch-max-experts-per-layer must be a non-negative integer"
    [[ "$UVM_WEIGHT_PREFETCH_SKIP_RESIDENT" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-skip-resident must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-expert-cache-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" =~ ^[0-9]+$ ]] || die "--uvm-weight-prefetch-region-cooldown-steps must be a non-negative integer"
    [ -n "$UVM_WEIGHT_PREFETCH_TARGET_ROLES" ] || die "--uvm-weight-prefetch-target-roles must not be empty"
    [[ "$UVM_WEIGHT_PREFETCH_DEVICE" =~ ^-?[0-9]+$ ]] || die "--uvm-weight-prefetch-device must be an integer"
    [[ "$UVM_WEIGHT_PREFETCH_REQUIRE_PLAN" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-require-plan must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_SOURCE" =~ ^(router_topk|plan_intersection|oracle_replay)$ ]] || die "--uvm-weight-prefetch-source must be router_topk, plan_intersection, or oracle_replay"
    [[ "$UVM_ORACLE_PREFETCH_LOOKAHEAD_STEPS" =~ ^[0-9]+$ ]] || die "--uvm-oracle-prefetch-lookahead-steps must be a non-negative integer"
    [[ "$UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-hit-trace-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-prefetch-hit-driver-query-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-weight-prefetch-hit-max-regions-per-step must be a non-negative integer"
    if [ "$UVM_WEIGHT_PREFETCH_SOURCE" = "oracle_replay" ] && [ -z "$UVM_ORACLE_EXPERT_TRACE_FILE" ]; then
      die "--uvm-oracle-expert-trace-file is required when --uvm-weight-prefetch-source=oracle_replay"
    fi
    [[ "$UVM_WEIGHT_BATCH_EXECUTOR_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-batch-executor-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_EXECUTOR" =~ ^(cpp_batch|driver_batch|auto)$ ]] || die "--uvm-weight-executor must be cpp_batch, driver_batch, or auto"
    [[ "$UVM_WEIGHT_OFFLOAD_ENABLE" =~ ^[01]$ ]] || die "--uvm-weight-offload-enable must be 0 or 1"
    [[ "$UVM_WEIGHT_OFFLOAD_MODE" =~ ^(trace_only|advise_cpu|prefetch_cpu)$ ]] || die "--uvm-weight-offload-mode must be trace_only, advise_cpu, or prefetch_cpu"
    [[ "$UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-weight-offload-max-bytes-per-step must be a non-negative integer"
    [[ "$UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]] || die "--uvm-weight-offload-max-experts-per-layer must be a non-negative integer"
    [ -n "$UVM_WEIGHT_OFFLOAD_TARGET_ROLES" ] || die "--uvm-weight-offload-target-roles must not be empty"
    [[ "$UVM_INIT_KV_SIZING_ENABLE" =~ ^[01]$ ]] || die "--uvm-init-kv-sizing-enable must be 0 or 1"
    [[ "$UVM_INIT_KV_SIZING_POLICY" =~ ^(report_only|clamp|override)$ ]] || die "--uvm-init-kv-sizing-policy must be report_only, clamp, or override"
    [[ "$UVM_INIT_KV_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-init-kv-max-bytes must be a non-negative integer"
    [[ "$UVM_INIT_KV_MAX_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-init-kv-max-blocks must be a non-negative integer"
    [[ "$UVM_INIT_KV_MIN_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-init-kv-min-blocks must be a non-negative integer"
    [[ "$UVM_INIT_WEIGHT_OFFLOAD_ENABLE" =~ ^[01]$ ]] || die "--uvm-init-weight-offload-enable must be 0 or 1"
    [[ "$UVM_INIT_WEIGHT_OFFLOAD_MODE" =~ ^(trace_only|advise_cpu|prefetch_cpu|offload_cpu)$ ]] || die "--uvm-init-weight-offload-mode must be trace_only, advise_cpu, prefetch_cpu, or offload_cpu"
    [ -n "$UVM_INIT_WEIGHT_TARGET_ROLES" ] || die "--uvm-init-weight-target-roles must not be empty"
    [[ "$UVM_INIT_WEIGHT_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-init-weight-max-bytes must be a non-negative integer"
    [[ "$UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]] || die "--uvm-init-weight-max-experts-per-layer must be a non-negative integer"
    [[ "$UVM_INIT_WEIGHT_SYNC" =~ ^[01]$ ]] || die "--uvm-init-weight-sync must be 0 or 1"
    [[ "$UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE" =~ ^[01]$ ]] || die "--uvm-init-weight-hot-prefetch-enable must be 0 or 1"
    [[ "$UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC" =~ ^[01]$ ]] || die "--uvm-init-weight-hot-prefetch-sync must be 0 or 1"
    [[ "$UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-init-weight-hot-prefetch-max-bytes must be a non-negative integer"
    [[ "$UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE" =~ ^-?[0-9]+$ ]] || die "--uvm-init-weight-hot-prefetch-device must be an integer"
    [[ "$UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE" =~ ^[01]$ ]] || die "--uvm-init-weight-hot-prefetch-real-query-enable must be 0 or 1"
    [[ "$UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE" =~ ^[01]$ ]] || die "--uvm-hot-residency-stage-trace-enable must be 0 or 1"
    [[ "$UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS" =~ ^[0-9]+$ ]] || die "--uvm-hot-residency-stage-max-num-tokens must be a non-negative integer"
    [[ "$UVM_HOT_RESIDENCY_STAGE_DEVICE" =~ ^-?[0-9]+$ ]] || die "--uvm-hot-residency-stage-device must be an integer"
    [[ "$UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN" =~ ^[01]$ ]] || die "--uvm-init-weight-require-routing-plan must be 0 or 1"
    [[ "$UVM_MATCHED_CPU_OFFLOAD_ENABLE" =~ ^[01]$ ]] || die "--uvm-matched-cpu-offload-enable must be 0 or 1"
    [[ "$UVM_MATCHED_CPU_OFFLOAD_MODE" =~ ^(trace_only|advise_cpu|prefetch_cpu|offload_cpu|prefetch_gpu)$ ]] || die "--uvm-matched-cpu-offload-mode must be trace_only, advise_cpu, prefetch_cpu, offload_cpu, or prefetch_gpu"
    [[ "$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR" =~ ^(cpp_batch|driver_batch|auto)$ ]] || die "--uvm-matched-cpu-offload-executor must be cpp_batch, driver_batch, or auto"
    [[ "$UVM_MATCHED_CPU_OFFLOAD_SYNC" =~ ^[01]$ ]] || die "--uvm-matched-cpu-offload-sync must be 0 or 1"
    [[ "$UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-matched-cpu-offload-max-bytes must be a non-negative integer"
    [[ "$UVM_POOL_COORDINATOR_ENABLE" =~ ^[01]$ ]] || die "--uvm-pool-coordinator-enable must be 0 or 1"
    [[ "$UVM_POOL_COORDINATOR_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-pool-coordinator-mode must be trace_only or enforce"
    [[ "$UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-pool-coordinator-global-bytes-per-step must be a non-negative integer"
    [[ "$UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-pool-coordinator-weight-bytes-per-step must be a non-negative integer"
    [[ "$UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-pool-coordinator-kv-bytes-per-step must be a non-negative integer"
    [[ "$UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-pool-coordinator-scratch-bytes-per-step must be a non-negative integer"
    [ -n "$UVM_POOL_COORDINATOR_PRIORITY" ] || die "--uvm-pool-coordinator-priority must not be empty"
    [[ "$UVM_RESIDENCY_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_TRACE_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-trace-enable must be 0 or 1"
    [ -n "$UVM_RESIDENCY_STAGE" ] || die "--uvm-residency-stage must not be empty"
    [[ "$UVM_RESIDENCY_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-residency-mode must be trace_only or enforce"
    [[ "$UVM_RESIDENCY_GLOBAL_GPU_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-global-gpu-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_WEIGHT_GPU_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-weight-gpu-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_KV_GPU_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-kv-gpu-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_SCRATCH_GPU_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-scratch-gpu-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_SAFETY_MARGIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-safety-margin-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_SCRATCH_RESERVED_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-scratch-reserved-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_SCRATCH_MARGIN_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-residency-scratch-margin-bytes must be a non-negative integer"
    [[ "$UVM_RESIDENCY_KV_SIZING_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-kv-sizing-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_KV_SIZING_POLICY" =~ ^(report_only|clamp|override)$ ]] || die "--uvm-residency-kv-sizing-policy must be report_only, clamp, or override"
    [[ "$UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS" =~ ^[0-9]+$ ]] || die "--uvm-residency-kv-sizing-max-blocks must be a non-negative integer"
    [[ "$UVM_RESIDENCY_KV_METADATA_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-kv-metadata-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-kv-data-offload-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE" =~ ^(trace_only|prefetch_cpu)$ ]] || die "--uvm-residency-kv-data-offload-mode must be trace_only or prefetch_cpu"
    [[ "$UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-kv-prefetch-back-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_KV_PREFETCH_BACK_MODE" =~ ^(trace_only|prefetch)$ ]] || die "--uvm-residency-kv-prefetch-back-mode must be trace_only or prefetch"
    [[ "$UVM_RESIDENCY_KV_PREFETCH_DEVICE" =~ ^-?[0-9]+$ ]] || die "--uvm-residency-kv-prefetch-device must be an integer"
    [[ "$UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-residency-kv-max-offload-blocks-per-step must be a non-negative integer"
    [ -n "$UVM_RESIDENCY_PRIORITY" ] || die "--uvm-residency-priority must not be empty"
    [[ "$UVM_RESIDENCY_WEIGHT_INITIAL_STATE" =~ ^(unknown|cpu|gpu)$ ]] || die "--uvm-residency-weight-initial-state must be unknown, cpu, or gpu"
    [[ "$UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET" =~ ^[01]$ ]] || die "--uvm-residency-weight-enforce-hot-set must be 0 or 1"
    [[ "$UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD" =~ ^[01]$ ]] || die "--uvm-residency-weight-sync-offload must be 0 or 1"
    [[ "$UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP" =~ ^[0-9]+$ ]] || die "--uvm-residency-weight-max-victims-per-step must be a non-negative integer"
    [ -n "$UVM_RESIDENCY_WEIGHT_VICTIM_POLICY" ] || die "--uvm-residency-weight-victim-policy must not be empty"
    [[ "$UVM_RESIDENCY_WEIGHT_VICTIM_ACTION" =~ ^(advise_cpu|prefetch_cpu|offload_cpu)$ ]] || die "--uvm-residency-weight-victim-action must be advise_cpu, prefetch_cpu, or offload_cpu"
    [[ "$UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_ENABLE" =~ ^[01]$ ]] || die "--uvm-residency-weight-active-evict-enable must be 0 or 1"
    [[ "$UVM_RESIDENCY_WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER" =~ ^[0-9]+$ ]] || die "--uvm-residency-weight-active-evict-max-experts-per-layer must be a non-negative integer"
    [[ "$UVM_RESIDENCY_WEIGHT_EVICT_PROTECT_RECENT_STEPS" =~ ^[0-9]+$ ]] || die "--uvm-residency-weight-evict-protect-recent-steps must be a non-negative integer"
    [[ "$UVM_POOL_REGISTRY_ENABLE" =~ ^[01]$ ]] || die "--uvm-pool-registry-enable must be 0 or 1"
    [[ "$UVM_SCRATCH_POOL_ENABLE" =~ ^[01]$ ]] || die "--uvm-scratch-pool-enable must be 0 or 1"
    [[ "$UVM_SCRATCH_POOL_BUDGET_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-scratch-pool-budget-bytes must be a non-negative integer"
    [[ "$UVM_SCRATCH_POOL_MODE" =~ ^(trace_only|enforce)$ ]] || die "--uvm-scratch-pool-mode must be trace_only or enforce"
    [ -n "$UVM_SCRATCH_POOL_TARGET_PHASES" ] || die "--uvm-scratch-pool-target-phases must not be empty"
    [[ "$UVM_SCRATCH_POOL_REUSE_CACHE_ENABLE" =~ ^[01]$ ]] || die "--uvm-scratch-pool-reuse-cache-enable must be 0 or 1"
    [[ "$UVM_SCRATCH_POOL_REUSE_CACHE_MAX_BYTES" =~ ^[0-9]+$ ]] || die "--uvm-scratch-pool-reuse-cache-max-bytes must be a non-negative integer"
    [[ "$UVM_SCRATCH_POOL_REUSE_CACHE_MAX_ENTRIES_PER_BUCKET" =~ ^[0-9]+$ ]] || die "--uvm-scratch-pool-reuse-cache-max-entries-per-bucket must be a non-negative integer"
    [[ "$UVM_MOE_ROUTING_TRACE_ENABLE" =~ ^[01]$ ]] || die "--uvm-moe-routing-trace-enable must be 0 or 1"
    [[ "$SERVER_MAX_NUM_SEQS" =~ ^[1-9][0-9]*$ ]] || die "--server-max-num-seqs must be a positive integer"
    [[ "$SERVER_GPU_MEMORY_UTILIZATION" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]] || die "--server-gpu-memory-utilization must be in (0, 1]"
    [[ "$AUTO_GAP_WATCH_ENABLE" =~ ^[01]$ ]] || die "--auto-gap-watch-enable must be 0 or 1"
    [[ "$AUTO_GAP_WATCH_PROBE_PROMPTS" =~ ^[0-9]+$ ]] || die "--auto-gap-watch-probe-prompts must be a non-negative integer"
    [[ "$AUTO_GAP_WATCH_TARGET_GAP" =~ ^[0-9]+$ ]] || die "--auto-gap-watch-target-gap must be a non-negative integer"
    [[ "$AUTO_GAP_WATCH_FALLBACK_TO_HOTTEST" =~ ^[01]$ ]] || die "--auto-gap-watch-fallback-to-hottest must be 0 or 1"
    if [ -n "$AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE" ]; then
      [[ "$AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE" =~ ^(observe|prefetch|advise_prefetch|device_direct_trace|device_direct)$ ]] || die "--auto-gap-watch-policy-action-override must be observe, prefetch, advise_prefetch, device_direct_trace, or device_direct"
    fi
    if [ -n "$UVM_GAP_WATCH_START" ]; then
      [[ "$UVM_GAP_WATCH_START" =~ ^0x[0-9a-fA-F]+$ ]] || die "--uvm-gap-watch-start must be a hex address"
    fi
    if [ -n "$UVM_GAP_WATCH_END" ]; then
      [[ "$UVM_GAP_WATCH_END" =~ ^0x[0-9a-fA-F]+$ ]] || die "--uvm-gap-watch-end must be a hex address"
    fi
    if [ "$UVM_GAP_WATCH_ENABLE" -eq 1 ]; then
      [ -n "$UVM_GAP_WATCH_START" ] || die "--uvm-gap-watch-enable=1 requires --uvm-gap-watch-start"
      [ -n "$UVM_GAP_WATCH_END" ] || die "--uvm-gap-watch-enable=1 requires --uvm-gap-watch-end"
    fi
    if [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ]; then
      [ "$NO_BENCH" -eq 0 ] || die "--auto-gap-watch-enable requires benchmark execution"
      [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] || die "--auto-gap-watch-enable requires --with-address-log or --address-trace-log"
      [ -n "$ALLOCATOR_TRACE_LOG" ] || die "--auto-gap-watch-enable requires --allocator-log"
      [ -z "$UVM_GAP_WATCH_START" ] || die "--auto-gap-watch-enable does not accept a pre-set --uvm-gap-watch-start"
      [ -z "$UVM_GAP_WATCH_END" ] || die "--auto-gap-watch-enable does not accept a pre-set --uvm-gap-watch-end"
      [ "$AUTO_GAP_WATCH_PROBE_PROMPTS" -gt 0 ] || die "--auto-gap-watch-probe-prompts must be > 0 when auto mode is enabled"
      if [ -z "$UVM_GAP_WATCH_CONTROL_FILE" ]; then
        UVM_GAP_WATCH_CONTROL_FILE="/tmp/vllm_uvm_gap_watch_control_$$.conf"
      fi
      if [ -z "$AUTO_GAP_WATCH_SUMMARY_JSON" ]; then
        AUTO_GAP_WATCH_SUMMARY_JSON="/tmp/vllm_auto_gap_watch_summary_$$.json"
      fi
      if [ -z "$AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON" ]; then
        AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON="/tmp/vllm_auto_gap_watch_post_main_summary_$$.json"
      fi
      UVM_GAP_WATCH_ENABLE=0
    fi

    # 5. 日志目录初始化：创建压测日志所在的文件夹
    mkdir -p "$(dirname "$BENCH_LOG")"

    # 6. 自动生成 stats/address 日志文件名（若用户未显式指定）
    if [ -z "$TRACE_LOG" ]; then
      local ts
      ts="$(date +%Y%m%d_%H%M%S)"
      TRACE_LOG="/tmp/uvm_kv_fault_stats_${ts}.log"
      if [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && [ -z "$ADDRESS_TRACE_LOG" ]; then
        ADDRESS_TRACE_LOG="/tmp/uvm_kv_fault_addrs_${ts}.log"
      fi
    elif [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && [ -z "$ADDRESS_TRACE_LOG" ]; then
      local ts
      ts="$(date +%Y%m%d_%H%M%S)"
      ADDRESS_TRACE_LOG="/tmp/uvm_kv_fault_addrs_${ts}.log"
    fi

    if [ ${#BENCH_CMD[@]} -eq 0 ] && [ "$NO_BENCH" -eq 0 ]; then
      CUSTOM_BENCH_CMD=0
    fi
}

write_gap_watch_control_file() {
  local enabled="$1"
  local name="$2"
  local start="$3"
  local end="$4"
  local all_classes="$5"
  local min_bytes="$6"
  local target_class="$7"
  local policy_action="$8"

  [ -n "$UVM_GAP_WATCH_CONTROL_FILE" ] || return 0
  mkdir -p "$(dirname "$UVM_GAP_WATCH_CONTROL_FILE")"
  {
    echo "enabled=$enabled"
    echo "name=$name"
    echo "start=$start"
    echo "end=$end"
    echo "all_classes=$all_classes"
    echo "min_bytes=$min_bytes"
    echo "target_class=$target_class"
    echo "policy_action=$policy_action"
  } > "$UVM_GAP_WATCH_CONTROL_FILE"
}

build_default_bench_cmd() {
  local prompts="$1"
  local -n out_ref="$2"

  out_ref=(
    uv run --directory "$SCRIPT_DIR" vllm bench serve
    --model "$MODEL"
    --dataset-name sharegpt
    --dataset-path "$DATASET_PATH"
    --num-prompts "$prompts"
    --sharegpt-output-len "$OUTPUT_LEN"
    --seed "$SEED"
    --request-rate "$REQUEST_RATE"
    --port "$PORT"
  )
}

build_bench_cmd_for_prompts() {
  local prompts="$1"
  local -n out_ref="$2"

  if [ "$CUSTOM_BENCH_CMD" -eq 0 ]; then
    build_default_bench_cmd "$prompts" "$2"
    return 0
  fi

  out_ref=()
  local replaced=0
  local i=0
  while [ $i -lt ${#BENCH_CMD[@]} ]; do
    if [ "${BENCH_CMD[$i]}" = "--num-prompts" ] && [ $((i + 1)) -lt ${#BENCH_CMD[@]} ]; then
      out_ref+=("--num-prompts" "$prompts")
      i=$((i + 2))
      replaced=1
      continue
    fi
    out_ref+=("${BENCH_CMD[$i]}")
    i=$((i + 1))
  done

  if [ "$replaced" -eq 0 ]; then
    out_ref+=("--num-prompts" "$prompts")
  fi
}

run_benchmark_phase() {
  local phase_label="$1"
  local prompts="$2"
  local phase_cmd=()
  build_bench_cmd_for_prompts "$prompts" phase_cmd

  mkdir -p "$(dirname "$BENCH_LOG")"
  {
    echo "===== Benchmark phase: $phase_label ====="
    printf 'command:'
    printf ' %q' "${phase_cmd[@]}"
    echo
  } | tee -a "$BENCH_LOG"

  set +e
  "${phase_cmd[@]}" 2>&1 | tee -a "$BENCH_LOG"
  local bench_status=$?
  set -e

  echo "Benchmark phase '$phase_label' exit code: $bench_status" | tee -a "$BENCH_LOG"
  if [ "$bench_status" -ne 0 ]; then
    echo "===== bench log tail ====="
    tail -n 80 "$BENCH_LOG" 2>/dev/null || true
    die "Benchmark phase '$phase_label' failed"
  fi
}

run_auto_gap_watch_discovery() {
  [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ] || return 0
  [ -n "$AUTO_GAP_WATCH_SUMMARY_JSON" ] || die "auto gap watch summary path is empty"

  local discover_cmd=(
    python3 "$SCRIPT_DIR/discover_gap_watch.py"
    --address-log "$ADDRESS_LOG"
    --fault-log "$RUN_ADDRESS_LOG"
    --allocator-log "$ALLOCATOR_TRACE_LOG"
    --control-file "$UVM_GAP_WATCH_CONTROL_FILE"
    --summary-json "$AUTO_GAP_WATCH_SUMMARY_JSON"
    --watch-name "$UVM_GAP_WATCH_NAME"
    --all-classes "$UVM_GAP_WATCH_ALL_CLASSES"
    --min-bytes "$UVM_GAP_WATCH_MIN_BYTES"
    --target-gap "$AUTO_GAP_WATCH_TARGET_GAP"
    --fallback-to-hottest "$AUTO_GAP_WATCH_FALLBACK_TO_HOTTEST"
  )
  if [ -n "$AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE" ]; then
    discover_cmd+=(--policy-action-override "$AUTO_GAP_WATCH_POLICY_ACTION_OVERRIDE")
  fi
  if [ -n "$AUTO_GAP_WATCH_TARGET_CLASS_OVERRIDE" ]; then
    discover_cmd+=(--target-class-override "$AUTO_GAP_WATCH_TARGET_CLASS_OVERRIDE")
  fi

  "${discover_cmd[@]}"

  echo "Auto gap-watch summary: $AUTO_GAP_WATCH_SUMMARY_JSON"
  python3 - <<'PY' "$AUTO_GAP_WATCH_SUMMARY_JSON"
import json
import sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    data = json.load(handle)
selected = data.get("selected_gap", {})
print(
    "Discovered gap watch:",
    f"gap_index={selected.get('gap_index')}",
    f"start={selected.get('start_hex')}",
    f"end={selected.get('end_hex')}",
    f"faults={selected.get('faults')}",
    f"fallback_used={data.get('fallback_used')}",
    f"effective_target_class={data.get('effective_target_class')}",
    f"effective_policy_action={data.get('effective_policy_action')}",
  )
PY
}

mark_auto_gap_watch_main_start() {
  [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ] || return 0
  if [ -n "$RUN_ADDRESS_LOG" ] && [ -e "$RUN_ADDRESS_LOG" ]; then
    AUTO_GAP_WATCH_MAIN_START_LINE=$(( $(wc -l < "$RUN_ADDRESS_LOG") + 1 ))
  else
    AUTO_GAP_WATCH_MAIN_START_LINE=1
  fi
  echo "Auto gap-watch main start line: $AUTO_GAP_WATCH_MAIN_START_LINE"
}

print_post_main_gap_watch_discovery() {
  [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ] || return 0
  [ -n "$AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON" ] || die "post-main gap watch summary path is empty"

  python3 "$SCRIPT_DIR/discover_gap_watch.py" \
    --address-log "$ADDRESS_LOG" \
    --fault-log "$RUN_ADDRESS_LOG" \
    --allocator-log "$ALLOCATOR_TRACE_LOG" \
    --summary-json "$AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON" \
    --watch-name "$UVM_GAP_WATCH_NAME" \
    --all-classes "$UVM_GAP_WATCH_ALL_CLASSES" \
    --min-bytes "$UVM_GAP_WATCH_MIN_BYTES" \
    --start-line "$AUTO_GAP_WATCH_MAIN_START_LINE" \
    --target-gap "$AUTO_GAP_WATCH_TARGET_GAP" \
    --fallback-to-hottest "$AUTO_GAP_WATCH_FALLBACK_TO_HOTTEST" \
    --no-write-control

  echo "Post-main gap-watch summary: $AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON"
  python3 - <<'PY' "$AUTO_GAP_WATCH_SUMMARY_JSON" "$AUTO_GAP_WATCH_POST_MAIN_SUMMARY_JSON"
import json
import sys

probe_path = sys.argv[1]
main_path = sys.argv[2]

with open(probe_path, "r", encoding="utf-8") as handle:
    probe = json.load(handle)
with open(main_path, "r", encoding="utf-8") as handle:
    main = json.load(handle)

probe_gap = probe.get("selected_gap", {})
main_gap = main.get("selected_gap", {})

same_index = probe_gap.get("gap_index") == main_gap.get("gap_index")
same_start = probe_gap.get("start_hex") == main_gap.get("start_hex")
same_end = probe_gap.get("end_hex") == main_gap.get("end_hex")
same_gap = same_index and same_start and same_end

print("Post-main Gap Watch Discovery")
print(
    f"- probe_gap={probe_gap.get('gap_index')} "
    f"start={probe_gap.get('start_hex')} end={probe_gap.get('end_hex')} "
    f"faults={probe_gap.get('faults')}"
)
print(
    f"- main_gap={main_gap.get('gap_index')} "
    f"start={main_gap.get('start_hex')} end={main_gap.get('end_hex')} "
    f"faults={main_gap.get('faults')}"
)
print(f"- same_gap={same_gap}")
print(f"- same_gap_index={same_index}")
print(f"- same_gap_start={same_start}")
print(f"- same_gap_end={same_end}")
print(f"- probe_fallback_used={probe.get('fallback_used')}")
print(f"- main_fallback_used={main.get('fallback_used')}")
PY
}

print_recent_stats() {
  echo "===== Recent Replayable fault stats ====="
  grep -E "$STATS_LINE_PATTERN" "$RUN_STATS_LOG" | tail -n 20 2>/dev/null || true
}

extract_stat_int() {
  local line="$1"
  local key="$2"

  # Preferred parse path: machine-readable key=value fields.
  local value
  value="$(echo "$line" | sed -n "s/.*${key}=\([0-9][0-9]*\).*/\1/p")"
  if [ -n "$value" ]; then
    echo "$value"
    return 0
  fi

  # Partial fallback for localized stats lines.
  # NOTE: Without key=value fields some counters can be ambiguous/unavailable.
  case "$key" in
    batch_faults)
      echo "$line" | sed -n 's/.*本批次总缺页实例数=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_after_dedup)
      echo "$line" | sed -n 's/.*本批次总缺页实例数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_kv_faults)
      echo "$line" | sed -n 's/.*KV类的总缺页数=\([0-9][0-9]*\).*/\1/p'
      ;;
    batch_kv_duplicates)
      # Localized logs expose the after-dedup value, not duplicate count.
      # Leave this empty so print_delta_stats can derive duplicates.
      true
      ;;
    batch_kv_after_dedup)
      echo "$line" | sed -n 's/.*KV类的总缺页数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_faults)
      echo "$line" | sed -n 's/.*|| 总缺页数=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_after_dedup)
      echo "$line" | sed -n 's/.*|| 总缺页数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_kv_faults)
      echo "$line" | sed -n 's/.*kv总错误数=\([0-9][0-9]*\).*/\1/p'
      ;;
    total_kv_duplicates)
      # Localized logs expose the after-dedup value, not duplicate count.
      # Leave this empty so print_delta_stats can derive duplicates.
      true
      ;;
    total_kv_after_dedup)
      echo "$line" | sed -n 's/.*kv总错误数=[0-9][0-9]*,去重后=\([0-9][0-9]*\).*/\1/p'
      ;;
    *)
      true
      ;;
  esac
}

safe_non_negative_delta() {
  local end="$1"
  local begin="$2"
  if [ "$end" -lt "$begin" ]; then
    echo 0
  else
    echo $(( end - begin ))
  fi
}

calc_ratio_percent() {
  local num="$1"
  local den="$2"
  awk -v n="$num" -v d="$den" 'BEGIN { if (d <= 0) printf "0.00"; else printf "%.2f", (100.0 * n) / d }'
}

print_delta_stats() {
  local stats_file="$RUN_STATS_LOG"
  [ -n "$stats_file" ] || return 0
  [ -s "$stats_file" ] || {
    echo "===== Delta Replayable fault stats (this workload) ====="
    echo "No stats captured in $stats_file"
    return 0
  }

  local first_line last_line
  first_line="$(grep -E "$STATS_LINE_PATTERN" "$stats_file" | head -n 1 || true)"
  last_line="$(grep -E "$STATS_LINE_PATTERN" "$stats_file" | tail -n 1 || true)"
  [ -n "$first_line" ] || return 0
  [ -n "$last_line" ] || return 0

  local first_batch_faults first_batch_duplicates first_batch_after_dedup
  local first_batch_kv_faults first_batch_kv_duplicates first_batch_kv_after_dedup
  local first_total_faults first_total_duplicates first_total_after_dedup
  local first_total_kv_faults first_total_kv_duplicates first_total_kv_after_dedup

  local last_total_faults last_total_duplicates last_total_after_dedup
  local last_total_kv_faults last_total_kv_duplicates last_total_kv_after_dedup

  first_batch_faults="$(extract_stat_int "$first_line" "batch_faults")"
  first_batch_duplicates="$(extract_stat_int "$first_line" "batch_duplicates")"
  first_batch_after_dedup="$(extract_stat_int "$first_line" "batch_after_dedup")"
  first_batch_kv_faults="$(extract_stat_int "$first_line" "batch_kv_faults")"
  first_batch_kv_duplicates="$(extract_stat_int "$first_line" "batch_kv_duplicates")"
  first_batch_kv_after_dedup="$(extract_stat_int "$first_line" "batch_kv_after_dedup")"

  first_total_faults="$(extract_stat_int "$first_line" "total_faults")"
  first_total_duplicates="$(extract_stat_int "$first_line" "total_duplicates")"
  first_total_after_dedup="$(extract_stat_int "$first_line" "total_after_dedup")"
  first_total_kv_faults="$(extract_stat_int "$first_line" "total_kv_faults")"
  first_total_kv_duplicates="$(extract_stat_int "$first_line" "total_kv_duplicates")"
  first_total_kv_after_dedup="$(extract_stat_int "$first_line" "total_kv_after_dedup")"

  last_total_faults="$(extract_stat_int "$last_line" "total_faults")"
  last_total_duplicates="$(extract_stat_int "$last_line" "total_duplicates")"
  last_total_after_dedup="$(extract_stat_int "$last_line" "total_after_dedup")"
  last_total_kv_faults="$(extract_stat_int "$last_line" "total_kv_faults")"
  last_total_kv_duplicates="$(extract_stat_int "$last_line" "total_kv_duplicates")"
  last_total_kv_after_dedup="$(extract_stat_int "$last_line" "total_kv_after_dedup")"

  for v in \
    "$first_batch_faults" "$first_batch_after_dedup" \
    "$first_batch_kv_faults" "$first_batch_kv_after_dedup" \
    "$first_total_faults" "$first_total_after_dedup" \
    "$first_total_kv_faults" "$first_total_kv_after_dedup" \
    "$last_total_faults" "$last_total_after_dedup" \
    "$last_total_kv_faults" "$last_total_kv_after_dedup"
  do
    [ -n "$v" ] || {
      echo "===== Delta Replayable fault stats (this workload) ====="
      echo "Failed to parse stats lines from $stats_file"
      echo "Hint: keep machine-readable key=value fields in driver stats logs (e.g., batch_faults=..., total_faults=...)."
      return 0
    }
  done

  # Derive duplicate counters from fault totals and after-dedup values when the
  # localized stats line does not expose explicit "*_duplicates" fields.
  [ -n "$first_batch_duplicates" ] || first_batch_duplicates=$(( first_batch_faults - first_batch_after_dedup ))
  [ -n "$first_batch_kv_duplicates" ] || first_batch_kv_duplicates=$(( first_batch_kv_faults - first_batch_kv_after_dedup ))
  [ -n "$first_total_duplicates" ] || first_total_duplicates=$(( first_total_faults - first_total_after_dedup ))
  [ -n "$first_total_kv_duplicates" ] || first_total_kv_duplicates=$(( first_total_kv_faults - first_total_kv_after_dedup ))
  [ -n "$last_total_duplicates" ] || last_total_duplicates=$(( last_total_faults - last_total_after_dedup ))
  [ -n "$last_total_kv_duplicates" ] || last_total_kv_duplicates=$(( last_total_kv_faults - last_total_kv_after_dedup ))

  # Baseline at benchmark start:
  # baseline = first_total - first_batch
  local base_total_faults base_total_duplicates base_total_after_dedup
  local base_total_kv_faults base_total_kv_duplicates base_total_kv_after_dedup
  base_total_faults=$(( first_total_faults - first_batch_faults ))
  base_total_duplicates=$(( first_total_duplicates - first_batch_duplicates ))
  base_total_after_dedup=$(( first_total_after_dedup - first_batch_after_dedup ))
  base_total_kv_faults=$(( first_total_kv_faults - first_batch_kv_faults ))
  base_total_kv_duplicates=$(( first_total_kv_duplicates - first_batch_kv_duplicates ))
  base_total_kv_after_dedup=$(( first_total_kv_after_dedup - first_batch_kv_after_dedup ))

  local delta_faults delta_duplicates delta_after_dedup
  local delta_kv_faults delta_kv_duplicates delta_kv_after_dedup
  delta_faults="$(safe_non_negative_delta "$last_total_faults" "$base_total_faults")"
  delta_duplicates="$(safe_non_negative_delta "$last_total_duplicates" "$base_total_duplicates")"
  delta_after_dedup="$(safe_non_negative_delta "$last_total_after_dedup" "$base_total_after_dedup")"
  delta_kv_faults="$(safe_non_negative_delta "$last_total_kv_faults" "$base_total_kv_faults")"
  delta_kv_duplicates="$(safe_non_negative_delta "$last_total_kv_duplicates" "$base_total_kv_duplicates")"
  delta_kv_after_dedup="$(safe_non_negative_delta "$last_total_kv_after_dedup" "$base_total_kv_after_dedup")"

  local delta_kv_ratio delta_kv_after_dedup_ratio
  delta_kv_ratio="$(calc_ratio_percent "$delta_kv_faults" "$delta_faults")"
  delta_kv_after_dedup_ratio="$(calc_ratio_percent "$delta_kv_after_dedup" "$delta_after_dedup")"

  echo "===== Delta Replayable fault stats (this workload) ====="
  echo "formula: baseline = first_total - first_batch; delta = last_total - baseline"
  echo "delta_faults=$delta_faults delta_duplicates=$delta_duplicates delta_after_dedup=$delta_after_dedup"
  echo "delta_kv_faults=$delta_kv_faults delta_kv_duplicates=$delta_kv_duplicates delta_kv_after_dedup=$delta_kv_after_dedup"
  echo "delta_kv_ratio=${delta_kv_ratio}% delta_kv_after_dedup_ratio=${delta_kv_after_dedup_ratio}%"
}

main() {
  parse_args "$@"

  trap cleanup EXIT INT TERM

  if [ "$NO_CLEANUP" -eq 0 ]; then
    python3 "$REPO_ROOT/workloads/cleanup_gpu.py" || true
    sleep 2
  fi

  if [ "$RESET_COUNTERS" -eq 1 ]; then
    reload_uvm_module_for_counter_reset
  fi

  validate_inputs
  : > "$BENCH_LOG"

  if [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ]; then
    write_gap_watch_control_file \
      0 \
      "$UVM_GAP_WATCH_NAME" \
      0x0 \
      0x0 \
      "$UVM_GAP_WATCH_ALL_CLASSES" \
      "$UVM_GAP_WATCH_MIN_BYTES" \
      "$UVM_GAP_WATCH_TARGET_CLASS" \
      "$UVM_GAP_WATCH_POLICY_ACTION"
    echo "Auto gap-watch control file initialized: $UVM_GAP_WATCH_CONTROL_FILE"
  fi

  echo "Starting single-process vLLM server..."
  start_server

  echo "Waiting for server readiness + fresh KV range..."
  wait_for_server_and_kv_range

  echo "KV range (same server process): start=$KV_START end=$KV_END"
  configure_uvm_params

  echo "UVM params configured."
  echo "Stats log: $TRACE_LOG"
  [ "$ENABLE_FAULT_ADDRESS_LOG" -eq 1 ] && echo "Address log: $ADDRESS_TRACE_LOG"

  start_trace_capture_if_needed
  start_dmesg_capture_if_needed

  if [ "$NO_BENCH" -eq 0 ]; then
    echo "Bench log: $BENCH_LOG"
    echo "Benchmark started. Waiting for completion..."

    if [ "$AUTO_GAP_WATCH_ENABLE" -eq 1 ]; then
      echo "Running auto gap-watch probe phase (same server process)..."
      run_benchmark_phase "probe" "$AUTO_GAP_WATCH_PROBE_PROMPTS"
      sleep 1
      run_auto_gap_watch_discovery
      sleep 1
      mark_auto_gap_watch_main_start
      echo "Running auto gap-watch main phase (same server process)..."
      run_benchmark_phase "main" "$PROMPTS"
      sleep 1
      print_post_main_gap_watch_discovery
    else
      run_benchmark_phase "main" "$PROMPTS"
    fi
  else
    echo "--no-bench enabled: server started and UVM params configured, benchmark skipped."
  fi

  stop_dmesg_capture_if_needed
  stop_trace_capture_if_needed

  print_recent_stats
  print_delta_stats
  if [ -n "$ALLOCATOR_TRACE_LOG" ]; then
    echo "Allocator trace log: $ALLOCATOR_TRACE_LOG"
  fi
  if [ -n "$GAP_WATCH_METRICS_SUMMARY_JSON" ] && [ -n "$ALLOCATOR_TRACE_LOG" ]; then
    python3 "$SCRIPT_DIR/summarize_gap_watch_metrics.py" \
      --allocator-log "$ALLOCATOR_TRACE_LOG" \
      --summary-json "$GAP_WATCH_METRICS_SUMMARY_JSON"
    echo "Gap-watch metrics summary: $GAP_WATCH_METRICS_SUMMARY_JSON"
  fi
}

main "$@"
