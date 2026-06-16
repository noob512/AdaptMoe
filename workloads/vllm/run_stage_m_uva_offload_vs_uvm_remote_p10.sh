#!/usr/bin/env bash
# Compare vLLM native CPU offload's pinned-host UVA remote access with UVM
# remote-access mechanisms.
#
# uvm_pure_remote matches the readonly microbenchmark definition:
#   1) cudaMemPrefetchAsync(..., cudaCpuDeviceId)
#   2) cudaMemAdviseSetPreferredLocation(CPU)
#   3) cudaMemAdviseSetAccessedBy(GPU)
#
# uvm_default_migration only enables VLLM_USE_UVM and applies no matched
# manifest placement/advise policy.
#
# uvm_all_alloc_default_migration additionally LD_PRELOADs a CUDA allocation
# interceptor so lower-level cudaMalloc/cudaMallocAsync/cuMemAlloc calls are
# redirected to managed memory too.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_uva_vs_uvm_native_remote_p10_run}"
PROMPTS="${PROMPTS:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-8}"
NATIVE_SERVER_GPU_MEMORY_UTILIZATION="${NATIVE_SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
UVM_SERVER_GPU_MEMORY_UTILIZATION="${UVM_SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="${UVM_MATCHED_CPU_OFFLOAD_EXECUTOR:-auto}"
UVM_MATCHED_CPU_OFFLOAD_SYNC="${UVM_MATCHED_CPU_OFFLOAD_SYNC:-1}"
UVM_TRACE_MIN_BYTES="${UVM_TRACE_MIN_BYTES:-1048576}"
LAYER_TIMING_ENABLE="${LAYER_TIMING_ENABLE:-1}"
LAYER_TIMING_SYNC="${LAYER_TIMING_SYNC:-1}"
LAYER_TIMING_PATTERN="${LAYER_TIMING_PATTERN:-model\\.layers\\.[0-9]+$|model\\.layers\\.[0-9]+\\.(self_attn|mlp)$}"
LAYER_TIMING_MIN_MS="${LAYER_TIMING_MIN_MS:-0}"
LAYER_TIMING_MAX_RECORDS="${LAYER_TIMING_MAX_RECORDS:-0}"
PORT_BASE="${PORT_BASE:-8000}"
SEED="${SEED:-42}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-600}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"

RUN_NATIVE="${RUN_NATIVE:-1}"
UVM_CASES="${UVM_CASES:-uvm_pure_remote uvm_default_migration uvm_all_alloc_default_migration}"
ALL_ALLOC_PRELOAD_SO="${ALL_ALLOC_PRELOAD_SO:-$REPO_ROOT/workloads/vllm/vllm/uvm_test/libcudamalloc_managed.so}"
ALL_ALLOC_PRELOAD_DIR="${ALL_ALLOC_PRELOAD_DIR:-$REPO_ROOT/workloads/vllm/vllm/uvm_test}"
PHASE_F_ROUTE_LOG_DELAY_MS="${PHASE_F_ROUTE_LOG_DELAY_MS:-0}"
DISABLE_ACCESS_COUNTER_MIGRATION="${DISABLE_ACCESS_COUNTER_MIGRATION:-1}"
RESTORE_PARAMS="${RESTORE_PARAMS:-1}"

case "$RUN_ROOT" in
  /*) ;;
  *) RUN_ROOT="$REPO_ROOT/$RUN_ROOT" ;;
esac

STATS_PATH="${STATS_PATH:-/proc/driver/nvidia-uvm/latency_stats}"
STATS_RESET_PATH="${STATS_RESET_PATH:-/proc/driver/nvidia-uvm/latency_stats_reset}"
PARAM_ROOT="${PARAM_ROOT:-/sys/module/nvidia_uvm/parameters}"

SUMMARY_JSON="$RUN_ROOT/uva_vs_uvm_native_remote_summary.json"
SUMMARY_CSV="$RUN_ROOT/uva_vs_uvm_native_remote_summary.csv"
SUMMARY_MD="$RUN_ROOT/uva_vs_uvm_native_remote_summary.md"
MANIFEST="${MANIFEST:-$RUN_ROOT/native_cpu_offload/offload_manifest.jsonl}"

case "$MANIFEST" in
  /*) ;;
  *) MANIFEST="$REPO_ROOT/$MANIFEST" ;;
esac

COUNTER_KEYS=(
  remote_first_faults
  remote_first_pages
  remote_map_pages
  sync_fallback_pages
  bg_promote_enqueued_pages
  bg_promote_copied_pages
  bg_promote_committed_pages
  bg_promote_cancelled_pages
  bg_promote_demand_pages_enqueued
  bg_promote_prefetch_pages_enqueued
  bg_promote_demand_pages_promoted
  bg_promote_prefetch_pages_promoted
  remote_prepare_calls
  remote_prepare_pages
  remote_prepare_mapped_pages
)

mkdir -p "$RUN_ROOT"

declare -A OLD_PARAMS=()

cleanup_leftovers() {
  pkill -f "vllm bench serve" >/dev/null 2>&1 || true
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
}

read_param() {
  local name="$1"
  local path="${PARAM_ROOT}/${name}"
  [[ -r "$path" ]] || return 1
  cat "$path"
}

write_param() {
  local name="$1"
  local value="$2"
  local path="${PARAM_ROOT}/${name}"
  [[ -e "$path" ]] || return 0

  if [[ ! -w "$path" && "${EUID}" -ne 0 ]]; then
    echo "$value" | sudo tee "$path" >/dev/null 2>/dev/null || true
  else
    echo "$value" > "$path" 2>/dev/null || true
  fi
}

save_param() {
  local name="$1"
  [[ -n "${OLD_PARAMS[$name]+set}" ]] && return
  local value
  if value="$(read_param "$name" 2>/dev/null)"; then
    OLD_PARAMS["$name"]="$value"
  fi
}

restore_params() {
  [[ "$RESTORE_PARAMS" == "1" ]] || return
  for name in "${!OLD_PARAMS[@]}"; do
    write_param "$name" "${OLD_PARAMS[$name]}" || true
  done
}

trap restore_params EXIT

set_uvm_native_remote_params() {
  for name in \
    uvm_latency_remote_first_enable \
    uvm_latency_bg_promote_enable \
    uvm_phase_f_route_log_delay_ms \
    uvm_perf_access_counter_migration_enable; do
    save_param "$name"
  done

  write_param uvm_latency_remote_first_enable 0
  write_param uvm_latency_bg_promote_enable 0
  write_param uvm_phase_f_route_log_delay_ms "$PHASE_F_ROUTE_LOG_DELAY_MS"

  if [[ "$DISABLE_ACCESS_COUNTER_MIGRATION" == "1" ]]; then
    write_param uvm_perf_access_counter_migration_enable 0
  fi
}

reset_counters() {
  [[ -e "$STATS_RESET_PATH" ]] || return 0
  if [[ ! -w "$STATS_RESET_PATH" && "${EUID}" -ne 0 ]]; then
    echo 1 | sudo tee "$STATS_RESET_PATH" >/dev/null || true
  else
    echo 1 > "$STATS_RESET_PATH" 2>/dev/null || true
  fi
}

snapshot_stats() {
  local out="$1"
  if [[ -r "$STATS_PATH" ]]; then
    cat "$STATS_PATH" > "$out"
  else
    : > "$out"
  fi
}

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before readiness. Log tail:" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$STARTUP_TIMEOUT" ]]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

write_case_metadata() {
  local case_root="$1"
  local case_name="$2"
  local mode="$3"
  local remote_mechanism="uvm_pure_remote"
  local uvm_allocator_scope="pytorch_cuda_allocator"
  local uvm_prefetch_cpu="1"
  local uvm_preferred_location_cpu="1"
  local uvm_accessed_by_gpu="1"
  local uvm_readmostly="0"
  local uvm_remote_first_enable="0"
  local uvm_bg_promote_enable="0"
  local uvm_matched_manifest_enable="1"

  if [[ "$mode" == "native_uva" ]]; then
    remote_mechanism="pinned_host_uva"
    uvm_allocator_scope="none"
    uvm_prefetch_cpu="0"
    uvm_preferred_location_cpu="0"
    uvm_accessed_by_gpu="0"
    uvm_readmostly="0"
    uvm_remote_first_enable="0"
    uvm_bg_promote_enable="0"
    uvm_matched_manifest_enable="0"
  elif [[ "$mode" == "default_uvm_migration" ]]; then
    remote_mechanism="default_uvm_demand_migration"
    uvm_allocator_scope="pytorch_cuda_allocator"
    uvm_prefetch_cpu="0"
    uvm_preferred_location_cpu="0"
    uvm_accessed_by_gpu="0"
    uvm_readmostly="0"
    uvm_remote_first_enable="0"
    uvm_bg_promote_enable="0"
    uvm_matched_manifest_enable="0"
  elif [[ "$mode" == "all_alloc_default_uvm_migration" ]]; then
    remote_mechanism="all_alloc_default_uvm_demand_migration"
    uvm_allocator_scope="pytorch_cuda_allocator_plus_ldpreload_cuda_alloc"
    uvm_prefetch_cpu="0"
    uvm_preferred_location_cpu="0"
    uvm_accessed_by_gpu="0"
    uvm_readmostly="0"
    uvm_remote_first_enable="0"
    uvm_bg_promote_enable="0"
    uvm_matched_manifest_enable="0"
  fi

  {
    echo "case=${case_name}"
    echo "matched_mode=${mode}"
    echo "uvm_remote_mechanism=${remote_mechanism}"
    echo "uvm_allocator_scope=${uvm_allocator_scope}"
    echo "uvm_prefetch_cpu=${uvm_prefetch_cpu}"
    echo "uvm_preferred_location_cpu=${uvm_preferred_location_cpu}"
    echo "uvm_accessed_by_gpu=${uvm_accessed_by_gpu}"
    echo "uvm_readmostly=${uvm_readmostly}"
    echo "uvm_remote_first_enable=${uvm_remote_first_enable}"
    echo "uvm_bg_promote_enable=${uvm_bg_promote_enable}"
    echo "uvm_matched_manifest_enable=${uvm_matched_manifest_enable}"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model=${MODEL}"
    echo "prompts=${PROMPTS}"
    echo "output_len=${OUTPUT_LEN}"
    echo "request_rate=${REQUEST_RATE}"
    echo "server_max_num_seqs=${SERVER_MAX_NUM_SEQS}"
    echo "max_model_len=${MAX_MODEL_LEN}"
    echo "uvm_executor=${UVM_MATCHED_CPU_OFFLOAD_EXECUTOR}"
    echo "uvm_sync=${UVM_MATCHED_CPU_OFFLOAD_SYNC}"
    echo "all_alloc_preload_so=${ALL_ALLOC_PRELOAD_SO}"
    echo "disable_access_counter_migration=${DISABLE_ACCESS_COUNTER_MIGRATION}"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    for p in \
      "$PARAM_ROOT/uvm_latency_remote_first_enable" \
      "$PARAM_ROOT/uvm_latency_bg_promote_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_migration_enable" \
      "$PARAM_ROOT/uvm_perf_access_counter_threshold" \
      "$PARAM_ROOT/uvm_ats_mode"; do
      [[ -r "$p" ]] && printf '%s=%s\n' "$(basename "$p")" "$(cat "$p")"
    done
  } > "$case_root/metadata.txt"
}

run_native_offload_case() {
  local case_root="$RUN_ROOT/native_cpu_offload"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local run_log="$case_root/run.log"
  local layer_timing_log="$case_root/layer_timing.jsonl"
  local port="$PORT_BASE"
  local server_pid=""
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$server_log"
  : > "$bench_log"
  : > "$layer_timing_log"
  : > "$MANIFEST"
  write_case_metadata "$case_root" native_cpu_offload native_uva

  echo "==========================================================="
  echo "Run native_cpu_offload: pinned CPU tensor + UVA remote view"
  echo "==========================================================="

  set_uvm_native_remote_params
  reset_counters
  snapshot_stats "$case_root/latency_stats_before.txt"
  cleanup_leftovers
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >>"$run_log" 2>&1 || true
  sleep 2

  (
    cd "$SCRIPT_DIR" || exit 1
    VLLM_CPU_OFFLOAD_MANIFEST_FILE="$MANIFEST" \
    VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
    VLLM_LAYER_TIMING_ENABLE="$LAYER_TIMING_ENABLE" \
    VLLM_LAYER_TIMING_FILE="$layer_timing_log" \
    VLLM_LAYER_TIMING_SYNC="$LAYER_TIMING_SYNC" \
    VLLM_LAYER_TIMING_PATTERN="$LAYER_TIMING_PATTERN" \
    VLLM_LAYER_TIMING_MIN_MS="$LAYER_TIMING_MIN_MS" \
    VLLM_LAYER_TIMING_MAX_RECORDS="$LAYER_TIMING_MAX_RECORDS" \
    uv run vllm serve "$MODEL" \
      --enforce-eager \
      --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$NATIVE_SERVER_GPU_MEMORY_UTILIZATION" \
      --cpu-offload-gb "$NATIVE_CPU_OFFLOAD_GB" \
      --port "$port" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    {
      echo "===== Benchmark phase: main ====="
      echo "command: uv run --directory $SCRIPT_DIR vllm bench serve --model $MODEL --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts $PROMPTS --sharegpt-output-len $OUTPUT_LEN --seed $SEED --request-rate $REQUEST_RATE --port $port"
    } | tee -a "$bench_log" >/dev/null
    set +e
    uv run --directory "$SCRIPT_DIR" vllm bench serve \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$PROMPTS" \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      --request-rate "$REQUEST_RATE" \
      --port "$port" 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
  else
    status=1
  fi

  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 2
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_leftovers
  snapshot_stats "$case_root/latency_stats_after.txt"
  echo "$status" > "$case_root/status.txt"
}

ensure_all_alloc_preload() {
  [[ -s "$ALL_ALLOC_PRELOAD_SO" ]] && return 0
  echo "Building all-allocation CUDA managed-memory preload library..."
  make -C "$ALL_ALLOC_PRELOAD_DIR" preload
  [[ -s "$ALL_ALLOC_PRELOAD_SO" ]]
}

case_mode() {
  case "$1" in
    uvm_pure_remote|uvm_native_remote_access|native_remote_access) echo native_remote_access ;;
    uvm_default|default_uvm|uvm_default_migration|default_uvm_migration) echo default_uvm_migration ;;
    uvm_all_alloc_default_migration|all_alloc_default_uvm_migration) echo all_alloc_default_uvm_migration ;;
    *)
      echo "ERROR: unsupported UVM case '$1'. Use uvm_pure_remote, uvm_default_migration, or uvm_all_alloc_default_migration." >&2
      return 1
      ;;
  esac
}

run_uvm_case() {
  local case_name="$1"
  local case_index="$2"
  local mode
  mode="$(case_mode "$case_name")" || exit 1

  local case_root="$RUN_ROOT/$case_name"
  local run_log="$case_root/run.log"
  local server_log="$case_root/vllm_server.log"
  local bench_log="$case_root/vllm_bench.log"
  local stats_log="$case_root/uvm_fault_stats.log"
  local addrs_log="$case_root/uvm_fault_addrs.log"
  local address_log="$case_root/vllm_uvm_address_regions.log"
  local allocator_log="$case_root/vllm_uvm_allocator_trace.log"
  local matched_trace="$case_root/uvm_matched_cpu_offload_trace.jsonl"
  local layer_timing_log="$case_root/layer_timing.jsonl"
  local port=$((PORT_BASE + case_index))
  local status=0

  mkdir -p "$case_root"
  : > "$run_log"
  : > "$matched_trace"
  : > "$stats_log"
  : > "$bench_log"
  : > "$server_log"
  : > "$allocator_log"
  : > "$address_log"
  : > "$addrs_log"
  : > "$layer_timing_log"
  write_case_metadata "$case_root" "$case_name" "$mode"

  echo "==========================================================="
  if [[ "$mode" == "default_uvm_migration" ]]; then
    echo "Run $case_name: PyTorch allocator UVM, default demand migration"
  elif [[ "$mode" == "all_alloc_default_uvm_migration" ]]; then
    echo "Run $case_name: all CUDA allocations UVM, default demand migration"
  else
    echo "Run $case_name: same manifest, UVM CPU Preferred + AccessedBy remote"
  fi
  echo "==========================================================="

  set_uvm_native_remote_params
  reset_counters
  snapshot_stats "$case_root/latency_stats_before.txt"
  cleanup_leftovers
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >>"$run_log" 2>&1 || true
  sleep 2

  local server_pid=""
  local matched_enable="1"
  local matched_env_mode="$mode"

  if [[ "$mode" == "default_uvm_migration" || "$mode" == "all_alloc_default_uvm_migration" ]]; then
    matched_enable="0"
    matched_env_mode="trace_only"
    rm -f "$matched_trace"
  fi

  local preload_env="${LD_PRELOAD:-}"
  if [[ "$mode" == "all_alloc_default_uvm_migration" ]]; then
    if ! ensure_all_alloc_preload >>"$run_log" 2>&1; then
      echo "ERROR: failed to build or find $ALL_ALLOC_PRELOAD_SO" | tee -a "$run_log" >&2
      echo "1" > "$case_root/status.txt"
      return
    fi
    if [[ -n "$preload_env" ]]; then
      preload_env="$ALL_ALLOC_PRELOAD_SO:$preload_env"
    else
      preload_env="$ALL_ALLOC_PRELOAD_SO"
    fi
  fi

  (
    cd "$SCRIPT_DIR" || exit 1
    export LD_PRELOAD="$preload_env"
    if [[ "$mode" == "all_alloc_default_uvm_migration" ]]; then
      export CUDA_MANAGED_VERBOSE="${CUDA_MANAGED_VERBOSE:-1}"
    fi
    VLLM_USE_UVM=1 \
    VLLM_UVM_ADDRESS_LOG_ENABLE=1 \
    VLLM_UVM_ADDRESS_LOG_FILE="$address_log" \
    VLLM_UVM_LOG_FILE="$allocator_log" \
    VLLM_UVM_TRACE_MIN_BYTES="$UVM_TRACE_MIN_BYTES" \
    VLLM_UVM_POLICY_ENABLE=0 \
    VLLM_UVM_POLICY_RUNTIME_ENABLE=0 \
    VLLM_UVM_DEVICE_DIRECT_ENABLE=0 \
    VLLM_UVM_RESIDENCY_ENABLE=0 \
    VLLM_UVM_KV_RUNTIME_ENABLE=0 \
    VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE=0 \
    VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE=1 \
    VLLM_UVM_WEIGHT_EXECUTOR="$UVM_MATCHED_CPU_OFFLOAD_EXECUTOR" \
    VLLM_UVM_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=0 \
    VLLM_UVM_POOL_COORDINATOR_ENABLE=0 \
    VLLM_UVM_POOL_REGISTRY_ENABLE=0 \
    VLLM_UVM_SCRATCH_POOL_ENABLE=0 \
    VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0 \
    VLLM_UVM_WEIGHT_MAP_ENABLE=0 \
    VLLM_UVM_UNKNOWN_DETAIL_ENABLE=0 \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE="$matched_enable" \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE="$MANIFEST" \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_TRACE_FILE="$matched_trace" \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_MODE="$matched_env_mode" \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="cpp_batch" \
    VLLM_UVM_MATCHED_CPU_OFFLOAD_SYNC="$UVM_MATCHED_CPU_OFFLOAD_SYNC" \
    VLLM_LAYER_TIMING_ENABLE="$LAYER_TIMING_ENABLE" \
    VLLM_LAYER_TIMING_FILE="$layer_timing_log" \
    VLLM_LAYER_TIMING_SYNC="$LAYER_TIMING_SYNC" \
    VLLM_LAYER_TIMING_PATTERN="$LAYER_TIMING_PATTERN" \
    VLLM_LAYER_TIMING_MIN_MS="$LAYER_TIMING_MIN_MS" \
    VLLM_LAYER_TIMING_MAX_RECORDS="$LAYER_TIMING_MAX_RECORDS" \
    uv run vllm serve "$MODEL" \
      --enforce-eager \
      --max-num-seqs "$SERVER_MAX_NUM_SEQS" \
      --max-model-len "$MAX_MODEL_LEN" \
      --gpu-memory-utilization "$UVM_SERVER_GPU_MEMORY_UTILIZATION" \
      --port "$port" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
    {
      echo "===== Benchmark phase: main ====="
      echo "command: uv run --directory $SCRIPT_DIR vllm bench serve --model $MODEL --dataset-name sharegpt --dataset-path $DATASET_PATH --num-prompts $PROMPTS --sharegpt-output-len $OUTPUT_LEN --seed $SEED --request-rate $REQUEST_RATE --port $port"
    } | tee -a "$bench_log" >/dev/null
    set +e
    uv run --directory "$SCRIPT_DIR" vllm bench serve \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$PROMPTS" \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      --request-rate "$REQUEST_RATE" \
      --port "$port" 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
  else
    status=1
  fi

  if kill -0 "$server_pid" 2>/dev/null; then
    kill -TERM "$server_pid" >/dev/null 2>&1 || true
    sleep 2
    kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  echo "$status" > "$case_root/status.txt"
  cleanup_leftovers
  snapshot_stats "$case_root/latency_stats_after.txt"
}

summarize() {
  python3 - "$RUN_ROOT" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$MANIFEST" "$UVM_CASES" "${COUNTER_KEYS[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
manifest_path = Path(sys.argv[5])
uvm_cases = sys.argv[6].split()
counter_keys = sys.argv[7:]

bench_patterns = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}
fault_patterns = {
    "delta_faults": r"\bdelta_faults=([0-9]+)",
    "delta_after_dedup": r"\bdelta_after_dedup=([0-9]+)",
    "delta_kv_faults": r"\bdelta_kv_faults=([0-9]+)",
    "delta_kv_after_dedup": r"\bdelta_kv_after_dedup=([0-9]+)",
}

def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def ratio(new, base):
    if new is None or base in (None, 0):
        return None
    return new / base

def pct(new, base):
    if new is None or base in (None, 0):
        return None
    return 100.0 * (new - base) / base

def first_float(s: str, pattern: str):
    m = re.search(pattern, s)
    return float(m.group(1)) if m else None

def first_int(s: str, pattern: str):
    m = re.search(pattern, s)
    return int(m.group(1)) if m else None

def status(path: Path):
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def parse_kv_file(path: Path):
    out = {}
    for line in text(path).splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = int(parts[1], 0)
            except Exception:
                pass
    return out

def counter_deltas(case_root: Path):
    before = parse_kv_file(case_root / "latency_stats_before.txt")
    after = parse_kv_file(case_root / "latency_stats_after.txt")
    return {
        f"{key}_delta": after.get(key, 0) - before.get(key, 0)
        for key in counter_keys
    }

def metadata(path: Path):
    data = {}
    for line in text(path).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k] = v
    return data

def manifest_stats(path: Path):
    count = 0
    total = 0
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") != "cpu_offload_parameter":
            continue
        count += 1
        total += int(rec.get("bytes") or 0)
    return count, total

def matched_trace_stats(path: Path):
    if not path.exists():
        return {
            "uvm_matched_parameter_count": None,
            "uvm_matched_bytes": None,
            "uvm_matched_batch_successes": None,
            "uvm_matched_missing_count": None,
            "native_remote_prepare_successes": None,
            "native_remote_accessed_by_result": None,
            "native_remote_prefetch_cpu_successes": None,
            "native_remote_preferred_cpu_result": None,
            "native_remote_driver_prefetch_result": None,
            "native_remote_preferred_location_cpu": None,
            "native_remote_readmostly": None,
        }
    names = set()
    attempted_bytes = 0
    batch_successes = None
    missing = 0
    native_prepare = {}
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        event = rec.get("event")
        if event == "matched_cpu_offload_parameter":
            name = rec.get("parameter_name")
            if isinstance(name, str):
                names.add(name)
            attempted_bytes += int(rec.get("bytes") or 0)
            if batch_successes is None and "batch_successes" in rec:
                batch_successes = rec.get("batch_successes")
        elif event == "matched_cpu_offload_missing_parameter":
            missing += 1
        elif event == "matched_cpu_offload_native_remote_prepare":
            native_prepare = rec
    return {
        "uvm_matched_parameter_count": len(names),
        "uvm_matched_bytes": attempted_bytes,
        "uvm_matched_batch_successes": batch_successes,
        "uvm_matched_missing_count": missing,
        "native_remote_prepare_successes": native_prepare.get("successes"),
        "native_remote_accessed_by_result": native_prepare.get("accessed_by_result"),
        "native_remote_prefetch_cpu_successes": native_prepare.get("prefetch_cpu_successes"),
        "native_remote_preferred_cpu_result": native_prepare.get("preferred_cpu_result"),
        "native_remote_driver_prefetch_result": native_prepare.get("driver_prefetch_result"),
        "native_remote_preferred_location_cpu": native_prepare.get("preferred_location_cpu"),
        "native_remote_readmostly": native_prepare.get("readmostly"),
    }

manifest_count, manifest_bytes = manifest_stats(manifest_path)

wanted_cases = ["native_cpu_offload", *uvm_cases]
case_dirs = [
    root / case
    for case in wanted_cases
    if (root / case).is_dir() and (root / case / "status.txt").exists()
]

rows = []
for case_root in case_dirs:
    bench = text(case_root / "vllm_bench.log")
    run_log = text(case_root / "run.log")
    meta = metadata(case_root / "metadata.txt")
    row = {
        "case": case_root.name,
        "matched_mode": meta.get("matched_mode"),
        "uvm_remote_mechanism": meta.get("uvm_remote_mechanism"),
        "uvm_allocator_scope": meta.get("uvm_allocator_scope"),
        "uvm_preferred_location_cpu": meta.get("uvm_preferred_location_cpu"),
        "uvm_accessed_by_gpu": meta.get("uvm_accessed_by_gpu"),
        "uvm_matched_manifest_enable": meta.get("uvm_matched_manifest_enable"),
        "uvm_readmostly": meta.get("uvm_readmostly"),
        "uvm_remote_first_enable": meta.get("uvm_remote_first_enable"),
        "exit_code": status(case_root / "status.txt"),
        "manifest_parameter_count": manifest_count,
        "manifest_bytes": manifest_bytes,
    }
    for key, pat in bench_patterns.items():
        row[key] = first_float(bench, pat)
    for key, pat in fault_patterns.items():
        row[key] = first_int(run_log, pat)
    row.update(counter_deltas(case_root))
    row.update(matched_trace_stats(case_root / "uvm_matched_cpu_offload_trace.jsonl"))
    rows.append(row)

native = next((r for r in rows if r["case"] == "native_cpu_offload"), None)
for row in rows:
    if native:
        row["output_tok_s_ratio_vs_native"] = ratio(row.get("output_tok_s"), native.get("output_tok_s"))
        row["mean_tpot_ratio_vs_native"] = ratio(row.get("mean_tpot_ms"), native.get("mean_tpot_ms"))
        row["mean_ttft_ratio_vs_native"] = ratio(row.get("mean_ttft_ms"), native.get("mean_ttft_ms"))
        row["output_tok_s_pct_vs_native"] = pct(row.get("output_tok_s"), native.get("output_tok_s"))
        row["mean_tpot_pct_vs_native"] = pct(row.get("mean_tpot_ms"), native.get("mean_tpot_ms"))

fields = [
    "case", "matched_mode", "uvm_remote_mechanism", "uvm_allocator_scope",
    "uvm_preferred_location_cpu", "uvm_accessed_by_gpu",
    "uvm_matched_manifest_enable", "uvm_readmostly", "uvm_remote_first_enable",
    "exit_code", "successful_requests", "failed_requests", "output_tok_s",
    "mean_tpot_ms", "mean_ttft_ms", "mean_itl_ms",
    "output_tok_s_ratio_vs_native", "mean_tpot_ratio_vs_native",
    "output_tok_s_pct_vs_native", "mean_tpot_pct_vs_native",
    "delta_faults", "delta_after_dedup", "delta_kv_faults",
    "remote_first_faults_delta", "remote_first_pages_delta",
    "remote_map_pages_delta", "sync_fallback_pages_delta",
    "manifest_parameter_count", "manifest_bytes",
    "uvm_matched_parameter_count", "uvm_matched_batch_successes",
    "uvm_matched_bytes", "uvm_matched_missing_count",
    "native_remote_prepare_successes", "native_remote_accessed_by_result",
    "native_remote_prefetch_cpu_successes",
    "native_remote_preferred_cpu_result",
    "native_remote_driver_prefetch_result",
    "native_remote_preferred_location_cpu",
    "native_remote_readmostly",
]
with summary_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k) for k in fields})

summary_json.write_text(json.dumps({"run_root": str(root), "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

lines = [
    "# vLLM UVA Offload vs UVM Remote Access",
    "",
    f"- run_root: `{root}`",
    f"- native_manifest: `{manifest_path}`",
    f"- manifest_parameters: `{manifest_count}`",
    f"- manifest_bytes: `{manifest_bytes}`",
    "",
    "## Performance",
    "",
    "| case | mode | allocator scope | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | matched | CPU prefetch | CPU preferred | AccessedBy | remote-first pages |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    matched = ""
    if row.get("uvm_matched_parameter_count") is not None:
        matched = f"{row.get('uvm_matched_parameter_count')}/{row.get('manifest_parameter_count')}"
    lines.append(
        "| {case} | {mode} | {scope} | {tok} | {tpot} | {ttft} | {tok_ratio} | {tpot_ratio} | "
        "{matched} | {prefetch_cpu} | {preferred_cpu} | {accessed_by} | {remote_pages} |".format(
            case=row.get("case"),
            mode=row.get("matched_mode"),
            scope=row.get("uvm_allocator_scope") or "",
            tok=fmt(row.get("output_tok_s")),
            tpot=fmt(row.get("mean_tpot_ms")),
            ttft=fmt(row.get("mean_ttft_ms")),
            tok_ratio=fmt(row.get("output_tok_s_ratio_vs_native")),
            tpot_ratio=fmt(row.get("mean_tpot_ratio_vs_native")),
            matched=matched,
            prefetch_cpu=row.get("native_remote_prefetch_cpu_successes") or "",
            preferred_cpu=row.get("native_remote_preferred_cpu_result") or "",
            accessed_by=row.get("native_remote_accessed_by_result") or "",
            remote_pages=row.get("remote_first_pages_delta"),
        )
    )

lines.extend([
    "",
    "## Reading The Result",
    "",
    "- `native_cpu_offload` is the vLLM pinned-host UVA remote-access baseline.",
    "- `uvm_pure_remote` uses the same parameter manifest and applies native UVM `CPU prefetch + PreferredLocation(CPU) + AccessedBy(GPU)`, matching the readonly microbenchmark's pure-remote mechanism.",
    "- `uvm_default_migration` enables the PyTorch CUDA UVM allocator and applies no matched manifest placement or advise policy; it is the default UVM demand-migration control.",
    "- `uvm_all_alloc_default_migration` also LD_PRELOADs a CUDA allocator interceptor so cudaMalloc/cudaMallocAsync/cuMemAlloc are redirected to managed memory where interposition works.",
    "- No UVM case uses ReadMostly, modified-driver remote-first, background promotion, or UVM CPU-offload controls.",
    "- `remote_first_*` counters are expected to stay near zero here; these cases are not testing the modified remote-first path.",
    "- Treat performance as matched only if TPOT and output tok/s are within about 5% of native with zero failed requests.",
])
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY
}

main() {
  echo "==========================================================="
  echo "vLLM UVA offload vs UVM pure/default remote access"
  echo "run_root=$RUN_ROOT"
  echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
  echo "uvm_cases=$UVM_CASES"
  echo "==========================================================="

  if [[ "$RUN_NATIVE" == "1" ]]; then
    run_native_offload_case
  fi

  if [[ ! -s "$MANIFEST" ]]; then
    echo "ERROR: missing or empty manifest: $MANIFEST" >&2
    echo "Run with RUN_NATIVE=1 first, or point MANIFEST to an existing native offload manifest." >&2
    exit 1
  fi

  local index=1
  for case_name in $UVM_CASES; do
    run_uvm_case "$case_name" "$index"
    index=$((index + 1))
  done

  summarize
  echo "summary_csv: $SUMMARY_CSV"
  echo "summary_md: $SUMMARY_MD"
}

main "$@"
