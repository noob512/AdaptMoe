#!/usr/bin/env bash
# Stage M three-layer diagnosis:
#   1. Macro serving metrics: tok/s, TPOT, TTFT, prefill/decode-derived fields.
#   2. UVM fault migration metrics: raw/dedup counters plus address classification.
#   3. Layer timing: model.layers.N plus self_attn/mlp forward timing.
#
# This is a thin wrapper around run_stage_m_matched_uvm_vs_vllm_offload_p10.sh
# so the matched native/pure-UVM/matched-UVM cases stay comparable.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ROOT="${RUN_ROOT:-log/stage_m_three_layer_diagnosis_p10_run}"
export PROMPTS="${PROMPTS:-10}"
export OUTPUT_LEN="${OUTPUT_LEN:-64}"
export REQUEST_RATE="${REQUEST_RATE:-1}"
export SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export NATIVE_CPU_OFFLOAD_GB="${NATIVE_CPU_OFFLOAD_GB:-8}"
export NATIVE_SERVER_GPU_MEMORY_UTILIZATION="${NATIVE_SERVER_GPU_MEMORY_UTILIZATION:-0.95}"
export UVM_SERVER_GPU_MEMORY_UTILIZATION="${UVM_SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
export UVM_MATCHED_CPU_OFFLOAD_MODE="${UVM_MATCHED_CPU_OFFLOAD_MODE:-offload_cpu}"
export UVM_MATCHED_CPU_OFFLOAD_EXECUTOR="${UVM_MATCHED_CPU_OFFLOAD_EXECUTOR:-auto}"
export UVM_MATCHED_CPU_OFFLOAD_SYNC="${UVM_MATCHED_CPU_OFFLOAD_SYNC:-1}"

export LAYER_TIMING_ENABLE="${LAYER_TIMING_ENABLE:-1}"
export LAYER_TIMING_SYNC="${LAYER_TIMING_SYNC:-1}"
export LAYER_TIMING_PATTERN="${LAYER_TIMING_PATTERN:-model\\.layers\\.[0-9]+$|model\\.layers\\.[0-9]+\\.(self_attn|mlp)$}"
export LAYER_TIMING_MIN_MS="${LAYER_TIMING_MIN_MS:-0}"
export LAYER_TIMING_MAX_RECORDS="${LAYER_TIMING_MAX_RECORDS:-0}"

exec "$SCRIPT_DIR/run_stage_m_matched_uvm_vs_vllm_offload_p10.sh"
