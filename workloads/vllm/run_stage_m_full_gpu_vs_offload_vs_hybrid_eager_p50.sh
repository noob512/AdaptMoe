#!/usr/bin/env bash
# Longer p50 run for comparing:
#   A. native full GPU with small KV
#   B. native pinned-UVA CPU offload
#   C. isolated UVM hybrid eager promotion
#
# Defaults intentionally amplify steady-state decode cost more than the smoke:
#   PROMPTS=50, OUTPUT_LEN=64, REQUEST_RATE=1, MAX_NUM_SEQS=1.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/full_gpu_vs_offload_vs_hybrid_eager_p50_o${OUTPUT_LEN:-64}_$(date -u +%Y%m%d_%H%M%S)}" \
PROMPTS="${PROMPTS:-50}" \
OUTPUT_LEN="${OUTPUT_LEN:-64}" \
REQUEST_RATE="${REQUEST_RATE:-1}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}" \
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}" \
FULL_GPU_KV_CACHE_GB="${FULL_GPU_KV_CACHE_GB:-0.5}" \
"$SCRIPT_DIR/run_stage_m_full_gpu_vs_offload_vs_hybrid_eager.sh"
