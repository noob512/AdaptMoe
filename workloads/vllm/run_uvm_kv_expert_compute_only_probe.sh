#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT/workloads/vllm"
SRC="$BENCH_DIR/bench_uvm_kv_expert_compute_only.cu"
BIN="$BENCH_DIR/bench_uvm_kv_expert_compute_only"

NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
DEVICE="${DEVICE:-0}"
EXPERT_MIB="${EXPERT_MIB:-1024}"
KV_TOTAL_MIB="${KV_TOTAL_MIB:-1536}"
KV_ACTIVE_MIB="${KV_ACTIVE_MIB:-512}"
REPEATS="${REPEATS:-5}"
WARMUPS="${WARMUPS:-2}"
RUN_ROOT="${RUN_ROOT:-$BENCH_DIR/log/uvm_kv_expert_compute_only_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "building $BIN"
"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"

{
  echo "=== nvidia-smi before ==="
  nvidia-smi || true
  echo
  echo "=== benchmark ==="
  "$BIN" \
    --device "$DEVICE" \
    --expert-mib "$EXPERT_MIB" \
    --kv-total-mib "$KV_TOTAL_MIB" \
    --kv-active-mib "$KV_ACTIVE_MIB" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS"
  echo
  echo "=== nvidia-smi after ==="
  nvidia-smi || true
} 2>&1 | tee "$RUN_ROOT/benchmark.log"

echo "LOG=$RUN_ROOT/benchmark.log"
