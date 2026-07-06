#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="$ROOT/workloads/vllm"
SRC="$BENCH_DIR/bench_native_vs_uvm_expert_prefetch.cu"
BIN="$BENCH_DIR/bench_native_vs_uvm_expert_prefetch"

NVCC="${NVCC:-/usr/local/cuda-12.8/bin/nvcc}"
DEVICE="${DEVICE:-0}"
EXPERT_MIB="${EXPERT_MIB:-8192}"
KV_MIB="${KV_MIB:-512}"
REPEATS="${REPEATS:-3}"
WARMUPS="${WARMUPS:-1}"
RUN_ROOT="${RUN_ROOT:-$BENCH_DIR/log/native_vs_uvm_expert_prefetch_$(date +%Y%m%d_%H%M%S)}"

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
    --kv-mib "$KV_MIB" \
    --repeats "$REPEATS" \
    --warmups "$WARMUPS"
  echo
  echo "=== nvidia-smi after ==="
  nvidia-smi || true
} 2>&1 | tee "$RUN_ROOT/benchmark.log"

echo "LOG=$RUN_ROOT/benchmark.log"
