#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/expert_reserved_layer_bench.cu"
BIN="${EXPERT_RESERVED_LAYER_BENCH_BIN:-/tmp/expert_reserved_layer_bench}"

nvcc -std=c++17 -O2 -I/usr/local/cuda/include "$SRC" -lcuda -o "$BIN"

exec "$BIN" "$@"
