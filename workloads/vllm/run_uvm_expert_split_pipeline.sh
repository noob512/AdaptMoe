#!/usr/bin/env bash
# Compare serialized UVM expert prefetch with split prefetch/compute overlap.
#
# The benchmark models the vLLM fused-MoE execution shape: one grouped W13
# phase over all active experts, activation, then one grouped W2 phase over the
# same active experts. It does not launch one CUDA kernel per expert.
#
# Default scenario:
#   - 10 experts
#   - each expert is 4 MiB, split into upper/lower 2 MiB halves
#   - compute touches all experts' upper/W13 half first, then all experts'
#     activation/quant workspace, then all experts' lower/W2 half
#   - split_overlap prefetches upper synchronously, then prefetches lower while
#     the grouped upper/W13 and activation/quant phases run; if lower is late,
#     the grouped lower/W2 phase falls back to UVM replayable fault handling.
#
# To exercise the modified UVM expert-prefetch handoff path:
#   1. Load the modified nvidia-uvm module.
#   2. Enable uvm_phase_n_expert_prefetch_wait_enable=1, and optionally tune
#      uvm_phase_n_expert_prefetch_wait_timeout_ms.
#   3. Run with TAG_EXPERT_REGIONS=1 MODE=compare_handoff.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/uvm_expert_split_pipeline_$(date -u +%Y%m%d_%H%M%S)}"
NVCC="${NVCC:-nvcc}"
DEVICE="${DEVICE:-0}"
EXPERTS="${EXPERTS:-10}"
EXPERT_MIB="${EXPERT_MIB:-4}"
REPEATS="${REPEATS:-5}"
WARMUPS="${WARMUPS:-1}"
UPPER_US="${UPPER_US:-350}"
ACTIVATION_US="${ACTIVATION_US:-50}"
LOWER_US="${LOWER_US:-350}"
BLOCKS_PER_SM="${BLOCKS_PER_SM:-1}"
MODE="${MODE:-all}"
ADVISE_READ_MOSTLY="${ADVISE_READ_MOSTLY:-0}"
TAG_EXPERT_REGIONS="${TAG_EXPERT_REGIONS:-0}"

SRC="$SCRIPT_DIR/bench_uvm_expert_split_pipeline.cu"
BIN="$RUN_ROOT/bench_uvm_expert_split_pipeline"
OUT="$RUN_ROOT/results.txt"

mkdir -p "$RUN_ROOT"

"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"

ARGS=(
  --device "$DEVICE"
  --experts "$EXPERTS"
  --expert-mib "$EXPERT_MIB"
  --repeats "$REPEATS"
  --warmups "$WARMUPS"
  --upper-us "$UPPER_US"
  --activation-us "$ACTIVATION_US"
  --lower-us "$LOWER_US"
  --blocks-per-sm "$BLOCKS_PER_SM"
  --mode "$MODE"
)

if [[ "$ADVISE_READ_MOSTLY" == "1" ]]; then
  ARGS+=(--advise-read-mostly)
fi

if [[ "$TAG_EXPERT_REGIONS" == "1" ]]; then
  ARGS+=(--tag-expert-regions)
fi

echo "tag_expert_regions=$TAG_EXPERT_REGIONS mode=$MODE"
for p in \
  /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_prefetch_wait_enable \
  /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_prefetch_wait_timeout_ms; do
  if [[ -r "$p" ]]; then
    echo "$(basename "$p")=$(cat "$p")"
  fi
done

"$BIN" "${ARGS[@]}" | tee "$OUT"

echo "results: $OUT"
