#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-log/current_hot_cold_expert_checkpoint_microbench}"
PYTHON="${PYTHON:-$ROOT/workloads/vllm/.venv/bin/python3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
mkdir -p "$OUT_ROOT"

LAYERS="${LAYERS:-}"
LAYER_START="${LAYER_START:-0}"
LAYER_COUNT="${LAYER_COUNT:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-128}"
HOT_EXPERTS="${HOT_EXPERTS:-64}"
COLD_EXPERTS="${COLD_EXPERTS:-64}"
STREAM_EXPERTS="${STREAM_EXPERTS:-}"
UVM_EXPERTS="${UVM_EXPERTS:-}"
SLOT_LAYERS="${SLOT_LAYERS:-2}"
HOT_BACKING="${HOT_BACKING:-cuda}"
SLOT_BACKING="${SLOT_BACKING:-cuda}"
UVM_BACKING="${UVM_BACKING:-uvm}"
UVM_MODE="${UVM_MODE:-fault}"
TOPK="${TOPK:-8}"
TOUCH_ELEMENTS="${TOUCH_ELEMENTS:-4096}"
POINTER_CHECKS_PER_LAYER="${POINTER_CHECKS_PER_LAYER:-4}"
POINTER_HIDDEN="${POINTER_HIDDEN:-32}"
POINTER_INTERMEDIATE="${POINTER_INTERMEDIATE:-32}"
POINTER_TOLERANCE="${POINTER_TOLERANCE:-0.01}"
DEVICE="${DEVICE:-0}"
PIN_MEMORY="${PIN_MEMORY:-1}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  "$PYTHON" workloads/vllm/bench_hot_cold_expert_checkpoint.py
  --checkpoint-dir "$CHECKPOINT_DIR"
  --layer-start "$LAYER_START"
  --layer-count "$LAYER_COUNT"
  --num-experts "$NUM_EXPERTS"
  --hot-experts "$HOT_EXPERTS"
  --cold-experts "$COLD_EXPERTS"
  --slot-layers "$SLOT_LAYERS"
  --device "$DEVICE"
  --hot-backing "$HOT_BACKING"
  --slot-backing "$SLOT_BACKING"
  --uvm-backing "$UVM_BACKING"
  --uvm-mode "$UVM_MODE"
  --topk "$TOPK"
  --touch-elements "$TOUCH_ELEMENTS"
  --pointer-checks-per-layer "$POINTER_CHECKS_PER_LAYER"
  --pointer-hidden "$POINTER_HIDDEN"
  --pointer-intermediate "$POINTER_INTERMEDIATE"
  --pointer-tolerance "$POINTER_TOLERANCE"
  --out-json "$OUT_ROOT/result.json"
)

if [ -n "$LAYERS" ]; then
  CMD+=(--layers "$LAYERS")
fi
if [ -n "$STREAM_EXPERTS" ]; then
  CMD+=(--stream-experts "$STREAM_EXPERTS")
fi
if [ -n "$UVM_EXPERTS" ]; then
  CMD+=(--uvm-experts "$UVM_EXPERTS")
fi
if [ "$PIN_MEMORY" = "1" ]; then
  CMD+=(--pin-memory)
fi
if [ "$DRY_RUN" = "1" ]; then
  CMD+=(--dry-run)
fi

{
  echo "===== real-checkpoint hot/cold expert layout microbench ====="
  echo "out_root=$OUT_ROOT"
  echo "checkpoint_dir=$CHECKPOINT_DIR"
  printf 'cmd='
  printf '%q ' "${CMD[@]}"
  echo
  "${CMD[@]}"
} 2>&1 | tee "$OUT_ROOT/run.log"
