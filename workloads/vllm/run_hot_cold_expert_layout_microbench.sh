#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-log/current_hot_cold_expert_layout_microbench}"
PYTHON="${PYTHON:-$ROOT/workloads/vllm/.venv/bin/python3}"
mkdir -p "$OUT_ROOT"

FULL_QWEN3_FP8="${FULL_QWEN3_FP8:-0}"
if [ "$FULL_QWEN3_FP8" = "1" ]; then
  NUM_LAYERS="${NUM_LAYERS:-48}"
  SHAPE_NAME="${SHAPE_NAME:-qwen3_fp8}"
  DTYPE="${DTYPE:-uint8}"
else
  NUM_LAYERS="${NUM_LAYERS:-6}"
  SHAPE_NAME="${SHAPE_NAME:-toy}"
  DTYPE="${DTYPE:-uint8}"
fi

NUM_EXPERTS="${NUM_EXPERTS:-128}"
HOT_EXPERTS="${HOT_EXPERTS:-64}"
COLD_EXPERTS="${COLD_EXPERTS:-64}"
SLOT_LAYERS="${SLOT_LAYERS:-2}"
HOT_BACKING="${HOT_BACKING:-auto}"
SLOT_BACKING="${SLOT_BACKING:-cuda}"
TOPK="${TOPK:-8}"
TOUCH_ELEMENTS="${TOUCH_ELEMENTS:-4096}"
THINK_MS="${THINK_MS:-0}"
DEVICE="${DEVICE:-0}"
PIN_MEMORY="${PIN_MEMORY:-1}"
VERIFY="${VERIFY:-1}"
DRY_RUN="${DRY_RUN:-0}"

CMD=(
  "$PYTHON" workloads/vllm/bench_hot_cold_expert_layout.py
  --num-layers "$NUM_LAYERS"
  --num-experts "$NUM_EXPERTS"
  --hot-experts "$HOT_EXPERTS"
  --cold-experts "$COLD_EXPERTS"
  --slot-layers "$SLOT_LAYERS"
  --shape-name "$SHAPE_NAME"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --hot-backing "$HOT_BACKING"
  --slot-backing "$SLOT_BACKING"
  --topk "$TOPK"
  --touch-elements "$TOUCH_ELEMENTS"
  --think-ms "$THINK_MS"
  --out-json "$OUT_ROOT/result.json"
)

if [ "$PIN_MEMORY" = "1" ]; then
  CMD+=(--pin-memory)
fi
if [ "$VERIFY" = "1" ]; then
  CMD+=(--verify)
fi
if [ "$DRY_RUN" = "1" ]; then
  CMD+=(--dry-run)
fi

{
  echo "===== hot/cold explicit expert layout microbench ====="
  echo "out_root=$OUT_ROOT"
  echo "full_qwen3_fp8=$FULL_QWEN3_FP8"
  printf 'cmd='
  printf '%q ' "${CMD[@]}"
  echo
  "${CMD[@]}"
} 2>&1 | tee "$OUT_ROOT/run.log"
