#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-log/stage_m_moe_slot_pool_state_sanity}"
BASE_PORT="${BASE_PORT:-8098}"
PROMPTS="${PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-4}"
REQUEST_RATE="${REQUEST_RATE:-2}"
COLD_SLOT_POOL_DRIVER_MODE="${COLD_SLOT_POOL_DRIVER_MODE:-validate}"
COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED="${COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED:-0}"

mkdir -p "$RUN_ROOT"

echo "== UVM MoE slot remap ioctl dry-run smoke =="
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode dry_run \
  --count 2 \
  --out-json "$RUN_ROOT/moe_slot_remap_ioctl_dry_run.json"

echo
echo "== UVM MoE slot remap ioctl copy-only smoke =="
python3 workloads/vllm/check_moe_slot_remap_ioctl_smoke.py \
  --mode copy_only \
  --count 2 \
  --out-json "$RUN_ROOT/moe_slot_remap_ioctl_copy_only.json"

echo
echo "== UVM MoE slot pool state smoke =="
python3 workloads/vllm/check_moe_slot_pool_state_smoke.py \
  --out-json "$RUN_ROOT/moe_slot_pool_state_smoke.json"

echo
echo "== vLLM driver-slot metadata sanity =="
RUN_ROOT="$RUN_ROOT/vllm_driver_iface" \
BASE_PORT="$BASE_PORT" \
PROMPTS="$PROMPTS" \
OUTPUT_LEN="$OUTPUT_LEN" \
REQUEST_RATE="$REQUEST_RATE" \
COLD_SLOT_POOL_DRIVER_MODE="$COLD_SLOT_POOL_DRIVER_MODE" \
COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED="$COLD_SLOT_POOL_DRIVER_VALIDATE_MANAGED" \
workloads/vllm/run_stage_m_cold_slot_pool_driver_iface_sanity.sh

echo
echo "Slot pool state sanity artifacts:"
find "$RUN_ROOT" -maxdepth 4 \
  \( -name 'moe_slot_*json' -o -name 'cold_slot_pool_report.md' -o -name 'cold_streaming_vs_fault_only_summary.md' \) \
  -print | sort
