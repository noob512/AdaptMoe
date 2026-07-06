#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$ROOT/workloads/vllm/vllm/uvm_test"
BIN="$TEST_DIR/phase_n_3gb_rebalance_smoke"

DEVICE="${DEVICE:-0}"
REGION_MIB="${REGION_MIB:-512}"
EXPERT_REGIONS="${EXPERT_REGIONS:-5}"
KV_REGIONS="${KV_REGIONS:-2}"
ASYNC="${ASYNC:-0}"
RUN_ROOT="${RUN_ROOT:-$ROOT/workloads/vllm/log/stage_n_3gb_rebalance_smoke_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT"
echo "building phase_n_3gb_rebalance_smoke"
make -C "$TEST_DIR" phase_n_3gb_rebalance_smoke

{
  echo "=== nvidia-smi before ==="
  nvidia-smi || true
  echo
  echo "=== phase_n_3gb_rebalance_smoke ==="
  "$BIN" \
    --device "$DEVICE" \
    --region-mib "$REGION_MIB" \
    --expert-regions "$EXPERT_REGIONS" \
    --kv-regions "$KV_REGIONS" \
    --async "$ASYNC"
  echo
  echo "=== nvidia-smi after ==="
  nvidia-smi || true
} 2>&1 | tee "$RUN_ROOT/smoke.log"

if grep -q 'phase_n_3gb_final .*total_ok=1' "$RUN_ROOT/smoke.log"; then
  echo "PASS stage_n_3gb_rebalance_smoke"
  exit 0
fi

echo "FAIL stage_n_3gb_rebalance_smoke"
exit 1
