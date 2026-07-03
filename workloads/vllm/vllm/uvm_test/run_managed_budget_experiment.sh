#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/log/managed_budget_$(date -u +%Y%m%d_%H%M%S)}"

DEVICE="${DEVICE:-0}"
POOL_MB="${POOL_MB:-128}"
HBM_RESERVE_MB="${HBM_RESERVE_MB:-128}"
ALLOC_A_MB="${ALLOC_A_MB:-64}"
ALLOC_B_MB="${ALLOC_B_MB:-80}"
TOUCH_GPU="${TOUCH_GPU:-1}"
PREFETCH_GPU="${PREFETCH_GPU:-0}"

mkdir -p "$LOG_DIR"

make -C "$SCRIPT_DIR" libuvm_managed_budget.so managed_pool_probe

run_case() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  echo "== $name =="
  set +e
  "$@" >"$log" 2>&1
  local rc=$?
  set -e
  echo "exit_code=$rc log=$log"
  tail -n 40 "$log" || true
  echo
  return 0
}

PROBE=("$SCRIPT_DIR/managed_pool_probe"
       --device "$DEVICE"
       --alloc-mb "$ALLOC_A_MB"
       --alloc-mb "$ALLOC_B_MB"
       --touch-gpu "$TOUCH_GPU"
       --prefetch-gpu "$PREFETCH_GPU")

run_case baseline "${PROBE[@]}"

run_case hbm_reserve \
  env LD_PRELOAD="$SCRIPT_DIR/libuvm_managed_budget.so" \
      UVM_MANAGED_BUDGET_DEVICE="$DEVICE" \
      UVM_HBM_RESERVE_MB="$HBM_RESERVE_MB" \
      UVM_MANAGED_BUDGET_VERBOSE=1 \
      "${PROBE[@]}"

run_case managed_pool_cap \
  env LD_PRELOAD="$SCRIPT_DIR/libuvm_managed_budget.so" \
      UVM_MANAGED_BUDGET_DEVICE="$DEVICE" \
      UVM_MANAGED_POOL_MB="$POOL_MB" \
      UVM_MANAGED_BUDGET_VERBOSE=1 \
      "${PROBE[@]}"

cat >"$LOG_DIR/README.txt" <<EOF
Managed budget experiment

Commands:
  baseline          : native cudaMallocManaged.
  hbm_reserve       : pre-reserve ${HBM_RESERVE_MB} MiB with cudaMalloc before first managed allocation.
  managed_pool_cap  : serve cudaMallocManaged from one ${POOL_MB} MiB managed arena.

Expected default behavior:
  - baseline succeeds for ${ALLOC_A_MB}+${ALLOC_B_MB} MiB if the GPU/runtime is healthy.
  - hbm_reserve also succeeds but reports lower cudaMemGetInfo free memory after preload init.
  - managed_pool_cap returns cudaErrorMemoryAllocation once live managed allocations exceed ${POOL_MB} MiB.

Tune with:
  DEVICE=0 POOL_MB=128 HBM_RESERVE_MB=128 ALLOC_A_MB=64 ALLOC_B_MB=80 \\
    $SCRIPT_DIR/run_managed_budget_experiment.sh
EOF

echo "Logs written to $LOG_DIR"
