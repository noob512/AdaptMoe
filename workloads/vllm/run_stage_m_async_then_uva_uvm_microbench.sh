#!/usr/bin/env bash
# Run the gap#2 cudaMallocAsync device_direct A/B first, then the UVA vs UVM
# page-touch microbenchmark with the same defaults discussed in Stage M.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

./run_stage_m_gap2_device_direct_async_ab_p10.sh

PAGES="${PAGES:-262144}" \
TOUCHES="${TOUCHES:-1}" \
ITERS="${ITERS:-7}" \
./run_stage_m_uva_uvm_page_touch_microbench.sh
