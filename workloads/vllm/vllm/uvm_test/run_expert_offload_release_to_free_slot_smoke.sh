#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ROOT="${RUN_ROOT:-/tmp/expert_offload_release_to_free_slot_smoke_$(date +%Y%m%d_%H%M%S)}"
export USE_DRIVER_RELEASE="${USE_DRIVER_RELEASE:-1}"
export SKIP_BEFORE_PREALLOC="${SKIP_BEFORE_PREALLOC:-0}"
export ALLOW_EVICT="${ALLOW_EVICT:-0}"
export EXPERT_COUNT="${EXPERT_COUNT:-8}"
export EXPERT_BYTES="${EXPERT_BYTES:-64M}"
export SLOT_COUNT="${SLOT_COUNT:-48}"
export SLOT_BYTES="${SLOT_BYTES:-4M}"
export TARGET_FREE_BEFORE_SLOTS="${TARGET_FREE_BEFORE_SLOTS:-128M}"
export PRESSURE_CHUNK_BYTES="${PRESSURE_CHUNK_BYTES:-256M}"

exec "${SCRIPT_DIR}/run_expert_offload_slot_reuse_smoke.sh"
