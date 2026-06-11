#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

PAGES="${PAGES:-4096}"
WRITE_PAGES="${WRITE_PAGES:-512}"
PASSES="${PASSES:-4}"
OPS_PER_PAGE="${OPS_PER_PAGE:-16}"
STREAMS="${STREAMS:-4}"
DEVICE="${DEVICE:-0}"
BG_MAX_PAGES="${BG_MAX_PAGES:-512}"
BG_DELAY_MS="${BG_DELAY_MS:-0}"
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS:-0}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/results/prepare_remote_bg_correctness_$(date -u +%Y%m%d_%H%M%S)}"

echo "run_root=${RUN_ROOT}"

RUN_ROOT="${RUN_ROOT}" \
PAGES="${PAGES}" \
WRITE_PAGES="${WRITE_PAGES}" \
PASSES="${PASSES}" \
OPS_PER_PAGE="${OPS_PER_PAGE}" \
STREAMS="${STREAMS}" \
DEVICE="${DEVICE}" \
BG_MAX_PAGES="${BG_MAX_PAGES}" \
BG_DELAY_MS="${BG_DELAY_MS}" \
BG_RESCHEDULE_DELAY_MS="${BG_RESCHEDULE_DELAY_MS}" \
REMOTE_FIRST=0 \
BG_PROMOTE=0 \
PREPARE_REMOTE=0 \
PREPARE_REMOTE_BG=1 \
RESET_COUNTERS=1 \
"${ROOT_DIR}/scripts/test_remote_first_correctness.sh"

echo
echo "Done."
echo "This correctness run uses explicit PREPARE_REMOTE_READ_BG, not the fault-path remote-first hook."
echo "Summary: ${RUN_ROOT}/correctness_summary.md"
