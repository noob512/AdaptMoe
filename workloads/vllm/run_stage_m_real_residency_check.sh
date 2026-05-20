#!/usr/bin/env bash
# Run a small real-vLLM Phase F experiment and sample page-granular UVM GPU
# residency from inside the vLLM process.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_real_residency_check_run}"
LOG_FILE="${LOG_FILE:-log/stage_m_real_residency_check.log}"
REAL_TRACE_FILE="${REAL_TRACE_FILE:-$RUN_ROOT/real_residency.jsonl}"
REAL_TRACE_FILE_EXPLICIT=0
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
REQUIRE_DRIVER="${REQUIRE_DRIVER:-1}"

INIT_WEIGHT_OFFLOAD_ENABLE="${INIT_WEIGHT_OFFLOAD_ENABLE:-1}"
INIT_WEIGHT_HOT_PREFETCH_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_ENABLE:-0}"
INIT_WEIGHT_HOT_PREFETCH_SYNC="${INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE:-0}"
HOT_RESIDENCY_STAGE_TRACE_ENABLE="${HOT_RESIDENCY_STAGE_TRACE_ENABLE:-0}"
HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="${HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS:-1}"
HOT_RESIDENCY_STAGE_DEVICE="${HOT_RESIDENCY_STAGE_DEVICE:--1}"
WEIGHT_RUNTIME_PREFETCH_ENABLE="${WEIGHT_RUNTIME_PREFETCH_ENABLE:-0}"
SCRATCH_DEVICE_DIRECT_ENABLE="${SCRATCH_DEVICE_DIRECT_ENABLE:-0}"
SCRATCH_POOL_ENABLE="${SCRATCH_POOL_ENABLE:-0}"
RESIDENCY_ENABLE="${RESIDENCY_ENABLE:-1}"
RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE:-1}"
KV_RUNTIME_ENABLE="${KV_RUNTIME_ENABLE:-0}"
POOL_COORDINATOR_ENABLE="${POOL_COORDINATOR_ENABLE:-1}"
POOL_COORDINATOR_TRACE_ENABLE="${POOL_COORDINATOR_TRACE_ENABLE:-1}"
POOL_REGISTRY_ENABLE="${POOL_REGISTRY_ENABLE:-1}"
AUTO_GAP_WATCH_ENABLE="${AUTO_GAP_WATCH_ENABLE:-0}"
FAULT_ADDRESS_LOG_ENABLE="${FAULT_ADDRESS_LOG_ENABLE:-0}"
WEIGHT_PREFETCH_TRACE_ENABLE="${WEIGHT_PREFETCH_TRACE_ENABLE:-1}"
WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="${WEIGHT_ROUTER_TOPK_EVERY_N_STEPS:-1}"
WEIGHT_PREFETCH_SKIP_RESIDENT="${WEIGHT_PREFETCH_SKIP_RESIDENT:-1}"
WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
WEIGHT_PREFETCH_SOURCE="${WEIGHT_PREFETCH_SOURCE:-router_topk}"
ORACLE_EXPERT_TRACE_FILE="${ORACLE_EXPERT_TRACE_FILE:-}"
ORACLE_PREFETCH_LOOKAHEAD_STEPS="${ORACLE_PREFETCH_LOOKAHEAD_STEPS:-1}"
WEIGHT_PREFETCH_HIT_TRACE_ENABLE="${WEIGHT_PREFETCH_HIT_TRACE_ENABLE:-0}"
WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="${WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE:-0}"
WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="${WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP:-64}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"

REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-128}"
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-16}"
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-32}"
REAL_MAX_REGIONS_PER_POOL="${REAL_MAX_REGIONS_PER_POOL:-0}"
REAL_DEVICE="${REAL_DEVICE:-0}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_real_residency_check.sh [options] [-- extra run_stage_m_phase_f_check.sh args]

Options:
  --run-root <path>                  default: log/stage_m_real_residency_check_run
  --log-file <path>                  default: log/stage_m_real_residency_check.log
  --real-trace-file <path>           default: <run-root>/real_residency.jsonl
  --prompts <n>                      default: 1
  --output-len <n>                   default: 64
  --weight-executor <mode>           default: driver_batch
  --no-require-driver
  --init-weight-offload-enable <0|1> default: 1
  --init-weight-hot-prefetch-enable <0|1> default: 0
  --init-weight-hot-prefetch-sync <0|1> default: 1
  --init-weight-hot-prefetch-max-bytes <n> default: 0 means unlimited
  --init-weight-hot-prefetch-device <id> default: -1/current
  --init-weight-hot-prefetch-real-query-enable <0|1> default: 0
  --hot-residency-stage-trace-enable <0|1> default: 0
  --hot-residency-stage-max-num-tokens <n> default: 1
  --hot-residency-stage-device <id> default: -1/current
  --weight-runtime-prefetch-enable <0|1>
                                     default: 0, set 1 to measure runtime prefetch too
  --weight-prefetch-trace-enable <0|1> default: 1
  --weight-router-topk-every-n-steps <n> default: 1
  --weight-prefetch-skip-resident <0|1> default: 1
  --weight-prefetch-expert-cache-enable <0|1> default: 1
  --weight-prefetch-region-cooldown-steps <n> default: 8
  --weight-prefetch-source <source>      router_topk|plan_intersection|oracle_replay
  --oracle-expert-trace-file <path>      routing trace JSONL for oracle_replay
  --oracle-prefetch-lookahead-steps <n>  per-layer replay lookahead
  --weight-prefetch-hit-trace-enable <0|1> default: 0
  --weight-prefetch-hit-driver-query-enable <0|1> default: 0
  --weight-prefetch-hit-max-regions-per-step <n> default: 64
  --weight-evict-protect-recent-steps <n> default: 8
  --real-on-register-every <n>       snapshot every n registered regions, default: 128
  --real-on-action-every <n>         snapshot every n residency actions, default: 16
  --real-max-snapshots <n>           default: 32
  --real-max-regions-per-pool <n>    default: 0 means all registered regions
  -h, --help

Example:
  VLLM_UVM_PHASE_F_IOCTL_MODE=policy UVM_PHASE_F_ROUTE_LOG_DELAY_MS=0 \
  ./run_stage_m_real_residency_check.sh
USAGE
}

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --run-root)
      RUN_ROOT="$2"
      if [ "$REAL_TRACE_FILE_EXPLICIT" = "0" ]; then
        REAL_TRACE_FILE="$RUN_ROOT/real_residency.jsonl"
      fi
      shift 2
      ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --real-trace-file) REAL_TRACE_FILE="$2"; REAL_TRACE_FILE_EXPLICIT=1; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --weight-executor) WEIGHT_EXECUTOR="$2"; shift 2 ;;
    --no-require-driver) REQUIRE_DRIVER=0; shift ;;
    --init-weight-offload-enable) INIT_WEIGHT_OFFLOAD_ENABLE="$2"; shift 2 ;;
    --init-weight-hot-prefetch-enable) INIT_WEIGHT_HOT_PREFETCH_ENABLE="$2"; shift 2 ;;
    --init-weight-hot-prefetch-sync) INIT_WEIGHT_HOT_PREFETCH_SYNC="$2"; shift 2 ;;
    --init-weight-hot-prefetch-max-bytes) INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="$2"; shift 2 ;;
    --init-weight-hot-prefetch-device) INIT_WEIGHT_HOT_PREFETCH_DEVICE="$2"; shift 2 ;;
    --init-weight-hot-prefetch-real-query-enable) INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="$2"; shift 2 ;;
    --hot-residency-stage-trace-enable) HOT_RESIDENCY_STAGE_TRACE_ENABLE="$2"; shift 2 ;;
    --hot-residency-stage-max-num-tokens) HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="$2"; shift 2 ;;
    --hot-residency-stage-device) HOT_RESIDENCY_STAGE_DEVICE="$2"; shift 2 ;;
    --weight-runtime-prefetch-enable) WEIGHT_RUNTIME_PREFETCH_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-trace-enable) WEIGHT_PREFETCH_TRACE_ENABLE="$2"; shift 2 ;;
    --weight-router-topk-every-n-steps) WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="$2"; shift 2 ;;
    --weight-prefetch-skip-resident) WEIGHT_PREFETCH_SKIP_RESIDENT="$2"; shift 2 ;;
    --weight-prefetch-expert-cache-enable) WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-region-cooldown-steps) WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="$2"; shift 2 ;;
    --weight-prefetch-source) WEIGHT_PREFETCH_SOURCE="$2"; shift 2 ;;
    --oracle-expert-trace-file) ORACLE_EXPERT_TRACE_FILE="$2"; shift 2 ;;
    --oracle-prefetch-lookahead-steps) ORACLE_PREFETCH_LOOKAHEAD_STEPS="$2"; shift 2 ;;
    --weight-prefetch-hit-trace-enable) WEIGHT_PREFETCH_HIT_TRACE_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-hit-driver-query-enable) WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="$2"; shift 2 ;;
    --weight-prefetch-hit-max-regions-per-step) WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="$2"; shift 2 ;;
    --weight-evict-protect-recent-steps) WEIGHT_EVICT_PROTECT_RECENT_STEPS="$2"; shift 2 ;;
    --real-on-register-every) REAL_ON_REGISTER_EVERY="$2"; shift 2 ;;
    --real-on-action-every) REAL_ON_ACTION_EVERY="$2"; shift 2 ;;
    --real-max-snapshots) REAL_MAX_SNAPSHOTS="$2"; shift 2 ;;
    --real-max-regions-per-pool) REAL_MAX_REGIONS_PER_POOL="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$RUN_ROOT" "$(dirname "$LOG_FILE")" "$(dirname "$REAL_TRACE_FILE")"
rm -f "$REAL_TRACE_FILE"

echo "Stage M real residency check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Real residency trace: $REAL_TRACE_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

COMMON_ARGS=(
  --log-file "$LOG_FILE"
  --run-root "$RUN_ROOT"
  --prompts "$PROMPTS"
  --output-len "$OUTPUT_LEN"
  --weight-executor "$WEIGHT_EXECUTOR"
  --init-weight-offload-enable "$INIT_WEIGHT_OFFLOAD_ENABLE"
  --init-weight-hot-prefetch-enable "$INIT_WEIGHT_HOT_PREFETCH_ENABLE"
  --init-weight-hot-prefetch-sync "$INIT_WEIGHT_HOT_PREFETCH_SYNC"
  --init-weight-hot-prefetch-max-bytes "$INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES"
  --init-weight-hot-prefetch-device "$INIT_WEIGHT_HOT_PREFETCH_DEVICE"
  --init-weight-hot-prefetch-real-query-enable "$INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE"
  --hot-residency-stage-trace-enable "$HOT_RESIDENCY_STAGE_TRACE_ENABLE"
  --hot-residency-stage-max-num-tokens "$HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS"
  --hot-residency-stage-device "$HOT_RESIDENCY_STAGE_DEVICE"
  --weight-runtime-prefetch-enable "$WEIGHT_RUNTIME_PREFETCH_ENABLE"
  --weight-prefetch-trace-enable "$WEIGHT_PREFETCH_TRACE_ENABLE"
  --weight-router-topk-every-n-steps "$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS"
  --weight-prefetch-skip-resident "$WEIGHT_PREFETCH_SKIP_RESIDENT"
  --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE"
  --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS"
  --weight-prefetch-source "$WEIGHT_PREFETCH_SOURCE"
  --oracle-expert-trace-file "$ORACLE_EXPERT_TRACE_FILE"
  --oracle-prefetch-lookahead-steps "$ORACLE_PREFETCH_LOOKAHEAD_STEPS"
  --weight-prefetch-hit-trace-enable "$WEIGHT_PREFETCH_HIT_TRACE_ENABLE"
  --weight-prefetch-hit-driver-query-enable "$WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE"
  --weight-prefetch-hit-max-regions-per-step "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP"
  --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS"
  --scratch-device-direct-enable "$SCRATCH_DEVICE_DIRECT_ENABLE"
  --scratch-pool-enable "$SCRATCH_POOL_ENABLE"
  --residency-enable "$RESIDENCY_ENABLE"
  --residency-trace-enable "$RESIDENCY_TRACE_ENABLE"
  --kv-runtime-enable "$KV_RUNTIME_ENABLE"
  --pool-coordinator-enable "$POOL_COORDINATOR_ENABLE"
  --pool-coordinator-trace-enable "$POOL_COORDINATOR_TRACE_ENABLE"
  --pool-registry-enable "$POOL_REGISTRY_ENABLE"
  --auto-gap-watch-enable "$AUTO_GAP_WATCH_ENABLE"
  --fault-address-log-enable "$FAULT_ADDRESS_LOG_ENABLE"
)
if [ "$REQUIRE_DRIVER" = "1" ]; then
  COMMON_ARGS+=(--require-driver)
fi

set +e
VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
VLLM_UVM_REAL_RESIDENCY_ENABLE=1 \
VLLM_UVM_REAL_RESIDENCY_TRACE_FILE="$REAL_TRACE_FILE" \
VLLM_UVM_REAL_RESIDENCY_DEVICE="$REAL_DEVICE" \
VLLM_UVM_REAL_RESIDENCY_ON_REGISTER_EVERY="$REAL_ON_REGISTER_EVERY" \
VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY="$REAL_ON_ACTION_EVERY" \
VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS="$REAL_MAX_SNAPSHOTS" \
VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL="$REAL_MAX_REGIONS_PER_POOL" \
"$SCRIPT_DIR/run_stage_m_phase_f_check.sh" "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}"
PHASE_F_STATUS=$?
set -e

python3 - "$REAL_TRACE_FILE" "$RUN_ROOT/real_residency_summary.json" "$RUN_ROOT/real_residency_summary.md" "$PHASE_F_STATUS" <<'PY'
import json
import sys
from pathlib import Path

trace = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
phase_f_status = int(sys.argv[4])

records = []
if trace.exists():
    with trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("action") == "real_residency_snapshot":
                records.append(rec)

latest = records[-1] if records else {}

def gib(value):
    return float(value or 0) / (1024 ** 3)

summary = {
    "trace": str(trace),
    "phase_f_exit_code": phase_f_status,
    "snapshots": len(records),
    "ok_snapshots": sum(1 for rec in records if rec.get("ok")),
    "latest": latest,
}
out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

lines = [
    "# Real UVM Residency Summary",
    "",
    f"- trace: {trace}",
    f"- phase_f_exit_code: {phase_f_status}",
    f"- snapshots: {summary['snapshots']}",
    f"- ok_snapshots: {summary['ok_snapshots']}",
]
if latest:
    lines.extend([
        f"- latest_total_queried_gib: {gib(latest.get('queried_bytes')):.3f}",
        f"- latest_total_gpu_gib: {gib(latest.get('gpu_bytes')):.3f}",
        f"- latest_total_cpu_gib: {gib(latest.get('cpu_bytes')):.3f}",
        f"- latest_total_unknown_gib: {gib(latest.get('unknown_bytes')):.3f}",
        "",
        "## By Pool",
    ])
    for pool in ("weights", "kv", "scratch"):
        item = (latest.get("by_pool") or {}).get(pool) or {}
        lines.append(
            "- {pool}: regions={regions} queried_regions={queried_regions} "
            "registered={registered:.3f}GiB queried={queried:.3f}GiB "
            "gpu={gpu:.3f}GiB cpu={cpu:.3f}GiB unknown={unknown:.3f}GiB "
            "batches={batches} failed_batches={failed_batches} ok={ok}".format(
                pool=pool,
                regions=item.get("region_count", 0),
                queried_regions=item.get("queried_region_count", 0),
                registered=gib(item.get("registered_bytes")),
                queried=gib(item.get("queried_bytes")),
                gpu=gib(item.get("gpu_bytes")),
                cpu=gib(item.get("cpu_bytes")),
                unknown=gib(item.get("unknown_bytes")),
                batches=item.get("query_batches", 0),
                failed_batches=item.get("failed_batches", 0),
                ok=item.get("ok"),
            )
        )
else:
    lines.append("- error: no real_residency_snapshot records found")

out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "Real residency check finished."
echo "summary_json: $RUN_ROOT/real_residency_summary.json"
echo "summary_md: $RUN_ROOT/real_residency_summary.md"
exit "$PHASE_F_STATUS"
