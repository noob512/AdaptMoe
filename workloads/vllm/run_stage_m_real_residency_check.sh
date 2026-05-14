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
WEIGHT_RUNTIME_PREFETCH_ENABLE="${WEIGHT_RUNTIME_PREFETCH_ENABLE:-0}"
SCRATCH_DEVICE_DIRECT_ENABLE="${SCRATCH_DEVICE_DIRECT_ENABLE:-0}"
SCRATCH_POOL_ENABLE="${SCRATCH_POOL_ENABLE:-0}"
RESIDENCY_ENABLE="${RESIDENCY_ENABLE:-1}"
RESIDENCY_TRACE_ENABLE="${RESIDENCY_TRACE_ENABLE:-1}"
KV_RUNTIME_ENABLE="${KV_RUNTIME_ENABLE:-0}"
POOL_COORDINATOR_ENABLE="${POOL_COORDINATOR_ENABLE:-1}"
POOL_REGISTRY_ENABLE="${POOL_REGISTRY_ENABLE:-1}"
AUTO_GAP_WATCH_ENABLE="${AUTO_GAP_WATCH_ENABLE:-0}"
FAULT_ADDRESS_LOG_ENABLE="${FAULT_ADDRESS_LOG_ENABLE:-0}"

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
  --weight-runtime-prefetch-enable <0|1>
                                     default: 0, set 1 to measure runtime prefetch too
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
    --weight-runtime-prefetch-enable) WEIGHT_RUNTIME_PREFETCH_ENABLE="$2"; shift 2 ;;
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
  --weight-runtime-prefetch-enable "$WEIGHT_RUNTIME_PREFETCH_ENABLE"
  --scratch-device-direct-enable "$SCRATCH_DEVICE_DIRECT_ENABLE"
  --scratch-pool-enable "$SCRATCH_POOL_ENABLE"
  --residency-enable "$RESIDENCY_ENABLE"
  --residency-trace-enable "$RESIDENCY_TRACE_ENABLE"
  --kv-runtime-enable "$KV_RUNTIME_ENABLE"
  --pool-coordinator-enable "$POOL_COORDINATOR_ENABLE"
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
