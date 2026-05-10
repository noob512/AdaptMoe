#!/usr/bin/env bash
# Check Phase D: runtime weight prefetch must be driven by router top-k only,
# while runtime cold offload is disabled.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_d_check_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
ROUTING_TRACE="${ROUTING_TRACE:-}"

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-8}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-8}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-10737418240}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_d_check.sh [options]

Purpose:
  Verify Phase D:
    - runtime weight prefetch source is router_topk
    - runtime cold weight offload is disabled
    - Stage M init cold offload still uses routing_heatmap_v1 plan

Options:
  --routing-trace <path>                 reuse an existing routing trace
  --run-root <path>
  --log-file <path>
  --model <name>
  --prompts <n>                          default: 1
  --output-len <n>                       default: 64
  --request-rate <f>                     default: 1
  --hot-experts-per-layer <n>            default: 8
  --cold-experts-per-layer <n>           default: 8
  -h, --help
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --routing-trace) ROUTING_TRACE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --hot-experts-per-layer) HOT_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --cold-experts-per-layer) COLD_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_d_check.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage M Phase D check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

PHASE_C_ROOT="$RUN_ROOT/phase_c"
PLAN_JSON="$RUN_ROOT/routing_phase_d_weight_plan.json"
PLAN_REPORT="$RUN_ROOT/routing_phase_d_weight_plan_report.md"
CHECK_DIR="$RUN_ROOT/routing_phase_d_full_impl"
CHECK_JSON="$RUN_ROOT/phase_d_check.json"
CHECK_MD="$RUN_ROOT/phase_d_check.md"
mkdir -p "$CHECK_DIR"

if [ -z "$ROUTING_TRACE" ]; then
  "$SCRIPT_DIR/run_stage_m_phase_c_check.sh" \
    --run-root "$PHASE_C_ROOT" \
    --log-file "$PHASE_C_ROOT/run.log" \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
    --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER"
  ROUTING_TRACE="$PHASE_C_ROOT/profile/vllm_uvm_moe_routing_trace.jsonl"
fi

python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$ROUTING_TRACE" \
  --output-plan "$PLAN_JSON" \
  --report-md "$PLAN_REPORT" \
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER"

"$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" \
  --run-root "$CHECK_DIR" \
  --log-file "$CHECK_DIR/run.log" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --repeat 1 \
  --skip-trace-control \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --init-weight-max-bytes "$WEIGHT_GPU_BYTES" \
  --init-weight-max-experts-per-layer 0 \
  --weight-plan-file "$PLAN_JSON" \
  --weight-budget-cpu-offload-enable 0

python3 - "$CHECK_DIR" "$CHECK_JSON" "$CHECK_MD" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

check_dir = Path(sys.argv[1])
out_json = Path(sys.argv[2])
out_md = Path(sys.argv[3])
run_dir = check_dir / "runs" / "full_impl_enforce_rep1"
prefetch_trace = run_dir / "vllm_uvm_weight_prefetch_full_impl_enforce_rep1.jsonl"
init_trace = run_dir / "vllm_uvm_init_weight_offload_full_impl_enforce_rep1.jsonl"
summary_csv = check_dir / "summary.csv"

prefetch_actions = Counter()
prefetch_sources = Counter()
offload_actions = Counter()
if prefetch_trace.exists():
    with prefetch_trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = str(rec.get("action") or "")
            if action:
                prefetch_actions[action] += 1
            source = rec.get("prefetch_source")
            if source:
                prefetch_sources[str(source)] += 1
            if action.startswith("expert_") or action.startswith("offload_"):
                offload_actions[action] += 1

init_policy_seen = False
if init_trace.exists():
    with init_trace.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("plan_policy") == "routing_heatmap_v1":
                init_policy_seen = True

summary_row = {}
if summary_csv.exists():
    with summary_csv.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                summary_row = row
                break

def as_int(value):
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None

weight_offload_prefetch_cpu_issued = as_int(
    summary_row.get("weight_offload_prefetch_cpu_issued")
)
failed_requests = as_int(summary_row.get("failed_requests"))
exit_code = as_int(summary_row.get("exit_code"))

checks = {
    "prefetch_trace_present": prefetch_trace.exists(),
    "router_topk_source_present": prefetch_sources.get("router_topk", 0) > 0,
    "no_plan_intersection_source": prefetch_sources.get("plan_intersection", 0) == 0,
    "runtime_cold_offload_disabled": weight_offload_prefetch_cpu_issued == 0,
    "no_runtime_offload_actions": sum(offload_actions.values()) == 0,
    "init_routing_policy_seen": init_policy_seen,
    "benchmark_exit_zero": exit_code == 0,
    "benchmark_failed_requests_zero": failed_requests == 0,
}
passed = all(checks.values())
payload = {
    "passed": passed,
    "checks": checks,
    "prefetch_actions": dict(prefetch_actions),
    "prefetch_sources": dict(prefetch_sources),
    "offload_actions": dict(offload_actions),
    "weight_offload_prefetch_cpu_issued": weight_offload_prefetch_cpu_issued,
    "prefetch_trace": str(prefetch_trace),
    "summary_csv": str(summary_csv),
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

lines = ["# Stage M Phase D Check", "", f"- passed: {passed}", ""]
lines.append("## Checks")
for key, value in checks.items():
    lines.append(f"- {key}: {value}")
lines.extend([
    "",
    "## Evidence",
    f"- prefetch_sources: {dict(prefetch_sources)}",
    f"- prefetch_actions: {dict(prefetch_actions)}",
    f"- offload_actions: {dict(offload_actions)}",
    f"- weight_offload_prefetch_cpu_issued: {weight_offload_prefetch_cpu_issued}",
])
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY

cat "$CHECK_MD"

echo "Stage M Phase D check finished."
echo "run_root: $RUN_ROOT"
echo "check_json: $CHECK_JSON"
echo "check_md: $CHECK_MD"
