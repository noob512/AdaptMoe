#!/usr/bin/env bash
# Check Phase C: Stage M init-time placement must consume a routing-aware plan,
# register hot experts as GPU/protected, and offload only planned cold experts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_phase_c_check_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
ROUTING_TRACE="${ROUTING_TRACE:-}"

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-8}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-8}"
WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-10737418240}"       # 10 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"                # 2 GiB
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_c_check.sh [options]

Purpose:
  Verify Phase C implementation:
    - Stage M accepts a routing_heatmap_v1 plan
    - hot prefetch_plan experts are registered as GPU/protected
    - cold offload_plan experts are init-time offloaded
    - allocator-level blind weight CPU offload remains disabled

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
  --weight-gpu-bytes <n>                 default: 10GiB
  --kv-gpu-bytes <n>                     default: 2GiB
  --server-max-num-seqs <n>              default: 1
  --server-gpu-memory-utilization <f>    default: 0.70
  -h, --help

Example:
  ./run_stage_m_phase_c_check.sh \
    --log-file log/stage_m_phase_c_check.log
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
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_phase_c_check.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage M Phase C check started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

PROFILE_DIR="$RUN_ROOT/profile"
CHECK_DIR="$RUN_ROOT/routing_phase_c_full_impl"
PLAN_JSON="$RUN_ROOT/routing_phase_c_weight_plan.json"
PLAN_REPORT="$RUN_ROOT/routing_phase_c_weight_plan_report.md"
CHECK_JSON="$RUN_ROOT/phase_c_check.json"
CHECK_MD="$RUN_ROOT/phase_c_check.md"
mkdir -p "$PROFILE_DIR" "$CHECK_DIR"

echo "==========================================================="
echo " Stage M Phase C Check"
echo " run_root: $RUN_ROOT"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " routing_trace: ${ROUTING_TRACE:-<generate>}"
echo "==========================================================="

if [ -z "$ROUTING_TRACE" ]; then
  echo "==========================================================="
  echo " [1/4] Generate routing trace under pure UVM"
  echo "==========================================================="
  "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
    --model "$MODEL" \
    --mode trace \
    --with-address-log \
    --allocator-log "$PROFILE_DIR/vllm_uvm_allocator_trace_profile.log" \
    --trace-log "$PROFILE_DIR/uvm_kv_fault_stats_profile.log" \
    --address-trace-log "$PROFILE_DIR/uvm_kv_fault_addrs_profile.log" \
    --address-log "$PROFILE_DIR/vllm_uvm_address_regions_profile.log" \
    --server-log "$PROFILE_DIR/vllm_server_profile.log" \
    --bench-log "$PROFILE_DIR/vllm_bench_profile.log" \
    --uvm-policy-enable 0 \
    --uvm-device-direct-enable 0 \
    --uvm-residency-enable 0 \
    --uvm-kv-runtime-enable 0 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-offload-enable 0 \
    --uvm-pool-coordinator-enable 0 \
    --uvm-pool-registry-enable 0 \
    --uvm-scratch-pool-enable 0 \
    --uvm-moe-routing-trace-enable 1 \
    --uvm-moe-routing-trace-file "$PROFILE_DIR/vllm_uvm_moe_routing_trace.jsonl" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN"
  ROUTING_TRACE="$PROFILE_DIR/vllm_uvm_moe_routing_trace.jsonl"
else
  echo "==========================================================="
  echo " [1/4] Reuse routing trace"
  echo "==========================================================="
  echo "routing_trace=$ROUTING_TRACE"
fi

echo "==========================================================="
echo " [2/4] Build routing-aware plan"
echo "==========================================================="
python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$ROUTING_TRACE" \
  --output-plan "$PLAN_JSON" \
  --report-md "$PLAN_REPORT" \
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --fallback-layers "$WEIGHT_PLAN_LAYERS" \
  --fallback-experts "$WEIGHT_PLAN_EXPERTS" \
  --min-routing-records 1

echo "==========================================================="
echo " [3/4] Run routing-aware full_impl with Phase C guard"
echo "==========================================================="
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

echo "==========================================================="
echo " [4/4] Validate Phase C evidence"
echo "==========================================================="
python3 - "$CHECK_DIR" "$PLAN_JSON" "$CHECK_JSON" "$CHECK_MD" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

check_dir = Path(sys.argv[1])
plan_path = Path(sys.argv[2])
out_json = Path(sys.argv[3])
out_md = Path(sys.argv[4])

plan = json.loads(plan_path.read_text(encoding="utf-8"))
policy = (plan.get("metadata") or {}).get("policy")
prefetch_records = int(plan.get("prefetch_plan_records") or 0)
offload_records = int(plan.get("offload_plan_records") or 0)

summary_csv = check_dir / "summary.csv"
summary_row = {}
if summary_csv.exists():
    with summary_csv.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                summary_row = row
                break

run_dir = check_dir / "runs" / "full_impl_enforce_rep1"
trace_path = run_dir / "vllm_uvm_init_weight_offload_full_impl_enforce_rep1.jsonl"
action_counts = Counter()
summary_records = []
plan_policies = Counter()
if trace_path.exists():
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            action = rec.get("action")
            if action:
                action_counts[str(action)] += 1
            if rec.get("plan_policy"):
                plan_policies[str(rec.get("plan_policy"))] += 1
            if action == "init_weight_offload_summary":
                summary_records.append(rec)

hot_registered = sum(int(rec.get("hot_registered_records") or 0) for rec in summary_records)
issued = sum(int(rec.get("issued_records") or 0) for rec in summary_records)
reject_non_routing = action_counts.get("init_weight_offload_reject_non_routing_plan", 0)

def as_int(value):
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None

allocator_blind_offload_bytes = as_int(
    summary_row.get("allocator_weight_budget_cpu_offload_bytes")
)
exit_code = as_int(summary_row.get("exit_code"))
failed_requests = as_int(summary_row.get("failed_requests"))

checks = {
    "plan_policy_routing_heatmap_v1": policy == "routing_heatmap_v1",
    "plan_has_hot_records": prefetch_records > 0,
    "plan_has_cold_records": offload_records > 0,
    "trace_present": trace_path.exists(),
    "no_non_routing_reject": reject_non_routing == 0,
    "trace_uses_routing_policy": plan_policies.get("routing_heatmap_v1", 0) > 0,
    "hot_registered_records_present": hot_registered > 0,
    "cold_offload_issued_present": issued > 0,
    "allocator_blind_offload_disabled": allocator_blind_offload_bytes == 0,
    "benchmark_exit_zero": exit_code == 0,
    "benchmark_failed_requests_zero": failed_requests == 0,
}
passed = all(checks.values())
payload = {
    "passed": passed,
    "checks": checks,
    "plan_policy": policy,
    "prefetch_plan_records": prefetch_records,
    "offload_plan_records": offload_records,
    "trace_path": str(trace_path),
    "action_counts": dict(action_counts),
    "plan_policy_counts": dict(plan_policies),
    "hot_registered_records": hot_registered,
    "issued_records": issued,
    "allocator_weight_budget_cpu_offload_bytes": allocator_blind_offload_bytes,
    "summary_csv": str(summary_csv),
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

lines = ["# Stage M Phase C Check", "", f"- passed: {passed}", ""]
lines.append("## Checks")
for key, value in checks.items():
    lines.append(f"- {key}: {value}")
lines.extend([
    "",
    "## Evidence",
    f"- plan_policy: {policy}",
    f"- prefetch_plan_records: {prefetch_records}",
    f"- offload_plan_records: {offload_records}",
    f"- hot_registered_records: {hot_registered}",
    f"- issued_records: {issued}",
    f"- allocator_weight_budget_cpu_offload_bytes: {allocator_blind_offload_bytes}",
    f"- trace_path: `{trace_path}`",
])
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if passed else 1)
PY

cat "$CHECK_MD"

echo "==========================================================="
echo " Stage M Phase C check finished"
echo " run_root: $RUN_ROOT"
echo " plan: $PLAN_JSON"
echo " check_json: $CHECK_JSON"
echo " check_md: $CHECK_MD"
echo "==========================================================="
