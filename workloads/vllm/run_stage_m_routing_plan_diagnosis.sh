#!/usr/bin/env bash
# Diagnose whether weight performance loss comes from inaccurate cold-expert
# selection or from migration/executor overhead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-2}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-2}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_m_routing_plan_diag_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-16}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"
WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"

WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-10737418240}"       # 10 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"                # 2 GiB
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-10737418240}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-2}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_routing_plan_diagnosis.sh [options]

Purpose:
  Explain whether the current full_impl slowdown is mostly caused by inaccurate
  broad cold-expert offload selection or by migration/executor overhead.

What it runs:
  1. routing_profile_pure_uvm
     Pure UVM + MoE routing trace enabled. This collects the actual expert
     heatmap for the same model/data/load.

  2. build routing-aware plan
     Reads routing trace and creates a plan where:
       - prefetch_plan = top hot experts per layer
       - offload_plan  = bottom cold experts per layer

  3. broad_cold_full_impl
     Existing full_impl with generated broad cold plan.

  4. routing_aware_full_impl
     Same full_impl knobs, but using the routing-aware plan.

Interpretation:
  - routing_aware better than broad_cold:
      weight offload selection was the main problem.
  - both equally slow:
      migration/executor overhead or KV/runtime policy is likely dominant.
  - routing_aware loses scratch fault benefit:
      plan or runtime phase interaction needs adjustment.

Options:
  --run-root <path>
  --log-file <path>
  --model <name>
  --prompts <n>                       default: 2
  --output-len <n>                    default: 128
  --request-rate <f>                  default: 2
  --hot-experts-per-layer <n>         default: 16
  --cold-experts-per-layer <n>        default: 16
  --weight-plan-layers <n>            default: 48
  --weight-plan-experts <n>           default: 128
  --weight-gpu-bytes <n>              default: 10GiB
  --kv-gpu-bytes <n>                  default: 2GiB
  --init-weight-max-bytes <n>         default: 10GiB
  --server-max-num-seqs <n>           default: 2
  --server-gpu-memory-utilization <f> default: 0.70
  -h, --help

Example:
  ./run_stage_m_routing_plan_diagnosis.sh \
    --log-file log/stage_m_routing_plan_diag.log
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --hot-experts-per-layer) HOT_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --cold-experts-per-layer) COLD_EXPERTS_PER_LAYER="$2"; shift 2 ;;
    --weight-plan-layers) WEIGHT_PLAN_LAYERS="$2"; shift 2 ;;
    --weight-plan-experts) WEIGHT_PLAN_EXPERTS="$2"; shift 2 ;;
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --init-weight-max-bytes) INIT_WEIGHT_MAX_BYTES="$2"; shift 2 ;;
    --server-max-num-seqs) SERVER_MAX_NUM_SEQS="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_m_routing_plan_diagnosis.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage M routing-aware weight plan diagnosis started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

PROFILE_DIR="$RUN_ROOT/profile_pure_uvm"
BROAD_DIR="$RUN_ROOT/broad_cold_full_impl"
ROUTING_DIR="$RUN_ROOT/routing_aware_full_impl"
PLAN_JSON="$RUN_ROOT/routing_aware_weight_plan.json"
PLAN_REPORT="$RUN_ROOT/routing_aware_weight_plan_report.md"
COMPARE_JSON="$RUN_ROOT/diagnosis_comparison.json"
COMPARE_MD="$RUN_ROOT/diagnosis_summary.md"

mkdir -p "$PROFILE_DIR" "$BROAD_DIR" "$ROUTING_DIR"

echo "==========================================================="
echo " Stage M Routing-Aware Weight Plan Diagnosis"
echo " run_root: $RUN_ROOT"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " hot_experts_per_layer: $HOT_EXPERTS_PER_LAYER"
echo " cold_experts_per_layer: $COLD_EXPERTS_PER_LAYER"
echo " weight_gpu_bytes: $WEIGHT_GPU_BYTES"
echo " kv_gpu_bytes: $KV_GPU_BYTES"
echo "==========================================================="

echo "==========================================================="
echo " [1/4] Routing profile under pure UVM"
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

echo "==========================================================="
echo " [2/4] Build routing-aware hot/cold weight plan"
echo "==========================================================="
python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$PROFILE_DIR/vllm_uvm_moe_routing_trace.jsonl" \
  --output-plan "$PLAN_JSON" \
  --report-md "$PLAN_REPORT" \
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --fallback-layers "$WEIGHT_PLAN_LAYERS" \
  --fallback-experts "$WEIGHT_PLAN_EXPERTS"

echo "routing_plan=$PLAN_JSON"
echo "routing_plan_report=$PLAN_REPORT"

echo "==========================================================="
echo " [3/4] Broad cold full_impl baseline"
echo "==========================================================="
"$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" \
  --run-root "$BROAD_DIR" \
  --log-file "$BROAD_DIR/run.log" \
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
  --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
  --init-weight-max-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --weight-budget-cpu-offload-enable 0

echo "==========================================================="
echo " [4/4] Routing-aware full_impl"
echo "==========================================================="
"$SCRIPT_DIR/run_stage_l_full_impl_uvm_ab.sh" \
  --run-root "$ROUTING_DIR" \
  --log-file "$ROUTING_DIR/run.log" \
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
  --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
  --init-weight-max-experts-per-layer 0 \
  --weight-plan-file "$PLAN_JSON" \
  --weight-budget-cpu-offload-enable 0

echo "==========================================================="
echo " Build diagnosis comparison"
echo "==========================================================="
python3 - "$BROAD_DIR/summary.csv" "$ROUTING_DIR/summary.csv" "$COMPARE_JSON" "$COMPARE_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

broad_csv, routing_csv, out_json, out_md = map(Path, sys.argv[1:])

def read_full(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("variant") == "full_impl_enforce":
            return row
    return rows[0] if rows else {}

def num(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None

broad = read_full(broad_csv)
routing = read_full(routing_csv)
keys = [
    "output_tok_s",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "delta_after_dedup",
    "delta_kv_after_dedup",
    "post_gap_faults",
    "post_unknown_faults",
    "init_weight_offload_cpu_issued_total",
    "residency_weight_estimated_gpu_bytes",
    "allocator_weight_budget_cpu_offload_bytes",
    "weight_prefetch_issued",
    "weight_offload_prefetch_cpu_issued",
]

comparison = {}
for key in keys:
    b = num(broad, key)
    r = num(routing, key)
    pct = None
    if b not in (None, 0) and r is not None:
        pct = (r - b) / b * 100.0
    comparison[key] = {
        "broad_cold": b,
        "routing_aware": r,
        "routing_vs_broad_pct": pct,
    }

diagnosis = []
bt = num(broad, "mean_tpot_ms")
rt = num(routing, "mean_tpot_ms")
bo = num(broad, "output_tok_s")
ro = num(routing, "output_tok_s")
if bt and rt and rt < bt * 0.85:
    diagnosis.append("routing-aware plan improves TPOT materially; cold expert selection was a major problem.")
elif bo and ro and ro > bo * 1.15:
    diagnosis.append("routing-aware plan improves throughput materially; cold expert selection was a major problem.")
else:
    diagnosis.append("routing-aware plan does not materially improve performance; migration/executor overhead or KV/runtime policy may dominate.")

bg = num(broad, "post_gap_faults")
rg = num(routing, "post_gap_faults")
if bg and rg and rg <= bg * 1.2:
    diagnosis.append("scratch/gap fault reduction is preserved under routing-aware plan.")
elif bg and rg:
    diagnosis.append("routing-aware plan changes scratch/gap behavior; inspect gap-watch phase/classification.")

payload = {
    "comparison": comparison,
    "diagnosis": diagnosis,
    "broad_summary_csv": str(broad_csv),
    "routing_summary_csv": str(routing_csv),
}
out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

lines = ["# Stage M Routing Plan Diagnosis", ""]
lines.append("## Diagnosis")
for item in diagnosis:
    lines.append(f"- {item}")
lines.append("")
lines.append("## Broad Cold vs Routing-Aware")
for key, values in comparison.items():
    lines.append("")
    lines.append(f"### {key}")
    lines.append(f"- broad_cold: {values['broad_cold']}")
    lines.append(f"- routing_aware: {values['routing_aware']}")
    lines.append(f"- routing_vs_broad_pct: {values['routing_vs_broad_pct']}")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

cat "$COMPARE_MD"

echo "==========================================================="
echo " Stage M routing-aware weight plan diagnosis finished"
echo " run_root: $RUN_ROOT"
echo " routing_plan: $PLAN_JSON"
echo " plan_report: $PLAN_REPORT"
echo " comparison_json: $COMPARE_JSON"
echo " comparison_md: $COMPARE_MD"
echo "==========================================================="
