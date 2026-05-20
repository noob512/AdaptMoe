#!/usr/bin/env bash
# Oracle static hotset upper-bound experiment for MoE expert weights.
#
# Run 1 records the exact experts used by a fixed request.
# Run 2 builds an exact-used plan:
#   - observed experts are prefetched to GPU during init
#   - unobserved experts are offloaded to CPU during init
#   - runtime prefetch and active eviction are disabled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_oracle_static_hotset_ab_run}"

WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

INIT_WEIGHT_HOT_PREFETCH_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_ENABLE:-1}"
INIT_WEIGHT_HOT_PREFETCH_SYNC="${INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-0}"
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-0}"

TRACE_SOURCE_INIT_OFFLOAD_ENABLE="${TRACE_SOURCE_INIT_OFFLOAD_ENABLE:-1}"
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-32}"
REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-128}"
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-16}"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M Oracle Static Hotset A/B"
echo "run_root=$RUN_ROOT"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "plan_layers=$WEIGHT_PLAN_LAYERS plan_experts=$WEIGHT_PLAN_EXPERTS"
echo "weight_gpu_bytes=$WEIGHT_GPU_BYTES kv_gpu_bytes=$KV_GPU_BYTES"
echo "init_hot_prefetch=$INIT_WEIGHT_HOT_PREFETCH_ENABLE sync=$INIT_WEIGHT_HOT_PREFETCH_SYNC"
echo "==========================================================="

TRACE_RUN_ROOT="$RUN_ROOT/trace_source"
TRACE_LOG="$RUN_ROOT/trace_source.log"

echo "==========================================================="
echo "Run trace_source: record exact routing trace"
echo "==========================================================="
"$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
  --run-root "$TRACE_RUN_ROOT" \
  --log-file "$TRACE_LOG" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --weight-executor "$WEIGHT_EXECUTOR" \
  --init-weight-offload-enable "$TRACE_SOURCE_INIT_OFFLOAD_ENABLE" \
  --init-weight-hot-prefetch-enable 0 \
  --weight-runtime-prefetch-enable 0 \
  --weight-prefetch-trace-enable 1 \
  --scratch-device-direct-enable 0 \
  --scratch-pool-enable 0 \
  --residency-enable 1 \
  --residency-trace-enable 1 \
  --kv-runtime-enable 0 \
  --pool-coordinator-enable 0 \
  --pool-coordinator-trace-enable 0 \
  --pool-registry-enable 1 \
  --auto-gap-watch-enable 0 \
  --fault-address-log-enable 1 \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --real-max-snapshots "$REAL_MAX_SNAPSHOTS" \
  --real-on-register-every "$REAL_ON_REGISTER_EVERY" \
  --real-on-action-every "$REAL_ON_ACTION_EVERY" \
  --no-require-driver

ORACLE_TRACE="$TRACE_RUN_ROOT/profile/vllm_uvm_moe_routing_trace.jsonl"
if [ ! -s "$ORACLE_TRACE" ]; then
  echo "Oracle trace was not generated: $ORACLE_TRACE" >&2
  exit 1
fi
echo "oracle_trace=$ORACLE_TRACE"

PLAN_JSON="$RUN_ROOT/oracle_static_hotset_weight_plan.json"
PLAN_REPORT="$RUN_ROOT/oracle_static_hotset_weight_plan_report.md"

echo "==========================================================="
echo "Build exact-used oracle static hotset plan"
echo "==========================================================="
python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$ORACLE_TRACE" \
  --output-plan "$PLAN_JSON" \
  --report-md "$PLAN_REPORT" \
  --policy exact_used_oracle \
  --fallback-layers "$WEIGHT_PLAN_LAYERS" \
  --fallback-experts "$WEIGHT_PLAN_EXPERTS" \
  --min-routing-records 1

STATIC_RUN_ROOT="$RUN_ROOT/oracle_static_hotset"
STATIC_LOG="$RUN_ROOT/oracle_static_hotset.log"

echo "==========================================================="
echo "Run oracle_static_hotset: init hot GPU + unused cold CPU"
echo "==========================================================="
"$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
  --run-root "$STATIC_RUN_ROOT" \
  --log-file "$STATIC_LOG" \
  --routing-trace "$ORACLE_TRACE" \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --weight-executor "$WEIGHT_EXECUTOR" \
  --init-weight-offload-enable 1 \
  --init-weight-hot-prefetch-enable "$INIT_WEIGHT_HOT_PREFETCH_ENABLE" \
  --init-weight-hot-prefetch-sync "$INIT_WEIGHT_HOT_PREFETCH_SYNC" \
  --init-weight-hot-prefetch-max-bytes "$INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES" \
  --init-weight-hot-prefetch-device "$INIT_WEIGHT_HOT_PREFETCH_DEVICE" \
  --weight-runtime-prefetch-enable 0 \
  --weight-prefetch-trace-enable 0 \
  --scratch-device-direct-enable 0 \
  --scratch-pool-enable 0 \
  --residency-enable 1 \
  --residency-trace-enable 1 \
  --kv-runtime-enable 0 \
  --pool-coordinator-enable 0 \
  --pool-coordinator-trace-enable 0 \
  --pool-registry-enable 1 \
  --auto-gap-watch-enable 0 \
  --fault-address-log-enable 1 \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --real-max-snapshots "$REAL_MAX_SNAPSHOTS" \
  --real-on-register-every "$REAL_ON_REGISTER_EVERY" \
  --real-on-action-every "$REAL_ON_ACTION_EVERY" \
  --no-require-driver \
  -- \
  --weight-plan-file "$PLAN_JSON" \
  --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
  --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" \
  --weight-active-evict-enable 0 \
  --weight-prefetch-max-experts-per-layer 0 \
  --weight-prefetch-max-bytes-per-step 0

python3 - "$RUN_ROOT" "$ORACLE_TRACE" "$PLAN_JSON" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
oracle_trace = Path(sys.argv[2])
plan_path = Path(sys.argv[3])

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_csv_row(path: Path, variant: str = "full_impl_enforce") -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == variant:
                return row
    return {}

def num(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None

def plan_counts(path: Path) -> tuple[int, int, int, int]:
    plan = read_json(path)
    prefetch = plan.get("prefetch_plan") or []
    offload = plan.get("offload_plan") or []
    layers = set()
    used_by_layer = {}
    for item in prefetch:
        layer = item.get("layer_id")
        layers.add(layer)
        used_by_layer[layer] = used_by_layer.get(layer, 0) + 1
    for item in offload:
        layers.add(item.get("layer_id"))
    return (
        len(prefetch),
        len(offload),
        len(layers),
        max(used_by_layer.values() or [0]),
    )

prefetch_plan_records, offload_plan_records, plan_layers, max_used_per_layer = plan_counts(plan_path)

rows = []
for case_name, case_root in (
    ("trace_source", root / "trace_source"),
    ("oracle_static_hotset", root / "oracle_static_hotset"),
):
    summary = read_csv_row(case_root / "routing_phase_f_full_impl" / "summary.csv")
    phase = read_json(case_root / "phase_f_check.json")
    residency = read_json(case_root / "real_residency_summary.json")
    rows.append({
        "case": case_name,
        "phase_passed": phase.get("passed"),
        "tok_s": num(summary, "output_tok_s"),
        "tpot_ms": num(summary, "mean_tpot_ms"),
        "faults_dedup": num(summary, "delta_after_dedup"),
        "init_hot_prefetch_issued": num(summary, "init_weight_hot_prefetch_issued"),
        "init_hot_prefetch_bytes": num(summary, "init_weight_hot_prefetch_summary_issued_bytes"),
        "init_hot_registered": num(summary, "init_weight_hot_registered_records"),
        "init_cold_offload_issued": num(summary, "init_weight_offload_offload_cpu_issued"),
        "runtime_prefetch_issued": num(summary, "weight_prefetch_issued"),
        "active_evict_issued": num(summary, "residency_weight_victim_offload_issued"),
        "latest_weight_gpu_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("gpu_gib"),
        "latest_weight_cpu_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("cpu_gib"),
        "queried_weight_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("queried_gib"),
    })

csv_path = root / "oracle_static_hotset_summary.csv"
fields = [
    "case",
    "phase_passed",
    "tok_s",
    "tpot_ms",
    "faults_dedup",
    "init_hot_prefetch_issued",
    "init_hot_prefetch_bytes",
    "init_hot_registered",
    "init_cold_offload_issued",
    "runtime_prefetch_issued",
    "active_evict_issued",
    "latest_weight_gpu_gib",
    "latest_weight_cpu_gib",
    "queried_weight_gib",
]
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "oracle_static_hotset_summary.md"
lines = [
    "# Stage M Oracle Static Hotset A/B",
    "",
    f"- oracle_trace: `{oracle_trace}`",
    f"- plan_json: `{plan_path}`",
    f"- summary_csv: `{csv_path}`",
    f"- prefetch_plan_records: {prefetch_plan_records}",
    f"- offload_plan_records: {offload_plan_records}",
    f"- plan_layers: {plan_layers}",
    f"- max_used_experts_per_layer: {max_used_per_layer}",
    "",
    "| case | passed | tok/s | TPOT ms | faults dedup | hot prefetch | hot GiB | cold offload | runtime prefetch | active evict | weight GPU GiB | weight CPU GiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    hot_gib = None
    if row["init_hot_prefetch_bytes"] is not None:
        hot_gib = row["init_hot_prefetch_bytes"] / (1024 ** 3)
    lines.append(
        "| {case} | {phase_passed} | {tok_s} | {tpot_ms} | {faults_dedup} | "
        "{init_hot_prefetch_issued} | {hot_gib} | {init_cold_offload_issued} | "
        "{runtime_prefetch_issued} | {active_evict_issued} | {latest_weight_gpu_gib} | "
        "{latest_weight_cpu_gib} |".format(
            **row,
            hot_gib=hot_gib,
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
PY

echo "Oracle static hotset A/B finished."
echo "summary_csv: $RUN_ROOT/oracle_static_hotset_summary.csv"
echo "summary_md: $RUN_ROOT/oracle_static_hotset_summary.md"
