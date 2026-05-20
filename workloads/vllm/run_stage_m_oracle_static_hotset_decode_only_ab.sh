#!/usr/bin/env bash
# Decode-only oracle static hotset upper-bound experiment.
#
# Run 1 records one fixed request's full MoE routing trace.
# Run 2 compares two static oracle placements built from that same trace:
#   - all_used: every observed prefill+decode expert is init-prefetched to GPU
#   - decode_only: only records with num_tokens <= DECODE_MAX_NUM_TOKENS count
#
# Fault deltas in the summary are still workload-level counters.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_oracle_static_hotset_decode_only_ab_run}"

WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
DECODE_MAX_NUM_TOKENS="${DECODE_MAX_NUM_TOKENS:-$SERVER_MAX_NUM_SEQS}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

INIT_WEIGHT_HOT_PREFETCH_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_ENABLE:-1}"
INIT_WEIGHT_HOT_PREFETCH_SYNC="${INIT_WEIGHT_HOT_PREFETCH_SYNC:-1}"
INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES="${INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES:-0}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:--1}"
INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE="${INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE:-1}"
HOT_RESIDENCY_STAGE_TRACE_ENABLE="${HOT_RESIDENCY_STAGE_TRACE_ENABLE:-0}"
HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="${HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS:-$DECODE_MAX_NUM_TOKENS}"
HOT_RESIDENCY_STAGE_DEVICE="${HOT_RESIDENCY_STAGE_DEVICE:-$INIT_WEIGHT_HOT_PREFETCH_DEVICE}"
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-0}"
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-0}"

TRACE_SOURCE_INIT_OFFLOAD_ENABLE="${TRACE_SOURCE_INIT_OFFLOAD_ENABLE:-1}"
TRACE_SOURCE_ALLOW_PHASE_F_FAIL="${TRACE_SOURCE_ALLOW_PHASE_F_FAIL:-1}"
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-256}"
REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-4096}"
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-1}"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M Oracle Static Hotset Decode-Only A/B"
echo "run_root=$RUN_ROOT"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS decode_max_num_tokens=$DECODE_MAX_NUM_TOKENS"
echo "plan_layers=$WEIGHT_PLAN_LAYERS plan_experts=$WEIGHT_PLAN_EXPERTS"
echo "weight_gpu_bytes=$WEIGHT_GPU_BYTES kv_gpu_bytes=$KV_GPU_BYTES"
echo "init_hot_prefetch=$INIT_WEIGHT_HOT_PREFETCH_ENABLE sync=$INIT_WEIGHT_HOT_PREFETCH_SYNC real_query=$INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE"
echo "hot_residency_stage_trace=$HOT_RESIDENCY_STAGE_TRACE_ENABLE stage_max_num_tokens=$HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS stage_device=$HOT_RESIDENCY_STAGE_DEVICE"
echo "==========================================================="

TRACE_RUN_ROOT="$RUN_ROOT/trace_source"
TRACE_LOG="$RUN_ROOT/trace_source.log"

echo "==========================================================="
echo "Run trace_source: record full routing trace"
echo "==========================================================="
set +e
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
TRACE_SOURCE_STATUS=$?
set -e

ORACLE_TRACE="$TRACE_RUN_ROOT/profile/vllm_uvm_moe_routing_trace.jsonl"
if [ ! -s "$ORACLE_TRACE" ]; then
  echo "Oracle trace was not generated: $ORACLE_TRACE" >&2
  exit 1
fi
if [ "$TRACE_SOURCE_STATUS" -ne 0 ]; then
  if [ "$TRACE_SOURCE_ALLOW_PHASE_F_FAIL" = "1" ]; then
    echo "trace_source_status=$TRACE_SOURCE_STATUS"
    echo "trace_source Phase F validation failed, but routing trace exists; continuing."
  else
    echo "trace_source failed with status $TRACE_SOURCE_STATUS" >&2
    exit "$TRACE_SOURCE_STATUS"
  fi
fi
echo "oracle_trace=$ORACLE_TRACE"

ALL_PLAN_JSON="$RUN_ROOT/oracle_static_hotset_all_used_weight_plan.json"
ALL_PLAN_REPORT="$RUN_ROOT/oracle_static_hotset_all_used_weight_plan_report.md"
DECODE_PLAN_JSON="$RUN_ROOT/oracle_static_hotset_decode_only_weight_plan.json"
DECODE_PLAN_REPORT="$RUN_ROOT/oracle_static_hotset_decode_only_weight_plan_report.md"

echo "==========================================================="
echo "Build all-used oracle static hotset plan"
echo "==========================================================="
python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$ORACLE_TRACE" \
  --output-plan "$ALL_PLAN_JSON" \
  --report-md "$ALL_PLAN_REPORT" \
  --policy exact_used_oracle \
  --fallback-layers "$WEIGHT_PLAN_LAYERS" \
  --fallback-experts "$WEIGHT_PLAN_EXPERTS" \
  --min-routing-records 1

echo "==========================================================="
echo "Build decode-only oracle static hotset plan"
echo "==========================================================="
python3 "$SCRIPT_DIR/build_stage_m_hot_cold_weight_plan.py" \
  --routing-trace "$ORACLE_TRACE" \
  --output-plan "$DECODE_PLAN_JSON" \
  --report-md "$DECODE_PLAN_REPORT" \
  --policy exact_used_oracle \
  --fallback-layers "$WEIGHT_PLAN_LAYERS" \
  --fallback-experts "$WEIGHT_PLAN_EXPERTS" \
  --max-num-tokens "$DECODE_MAX_NUM_TOKENS" \
  --min-routing-records 1

run_static_case() {
  local case_name="$1"
  local plan_json="$2"
  local case_root="$RUN_ROOT/$case_name"
  local case_log="$RUN_ROOT/$case_name.log"

  echo "==========================================================="
  echo "Run $case_name: init hot GPU + unused cold CPU"
  echo "==========================================================="
  "$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
    --run-root "$case_root" \
    --log-file "$case_log" \
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
    --init-weight-hot-prefetch-real-query-enable "$INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE" \
    --hot-residency-stage-trace-enable "$HOT_RESIDENCY_STAGE_TRACE_ENABLE" \
    --hot-residency-stage-max-num-tokens "$HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS" \
    --hot-residency-stage-device "$HOT_RESIDENCY_STAGE_DEVICE" \
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
    --weight-plan-file "$plan_json" \
    --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
    --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" \
    --weight-active-evict-enable 0 \
    --weight-prefetch-max-experts-per-layer 0 \
    --weight-prefetch-max-bytes-per-step 0
}

run_static_case "oracle_static_hotset_all_used" "$ALL_PLAN_JSON"
run_static_case "oracle_static_hotset_decode_only" "$DECODE_PLAN_JSON"

python3 - "$RUN_ROOT" "$ORACLE_TRACE" "$ALL_PLAN_JSON" "$DECODE_PLAN_JSON" "$DECODE_MAX_NUM_TOKENS" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
oracle_trace = Path(sys.argv[2])
all_plan_path = Path(sys.argv[3])
decode_plan_path = Path(sys.argv[4])
decode_max_num_tokens = sys.argv[5]

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

def plan_stats(path: Path) -> dict:
    plan = read_json(path)
    metadata = plan.get("metadata") or {}
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
    return {
        "plan_json": str(path),
        "routing_records_seen": metadata.get("routing_records_seen", metadata.get("routing_records")),
        "routing_records_used": metadata.get("routing_records_used", metadata.get("routing_records")),
        "min_num_tokens": metadata.get("min_num_tokens"),
        "max_num_tokens": metadata.get("max_num_tokens"),
        "prefetch_plan_records": len(prefetch),
        "offload_plan_records": len(offload),
        "plan_layers": len(layers),
        "max_used_experts_per_layer": max(used_by_layer.values() or [0]),
    }

plans = {
    "trace_source": {},
    "oracle_static_hotset_all_used": plan_stats(all_plan_path),
    "oracle_static_hotset_decode_only": plan_stats(decode_plan_path),
}
all_prefetch = plans["oracle_static_hotset_all_used"].get("prefetch_plan_records") or 0
decode_prefetch = plans["oracle_static_hotset_decode_only"].get("prefetch_plan_records") or 0
plan_shrink_ratio = (
    decode_prefetch / all_prefetch
    if isinstance(all_prefetch, int) and all_prefetch
    else None
)

rows = []
for case_name in (
    "trace_source",
    "oracle_static_hotset_all_used",
    "oracle_static_hotset_decode_only",
):
    case_root = root / case_name
    summary = read_csv_row(case_root / "routing_phase_f_full_impl" / "summary.csv")
    phase = read_json(case_root / "phase_f_check.json")
    residency = read_json(case_root / "real_residency_summary.json")
    plan = plans.get(case_name) or {}
    row = {
        "case": case_name,
        "phase_passed": phase.get("passed"),
        "tok_s": num(summary, "output_tok_s"),
        "tpot_ms": num(summary, "mean_tpot_ms"),
        "workload_faults_dedup": num(summary, "delta_after_dedup"),
        "routing_records_seen": plan.get("routing_records_seen"),
        "routing_records_used": plan.get("routing_records_used"),
        "min_num_tokens": plan.get("min_num_tokens"),
        "max_num_tokens": plan.get("max_num_tokens"),
        "prefetch_plan_records": plan.get("prefetch_plan_records"),
        "offload_plan_records": plan.get("offload_plan_records"),
        "plan_layers": plan.get("plan_layers"),
        "max_used_experts_per_layer": plan.get("max_used_experts_per_layer"),
        "init_hot_prefetch_issued": num(summary, "init_weight_hot_prefetch_issued"),
        "init_hot_prefetch_bytes": num(summary, "init_weight_hot_prefetch_summary_issued_bytes"),
        "init_hot_real_query_ranges": num(summary, "init_weight_hot_prefetch_real_query_ranges"),
        "init_hot_real_queried_bytes": num(summary, "init_weight_hot_prefetch_real_queried_bytes"),
        "init_hot_real_gpu_bytes": num(summary, "init_weight_hot_prefetch_real_gpu_bytes"),
        "init_hot_real_gpu_ratio": num(summary, "init_weight_hot_prefetch_real_gpu_ratio"),
        "init_cold_offload_issued": num(summary, "init_weight_offload_offload_cpu_issued"),
        "runtime_prefetch_issued": num(summary, "weight_prefetch_issued"),
        "active_evict_issued": num(summary, "residency_weight_victim_offload_issued"),
        "latest_weight_gpu_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("gpu_gib"),
        "latest_weight_cpu_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("cpu_gib"),
        "queried_weight_gib": ((residency.get("by_pool") or {}).get("weights") or {}).get("queried_gib"),
        "plan_shrink_ratio_vs_all_used": (
            plan_shrink_ratio if case_name == "oracle_static_hotset_decode_only" else None
        ),
    }
    rows.append(row)

csv_path = root / "oracle_static_hotset_decode_only_summary.csv"
fields = [
    "case",
    "phase_passed",
    "tok_s",
    "tpot_ms",
    "workload_faults_dedup",
    "routing_records_seen",
    "routing_records_used",
    "min_num_tokens",
    "max_num_tokens",
    "prefetch_plan_records",
    "offload_plan_records",
    "plan_shrink_ratio_vs_all_used",
    "plan_layers",
    "max_used_experts_per_layer",
    "init_hot_prefetch_issued",
    "init_hot_prefetch_bytes",
    "init_hot_real_query_ranges",
    "init_hot_real_queried_bytes",
    "init_hot_real_gpu_bytes",
    "init_hot_real_gpu_ratio",
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

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

md_path = root / "oracle_static_hotset_decode_only_summary.md"
lines = [
    "# Stage M Oracle Static Hotset Decode-Only A/B",
    "",
    f"- oracle_trace: `{oracle_trace}`",
    f"- all_used_plan_json: `{all_plan_path}`",
    f"- decode_only_plan_json: `{decode_plan_path}`",
    f"- decode_max_num_tokens: {decode_max_num_tokens}",
    f"- summary_csv: `{csv_path}`",
    f"- decode_only_plan_shrink_ratio_vs_all_used: {fmt(plan_shrink_ratio)}",
    "- note: `workload faults dedup` is not decode-only; it is the existing workload-level counter.",
    "",
    "| case | passed | tok/s | TPOT ms | workload faults dedup | records used/seen | plan hot | plan cold | shrink | hot GiB | hot real GPU ratio | runtime prefetch | active evict | weight GPU GiB | weight CPU GiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    hot_gib = None
    if row["init_hot_prefetch_bytes"] is not None:
        hot_gib = row["init_hot_prefetch_bytes"] / (1024 ** 3)
    records = ""
    if row["routing_records_used"] is not None or row["routing_records_seen"] is not None:
        records = f"{row['routing_records_used']}/{row['routing_records_seen']}"
    lines.append(
        "| {case} | {passed} | {tok_s} | {tpot} | {faults} | {records} | "
        "{hot} | {cold} | {shrink} | {hot_gib} | {ratio} | {runtime} | "
        "{evict} | {gpu} | {cpu} |".format(
            case=row["case"],
            passed=fmt(row["phase_passed"]),
            tok_s=fmt(row["tok_s"]),
            tpot=fmt(row["tpot_ms"]),
            faults=fmt(row["workload_faults_dedup"]),
            records=records,
            hot=fmt(row["prefetch_plan_records"]),
            cold=fmt(row["offload_plan_records"]),
            shrink=fmt(row["plan_shrink_ratio_vs_all_used"]),
            hot_gib=fmt(hot_gib),
            ratio=fmt(row["init_hot_real_gpu_ratio"]),
            runtime=fmt(row["runtime_prefetch_issued"]),
            evict=fmt(row["active_evict_issued"]),
            gpu=fmt(row["latest_weight_gpu_gib"]),
            cpu=fmt(row["latest_weight_cpu_gib"]),
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
PY

echo "Oracle static hotset decode-only A/B finished."
echo "summary_csv: $RUN_ROOT/oracle_static_hotset_decode_only_summary.csv"
echo "summary_md: $RUN_ROOT/oracle_static_hotset_decode_only_summary.md"
