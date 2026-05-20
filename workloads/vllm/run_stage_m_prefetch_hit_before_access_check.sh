#!/usr/bin/env bash
# Verify whether a successfully issued expert-weight prefetch becomes GPU
# resident before the same region is accessed in a later MoE layer step.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_prefetch_hit_before_access_check_run}"

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-64}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-64}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-0}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-0}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-268435456}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"
WEIGHT_PREFETCH_HIT_TRACE_ENABLE="${WEIGHT_PREFETCH_HIT_TRACE_ENABLE:-1}"
WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE="${WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE:-1}"
WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP="${WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP:-64}"
ORACLE_LOOKAHEAD_STEPS="${ORACLE_LOOKAHEAD_STEPS:-1 2}"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M Prefetch Hit-Before-Access Check"
echo "run_root=$RUN_ROOT"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "hot=$HOT_EXPERTS_PER_LAYER cold=$COLD_EXPERTS_PER_LAYER"
echo "lookaheads=$ORACLE_LOOKAHEAD_STEPS"
echo "hit_driver_query=$WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE max_regions=$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP"
echo "==========================================================="

TRACE_RUN_ROOT="$RUN_ROOT/trace_source"
TRACE_LOG="$RUN_ROOT/trace_source.log"

echo "==========================================================="
echo "Run trace_source: record routing trace with runtime prefetch disabled"
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
  --init-weight-offload-enable 1 \
  --weight-runtime-prefetch-enable 0 \
  --weight-prefetch-trace-enable 1 \
  --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
  --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
  --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
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
  --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
  --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
  --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
  --kv-gpu-bytes "$KV_GPU_BYTES" \
  --no-require-driver

ORACLE_TRACE="$TRACE_RUN_ROOT/profile/vllm_uvm_moe_routing_trace.jsonl"
if [ ! -s "$ORACLE_TRACE" ]; then
  echo "Oracle trace was not generated: $ORACLE_TRACE" >&2
  exit 1
fi
echo "oracle_trace=$ORACLE_TRACE"

run_case() {
  local case_name="$1"
  local source="$2"
  local lookahead="$3"
  local case_root="$RUN_ROOT/$case_name"
  local case_log="$RUN_ROOT/${case_name}.log"

  echo "==========================================================="
  echo "Running $case_name source=$source lookahead=$lookahead"
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
    --weight-runtime-prefetch-enable 1 \
    --weight-prefetch-trace-enable 1 \
    --weight-prefetch-source "$source" \
    --oracle-expert-trace-file "$ORACLE_TRACE" \
    --oracle-prefetch-lookahead-steps "$lookahead" \
    --weight-prefetch-hit-trace-enable "$WEIGHT_PREFETCH_HIT_TRACE_ENABLE" \
    --weight-prefetch-hit-driver-query-enable "$WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE" \
    --weight-prefetch-hit-max-regions-per-step "$WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP" \
    --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
    --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
    --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
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
    --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
    --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
    --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
    --kv-gpu-bytes "$KV_GPU_BYTES" \
    --no-require-driver \
    -- \
    --weight-prefetch-max-experts-per-layer "$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" \
    --weight-prefetch-max-bytes-per-step "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP"
}

run_case "router_topk_prefetch" "router_topk" 0
for lookahead in $ORACLE_LOOKAHEAD_STEPS; do
  run_case "oracle_replay_lookahead${lookahead}" "oracle_replay" "$lookahead"
done

python3 - "$RUN_ROOT" "$ORACLE_TRACE" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
oracle_trace = Path(sys.argv[2])

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_csv_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
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

rows = []
for case_dir in sorted(root.iterdir()):
    if not case_dir.is_dir() or case_dir.name == "trace_source":
        continue
    summary = read_csv_row(case_dir / "routing_phase_f_full_impl" / "summary.csv")
    phase = read_json(case_dir / "phase_f_check.json")
    replay = num(summary, "weight_prefetch_oracle_replay_records") or 0
    matches = num(summary, "weight_prefetch_oracle_match_records") or 0
    mismatch = num(summary, "weight_prefetch_oracle_mismatch_records") or 0
    before_candidates = num(summary, "weight_prefetch_before_access_candidates") or 0
    estimated_hits = num(summary, "weight_prefetch_hit_before_access_estimated") or 0
    real_queried = num(summary, "weight_prefetch_hit_before_access_real_queried_bytes") or 0
    real_gpu = num(summary, "weight_prefetch_hit_before_access_real_gpu_bytes") or 0
    rows.append({
        "case": case_dir.name,
        "passed": phase.get("passed"),
        "tok_s": num(summary, "output_tok_s"),
        "tpot_ms": num(summary, "mean_tpot_ms"),
        "faults_dedup": num(summary, "delta_after_dedup"),
        "prefetch_issued": num(summary, "weight_prefetch_issued"),
        "before_access_candidates": before_candidates,
        "estimated_hits": estimated_hits,
        "estimated_hit_rate": estimated_hits / before_candidates if before_candidates else None,
        "estimated_misses": num(summary, "weight_prefetch_miss_before_access_estimated"),
        "real_query_records": num(summary, "weight_prefetch_hit_before_access_real_query_records"),
        "real_query_ranges": num(summary, "weight_prefetch_hit_before_access_real_query_ranges"),
        "real_queried_bytes": real_queried,
        "real_gpu_bytes": real_gpu,
        "real_gpu_ratio": real_gpu / real_queried if real_queried else None,
        "same_step_access": num(summary, "weight_prefetch_same_step_access"),
        "no_prior_prefetch": num(summary, "weight_prefetch_access_no_prior_prefetch"),
        "hit_step_lag_avg": num(summary, "weight_prefetch_hit_step_lag_avg"),
        "oracle_match_rate": matches / replay if replay else None,
        "oracle_mismatch": mismatch,
    })

csv_path = root / "prefetch_hit_before_access_summary.csv"
fields = [
    "case",
    "passed",
    "tok_s",
    "tpot_ms",
    "faults_dedup",
    "prefetch_issued",
    "before_access_candidates",
    "estimated_hits",
    "estimated_hit_rate",
    "estimated_misses",
    "real_query_records",
    "real_query_ranges",
    "real_queried_bytes",
    "real_gpu_bytes",
    "real_gpu_ratio",
    "same_step_access",
    "no_prior_prefetch",
    "hit_step_lag_avg",
    "oracle_match_rate",
    "oracle_mismatch",
]
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

md_path = root / "prefetch_hit_before_access_summary.md"
lines = [
    "# Stage M Prefetch Hit-Before-Access Check",
    "",
    f"- oracle_trace: `{oracle_trace}`",
    f"- summary_csv: `{csv_path}`",
    "",
    "| case | passed | tok/s | TPOT ms | faults dedup | prefetch issued | before candidates | est hit rate | real GPU ratio | same-step | no prior | oracle match |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
for row in rows:
    lines.append(
        "| {case} | {passed} | {tok_s} | {tpot_ms} | {faults} | {issued} | {cands} | {est_rate} | {real_ratio} | {same} | {noprior} | {match} |".format(
            case=row["case"],
            passed=row["passed"],
            tok_s=fmt(row["tok_s"]),
            tpot_ms=fmt(row["tpot_ms"]),
            faults=fmt(row["faults_dedup"]),
            issued=fmt(row["prefetch_issued"]),
            cands=fmt(row["before_access_candidates"]),
            est_rate=fmt(row["estimated_hit_rate"]),
            real_ratio=fmt(row["real_gpu_ratio"]),
            same=fmt(row["same_step_access"]),
            noprior=fmt(row["no_prior_prefetch"]),
            match=fmt(row["oracle_match_rate"]),
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY

echo "Prefetch hit-before-access check finished."
echo "summary_csv: $RUN_ROOT/prefetch_hit_before_access_summary.csv"
echo "summary_md: $RUN_ROOT/prefetch_hit_before_access_summary.md"
