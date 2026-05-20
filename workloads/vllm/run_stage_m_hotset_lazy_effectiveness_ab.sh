#!/usr/bin/env bash
# Focused A/B for the lazy hot-set path:
# - expert_cache_on should skip fully resident experts before region expansion
# - active_region_ids and cold candidate registration should only happen when a
#   real non-resident region reaches residency request

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_hotset_lazy_effectiveness_ab_run}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-112}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-536870912}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-32}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"

mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  local expert_cache="$2"
  local case_root="$RUN_ROOT/$label"
  local log_file="$RUN_ROOT/${label}.log"

  echo "==========================================================="
  echo "Running $label expert_cache=$expert_cache"
  echo "Run root: $case_root"
  echo "Log file: $log_file"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" \
  WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" \
  REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-8192}" \
  REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-1024}" \
  REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-16}" \
  "$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
    --run-root "$case_root" \
    --log-file "$log_file" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --weight-executor driver_batch \
    --init-weight-offload-enable 1 \
    --weight-runtime-prefetch-enable 1 \
    --weight-prefetch-trace-enable 1 \
    --weight-router-topk-every-n-steps 1 \
    --weight-prefetch-skip-resident 1 \
    --weight-prefetch-expert-cache-enable "$expert_cache" \
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
    --fault-address-log-enable 0 \
    --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
    --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
    --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
    --kv-gpu-bytes "$KV_GPU_BYTES" \
    --weight-active-evict-enable 0 \
    --weight-active-evict-max-experts-per-layer 0 \
    --weight-max-victims-per-step 0 \
    --server-max-num-seqs 1 \
    --request-rate 1 || true
}

run_case expert_cache_on 1
run_case expert_cache_off 0

SUMMARY_CSV="$RUN_ROOT/hotset_lazy_effectiveness_summary.csv"
SUMMARY_MD="$RUN_ROOT/hotset_lazy_effectiveness_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
cases = ["expert_cache_on", "expert_cache_off"]

def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def summary_row(case_root):
    path = case_root / "routing_phase_f_full_impl" / "summary.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                return row
    return {}

def phase_f_passed(case_root):
    path = case_root / "phase_f_check.json"
    if not path.exists():
        return None
    return bool(json.loads(path.read_text(encoding="utf-8")).get("passed"))

def load(case):
    case_root = run_root / case
    run_dir = case_root / "routing_phase_f_full_impl" / "runs" / "full_impl_enforce_rep1"
    row = summary_row(case_root)
    records = read_jsonl(run_dir / "vllm_uvm_weight_prefetch_full_impl_enforce_rep1.jsonl")
    steps = [r for r in records if r.get("action") == "step_summary"]
    actions = Counter(str(r.get("action")) for r in records)
    return {
        "case": case,
        "phase_f_passed": phase_f_passed(case_root),
        "tok_s": as_float(row.get("output_tok_s")),
        "tpot_ms": as_float(row.get("mean_tpot_ms")),
        "faults_dedup": as_float(row.get("delta_after_dedup")),
        "candidate_experts": sum(int(r.get("candidate_expert_records") or 0) for r in steps),
        "expert_skip": sum(int(r.get("expert_resident_skip_records") or 0) for r in steps),
        "expert_expanded": sum(int(r.get("expert_expanded_records") or 0) for r in steps),
        "candidate_regions": sum(int(r.get("candidate_region_records") or 0) for r in steps),
        "active_id_builds": sum(int(r.get("active_region_ids_build_records") or 0) for r in steps),
        "active_region_ids": sum(int(r.get("active_region_id_records") or 0) for r in steps),
        "cold_reg_calls": sum(int(r.get("cold_candidate_register_calls") or 0) for r in steps),
        "residency_requests": sum(1 for r in read_jsonl(run_dir / "vllm_uvm_pool_residency_full_impl_enforce_rep1.jsonl") if r.get("action") == "residency_request" and (r.get("metadata") or {}).get("source_action") == "prefetch"),
        "prefetch_issued": actions.get("prefetch_issued", 0),
    }

rows = [load(case) for case in cases]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage M Hotset Lazy Effectiveness A/B",
    "",
    f"- summary_csv: `{out_csv}`",
    "",
    "| case | passed | tok/s | TPOT ms | expert skip | region candidates | active id builds | active region ids | cold reg calls | residency req | prefetch issued |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {phase_f_passed} | {tok_s} | {tpot_ms} | "
        "{expert_skip} | {candidate_regions} | {active_id_builds} | "
        "{active_region_ids} | {cold_reg_calls} | {residency_requests} | "
        "{prefetch_issued} |".format(**row)
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
PY

echo "Hotset lazy effectiveness A/B finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
