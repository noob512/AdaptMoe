#!/usr/bin/env bash
# Focused check for the region_id driven weight policy:
# 1) prefetch should skip already-resident regions before coordinator/driver
# 2) active eviction should protect recently active/prefetching regions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_region_state_policy_check_run}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"  # 24 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"           # 2 GiB
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-112}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"
ACTIVE_EXPERTS_PER_LAYER="${ACTIVE_EXPERTS_PER_LAYER:-32}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"

mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  local active_evict="$2"
  local case_root="$RUN_ROOT/$label"
  local log_file="$RUN_ROOT/${label}.log"

  echo "==========================================================="
  echo "Running $label active_evict=$active_evict"
  echo "Run root: $case_root"
  echo "Log file: $log_file"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-536870912}" \
  WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$ACTIVE_EXPERTS_PER_LAYER" \
  WEIGHT_PREFETCH_SKIP_RESIDENT=1 \
  WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
  WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
  WEIGHT_EVICT_PROTECT_RECENT_STEPS="$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
  WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER="${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-16}" \
  WEIGHT_MAX_VICTIMS_PER_STEP="${WEIGHT_MAX_VICTIMS_PER_STEP:-16}" \
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
    --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
    --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
    --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
    --scratch-device-direct-enable 0 \
    --scratch-pool-enable 0 \
    --residency-enable 1 \
    --residency-trace-enable 1 \
    --kv-runtime-enable 0 \
    --pool-coordinator-enable 1 \
    --pool-coordinator-trace-enable 0 \
    --pool-registry-enable 1 \
    --auto-gap-watch-enable 0 \
    --fault-address-log-enable 0 \
    --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
    --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
    --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
    --kv-gpu-bytes "$KV_GPU_BYTES" \
    --weight-active-evict-enable "$active_evict" \
    --weight-active-evict-max-experts-per-layer "${WEIGHT_ACTIVE_EVICT_MAX_EXPERTS_PER_LAYER:-16}" \
    --weight-max-victims-per-step "${WEIGHT_MAX_VICTIMS_PER_STEP:-16}" \
    --server-max-num-seqs 1 \
    --request-rate 1
}

run_case prefetch_region_filter 0
run_case prefetch_plus_active_evict_recent 1

SUMMARY_CSV="$RUN_ROOT/region_state_policy_summary.csv"
SUMMARY_MD="$RUN_ROOT/region_state_policy_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])
cases = ["prefetch_region_filter", "prefetch_plus_active_evict_recent"]

def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def as_float(value):
    if value in (None, "", "None"):
        return None
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

def load(case):
    case_root = run_root / case
    run_dir = case_root / "routing_phase_f_full_impl" / "runs" / "full_impl_enforce_rep1"
    row = summary_row(case_root)
    weight_records = read_jsonl(run_dir / "vllm_uvm_weight_prefetch_full_impl_enforce_rep1.jsonl")
    residency_records = read_jsonl(run_dir / "vllm_uvm_pool_residency_full_impl_enforce_rep1.jsonl")
    step_summaries = [r for r in weight_records if r.get("action") == "step_summary"]
    weight_actions = Counter(str(r.get("action")) for r in weight_records)
    residency_actions = Counter(str(r.get("action")) for r in residency_records)
    prefetch_residency_records = [
        r
        for r in residency_records
        if (r.get("metadata") or {}).get("source_action") == "prefetch"
    ]
    prefetch_residency_actions = Counter(
        str(r.get("action")) for r in prefetch_residency_records
    )
    return {
        "case": case,
        "tok_s": as_float(row.get("output_tok_s")),
        "tpot_ms": as_float(row.get("mean_tpot_ms")),
        "faults_dedup": as_float(row.get("delta_after_dedup")),
        "candidate_experts": sum(int(r.get("candidate_expert_records") or 0) for r in step_summaries),
        "expert_resident_skip": sum(int(r.get("expert_resident_skip_records") or 0) for r in step_summaries),
        "expert_expanded": sum(int(r.get("expert_expanded_records") or 0) for r in step_summaries),
        "candidate_regions": sum(int(r.get("candidate_region_records") or 0) for r in step_summaries),
        "active_region_id_builds": sum(int(r.get("active_region_ids_build_records") or 0) for r in step_summaries),
        "cold_candidate_register_calls": sum(int(r.get("cold_candidate_register_calls") or 0) for r in step_summaries),
        "resident_skip": sum(int(r.get("resident_skip_records") or 0) for r in step_summaries),
        "cooldown_skip": sum(int(r.get("cooldown_skip_records") or 0) for r in step_summaries),
        "cpu_unknown_regions": sum(int(r.get("cpu_unknown_region_records") or 0) for r in step_summaries),
        "prefetch_residency_register": prefetch_residency_actions.get("residency_register_region", 0),
        "prefetch_residency_access": prefetch_residency_actions.get("residency_access", 0),
        "prefetch_residency_request": prefetch_residency_actions.get("residency_request", 0),
        "prefetch_issued": weight_actions.get("prefetch_issued", 0),
        "coordinator_reject": weight_actions.get("coordinator_reject", 0),
        "reaccess_after_evict": sum(int(r.get("reaccess_after_evict_records") or 0) for r in step_summaries),
        "active_evict_issued": residency_actions.get("residency_weight_active_evict_issued", 0),
        "active_evict_skip_recent": sum(
            int((r.get("metadata") or {}).get("skipped_recent_records") or 0)
            + int((r.get("metadata") or {}).get("skipped_recent_expert_records") or 0)
            for r in residency_records
            if r.get("action") == "residency_weight_active_evict_noop"
        ),
        "victim_skip_recent": residency_actions.get("residency_weight_victim_skip_recent", 0),
    }

rows = [load(case) for case in cases]
fields = list(rows[0].keys())
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage M Region State Policy Check",
    "",
    f"- summary_csv: `{out_csv}`",
    "",
    "| case | tok/s | TPOT ms | expert skip | region candidates | active id builds | cold reg calls | residency req | prefetch issued | reaccess after evict | active evict issued | evict skip recent |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {tok_s} | {tpot_ms} | {expert_resident_skip} | "
        "{candidate_regions} | {active_region_id_builds} | "
        "{cold_candidate_register_calls} | "
        "{prefetch_residency_request} | {prefetch_issued} | "
        "{reaccess_after_evict} | {active_evict_issued} | "
        "{active_evict_skip_recent} |".format(**row)
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
PY

echo "Region state policy check finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
