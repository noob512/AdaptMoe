#!/usr/bin/env bash
# A/B test for runtime MoE weight prefetch effectiveness.
# Keeps active eviction disabled so prefetch can be measured without
# eviction/ping-pong effects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_prefetch_effectiveness_ab_run}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"  # 24 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"           # 2 GiB
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-112}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-536870912}"
WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="${WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER:-32}"
WEIGHT_PREFETCH_SKIP_RESIDENT="${WEIGHT_PREFETCH_SKIP_RESIDENT:-1}"
WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE="${WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE:-1}"
WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS="${WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS:-8}"
WEIGHT_EVICT_PROTECT_RECENT_STEPS="${WEIGHT_EVICT_PROTECT_RECENT_STEPS:-8}"
COORDINATOR_WEIGHT_BYTES_PER_STEP="${COORDINATOR_WEIGHT_BYTES_PER_STEP:-536870912}"
COORDINATOR_GLOBAL_BYTES_PER_STEP="${COORDINATOR_GLOBAL_BYTES_PER_STEP:-1073741824}"

mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  local prefetch_enable="$2"
  local coordinator_enable="$3"
  local prefetch_trace="$4"
  local coordinator_trace="$5"
  local router_every="$6"
  local case_root="$RUN_ROOT/$label"
  local log_file="$RUN_ROOT/${label}.log"

  echo "==========================================================="
  echo "Running $label"
  echo "  runtime_prefetch=$prefetch_enable coordinator=$coordinator_enable"
  echo "  prefetch_trace=$prefetch_trace coordinator_trace=$coordinator_trace router_every=$router_every expert_cache=$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE"
  echo "  run_root=$case_root"
  echo "  log_file=$log_file"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP" \
  WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER="$WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER" \
  COORDINATOR_WEIGHT_BYTES_PER_STEP="$COORDINATOR_WEIGHT_BYTES_PER_STEP" \
  COORDINATOR_GLOBAL_BYTES_PER_STEP="$COORDINATOR_GLOBAL_BYTES_PER_STEP" \
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
    --weight-runtime-prefetch-enable "$prefetch_enable" \
    --weight-prefetch-trace-enable "$prefetch_trace" \
    --weight-router-topk-every-n-steps "$router_every" \
    --weight-prefetch-skip-resident "$WEIGHT_PREFETCH_SKIP_RESIDENT" \
    --weight-prefetch-expert-cache-enable "$WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE" \
    --weight-prefetch-region-cooldown-steps "$WEIGHT_PREFETCH_REGION_COOLDOWN_STEPS" \
    --weight-evict-protect-recent-steps "$WEIGHT_EVICT_PROTECT_RECENT_STEPS" \
    --scratch-device-direct-enable 0 \
    --scratch-pool-enable 0 \
    --residency-enable 1 \
    --residency-trace-enable 1 \
    --kv-runtime-enable 0 \
    --pool-coordinator-enable "$coordinator_enable" \
    --pool-coordinator-trace-enable "$coordinator_trace" \
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
    --request-rate 1
}

run_case baseline_no_prefetch 0 1 0 0 4
run_case prefetch_no_coordinator_trace 1 0 1 0 1
run_case prefetch_with_coordinator_trace 1 1 1 1 1
run_case prefetch_quiet_every4 1 1 0 0 4

SUMMARY_CSV="$RUN_ROOT/prefetch_effectiveness_summary.csv"
SUMMARY_MD="$RUN_ROOT/prefetch_effectiveness_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from collections import Counter
from pathlib import Path

run_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])

cases = [
    "baseline_no_prefetch",
    "prefetch_no_coordinator_trace",
    "prefetch_with_coordinator_trace",
    "prefetch_quiet_every4",
]

def as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def read_jsonl(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records

def load_summary_row(case_root):
    summary = case_root / "routing_phase_f_full_impl" / "summary.csv"
    if not summary.exists():
        return {}
    with summary.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                return row
    return {}

def latest_weight_residency_gib(case_root):
    path = case_root / "real_residency_summary.json"
    if not path.exists():
        return (None, None)
    data = json.loads(path.read_text(encoding="utf-8"))
    latest = data.get("latest") or {}
    weights = (latest.get("by_pool") or {}).get("weights") or {}
    return (
        float(weights.get("gpu_bytes") or 0) / (1024 ** 3),
        float(weights.get("cpu_bytes") or 0) / (1024 ** 3),
    )

def load_case(label):
    case_root = run_root / label
    full_root = case_root / "routing_phase_f_full_impl"
    run_dir = full_root / "runs" / "full_impl_enforce_rep1"
    row = load_summary_row(case_root)
    prefetch_records = read_jsonl(
        run_dir / "vllm_uvm_weight_prefetch_full_impl_enforce_rep1.jsonl"
    )
    residency_records = read_jsonl(
        run_dir / "vllm_uvm_pool_residency_full_impl_enforce_rep1.jsonl"
    )
    coord_records = read_jsonl(
        run_dir / "vllm_uvm_pool_coordinator_full_impl_enforce_rep1.jsonl"
    )
    prefetch_actions = Counter(str(r.get("action")) for r in prefetch_records)
    prefetch_sources = Counter(
        str(r.get("prefetch_source"))
        for r in prefetch_records
        if r.get("prefetch_source") is not None
    )
    prefetch_executors = Counter(
        str(r.get("executor")) for r in prefetch_records if r.get("executor")
    )
    step_summaries = [
        r for r in prefetch_records if r.get("action") == "step_summary"
    ]
    coord_outcomes = Counter(
        str(r.get("outcome")) for r in coord_records if r.get("action") == "coordinator_request"
    )
    coord_reasons = Counter(
        str(r.get("reason")) for r in coord_records if r.get("action") == "coordinator_request"
    )
    prefetch_residency_records = [
        r
        for r in residency_records
        if (r.get("metadata") or {}).get("source_action") == "prefetch"
    ]
    prefetch_residency_actions = Counter(
        str(r.get("action")) for r in prefetch_residency_records
    )
    issued_bytes = sum(
        int(r.get("bytes") or 0)
        for r in prefetch_records
        if r.get("action") == "prefetch_issued"
    )
    candidate_regions = sum(
        int(r.get("candidate_region_records") or 0) for r in step_summaries
    )
    candidate_experts = sum(
        int(r.get("candidate_expert_records") or 0) for r in step_summaries
    )
    expert_resident_skip = sum(
        int(r.get("expert_resident_skip_records") or 0) for r in step_summaries
    )
    expert_expanded = sum(
        int(r.get("expert_expanded_records") or 0) for r in step_summaries
    )
    active_region_builds = sum(
        int(r.get("active_region_ids_build_records") or 0) for r in step_summaries
    )
    active_region_ids = sum(
        int(r.get("active_region_id_records") or 0) for r in step_summaries
    )
    cold_candidate_register_calls = sum(
        int(r.get("cold_candidate_register_calls") or 0) for r in step_summaries
    )
    cold_candidate_register_records = sum(
        int(r.get("cold_candidate_register_records") or 0) for r in step_summaries
    )
    resident_skip = sum(
        int(r.get("resident_skip_records") or 0) for r in step_summaries
    )
    cooldown_skip = sum(
        int(r.get("cooldown_skip_records") or 0) for r in step_summaries
    )
    cpu_unknown_regions = sum(
        int(r.get("cpu_unknown_region_records") or 0) for r in step_summaries
    )
    reaccess_after_evict = sum(
        int(r.get("reaccess_after_evict_records") or 0) for r in step_summaries
    )
    issued_experts = {
        (r.get("layer_name"), r.get("local_expert_id"), r.get("role"))
        for r in prefetch_records
        if r.get("action") == "prefetch_issued"
    }
    gpu_gib, cpu_gib = latest_weight_residency_gib(case_root)
    return {
        "case": label,
        "exit_code": row.get("exit_code"),
        "tok_s": as_float(row.get("output_tok_s")),
        "tpot_ms": as_float(row.get("mean_tpot_ms")),
        "ttft_ms": as_float(row.get("mean_ttft_ms")),
        "faults_dedup": as_float(row.get("delta_after_dedup")),
        "kv_faults_dedup": as_float(row.get("delta_kv_after_dedup")),
        "summary_weight_prefetch_issued": as_float(row.get("weight_prefetch_issued")),
        "trace_prefetch_issued": prefetch_actions.get("prefetch_issued", 0),
        "trace_prefetch_records": len(prefetch_records),
        "router_topk_records": prefetch_sources.get("router_topk", 0),
        "prefetch_candidate_experts": candidate_experts,
        "prefetch_expert_resident_skip": expert_resident_skip,
        "prefetch_expert_expanded": expert_expanded,
        "prefetch_candidate_regions": candidate_regions,
        "prefetch_resident_skip": resident_skip,
        "prefetch_cooldown_skip": cooldown_skip,
        "prefetch_cpu_unknown_regions": cpu_unknown_regions,
        "reaccess_after_evict": reaccess_after_evict,
        "active_region_id_builds": active_region_builds,
        "active_region_ids": active_region_ids,
        "cold_candidate_register_calls": cold_candidate_register_calls,
        "cold_candidate_register_records": cold_candidate_register_records,
        "prefetch_residency_register_records": prefetch_residency_actions.get("residency_register_region", 0),
        "prefetch_residency_access_records": prefetch_residency_actions.get("residency_access", 0),
        "prefetch_residency_request_records": prefetch_residency_actions.get("residency_request", 0),
        "driver_batch_records": prefetch_executors.get("driver_batch", 0),
        "prefetch_issued_mib": issued_bytes / (1024 ** 2),
        "prefetch_unique_layer_expert_role": len(issued_experts),
        "coordinator_requests": sum(coord_outcomes.values()),
        "coordinator_grants": coord_outcomes.get("coordinator_grant", 0),
        "coordinator_denies": coord_outcomes.get("coordinator_deny", 0),
        "coordinator_top_reasons": ";".join(
            f"{key}:{value}" for key, value in coord_reasons.most_common(4)
        ),
        "weight_gpu_gib": gpu_gib,
        "weight_cpu_gib": cpu_gib,
    }

rows = [load_case(label) for label in cases]
fields = [
    "case",
    "exit_code",
    "tok_s",
    "tpot_ms",
    "ttft_ms",
    "faults_dedup",
    "kv_faults_dedup",
    "summary_weight_prefetch_issued",
    "trace_prefetch_issued",
    "trace_prefetch_records",
    "router_topk_records",
    "prefetch_candidate_experts",
    "prefetch_expert_resident_skip",
    "prefetch_expert_expanded",
    "prefetch_candidate_regions",
    "prefetch_resident_skip",
    "prefetch_cooldown_skip",
    "prefetch_cpu_unknown_regions",
    "reaccess_after_evict",
    "active_region_id_builds",
    "active_region_ids",
    "cold_candidate_register_calls",
    "cold_candidate_register_records",
    "prefetch_residency_register_records",
    "prefetch_residency_access_records",
    "prefetch_residency_request_records",
    "driver_batch_records",
    "prefetch_issued_mib",
    "prefetch_unique_layer_expert_role",
    "coordinator_requests",
    "coordinator_grants",
    "coordinator_denies",
    "coordinator_top_reasons",
    "weight_gpu_gib",
    "weight_cpu_gib",
]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

base = rows[0]
def pct(row, key):
    b = base.get(key)
    v = row.get(key)
    if not isinstance(b, (int, float)) or not isinstance(v, (int, float)) or b == 0:
        return None
    return (v - b) / b * 100.0

lines = [
    "# Stage M Prefetch Effectiveness A/B",
    "",
    f"- summary_csv: `{out_csv}`",
    "",
    "| case | tok/s | TPOT ms | faults dedup | expert skip | region candidates | active id builds | cold reg calls | residency req | prefetch issued | weight GPU GiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {tok_s} | {tpot_ms} | {faults_dedup} | "
        "{prefetch_expert_resident_skip} | {prefetch_candidate_regions} | "
        "{active_region_id_builds} | {cold_candidate_register_calls} | "
        "{prefetch_residency_request_records} | "
        "{trace_prefetch_issued} | {weight_gpu_gib} |".format(**row)
    )
lines.extend(["", "## Relative To Baseline"])
for row in rows[1:]:
    lines.append(
        "- {case}: tok_s_pct={tok}, tpot_pct={tpot}, faults_pct={faults}, "
        "expert_skip={expert_skip}, region_candidates={candidates}, "
        "active_id_builds={active_builds}, cold_reg_calls={cold_calls}, "
        "prefetch_issued={issued}, residency_requests={req}, coordinator_top_reasons={reasons}".format(
            case=row["case"],
            tok=pct(row, "tok_s"),
            tpot=pct(row, "tpot_ms"),
            faults=pct(row, "faults_dedup"),
            expert_skip=row["prefetch_expert_resident_skip"],
            candidates=row["prefetch_candidate_regions"],
            active_builds=row["active_region_id_builds"],
            cold_calls=row["cold_candidate_register_calls"],
            issued=row["trace_prefetch_issued"],
            req=row["prefetch_residency_request_records"],
            reasons=row["coordinator_top_reasons"],
        )
    )
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
PY

echo "Prefetch effectiveness A/B finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
