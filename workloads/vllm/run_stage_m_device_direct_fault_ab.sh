#!/usr/bin/env bash
# A/B test for scratch device_direct fault impact.
# Keeps init weight offload enabled and disables runtime prefetch/active eviction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_device_direct_fault_ab_run}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"  # 24 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"           # 2 GiB
HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-112}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-16}"

mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  local scratch_direct="$2"
  local scratch_pool="$3"
  local case_root="$RUN_ROOT/$label"
  local log_file="$RUN_ROOT/${label}.log"

  echo "==========================================================="
  echo "Running $label: scratch_device_direct=$scratch_direct scratch_pool=$scratch_pool"
  echo "Run root: $case_root"
  echo "Log file: $log_file"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
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
    --weight-runtime-prefetch-enable 0 \
    --weight-prefetch-trace-enable 0 \
    --weight-router-topk-every-n-steps 4 \
    --scratch-device-direct-enable "$scratch_direct" \
    --scratch-pool-enable "$scratch_pool" \
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
    --weight-active-evict-enable 0 \
    --weight-active-evict-max-experts-per-layer 0 \
    --weight-max-victims-per-step 0 \
    --server-max-num-seqs 1 \
    --request-rate 1
}

run_case scratch_uvm 0 0
run_case scratch_device_direct 1 1

SUMMARY_CSV="$RUN_ROOT/device_direct_fault_ab_summary.csv"
SUMMARY_MD="$RUN_ROOT/device_direct_fault_ab_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
out_md = Path(sys.argv[3])

def as_float(value):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None

def pct(numer, denom):
    if numer is None or denom in (None, 0):
        return None
    return 100.0 * numer / denom

def first_present(row, metrics, key):
    value = row.get(key)
    if value not in (None, "", "None"):
        return value
    return metrics.get(key)

def load_case(label):
    summary = run_root / label / "routing_phase_f_full_impl" / "summary.csv"
    row = {}
    if summary.exists():
        with summary.open("r", encoding="utf-8", errors="replace") as handle:
            for item in csv.DictReader(handle):
                if item.get("variant") == "full_impl_enforce":
                    row = item
                    break
    metrics = {}
    metrics_files = sorted(
        (run_root / label / "routing_phase_f_full_impl" / "runs").glob(
            "full_impl_enforce_rep*/gap_watch_metrics_full_impl_enforce_rep*.json"
        )
    )
    if metrics_files:
        metrics = json.loads(metrics_files[-1].read_text(encoding="utf-8"))
    values = {
        "label": label,
        "output_tok_s": as_float(row.get("output_tok_s")),
        "mean_tpot_ms": as_float(row.get("mean_tpot_ms")),
        "mean_ttft_ms": as_float(row.get("mean_ttft_ms")),
        "delta_faults": as_float(row.get("delta_faults")),
        "delta_after_dedup": as_float(row.get("delta_after_dedup")),
        "delta_kv_faults": as_float(row.get("delta_kv_faults")),
        "delta_kv_after_dedup": as_float(row.get("delta_kv_after_dedup")),
        "device_direct_actual_records": as_float(first_present(row, metrics, "device_direct_actual_records")),
        "device_direct_eligible_records": as_float(first_present(row, metrics, "device_direct_eligible_records")),
        "device_direct_budget_reject_records": as_float(first_present(row, metrics, "device_direct_budget_reject_records")),
        "scratch_pool_trace_records": as_float(first_present(row, metrics, "scratch_pool_trace_records")),
        "scratch_pool_eligible_records": as_float(first_present(row, metrics, "scratch_pool_eligible_records")),
        "scratch_pool_device_direct_records": as_float(first_present(row, metrics, "scratch_pool_device_direct_records")),
        "scratch_pool_budget_over_records": as_float(first_present(row, metrics, "scratch_pool_budget_over_records")),
        "scratch_pool_budget_reject_records": as_float(first_present(row, metrics, "scratch_pool_budget_reject_records")),
        "scratch_pool_peak_live_mib": (
            as_float(first_present(row, metrics, "scratch_pool_peak_live_bytes")) / (1024 ** 2)
            if as_float(first_present(row, metrics, "scratch_pool_peak_live_bytes")) is not None
            else None
        ),
        "init_weight_offload_offload_cpu_issued": as_float(row.get("init_weight_offload_offload_cpu_issued")),
        "weight_prefetch_issued": as_float(row.get("weight_prefetch_issued")),
    }
    values["kv_after_dedup_ratio_pct"] = pct(
        values["delta_kv_after_dedup"], values["delta_after_dedup"]
    )
    values["non_kv_after_dedup"] = (
        values["delta_after_dedup"] - values["delta_kv_after_dedup"]
        if values["delta_after_dedup"] is not None
        and values["delta_kv_after_dedup"] is not None
        else None
    )
    residency = run_root / label / "real_residency_summary.json"
    if residency.exists():
        data = json.loads(residency.read_text(encoding="utf-8"))
        latest = data.get("latest") or {}
        weights = (latest.get("by_pool") or {}).get("weights") or {}
        values["real_weight_gpu_gib"] = (
            float(weights.get("gpu_bytes") or 0) / (1024 ** 3)
        )
    else:
        values["real_weight_gpu_gib"] = None
    return values

rows = [load_case("scratch_uvm"), load_case("scratch_device_direct")]

fields = [
    "label",
    "output_tok_s",
    "mean_tpot_ms",
    "mean_ttft_ms",
    "delta_faults",
    "delta_after_dedup",
    "delta_kv_faults",
    "delta_kv_after_dedup",
    "kv_after_dedup_ratio_pct",
    "non_kv_after_dedup",
    "device_direct_actual_records",
    "device_direct_eligible_records",
    "device_direct_budget_reject_records",
    "scratch_pool_trace_records",
    "scratch_pool_eligible_records",
    "scratch_pool_device_direct_records",
    "scratch_pool_budget_over_records",
    "scratch_pool_budget_reject_records",
    "scratch_pool_peak_live_mib",
    "init_weight_offload_offload_cpu_issued",
    "weight_prefetch_issued",
    "real_weight_gpu_gib",
]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

base, direct = rows
def change(key):
    b = base.get(key)
    d = direct.get(key)
    if b in (None, 0) or d is None:
        return None
    return 100.0 * (d - b) / b

lines = [
    "# Stage M Device Direct Fault A/B",
    "",
    f"- summary_csv: `{out_csv}`",
    "",
    "| case | tok/s | TPOT ms | dedup faults | KV dedup faults | KV ratio % | non-KV dedup | device_direct actual | scratch direct | scratch eligible | scratch budget reject | scratch peak MiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {label} | {output_tok_s} | {mean_tpot_ms} | {delta_after_dedup} | "
        "{delta_kv_after_dedup} | {kv_after_dedup_ratio_pct} | "
        "{non_kv_after_dedup} | {device_direct_actual_records} | "
        "{scratch_pool_device_direct_records} | {scratch_pool_eligible_records} | "
        "{scratch_pool_budget_reject_records} | {scratch_pool_peak_live_mib} |".format(**row)
    )
lines.extend([
    "",
    "## Relative Change: device_direct vs scratch_uvm",
    f"- output_tok_s_pct: {change('output_tok_s')}",
    f"- mean_tpot_ms_pct: {change('mean_tpot_ms')}",
    f"- delta_after_dedup_pct: {change('delta_after_dedup')}",
    f"- non_kv_after_dedup_pct: {change('non_kv_after_dedup')}",
    f"- scratch_pool_device_direct_records_delta: {direct.get('scratch_pool_device_direct_records') - base.get('scratch_pool_device_direct_records') if direct.get('scratch_pool_device_direct_records') is not None and base.get('scratch_pool_device_direct_records') is not None else None}",
])
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
PY

echo "Device direct fault A/B finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
