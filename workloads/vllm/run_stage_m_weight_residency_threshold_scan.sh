#!/usr/bin/env bash
# Scan MoE weight GPU residency levels to find the throughput cliff.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCAN_ROOT="${SCAN_ROOT:-log/stage_m_weight_residency_threshold_scan_25_27_30}"
PROMPTS="${PROMPTS:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
ACTIVE_EXPERTS_PER_LAYER="${ACTIVE_EXPERTS_PER_LAYER:-32}"
WEIGHT_PREFETCH_TRACE_ENABLE="${WEIGHT_PREFETCH_TRACE_ENABLE:-0}"
WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="${WEIGHT_ROUTER_TOPK_EVERY_N_STEPS:-4}"
POOL_COORDINATOR_TRACE_ENABLE="${POOL_COORDINATOR_TRACE_ENABLE:-0}"

# Format: label:weight_budget_bytes:hot_experts:cold_experts
CASES="${CASES:-w25_hot116:26843545600:116:12 w27_hot124:28991029248:124:4 w30_hot128:32212254720:128:0}"

mkdir -p "$SCAN_ROOT"
SUMMARY_CSV="$SCAN_ROOT/threshold_summary.csv"
SUMMARY_MD="$SCAN_ROOT/threshold_summary.md"

echo "label,weight_budget_gib,hot,cold,phase_f_exit_code,snapshots,weight_gpu_gib,weight_cpu_gib,output_tok_s,mean_tpot_ms,mean_ttft_ms,delta_after_dedup,init_offload_issued" > "$SUMMARY_CSV"

for case_spec in $CASES; do
  IFS=: read -r label weight_bytes hot cold <<< "$case_spec"
  run_root="$SCAN_ROOT/${label}_run"
  log_file="$SCAN_ROOT/${label}.log"

  echo "==========================================================="
  echo "Running $label: weight_bytes=$weight_bytes hot=$hot cold=$cold"
  echo "Run root: $run_root"
  echo "Log file: $log_file"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve Qwen/Qwen3-30B-A3B-FP8" || true

  VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
  UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
  WEIGHT_GPU_BYTES="$weight_bytes" \
  KV_GPU_BYTES="$KV_GPU_BYTES" \
  HOT_EXPERTS_PER_LAYER="$hot" \
  COLD_EXPERTS_PER_LAYER="$cold" \
  ACTIVE_EXPERTS_PER_LAYER="$ACTIVE_EXPERTS_PER_LAYER" \
  WEIGHT_PREFETCH_TRACE_ENABLE="$WEIGHT_PREFETCH_TRACE_ENABLE" \
  WEIGHT_ROUTER_TOPK_EVERY_N_STEPS="$WEIGHT_ROUTER_TOPK_EVERY_N_STEPS" \
  POOL_COORDINATOR_TRACE_ENABLE="$POOL_COORDINATOR_TRACE_ENABLE" \
  RUN_ROOT="$run_root" \
  LOG_FILE="$log_file" \
  PROMPTS="$PROMPTS" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  "$SCRIPT_DIR/run_stage_m_active_experts_residency_10g_quiet.sh"

  python3 - "$run_root" "$label" "$weight_bytes" "$hot" "$cold" >> "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
label = sys.argv[2]
weight_bytes = int(sys.argv[3])
hot = sys.argv[4]
cold = sys.argv[5]

def gib(value):
    return float(value or 0) / (1024 ** 3)

residency_path = run_root / "real_residency_summary.json"
phase_f_exit_code = ""
snapshots = ""
weight_gpu_gib = ""
weight_cpu_gib = ""
if residency_path.exists():
    data = json.loads(residency_path.read_text(encoding="utf-8"))
    phase_f_exit_code = data.get("phase_f_exit_code", "")
    snapshots = data.get("snapshots", "")
    latest = data.get("latest") or {}
    weights = (latest.get("by_pool") or {}).get("weights") or {}
    weight_gpu_gib = f"{gib(weights.get('gpu_bytes')):.3f}"
    weight_cpu_gib = f"{gib(weights.get('cpu_bytes')):.3f}"

summary_csv = run_root / "routing_phase_f_full_impl" / "summary.csv"
output_tok_s = ""
mean_tpot_ms = ""
mean_ttft_ms = ""
delta_after_dedup = ""
init_offload_issued = ""
if summary_csv.exists():
    with summary_csv.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == "full_impl_enforce":
                output_tok_s = row.get("output_tok_s", "")
                mean_tpot_ms = row.get("mean_tpot_ms", "")
                mean_ttft_ms = row.get("mean_ttft_ms", "")
                delta_after_dedup = row.get("delta_after_dedup", "")
                init_offload_issued = row.get("init_weight_offload_offload_cpu_issued", "")
                break

print(",".join([
    label,
    f"{gib(weight_bytes):.3f}",
    hot,
    cold,
    str(phase_f_exit_code),
    str(snapshots),
    str(weight_gpu_gib),
    str(weight_cpu_gib),
    str(output_tok_s),
    str(mean_tpot_ms),
    str(mean_ttft_ms),
    str(delta_after_dedup),
    str(init_offload_issued),
]))
PY

  echo "Finished $label"
done

python3 - "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
from pathlib import Path

summary_csv = Path(sys.argv[1])
summary_md = Path(sys.argv[2])
rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))

lines = [
    "# Stage M Weight Residency Threshold Scan",
    "",
    f"- summary_csv: `{summary_csv}`",
    "",
    "| label | budget GiB | hot | cold | GPU GiB | CPU GiB | tok/s | TPOT ms | faults |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {label} | {weight_budget_gib} | {hot} | {cold} | "
        "{weight_gpu_gib} | {weight_cpu_gib} | {output_tok_s} | "
        "{mean_tpot_ms} | {delta_after_dedup} |".format(**row)
    )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "Threshold scan finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
