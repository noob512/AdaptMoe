#!/usr/bin/env bash
# Matrix runner for cuda_malloc_async device_direct phase ablation.
# It reuses the p10 fault/perf A/B script and runs three phase sets:
#   1) moe + model_forward
#   2) attention + moe + model_forward
#   3) attention + moe + model_forward + compute_logits + sampler

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_device_direct_async_phase_matrix_p10_run}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-10}"
PROBE_PROMPTS="${PROBE_PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TARGET_GAP="${TARGET_GAP:-2}"
TARGET_CLASS="${TARGET_CLASS:-gap_hot_runtime_scratch}"

DEVICE_DIRECT_MIN_BYTES="${DEVICE_DIRECT_MIN_BYTES:-4096}"
DEVICE_DIRECT_MAX_BYTES="${DEVICE_DIRECT_MAX_BYTES:-67108864}"
DEVICE_DIRECT_MAX_TOTAL_BYTES="${DEVICE_DIRECT_MAX_TOTAL_BYTES:-1073741824}"
DEVICE_DIRECT_BACKEND="${DEVICE_DIRECT_BACKEND:-cuda_malloc_async}"
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="${DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-1073741824}"
SCRATCH_POOL_BUDGET_BYTES="${SCRATCH_POOL_BUDGET_BYTES:-1073741824}"
SCRATCH_POOL_MODE="${SCRATCH_POOL_MODE:-enforce}"

mkdir -p "$RUN_ROOT"

declare -a CASE_LABELS=(
  "moe_model_forward"
  "attention_moe_model_forward"
  "wide_all_decode_phases"
)

declare -a CASE_PHASES=(
  "enabled:moe,enabled:model_forward"
  "enabled:attention,enabled:moe,enabled:model_forward"
  "enabled:attention,enabled:moe,enabled:model_forward,enabled:compute_logits,enabled:sampler"
)

echo "==========================================================="
echo "Stage M device_direct cuda_malloc_async phase matrix"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS probe_prompts=$PROBE_PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "target_gap=$TARGET_GAP target_class=$TARGET_CLASS"
echo "device_direct_bytes=min:$DEVICE_DIRECT_MIN_BYTES max:$DEVICE_DIRECT_MAX_BYTES total:$DEVICE_DIRECT_MAX_TOTAL_BYTES backend:$DEVICE_DIRECT_BACKEND"
echo "device_direct_pool_release_threshold=$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD"
echo "scratch_pool_budget_bytes=$SCRATCH_POOL_BUDGET_BYTES mode=$SCRATCH_POOL_MODE"
echo "==========================================================="

for idx in "${!CASE_LABELS[@]}"; do
  label="${CASE_LABELS[$idx]}"
  phases="${CASE_PHASES[$idx]}"
  case_root="$RUN_ROOT/$label"

  echo "==========================================================="
  echo "Matrix case: $label"
  echo "target_phases=$phases"
  echo "case_root=$case_root"
  echo "==========================================================="

  RUN_ROOT="$case_root" \
  MODEL="$MODEL" \
  PROMPTS="$PROMPTS" \
  PROBE_PROMPTS="$PROBE_PROMPTS" \
  OUTPUT_LEN="$OUTPUT_LEN" \
  REQUEST_RATE="$REQUEST_RATE" \
  SERVER_MAX_NUM_SEQS="$SERVER_MAX_NUM_SEQS" \
  SERVER_GPU_MEMORY_UTILIZATION="$SERVER_GPU_MEMORY_UTILIZATION" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  TARGET_GAP="$TARGET_GAP" \
  TARGET_CLASS="$TARGET_CLASS" \
  TARGET_PHASES="$phases" \
  DEVICE_DIRECT_MIN_BYTES="$DEVICE_DIRECT_MIN_BYTES" \
  DEVICE_DIRECT_MAX_BYTES="$DEVICE_DIRECT_MAX_BYTES" \
  DEVICE_DIRECT_MAX_TOTAL_BYTES="$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
  DEVICE_DIRECT_BACKEND="$DEVICE_DIRECT_BACKEND" \
  DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" \
  SCRATCH_POOL_BUDGET_BYTES="$SCRATCH_POOL_BUDGET_BYTES" \
  SCRATCH_POOL_MODE="$SCRATCH_POOL_MODE" \
  ./run_stage_m_device_direct_wide_phase_fault_perf_ab_p10.sh
done

SUMMARY_JSON="$RUN_ROOT/async_phase_matrix_summary.json"
SUMMARY_CSV="$RUN_ROOT/async_phase_matrix_summary.csv"
SUMMARY_MD="$RUN_ROOT/async_phase_matrix_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])

case_labels = [
    "moe_model_forward",
    "attention_moe_model_forward",
    "wide_all_decode_phases",
]

case_phases = {
    "moe_model_forward": "enabled:moe,enabled:model_forward",
    "attention_moe_model_forward": "enabled:attention,enabled:moe,enabled:model_forward",
    "wide_all_decode_phases": "enabled:attention,enabled:moe,enabled:model_forward,enabled:compute_logits,enabled:sampler",
}

def pct_change(base, direct):
    if base in (None, 0) or direct is None:
        return None
    return 100.0 * (direct - base) / base

def fmt_pct(value):
    return "n/a" if value is None else f"{value:.2f}%"

def get(case, key):
    return case.get(key)

rows = []
payload = {}
for label in case_labels:
    path = run_root / label / "device_direct_wide_phase_fault_perf_comparison.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    payload[label] = data
    cases = {case["case"]: case for case in data["cases"]}
    base = cases["baseline_trace"]
    direct = cases["device_direct"]
    row = {
        "phase_case": label,
        "target_phases": case_phases[label],
        "base_tok_s": get(base, "output_tok_s"),
        "direct_tok_s": get(direct, "output_tok_s"),
        "tok_s_delta_pct": pct_change(get(base, "output_tok_s"), get(direct, "output_tok_s")),
        "base_tpot_ms": get(base, "mean_tpot_ms"),
        "direct_tpot_ms": get(direct, "mean_tpot_ms"),
        "tpot_delta_pct": pct_change(get(base, "mean_tpot_ms"), get(direct, "mean_tpot_ms")),
        "base_ttft_ms": get(base, "mean_ttft_ms"),
        "direct_ttft_ms": get(direct, "mean_ttft_ms"),
        "ttft_delta_pct": pct_change(get(base, "mean_ttft_ms"), get(direct, "mean_ttft_ms")),
        "base_parsed_faults": get(base, "parsed_faults_main"),
        "direct_parsed_faults": get(direct, "parsed_faults_main"),
        "parsed_fault_delta_pct": pct_change(get(base, "parsed_faults_main"), get(direct, "parsed_faults_main")),
        "base_unknown_faults": get(base, "unknown_faults_main"),
        "direct_unknown_faults": get(direct, "unknown_faults_main"),
        "unknown_fault_delta_pct": pct_change(get(base, "unknown_faults_main"), get(direct, "unknown_faults_main")),
        "base_gap2_faults": get(base, "gap2_faults_main"),
        "direct_gap2_faults": get(direct, "gap2_faults_main"),
        "gap2_fault_delta_pct": pct_change(get(base, "gap2_faults_main"), get(direct, "gap2_faults_main")),
        "base_weight_faults": get(base, "weight_faults_main"),
        "direct_weight_faults": get(direct, "weight_faults_main"),
        "weight_fault_delta_pct": pct_change(get(base, "weight_faults_main"), get(direct, "weight_faults_main")),
        "base_kv_faults": get(base, "kv_cache_faults_main"),
        "direct_kv_faults": get(direct, "kv_cache_faults_main"),
        "kv_fault_delta_pct": pct_change(get(base, "kv_cache_faults_main"), get(direct, "kv_cache_faults_main")),
        "base_raw_delta": get(base, "delta_faults"),
        "direct_raw_delta": get(direct, "delta_faults"),
        "raw_delta_pct": pct_change(get(base, "delta_faults"), get(direct, "delta_faults")),
        "base_dedup_delta": get(base, "delta_after_dedup"),
        "direct_dedup_delta": get(direct, "delta_after_dedup"),
        "dedup_delta_pct": pct_change(get(base, "delta_after_dedup"), get(direct, "delta_after_dedup")),
        "direct_actual_records": get(direct, "device_direct_actual_records"),
        "direct_budget_reject_records": get(direct, "device_direct_budget_reject_records"),
        "direct_peak_live_mib": (
            float(get(direct, "device_direct_peak_live_bytes")) / (1024 * 1024)
            if get(direct, "device_direct_peak_live_bytes") is not None
            else None
        ),
        "scratch_direct_records": get(direct, "scratch_pool_device_direct_records"),
        "scratch_peak_live_mib": (
            float(get(direct, "scratch_pool_peak_live_bytes")) / (1024 * 1024)
            if get(direct, "scratch_pool_peak_live_bytes") is not None
            else None
        ),
    }
    rows.append(row)

summary_json.write_text(json.dumps({"run_root": str(run_root), "cases": rows, "raw": payload}, indent=2, sort_keys=True) + "\n")

fields = list(rows[0].keys())
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage M Device Direct cuda_malloc_async Phase Matrix",
    "",
    f"- run_root: `{run_root}`",
    f"- summary_json: `{summary_json}`",
    f"- summary_csv: `{summary_csv}`",
    "- backend: `cuda_malloc_async`",
    "- pool_release_threshold: `1073741824` unless overridden",
    "- baseline per phase: trace only; treatment per phase: real device_direct",
    "",
    "| phase case | tok/s delta | TPOT delta | parsed faults delta | gap2 delta | unknown delta | weight delta | KV delta | raw delta | dedup delta | direct actual | direct peak MiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {phase_case} | {tok} | {tpot} | {parsed} | {gap2} | {unknown} | "
        "{weight} | {kv} | {raw} | {dedup} | {actual} | {peak} |".format(
            phase_case=row["phase_case"],
            tok=fmt_pct(row["tok_s_delta_pct"]),
            tpot=fmt_pct(row["tpot_delta_pct"]),
            parsed=fmt_pct(row["parsed_fault_delta_pct"]),
            gap2=fmt_pct(row["gap2_fault_delta_pct"]),
            unknown=fmt_pct(row["unknown_fault_delta_pct"]),
            weight=fmt_pct(row["weight_fault_delta_pct"]),
            kv=fmt_pct(row["kv_fault_delta_pct"]),
            raw=fmt_pct(row["raw_delta_pct"]),
            dedup=fmt_pct(row["dedup_delta_pct"]),
            actual=row["direct_actual_records"],
            peak="" if row["direct_peak_live_mib"] is None else f"{row['direct_peak_live_mib']:.2f}",
        )
    )

lines.extend([
    "",
    "## Per-Case Values",
    "",
    "| phase case | base tok/s | direct tok/s | base TPOT | direct TPOT | base gap2 | direct gap2 | base weight | direct weight | base KV | direct KV |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
])
for row in rows:
    lines.append(
        "| {phase_case} | {base_tok_s} | {direct_tok_s} | {base_tpot_ms} | {direct_tpot_ms} | "
        "{base_gap2_faults} | {direct_gap2_faults} | {base_weight_faults} | {direct_weight_faults} | "
        "{base_kv_faults} | {direct_kv_faults} |".format(**row)
    )

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "Stage M device_direct cuda_malloc_async phase matrix finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
