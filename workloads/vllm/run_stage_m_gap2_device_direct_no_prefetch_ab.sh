#!/usr/bin/env bash
# Strict A/B for gap#2: can device_direct reduce frequent runtime-scratch
# UVM faults when dynamic prefetch/offload/residency policies are disabled?

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_gap2_device_direct_no_prefetch_ab_run}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-5}"
PROBE_PROMPTS="${PROBE_PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TARGET_GAP="${TARGET_GAP:-2}"
TARGET_CLASS="${TARGET_CLASS:-gap_hot_runtime_scratch}"
TARGET_PHASES="${TARGET_PHASES:-enabled:attention,enabled:moe,enabled:model_forward}"
DEVICE_DIRECT_MIN_BYTES="${DEVICE_DIRECT_MIN_BYTES:-4096}"
DEVICE_DIRECT_MAX_BYTES="${DEVICE_DIRECT_MAX_BYTES:-67108864}"
DEVICE_DIRECT_MAX_TOTAL_BYTES="${DEVICE_DIRECT_MAX_TOTAL_BYTES:-1073741824}"
DEVICE_DIRECT_BACKEND="${DEVICE_DIRECT_BACKEND:-cuda_malloc}"
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="${DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-}"
SCRATCH_POOL_BUDGET_BYTES="${SCRATCH_POOL_BUDGET_BYTES:-1073741824}"
SCRATCH_POOL_MODE="${SCRATCH_POOL_MODE:-enforce}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

mkdir -p "$RUN_ROOT"

COMMON_DISABLE_ARGS=(
  --uvm-policy-enable 1
  --uvm-policy-mode trace_only
  --uvm-policy-warmup-advise-gpu 0
  --uvm-kv-runtime-enable 0
  --uvm-kv-runtime-prefix-evict-enable 0
  --uvm-weight-prefetch-enable 0
  --uvm-weight-prefetch-mode trace_only
  --uvm-weight-prefetch-trace-enable 0
  --uvm-weight-prefetch-hit-trace-enable 0
  --uvm-weight-prefetch-hit-driver-query-enable 0
  --uvm-weight-batch-executor-enable 0
  --uvm-weight-executor "$WEIGHT_EXECUTOR"
  --uvm-weight-offload-enable 0
  --uvm-weight-offload-mode trace_only
  --uvm-init-kv-sizing-enable 0
  --uvm-init-weight-offload-enable 0
  --uvm-init-weight-offload-mode trace_only
  --uvm-init-weight-hot-prefetch-enable 0
  --uvm-init-weight-hot-prefetch-real-query-enable 0
  --uvm-hot-residency-stage-trace-enable 0
  --uvm-pool-coordinator-enable 0
  --uvm-pool-coordinator-trace-enable 0
  --uvm-residency-enable 0
  --uvm-residency-trace-enable 0
  --uvm-residency-kv-sizing-enable 0
  --uvm-residency-kv-metadata-enable 0
  --uvm-residency-kv-data-offload-enable 0
  --uvm-residency-kv-prefetch-back-enable 0
  --uvm-residency-weight-enforce-hot-set 0
  --uvm-residency-weight-active-evict-enable 0
  --uvm-pool-registry-enable 0
  --uvm-moe-routing-trace-enable 0
)

run_case() {
  local label="$1"
  local action="$2"
  local device_direct_enable="$3"
  local scratch_pool_enable="$4"
  local case_dir="$RUN_ROOT/$label"

  mkdir -p "$case_dir"

  echo "==========================================================="
  echo "Run $label"
  echo "  action=$action"
  echo "  device_direct_enable=$device_direct_enable"
  echo "  scratch_pool_enable=$scratch_pool_enable"
  echo "  target_gap=$TARGET_GAP target_class=$TARGET_CLASS"
  echo "  target_phases=$TARGET_PHASES"
  echo "==========================================================="

  pkill -f "vllm bench serve" || true
  pkill -f "vllm serve $MODEL" || true
  python3 /home/ubuntu/nvidia-uvm-gpu/workloads/cleanup_gpu.py || true

  ./run_kv_fault_ratio.sh \
    --model "$MODEL" \
    --mode trace \
    --with-address-log \
    --trace-log "$case_dir/uvm_fault_stats.log" \
    --address-trace-log "$case_dir/uvm_fault_addrs.log" \
    --allocator-log "$case_dir/vllm_uvm_allocator_trace.log" \
    --bench-log "$case_dir/vllm_bench.log" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --uvm-trace-min-bytes 4096 \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --uvm-gap-watch-name "gap${TARGET_GAP}_${label}_no_prefetch" \
    --uvm-gap-watch-all-classes 1 \
    --uvm-gap-watch-min-bytes 4096 \
    --uvm-device-direct-enable "$device_direct_enable" \
    --uvm-device-direct-min-bytes "$DEVICE_DIRECT_MIN_BYTES" \
    --uvm-device-direct-max-bytes "$DEVICE_DIRECT_MAX_BYTES" \
    --uvm-device-direct-max-total-bytes "$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
    --uvm-device-direct-backend "$DEVICE_DIRECT_BACKEND" \
    --uvm-device-direct-pool-release-threshold "$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" \
    --uvm-device-direct-target-phases "$TARGET_PHASES" \
    --uvm-scratch-pool-enable "$scratch_pool_enable" \
    --uvm-scratch-pool-budget-bytes "$SCRATCH_POOL_BUDGET_BYTES" \
    --uvm-scratch-pool-mode "$SCRATCH_POOL_MODE" \
    --uvm-scratch-pool-target-phases "$TARGET_PHASES" \
    "${COMMON_DISABLE_ARGS[@]}" \
    --auto-gap-watch-enable 1 \
    --auto-gap-watch-probe-prompts "$PROBE_PROMPTS" \
    --auto-gap-watch-target-gap "$TARGET_GAP" \
    --auto-gap-watch-fallback-to-hottest 0 \
    --auto-gap-watch-policy-action-override "$action" \
    --auto-gap-watch-target-class-override "$TARGET_CLASS" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --gap-watch-metrics-summary-json "$case_dir/gap_watch_metrics.json" \
    --auto-gap-watch-summary-json "$case_dir/auto_gap_watch_probe.json" \
    --auto-gap-watch-post-main-summary-json "$case_dir/auto_gap_watch_post_main.json"
}

echo "==========================================================="
echo "Stage M gap#2 device_direct no-prefetch A/B"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS probe_prompts=$PROBE_PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "target_gap=$TARGET_GAP target_class=$TARGET_CLASS"
echo "target_phases=$TARGET_PHASES"
echo "device_direct_bytes=min:$DEVICE_DIRECT_MIN_BYTES max:$DEVICE_DIRECT_MAX_BYTES total:$DEVICE_DIRECT_MAX_TOTAL_BYTES backend:$DEVICE_DIRECT_BACKEND"
echo "scratch_pool_budget_bytes=$SCRATCH_POOL_BUDGET_BYTES mode=$SCRATCH_POOL_MODE"
echo "==========================================================="

run_case baseline_trace device_direct_trace 0 0
run_case device_direct device_direct 1 1

COMPARISON_JSON="$RUN_ROOT/gap${TARGET_GAP}_device_direct_no_prefetch_comparison.json"
SUMMARY_CSV="$RUN_ROOT/gap${TARGET_GAP}_device_direct_no_prefetch_summary.csv"
SUMMARY_MD="$RUN_ROOT/gap${TARGET_GAP}_device_direct_no_prefetch_summary.md"

python3 "$SCRIPT_DIR/compare_stage_c_attention_p20_ab.py" \
  --trace-probe "$RUN_ROOT/baseline_trace/auto_gap_watch_probe.json" \
  --trace-post-main "$RUN_ROOT/baseline_trace/auto_gap_watch_post_main.json" \
  --trace-metrics "$RUN_ROOT/baseline_trace/gap_watch_metrics.json" \
  --trace-bench-log "$RUN_ROOT/baseline_trace/vllm_bench.log" \
  --device-probe "$RUN_ROOT/device_direct/auto_gap_watch_probe.json" \
  --device-post-main "$RUN_ROOT/device_direct/auto_gap_watch_post_main.json" \
  --device-metrics "$RUN_ROOT/device_direct/gap_watch_metrics.json" \
  --device-bench-log "$RUN_ROOT/device_direct/vllm_bench.log" \
  --output-json "$COMPARISON_JSON"

python3 - "$COMPARISON_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$TARGET_GAP" <<'PY'
import csv
import json
import sys
from pathlib import Path

comparison_path = Path(sys.argv[1])
csv_path = Path(sys.argv[2])
md_path = Path(sys.argv[3])
target_gap = sys.argv[4]

data = json.loads(comparison_path.read_text(encoding="utf-8"))
baseline = data["trace_only"]
device = data["stage_c_attention_device_direct"]
comp = data["comparison"]

def get(run, *keys):
    current = run
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current

def pct(value):
    if value is None:
        return ""
    return f"{value * 100:.2f}"

rows = []
for label, run in (("baseline_trace", baseline), ("device_direct", device)):
    rows.append({
        "case": label,
        "gap_index": get(run, "main_gap", "gap_index"),
        "gap_faults": get(run, "main_gap", "faults"),
        "gap_fault_share_of_unknown": get(run, "main_gap", "fault_share_of_unknown"),
        "total_unknown_faults": get(run, "total_unknown_faults"),
        "device_direct_actual_records": get(run, "metrics", "device_direct_actual_records"),
        "device_direct_eligible_records": get(run, "metrics", "device_direct_eligible_records"),
        "device_direct_budget_reject_records": get(run, "metrics", "device_direct_budget_reject_records"),
        "gap_policy_success": get(run, "metrics", "gap_policy_success"),
        "gap_policy_fail": get(run, "metrics", "gap_policy_fail"),
        "placement_backend_counts": json.dumps(get(run, "metrics", "placement_backend_counts"), sort_keys=True),
        "output_tok_s": get(run, "bench_main", "output_throughput_tok_s"),
        "mean_tpot_ms": get(run, "bench_main", "mean_tpot_ms"),
        "mean_ttft_ms": get(run, "bench_main", "mean_ttft_ms"),
    })

fields = list(rows[0].keys())
csv_path.parent.mkdir(parents=True, exist_ok=True)
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

checks = comp.get("checks", {})
success_items = [
    ("target_gap_selected_baseline", get(baseline, "main_gap", "gap_index") == int(target_gap)),
    ("target_gap_selected_device", get(device, "main_gap", "gap_index") == int(target_gap)),
    ("device_direct_records_present", checks.get("device_direct_records_present")),
    ("device_gap_faults_lower_than_baseline", checks.get("device_gap_faults_lower_than_trace")),
    ("device_unknown_faults_lower_than_baseline", checks.get("device_unknown_faults_lower_than_trace")),
    ("device_policy_fail_zero", checks.get("device_policy_fail_zero")),
]

lines = [
    f"# Stage M gap#{target_gap} device_direct no-prefetch A/B",
    "",
    f"- comparison_json: `{comparison_path}`",
    f"- summary_csv: `{csv_path}`",
    "- dynamic prefetch/offload/residency policies: disabled; allocator policy kept in `trace_only` for gap-watch control.",
    "",
    "| case | gap | gap faults | unknown faults | actual direct | eligible direct | policy success/fail | tok/s | TPOT ms | TTFT ms |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {gap_index} | {gap_faults} | {total_unknown_faults} | "
        "{device_direct_actual_records} | {device_direct_eligible_records} | "
        "{gap_policy_success}/{gap_policy_fail} | {output_tok_s} | {mean_tpot_ms} | {mean_ttft_ms} |".format(**row)
    )
lines.extend([
    "",
    "## Delta: device_direct vs baseline_trace",
    f"- gap_fault_delta: {comp.get('gap_fault_delta')}",
    f"- gap_fault_delta_pct: {pct(comp.get('gap_fault_delta_pct'))}%",
    f"- unknown_fault_delta: {comp.get('unknown_fault_delta')}",
    f"- unknown_fault_delta_pct: {pct(comp.get('unknown_fault_delta_pct'))}%",
    f"- output_tok_s_delta_pct: {pct(comp.get('output_throughput_delta_pct'))}%",
    f"- mean_tpot_delta_pct: {pct(comp.get('mean_tpot_delta_pct'))}%",
    "",
    "## Checks",
])
for key, value in success_items:
    lines.append(f"- {key}: {value}")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md_path.read_text(encoding="utf-8"))
PY

echo "Gap#${TARGET_GAP} device_direct no-prefetch A/B finished."
echo "comparison_json: $COMPARISON_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
