#!/usr/bin/env bash
# A/B test for gap#2 runtime-scratch device_direct with the best wide-phase
# configuration found so far. Dynamic prefetch/offload/residency policies stay
# disabled; the only behavioral difference is real cudaMalloc-backed
# device_direct for the watched gap/scratch allocations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_device_direct_wide_phase_fault_perf_ab_p10_run}"
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
TARGET_PHASES="${TARGET_PHASES:-enabled:attention,enabled:moe,enabled:model_forward,enabled:compute_logits,enabled:sampler}"

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
  --uvm-weight-budget-cpu-offload-enable 0
  --uvm-weight-map-enable 1
  --uvm-moe-routing-trace-enable 0
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
)

run_case() {
  local label="$1"
  local action="$2"
  local device_direct_enable="$3"
  local scratch_pool_enable="$4"
  local case_dir="$RUN_ROOT/$label"

  mkdir -p "$case_dir/analysis"

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

  local device_direct_optional_args=()
  if [ -n "$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" ]; then
    device_direct_optional_args+=(
      --uvm-device-direct-pool-release-threshold "$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD"
    )
  fi

  ./run_kv_fault_ratio.sh \
    --model "$MODEL" \
    --mode trace \
    --with-address-log \
    --trace-log "$case_dir/uvm_fault_stats.log" \
    --address-trace-log "$case_dir/uvm_fault_addrs.log" \
    --address-log "$case_dir/vllm_uvm_address_regions.log" \
    --allocator-log "$case_dir/vllm_uvm_allocator_trace.log" \
    --server-log "$case_dir/vllm_server.log" \
    --bench-log "$case_dir/vllm_bench.log" \
    --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
    --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --uvm-trace-min-bytes 4096 \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --uvm-gap-watch-name "gap${TARGET_GAP}_${label}_wide_phase_ab" \
    --uvm-gap-watch-all-classes 1 \
    --uvm-gap-watch-min-bytes 4096 \
    --uvm-device-direct-enable "$device_direct_enable" \
    --uvm-device-direct-min-bytes "$DEVICE_DIRECT_MIN_BYTES" \
    --uvm-device-direct-max-bytes "$DEVICE_DIRECT_MAX_BYTES" \
    --uvm-device-direct-max-total-bytes "$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
    --uvm-device-direct-backend "$DEVICE_DIRECT_BACKEND" \
    --uvm-device-direct-target-phases "$TARGET_PHASES" \
    --uvm-scratch-pool-enable "$scratch_pool_enable" \
    --uvm-scratch-pool-budget-bytes "$SCRATCH_POOL_BUDGET_BYTES" \
    --uvm-scratch-pool-mode "$SCRATCH_POOL_MODE" \
    --uvm-scratch-pool-target-phases "$TARGET_PHASES" \
    "${device_direct_optional_args[@]}" \
    "${COMMON_DISABLE_ARGS[@]}" \
    --auto-gap-watch-enable 1 \
    --auto-gap-watch-probe-prompts "$PROBE_PROMPTS" \
    --auto-gap-watch-target-gap "$TARGET_GAP" \
    --auto-gap-watch-fallback-to-hottest 1 \
    --auto-gap-watch-policy-action-override "$action" \
    --auto-gap-watch-target-class-override "$TARGET_CLASS" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN" \
    --gap-watch-metrics-summary-json "$case_dir/gap_watch_metrics.json" \
    --auto-gap-watch-summary-json "$case_dir/auto_gap_watch_probe.json" \
    --auto-gap-watch-post-main-summary-json "$case_dir/auto_gap_watch_post_main.json" \
    2>&1 | tee "$case_dir/run.log"

  local main_start_line
  main_start_line="$(
    sed -n 's/^Auto gap-watch main start line: \([0-9][0-9]*\)$/\1/p' "$case_dir/run.log" | tail -n 1
  )"
  if [ -z "$main_start_line" ]; then
    echo "Could not find Auto gap-watch main start line in $case_dir/run.log" >&2
    return 1
  fi
  echo "$main_start_line" > "$case_dir/analysis/main_start_line.txt"

  python3 "$SCRIPT_DIR/analyze_uvm_fault_addresses.py" \
    --address-log "$case_dir/vllm_uvm_address_regions.log" \
    --fault-log "$case_dir/uvm_fault_addrs.log" \
    --regions-csv "$case_dir/analysis/regions.csv" \
    --faults-csv "$case_dir/analysis/faults.csv" \
    --summary-json "$case_dir/analysis/addr_summary.json" \
    --top-n 20

  python3 "$SCRIPT_DIR/deep_dive_uvm_faults.py" \
    --address-log "$case_dir/vllm_uvm_address_regions.log" \
    --fault-log "$case_dir/uvm_fault_addrs.log" \
    --allocator-log "$case_dir/vllm_uvm_allocator_trace.log" \
    --fault-start-line "$main_start_line" \
    --top-gaps 20 \
    --gap-csv "$case_dir/analysis/gaps.csv" \
    --summary-json "$case_dir/analysis/deep_summary.json"
}

echo "==========================================================="
echo "Stage M device_direct wide-phase fault/perf A/B"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS probe_prompts=$PROBE_PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "target_gap=$TARGET_GAP target_class=$TARGET_CLASS"
echo "target_phases=$TARGET_PHASES"
echo "device_direct_bytes=min:$DEVICE_DIRECT_MIN_BYTES max:$DEVICE_DIRECT_MAX_BYTES total:$DEVICE_DIRECT_MAX_TOTAL_BYTES backend:$DEVICE_DIRECT_BACKEND"
echo "scratch_pool_budget_bytes=$SCRATCH_POOL_BUDGET_BYTES mode=$SCRATCH_POOL_MODE"
echo "dynamic prefetch/offload/residency policies=disabled"
echo "==========================================================="

run_case baseline_trace device_direct_trace 0 0
run_case device_direct device_direct 1 1

SUMMARY_JSON="$RUN_ROOT/device_direct_wide_phase_fault_perf_comparison.json"
SUMMARY_CSV="$RUN_ROOT/device_direct_wide_phase_fault_perf_summary.csv"
SUMMARY_MD="$RUN_ROOT/device_direct_wide_phase_fault_perf_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_JSON" "$SUMMARY_CSV" "$SUMMARY_MD" "$TARGET_GAP" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_csv = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
target_gap = int(sys.argv[5])

def pct(numer, denom):
    if numer is None or denom in (None, 0):
        return None
    return 100.0 * float(numer) / float(denom)

def fmt_pct(value):
    return "n/a" if value is None else f"{value:.2f}%"

def parse_last(pattern, text, cast=float):
    matches = re.findall(pattern, text)
    return cast(matches[-1]) if matches else None

def parse_bench(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "successful_requests": parse_last(r"Successful requests:\s+(\d+)", text, int),
        "failed_requests": parse_last(r"Failed requests:\s+(\d+)", text, int),
        "output_tok_s": parse_last(r"Output token throughput \(tok/s\):\s+([\d.]+)", text),
        "mean_tpot_ms": parse_last(r"Mean TPOT \(ms\):\s+([\d.]+)", text),
        "mean_ttft_ms": parse_last(r"Mean TTFT \(ms\):\s+([\d.]+)", text),
    }

def parse_delta_stats(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    values = {}
    for key in (
        "delta_faults",
        "delta_duplicates",
        "delta_after_dedup",
        "delta_kv_faults",
        "delta_kv_duplicates",
        "delta_kv_after_dedup",
    ):
        matches = re.findall(rf"{key}=(\d+)", text)
        values[key] = int(matches[-1]) if matches else None
    return values

def load_case(label):
    case_dir = run_root / label
    deep = json.loads((case_dir / "analysis" / "deep_summary.json").read_text())
    metrics_path = case_dir / "gap_watch_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    probe = json.loads((case_dir / "auto_gap_watch_probe.json").read_text())
    post = json.loads((case_dir / "auto_gap_watch_post_main.json").read_text())

    fault_analysis = deep.get("fault_analysis", {})
    parsed = fault_analysis.get("parsed_fault_records")
    classes = fault_analysis.get("class_summaries", {})
    gaps = fault_analysis.get("unknown_gap_summaries", [])
    target = next((gap for gap in gaps if gap.get("gap_index") == target_gap), None)
    if target is None:
        target = {"gap_index": target_gap, "faults": 0, "fault_share_of_unknown": 0.0}

    class_faults = {
        name: int((classes.get(name) or {}).get("faults") or 0)
        for name in ("weight", "unknown", "kv_cache")
    }
    row = {
        "case": label,
        "parsed_faults_main": parsed,
        "weight_faults_main": class_faults["weight"],
        "unknown_faults_main": class_faults["unknown"],
        "kv_cache_faults_main": class_faults["kv_cache"],
        "gap2_faults_main": int(target.get("faults") or 0),
        "gap2_share_unknown_main_pct": pct(target.get("faults"), class_faults["unknown"]),
        "gap2_share_parsed_main_pct": pct(target.get("faults"), parsed),
        "weight_share_parsed_main_pct": pct(class_faults["weight"], parsed),
        "unknown_share_parsed_main_pct": pct(class_faults["unknown"], parsed),
        "kv_share_parsed_main_pct": pct(class_faults["kv_cache"], parsed),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "device_direct_eligible_records": metrics.get("device_direct_eligible_records"),
        "device_direct_budget_reject_records": metrics.get("device_direct_budget_reject_records"),
        "device_direct_peak_live_bytes": metrics.get("device_direct_peak_live_bytes_observed"),
        "scratch_pool_device_direct_records": metrics.get("scratch_pool_device_direct_records"),
        "scratch_pool_peak_live_bytes": metrics.get("scratch_pool_peak_live_bytes"),
        "scratch_pool_budget_reject_records": metrics.get("scratch_pool_budget_reject_records"),
        "gap_policy_success": metrics.get("gap_policy_success"),
        "gap_policy_fail": metrics.get("gap_policy_fail"),
        "probe_selected_gap": (probe.get("selected_gap") or {}).get("gap_index"),
        "post_selected_gap": (post.get("selected_gap") or {}).get("gap_index"),
        "post_fallback_used": post.get("fallback_used"),
    }
    stats = parse_delta_stats(case_dir / "uvm_fault_stats.log")
    if stats.get("delta_faults") is None:
        stats = parse_delta_stats(case_dir / "run.log")
    row.update(stats)
    row.update(parse_bench(case_dir / "vllm_bench.log"))
    return row

rows = [load_case("baseline_trace"), load_case("device_direct")]
base, direct = rows

def delta(key):
    b = base.get(key)
    d = direct.get(key)
    return None if b is None or d is None else d - b

def delta_pct(key):
    b = base.get(key)
    d = direct.get(key)
    if b in (None, 0) or d is None:
        return None
    return 100.0 * (d - b) / b

comparison = {
    "gap2_fault_delta": delta("gap2_faults_main"),
    "gap2_fault_delta_pct": delta_pct("gap2_faults_main"),
    "unknown_fault_delta": delta("unknown_faults_main"),
    "unknown_fault_delta_pct": delta_pct("unknown_faults_main"),
    "weight_fault_delta": delta("weight_faults_main"),
    "weight_fault_delta_pct": delta_pct("weight_faults_main"),
    "parsed_fault_delta": delta("parsed_faults_main"),
    "parsed_fault_delta_pct": delta_pct("parsed_faults_main"),
    "delta_replayable_faults_delta": delta("delta_faults"),
    "delta_replayable_faults_delta_pct": delta_pct("delta_faults"),
    "delta_after_dedup_delta": delta("delta_after_dedup"),
    "delta_after_dedup_delta_pct": delta_pct("delta_after_dedup"),
    "output_tok_s_delta_pct": delta_pct("output_tok_s"),
    "mean_tpot_ms_delta_pct": delta_pct("mean_tpot_ms"),
    "mean_ttft_ms_delta_pct": delta_pct("mean_ttft_ms"),
}

payload = {"run_root": str(run_root), "cases": rows, "comparison": comparison}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

fields = list(rows[0].keys())
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage M Device Direct Wide-Phase Fault/Perf A/B",
    "",
    f"- run_root: `{run_root}`",
    f"- summary_json: `{summary_json}`",
    f"- summary_csv: `{summary_csv}`",
    "- baseline: gap-watch trace only; device_direct disabled",
    f"- treatment: gap-watch device_direct; scratch_pool enforce; backend {direct.get('device_direct_backend_counts')}",
    "- dynamic prefetch/offload/residency policies: disabled",
    "",
    "| case | parsed main faults | weight | unknown | gap2 | gap2/unknown | raw replayable delta | dedup delta | tok/s | TPOT ms | TTFT ms | direct actual | direct peak MiB |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    direct_peak_mib = (
        float(row["device_direct_peak_live_bytes"]) / (1024 * 1024)
        if row.get("device_direct_peak_live_bytes") is not None
        else None
    )
    lines.append(
        "| {case} | {parsed} | {weight} | {unknown} | {gap2} | {gap2_share} | "
        "{delta_faults} | {delta_after_dedup} | {tok} | {tpot} | {ttft} | {actual} | {peak} |".format(
            case=row["case"],
            parsed=row["parsed_faults_main"],
            weight=row["weight_faults_main"],
            unknown=row["unknown_faults_main"],
            gap2=row["gap2_faults_main"],
            gap2_share=fmt_pct(row["gap2_share_unknown_main_pct"]),
            delta_faults=row.get("delta_faults"),
            delta_after_dedup=row.get("delta_after_dedup"),
            tok=row.get("output_tok_s"),
            tpot=row.get("mean_tpot_ms"),
            ttft=row.get("mean_ttft_ms"),
            actual=row.get("device_direct_actual_records"),
            peak="" if direct_peak_mib is None else f"{direct_peak_mib:.2f}",
        )
    )

lines.extend([
    "",
    "## Delta: device_direct vs baseline_trace",
    f"- parsed_main_fault_delta: {comparison['parsed_fault_delta']} ({fmt_pct(comparison['parsed_fault_delta_pct'])})",
    f"- gap2_fault_delta: {comparison['gap2_fault_delta']} ({fmt_pct(comparison['gap2_fault_delta_pct'])})",
    f"- unknown_fault_delta: {comparison['unknown_fault_delta']} ({fmt_pct(comparison['unknown_fault_delta_pct'])})",
    f"- weight_fault_delta: {comparison['weight_fault_delta']} ({fmt_pct(comparison['weight_fault_delta_pct'])})",
    f"- replayable_delta_faults_delta: {comparison['delta_replayable_faults_delta']} ({fmt_pct(comparison['delta_replayable_faults_delta_pct'])})",
    f"- replayable_delta_after_dedup_delta: {comparison['delta_after_dedup_delta']} ({fmt_pct(comparison['delta_after_dedup_delta_pct'])})",
    f"- output_tok_s_delta_pct: {fmt_pct(comparison['output_tok_s_delta_pct'])}",
    f"- mean_tpot_ms_delta_pct: {fmt_pct(comparison['mean_tpot_ms_delta_pct'])}",
    f"- mean_ttft_ms_delta_pct: {fmt_pct(comparison['mean_ttft_ms_delta_pct'])}",
])

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "Stage M device_direct wide-phase fault/perf A/B finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
