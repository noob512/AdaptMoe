#!/usr/bin/env bash
# Stage M Phase 5/6: KV active protection experiments.
#
# Phase 5 validates trace-only active KV identification and telemetry.
# Phase 6 enables active KV advise+prefetch using the Stage L6 KV pointer map.
# Both run on top of the current best wide-phase cuda_malloc_async
# device_direct scratch configuration.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_kv_active_protection_p10_run}"
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
DEVICE_DIRECT_BACKEND="${DEVICE_DIRECT_BACKEND:-cuda_malloc_async}"
DEVICE_DIRECT_POOL_RELEASE_THRESHOLD="${DEVICE_DIRECT_POOL_RELEASE_THRESHOLD:-1073741824}"
SCRATCH_POOL_BUDGET_BYTES="${SCRATCH_POOL_BUDGET_BYTES:-1073741824}"
SCRATCH_POOL_MODE="${SCRATCH_POOL_MODE:-enforce}"

mkdir -p "$RUN_ROOT"

run_case() {
  local label="$1"
  local active_enable="$2"
  local active_mode="$3"
  local case_dir="$RUN_ROOT/$label"
  local kv_runtime_trace="$case_dir/vllm_uvm_kv_runtime_stage_m.jsonl"
  local residency_trace="$case_dir/vllm_uvm_pool_residency_stage_m.jsonl"
  local coordinator_trace="$case_dir/vllm_uvm_pool_coordinator_stage_m.jsonl"
  local kv_pointer_map="$case_dir/vllm_uvm_kv_pointer_map_stage_m.jsonl"

  mkdir -p "$case_dir/analysis"

  echo "==========================================================="
  echo "Run $label"
  echo "  active_enable=$active_enable"
  echo "  active_mode=$active_mode"
  echo "  run_root=$case_dir"
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
    --uvm-policy-enable 1 \
    --uvm-policy-mode trace_only \
    --uvm-policy-runtime-enable "$active_enable" \
    --uvm-gap-watch-name "gap${TARGET_GAP}_${label}_kv_active" \
    --uvm-gap-watch-all-classes 1 \
    --uvm-gap-watch-min-bytes 4096 \
    --uvm-device-direct-enable 1 \
    --uvm-device-direct-min-bytes "$DEVICE_DIRECT_MIN_BYTES" \
    --uvm-device-direct-max-bytes "$DEVICE_DIRECT_MAX_BYTES" \
    --uvm-device-direct-max-total-bytes "$DEVICE_DIRECT_MAX_TOTAL_BYTES" \
    --uvm-device-direct-backend "$DEVICE_DIRECT_BACKEND" \
    --uvm-device-direct-pool-release-threshold "$DEVICE_DIRECT_POOL_RELEASE_THRESHOLD" \
    --uvm-device-direct-target-phases "$TARGET_PHASES" \
    --uvm-scratch-pool-enable 1 \
    --uvm-scratch-pool-budget-bytes "$SCRATCH_POOL_BUDGET_BYTES" \
    --uvm-scratch-pool-mode "$SCRATCH_POOL_MODE" \
    --uvm-scratch-pool-target-phases "$TARGET_PHASES" \
    --uvm-kv-runtime-enable "$active_enable" \
    --uvm-kv-runtime-mode trace_only \
    --uvm-kv-runtime-trace-file "$kv_runtime_trace" \
    --uvm-kv-runtime-prefix-evict-enable 0 \
    --uvm-kv-active-protect-enable "$active_enable" \
    --uvm-kv-active-protect-mode "$active_mode" \
    --uvm-kv-active-protect-device 0 \
    --uvm-kv-active-protect-max-blocks 0 \
    --uvm-residency-enable "$active_enable" \
    --uvm-residency-stage stage_l6 \
    --uvm-residency-mode trace_only \
    --uvm-residency-trace-enable "$active_enable" \
    --uvm-residency-trace-file "$residency_trace" \
    --uvm-residency-kv-metadata-enable "$active_enable" \
    --uvm-residency-kv-data-offload-enable 0 \
    --uvm-residency-kv-prefetch-back-enable 0 \
    --uvm-residency-kv-pointer-map-file "$kv_pointer_map" \
    --uvm-pool-coordinator-enable "$active_enable" \
    --uvm-pool-coordinator-mode trace_only \
    --uvm-pool-coordinator-trace-enable "$active_enable" \
    --uvm-pool-coordinator-trace-file "$coordinator_trace" \
    --uvm-weight-budget-cpu-offload-enable 0 \
    --uvm-weight-map-enable 1 \
    --uvm-moe-routing-trace-enable 0 \
    --uvm-weight-prefetch-enable 0 \
    --uvm-weight-prefetch-mode trace_only \
    --uvm-weight-prefetch-trace-enable 0 \
    --uvm-weight-prefetch-hit-trace-enable 0 \
    --uvm-weight-prefetch-hit-driver-query-enable 0 \
    --uvm-weight-batch-executor-enable 0 \
    --uvm-weight-offload-enable 0 \
    --uvm-weight-offload-mode trace_only \
    --uvm-init-kv-sizing-enable 0 \
    --uvm-init-weight-offload-enable 0 \
    --uvm-init-weight-offload-mode trace_only \
    --uvm-init-weight-hot-prefetch-enable 0 \
    --uvm-init-weight-hot-prefetch-real-query-enable 0 \
    --uvm-hot-residency-stage-trace-enable 0 \
    --uvm-pool-registry-enable 0 \
    --auto-gap-watch-enable 1 \
    --auto-gap-watch-probe-prompts "$PROBE_PROMPTS" \
    --auto-gap-watch-target-gap "$TARGET_GAP" \
    --auto-gap-watch-fallback-to-hottest 1 \
    --auto-gap-watch-policy-action-override device_direct \
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
echo "Stage M KV active protection Phase 5/6"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS probe_prompts=$PROBE_PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "scratch/device_direct backend=$DEVICE_DIRECT_BACKEND phases=$TARGET_PHASES"
echo "==========================================================="

run_case device_direct_baseline 0 trace_only
run_case kv_active_trace 1 trace_only
run_case kv_active_advise_prefetch 1 advise_prefetch

SUMMARY_JSON="$RUN_ROOT/kv_active_protection_summary.json"
SUMMARY_CSV="$RUN_ROOT/kv_active_protection_summary.csv"
SUMMARY_MD="$RUN_ROOT/kv_active_protection_summary.md"

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

case_names = [
    "device_direct_baseline",
    "kv_active_trace",
    "kv_active_advise_prefetch",
]

def pct_change(base, value):
    if base in (None, 0) or value is None:
        return None
    return 100.0 * (value - base) / base

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
    for key in ("delta_faults", "delta_after_dedup", "delta_kv_faults", "delta_kv_after_dedup"):
        matches = re.findall(rf"{key}=(\d+)", text)
        values[key] = int(matches[-1]) if matches else None
    return values

def load_last_jsonl_action(path, action):
    if not path.exists():
        return {}
    last = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("action") == action:
                last = record
    return last

def count_jsonl_action(path, action):
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("action") == action:
                count += 1
    return count

def load_case(name):
    case_dir = run_root / name
    deep = json.loads((case_dir / "analysis" / "deep_summary.json").read_text())
    metrics = json.loads((case_dir / "gap_watch_metrics.json").read_text())
    fault_analysis = deep.get("fault_analysis", {})
    classes = fault_analysis.get("class_summaries", {})
    gaps = fault_analysis.get("unknown_gap_summaries", [])
    target = next((gap for gap in gaps if gap.get("gap_index") == target_gap), None) or {}
    runtime_path = case_dir / "vllm_uvm_kv_runtime_stage_m.jsonl"
    runtime_summary = load_last_jsonl_action(runtime_path, "runtime_summary")
    pointer_map = load_last_jsonl_action(runtime_path, "stage_l6_kv_pointer_map_loaded")
    row = {
        "case": name,
        "parsed_faults_main": fault_analysis.get("parsed_fault_records"),
        "weight_faults_main": int((classes.get("weight") or {}).get("faults") or 0),
        "unknown_faults_main": int((classes.get("unknown") or {}).get("faults") or 0),
        "kv_cache_faults_main": int((classes.get("kv_cache") or {}).get("faults") or 0),
        "gap2_faults_main": int(target.get("faults") or 0),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "scratch_pool_device_direct_records": metrics.get("scratch_pool_device_direct_records"),
        "active_protect_attempt_records": runtime_summary.get("active_protect_attempt_records", 0),
        "active_protect_issued_records": runtime_summary.get("active_protect_issued_records", 0),
        "active_protect_fallback_records": runtime_summary.get("active_protect_fallback_records", 0),
        "active_protect_blocks": runtime_summary.get("active_protect_blocks", 0),
        "active_protect_bytes": runtime_summary.get("active_protect_bytes", 0),
        "active_kv_protect_issued_lines": count_jsonl_action(runtime_path, "active_kv_protect_issued"),
        "active_kv_protect_fallback_lines": count_jsonl_action(runtime_path, "active_kv_protect_fallback"),
        "kv_pointer_map_entries": pointer_map.get("entries", 0),
    }
    row.update(parse_delta_stats(case_dir / "uvm_fault_stats.log"))
    row.update(parse_bench(case_dir / "vllm_bench.log"))
    return row

rows = [load_case(name) for name in case_names]
baseline = rows[0]
for row in rows:
    row["kv_fault_delta_vs_baseline_pct"] = pct_change(
        baseline.get("kv_cache_faults_main"), row.get("kv_cache_faults_main")
    )
    row["tpot_delta_vs_baseline_pct"] = pct_change(
        baseline.get("mean_tpot_ms"), row.get("mean_tpot_ms")
    )
    row["tok_s_delta_vs_baseline_pct"] = pct_change(
        baseline.get("output_tok_s"), row.get("output_tok_s")
    )

payload = {"run_root": str(run_root), "cases": rows}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

fields = list(rows[0].keys())
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Stage M KV Active Protection Phase 5/6",
    "",
    f"- run_root: `{run_root}`",
    f"- summary_json: `{summary_json}`",
    f"- summary_csv: `{summary_csv}`",
    "- baseline: wide-phase cuda_malloc_async device_direct scratch, KV active protection disabled",
    "- Phase 5: `kv_active_trace`, active KV trace-only protection",
    "- Phase 6: `kv_active_advise_prefetch`, active KV advise+prefetch",
    "",
    "| case | KV faults | KV delta | TPOT ms | TPOT delta | tok/s | tok/s delta | active attempts | active issued | active fallback | pointer map entries |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {kv} | {kv_delta} | {tpot} | {tpot_delta} | {tok} | {tok_delta} | {attempts} | {issued} | {fallback} | {entries} |".format(
            case=row["case"],
            kv=row["kv_cache_faults_main"],
            kv_delta=fmt_pct(row["kv_fault_delta_vs_baseline_pct"]),
            tpot=row.get("mean_tpot_ms"),
            tpot_delta=fmt_pct(row["tpot_delta_vs_baseline_pct"]),
            tok=row.get("output_tok_s"),
            tok_delta=fmt_pct(row["tok_s_delta_vs_baseline_pct"]),
            attempts=row["active_protect_attempt_records"],
            issued=row["active_protect_issued_records"],
            fallback=row["active_protect_fallback_records"],
            entries=row["kv_pointer_map_entries"],
        )
    )
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "Stage M KV active protection Phase 5/6 finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
