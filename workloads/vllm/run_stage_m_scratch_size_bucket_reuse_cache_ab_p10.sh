#!/usr/bin/env bash
# Phase 7 A/B for scratch cuda_malloc_async device-direct size-bucket reuse.
# The treatment keeps freed scratch direct blocks in an exact-size reuse cache
# within each size bucket, reducing short-lived cudaMallocAsync churn.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_scratch_size_bucket_reuse_cache_ab_p10_run}"
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
REUSE_CACHE_MAX_BYTES="${REUSE_CACHE_MAX_BYTES:-134217728}"
REUSE_CACHE_MAX_ENTRIES_PER_BUCKET="${REUSE_CACHE_MAX_ENTRIES_PER_BUCKET:-8}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"

mkdir -p "$RUN_ROOT"

COMMON_DISABLE_ARGS=(
  --uvm-policy-enable 1
  --uvm-policy-mode trace_only
  --uvm-policy-runtime-enable 0
  --uvm-policy-warmup-advise-gpu 0
  --uvm-kv-runtime-enable 0
  --uvm-kv-runtime-prefix-evict-enable 0
  --uvm-kv-active-protect-enable 0
  --uvm-kv-active-protect-mode trace_only
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
  local reuse_cache_enable="$2"
  local case_dir="$RUN_ROOT/$label"

  mkdir -p "$case_dir/analysis"

  echo "==========================================================="
  echo "Run $label"
  echo "  scratch_pool_reuse_cache_enable=$reuse_cache_enable"
  echo "  backend=$DEVICE_DIRECT_BACKEND"
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
    --uvm-gap-watch-name "gap${TARGET_GAP}_${label}_scratch_reuse_cache_ab" \
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
    --uvm-scratch-pool-reuse-cache-enable "$reuse_cache_enable" \
    --uvm-scratch-pool-reuse-cache-max-bytes "$REUSE_CACHE_MAX_BYTES" \
    --uvm-scratch-pool-reuse-cache-max-entries-per-bucket "$REUSE_CACHE_MAX_ENTRIES_PER_BUCKET" \
    "${COMMON_DISABLE_ARGS[@]}" \
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

  python3 "$SCRIPT_DIR/summarize_gap_watch_metrics.py" \
    --allocator-log "$case_dir/vllm_uvm_allocator_trace.log" \
    --summary-json "$case_dir/gap_watch_metrics.json"
}

echo "==========================================================="
echo "Stage M scratch size-bucket reuse cache A/B"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS probe_prompts=$PROBE_PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS max_model_len=$MAX_MODEL_LEN gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "target_gap=$TARGET_GAP target_class=$TARGET_CLASS"
echo "target_phases=$TARGET_PHASES"
echo "device_direct_bytes=min:$DEVICE_DIRECT_MIN_BYTES max:$DEVICE_DIRECT_MAX_BYTES total:$DEVICE_DIRECT_MAX_TOTAL_BYTES backend:$DEVICE_DIRECT_BACKEND"
echo "scratch_pool_budget_bytes=$SCRATCH_POOL_BUDGET_BYTES mode=$SCRATCH_POOL_MODE"
echo "reuse_cache_max_bytes=$REUSE_CACHE_MAX_BYTES max_entries_per_bucket=$REUSE_CACHE_MAX_ENTRIES_PER_BUCKET"
echo "dynamic KV/weight/offload/residency policies=disabled"
echo "==========================================================="

run_case scratch_direct_no_cache 0
run_case scratch_direct_bucket_cache 1

SUMMARY_JSON="$RUN_ROOT/scratch_size_bucket_reuse_cache_comparison.json"
SUMMARY_CSV="$RUN_ROOT/scratch_size_bucket_reuse_cache_summary.csv"
SUMMARY_MD="$RUN_ROOT/scratch_size_bucket_reuse_cache_summary.md"

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

def delta_pct(new_value, old_value):
    if old_value in (None, 0) or new_value is None:
        return None
    return 100.0 * (float(new_value) - float(old_value)) / float(old_value)

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

def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def load_case(label):
    case_dir = run_root / label
    deep = load_json(case_dir / "analysis" / "deep_summary.json", {})
    metrics = load_json(case_dir / "gap_watch_metrics.json", {})
    probe = load_json(case_dir / "auto_gap_watch_probe.json", {})
    post = load_json(case_dir / "auto_gap_watch_post_main.json", {})

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
    hits = metrics.get("scratch_pool_reuse_cache_hits")
    misses = metrics.get("scratch_pool_reuse_cache_misses")
    hit_rate = pct(hits, (hits or 0) + (misses or 0))

    row = {
        "case": label,
        "parsed_faults_main": parsed,
        "weight_faults_main": class_faults["weight"],
        "unknown_faults_main": class_faults["unknown"],
        "kv_cache_faults_main": class_faults["kv_cache"],
        "gap2_faults_main": int(target.get("faults") or 0),
        "gap2_share_unknown_main_pct": pct(target.get("faults"), class_faults["unknown"]),
        "device_direct_actual_records": metrics.get("device_direct_actual_records"),
        "device_direct_eligible_records": metrics.get("device_direct_eligible_records"),
        "device_direct_budget_reject_records": metrics.get("device_direct_budget_reject_records"),
        "device_direct_peak_live_bytes": metrics.get("device_direct_peak_live_bytes_observed"),
        "scratch_pool_device_direct_records": metrics.get("scratch_pool_device_direct_records"),
        "scratch_pool_peak_live_bytes": metrics.get("scratch_pool_peak_live_bytes"),
        "scratch_pool_budget_reject_records": metrics.get("scratch_pool_budget_reject_records"),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_rate_pct": hit_rate,
        "cache_fresh_backend_allocations": metrics.get("scratch_pool_reuse_cache_fresh_backend_allocations"),
        "cache_inserts": metrics.get("scratch_pool_reuse_cache_inserts"),
        "cache_insert_rejects": metrics.get("scratch_pool_reuse_cache_insert_rejects"),
        "cache_hit_bytes": metrics.get("scratch_pool_reuse_cache_hit_bytes"),
        "cache_insert_bytes": metrics.get("scratch_pool_reuse_cache_insert_bytes"),
        "cache_peak_cached_bytes": metrics.get("scratch_pool_reuse_cache_peak_cached_bytes"),
        "cache_cached_entries": metrics.get("scratch_pool_reuse_cache_cached_entries"),
        "cache_event_not_ready": metrics.get("scratch_pool_reuse_cache_event_not_ready"),
        "cache_event_errors": metrics.get("scratch_pool_reuse_cache_event_errors"),
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

rows = [load_case("scratch_direct_no_cache"), load_case("scratch_direct_bucket_cache")]
base, cached = rows

def delta(key):
    b = base.get(key)
    c = cached.get(key)
    return None if b is None or c is None else c - b

comparison = {
    "output_tok_s_delta_pct": delta_pct(cached.get("output_tok_s"), base.get("output_tok_s")),
    "mean_tpot_ms_delta_pct": delta_pct(cached.get("mean_tpot_ms"), base.get("mean_tpot_ms")),
    "mean_ttft_ms_delta_pct": delta_pct(cached.get("mean_ttft_ms"), base.get("mean_ttft_ms")),
    "parsed_fault_delta": delta("parsed_faults_main"),
    "unknown_fault_delta": delta("unknown_faults_main"),
    "gap2_fault_delta": delta("gap2_faults_main"),
    "delta_replayable_faults_delta": delta("delta_faults"),
    "delta_after_dedup_delta": delta("delta_after_dedup"),
    "cache_hits": cached.get("cache_hits"),
    "cache_misses": cached.get("cache_misses"),
    "cache_hit_rate_pct": cached.get("cache_hit_rate_pct"),
    "cache_fresh_backend_allocations": cached.get("cache_fresh_backend_allocations"),
}

payload = {"run_root": str(run_root), "cases": rows, "comparison": comparison}
summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

fields = list(rows[0].keys())
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def mib(value):
    if value is None:
        return "n/a"
    return f"{float(value) / (1024 * 1024):.2f}"

lines = [
    "# Stage M Scratch Size-Bucket Reuse Cache A/B",
    "",
    f"- run_root: `{run_root}`",
    f"- summary_json: `{summary_json}`",
    f"- summary_csv: `{summary_csv}`",
    "- baseline: scratch cuda_malloc_async device_direct, reuse cache disabled",
    "- treatment: same scratch device_direct path, exact-size reuse cache enabled",
    "- dynamic KV/weight/offload/residency policies: disabled",
    "",
    "| case | tok/s | TPOT ms | TTFT ms | parsed faults | unknown | gap2 | raw replayable delta | dedup delta | direct actual | cache hits | cache misses | hit rate | fresh backend | cache peak MiB | rejects |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {case} | {tok} | {tpot} | {ttft} | {parsed} | {unknown} | {gap2} | "
        "{delta_faults} | {delta_after_dedup} | {actual} | {hits} | {misses} | "
        "{hit_rate} | {fresh} | {peak} | {rejects} |".format(
            case=row["case"],
            tok=row.get("output_tok_s"),
            tpot=row.get("mean_tpot_ms"),
            ttft=row.get("mean_ttft_ms"),
            parsed=row["parsed_faults_main"],
            unknown=row["unknown_faults_main"],
            gap2=row["gap2_faults_main"],
            delta_faults=row.get("delta_faults"),
            delta_after_dedup=row.get("delta_after_dedup"),
            actual=row.get("device_direct_actual_records"),
            hits=row.get("cache_hits"),
            misses=row.get("cache_misses"),
            hit_rate=fmt_pct(row.get("cache_hit_rate_pct")),
            fresh=row.get("cache_fresh_backend_allocations"),
            peak=mib(row.get("cache_peak_cached_bytes")),
            rejects=row.get("cache_insert_rejects"),
        )
    )

lines.extend([
    "",
    "## Delta: bucket_cache vs no_cache",
    f"- output_tok_s_delta_pct: {fmt_pct(comparison['output_tok_s_delta_pct'])}",
    f"- mean_tpot_ms_delta_pct: {fmt_pct(comparison['mean_tpot_ms_delta_pct'])}",
    f"- mean_ttft_ms_delta_pct: {fmt_pct(comparison['mean_ttft_ms_delta_pct'])}",
    f"- parsed_fault_delta: {comparison['parsed_fault_delta']}",
    f"- unknown_fault_delta: {comparison['unknown_fault_delta']}",
    f"- gap2_fault_delta: {comparison['gap2_fault_delta']}",
    f"- replayable_delta_faults_delta: {comparison['delta_replayable_faults_delta']}",
    f"- replayable_delta_after_dedup_delta: {comparison['delta_after_dedup_delta']}",
    f"- cache_hits: {comparison['cache_hits']}",
    f"- cache_misses: {comparison['cache_misses']}",
    f"- cache_hit_rate: {fmt_pct(comparison['cache_hit_rate_pct'])}",
    f"- cache_fresh_backend_allocations: {comparison['cache_fresh_backend_allocations']}",
])

summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY

echo "Stage M scratch size-bucket reuse cache A/B finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $SUMMARY_MD"
