#!/usr/bin/env bash
# Pure UVM fault-profile run.
# This intentionally disables device_direct, prefetch, offload, residency,
# gap-watch, allocator trace, and policy actions. It records only UVM fault
# counters/address logs plus vLLM address regions for offline classification.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_pure_uvm_fault_profile_p10_run}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-10}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

mkdir -p "$RUN_ROOT/analysis"

echo "==========================================================="
echo "Pure UVM fault profile"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL"
echo "prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "server_max_num_seqs=$SERVER_MAX_NUM_SEQS"
echo "server_gpu_memory_utilization=$SERVER_GPU_MEMORY_UTILIZATION"
echo "max_model_len=$MAX_MODEL_LEN"
echo "==========================================================="

pkill -f "vllm bench serve" || true
pkill -f "vllm serve $MODEL" || true
python3 /home/ubuntu/nvidia-uvm-gpu/workloads/cleanup_gpu.py || true

./run_kv_fault_ratio.sh \
  --model "$MODEL" \
  --mode trace \
  --with-address-log \
  --trace-log "$RUN_ROOT/uvm_fault_stats.log" \
  --address-trace-log "$RUN_ROOT/uvm_fault_addrs.log" \
  --address-log "$RUN_ROOT/vllm_uvm_address_regions.log" \
  --server-log "$RUN_ROOT/vllm_server.log" \
  --bench-log "$RUN_ROOT/vllm_bench.log" \
  --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
  --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --uvm-policy-enable 0 \
  --uvm-policy-mode trace_only \
  --uvm-policy-warmup-advise-gpu 0 \
  --uvm-unknown-detail-enable 0 \
  --uvm-gap-watch-enable 0 \
  --uvm-device-direct-enable 0 \
  --uvm-kv-runtime-enable 0 \
  --uvm-kv-runtime-prefix-evict-enable 0 \
  --uvm-weight-budget-cpu-offload-enable 0 \
  --uvm-weight-map-enable 0 \
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
  --uvm-pool-coordinator-enable 0 \
  --uvm-pool-coordinator-trace-enable 0 \
  --uvm-residency-enable 0 \
  --uvm-residency-trace-enable 0 \
  --uvm-residency-kv-sizing-enable 0 \
  --uvm-residency-kv-metadata-enable 0 \
  --uvm-residency-kv-data-offload-enable 0 \
  --uvm-residency-kv-prefetch-back-enable 0 \
  --uvm-residency-weight-enforce-hot-set 0 \
  --uvm-residency-weight-active-evict-enable 0 \
  --uvm-pool-registry-enable 0 \
  --uvm-scratch-pool-enable 0 \
  --prompts "$PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --output-len "$OUTPUT_LEN"

python3 "$SCRIPT_DIR/analyze_uvm_fault_addresses.py" \
  --address-log "$RUN_ROOT/vllm_uvm_address_regions.log" \
  --fault-log "$RUN_ROOT/uvm_fault_addrs.log" \
  --regions-csv "$RUN_ROOT/analysis/regions.csv" \
  --faults-csv "$RUN_ROOT/analysis/faults.csv" \
  --summary-json "$RUN_ROOT/analysis/addr_summary.json" \
  --top-n 20

python3 "$SCRIPT_DIR/deep_dive_uvm_faults.py" \
  --address-log "$RUN_ROOT/vllm_uvm_address_regions.log" \
  --fault-log "$RUN_ROOT/uvm_fault_addrs.log" \
  --top-gaps 20 \
  --gap-csv "$RUN_ROOT/analysis/gaps.csv" \
  --summary-json "$RUN_ROOT/analysis/deep_summary.json"

SUMMARY_JSON="$RUN_ROOT/analysis/fault_profile_summary.json"
SUMMARY_MD="$RUN_ROOT/analysis/fault_profile_summary.md"

python3 - "$RUN_ROOT" "$SUMMARY_JSON" "$SUMMARY_MD" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])

addr = json.loads((run_root / "analysis" / "addr_summary.json").read_text())
deep = json.loads((run_root / "analysis" / "deep_summary.json").read_text())

def pct(numer, denom):
    if denom in (None, 0) or numer is None:
        return None
    return 100.0 * float(numer) / float(denom)

def pct_s(value):
    return "n/a" if value is None else f"{value:.2f}%"

fault_analysis = deep.get("fault_analysis", {})
class_summaries = fault_analysis.get("class_summaries", {})
parsed_faults = fault_analysis.get("parsed_fault_records")

classes = {}
for name in ("weight", "unknown", "kv_cache"):
    faults = (class_summaries.get(name) or {}).get("faults", 0)
    classes[name] = {
        "faults": faults,
        "share_of_parsed": pct(faults, parsed_faults),
    }

top_gaps = fault_analysis.get("unknown_gap_summaries", [])
for gap in top_gaps:
    gap["share_of_parsed"] = pct(gap.get("faults"), parsed_faults)

stats_text = (run_root / "uvm_fault_stats.log").read_text(
    encoding="utf-8", errors="replace"
)
delta = {}
for key in (
    "delta_faults",
    "delta_duplicates",
    "delta_after_dedup",
    "delta_kv_faults",
    "delta_kv_duplicates",
    "delta_kv_after_dedup",
):
    matches = re.findall(rf"{key}=(\d+)", stats_text)
    delta[key] = int(matches[-1]) if matches else None

if delta["delta_faults"] is None:
    pattern = re.compile(
        r"batch_faults=(\d+) batch_duplicates=(\d+) "
        r"batch_after_dedup=(\d+) batch_kv_faults=(\d+) "
        r"batch_kv_duplicates=(\d+) batch_kv_after_dedup=(\d+).*?"
        r"total_faults=(\d+) total_duplicates=(\d+) "
        r"total_after_dedup=(\d+) total_kv_faults=(\d+) "
        r"total_kv_duplicates=(\d+) total_kv_after_dedup=(\d+)"
    )
    rows = [list(map(int, m.groups())) for m in pattern.finditer(stats_text)]
    if rows:
        first, last = rows[0], rows[-1]
        mapping = [
            ("delta_faults", 0, 6),
            ("delta_duplicates", 1, 7),
            ("delta_after_dedup", 2, 8),
            ("delta_kv_faults", 3, 9),
            ("delta_kv_duplicates", 4, 10),
            ("delta_kv_after_dedup", 5, 11),
        ]
        for key, batch_i, total_i in mapping:
            delta[key] = last[total_i] - (first[total_i] - first[batch_i])

bench = {}
bench_text = (run_root / "vllm_bench.log").read_text(
    encoding="utf-8", errors="replace"
)
patterns = {
    "successful_requests": (r"Successful requests:\s+(\d+)", int),
    "failed_requests": (r"Failed requests:\s+(\d+)", int),
    "output_tok_s": (r"Output token throughput \(tok/s\):\s+([\d.]+)", float),
    "mean_tpot_ms": (r"Mean TPOT \(ms\):\s+([\d.]+)", float),
    "mean_ttft_ms": (r"Mean TTFT \(ms\):\s+([\d.]+)", float),
}
for key, (pattern, caster) in patterns.items():
    matches = re.findall(pattern, bench_text)
    bench[key] = caster(matches[-1]) if matches else None

summary = {
    "run_root": str(run_root),
    "parsed_fault_records": parsed_faults,
    "classes": classes,
    "top_unknown_gaps": top_gaps[:20],
    "delta_counters": delta,
    "bench": bench,
}

summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

lines = [
    "# Pure UVM Fault Profile",
    "",
    f"- run_root: `{run_root}`",
    f"- parsed_address_faults: {parsed_faults}",
    f"- delta_faults: {delta.get('delta_faults')}",
    f"- delta_after_dedup: {delta.get('delta_after_dedup')}",
    "",
    "## Fault Classes",
    "",
    "| class | faults | share of parsed |",
    "|---|---:|---:|",
]
for name in ("weight", "unknown", "kv_cache"):
    row = classes[name]
    lines.append(
        f"| {name} | {row['faults']} | {pct_s(row['share_of_parsed'])} |"
    )

lines.extend([
    "",
    "## Top Unknown Gaps",
    "",
    "| gap | faults | share of unknown | share of parsed | size MiB | left | right |",
    "|---:|---:|---:|---:|---:|---|---|",
])
for gap in top_gaps[:20]:
    left = gap.get("left_region") or {}
    right = gap.get("right_region") or {}
    if isinstance(left, dict):
        left_name = left.get("name")
    else:
        left_name = None
    if isinstance(right, dict):
        right_name = right.get("name")
    else:
        right_name = None
    lines.append(
        "| {gap_index} | {faults} | {share_unknown} | {share_parsed} | "
        "{size_mib} | {left} | {right} |".format(
            gap_index=gap.get("gap_index"),
            faults=gap.get("faults"),
            share_unknown=pct_s(
                100.0 * gap.get("fault_share_of_unknown", 0.0)
                if gap.get("fault_share_of_unknown") is not None
                else None
            ),
            share_parsed=pct_s(gap.get("share_of_parsed")),
            size_mib=gap.get("size_mib"),
            left=left_name,
            right=right_name,
        )
    )

lines.extend([
    "",
    "## Bench",
    "",
    f"- successful_requests: {bench.get('successful_requests')}",
    f"- failed_requests: {bench.get('failed_requests')}",
    f"- output_tok_s: {bench.get('output_tok_s')}",
    f"- mean_tpot_ms: {bench.get('mean_tpot_ms')}",
    f"- mean_ttft_ms: {bench.get('mean_ttft_ms')}",
])

summary_md.write_text("\n".join(lines) + "\n")
print(summary_md.read_text())
PY

echo "Pure UVM fault profile finished."
echo "summary_json: $SUMMARY_JSON"
echo "summary_md: $SUMMARY_MD"
echo "gaps_csv: $RUN_ROOT/analysis/gaps.csv"
