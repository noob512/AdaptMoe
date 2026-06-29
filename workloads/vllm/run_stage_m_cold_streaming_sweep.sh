#!/usr/bin/env bash
set -euo pipefail

# Stage M long-term sweep:
#   KV native + fixed non-expert native + expert UVM
#   hotK pre-resident + init-time non-hot cold CPU offload
#   cold streaming buffer sweep with driver-visible hot/cold/streaming tags.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_cold_streaming_sweep}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
TRACE_PATH="${TRACE_PATH:-log/moe_top64_numtoken32_trace/s32/vllm_uvm_moe_routing_trace.jsonl}"
PLAN_PATH="${PLAN_PATH:-}"
HOT_K="${HOT_K:-64}"
KV_BYTES="${KV_BYTES:-7516192768}"
STREAMING_BYTES_LIST="${STREAMING_BYTES_LIST:-1073741824 2147483648 3221225472}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-16 32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-4}"
REQUEST_RATE="${REQUEST_RATE:-8}"
BASE_PORT="${BASE_PORT:-8030}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
STREAMING_TTL_STEPS="${STREAMING_TTL_STEPS:-32}"
STREAMING_MAX_EXPERTS_PER_LAYER="${STREAMING_MAX_EXPERTS_PER_LAYER:-0}"
STREAMING_SYNC_PREFETCH="${STREAMING_SYNC_PREFETCH:-0}"
STREAMING_SYNC_EVICT="${STREAMING_SYNC_EVICT:-1}"
STREAMING_REAL_QUERY_ENABLE="${STREAMING_REAL_QUERY_ENABLE:-0}"
INIT_COLD_OFFLOAD_MODE="${INIT_COLD_OFFLOAD_MODE:-offload_cpu}"
INIT_COLD_OFFLOAD_SYNC="${INIT_COLD_OFFLOAD_SYNC:-1}"
DRY_RUN="${DRY_RUN:-0}"

cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"
COMMANDS_LOG="$RUN_ROOT/commands.log"
touch "$COMMANDS_LOG"

abs_path() {
  python3 - "$1" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
}

append_command() {
  printf 'command:' >> "$COMMANDS_LOG"
  printf ' %q' "$@" >> "$COMMANDS_LOG"
  printf '\n' >> "$COMMANDS_LOG"
}

run_cmd() {
  append_command "$@"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

echo "==========================================================="
echo "Stage M Cold Streaming Long-Term Sweep"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL hotK=$HOT_K kv_bytes=$KV_BYTES"
echo "streaming_bytes=$STREAMING_BYTES_LIST"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "==========================================================="

if [ -z "$PLAN_PATH" ]; then
  if [ ! -s "$TRACE_PATH" ]; then
    echo "ERROR: TRACE_PATH not found or empty: $TRACE_PATH" >&2
    exit 2
  fi
  plan_dir="$RUN_ROOT/plan"
  run_cmd python3 "$SCRIPT_DIR/analyze_moe_hotset_train_test.py" \
    --trace "$TRACE_PATH" \
    --out-dir "$plan_dir" \
    --k-values "$HOT_K" \
    --split-fraction 0.5 \
    --score-mode weighted \
    --expert-bytes 4718592 \
    --write-plans
  PLAN_PATH="$(abs_path "$plan_dir/plans/train_static_hotset_k${HOT_K}.json")"
else
  PLAN_PATH="$(abs_path "$PLAN_PATH")"
fi

if [ "$DRY_RUN" != "1" ] && [ ! -s "$PLAN_PATH" ]; then
  echo "ERROR: hot plan not found: $PLAN_PATH" >&2
  exit 2
fi

case_index=0
for streaming_bytes in $STREAMING_BYTES_LIST; do
  for max_num_seqs in $MAX_NUM_SEQS_LIST; do
    port=$((BASE_PORT + case_index))
    case_index=$((case_index + 1))
    echo "---- case streaming=$streaming_bytes max_num_seqs=$max_num_seqs port=$port ----"
    run_cmd env \
      RUN_ROOT="$RUN_ROOT" \
      MODEL="$MODEL" \
      PLAN_PATH="$PLAN_PATH" \
      HOT_K="$HOT_K" \
      KV_BYTES="$KV_BYTES" \
      STREAMING_BYTES="$streaming_bytes" \
      STREAMING_TTL_STEPS="$STREAMING_TTL_STEPS" \
      STREAMING_MAX_BYTES_PER_STEP="$streaming_bytes" \
      STREAMING_MAX_EXPERTS_PER_LAYER="$STREAMING_MAX_EXPERTS_PER_LAYER" \
      STREAMING_SYNC_PREFETCH="$STREAMING_SYNC_PREFETCH" \
      STREAMING_SYNC_EVICT="$STREAMING_SYNC_EVICT" \
      STREAMING_REAL_QUERY_ENABLE="$STREAMING_REAL_QUERY_ENABLE" \
      EXPECT_EVICTIONS=0 \
      EXPECT_DRIVER_TAGS=1 \
      DRIVER_TAG_ENABLE=1 \
      INIT_COLD_OFFLOAD_ENABLE=1 \
      INIT_COLD_OFFLOAD_MODE="$INIT_COLD_OFFLOAD_MODE" \
      INIT_COLD_OFFLOAD_SYNC="$INIT_COLD_OFFLOAD_SYNC" \
      MAX_NUM_SEQS="$max_num_seqs" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      PROMPTS="$PROMPTS" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      REQUEST_RATE="$REQUEST_RATE" \
      PORT="$port" \
      GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
      WEIGHT_EXECUTOR="$WEIGHT_EXECUTOR" \
      DRY_RUN="$DRY_RUN" \
      "$SCRIPT_DIR/run_stage_m_cold_streaming_sanity.sh"
  done
done

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

python3 - "$RUN_ROOT" "$STREAMING_BYTES_LIST" "$MAX_NUM_SEQS_LIST" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
rows = []
for static_path in sorted(run_root.glob("cold_stream_hot*_stream*_kv*_s*/static_hot_residency_summary.json")):
    case_dir = static_path.parent
    cold_path = case_dir / "cold_streaming_summary.json"
    try:
        static = json.loads(static_path.read_text())
        cold = json.loads(cold_path.read_text())
    except Exception:
        continue
    match = re.search(r"_stream(\d+)_kv(\d+)_s(\d+)$", case_dir.name)
    streaming_bytes = int(match.group(1)) if match else None
    kv_bytes = int(match.group(2)) if match else None
    max_num_seqs = int(match.group(3)) if match else None
    bench = cold.get("benchmark", {})
    cstats = cold.get("cold_streaming", {})
    tags = cold.get("driver_tags", {})
    hot = static.get("hot_prefetch", {})
    rows.append({
        "case": case_dir.name,
        "streaming_bytes": streaming_bytes,
        "streaming_gib": (streaming_bytes or 0) / (1024 ** 3),
        "kv_bytes": kv_bytes,
        "max_num_seqs": max_num_seqs,
        "static_passed": static.get("passed"),
        "cold_passed": cold.get("passed"),
        "successful_requests": bench.get("successful_requests"),
        "failed_requests": bench.get("failed_requests"),
        "output_tok_s": bench.get("output_tok_s"),
        "mean_ttft_ms": bench.get("mean_ttft_ms"),
        "mean_tpot_ms": bench.get("mean_tpot_ms"),
        "hot_gpu_ratio": hot.get("hot_real_gpu_ratio"),
        "cold_offload_records": hot.get("init_weight_offload_issued_records"),
        "cold_offload_gib": (hot.get("init_weight_offload_issued_bytes") or 0) / (1024 ** 3),
        "cold_miss_experts": cstats.get("cold_miss_experts"),
        "cold_prefetch_gib": cstats.get("streaming_prefetch_issued_gib"),
        "cold_eviction_count": cstats.get("streaming_eviction_count"),
        "cold_eviction_gib": cstats.get("streaming_eviction_gib"),
        "max_live_gib": cstats.get("max_streaming_live_gib"),
        "hot_evicted_by_cold": cstats.get("hot_evicted_by_cold_count"),
        "driver_hot_tags": tags.get("hot_driver_tag_records"),
        "driver_cold_tags": tags.get("cold_driver_tag_records"),
        "driver_streaming_tags": tags.get("streaming_driver_tag_records"),
        "driver_streaming_protected": tags.get("streaming_driver_protected_records"),
    })

csv_path = run_root / "cold_streaming_sweep_summary.csv"
md_path = run_root / "cold_streaming_sweep_summary.md"
if rows:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
lines = [
    "# Stage M Cold Streaming Sweep Summary",
    "",
    "| streaming GiB | max seqs | static | cold | tok/s | TTFT ms | TPOT ms | hot GPU ratio | cold offload GiB | prefetch GiB | evict GiB | max live GiB | hot evicted | driver tags hot/cold/stream |",
    "|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    lines.append(
        "| {streaming_gib:.1f} | {max_num_seqs} | {static_passed} | {cold_passed} | "
        "{output_tok_s} | {mean_ttft_ms} | {mean_tpot_ms} | {hot_gpu_ratio} | "
        "{cold_offload_gib:.3f} | {cold_prefetch_gib} | {cold_eviction_gib} | "
        "{max_live_gib} | {hot_evicted_by_cold} | {driver_hot_tags}/{driver_cold_tags}/{driver_streaming_tags} |".format(**row)
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote {csv_path}")
print(f"wrote {md_path}")
PY

echo "Summary: $RUN_ROOT/cold_streaming_sweep_summary.md"
