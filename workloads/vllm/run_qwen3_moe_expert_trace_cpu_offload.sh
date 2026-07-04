#!/usr/bin/env bash
# Run the README CPU-offload baseline and trace Qwen3 MoE expert activation.
# This script only enables Python-level routing trace. It does not rebuild vLLM.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/results/qwen3_moe_expert_trace_cpu_offload_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-100}"
OUTPUT_LEN="${OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-5}"
SEED="${SEED:-42}"
PORT="${PORT:-8130}"
CPU_MAX_NUM_SEQS="${CPU_MAX_NUM_SEQS:-100}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
VLLM_BIN="${VLLM_BIN:-}"
USE_UV_RUN="${USE_UV_RUN:-0}"

mkdir -p "$RUN_ROOT"

VLLM_CMD=()
if [[ -n "$VLLM_BIN" ]]; then
  VLLM_CMD=("$VLLM_BIN")
elif [[ -x "$SCRIPT_DIR/.venv/bin/vllm" ]]; then
  VLLM_CMD=("$SCRIPT_DIR/.venv/bin/vllm")
elif [[ "$USE_UV_RUN" == "1" ]]; then
  VLLM_CMD=(uv run --directory "$SCRIPT_DIR" --no-sync vllm)
else
  echo "ERROR: no existing vllm executable found." >&2
  echo "Set VLLM_BIN=/path/to/vllm, or set USE_UV_RUN=1 to use uv run --no-sync." >&2
  exit 1
fi

TRACE_JSONL="$RUN_ROOT/qwen3_moe_expert_trace.jsonl"
SERVER_LOG="$RUN_ROOT/server.log"
BENCH_LOG="$RUN_ROOT/bench.log"
SUMMARY_MD="$RUN_ROOT/qwen3_moe_expert_trace_summary.md"
SUMMARY_CSV="$RUN_ROOT/qwen3_moe_expert_trace_layer_summary.csv"
SUMMARY_ROUND_CSV="$RUN_ROOT/qwen3_moe_expert_trace_round_summary.csv"
SUMMARY_JSON="$RUN_ROOT/qwen3_moe_expert_trace_summary.json"
TOKEN_ANALYSIS_MD="$RUN_ROOT/qwen3_moe_token_expert_trace_analysis.md"
TOKEN_ANALYSIS_JSON="$RUN_ROOT/qwen3_moe_token_expert_trace_analysis.json"
TOKEN_LAYER_CSV="$RUN_ROOT/qwen3_moe_token_expert_trace_layer_summary.csv"
TOKEN_LAYER_PAIR_CSV="$RUN_ROOT/qwen3_moe_token_expert_trace_layer_pair_summary.csv"
TIER_ANALYSIS_MD="$RUN_ROOT/qwen3_moe_tier_adjacent_similarity.md"
TIER_ANALYSIS_JSON="$RUN_ROOT/qwen3_moe_tier_adjacent_similarity.json"
TIER_ANALYSIS_CSV="$RUN_ROOT/qwen3_moe_tier_adjacent_similarity.csv"
FIXED_TIER_STABILITY_MD="$RUN_ROOT/qwen3_moe_fixed_tier_stability.md"
FIXED_TIER_STABILITY_JSON="$RUN_ROOT/qwen3_moe_fixed_tier_stability.json"
FIXED_TIER_STABILITY_CSV="$RUN_ROOT/qwen3_moe_fixed_tier_stability.csv"

cleanup_gpu() {
  pkill -f "vllm bench serve" >/dev/null 2>&1 || true
  pkill -f "vllm serve $MODEL" >/dev/null 2>&1 || true
  python3 "$REPO_ROOT/workloads/cleanup_gpu.py" >/dev/null 2>&1 || true
}

wait_for_server() {
  local port="$1"
  local server_pid="$2"
  local server_log="$3"
  local start_ts now elapsed
  start_ts="$(date +%s)"
  while true; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Server exited before readiness. Log tail:" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    now="$(date +%s)"
    elapsed=$((now - start_ts))
    if [[ "$elapsed" -ge "$STARTUP_TIMEOUT" ]]; then
      echo "Timed out waiting for server readiness after ${elapsed}s" >&2
      tail -n 120 "$server_log" >&2 2>/dev/null || true
      return 1
    fi
    sleep "$CHECK_INTERVAL"
  done
}

write_metadata() {
  {
    echo "case=qwen3_moe_expert_trace_cpu_offload"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "model=$MODEL"
    echo "dataset_path=$DATASET_PATH"
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "seed=$SEED"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$CPU_MAX_NUM_SEQS"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "vllm_cmd=${VLLM_CMD[*]}"
    echo "trace_jsonl=$TRACE_JSONL"
    echo "token_analysis_md=$TOKEN_ANALYSIS_MD"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    modinfo nvidia_uvm 2>/dev/null | sed -n '1,20p' || true
  } > "$RUN_ROOT/metadata.txt"
}

summarize_trace() {
  python3 - "$RUN_ROOT" "$TRACE_JSONL" "$BENCH_LOG" "$SUMMARY_MD" "$SUMMARY_CSV" "$SUMMARY_ROUND_CSV" "$SUMMARY_JSON" <<'PY'
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
trace_path = Path(sys.argv[2])
bench_path = Path(sys.argv[3])
summary_md = Path(sys.argv[4])
summary_csv = Path(sys.argv[5])
summary_round_csv = Path(sys.argv[6])
summary_json = Path(sys.argv[7])

def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(blob: str, pattern: str):
    match = re.search(pattern, blob)
    return float(match.group(1)) if match else None

def percentile(values: list[int], q: float):
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]

bench_blob = text(bench_path)
bench = {
    "successful_requests": first_float(bench_blob, r"Successful requests:\s+([0-9.]+)"),
    "failed_requests": first_float(bench_blob, r"Failed requests:\s+([0-9.]+)"),
    "output_tok_s": first_float(bench_blob, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
    "mean_tpot_ms": first_float(bench_blob, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
    "mean_ttft_ms": first_float(bench_blob, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
    "duration_s": first_float(bench_blob, r"Benchmark duration \(s\):\s+([0-9.]+)"),
}

records = []
for line in text(trace_path).splitlines():
    try:
        rec = json.loads(line)
    except Exception:
        continue
    if rec.get("event") == "qwen3_moe_expert_trace":
        records.append(rec)

by_layer: dict[int, list[dict]] = defaultdict(list)
global_counter: Counter[int] = Counter()
step_union: dict[int, set[int]] = defaultdict(set)
for rec in records:
    layer = rec.get("layer_id")
    if layer is None:
        continue
    layer = int(layer)
    by_layer[layer].append(rec)
    step = int(rec.get("step") or 0)
    for expert_id in rec.get("active_experts") or []:
        expert_id = int(expert_id)
        global_counter[expert_id] += 1
        step_union[step].add(expert_id)

rounds = []
records_by_pid: dict[int, list[dict]] = defaultdict(list)
for rec in records:
    if rec.get("layer_id") is None:
        continue
    records_by_pid[int(rec.get("pid") or 0)].append(rec)

for pid, pid_records in sorted(records_by_pid.items()):
    current = []
    prev_layer = None
    round_id = 0
    for rec in sorted(pid_records, key=lambda x: int(x.get("step") or 0)):
        layer = int(rec["layer_id"])
        if current and prev_layer is not None and layer <= prev_layer:
            rounds.append((pid, round_id, current))
            round_id += 1
            current = []
        current.append(rec)
        prev_layer = layer
    if current:
        rounds.append((pid, round_id, current))

rows = []
for layer in sorted(by_layer):
    items = by_layer[layer]
    active_counts = [int(x.get("active_expert_count") or 0) for x in items]
    token_counts = [int(x.get("num_tokens") or 0) for x in items]
    assignment_counts = [int(x.get("expert_assignments") or 0) for x in items]
    union = set()
    expert_hits = Counter()
    for rec in items:
        for expert_id in rec.get("active_experts") or []:
            union.add(int(expert_id))
        for key, value in (rec.get("expert_token_counts") or {}).items():
            expert_hits[int(key)] += int(value)
    rows.append({
        "layer_id": layer,
        "records": len(items),
        "mean_tokens": statistics.mean(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "mean_expert_assignments": statistics.mean(assignment_counts) if assignment_counts else 0,
        "mean_active_experts": statistics.mean(active_counts) if active_counts else 0,
        "median_active_experts": statistics.median(active_counts) if active_counts else 0,
        "p95_active_experts": percentile(active_counts, 0.95),
        "max_active_experts": max(active_counts) if active_counts else 0,
        "union_active_experts": len(union),
        "top_experts_by_token_hits": expert_hits.most_common(16),
    })

with summary_csv.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "layer_id", "records", "mean_tokens", "max_tokens",
        "mean_expert_assignments", "mean_active_experts",
        "median_active_experts", "p95_active_experts", "max_active_experts",
        "union_active_experts", "top_experts_by_token_hits",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

round_rows = []
for pid, round_id, items in rounds:
    token_counts = [int(x.get("num_tokens") or 0) for x in items]
    active_counts = [int(x.get("active_expert_count") or 0) for x in items]
    assignment_counts = [int(x.get("expert_assignments") or 0) for x in items]
    layer_ids = [int(x.get("layer_id")) for x in items]
    active_layer_experts = {
        (int(x.get("layer_id")), int(expert_id))
        for x in items
        for expert_id in (x.get("active_experts") or [])
    }
    round_rows.append({
        "pid": pid,
        "round_id": round_id,
        "records": len(items),
        "first_step": min(int(x.get("step") or 0) for x in items),
        "last_step": max(int(x.get("step") or 0) for x in items),
        "first_layer": min(layer_ids) if layer_ids else None,
        "last_layer": max(layer_ids) if layer_ids else None,
        "mean_tokens": statistics.mean(token_counts) if token_counts else 0,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "total_token_expert_assignments": sum(assignment_counts),
        "total_active_layer_experts": len(active_layer_experts),
        "sum_active_experts_by_layer": sum(active_counts),
        "max_layer_active_experts": max(active_counts) if active_counts else 0,
    })

with summary_round_csv.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "pid", "round_id", "records", "first_step", "last_step",
        "first_layer", "last_layer", "mean_tokens", "min_tokens",
        "max_tokens", "total_token_expert_assignments",
        "total_active_layer_experts", "sum_active_experts_by_layer",
        "max_layer_active_experts",
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in round_rows:
        writer.writerow(row)

step_union_counts = [len(v) for v in step_union.values()]
round_token_counts = [int(row["max_tokens"]) for row in round_rows]
round_active_layer_expert_counts = [
    int(row["total_active_layer_experts"]) for row in round_rows
]
summary = {
    "run_root": str(root),
    "trace_records": len(records),
    "layers": rows,
    "rounds": round_rows,
    "bench": bench,
    "global_unique_experts": len(global_counter),
    "global_top_experts_by_layer_hits": global_counter.most_common(32),
    "mean_step_global_union_experts": statistics.mean(step_union_counts) if step_union_counts else 0,
    "max_step_global_union_experts": max(step_union_counts) if step_union_counts else 0,
    "round_count": len(round_rows),
    "mean_round_tokens": statistics.mean(round_token_counts) if round_token_counts else 0,
    "p95_round_tokens": percentile(round_token_counts, 0.95),
    "max_round_tokens": max(round_token_counts) if round_token_counts else 0,
    "mean_round_active_layer_experts": (
        statistics.mean(round_active_layer_expert_counts)
        if round_active_layer_expert_counts
        else 0
    ),
    "p95_round_active_layer_experts": percentile(
        round_active_layer_expert_counts, 0.95
    ),
    "max_round_active_layer_experts": (
        max(round_active_layer_expert_counts)
        if round_active_layer_expert_counts
        else 0
    ),
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

def fmt(value, digits=2):
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

lines = [
    "# Qwen3 MoE Expert Activation Trace",
    "",
    f"- run_root: `{root}`",
    f"- trace_records: `{len(records)}`",
    f"- successful/failed requests: `{bench['successful_requests']}` / `{bench['failed_requests']}`",
    f"- output tok/s: `{bench['output_tok_s']}`",
    f"- mean TPOT ms: `{bench['mean_tpot_ms']}`",
    f"- mean TTFT ms: `{bench['mean_ttft_ms']}`",
    f"- global unique expert ids seen: `{len(global_counter)}`",
    f"- observed rounds: `{summary['round_count']}`",
    f"- mean/max tokens per round: `{fmt(summary['mean_round_tokens'])}` / `{summary['max_round_tokens']}`",
    f"- mean/max active layer-experts per round: `{fmt(summary['mean_round_active_layer_experts'])}` / `{summary['max_round_active_layer_experts']}`",
    f"- round summary csv: `{summary_round_csv}`",
    "",
    "## Per-Layer Summary",
    "",
    "| layer | records | mean tokens | max tokens | mean active experts | p95 active | max active | union experts | top experts by token hits |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    top = " ".join(f"{expert}:{hits}" for expert, hits in row["top_experts_by_token_hits"][:8])
    lines.append(
        "| {layer} | {records} | {mean_tokens} | {max_tokens} | "
        "{mean_active} | {p95_active} | {max_active} | {union} | {top} |".format(
            layer=row["layer_id"],
            records=row["records"],
            mean_tokens=fmt(row["mean_tokens"]),
            max_tokens=row["max_tokens"],
            mean_active=fmt(row["mean_active_experts"]),
            p95_active=row["p95_active_experts"],
            max_active=row["max_active_experts"],
            union=row["union_active_experts"],
            top=top,
        )
    )
lines.extend([
    "",
    "## Per-Round Summary",
    "",
    "| round | layers | mean tokens | max tokens | active layer-experts | max layer active experts |",
    "|---:|---:|---:|---:|---:|---:|",
])
for row in round_rows[:32]:
    lines.append(
        "| {round_id} | {records} | {mean_tokens} | {max_tokens} | "
        "{active_layer_experts} | {max_layer_active} |".format(
            round_id=row["round_id"],
            records=row["records"],
            mean_tokens=fmt(row["mean_tokens"]),
            max_tokens=row["max_tokens"],
            active_layer_experts=row["total_active_layer_experts"],
            max_layer_active=row["max_layer_active_experts"],
        )
    )
if len(round_rows) > 32:
    lines.append(f"| ... | ... | ... | ... | ... | ... |")

lines.extend([
    "",
    "## Reading The Trace",
    "",
    "- `records` is the number of MoE layer forwards observed for that layer.",
    "- `mean tokens` is how many hidden-state rows reached the MoE router in one forward.",
    "- `mean active experts` is the per-forward expert union size for that layer.",
    "- `union experts` is how many distinct expert ids appeared across the whole run for that layer.",
    "- `top experts by token hits` counts routed token-expert assignments, not unique requests.",
    "- `round` starts a new group whenever traced layer ids wrap back to an earlier layer.",
    "- `active layer-experts` treats `(layer_id, expert_id)` as the cache/prefetch unit.",
])
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(summary_md.read_text(encoding="utf-8"))
PY
}

echo "==========================================================="
echo "Qwen3 MoE expert trace with vLLM CPU offload baseline"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "vllm_cmd=${VLLM_CMD[*]}"
echo "==========================================================="

write_metadata
: > "$TRACE_JSONL"
: > "$SERVER_LOG"
: > "$BENCH_LOG"

cleanup_gpu
sleep 2

(
  cd "$SCRIPT_DIR" || exit 1
  env -u VLLM_USE_UVM -u LD_PRELOAD \
    VLLM_QWEN3_MOE_EXPERT_TRACE_ENABLE=1 \
    VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE=1 \
    VLLM_QWEN3_MOE_EXPERT_TRACE_FILE="$TRACE_JSONL" \
    "${VLLM_CMD[@]}" serve "$MODEL" \
      --enforce-eager \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$CPU_MAX_NUM_SEQS" \
      --cpu-offload-gb "$CPU_OFFLOAD_GB" \
      --port "$PORT" >"$SERVER_LOG" 2>&1
) &
server_pid=$!

status=0
if wait_for_server "$PORT" "$server_pid" "$SERVER_LOG"; then
  set +e
  (
    cd "$SCRIPT_DIR" || exit 1
    "${VLLM_CMD[@]}" bench serve \
      --model "$MODEL" \
      --dataset-name sharegpt \
      --dataset-path "$DATASET_PATH" \
      --num-prompts "$PROMPTS" \
      --sharegpt-output-len "$OUTPUT_LEN" \
      --seed "$SEED" \
      --request-rate "$REQUEST_RATE" \
      --port "$PORT"
  ) 2>&1 | tee -a "$BENCH_LOG"
  status=${PIPESTATUS[0]}
  set -u
else
  status=1
fi

echo "$status" > "$RUN_ROOT/status.txt"
if kill -0 "$server_pid" 2>/dev/null; then
  kill -TERM "$server_pid" >/dev/null 2>&1 || true
  sleep 3
  kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
  wait "$server_pid" >/dev/null 2>&1 || true
fi
cleanup_gpu

summarize_trace
python3 "$SCRIPT_DIR/analyze_qwen3_moe_token_expert_trace.py" "$TRACE_JSONL" \
  --output-md "$TOKEN_ANALYSIS_MD" \
  --output-json "$TOKEN_ANALYSIS_JSON" \
  --layer-csv "$TOKEN_LAYER_CSV" \
  --layer-pair-csv "$TOKEN_LAYER_PAIR_CSV"
python3 "$SCRIPT_DIR/analyze_qwen3_moe_tier_adjacent_similarity.py" "$TRACE_JSONL" \
  --output-md "$TIER_ANALYSIS_MD" \
  --output-json "$TIER_ANALYSIS_JSON" \
  --output-csv "$TIER_ANALYSIS_CSV"
python3 "$SCRIPT_DIR/analyze_qwen3_moe_fixed_tier_stability.py" "$TRACE_JSONL" \
  --output-md "$FIXED_TIER_STABILITY_MD" \
  --output-json "$FIXED_TIER_STABILITY_JSON" \
  --output-csv "$FIXED_TIER_STABILITY_CSV"

echo "summary_md=$SUMMARY_MD"
echo "summary_csv=$SUMMARY_CSV"
echo "token_analysis_md=$TOKEN_ANALYSIS_MD"
echo "token_layer_csv=$TOKEN_LAYER_CSV"
echo "token_layer_pair_csv=$TOKEN_LAYER_PAIR_CSV"
echo "tier_analysis_md=$TIER_ANALYSIS_MD"
echo "tier_analysis_csv=$TIER_ANALYSIS_CSV"
echo "fixed_tier_stability_md=$FIXED_TIER_STABILITY_MD"
echo "fixed_tier_stability_csv=$FIXED_TIER_STABILITY_CSV"
echo "trace_jsonl=$TRACE_JSONL"
exit "$status"
