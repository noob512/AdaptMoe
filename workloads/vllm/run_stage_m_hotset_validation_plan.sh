#!/usr/bin/env bash
# Stage M hotset validation plan.
#
# This script follows the "prove value before adding complexity" sequence:
#   0) collect routing trace without prefetch/offload
#   1) replay hotset policies offline
#   2) microbenchmark UVM expert range migration
#   3) measure KV sizing/residency value with runtime expert prefetch disabled
#   4) run live prefetch K sweep with hit-before-access tracing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-32}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
REQUEST_RATE="${REQUEST_RATE:-8}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
MAX_NUM_SEQS_LIST="${MAX_NUM_SEQS_LIST:-1 4 8 16 32 64}"
MAX_NUM_BATCHED_TOKENS_LIST="${MAX_NUM_BATCHED_TOKENS_LIST:-0}"
K_VALUES="${K_VALUES:-0 16 32 64 96 128}"
REPEAT="${REPEAT:-1}"
RUN_ROOT="${RUN_ROOT:-log/stage_m_hotset_validation_$(date -u +%Y%m%d_%H%M%S)}"

RUN_STAGE0="${RUN_STAGE0:-1}"
RUN_STAGE1="${RUN_STAGE1:-1}"
RUN_STAGE2="${RUN_STAGE2:-1}"
RUN_STAGE3="${RUN_STAGE3:-0}"
RUN_STAGE4="${RUN_STAGE4:-0}"
DRY_RUN="${DRY_RUN:-0}"

DECODE_MAX_TOKENS="${DECODE_MAX_TOKENS:-1}"
EXPERT_BYTES="${EXPERT_BYTES:-4718592}"
MICROBENCH_BATCH_SIZES="${MICROBENCH_BATCH_SIZES:-1,16,32,64,96,128}"
MICROBENCH_REPEAT="${MICROBENCH_REPEAT:-5}"
MICROBENCH_EXECUTORS="${MICROBENCH_EXECUTORS:-single,cpp_batch,driver_batch}"

HOT_EXPERTS_PER_LAYER="${HOT_EXPERTS_PER_LAYER:-64}"
COLD_EXPERTS_PER_LAYER="${COLD_EXPERTS_PER_LAYER:-64}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES_LIST="${KV_GPU_BYTES_LIST:-1073741824 2147483648 4294967296 8589934592}"
INIT_WEIGHT_MAX_BYTES="${INIT_WEIGHT_MAX_BYTES:-8589934592}"
INIT_WEIGHT_MAX_EXPERTS_PER_LAYER="${INIT_WEIGHT_MAX_EXPERTS_PER_LAYER:-16}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-auto}"
WEIGHT_PREFETCH_MAX_BYTES_PER_STEP="${WEIGHT_PREFETCH_MAX_BYTES_PER_STEP:-536870912}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_hotset_validation_plan.sh [options]

Options can also be set via environment variables.

Main options:
  --run-root <path>
  --model <name>
  --prompts <n>
  --output-len <n>
  --request-rate <f>
  --max-num-seqs-list "1 4 8 16 32 64"
  --max-num-batched-tokens-list "0 1024 2048"
  --k-values "0 16 32 64 96 128"
  --repeat <n>
  --dry-run <0|1>

Stage gates:
  --run-stage0 <0|1>
  --run-stage1 <0|1>
  --run-stage2 <0|1>
  --run-stage3 <0|1>
  --run-stage4 <0|1>

Notes:
  Stage 0/1/2 are safe first passes. Stage 3/4 start heavier full_impl vLLM
  sweeps and are disabled by default.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --server-gpu-memory-utilization) SERVER_GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
    --max-num-seqs-list) MAX_NUM_SEQS_LIST="$2"; shift 2 ;;
    --max-num-batched-tokens-list) MAX_NUM_BATCHED_TOKENS_LIST="$2"; shift 2 ;;
    --k-values) K_VALUES="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    --run-stage0) RUN_STAGE0="$2"; shift 2 ;;
    --run-stage1) RUN_STAGE1="$2"; shift 2 ;;
    --run-stage2) RUN_STAGE2="$2"; shift 2 ;;
    --run-stage3) RUN_STAGE3="$2"; shift 2 ;;
    --run-stage4) RUN_STAGE4="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RUN_ROOT" != /* ]]; then
  RUN_ROOT="$PWD/$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"
SUMMARY_CSV="$RUN_ROOT/stage_m_hotset_validation_summary.csv"
COMMAND_LOG="$RUN_ROOT/commands.log"
: > "$COMMAND_LOG"

run_cmd() {
  printf 'command:' | tee -a "$COMMAND_LOG"
  printf ' %q' "$@" | tee -a "$COMMAND_LOG"
  printf '\n' | tee -a "$COMMAND_LOG"
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

batch_token_args() {
  local value="$1"
  if [ "$value" != "0" ] && [ -n "$value" ]; then
    printf '%s\0%s\0' --server-max-num-batched-tokens "$value"
  fi
}

trace_path_for() {
  local seqs="$1"
  local batched_tokens="$2"
  printf '%s/stage0_trace/s%s_bt%s/vllm_uvm_moe_routing_trace.jsonl' \
    "$RUN_ROOT" "$seqs" "$batched_tokens"
}

echo "==========================================================="
echo "Stage M Hotset Validation Plan"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "max_num_seqs=$MAX_NUM_SEQS_LIST"
echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS_LIST"
echo "K=$K_VALUES"
echo "dry_run=$DRY_RUN stage0=$RUN_STAGE0 stage1=$RUN_STAGE1 stage2=$RUN_STAGE2 stage3=$RUN_STAGE3 stage4=$RUN_STAGE4"
echo "==========================================================="

if [ "$RUN_STAGE0" = "1" ]; then
  for seqs in $MAX_NUM_SEQS_LIST; do
    for batched_tokens in $MAX_NUM_BATCHED_TOKENS_LIST; do
      case_dir="$RUN_ROOT/stage0_trace/s${seqs}_bt${batched_tokens}"
      mkdir -p "$case_dir"
      extra=()
      while IFS= read -r -d '' item; do
        extra+=("$item")
      done < <(batch_token_args "$batched_tokens")
      run_cmd "$SCRIPT_DIR/run_kv_fault_ratio.sh" \
        --model "$MODEL" \
        --mode trace \
        --with-address-log \
        --allocator-log "$case_dir/vllm_uvm_allocator.log" \
        --trace-log "$case_dir/uvm_fault_stats.log" \
        --address-trace-log "$case_dir/uvm_fault_addrs.log" \
        --address-log "$case_dir/vllm_uvm_address_regions.log" \
        --server-log "$case_dir/vllm_server.log" \
        --bench-log "$case_dir/vllm_bench.log" \
        --uvm-policy-enable 0 \
        --uvm-device-direct-enable 0 \
        --uvm-residency-enable 0 \
        --uvm-kv-runtime-enable 0 \
        --uvm-weight-prefetch-enable 0 \
        --uvm-weight-offload-enable 0 \
        --uvm-weight-batch-executor-enable 0 \
        --uvm-pool-coordinator-enable 0 \
        --uvm-pool-registry-enable 0 \
        --uvm-scratch-pool-enable 0 \
        --uvm-moe-routing-trace-enable 1 \
        --uvm-moe-routing-trace-file "$case_dir/vllm_uvm_moe_routing_trace.jsonl" \
        --server-max-num-seqs "$seqs" \
        "${extra[@]}" \
        --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
        --prompts "$PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --output-len "$OUTPUT_LEN"
    done
  done
fi

if [ "$RUN_STAGE1" = "1" ]; then
  k_csv="$(echo "$K_VALUES" | tr ' ' ',')"
  for seqs in $MAX_NUM_SEQS_LIST; do
    for batched_tokens in $MAX_NUM_BATCHED_TOKENS_LIST; do
      trace="$(trace_path_for "$seqs" "$batched_tokens")"
      out_dir="$RUN_ROOT/stage1_replay/s${seqs}_bt${batched_tokens}"
      if [ ! -s "$trace" ]; then
        echo "skip_stage1_missing_trace=$trace" | tee -a "$COMMAND_LOG"
        continue
      fi
      run_cmd python3 "$SCRIPT_DIR/analyze_moe_routing_trace.py" \
        --trace "$trace" \
        --out-dir "$out_dir" \
        --k-values "$k_csv" \
        --decode-max-tokens "$DECODE_MAX_TOKENS" \
        --expert-bytes "$EXPERT_BYTES"
    done
  done
fi

if [ "$RUN_STAGE2" = "1" ]; then
  run_cmd python3 "$SCRIPT_DIR/bench_uvm_expert_prefetch.py" \
    --out-dir "$RUN_ROOT/stage2_microbench" \
    --expert-bytes "$EXPERT_BYTES" \
    --batch-sizes "$MICROBENCH_BATCH_SIZES" \
    --executors "$MICROBENCH_EXECUTORS" \
    --repeat "$MICROBENCH_REPEAT"
fi

if [ "$RUN_STAGE3" = "1" ]; then
  for seqs in $MAX_NUM_SEQS_LIST; do
    for kv_bytes in $KV_GPU_BYTES_LIST; do
      for rep in $(seq 1 "$REPEAT"); do
        case_dir="$RUN_ROOT/stage3_kv_fixed/s${seqs}_kv${kv_bytes}_rep${rep}"
        run_cmd "$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
          --run-root "$case_dir" \
          --log-file "$case_dir/run.log" \
          --model "$MODEL" \
          --prompts "$PROMPTS" \
          --output-len "$OUTPUT_LEN" \
          --weight-executor "$WEIGHT_EXECUTOR" \
          --init-weight-offload-enable 1 \
          --weight-runtime-prefetch-enable 0 \
          --weight-prefetch-trace-enable 1 \
          --scratch-device-direct-enable 0 \
          --scratch-pool-enable 0 \
          --residency-enable 1 \
          --residency-trace-enable 1 \
          --kv-runtime-enable 0 \
          --pool-coordinator-enable 0 \
          --pool-coordinator-trace-enable 0 \
          --pool-registry-enable 1 \
          --auto-gap-watch-enable 0 \
          --fault-address-log-enable 1 \
          --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
          --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
          --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
          --kv-gpu-bytes "$kv_bytes" \
          --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
          --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" \
          --server-max-num-seqs "$seqs" \
          --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
          --no-require-driver
      done
    done
  done
fi

if [ "$RUN_STAGE4" = "1" ]; then
  for seqs in $MAX_NUM_SEQS_LIST; do
    for k in $K_VALUES; do
      for rep in $(seq 1 "$REPEAT"); do
        case_dir="$RUN_ROOT/stage4_live_prefetch/s${seqs}_k${k}_rep${rep}"
        enable=1
        if [ "$k" = "0" ]; then
          enable=0
        fi
        run_cmd "$SCRIPT_DIR/run_stage_m_real_residency_check.sh" \
          --run-root "$case_dir" \
          --log-file "$case_dir/run.log" \
          --model "$MODEL" \
          --prompts "$PROMPTS" \
          --output-len "$OUTPUT_LEN" \
          --weight-executor "$WEIGHT_EXECUTOR" \
          --init-weight-offload-enable 1 \
          --weight-runtime-prefetch-enable "$enable" \
          --weight-prefetch-trace-enable 1 \
          --weight-prefetch-source router_topk \
          --weight-prefetch-hit-trace-enable 1 \
          --weight-prefetch-hit-driver-query-enable 1 \
          --weight-prefetch-hit-max-regions-per-step 128 \
          --weight-prefetch-region-cooldown-steps 0 \
          --weight-prefetch-expert-cache-enable 1 \
          --scratch-device-direct-enable 0 \
          --scratch-pool-enable 0 \
          --residency-enable 1 \
          --residency-trace-enable 1 \
          --kv-runtime-enable 0 \
          --pool-coordinator-enable 0 \
          --pool-coordinator-trace-enable 0 \
          --pool-registry-enable 1 \
          --auto-gap-watch-enable 0 \
          --fault-address-log-enable 1 \
          --hot-experts-per-layer "$HOT_EXPERTS_PER_LAYER" \
          --cold-experts-per-layer "$COLD_EXPERTS_PER_LAYER" \
          --weight-gpu-bytes "$WEIGHT_GPU_BYTES" \
          --kv-gpu-bytes "$(echo "$KV_GPU_BYTES_LIST" | awk '{print $1}')" \
          --init-weight-max-bytes "$INIT_WEIGHT_MAX_BYTES" \
          --init-weight-max-experts-per-layer "$INIT_WEIGHT_MAX_EXPERTS_PER_LAYER" \
          --server-max-num-seqs "$seqs" \
          --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
          --no-require-driver \
          -- \
          --weight-prefetch-max-experts-per-layer "$k" \
          --weight-prefetch-max-bytes-per-step "$WEIGHT_PREFETCH_MAX_BYTES_PER_STEP"
      done
    done
  done
fi

python3 - "$RUN_ROOT" "$SUMMARY_CSV" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
out_csv = Path(sys.argv[2])

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

def first_float(blob: str, pattern: str):
    match = re.search(pattern, blob)
    return float(match.group(1).replace(",", "")) if match else None

def read_csv_row(path: Path, variant: str = "full_impl_enforce"):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("variant") == variant:
                return row
    return {}

rows = []
for stage in ("stage3_kv_fixed", "stage4_live_prefetch"):
    stage_dir = root / stage
    if not stage_dir.exists():
        continue
    for case in sorted(stage_dir.iterdir()):
        if not case.is_dir():
            continue
        summary = read_csv_row(case / "routing_phase_f_full_impl" / "summary.csv")
        server = read_text(case / "routing_phase_f_full_impl" / "runs" / "full_impl_enforce_rep1" / "vllm_server_full_impl_enforce_rep1.log")
        bench = read_text(case / "routing_phase_f_full_impl" / "runs" / "full_impl_enforce_rep1" / "vllm_bench_full_impl_enforce_rep1.log")
        phase = {}
        try:
            phase = json.loads((case / "phase_f_check.json").read_text(encoding="utf-8"))
        except Exception:
            pass
        row = {
            "stage": stage,
            "case": case.name,
            "passed": phase.get("passed"),
            "request_throughput_req_s": first_float(bench, r"Request throughput \(req/s\):\s+([0-9.]+)"),
            "output_tok_s": first_float(bench, r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
            "mean_ttft_ms": first_float(bench, r"Mean TTFT \(ms\):\s+([0-9.]+)"),
            "mean_tpot_ms": first_float(bench, r"Mean TPOT \(ms\):\s+([0-9.]+)"),
            "gpu_kv_cache_tokens": first_float(server, r"GPU KV cache size:\s+([0-9,]+)\s+tokens"),
            "available_kv_cache_gib": first_float(server, r"Available KV cache memory:\s+([0-9.]+)\s+GiB"),
            "prefetch_issued": summary.get("weight_prefetch_issued"),
            "prefetch_candidate_regions": summary.get("weight_prefetch_candidate_regions"),
            "prefetch_hit_records": summary.get("weight_prefetch_hit_before_access_records"),
            "prefetch_hit_gpu_records": summary.get("weight_prefetch_hit_before_access_gpu_records"),
            "faults_dedup": summary.get("delta_after_dedup"),
        }
        rows.append(row)

fields = [
    "stage", "case", "passed", "request_throughput_req_s", "output_tok_s",
    "mean_ttft_ms", "mean_tpot_ms", "gpu_kv_cache_tokens",
    "available_kv_cache_gib", "prefetch_issued", "prefetch_candidate_regions",
    "prefetch_hit_records", "prefetch_hit_gpu_records", "faults_dedup",
]
with out_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

md = root / "stage_m_hotset_validation_summary.md"
lines = [
    "# Stage M Hotset Validation Summary",
    "",
    f"- summary_csv: `{out_csv}`",
    f"- commands: `{root / 'commands.log'}`",
    "",
    "| stage | case | passed | tok/s | TPOT ms | GPU KV tokens | prefetch issued | hit gpu records | faults dedup |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
def fmt(value):
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)
for row in rows:
    lines.append(
        "| {stage} | {case} | {passed} | {tok} | {tpot} | {kv} | {issued} | {hit} | {faults} |".format(
            stage=row["stage"],
            case=row["case"],
            passed=row["passed"],
            tok=fmt(row["output_tok_s"]),
            tpot=fmt(row["mean_tpot_ms"]),
            kv=fmt(row["gpu_kv_cache_tokens"]),
            issued=fmt(row["prefetch_issued"]),
            hit=fmt(row["prefetch_hit_gpu_records"]),
            faults=fmt(row["faults_dedup"]),
        )
    )
md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(md.read_text(encoding="utf-8"))
PY

echo "Stage M hotset validation plan finished."
echo "summary_csv: $SUMMARY_CSV"
echo "summary_md: $RUN_ROOT/stage_m_hotset_validation_summary.md"
echo "commands: $COMMAND_LOG"
