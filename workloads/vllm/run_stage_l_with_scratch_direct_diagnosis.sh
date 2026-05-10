#!/usr/bin/env bash
# Layered Stage L/KV diagnosis under a common Stage C scratch-direct setup.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REPEAT="${REPEAT:-1}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_l_with_scratch_direct_$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE=""
SCRATCH_PROFILE="${SCRATCH_PROFILE:-16m}"
RUN_STAGE_L_BASELINE="${RUN_STAGE_L_BASELINE:-1}"
RUN_SCRATCH_BASELINE="${RUN_SCRATCH_BASELINE:-1}"
RUN_L7_CROSS="${RUN_L7_CROSS:-0}"
POLICY_ENABLE="${POLICY_ENABLE:-1}"

SCRATCH_TARGET_PHASES="${SCRATCH_TARGET_PHASES:-enabled:attention,enabled:moe}"
SCRATCH_TARGET_CLASS="${SCRATCH_TARGET_CLASS:-gap_hot_runtime_scratch}"
SCRATCH_TARGET_GAP="${SCRATCH_TARGET_GAP:-2}"
SCRATCH_MIN_BYTES="${SCRATCH_MIN_BYTES:-4096}"
SCRATCH_BACKEND="${SCRATCH_BACKEND:-cuda_malloc_async}"
SCRATCH_POOL_RELEASE_THRESHOLD="${SCRATCH_POOL_RELEASE_THRESHOLD:-1048576}"
SCRATCH_MAX_TOTAL_BYTES=""
SCRATCH_MAX_BYTES=""

GLOBAL_GPU_BYTES="${GLOBAL_GPU_BYTES:-25769803776}"    # 24 GiB
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-8589934592}"     # 8 GiB
KV_GPU_BYTES="${KV_GPU_BYTES:-1073741824}"             # 1 GiB
SCRATCH_GPU_BYTES="${SCRATCH_GPU_BYTES:-2147483648}"   # 2 GiB Stage L ledger
SAFETY_MARGIN_BYTES="${SAFETY_MARGIN_BYTES:-1073741824}"
RUNTIME_BUDGET_BLOCKS="${RUNTIME_BUDGET_BLOCKS:-1}"
RUNTIME_CANDIDATE_LIMIT="${RUNTIME_CANDIDATE_LIMIT:-16}"
MAX_OFFLOAD_BLOCKS="${MAX_OFFLOAD_BLOCKS:-4}"

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_l_with_scratch_direct_diagnosis.sh [options]

Options:
  --run-root <path>          Output root. Default: /tmp/vllm_stage_l_with_scratch_direct_<utc_stamp>
  --log-file <path>          Full shell output log. Default: <run-root>/run.log
  --model <name>             Model name. Default: Qwen/Qwen3-30B-A3B-FP8
  --prompts <int>            Benchmark prompts. Default: 4
  --output-len <int>         Generated tokens per request. Default: 256
  --request-rate <float>     Benchmark request rate. Default: 2
  --repeat <int>             Repeats per variant. Default: 1

  --scratch-profile <name>   Scratch direct profile:
                               4m   => total budget 4MiB, max allocation 4MiB
                               16m  => total budget 16MiB, max allocation 8MiB (default)
                               custom => use --scratch-total-bytes and --scratch-max-bytes
  --scratch-total-bytes <n>  Stage C scratch device-direct max live bytes.
  --scratch-max-bytes <n>    Stage C scratch device-direct max single allocation bytes.
  --scratch-target-phases <csv>
                             Default: enabled:attention,enabled:moe
  --scratch-backend <name>   cuda_malloc or cuda_malloc_async. Default: cuda_malloc_async
  --scratch-pool-release-threshold <n>
                             cuda_malloc_async mempool release threshold. Default: 1048576

  --global-gpu-bytes <n>     Stage L residency global budget. Default: 24GiB
  --weight-gpu-bytes <n>     Stage L weight pool budget. Default: 8GiB
  --kv-gpu-bytes <n>         Stage L KV pool budget. Default: 1GiB
  --scratch-gpu-bytes <n>    Stage L scratch ledger budget. Default: 2GiB
  --safety-margin-bytes <n>  Stage L safety margin. Default: 1GiB

  --skip-stage-l-baseline    Skip pure UVM-style baseline without scratch direct.
  --skip-scratch-baseline    Skip scratch-direct-only baseline.
  --with-l7-cross            Also run heavier L7 cross-pool variant.
  --policy-enable <0|1>      Allocator policy/classification logs. Default: 1.
                             Keep this enabled for scratch direct target-class matching.
  -h, --help                 Show this help.

This script runs a same-load layered diagnosis:
  baseline_pure_uvm              Stage L/J/K and scratch direct off
  scratch_direct_baseline        Stage C scratch direct on, Stage L/J/K off
  l5_metadata_scratch_direct     Stage L5 KV metadata + Stage C scratch direct
  l6_decision_trace_scratch      L6 decision trace + Stage C scratch direct
  l6_real_prefetch_scratch       L6 real KV prefetch/offload + Stage C scratch direct
  l7_coordinator_scratch         L7 coordinator + Stage C scratch direct
  l7_cross_pool_scratch          Optional heavier cross-pool L7 + Stage C scratch direct

Default recommended scratch profile:
  target phases: enabled:attention,enabled:moe
  total budget:  16777216 bytes (16MiB)
  max bytes:     8388608 bytes (8MiB)
  backend:       cuda_malloc_async

Main outputs:
  <run-root>/summary.csv
  <run-root>/summary.md
  <run-root>/runs/<variant>_rep<n>/*
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --scratch-profile) SCRATCH_PROFILE="$2"; shift 2 ;;
    --scratch-total-bytes) SCRATCH_MAX_TOTAL_BYTES="$2"; SCRATCH_PROFILE="custom"; shift 2 ;;
    --scratch-max-bytes) SCRATCH_MAX_BYTES="$2"; SCRATCH_PROFILE="custom"; shift 2 ;;
    --scratch-target-phases) SCRATCH_TARGET_PHASES="$2"; shift 2 ;;
    --scratch-backend) SCRATCH_BACKEND="$2"; shift 2 ;;
  --scratch-pool-release-threshold) SCRATCH_POOL_RELEASE_THRESHOLD="$2"; shift 2 ;;
    --policy-enable) POLICY_ENABLE="$2"; shift 2 ;;
    --global-gpu-bytes) GLOBAL_GPU_BYTES="$2"; shift 2 ;;
    --weight-gpu-bytes) WEIGHT_GPU_BYTES="$2"; shift 2 ;;
    --kv-gpu-bytes) KV_GPU_BYTES="$2"; shift 2 ;;
    --scratch-gpu-bytes) SCRATCH_GPU_BYTES="$2"; shift 2 ;;
    --safety-margin-bytes) SAFETY_MARGIN_BYTES="$2"; shift 2 ;;
    --skip-stage-l-baseline) RUN_STAGE_L_BASELINE=0; shift ;;
    --skip-scratch-baseline) RUN_SCRATCH_BASELINE=0; shift ;;
    --with-l7-cross) RUN_L7_CROSS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SCRATCH_PROFILE" in
  4m)
    SCRATCH_MAX_TOTAL_BYTES="${SCRATCH_MAX_TOTAL_BYTES:-4194304}"
    SCRATCH_MAX_BYTES="${SCRATCH_MAX_BYTES:-4194304}"
    ;;
  16m)
    SCRATCH_MAX_TOTAL_BYTES="${SCRATCH_MAX_TOTAL_BYTES:-16777216}"
    SCRATCH_MAX_BYTES="${SCRATCH_MAX_BYTES:-8388608}"
    ;;
  custom)
    if [ -z "$SCRATCH_MAX_TOTAL_BYTES" ] || [ -z "$SCRATCH_MAX_BYTES" ]; then
      echo "--scratch-profile custom requires --scratch-total-bytes and --scratch-max-bytes" >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown --scratch-profile: $SCRATCH_PROFILE" >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_l_with_scratch_direct.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage L with scratch-direct diagnosis started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

echo "==========================================================="
echo " Stage L With Scratch-Direct Diagnosis"
echo " run_root: $RUN_ROOT"
echo " log_file: $LOG_FILE"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " repeat: $REPEAT"
echo " scratch_profile: $SCRATCH_PROFILE"
echo " scratch_target_phases: $SCRATCH_TARGET_PHASES"
echo " scratch_total_bytes: $SCRATCH_MAX_TOTAL_BYTES"
echo " scratch_max_bytes: $SCRATCH_MAX_BYTES"
echo " scratch_backend: $SCRATCH_BACKEND"
echo " scratch_pool_release_threshold: ${SCRATCH_POOL_RELEASE_THRESHOLD:-<cuda-default>}"
echo " policy_enable: $POLICY_ENABLE"
echo " stage_l_global_gpu_bytes: $GLOBAL_GPU_BYTES"
echo " stage_l_weight_gpu_bytes: $WEIGHT_GPU_BYTES"
echo " stage_l_kv_gpu_bytes: $KV_GPU_BYTES"
echo " stage_l_scratch_gpu_bytes: $SCRATCH_GPU_BYTES"
echo " stage_l_safety_margin_bytes: $SAFETY_MARGIN_BYTES"
echo "==========================================================="

base_run_args() {
  local run_dir="$1"
  local label="$2"
  printf '%s\0' \
    --model "$MODEL" \
    --mode trace \
    --with-address-log \
    --allocator-log "$run_dir/vllm_uvm_allocator_trace_${label}.log" \
    --trace-log "$run_dir/uvm_kv_fault_stats_${label}.log" \
    --address-trace-log "$run_dir/uvm_kv_fault_addrs_${label}.log" \
    --address-log "$run_dir/vllm_uvm_address_regions_${label}.log" \
    --server-log "$run_dir/vllm_server_${label}.log" \
    --bench-log "$run_dir/vllm_bench_${label}.log" \
    --uvm-trace-min-bytes 1048576 \
    --uvm-unknown-detail-enable 1 \
    --uvm-unknown-detail-min-bytes 4096 \
    --uvm-gap-watch-name "$label" \
    --uvm-gap-watch-all-classes 1 \
    --uvm-gap-watch-min-bytes 4096 \
    --auto-gap-watch-enable 1 \
    --auto-gap-watch-probe-prompts 1 \
    --auto-gap-watch-target-gap "$SCRATCH_TARGET_GAP" \
    --auto-gap-watch-target-class-override "$SCRATCH_TARGET_CLASS" \
    --auto-gap-watch-summary-json "$run_dir/auto_gap_watch_probe_${label}.json" \
    --auto-gap-watch-post-main-summary-json "$run_dir/auto_gap_watch_post_main_${label}.json" \
    --gap-watch-metrics-summary-json "$run_dir/gap_watch_metrics_${label}.json" \
    --prompts "$PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --output-len "$OUTPUT_LEN"
}

run_kv() {
  local name="$1"
  local rep="$2"
  local scratch_mode="$3"
  local stage_mode="$4"
  local label="${name}_rep${rep}"
  local run_dir="$RUN_ROOT/runs/$label"
  mkdir -p "$run_dir"

  echo "==========================================================="
  echo " Running $name rep $rep/$REPEAT"
  echo " scratch_mode: $scratch_mode"
  echo " stage_mode: $stage_mode"
  echo " run_dir: $run_dir"
  echo "==========================================================="

  local args=()
  while IFS= read -r -d '' item; do
    args+=("$item")
  done < <(base_run_args "$run_dir" "$label")

  if [ "$scratch_mode" = "trace" ]; then
    args+=(
      --uvm-device-direct-enable 0
      --uvm-device-direct-min-bytes "$SCRATCH_MIN_BYTES"
      --uvm-device-direct-max-bytes "$SCRATCH_MAX_BYTES"
      --uvm-device-direct-max-total-bytes "$SCRATCH_MAX_TOTAL_BYTES"
      --uvm-device-direct-backend "$SCRATCH_BACKEND"
      --uvm-device-direct-pool-release-threshold "$SCRATCH_POOL_RELEASE_THRESHOLD"
      --uvm-device-direct-target-phases "$SCRATCH_TARGET_PHASES"
      --auto-gap-watch-policy-action-override device_direct_trace
    )
  else
    args+=(
      --uvm-device-direct-enable 1
      --uvm-device-direct-min-bytes "$SCRATCH_MIN_BYTES"
      --uvm-device-direct-max-bytes "$SCRATCH_MAX_BYTES"
      --uvm-device-direct-max-total-bytes "$SCRATCH_MAX_TOTAL_BYTES"
      --uvm-device-direct-backend "$SCRATCH_BACKEND"
      --uvm-device-direct-pool-release-threshold "$SCRATCH_POOL_RELEASE_THRESHOLD"
      --uvm-device-direct-target-phases "$SCRATCH_TARGET_PHASES"
      --auto-gap-watch-policy-action-override device_direct
    )
  fi

  case "$stage_mode" in
    none)
      ;;
    l5)
      args+=(
        --uvm-residency-enable 1
        --uvm-residency-stage stage_l5
        --uvm-residency-mode trace_only
        --uvm-residency-trace-file "$run_dir/vllm_uvm_pool_residency_l5.jsonl"
        --uvm-residency-global-gpu-bytes "$GLOBAL_GPU_BYTES"
        --uvm-residency-weight-gpu-bytes "$WEIGHT_GPU_BYTES"
        --uvm-residency-kv-gpu-bytes "$KV_GPU_BYTES"
        --uvm-residency-scratch-gpu-bytes "$SCRATCH_GPU_BYTES"
        --uvm-residency-safety-margin-bytes "$SAFETY_MARGIN_BYTES"
        --uvm-residency-kv-metadata-enable 1
        --uvm-kv-runtime-enable 1
        --uvm-kv-runtime-mode trace_only
        --uvm-kv-runtime-budget-blocks "$RUNTIME_BUDGET_BLOCKS"
        --uvm-kv-runtime-trace-file "$run_dir/vllm_uvm_kv_runtime_l5.jsonl"
        --uvm-kv-runtime-candidate-limit "$RUNTIME_CANDIDATE_LIMIT"
        --uvm-kv-runtime-prefix-evict-enable 0
      )
      ;;
    l6_trace)
      args+=(
        --uvm-residency-enable 1
        --uvm-residency-stage stage_l6
        --uvm-residency-mode trace_only
        --uvm-residency-trace-file "$run_dir/vllm_uvm_pool_residency_l6_trace.jsonl"
        --uvm-residency-global-gpu-bytes "$GLOBAL_GPU_BYTES"
        --uvm-residency-weight-gpu-bytes "$WEIGHT_GPU_BYTES"
        --uvm-residency-kv-gpu-bytes "$KV_GPU_BYTES"
        --uvm-residency-scratch-gpu-bytes "$SCRATCH_GPU_BYTES"
        --uvm-residency-safety-margin-bytes "$SAFETY_MARGIN_BYTES"
        --uvm-residency-kv-metadata-enable 1
        --uvm-residency-kv-data-offload-enable 1
        --uvm-residency-kv-data-offload-mode trace_only
        --uvm-residency-kv-prefetch-back-enable 1
        --uvm-residency-kv-prefetch-back-mode trace_only
        --uvm-residency-kv-max-offload-blocks-per-step "$MAX_OFFLOAD_BLOCKS"
        --uvm-residency-kv-pointer-map-file "$run_dir/vllm_uvm_kv_pointer_map_l6_trace.jsonl"
        --uvm-kv-runtime-enable 1
        --uvm-kv-runtime-mode trace_only
        --uvm-kv-runtime-budget-blocks "$RUNTIME_BUDGET_BLOCKS"
        --uvm-kv-runtime-trace-file "$run_dir/vllm_uvm_kv_runtime_l6_trace.jsonl"
        --uvm-kv-runtime-candidate-limit "$RUNTIME_CANDIDATE_LIMIT"
        --uvm-kv-runtime-prefix-evict-enable 0
      )
      ;;
    l6_real|l7)
      args+=(
        --uvm-residency-enable 1
        --uvm-residency-stage stage_l6
        --uvm-residency-mode trace_only
        --uvm-residency-trace-file "$run_dir/vllm_uvm_pool_residency_${stage_mode}.jsonl"
        --uvm-residency-global-gpu-bytes "$GLOBAL_GPU_BYTES"
        --uvm-residency-weight-gpu-bytes "$WEIGHT_GPU_BYTES"
        --uvm-residency-kv-gpu-bytes "$KV_GPU_BYTES"
        --uvm-residency-scratch-gpu-bytes "$SCRATCH_GPU_BYTES"
        --uvm-residency-safety-margin-bytes "$SAFETY_MARGIN_BYTES"
        --uvm-residency-kv-metadata-enable 1
        --uvm-residency-kv-data-offload-enable 1
        --uvm-residency-kv-data-offload-mode prefetch_cpu
        --uvm-residency-kv-prefetch-back-enable 1
        --uvm-residency-kv-prefetch-back-mode prefetch
        --uvm-residency-kv-max-offload-blocks-per-step "$MAX_OFFLOAD_BLOCKS"
        --uvm-residency-kv-pointer-map-file "$run_dir/vllm_uvm_kv_pointer_map_${stage_mode}.jsonl"
        --uvm-kv-runtime-enable 1
        --uvm-kv-runtime-mode trace_only
        --uvm-kv-runtime-budget-blocks "$RUNTIME_BUDGET_BLOCKS"
        --uvm-kv-runtime-trace-file "$run_dir/vllm_uvm_kv_runtime_${stage_mode}.jsonl"
        --uvm-kv-runtime-candidate-limit "$RUNTIME_CANDIDATE_LIMIT"
        --uvm-kv-runtime-prefix-evict-enable 0
      )
      if [ "$stage_mode" = "l7" ]; then
        args+=(
          --uvm-pool-coordinator-enable 1
          --uvm-pool-coordinator-mode trace_only
          --uvm-pool-coordinator-trace-file "$run_dir/vllm_uvm_pool_coordinator_l7.jsonl"
          --uvm-pool-coordinator-global-bytes-per-step 536870912
          --uvm-pool-coordinator-kv-bytes-per-step 536870912
          --uvm-pool-coordinator-weight-bytes-per-step 0
          --uvm-pool-coordinator-scratch-bytes-per-step 0
          --uvm-pool-coordinator-priority kv,scratch,weights
        )
      fi
      ;;
    *)
      echo "unknown stage mode: $stage_mode" >&2
      exit 2
      ;;
  esac

  VLLM_UVM_POLICY_ENABLE="$POLICY_ENABLE" "$SCRIPT_DIR/run_kv_fault_ratio.sh" "${args[@]}"
}

run_variant() {
  local name="$1"
  local scratch_mode="$2"
  local stage_mode="$3"
  local rep
  for rep in $(seq 1 "$REPEAT"); do
    run_kv "$name" "$rep" "$scratch_mode" "$stage_mode"
  done
}

if [ "$RUN_STAGE_L_BASELINE" = "1" ]; then
  run_variant baseline_pure_uvm trace none
fi
if [ "$RUN_SCRATCH_BASELINE" = "1" ]; then
  run_variant scratch_direct_baseline direct none
fi
run_variant l5_metadata_scratch_direct direct l5
run_variant l6_decision_trace_scratch direct l6_trace
run_variant l6_real_prefetch_scratch direct l6_real
run_variant l7_coordinator_scratch direct l7
if [ "$RUN_L7_CROSS" = "1" ]; then
  echo "L7 cross-pool combined mode is not implemented in this direct runner yet." >&2
  echo "Use the non-combined checker for cross-pool, or add a dedicated path if needed." >&2
fi

echo "==========================================================="
echo " Building combined diagnosis summary"
echo "==========================================================="
python3 - "$RUN_ROOT" "$SCRATCH_PROFILE" "$SCRATCH_MAX_TOTAL_BYTES" "$SCRATCH_MAX_BYTES" "$SCRATCH_TARGET_PHASES" <<'PY'
import csv
import json
import re
import statistics
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
scratch_profile = sys.argv[2]
scratch_total = sys.argv[3]
scratch_max = sys.argv[4]
scratch_phases = sys.argv[5]
runs_dir = run_root / "runs"

bench_patterns = {
    "successful_requests": r"Successful requests:\s+(\d+)",
    "failed_requests": r"Failed requests:\s+(\d+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
}
fault_keys = (
    "delta_faults",
    "delta_after_dedup",
    "delta_kv_faults",
    "delta_kv_after_dedup",
)

def num(raw):
    return int(raw) if raw.isdigit() else float(raw)

def parse_bench(path):
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {}
    for key, pat in bench_patterns.items():
        m = re.findall(pat, text)
        if m:
            out[key] = num(m[-1])
    return out

def parse_text_metrics(run_dir):
    text = "\n".join(
        p.read_text(encoding="utf-8", errors="replace")
        for p in run_dir.glob("*.log")
        if p.is_file()
    )
    out = {}
    for key in fault_keys:
        m = re.findall(rf"\b{key}=([0-9]+)", text)
        if m:
            out[key] = int(m[-1])
    # Last gap-watch metrics summary in the run output.
    for key, pat in {
        "device_direct_actual_records": r"device_direct_actual_records=([0-9]+)",
        "device_direct_budget_reject_records": r"device_direct_budget_reject_records=([0-9]+)",
        "device_direct_peak_live_bytes_observed": r"device_direct_peak_live_bytes_observed=([0-9]+)",
        "gap_policy_records": r"gap_policy_records=([0-9]+)",
        "gap_policy_success": r"gap_policy_success=([0-9]+)",
        "gap_policy_fail": r"gap_policy_fail=([0-9]+)",
        "hot_gap_match_records": r"hot_gap_match_records=([0-9]+)",
    }.items():
        m = re.findall(pat, text)
        if m:
            out[key] = int(m[-1])
    return out

def load_json(path):
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def flatten(run_dir):
    variant = re.sub(r"_rep[0-9]+$", "", run_dir.name)
    rep_match = re.search(r"_rep([0-9]+)$", run_dir.name)
    rep = int(rep_match.group(1)) if rep_match else None
    bench = parse_bench(next(iter(run_dir.glob("vllm_bench_*.log")), Path()))
    metrics = parse_text_metrics(run_dir)
    gap = load_json(next(iter(run_dir.glob("gap_watch_metrics_*.json")), Path()))
    post = load_json(next(iter(run_dir.glob("auto_gap_watch_post_main_*.json")), Path()))
    selected_gap = post.get("selected_gap") if isinstance(post.get("selected_gap"), dict) else {}
    row = {
        "variant": variant,
        "rep": rep,
        "run_dir": str(run_dir),
        **bench,
        **metrics,
        "main_gap_faults": selected_gap.get("faults"),
        "main_gap_unique_pages": selected_gap.get("unique_pages"),
        "main_gap_size_mib": selected_gap.get("size_mib"),
        "dominant_phase": gap.get("dominant_phase"),
        "phase_record_ratios": json.dumps(gap.get("phase_record_ratios") or {}, sort_keys=True),
        "placement_backend_counts": json.dumps(gap.get("placement_backend_counts") or {}, sort_keys=True),
        "device_direct_backend_counts": json.dumps(gap.get("device_direct_backend_counts") or {}, sort_keys=True),
    }
    return row

rows = [flatten(p) for p in sorted(runs_dir.glob("*_rep*")) if p.is_dir()]

baseline = next((r for r in rows if r["variant"] == "baseline_pure_uvm"), None)
scratch_base = next((r for r in rows if r["variant"] == "scratch_direct_baseline"), None)

def pct(new, old):
    if new is None or old in (None, 0):
        return None
    return (float(new) - float(old)) / float(old) * 100.0

for r in rows:
    base = scratch_base or baseline
    if base:
        r["output_vs_reference_pct"] = pct(r.get("output_tok_s"), base.get("output_tok_s"))
        r["tpot_vs_reference_pct"] = pct(r.get("mean_tpot_ms"), base.get("mean_tpot_ms"))
        r["gap_faults_vs_reference_pct"] = pct(r.get("main_gap_faults"), base.get("main_gap_faults"))
    if baseline:
        r["output_vs_pure_uvm_pct"] = pct(r.get("output_tok_s"), baseline.get("output_tok_s"))
        r["tpot_vs_pure_uvm_pct"] = pct(r.get("mean_tpot_ms"), baseline.get("mean_tpot_ms"))
        r["gap_faults_vs_pure_uvm_pct"] = pct(r.get("main_gap_faults"), baseline.get("main_gap_faults"))

fieldnames = sorted({k for r in rows for k in r})
summary_csv = run_root / "summary.csv"
with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

def median_for(variant, key):
    vals = [r.get(key) for r in rows if r.get("variant") == variant and isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None

variants = []
for r in rows:
    if r["variant"] not in variants:
        variants.append(r["variant"])

def fmt(v, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return f"{v}{suffix}"
    return f"{v:.2f}{suffix}"

lines = [
    "# Stage L With Scratch-Direct Diagnosis",
    "",
    f"- run_root: `{run_root}`",
    f"- scratch_profile: `{scratch_profile}`",
    f"- scratch_total_bytes: `{scratch_total}`",
    f"- scratch_max_bytes: `{scratch_max}`",
    f"- scratch_target_phases: `{scratch_phases}`",
    f"- runs: {len(rows)}",
    "",
    "## Median By Variant",
    "",
    "| variant | output_tok_s | mean_tpot_ms | mean_ttft_ms | gap_faults | kv_faults | kv_dedup_faults | direct_records | budget_rejects | output_vs_scratch_base | TPOT_vs_scratch_base |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for v in variants:
    lines.append(
        "| "
        f"{v} | "
        f"{fmt(median_for(v,'output_tok_s'))} | "
        f"{fmt(median_for(v,'mean_tpot_ms'))} | "
        f"{fmt(median_for(v,'mean_ttft_ms'))} | "
        f"{fmt(median_for(v,'main_gap_faults'))} | "
        f"{fmt(median_for(v,'delta_kv_faults'))} | "
        f"{fmt(median_for(v,'delta_kv_after_dedup'))} | "
        f"{fmt(median_for(v,'device_direct_actual_records'))} | "
        f"{fmt(median_for(v,'device_direct_budget_reject_records'))} | "
        f"{fmt(median_for(v,'output_vs_reference_pct'), '%')} | "
        f"{fmt(median_for(v,'tpot_vs_reference_pct'), '%')} |"
    )

lines += [
    "",
    "## Interpretation",
    "",
    "- `baseline_pure_uvm` keeps scratch direct off and Stage L/K/V coordinator paths off.",
    "- `scratch_direct_baseline` enables the selected scratch direct profile without Stage L KV actions.",
    "- Stage L variants after that show whether KV metadata/offload/coordinator still add value after scratch faults are reduced.",
    "- Compare Stage L rows primarily against `scratch_direct_baseline`; compare against `baseline_pure_uvm` only to estimate total scratch-direct uplift.",
    "",
    f"- CSV: `{summary_csv}`",
]
summary_md = run_root / "summary.md"
summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"summary_csv={summary_csv}")
print(f"summary_md={summary_md}")
PY

echo "==========================================================="
echo " Done"
echo " Summary CSV: $RUN_ROOT/summary.csv"
echo " Summary report: $RUN_ROOT/summary.md"
echo " Raw runs: $RUN_ROOT/runs"
echo "==========================================================="
