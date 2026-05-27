#!/usr/bin/env bash
# Stage N P7 stability/performance A/B harness.
#
# This script prepares and runs longer P7 experiments. It compares native UVM,
# Stage N trace-only, Stage N automatic soft reclaim, and optional tuning sweeps.
# Use DRY_RUN=1 to emit commands without running the expensive workloads.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_n_p7_ab_sweep_$(date -u +%Y%m%d_%H%M%S)}"
P7_VARIANTS="${P7_VARIANTS:-native_uvm,stage_n_trace,stage_n_soft}"
REPEATS="${REPEATS:-1}"
DRY_RUN="${DRY_RUN:-0}"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
REQUEST_RATE="${REQUEST_RATE:-1}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-4}"
SERVER_GPU_MEMORY_UTILIZATION="${SERVER_GPU_MEMORY_UTILIZATION:-0.70}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
PORT_BASE="${PORT_BASE:-8100}"
RESET_COUNTERS="${RESET_COUNTERS:-1}"
UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}"

KV_BUDGET_BYTES="${KV_BUDGET_BYTES:-10737418240}"
TRACE_WEIGHT_BUDGET_BYTES="${TRACE_WEIGHT_BUDGET_BYTES:-10737418240}"
SOFT_WEIGHT_BUDGET_BYTES="${SOFT_WEIGHT_BUDGET_BYTES:-21474836480}"
TRACE_SCRATCH_BUDGET_BYTES="${TRACE_SCRATCH_BUDGET_BYTES:-104857600}"
SOFT_SCRATCH_BUDGET_BYTES="${SOFT_SCRATCH_BUDGET_BYTES:-16777216}"

UVM_PHASE_N_AUTO_RECLAIM_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_EVERY:-64}"
UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="${UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY:-256}"
UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="${UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS:-1}"
UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS:-4096}"
UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="${UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS:-256}"

P7_SCRATCH_SWEEP_BYTES_CSV="${P7_SCRATCH_SWEEP_BYTES_CSV:-16777216,67108864,104857600}"
P7_COOLDOWN_SWEEP_EPOCHS_CSV="${P7_COOLDOWN_SWEEP_EPOCHS_CSV:-0,1024,4096}"
P7_MAX_VICTIMS_SWEEP_CSV="${P7_MAX_VICTIMS_SWEEP_CSV:-1,2}"

MAX_TTFT_REGRESSION_PCT="${MAX_TTFT_REGRESSION_PCT:-50}"
MAX_TPOT_REGRESSION_PCT="${MAX_TPOT_REGRESSION_PCT:-50}"

mkdir -p "$RUN_ROOT"
METADATA="$RUN_ROOT/p7_runs.tsv"
SUMMARY_TSV="$RUN_ROOT/p7_ab_summary.tsv"
SUMMARY_MD="$RUN_ROOT/p7_ab_summary.md"
COMMAND_FILE="$RUN_ROOT/command_stage_n_p7_ab_sweep.sh"

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  local old_ifs="$IFS"

  IFS=,
  read -r -a out_ref <<< "$csv"
  IFS="$old_ifs"
}

write_command_file() {
  {
    printf 'cd %q\n' "$SCRIPT_DIR"
    printf 'RUN_ROOT=%q P7_VARIANTS=%q REPEATS=%q PROMPTS=%q REQUEST_RATE=%q OUTPUT_LEN=%q SERVER_MAX_NUM_SEQS=%q PORT_BASE=%q RESET_COUNTERS=%q UVM_PHASE_F_ROUTE_LOG_DELAY_MS=%q ./run_stage_n_p7_ab_sweep.sh\n' \
      "$RUN_ROOT" "$P7_VARIANTS" "$REPEATS" "$PROMPTS" "$REQUEST_RATE" "$OUTPUT_LEN" \
      "$SERVER_MAX_NUM_SEQS" "$PORT_BASE" "$RESET_COUNTERS" "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS"
  } > "$COMMAND_FILE"
  chmod +x "$COMMAND_FILE"
}

record_run() {
  local variant="$1"
  local repeat="$2"
  local status="$3"
  local kind="$4"
  local require_auto_reclaim="$5"
  local run_dir="$6"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$variant" "$repeat" "$status" "$kind" "$require_auto_reclaim" "$run_dir" >> "$METADATA"
}

run_or_print() {
  local log_file="$1"
  shift

  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN: ' | tee -a "$log_file"
    printf '%q ' "$@" | tee -a "$log_file"
    printf '\n' | tee -a "$log_file"
    return 125
  fi

  "$@" > "$log_file" 2>&1
}

run_stage_n_variant() {
  local variant="$1"
  local repeat="$2"
  local port="$3"
  local run_dir="$4"
  local budget_mode="$5"
  local require_auto_reclaim="$6"
  local weight_budget="$7"
  local scratch_budget="$8"
  local max_victims="$9"
  local cooldown_epochs="${10}"
  local log_file="$run_dir/driver.log"
  local status=0

  mkdir -p "$run_dir"
  {
    printf 'cd %q\n' "$SCRIPT_DIR"
    printf 'env MODEL=%q PROMPTS=%q REQUEST_RATE=%q OUTPUT_LEN=%q MAX_MODEL_LEN=%q SERVER_MAX_NUM_SEQS=%q SERVER_GPU_MEMORY_UTILIZATION=%q STARTUP_TIMEOUT=%q PORT=%q RESET_COUNTERS=%q UVM_PHASE_F_ROUTE_LOG_DELAY_MS=%q KV_BUDGET_BYTES=%q WEIGHT_BUDGET_BYTES=%q SCRATCH_TEMP_BUDGET_BYTES=%q DRIVER_POOL_BUDGET_MODE=%q REQUIRE_PHASE_N_AUTO_HOOK=%q REQUIRE_PHASE_N_AUTO_RECLAIM=%q UVM_PHASE_N_AUTO_RECLAIM_EVERY=%q UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY=%q UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS=%q UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS=%q UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS=%q RUN_ROOT=%q ./run_stage_n_tag_only_three_pool_smoke.sh\n' \
      "$MODEL" "$PROMPTS" "$REQUEST_RATE" "$OUTPUT_LEN" "$MAX_MODEL_LEN" "$SERVER_MAX_NUM_SEQS" \
      "$SERVER_GPU_MEMORY_UTILIZATION" "$STARTUP_TIMEOUT" "$port" "$RESET_COUNTERS" "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
      "$KV_BUDGET_BYTES" "$weight_budget" "$scratch_budget" "$budget_mode" "$require_auto_reclaim" "$require_auto_reclaim" \
      "$UVM_PHASE_N_AUTO_RECLAIM_EVERY" "$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY" "$max_victims" "$cooldown_epochs" \
      "$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS" "$run_dir"
  } > "$run_dir/command.sh"
  chmod +x "$run_dir/command.sh"

  set +e
  run_or_print "$log_file" \
    env \
      MODEL="$MODEL" \
      PROMPTS="$PROMPTS" \
      REQUEST_RATE="$REQUEST_RATE" \
      OUTPUT_LEN="$OUTPUT_LEN" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" \
      SERVER_MAX_NUM_SEQS="$SERVER_MAX_NUM_SEQS" \
      SERVER_GPU_MEMORY_UTILIZATION="$SERVER_GPU_MEMORY_UTILIZATION" \
      STARTUP_TIMEOUT="$STARTUP_TIMEOUT" \
      PORT="$port" \
      RESET_COUNTERS="$RESET_COUNTERS" \
      UVM_PHASE_F_ROUTE_LOG_DELAY_MS="$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
      KV_BUDGET_BYTES="$KV_BUDGET_BYTES" \
      WEIGHT_BUDGET_BYTES="$weight_budget" \
      SCRATCH_TEMP_BUDGET_BYTES="$scratch_budget" \
      DRIVER_POOL_BUDGET_MODE="$budget_mode" \
      REQUIRE_PHASE_N_AUTO_HOOK="$require_auto_reclaim" \
      REQUIRE_PHASE_N_AUTO_RECLAIM="$require_auto_reclaim" \
      UVM_PHASE_N_AUTO_RECLAIM_EVERY="$UVM_PHASE_N_AUTO_RECLAIM_EVERY" \
      UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY="$UVM_PHASE_N_AUTO_RECLAIM_SCAN_EVERY" \
      UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS="$max_victims" \
      UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS="$cooldown_epochs" \
      UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS="$UVM_PHASE_N_AUTO_RECLAIM_FAILURE_BACKOFF_EPOCHS" \
      RUN_ROOT="$run_dir" \
      ./run_stage_n_tag_only_three_pool_smoke.sh
  status=$?
  set -e

  record_run "$variant" "$repeat" "$status" "stage_n" "$require_auto_reclaim" "$run_dir"
}

run_native_uvm_variant() {
  local variant="$1"
  local repeat="$2"
  local port="$3"
  local run_dir="$4"
  local log_file="$run_dir/driver.log"
  local status=0

  mkdir -p "$run_dir"
  {
    printf 'cd %q\n' "$SCRIPT_DIR"
    printf './run_kv_fault_ratio.sh --model %q --mode trace --startup-timeout %q --trace-log %q --address-trace-log %q --server-log %q --bench-log %q --port %q --server-max-num-seqs %q --server-gpu-memory-utilization %q --max-model-len %q --prompts %q --request-rate %q --output-len %q --uvm-policy-enable 0 --uvm-weight-map-enable 0 --uvm-kv-runtime-enable 0 --uvm-residency-enable 0 --uvm-pool-registry-enable 0 --uvm-driver-scratch-tag-enable 0 --uvm-phase-n-auto-reclaim-enable 0 --uvm-phase-n-auto-reclaim-execute 0 --uvm-phase-f-route-log-delay-ms %q %s\n' \
      "$MODEL" "$STARTUP_TIMEOUT" "$run_dir/uvm_fault_stats.log" "$run_dir/uvm_fault_addrs.log" \
      "$run_dir/vllm_server.log" "$run_dir/vllm_bench.log" "$port" "$SERVER_MAX_NUM_SEQS" \
      "$SERVER_GPU_MEMORY_UTILIZATION" "$MAX_MODEL_LEN" "$PROMPTS" "$REQUEST_RATE" "$OUTPUT_LEN" \
      "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
      "$([ "$RESET_COUNTERS" = "1" ] && printf -- "--reset-counters" || printf -- "--no-reset-counters")"
  } > "$run_dir/command.sh"
  chmod +x "$run_dir/command.sh"

  set +e
  run_or_print "$log_file" \
    ./run_kv_fault_ratio.sh \
      --model "$MODEL" \
      --mode trace \
      --startup-timeout "$STARTUP_TIMEOUT" \
      --trace-log "$run_dir/uvm_fault_stats.log" \
      --address-trace-log "$run_dir/uvm_fault_addrs.log" \
      --server-log "$run_dir/vllm_server.log" \
      --bench-log "$run_dir/vllm_bench.log" \
      --port "$port" \
      --server-max-num-seqs "$SERVER_MAX_NUM_SEQS" \
      --server-gpu-memory-utilization "$SERVER_GPU_MEMORY_UTILIZATION" \
      --max-model-len "$MAX_MODEL_LEN" \
      --prompts "$PROMPTS" \
      --request-rate "$REQUEST_RATE" \
      --output-len "$OUTPUT_LEN" \
      --uvm-policy-enable 0 \
      --uvm-weight-map-enable 0 \
      --uvm-kv-runtime-enable 0 \
      --uvm-residency-enable 0 \
      --uvm-pool-registry-enable 0 \
      --uvm-driver-scratch-tag-enable 0 \
      --uvm-phase-n-auto-reclaim-enable 0 \
      --uvm-phase-n-auto-reclaim-execute 0 \
      --uvm-phase-f-route-log-delay-ms "$UVM_PHASE_F_ROUTE_LOG_DELAY_MS" \
      "$([ "$RESET_COUNTERS" = "1" ] && printf -- "--reset-counters" || printf -- "--no-reset-counters")"
  status=$?
  set -e

  record_run "$variant" "$repeat" "$status" "native_uvm" "0" "$run_dir"
}

write_p7_summary() {
  python3 - "$METADATA" "$SUMMARY_TSV" "$SUMMARY_MD" "$MAX_TTFT_REGRESSION_PCT" "$MAX_TPOT_REGRESSION_PCT" <<'PY'
import os
import re
import statistics
import sys

metadata_path, tsv_path, md_path = sys.argv[1:4]
max_ttft = float(sys.argv[4])
max_tpot = float(sys.argv[5])

bench_patterns = {
    "successful_requests": re.compile(r"Successful requests:\s+([0-9.]+)"),
    "failed_requests": re.compile(r"Failed requests:\s+([0-9.]+)"),
    "duration_s": re.compile(r"Benchmark duration \(s\):\s+([0-9.]+)"),
    "output_tok_s": re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)"),
    "mean_ttft_ms": re.compile(r"Mean TTFT \(ms\):\s+([0-9.]+)"),
    "mean_tpot_ms": re.compile(r"Mean TPOT \(ms\):\s+([0-9.]+)"),
}

summary_keys = (
    "phase_n_auto_hook_records",
    "phase_n_auto_hook_would_reclaim_records",
    "phase_n_auto_hook_locked_context_records",
    "phase_n_auto_hook_outer_records",
    "phase_n_auto_hook_pool_records",
    "phase_n_auto_hook_no_region_records",
    "phase_n_auto_hook_execute_records",
    "phase_n_auto_hook_success_records",
    "phase_n_auto_hook_reclaimed_records",
    "driver_pool_state_failures",
    "driver_pool_state_all_pools_present",
    "driver_pool_state_all_required_fields_present",
    "allocator_driver_pool_state_all_status_ok",
    "scratch_driver_over_budget_records",
)

def read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""

def parse_key_value(text: str, key: str, default: str = "0") -> str:
    pattern = re.compile(rf"^{re.escape(key)}=(.*)$", re.MULTILINE)
    matches = pattern.findall(text)
    return matches[-1].strip() if matches else default

def parse_bench(text: str) -> dict[str, str]:
    out = {}
    for key, pattern in bench_patterns.items():
        matches = pattern.findall(text)
        out[key] = matches[-1] if matches else ""
    return out

def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

rows = []
with open(metadata_path, encoding="utf-8") as handle:
    header = next(handle, "")
    for raw in handle:
        variant, repeat, status, kind, require_auto, run_dir = raw.rstrip("\n").split("\t")
        stage_summary = os.path.join(run_dir, "stage_n_acceptance_summary.txt")
        bench_log = os.path.join(run_dir, "vllm_bench.log")
        text = read_text(stage_summary)
        bench_text = text if text else read_text(bench_log)
        bench = parse_bench(bench_text)
        row = {
            "variant": variant,
            "repeat": repeat,
            "status": status,
            "kind": kind,
            "require_auto_reclaim": require_auto,
            "run_root": run_dir,
            **bench,
        }
        for key in summary_keys:
            row[key] = parse_key_value(text, key, "0")
        row["verdict"] = "PASS" if re.search(r"^PASS:", text, re.MULTILINE) else ("DRY_RUN" if status == "125" else "")
        rows.append(row)

baseline = None
for preferred in ("native_uvm", "stage_n_trace"):
    candidates = [r for r in rows if r["variant"] == preferred and as_float(r.get("mean_tpot_ms"))]
    if candidates:
        baseline = candidates[0]
        break

baseline_ttft = as_float(baseline["mean_ttft_ms"]) if baseline else None
baseline_tpot = as_float(baseline["mean_tpot_ms"]) if baseline else None

for row in rows:
    ttft = as_float(row.get("mean_ttft_ms"))
    tpot = as_float(row.get("mean_tpot_ms"))
    row["ttft_delta_pct"] = ""
    row["tpot_delta_pct"] = ""
    if baseline_ttft and ttft is not None:
        row["ttft_delta_pct"] = f"{((ttft - baseline_ttft) / baseline_ttft) * 100:.2f}"
    if baseline_tpot and tpot is not None:
        row["tpot_delta_pct"] = f"{((tpot - baseline_tpot) / baseline_tpot) * 100:.2f}"

    failed = as_float(row.get("failed_requests"))
    ttft_delta = as_float(row["ttft_delta_pct"])
    tpot_delta = as_float(row["tpot_delta_pct"])
    ok = row["status"] == "0" and (failed is None or failed == 0)
    if row["require_auto_reclaim"] == "1":
        ok = ok and int(row["phase_n_auto_hook_would_reclaim_records"] or 0) > 0
        ok = ok and int(row["phase_n_auto_hook_execute_records"] or 0) > 0
        ok = ok and int(row["phase_n_auto_hook_success_records"] or 0) > 0
        ok = ok and int(row["phase_n_auto_hook_reclaimed_records"] or 0) > 0
    if ttft_delta is not None:
        ok = ok and ttft_delta <= max_ttft
    if tpot_delta is not None:
        ok = ok and tpot_delta <= max_tpot
    row["p7_verdict"] = "PASS" if ok else ("DRY_RUN" if row["status"] == "125" else "WARN")

fields = [
    "variant", "repeat", "p7_verdict", "status", "successful_requests", "failed_requests",
    "duration_s", "output_tok_s", "mean_ttft_ms", "mean_tpot_ms", "ttft_delta_pct",
    "tpot_delta_pct", "phase_n_auto_hook_records", "phase_n_auto_hook_would_reclaim_records",
    "phase_n_auto_hook_execute_records", "phase_n_auto_hook_success_records",
    "phase_n_auto_hook_reclaimed_records", "driver_pool_state_failures",
    "allocator_driver_pool_state_all_status_ok", "scratch_driver_over_budget_records", "run_root",
]

with open(tsv_path, "w", encoding="utf-8") as out:
    out.write("\t".join(fields) + "\n")
    for row in rows:
        out.write("\t".join(str(row.get(field, "")) for field in fields) + "\n")

variant_groups = {}
for row in rows:
    variant_groups.setdefault(row["variant"], []).append(row)

def mean_for(items, field):
    values = [as_float(item.get(field)) for item in items]
    values = [value for value in values if value is not None]
    return f"{statistics.mean(values):.2f}" if values else ""

with open(md_path, "w", encoding="utf-8") as out:
    out.write("# Stage N P7 A/B Summary\n\n")
    out.write(f"Baseline: {baseline['variant'] if baseline else 'missing'}\n\n")
    out.write("| Variant | Runs | Verdicts | Mean TTFT ms | Mean TPOT ms | Mean output tok/s | Auto exec | Auto reclaimed | Failed req |\n")
    out.write("|---|---:|---|---:|---:|---:|---:|---:|---:|\n")
    for variant, items in variant_groups.items():
        verdicts = ",".join(sorted(set(item["p7_verdict"] for item in items)))
        auto_exec = sum(int(item.get("phase_n_auto_hook_execute_records") or 0) for item in items)
        auto_reclaimed = sum(int(item.get("phase_n_auto_hook_reclaimed_records") or 0) for item in items)
        failed = sum(int(float(item.get("failed_requests") or 0)) for item in items if item.get("failed_requests"))
        out.write(
            f"| {variant} | {len(items)} | {verdicts} | {mean_for(items, 'mean_ttft_ms')} | "
            f"{mean_for(items, 'mean_tpot_ms')} | {mean_for(items, 'output_tok_s')} | "
            f"{auto_exec} | {auto_reclaimed} | {failed} |\n"
        )
    out.write("\nDetailed TSV: `p7_ab_summary.tsv`\n")
PY
}

write_command_file
printf 'variant\trepeat\tstatus\tkind\trequire_auto_reclaim\trun_root\n' > "$METADATA"

declare -a variants
csv_to_array "$P7_VARIANTS" variants

run_index=0
for repeat in $(seq 1 "$REPEATS"); do
  for variant in "${variants[@]}"; do
    variant="${variant// /}"
    [ -n "$variant" ] || continue
    case "$variant" in
      native_uvm)
        run_dir="$RUN_ROOT/${repeat}_${variant}"
        run_native_uvm_variant "$variant" "$repeat" "$((PORT_BASE + run_index))" "$run_dir"
        run_index=$((run_index + 1))
        ;;
      stage_n_trace)
        run_dir="$RUN_ROOT/${repeat}_${variant}"
        run_stage_n_variant "$variant" "$repeat" "$((PORT_BASE + run_index))" "$run_dir" \
          trace_only 0 "$TRACE_WEIGHT_BUDGET_BYTES" "$TRACE_SCRATCH_BUDGET_BYTES" 2 256
        run_index=$((run_index + 1))
        ;;
      stage_n_soft)
        run_dir="$RUN_ROOT/${repeat}_${variant}"
        run_stage_n_variant "$variant" "$repeat" "$((PORT_BASE + run_index))" "$run_dir" \
          soft 1 "$SOFT_WEIGHT_BUDGET_BYTES" "$SOFT_SCRATCH_BUDGET_BYTES" \
          "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS" "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
        run_index=$((run_index + 1))
        ;;
      stage_n_enforce_probe)
        run_dir="$RUN_ROOT/${repeat}_${variant}"
        run_stage_n_variant "$variant" "$repeat" "$((PORT_BASE + run_index))" "$run_dir" \
          enforce 1 "$SOFT_WEIGHT_BUDGET_BYTES" "$SOFT_SCRATCH_BUDGET_BYTES" \
          "$UVM_PHASE_N_AUTO_RECLAIM_MAX_VICTIMS" "$UVM_PHASE_N_AUTO_RECLAIM_COOLDOWN_EPOCHS"
        run_index=$((run_index + 1))
        ;;
      stage_n_soft_sweep)
        declare -a scratch_values cooldown_values victim_values
        csv_to_array "$P7_SCRATCH_SWEEP_BYTES_CSV" scratch_values
        csv_to_array "$P7_COOLDOWN_SWEEP_EPOCHS_CSV" cooldown_values
        csv_to_array "$P7_MAX_VICTIMS_SWEEP_CSV" victim_values
        for scratch_budget in "${scratch_values[@]}"; do
          for cooldown_epochs in "${cooldown_values[@]}"; do
            for max_victims in "${victim_values[@]}"; do
              sweep_variant="stage_n_soft_s${scratch_budget}_c${cooldown_epochs}_v${max_victims}"
              run_dir="$RUN_ROOT/${repeat}_${sweep_variant}"
              run_stage_n_variant "$sweep_variant" "$repeat" "$((PORT_BASE + run_index))" "$run_dir" \
                soft 1 "$SOFT_WEIGHT_BUDGET_BYTES" "$scratch_budget" "$max_victims" "$cooldown_epochs"
              run_index=$((run_index + 1))
            done
          done
        done
        ;;
      *)
        echo "Unknown P7 variant: $variant" >&2
        exit 2
        ;;
    esac
  done
done

write_p7_summary

echo "Stage N P7 A/B sweep prepared."
echo "RUN_ROOT=$RUN_ROOT"
echo "metadata=$METADATA"
echo "summary_tsv=$SUMMARY_TSV"
echo "summary_md=$SUMMARY_MD"
echo "command_file=$COMMAND_FILE"
cat "$SUMMARY_MD"
