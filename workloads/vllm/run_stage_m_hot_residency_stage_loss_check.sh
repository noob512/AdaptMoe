#!/usr/bin/env bash
# Diagnose whether init-prefetched decode-hot expert pages are later migrated
# back to CPU. This script runs decode-only oracle static hotset experiments
# across output lengths and compares hot-prefetch-only vs hot+cold-offload.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-log/stage_m_hot_residency_stage_loss_check_run}"
OUTPUT_LENS="${OUTPUT_LENS:-1 4 8 16}"
COLD_MODES="${COLD_MODES:-hot_only cold_offload}"

PROMPTS="${PROMPTS:-1}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SERVER_MAX_NUM_SEQS="${SERVER_MAX_NUM_SEQS:-1}"
DECODE_MAX_NUM_TOKENS="${DECODE_MAX_NUM_TOKENS:-$SERVER_MAX_NUM_SEQS}"
WEIGHT_GPU_BYTES="${WEIGHT_GPU_BYTES:-25769803776}"
KV_GPU_BYTES="${KV_GPU_BYTES:-2147483648}"
WEIGHT_PLAN_LAYERS="${WEIGHT_PLAN_LAYERS:-48}"
WEIGHT_PLAN_EXPERTS="${WEIGHT_PLAN_EXPERTS:-128}"
WEIGHT_EXECUTOR="${WEIGHT_EXECUTOR:-driver_batch}"
INIT_WEIGHT_HOT_PREFETCH_DEVICE="${INIT_WEIGHT_HOT_PREFETCH_DEVICE:-0}"
REAL_MAX_SNAPSHOTS="${REAL_MAX_SNAPSHOTS:-256}"
REAL_ON_REGISTER_EVERY="${REAL_ON_REGISTER_EVERY:-4096}"
REAL_ON_ACTION_EVERY="${REAL_ON_ACTION_EVERY:-1}"

mkdir -p "$RUN_ROOT"

echo "==========================================================="
echo "Stage M Hot Residency Stage-Loss Check"
echo "run_root=$RUN_ROOT"
echo "output_lens=$OUTPUT_LENS"
echo "cold_modes=$COLD_MODES"
echo "decode_max_num_tokens=$DECODE_MAX_NUM_TOKENS"
echo "==========================================================="

for output_len in $OUTPUT_LENS; do
  for cold_mode in $COLD_MODES; do
    case "$cold_mode" in
      hot_only)
        init_weight_max_bytes=1
        ;;
      cold_offload)
        init_weight_max_bytes=0
        ;;
      *)
        echo "Unknown cold mode: $cold_mode" >&2
        exit 2
        ;;
    esac

    case_root="$RUN_ROOT/output${output_len}_${cold_mode}"
    echo "==========================================================="
    echo "Run output_len=$output_len cold_mode=$cold_mode"
    echo "case_root=$case_root"
    echo "==========================================================="
    TRACE_SOURCE_ALLOW_PHASE_F_FAIL=1 \
    TRACE_SOURCE_INIT_OFFLOAD_ENABLE=0 \
    VLLM_UVM_PHASE_F_IOCTL_MODE="${VLLM_UVM_PHASE_F_IOCTL_MODE:-policy}" \
    UVM_PHASE_F_ROUTE_LOG_DELAY_MS="${UVM_PHASE_F_ROUTE_LOG_DELAY_MS:-0}" \
    RUN_ROOT="$case_root" \
    PROMPTS="$PROMPTS" \
    OUTPUT_LEN="$output_len" \
    REQUEST_RATE="$REQUEST_RATE" \
    SERVER_MAX_NUM_SEQS="$SERVER_MAX_NUM_SEQS" \
    DECODE_MAX_NUM_TOKENS="$DECODE_MAX_NUM_TOKENS" \
    WEIGHT_GPU_BYTES="$WEIGHT_GPU_BYTES" \
    KV_GPU_BYTES="$KV_GPU_BYTES" \
    WEIGHT_PLAN_LAYERS="$WEIGHT_PLAN_LAYERS" \
    WEIGHT_PLAN_EXPERTS="$WEIGHT_PLAN_EXPERTS" \
    WEIGHT_EXECUTOR="$WEIGHT_EXECUTOR" \
    INIT_WEIGHT_HOT_PREFETCH_ENABLE=1 \
    INIT_WEIGHT_HOT_PREFETCH_SYNC=1 \
    INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE=1 \
    INIT_WEIGHT_HOT_PREFETCH_DEVICE="$INIT_WEIGHT_HOT_PREFETCH_DEVICE" \
    INIT_WEIGHT_MAX_BYTES="$init_weight_max_bytes" \
    INIT_WEIGHT_MAX_EXPERTS_PER_LAYER=0 \
    HOT_RESIDENCY_STAGE_TRACE_ENABLE=1 \
    HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS="$DECODE_MAX_NUM_TOKENS" \
    HOT_RESIDENCY_STAGE_DEVICE="$INIT_WEIGHT_HOT_PREFETCH_DEVICE" \
    REAL_MAX_SNAPSHOTS="$REAL_MAX_SNAPSHOTS" \
    REAL_ON_REGISTER_EVERY="$REAL_ON_REGISTER_EVERY" \
    REAL_ON_ACTION_EVERY="$REAL_ON_ACTION_EVERY" \
    "$SCRIPT_DIR/run_stage_m_oracle_static_hotset_decode_only_ab.sh"
  done
done

python3 - "$RUN_ROOT" "$OUTPUT_LENS" "$COLD_MODES" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output_lens = sys.argv[2].split()
cold_modes = sys.argv[3].split()

def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def read_csv_row(path: Path, case: str) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            if row.get("case") == case:
                return row
    return {}

def num(row: dict, key: str):
    value = row.get(key)
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None

def marker_stats(init_trace: Path) -> dict[str, dict[str, float | int]]:
    markers: dict[str, dict[str, float | int]] = {}

    def add(marker: str, queried: int, gpu: int, cpu: int, ranges: int) -> None:
        item = markers.setdefault(
            marker,
            {
                "records": 0,
                "ranges": 0,
                "queried_bytes": 0,
                "gpu_bytes": 0,
                "cpu_bytes": 0,
            },
        )
        item["records"] = int(item["records"]) + 1
        item["ranges"] = int(item["ranges"]) + ranges
        item["queried_bytes"] = int(item["queried_bytes"]) + queried
        item["gpu_bytes"] = int(item["gpu_bytes"]) + gpu
        item["cpu_bytes"] = int(item["cpu_bytes"]) + cpu

    if init_trace.exists():
        with init_trace.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = rec.get("action")
                if action == "init_weight_hot_prefetch_real_query":
                    add(
                        "after_hot_prefetch",
                        int(rec.get("queried_bytes") or 0),
                        int(rec.get("gpu_bytes") or 0),
                        int(rec.get("cpu_bytes") or 0),
                        int(rec.get("query_ranges") or 0),
                    )
                elif action == "init_hot_residency_stage_query":
                    marker = str(rec.get("marker") or "unknown")
                    add(
                        marker,
                        int(rec.get("hot_residency_queried_bytes") or 0),
                        int(rec.get("hot_residency_gpu_bytes") or 0),
                        int(rec.get("hot_residency_cpu_bytes") or 0),
                        int(rec.get("hot_residency_query_ranges") or 0),
                    )

    for item in markers.values():
        queried = int(item["queried_bytes"])
        item["gpu_ratio"] = (int(item["gpu_bytes"]) / queried) if queried else None
    return markers

rows = []
for output_len in output_lens:
    for cold_mode in cold_modes:
        case_root = root / f"output{output_len}_{cold_mode}"
        summary_csv = case_root / "oracle_static_hotset_decode_only_summary.csv"
        row = read_csv_row(summary_csv, "oracle_static_hotset_decode_only")
        case_run = case_root / "oracle_static_hotset_decode_only"
        init_trace = (
            case_run
            / "routing_phase_f_full_impl"
            / "runs"
            / "full_impl_enforce_rep1"
            / "vllm_uvm_init_weight_offload_full_impl_enforce_rep1.jsonl"
        )
        markers = marker_stats(init_trace)
        residency = read_json(case_run / "real_residency_summary.json")
        weights = (residency.get("by_pool") or {}).get("weights") or {}
        rows.append(
            {
                "output_len": output_len,
                "cold_mode": cold_mode,
                "passed": row.get("phase_passed"),
                "tok_s": row.get("tok_s"),
                "tpot_ms": row.get("tpot_ms"),
                "workload_faults_dedup": row.get("workload_faults_dedup"),
                "plan_hot": row.get("prefetch_plan_records"),
                "plan_cold": row.get("offload_plan_records"),
                "hot_gib": (
                    num(row, "init_hot_prefetch_bytes") / (1024**3)
                    if num(row, "init_hot_prefetch_bytes") is not None
                    else None
                ),
                "summary_after_hot_gpu_ratio": row.get("init_hot_real_gpu_ratio"),
                "after_hot_prefetch_gpu_ratio": (
                    markers.get("after_hot_prefetch") or {}
                ).get("gpu_ratio"),
                "after_cold_offload_gpu_ratio": (
                    markers.get("after_cold_offload") or {}
                ).get("gpu_ratio"),
                "before_first_decode_gpu_ratio": (
                    markers.get("before_first_decode") or {}
                ).get("gpu_ratio"),
                "final_weight_gpu_gib": weights.get("gpu_gib"),
                "final_weight_cpu_gib": weights.get("cpu_gib"),
                "init_trace": str(init_trace),
            }
        )

csv_path = root / "hot_residency_stage_loss_summary.csv"
fields = [
    "output_len",
    "cold_mode",
    "passed",
    "tok_s",
    "tpot_ms",
    "workload_faults_dedup",
    "plan_hot",
    "plan_cold",
    "hot_gib",
    "summary_after_hot_gpu_ratio",
    "after_hot_prefetch_gpu_ratio",
    "after_cold_offload_gpu_ratio",
    "before_first_decode_gpu_ratio",
    "final_weight_gpu_gib",
    "final_weight_cpu_gib",
    "init_trace",
]
with csv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

md_path = root / "hot_residency_stage_loss_summary.md"
lines = [
    "# Stage M Hot Residency Stage-Loss Check",
    "",
    f"- summary_csv: `{csv_path}`",
    "",
    "| output_len | mode | TPOT ms | faults | hot experts | hot GiB | after hot | after cold | before decode | final GPU GiB | final CPU GiB |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    lines.append(
        "| {output_len} | {cold_mode} | {tpot} | {faults} | {hot} | {hot_gib} | {after_hot} | {after_cold} | {before_decode} | {gpu} | {cpu} |".format(
            output_len=row["output_len"],
            cold_mode=row["cold_mode"],
            tpot=fmt(row["tpot_ms"]),
            faults=fmt(row["workload_faults_dedup"]),
            hot=fmt(row["plan_hot"]),
            hot_gib=fmt(row["hot_gib"]),
            after_hot=fmt(row["after_hot_prefetch_gpu_ratio"]),
            after_cold=fmt(row["after_cold_offload_gpu_ratio"]),
            before_decode=fmt(row["before_first_decode_gpu_ratio"]),
            gpu=fmt(row["final_weight_gpu_gib"]),
            cpu=fmt(row["final_weight_cpu_gib"]),
        )
    )
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"summary_csv={csv_path}")
print(f"summary_md={md_path}")
PY

echo "Hot residency stage-loss check finished."
echo "summary_csv: $RUN_ROOT/hot_residency_stage_loss_summary.csv"
echo "summary_md: $RUN_ROOT/hot_residency_stage_loss_summary.md"
