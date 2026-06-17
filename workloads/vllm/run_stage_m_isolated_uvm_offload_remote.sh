#!/usr/bin/env bash
# Compare native vLLM CPU offload with isolated UVM replacements for only
# the parameters selected by --cpu-offload-gb.
# Cases:
# - native_cpu_offload: pinned CPU tensor + UVA view
# - isolated_uvm_migration_offload: managed UVM + CPU prefetch only,
#   leaving later GPU reads to UVM's default demand migration/fault path
# - isolated_uvm_remote_offload: managed UVM + CPU prefetch/preferred +
#   AccessedBy(GPU), matching the remote-read path
# - isolated_uvm_hybrid_remote_offload: remote path plus ReadMostly and
#   deferred bg-promotion prepare/gate
#
# This script does not enable VLLM_USE_UVM and does not LD_PRELOAD CUDA
# allocators. All non-offloaded allocations keep the ordinary vLLM CUDA path.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
DATASET_PATH="${DATASET_PATH:-$SCRIPT_DIR/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/log/stage_m_isolated_uvm_offload_remote_$(date -u +%Y%m%d_%H%M%S)}"
PROMPTS="${PROMPTS:-5}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
REQUEST_RATE="${REQUEST_RATE:-1}"
SEED="${SEED:-42}"
PORT_BASE="${PORT_BASE:-8220}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
CHECK_INTERVAL="${CHECK_INTERVAL:-2}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
VLLM_UVM_ALLOCATOR_SO="${VLLM_UVM_ALLOCATOR_SO:-$SCRIPT_DIR/vllm/uvm_test/uvm_allocator.so}"
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

if [[ ! -s "$VLLM_UVM_ALLOCATOR_SO" ]]; then
  echo "ERROR: missing UVM pluggable allocator: $VLLM_UVM_ALLOCATOR_SO" >&2
  echo "This script will not build it automatically. Build once with:" >&2
  echo "  make -C $SCRIPT_DIR/vllm/uvm_test uvm" >&2
  exit 1
fi

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
  local case_root="$1"
  local case_name="$2"
  local backend="$3"
  {
    echo "case=$case_name"
    echo "cpu_offload_backend=$backend"
    echo "vllm_cmd=${VLLM_CMD[*]}"
    echo "model=$MODEL"
    echo "dataset_path=$DATASET_PATH"
    echo "prompts=$PROMPTS"
    echo "output_len=$OUTPUT_LEN"
    echo "request_rate=$REQUEST_RATE"
    echo "seed=$SEED"
    echo "cpu_offload_gb=$CPU_OFFLOAD_GB"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    echo "vllm_use_uvm=unset"
    echo "ld_preload=unset"
    echo "uvm_allocator_so=$VLLM_UVM_ALLOCATOR_SO"
    echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total,memory.free --format=csv,noheader || true
    echo
    modinfo nvidia_uvm 2>/dev/null | sed -n '1,20p' || true
  } > "$case_root/metadata.txt"
}

run_case() {
  local case_name="$1"
  local backend="$2"
  local port="$3"
  local case_root="$RUN_ROOT/$case_name"
  local server_log="$case_root/server.log"
  local bench_log="$case_root/bench.log"
  local run_log="$case_root/run.log"
  local manifest="$case_root/offload_manifest.jsonl"
  local status=0
  local server_pid=""

  mkdir -p "$case_root"
  : > "$server_log"
  : > "$bench_log"
  : > "$run_log"
  : > "$manifest"
  write_metadata "$case_root" "$case_name" "$backend"

  echo "==========================================================="
  echo "Run $case_name: backend=$backend"
  echo "logs=$case_root"
  echo "==========================================================="

  cleanup_gpu
  sleep 2

  (
    cd "$SCRIPT_DIR" || exit 1
    env -u VLLM_USE_UVM -u LD_PRELOAD \
      VLLM_CPU_OFFLOAD_BACKEND="$backend" \
      VLLM_CPU_OFFLOAD_UVM_SYNC=1 \
      VLLM_CPU_OFFLOAD_MANIFEST_FILE="$manifest" \
      VLLM_CPU_OFFLOAD_MANIFEST_RESET=1 \
      VLLM_UVM_LOG_FILE="$case_root/uvm_allocator.log" \
      VLLM_UVM_POLICY_ENABLE=0 \
      VLLM_UVM_POLICY_RUNTIME_ENABLE=0 \
      "${VLLM_CMD[@]}" serve "$MODEL" \
        --enforce-eager \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --cpu-offload-gb "$CPU_OFFLOAD_GB" \
        --port "$port" >"$server_log" 2>&1
  ) &
  server_pid=$!

  if wait_for_server "$port" "$server_pid" "$server_log"; then
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
        --port "$port"
    ) 2>&1 | tee -a "$bench_log"
    status=${PIPESTATUS[0]}
    set -u
  else
    status=1
  fi

  echo "$status" > "$case_root/status.txt"
  if kill -0 "$server_pid" 2>/dev/null; then
    if [[ "$backend" == "isolated_uvm_migration" || "$backend" == "isolated_uvm_remote" || "$backend" == "isolated_uvm_hybrid_remote" ]]; then
      # The current local MemPool + uvm_allocator.so path can crash during
      # Python interpreter teardown. The benchmark and manifest are already
      # written, so avoid the teardown path for this isolated smoke experiment.
      kill -KILL "$server_pid" >/dev/null 2>&1 || true
    else
      kill -TERM "$server_pid" >/dev/null 2>&1 || true
      sleep 3
      kill -0 "$server_pid" 2>/dev/null && kill -KILL "$server_pid" >/dev/null 2>&1 || true
    fi
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
  cleanup_gpu
}

summarize() {
  python3 - "$RUN_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = [
    "native_cpu_offload",
    "isolated_uvm_migration_offload",
    "isolated_uvm_remote_offload",
    "isolated_uvm_hybrid_remote_offload",
]
patterns = {
    "successful_requests": r"Successful requests:\s+([0-9.]+)",
    "failed_requests": r"Failed requests:\s+([0-9.]+)",
    "output_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
}

def text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""

def first_float(blob: str, pattern: str):
    match = re.search(pattern, blob)
    return float(match.group(1)) if match else None

def ratio(value, base):
    if value is None or base in (None, 0):
        return None
    return value / base

def manifest_stats(path: Path):
    out = {
        "manifest_parameters": 0,
        "manifest_bytes": 0,
        "backend_counts": {},
        "uvm_managed": 0,
        "uvm_prefetch_cpu_success": 0,
        "uvm_preferred_cpu_success": 0,
        "uvm_accessed_by_success": 0,
        "uvm_readmostly_success": 0,
        "uvm_remote_read_bg_success": 0,
        "uvm_remote_read_bg_failure": 0,
        "uvm_remote_read_bg_ioctl_ok": 0,
        "uvm_bg_start_success": 0,
        "uvm_bg_start_failure": 0,
        "uvm_bg_start_ioctl_ok": 0,
        "uvm_gate_events": 0,
        "uvm_gate_pause_success": 0,
        "uvm_gate_resume_success": 0,
        "uvm_gate_start_success": 0,
        "names": [],
    }
    for line in text(path).splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("event") == "cpu_offload_hybrid_remote_gate":
            out["uvm_gate_events"] += 1
            if int(rec.get("pause_success") or 0) > 0:
                out["uvm_gate_pause_success"] += 1
            if int(rec.get("resume_success") or 0) > 0:
                out["uvm_gate_resume_success"] += 1
            if int(rec.get("start_success") or 0) > 0:
                out["uvm_gate_start_success"] += 1
            continue
        if rec.get("event") != "cpu_offload_parameter":
            continue
        out["manifest_parameters"] += 1
        out["manifest_bytes"] += int(rec.get("bytes") or 0)
        backend = rec.get("offload_backend", "unknown")
        out["backend_counts"][backend] = out["backend_counts"].get(backend, 0) + 1
        if rec.get("uvm_managed") is True:
            out["uvm_managed"] += 1
        if int(rec.get("uvm_prefetch_cpu_result") or 0) > 0:
            out["uvm_prefetch_cpu_success"] += 1
        if int(rec.get("uvm_preferred_cpu_result") or 0) > 0:
            out["uvm_preferred_cpu_success"] += 1
        if int(rec.get("uvm_accessed_by_result") or 0) > 0:
            out["uvm_accessed_by_success"] += 1
        if int(rec.get("uvm_readmostly_result") or 0) > 0:
            out["uvm_readmostly_success"] += 1
        if int(rec.get("uvm_remote_read_bg_success") or 0) > 0:
            out["uvm_remote_read_bg_success"] += 1
        out["uvm_remote_read_bg_failure"] += int(
            rec.get("uvm_remote_read_bg_failure") or 0
        )
        if int(rec.get("uvm_remote_read_bg_ioctl_ret") or 0) == 0 and int(
            rec.get("uvm_remote_read_bg_success") or 0
        ) > 0:
            out["uvm_remote_read_bg_ioctl_ok"] += 1
        if int(rec.get("uvm_bg_start_success") or 0) > 0:
            out["uvm_bg_start_success"] += 1
        out["uvm_bg_start_failure"] += int(
            rec.get("uvm_bg_start_failure") or 0
        )
        if int(rec.get("uvm_bg_start_ioctl_ret") or 0) == 0 and int(
            rec.get("uvm_bg_start_success") or 0
        ) > 0:
            out["uvm_bg_start_ioctl_ok"] += 1
        name = rec.get("full_parameter_name")
        if isinstance(name, str):
            out["names"].append(name)
    return out

rows = []
for case in cases:
    case_root = root / case
    bench = text(case_root / "bench.log")
    row = {"case": case}
    try:
        row["exit_code"] = int((case_root / "status.txt").read_text().strip())
    except Exception:
        row["exit_code"] = None
    for key, pattern in patterns.items():
        row[key] = first_float(bench, pattern)
    stats = manifest_stats(case_root / "offload_manifest.jsonl")
    row.update({k: v for k, v in stats.items() if k != "names"})
    row["_names"] = stats["names"]
    rows.append(row)

native = rows[0] if rows else None
if native:
    for row in rows:
        row["output_tok_s_ratio_vs_native"] = ratio(
            row.get("output_tok_s"), native.get("output_tok_s")
        )
        row["mean_tpot_ratio_vs_native"] = ratio(
            row.get("mean_tpot_ms"), native.get("mean_tpot_ms")
        )
if native:
    for row in rows:
        row["same_offload_names_as_native"] = native["_names"] == row["_names"]
    native["same_offload_names_as_native"] = True

for row in rows:
    row.pop("_names", None)

fields = [
    "case", "exit_code", "successful_requests", "failed_requests",
    "output_tok_s", "mean_tpot_ms", "mean_ttft_ms", "mean_itl_ms",
    "output_tok_s_ratio_vs_native", "mean_tpot_ratio_vs_native",
    "manifest_parameters", "manifest_bytes", "backend_counts",
    "uvm_managed", "uvm_prefetch_cpu_success",
    "uvm_preferred_cpu_success", "uvm_accessed_by_success",
    "uvm_readmostly_success",
    "uvm_remote_read_bg_success",
    "uvm_remote_read_bg_failure", "uvm_remote_read_bg_ioctl_ok",
    "uvm_bg_start_success", "uvm_bg_start_failure", "uvm_bg_start_ioctl_ok",
    "uvm_gate_events", "uvm_gate_pause_success", "uvm_gate_resume_success",
    "uvm_gate_start_success",
    "same_offload_names_as_native",
]
with (root / "isolated_uvm_offload_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fields})

(root / "isolated_uvm_offload_summary.json").write_text(
    json.dumps({"run_root": str(root), "rows": rows}, indent=2, sort_keys=True),
    encoding="utf-8",
)

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

lines = [
    "# Isolated vLLM CPU Offload UVM Remote Experiment",
    "",
    f"- run_root: `{root}`",
    "- global UVM allocator: not enabled (`VLLM_USE_UVM` is unset)",
    "- LD_PRELOAD CUDA allocator interception: not used",
    "",
    "| case | ok/fail | tok/s | TPOT ms | TTFT ms | tok/s/native | TPOT/native | params | bytes | backend | managed | prefetch CPU | preferred CPU | AccessedBy | ReadMostly | remote-bg prepare | gate events | gate start | same names |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
]
for row in rows:
    ok = row.get("successful_requests")
    fail = row.get("failed_requests")
    lines.append(
        "| {case} | {ok}/{fail} | {tok} | {tpot} | {ttft} | {tok_ratio} | "
        "{tpot_ratio} | {params} | {bytes} | {backend} | {managed} | "
        "{prefetch} | {preferred} | {accessed} | {readmostly} | "
        "{remote_bg} | {gate_events} | {gate_start} | {same} |".format(
            case=row.get("case"),
            ok=fmt(ok, 0),
            fail=fmt(fail, 0),
            tok=fmt(row.get("output_tok_s")),
            tpot=fmt(row.get("mean_tpot_ms")),
            ttft=fmt(row.get("mean_ttft_ms")),
            tok_ratio=fmt(row.get("output_tok_s_ratio_vs_native")),
            tpot_ratio=fmt(row.get("mean_tpot_ratio_vs_native")),
            params=row.get("manifest_parameters"),
            bytes=row.get("manifest_bytes"),
            backend=json.dumps(row.get("backend_counts"), sort_keys=True),
            managed=row.get("uvm_managed"),
            prefetch=row.get("uvm_prefetch_cpu_success"),
            preferred=row.get("uvm_preferred_cpu_success"),
            accessed=row.get("uvm_accessed_by_success"),
            readmostly=row.get("uvm_readmostly_success"),
            remote_bg=row.get("uvm_remote_read_bg_success"),
            gate_events=row.get("uvm_gate_events"),
            gate_start=row.get("uvm_gate_start_success"),
            same=row.get("same_offload_names_as_native"),
        )
    )
lines.extend([
    "",
    "## Interpretation",
    "",
    "- `native_cpu_offload` is the vLLM pinned CPU tensor + UVA CUDA view baseline.",
    "- `isolated_uvm_migration_offload` uses the same `--cpu-offload-gb` selection, but those selected weights are allocated from a local UVM MemPool and only prefetched to CPU, leaving later GPU reads to the default UVM demand migration/fault path.",
    "- `isolated_uvm_remote_offload` uses the same selected weights, additionally sets PreferredLocation(CPU) and tags AccessedBy(GPU), so it exercises the native UVM remote-read path.",
    "- `isolated_uvm_hybrid_remote_offload` uses the same selected weights, additionally sets ReadMostly, calls the modified UVM remote-read/background-promotion prepare path, and gates background promotion around module forward calls.",
    "- The key validity checks are: same offload names, same manifest bytes, UVM managed count equals parameter count, remote-bg prepare count equals parameter count for the hybrid case, and no failed requests.",
])
(root / "isolated_uvm_offload_summary.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print((root / "isolated_uvm_offload_summary.md").read_text(encoding="utf-8"))
PY
}

echo "==========================================================="
echo "Isolated vLLM CPU offload: native UVA vs UVM remote weights"
echo "run_root=$RUN_ROOT"
echo "model=$MODEL prompts=$PROMPTS output_len=$OUTPUT_LEN request_rate=$REQUEST_RATE"
echo "vllm_cmd=${VLLM_CMD[*]}"
echo "==========================================================="

run_case native_cpu_offload uva "$PORT_BASE"
run_case isolated_uvm_migration_offload isolated_uvm_migration "$((PORT_BASE + 1))"
run_case isolated_uvm_remote_offload isolated_uvm_remote "$((PORT_BASE + 2))"
run_case isolated_uvm_hybrid_remote_offload isolated_uvm_hybrid_remote "$((PORT_BASE + 3))"
summarize

echo "summary_md=$RUN_ROOT/isolated_uvm_offload_summary.md"
