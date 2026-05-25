#!/usr/bin/env bash
# Re-measure UVA vs UVM overhead with two focused scenarios:
# 1) repeat: repeatedly touch the same pages to expose cold vs hot behavior.
# 2) partial: compare a mixed remote/local footprint with the same remote share.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_uva_uvm_repeat_mix_microbench}"
PAGES="${PAGES:-262144}"
REMOTE_FRACTION="${REMOTE_FRACTION:-0.10}"
TOUCHES="${TOUCHES:-1}"
PASSES="${PASSES:-8}"
DEVICE="${DEVICE:-0}"
NVCC="${NVCC:-nvcc}"
REPEAT_RESET_UVM_EACH_PASS="${REPEAT_RESET_UVM_EACH_PASS:-0}"

SRC="$SCRIPT_DIR/uvm/microbench/uva_uvm_repeat_mix.cu"
BIN="$RUN_ROOT/uva_uvm_repeat_mix"
OUT="$RUN_ROOT/results.jsonl"
SUMMARY_MD="$RUN_ROOT/summary.md"

mkdir -p "$RUN_ROOT"
"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"
: > "$OUT"

run_case() {
  "$BIN" "$@" | tee -a "$OUT"
}

run_case --scenario repeat --remote-kind uva \
  --pages "$PAGES" --touches "$TOUCHES" --passes "$PASSES" --device "$DEVICE"
repeat_reset_args=()
if [ "$REPEAT_RESET_UVM_EACH_PASS" = "1" ]; then
  repeat_reset_args=(--reset-uvm-each-pass)
fi
run_case --scenario repeat --remote-kind uvm_cpu \
  --pages "$PAGES" --touches "$TOUCHES" --passes "$PASSES" --device "$DEVICE" \
  "${repeat_reset_args[@]}"
run_case --scenario repeat --remote-kind uvm_cpu_no_advise \
  --pages "$PAGES" --touches "$TOUCHES" --passes "$PASSES" --device "$DEVICE" \
  "${repeat_reset_args[@]}"
run_case --scenario repeat --remote-kind uvm_gpu \
  --pages "$PAGES" --touches "$TOUCHES" --passes "$PASSES" --device "$DEVICE"

run_case --scenario partial --remote-kind uva \
  --pages "$PAGES" --remote-fraction "$REMOTE_FRACTION" --touches "$TOUCHES" \
  --passes "$PASSES" --device "$DEVICE"
run_case --scenario partial --remote-kind uvm_cpu \
  --pages "$PAGES" --remote-fraction "$REMOTE_FRACTION" --touches "$TOUCHES" \
  --passes "$PASSES" --device "$DEVICE"
run_case --scenario partial --remote-kind uvm_cpu_no_advise \
  --pages "$PAGES" --remote-fraction "$REMOTE_FRACTION" --touches "$TOUCHES" \
  --passes "$PASSES" --device "$DEVICE"
run_case --scenario partial --remote-kind uvm_gpu \
  --pages "$PAGES" --remote-fraction "$REMOTE_FRACTION" --touches "$TOUCHES" \
  --passes "$PASSES" --device "$DEVICE"

python3 - "$OUT" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
summary = Path(sys.argv[2])

def key(row):
    return (row["scenario"], row["remote_kind"])

def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"

lookup = {key(row): row for row in rows}
lines = [
    "# UVA vs UVM Repeat and Partial Mix Microbench",
    "",
    "## Repeat Scenario",
    "| case | first ms | median later ms | first us/page | later us/page | pass0/pass1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for remote_kind in ("uva", "uvm_cpu", "uvm_cpu_no_advise", "uvm_gpu"):
    row = lookup.get(("repeat", remote_kind))
    if row is None:
        continue
    ratio = "n/a"
    if row["first_ms"] and row["median_later_ms"]:
        ratio = f"{row['first_ms'] / row['median_later_ms']:.2f}x"
    lines.append(
        f"| repeat_{remote_kind} | {fmt(row['first_ms'])} | {fmt(row['median_later_ms'])} | "
        f"{fmt(row['first_per_page_us'])} | {fmt(row['median_later_per_page_us'])} | {ratio} |"
    )

lines.extend([
    "",
    "## Partial Scenario",
    "| case | remote pages | local pages | first ms | median later ms | first us/page | later us/page |",
    "|---|---:|---:|---:|---:|---:|---:|",
])
for remote_kind in ("uva", "uvm_cpu", "uvm_cpu_no_advise", "uvm_gpu"):
    row = lookup.get(("partial", remote_kind))
    if row is None:
        continue
    lines.append(
        f"| partial_{remote_kind} | {row['remote_pages']} | {row['local_pages']} | "
        f"{fmt(row['first_ms'])} | {fmt(row['median_later_ms'])} | "
        f"{fmt(row['first_per_page_us'])} | {fmt(row['median_later_per_page_us'])} |"
    )

lines.extend([
    "",
    "## Interpretation",
    "- repeat scenario shows how the same pages behave on the first pass versus later passes.",
    "- partial scenario keeps the same remote share and compares UVA vs UVM with the rest of the pages resident locally.",
    "- uvm_cpu uses cudaMemAdviseSetPreferredLocation(CPU) plus prefetch-to-CPU before GPU access.",
    "- uvm_cpu_no_advise prefetches managed pages to CPU but does not set preferred location, isolating the advise effect.",
    "- uva is direct mapped pinned host access.",
    "- uvm_gpu is the control for the same footprint once the pages already live on the GPU.",
])
summary.write_text("\n".join(lines) + "\n")
print(summary.read_text())
PY

echo "summary_md: $SUMMARY_MD"
echo "results_jsonl: $OUT"
