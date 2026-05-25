#!/usr/bin/env bash
# Microbenchmark for the mechanism gap:
#   UVA: GPU directly touches pinned/mapped host memory.
#   UVM: GPU first-touches managed pages resident/preferred on CPU.
#
# The output is JSONL plus a small Markdown summary. Pair this with UVM kernel
# fault counters if you want exact replayable fault totals for the UVM case.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_uva_uvm_page_touch_microbench}"
PAGES="${PAGES:-262144}"
TOUCHES="${TOUCHES:-1}"
ITERS="${ITERS:-7}"
WARMUPS="${WARMUPS:-1}"
DEVICE="${DEVICE:-0}"
NVCC="${NVCC:-nvcc}"

SRC="$SCRIPT_DIR/uvm/microbench/uva_uvm_page_touch.cu"
BIN="$RUN_ROOT/uva_uvm_page_touch"
OUT="$RUN_ROOT/results.jsonl"
SUMMARY_MD="$RUN_ROOT/summary.md"

mkdir -p "$RUN_ROOT"
"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"
: > "$OUT"

"$BIN" --mode uva --pages "$PAGES" --touches "$TOUCHES" \
  --iters "$ITERS" --warmups "$WARMUPS" --device "$DEVICE" | tee -a "$OUT"
"$BIN" --mode uvm --pages "$PAGES" --touches "$TOUCHES" \
  --iters "$ITERS" --warmups 0 --device "$DEVICE" \
  --advise-cpu --prefetch-cpu | tee -a "$OUT"
"$BIN" --mode uvm --pages "$PAGES" --touches "$TOUCHES" \
  --iters "$ITERS" --warmups "$WARMUPS" --device "$DEVICE" \
  --prefetch-gpu | tee -a "$OUT"

python3 - "$OUT" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
summary = Path(sys.argv[2])

def label(row):
    if row["mode"] == "uva":
        return "uva_pinned_host"
    if row.get("prefetch_cpu"):
        return "uvm_cpu_resident_first_touch"
    if row.get("prefetch_gpu"):
        return "uvm_gpu_resident_control"
    return "uvm"

base = next((r for r in rows if label(r) == "uva_pinned_host"), None)
lines = [
    "# UVA vs UVM Page Touch Microbench",
    "",
    "| case | pages | MiB | median ms | us/page | effective GiB/s | relative to UVA |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    rel = ""
    if base and row is not base:
        rel = f"{row['median_ms'] / base['median_ms']:.2f}x"
    elif base:
        rel = "1.00x"
    lines.append(
        f"| {label(row)} | {row['pages']} | {row['mib']:.2f} | "
        f"{row['median_ms']:.3f} | {row['per_page_us']:.3f} | "
        f"{row['effective_gib_s']:.3f} | {rel} |"
    )

lines.extend([
    "",
    "Interpretation:",
    "- `uva_pinned_host` measures direct GPU reads from pinned host memory mapped into the GPU address space.",
    "- `uvm_cpu_resident_first_touch` measures managed pages deliberately moved/preferred to CPU, then touched by GPU, so replayable UVM faults and migration dominate.",
    "- `uvm_gpu_resident_control` is the lower-bound control when managed pages are already on GPU.",
])
summary.write_text("\n".join(lines) + "\n")
print(summary.read_text())
PY

echo "summary_md: $SUMMARY_MD"
echo "results_jsonl: $OUT"
