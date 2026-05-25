#!/usr/bin/env bash
# Order-control check for UVM CPU preferred-location advice. The two cases use
# fresh processes and fresh managed allocations; this script runs both orders so
# we can separate policy effects from run-order/cache effects.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ROOT="${RUN_ROOT:-log/stage_m_uva_uvm_advise_order_check}"
PAGES="${PAGES:-262144}"
PASSES="${PASSES:-8}"
TOUCHES="${TOUCHES:-1}"
DEVICE="${DEVICE:-0}"
NVCC="${NVCC:-nvcc}"

SRC="$SCRIPT_DIR/uvm/microbench/uva_uvm_repeat_mix.cu"
BIN="$RUN_ROOT/uva_uvm_repeat_mix"
OUT="$RUN_ROOT/results.jsonl"
SUMMARY_MD="$RUN_ROOT/summary.md"

mkdir -p "$RUN_ROOT"
"$NVCC" -O3 -std=c++17 "$SRC" -o "$BIN"
: > "$OUT"

run_case() {
  local order="$1"
  local kind="$2"
  "$BIN" --scenario repeat --remote-kind "$kind" \
    --label "${order}_${kind}" \
    --pages "$PAGES" --touches "$TOUCHES" --passes "$PASSES" \
    --device "$DEVICE" | tee -a "$OUT"
}

run_case order_a uvm_cpu
run_case order_a uvm_cpu_no_advise
run_case order_b uvm_cpu_no_advise
run_case order_b uvm_cpu

python3 - "$OUT" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines() if line.strip()]
summary = Path(sys.argv[2])

def fmt(value):
    return "n/a" if value is None else f"{value:.3f}"

lines = [
    "# UVM Advise Order Check",
    "",
    "| run label | first ms | median later ms | first us/page | later us/page | pass0/pass1 |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    ratio = "n/a"
    if row.get("first_ms") and row.get("median_later_ms"):
        ratio = f"{row['first_ms'] / row['median_later_ms']:.2f}x"
    lines.append(
        f"| {row['label']} | {fmt(row['first_ms'])} | {fmt(row['median_later_ms'])} | "
        f"{fmt(row['first_per_page_us'])} | {fmt(row['median_later_per_page_us'])} | {ratio} |"
    )

lines.extend([
    "",
    "Interpretation:",
    "- If the gap follows uvm_cpu vs uvm_cpu_no_advise across both orders, it is a policy effect.",
    "- If the first case in each order is always much slower, then run-order/context warmup is contaminating the measurement.",
])
summary.write_text("\n".join(lines) + "\n")
print(summary.read_text())
PY

echo "summary_md: $SUMMARY_MD"
echo "results_jsonl: $OUT"
