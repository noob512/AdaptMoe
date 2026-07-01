#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_ROOT="${OUT_ROOT:-log/current_uvm_remote_bg_vs_copy_microbench}"
PYTHON="${PYTHON:-$ROOT/workloads/vllm/.venv/bin/python3}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
LAYER="${LAYER:-0}"
EXPERT="${EXPERT:-113}"
EXPERT_COUNT="${EXPERT_COUNT:-1}"
DEVICE="${DEVICE:-0}"
HIDDEN="${HIDDEN:-64}"
INTERMEDIATE="${INTERMEDIATE:-64}"
REPEATS="${REPEATS:-5}"
PROMOTION_WAIT_MS="${PROMOTION_WAIT_MS:-20}"
PIN_MEMORY="${PIN_MEMORY:-1}"
CASES="${CASES:-sync_copy_miss uvm_fault uvm_prefetch uvm_remote_bg}"

mkdir -p "$OUT_ROOT"

echo "===== UVM remote-bg vs explicit copy microbench =====" | tee "$OUT_ROOT/run.log"
echo "out_root=$OUT_ROOT" | tee -a "$OUT_ROOT/run.log"
echo "checkpoint_dir=$CHECKPOINT_DIR" | tee -a "$OUT_ROOT/run.log"

for case_name in $CASES; do
  case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  CMD=(
    "$PYTHON" workloads/vllm/bench_uvm_remote_bg_vs_copy.py
    --case "$case_name"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --layer "$LAYER"
    --expert "$EXPERT"
    --expert-count "$EXPERT_COUNT"
    --device "$DEVICE"
    --hidden "$HIDDEN"
    --intermediate "$INTERMEDIATE"
    --repeats "$REPEATS"
    --promotion-wait-ms "$PROMOTION_WAIT_MS"
    --out-json "$case_dir/result.json"
  )
  if [ "$PIN_MEMORY" = "1" ]; then
    CMD+=(--pin-memory)
  fi

  {
    echo
    echo "===== case=$case_name ====="
    printf 'cmd='
    printf '%q ' "${CMD[@]}"
    echo
  } | tee -a "$OUT_ROOT/run.log"

  set +e
  "${CMD[@]}" >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  status=$?
  set -e
  echo "$status" >"$case_dir/status.txt"

  {
    echo "case=$case_name status=$status"
    tail -n 20 "$case_dir/stderr.log" || true
    tail -n 80 "$case_dir/stdout.log" || true
  } | tee -a "$OUT_ROOT/run.log"
done

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {"out_root": str(root), "cases": {}}
for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    status = int((case_dir / "status.txt").read_text().strip())
    entry = {"status": status, "has_result": (case_dir / "result.json").exists()}
    if entry["has_result"]:
        data = json.loads((case_dir / "result.json").read_text())
        entry.update({
            "passed": data.get("passed"),
            "effective_backing": data.get("effective_backing"),
            "batch_bytes": data.get("batch_bytes"),
            "per_expert_bytes": data.get("per_expert_bytes"),
            "first_mean_ms": data.get("first_access", {}).get("mean_ms"),
            "second_mean_ms": data.get("second_access", {}).get("mean_ms"),
            "first_p95_ms": data.get("first_access", {}).get("p95_ms"),
            "second_p95_ms": data.get("second_access", {}).get("p95_ms"),
            "events": data.get("events", {}),
            "correctness": data.get("correctness", {}),
        })
        events = data.get("events", {})
        prepare = events.get("remote_bg_prepare_result")
        promote = events.get("remote_bg_start_promotion_result")
        if prepare is not None or promote is not None:
            entry["remote_bg_prepare_effective"] = bool(
                isinstance(prepare, list) and len(prepare) >= 4
                and int(prepare[0]) > 0 and int(prepare[3]) == 0
            )
            entry["remote_bg_promotion_effective"] = bool(
                isinstance(promote, list) and len(promote) >= 4
                and int(promote[0]) > 0 and int(promote[3]) == 0
            )
    summary["cases"][case_dir.name] = entry

text = json.dumps(summary, indent=2, sort_keys=True)
print(text)
(root / "summary.json").write_text(text + "\n", encoding="utf-8")
PY
