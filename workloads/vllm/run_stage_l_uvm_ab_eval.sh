#!/usr/bin/env bash
# Run Stage L validation plus a pure-UVM-vs-Stage-L performance comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
PROMPTS="${PROMPTS:-4}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-2}"
REPEAT="${REPEAT:-3}"
MAX_SWEEP_RUNS="${MAX_SWEEP_RUNS:-3}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_stage_l_uvm_ab_eval_$(date -u +%Y%m%d_%H%M%S)}"
RUN_KV_SWEEP="${RUN_KV_SWEEP:-1}"
RUN_FUNCTIONAL="${RUN_FUNCTIONAL:-1}"
LOG_FILE=""

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_l_uvm_ab_eval.sh [options]

Options:
  --run-root <path>          Output root. Default: /tmp/vllm_stage_l_uvm_ab_eval_<utc_stamp>
  --model <name>             Model name. Default: Qwen/Qwen3-30B-A3B-FP8
  --prompts <int>            Benchmark prompts. Default: 4
  --output-len <int>         Generated tokens per request. Default: 256
  --request-rate <float>     Benchmark request rate. Default: 2
  --repeat <int>             Repeats for performance baseline. Default: 3
  --max-sweep-runs <int>     Cap KV sweep variants. Default: 3
  --log-file <path>          Full shell output log. Default: <run-root>/run_stage_l_uvm_ab_eval.log
  --skip-functional          Skip gpu-functional gate.
  --skip-kv-sweep            Skip KV parameter sweep.
  -h, --help                 Show this help.

Environment overrides use the same names: MODEL, PROMPTS, OUTPUT_LEN,
REQUEST_RATE, REPEAT, MAX_SWEEP_RUNS, RUN_ROOT, RUN_KV_SWEEP, RUN_FUNCTIONAL.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --output-len) OUTPUT_LEN="$2"; shift 2 ;;
    --request-rate) REQUEST_RATE="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --max-sweep-runs) MAX_SWEEP_RUNS="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --skip-functional) RUN_FUNCTIONAL=0; shift ;;
    --skip-kv-sweep) RUN_KV_SWEEP=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "$RUN_ROOT"
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$RUN_ROOT/run_stage_l_uvm_ab_eval.log"
fi
mkdir -p "$(dirname "$LOG_FILE")"

echo "Stage L UVM A/B evaluation started."
echo "Full shell output: $LOG_FILE"
echo "Run root: $RUN_ROOT"
echo "Follow progress with: tail -f '$LOG_FILE'"

exec >"$LOG_FILE" 2>&1

echo "==========================================================="
echo " Stage L UVM A/B Evaluation"
echo " run_root: $RUN_ROOT"
echo " log_file: $LOG_FILE"
echo " model: $MODEL"
echo " prompts: $PROMPTS"
echo " output_len: $OUTPUT_LEN"
echo " request_rate: $REQUEST_RATE"
echo " repeat: $REPEAT"
echo " max_sweep_runs: $MAX_SWEEP_RUNS"
echo " pure_uvm_policy: VLLM_UVM_POLICY_ENABLE=0"
echo "==========================================================="

if [ "$RUN_FUNCTIONAL" = "1" ]; then
  echo "==========================================================="
  echo " [1/3] GPU functional gate"
  echo "==========================================================="
  "$SCRIPT_DIR/stage_l_full_eval.py" \
    --suite gpu-functional \
    --gpu-run \
    --fail-fast \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --run-root "$RUN_ROOT/gpu-functional"
else
  echo "Skipping GPU functional gate."
fi

echo "==========================================================="
echo " [2/3] Pure UVM baseline vs Stage L variants"
echo "==========================================================="
VLLM_UVM_POLICY_ENABLE=0 \
"$SCRIPT_DIR/stage_l_full_eval.py" \
  --suite performance-baseline \
  --gpu-run \
  --model "$MODEL" \
  --prompts "$PROMPTS" \
  --output-len "$OUTPUT_LEN" \
  --request-rate "$REQUEST_RATE" \
  --repeat "$REPEAT" \
  --run-root "$RUN_ROOT/performance-baseline"

if [ "$RUN_KV_SWEEP" = "1" ]; then
  echo "==========================================================="
  echo " [3/3] KV runtime/offload sweep"
  echo "==========================================================="
  VLLM_UVM_POLICY_ENABLE=0 \
  "$SCRIPT_DIR/stage_l_full_eval.py" \
    --suite kv-sweep \
    --gpu-run \
    --model "$MODEL" \
    --prompts "$PROMPTS" \
    --output-len "$OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    --max-sweep-runs "$MAX_SWEEP_RUNS" \
    --run-root "$RUN_ROOT/kv-sweep"
else
  echo "Skipping KV sweep."
fi

echo "==========================================================="
echo " Done"
echo " Reports:"
echo " - $RUN_ROOT/gpu-functional/stage_l_full_eval_report.md"
echo " - $RUN_ROOT/performance-baseline/stage_l_full_eval_report.md"
echo " - $RUN_ROOT/performance-baseline/stage_l_full_eval_summary.csv"
if [ "$RUN_KV_SWEEP" = "1" ]; then
  echo " - $RUN_ROOT/kv-sweep/stage_l_full_eval_report.md"
  echo " - $RUN_ROOT/kv-sweep/stage_l_full_eval_summary.csv"
fi
echo "==========================================================="
