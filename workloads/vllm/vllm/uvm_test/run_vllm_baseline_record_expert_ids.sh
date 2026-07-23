#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
RUN_ROOT="${RUN_ROOT:-/tmp/vllm_baseline_record_expert_ids_$(date +%Y%m%d_%H%M%S)}"
VLLM_PY="${VLLM_PY:-.venv/bin/python}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1800}"

DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
NUM_REQUESTS="${NUM_REQUESTS:-32}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
SHAREGPT_SEED="${SHAREGPT_SEED:-42}"
FIXTURE="${FIXTURE:-${RUN_ROOT}/sharegpt_32x64_fixture.json}"
FIXTURE_META="${FIXTURE_META:-${RUN_ROOT}/fixture_metadata.json}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-0}"
EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED:-0}"
KV_CACHE_MANAGED="${KV_CACHE_MANAGED:-0}"

mkdir -p "${RUN_ROOT}"

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    head -n 1 | tr -d ' '
}

require_clean_gpu() {
  local label="$1"
  local used
  used="$(gpu_used_mib || echo 999999)"
  if [[ ! "${used}" =~ ^[0-9]+$ ]] || [[ "${used}" -gt 1024 ]]; then
    echo "GPU is not clean before ${label}: ${used} MiB" >&2
    nvidia-smi >&2 || true
    exit 3
  fi
}

generate_fixture_if_missing() {
  if [[ -f "${FIXTURE}" ]]; then
    return
  fi

  mkdir -p "$(dirname "${FIXTURE}")" "$(dirname "${FIXTURE_META}")"
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" - "${DATASET_PATH}" "${MODEL}" "${FIXTURE}" \
      "${FIXTURE_META}" "${NUM_REQUESTS}" "${OUTPUT_TOKENS}" \
      "${MAX_MODEL_LEN}" "${SHAREGPT_SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

dataset_path = Path(sys.argv[1])
model = sys.argv[2]
fixture_path = Path(sys.argv[3])
meta_path = Path(sys.argv[4])
num_requests = int(sys.argv[5])
output_tokens = int(sys.argv[6])
max_model_len = int(sys.argv[7])
seed = int(sys.argv[8])

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
data = json.loads(dataset_path.read_text(encoding="utf-8"))
random.Random(seed).shuffle(data)

requests = []
prompt_lengths = []
for item in data:
    conversations = item.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        continue
    prompt = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if turn.get("from") in ("human", "user"):
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                prompt = value.strip()
                break
    if not prompt:
        continue
    token_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if token_len <= 0 or token_len + output_tokens > max_model_len:
        continue
    requests.append({
        "request": len(requests),
        "prompt": prompt,
        "max_tokens": output_tokens,
        "prompt_tokens": token_len,
    })
    prompt_lengths.append(token_len)
    if len(requests) >= num_requests:
        break

if len(requests) != num_requests:
    raise SystemExit(f"selected {len(requests)} requests, expected {num_requests}")

metadata = {
    "source": "ShareGPT_V3_unfiltered_cleaned_split",
    "selection": "shuffled_valid_prompts",
    "seed": seed,
    "num_requests": num_requests,
    "output_tokens": output_tokens,
    "max_model_len": max_model_len,
    "min_prompt_tokens": min(prompt_lengths),
    "max_prompt_tokens": max(prompt_lengths),
    "avg_prompt_tokens": sum(prompt_lengths) / len(prompt_lengths),
}
fixture = {"metadata": metadata, "requests": requests}
fixture_path.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
print(json.dumps(metadata, indent=2, sort_keys=True))
PY
  )
}

run_case() {
  local name="$1"
  local trace_enable="$2"
  local dir="${RUN_ROOT}/${name}"
  mkdir -p "${dir}"

  require_clean_gpu "${name}"
  echo "running ${name}: routing_trace=${trace_enable}"
  local kv_args=()
  if [[ "${KV_CACHE_MEMORY_BYTES}" =~ ^[0-9]+$ ]] && [[ "${KV_CACHE_MEMORY_BYTES}" -gt 0 ]]; then
    kv_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
  fi
  set +e
  (
    cd "${VLLM_ROOT}"
    env \
      VLLM_USE_UVM=0 \
      VLLM_UVM_EXPERT_WEIGHTS_MANAGED="${EXPERT_WEIGHTS_MANAGED}" \
      VLLM_UVM_KV_CACHE_MANAGED="${KV_CACHE_MANAGED}" \
      VLLM_DISABLE_OPENAI_METRICS=1 \
      VLLM_UVM_MOE_ROUTING_TRACE_ENABLE="${trace_enable}" \
      VLLM_UVM_MOE_ROUTING_TRACE_FILE="${dir}/routing_trace.jsonl" \
      VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0 \
      VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0 \
      VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP=0 \
      VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=0 \
      VLLM_UVM_KV_RUNTIME_ENABLE=0 \
      timeout "${TIMEOUT_SEC}s" "${VLLM_PY}" "${SCRIPT_DIR}/vllm_generate_sequence.py" \
        --model "${MODEL}" \
        --request-fixture-json "${FIXTURE}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        "${kv_args[@]}" \
        --output-json "${dir}/result.json" \
        --progress-jsonl "${dir}/progress.jsonl"
  ) > "${dir}/run.log" 2>&1
  local rc=$?
  set -e
  echo "${rc}" > "${dir}/returncode.txt"
}

analyze() {
  "${VLLM_ROOT}/${VLLM_PY}" - "${RUN_ROOT}" <<'PY'
import hashlib
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": repr(exc)}

def read_rc(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
               if line.strip())

def metrics(name: str) -> dict:
    case_dir = root / name
    result = read_json(case_dir / "result.json")
    requests = result.get("request_results") or []
    req_seconds = [float(item.get("generate_seconds") or 0.0) for item in requests]
    output_tokens = sum(int(item.get("output_tokens") or 0) for item in requests)
    total = float(result.get("total_generate_seconds") or sum(req_seconds) or 0.0)
    output_hashes = [
        hashlib.sha256(str(item.get("generated", "")).encode()).hexdigest()
        for item in requests
    ]
    run_log = case_dir / "run.log"
    log_lines = (
        run_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if run_log.exists()
        else []
    )
    return {
        "returncode": read_rc(case_dir / "returncode.txt"),
        "status": result.get("status"),
        "completed_requests": int(result.get("completed_requests") or 0),
        "load_seconds": result.get("load_seconds"),
        "total_generate_seconds": total,
        "tokens_per_second": output_tokens / total if total else None,
        "p50_request_seconds": statistics.median(req_seconds) if req_seconds else None,
        "output_tokens": output_tokens,
        "routing_trace_records": line_count(case_dir / "routing_trace.jsonl"),
        "output_hashes_sha256": output_hashes,
        "gpu_memory_after_load": result.get("gpu_memory_after_load"),
        "gpu_memory_after_generate": result.get("gpu_memory_after_generate"),
        "log_tail": log_lines[-60:],
    }

summary = {
    "run_root": str(root),
    "cases": {
        "no_trace": metrics("no_trace"),
        "routing_trace": metrics("routing_trace"),
    },
}

no_trace = summary["cases"]["no_trace"]
routing_trace = summary["cases"]["routing_trace"]
if no_trace["total_generate_seconds"] and routing_trace["total_generate_seconds"]:
    summary["delta"] = {
        "routing_trace_extra_seconds": (
            routing_trace["total_generate_seconds"]
            - no_trace["total_generate_seconds"]
        ),
        "routing_trace_extra_pct": (
            routing_trace["total_generate_seconds"]
            / no_trace["total_generate_seconds"]
            - 1.0
        )
        * 100.0,
        "tokens_per_second_change_pct": (
            routing_trace["tokens_per_second"] / no_trace["tokens_per_second"]
            - 1.0
        )
        * 100.0,
    }

(root / "summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

generate_fixture_if_missing

cat > "${RUN_ROOT}/env.txt" <<EOF
model=${MODEL}
dataset=${DATASET_PATH}
fixture=${FIXTURE}
fixture_meta=${FIXTURE_META}
num_requests=${NUM_REQUESTS}
output_tokens=${OUTPUT_TOKENS}
sharegpt_seed=${SHAREGPT_SEED}
max_model_len=${MAX_MODEL_LEN}
max_num_seqs=${MAX_NUM_SEQS}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
kv_cache_memory_bytes=$([[ "${KV_CACHE_MEMORY_BYTES}" =~ ^[0-9]+$ ]] && [[ "${KV_CACHE_MEMORY_BYTES}" -gt 0 ]] && echo "${KV_CACHE_MEMORY_BYTES}" || echo auto)
expert_weights_managed=${EXPERT_WEIGHTS_MANAGED}
kv_cache_managed=${KV_CACHE_MANAGED}
EOF

run_case no_trace 0
run_case routing_trace 1
analyze
