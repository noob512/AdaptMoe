#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
FORMAL_SCRIPT_DIR="${SCRIPT_DIR}"
RUN_ROOT="${1:-}"
PROTOCOL_ID="eurosys-pa-v1-20260823"
MODEL="/home/ubuntu/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
VLLM_PY="${REPO_ROOT}/workloads/vllm/vllm/.venv/bin/python"
KT_PY="${REPO_ROOT}/KTransformers/.venv/bin/python"
BLOCK_RUNNER="${SCRIPT_DIR}/formal_openai_block.py"
CLIENT="${SCRIPT_DIR}/streaming_client.py"
WORKLOADS="${SCRIPT_DIR}/configs/e2_matrix_cells.json"
FIXTURE_ROOT="${SCRIPT_DIR}/inputs/generated/${PROTOCOL_ID}/deepseek_v2_lite_bf16"
ASSET_ROOT="${FIXTURE_ROOT}/calibration_assets_attempt3"
BASE_HARNESS="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"
E2_ROOT=""
EXECUTION_TSV=""

die() {
  echo "error: $*" >&2
  exit 1
}

contains_csv() {
  local needle="$1" haystack=",$2,"
  [[ "${haystack}" == *",${needle},"* ]]
}

[[ -n "${RUN_ROOT}" ]] || die "usage: $0 RUN_ROOT"
[[ -d "${RUN_ROOT}" && -f "${RUN_ROOT}/protocol.json" ]] \
  || die "formal run root is invalid: ${RUN_ROOT}"
[[ "$(jq -r '.protocol_id' "${RUN_ROOT}/protocol.json")" == "${PROTOCOL_ID}" ]] \
  || die "formal protocol ID differs"
[[ -x "${VLLM_PY}" && -x "${KT_PY}" ]] || die "missing Python environment"
[[ -f "${MODEL}/config.json" && -f "${WORKLOADS}" ]] || die "missing frozen input"

SYSTEM_FILTER="${FORMAL_SYSTEMS:-ours,ktransformers,moe_infinity}"
TIER_FILTER="${FORMAL_TIERS:-0g,2g,4g,6g,8g}"
REPEAT_FILTER="${FORMAL_REPEATS:-1,2,3,4,5}"
E2_ROOT="${RUN_ROOT}/cases/E2/track_b_equal_system_hbm"
mkdir -p "${E2_ROOT}"
EXECUTION_TSV="${E2_ROOT}/execution.tsv"
if [[ ! -e "${EXECUTION_TSV}" ]]; then
  printf 'created_at\tsystem\ttier\trepeat\tattempt\tcase_root\treturncode\n' \
    >"${EXECUTION_TSV}"
fi

successful_attempt_exists() {
  local repeat_root="$1" verification
  [[ -d "${repeat_root}" ]] || return 1
  while IFS= read -r verification; do
    jq -e '.status == "ok"' "${verification}" >/dev/null 2>&1 && return 0
  done < <(find "${repeat_root}" -mindepth 2 -maxdepth 2 \
    -name verification.json -type f | sort)
  return 1
}

next_attempt() {
  local repeat_root="$1" attempt=1
  while [[ -e "${repeat_root}/attempt_${attempt}" ]]; do
    attempt=$((attempt + 1))
  done
  echo "${attempt}"
}

run_block() {
  local system="$1" tier="$2" repeat="$3"
  local repeat_root attempt case_root port base_url env_file client_python
  local gpu_experts frequency kv_bytes kv_ratio device_ratio
  local -a server_command runner_command overrides

  repeat_root="${E2_ROOT}/${system}/deepseek_v2_lite_bf16/budget_${tier}/repeat_$(printf '%02d' "${repeat}")"
  if [[ "${FORMAL_FORCE_RETRY:-0}" != 1 ]] \
      && successful_attempt_exists "${repeat_root}"; then
    echo "skip verified E2 case: ${system}/${tier}/repeat_${repeat}"
    return 0
  fi
  mkdir -p "${repeat_root}"
  attempt="$(next_attempt "${repeat_root}")"
  case_root="${repeat_root}/attempt_${attempt}"
  overrides=()

  case "${system}" in
    ours)
      port=30201
      env_file="${FORMAL_SCRIPT_DIR}/configs/e2_ours_${tier}_env.json"
      client_python="${VLLM_PY}"
      if [[ "${tier}" == 0g ]]; then
        server_command=(
          taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
          --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing
          --model "${MODEL}" --trust-remote-code --dtype bfloat16
          --max-model-len 8320 --gpu-memory-utilization 0.97
          --cpu-offload-gb 0 --max-num-batched-tokens 131072
          --max-num-seqs 16 --generation-config vllm
        )
      else
        case "${tier}" in
          2g) kv_bytes=2147483648 ;;
          4g) kv_bytes=4294967296 ;;
          6g) kv_bytes=6442450944 ;;
          8g) kv_bytes=8589934592 ;;
          *) die "unsupported ours tier: ${tier}" ;;
        esac
        overrides+=(
          --server-env-override "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE=${case_root}/phase_transition.json"
          --server-env-override "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_root}/offload_manifest.jsonl"
          --server-env-override "VLLM_UVM_LOG_FILE=${case_root}/uvm.log"
        )
        server_command=(
          taskset -c 0-13 "${VLLM_PY}" -m vllm.entrypoints.openai.api_server
          --host 127.0.0.1 --port "${port}" --disable-frontend-multiprocessing
          --model "${MODEL}" --trust-remote-code --dtype bfloat16
          --max-model-len 8320 --gpu-memory-utilization 0.97
          --kv-cache-memory-bytes "${kv_bytes}" --cpu-offload-gb 30
          --max-num-batched-tokens 131072 --max-num-seqs 16
          --generation-config vllm
          --compilation-config '{"cudagraph_mode":"FULL_SEPARATE","cudagraph_capture_sizes":[1,4,8,16]}'
        )
      fi
      ;;
    ktransformers)
      port=30202
      env_file="${FORMAL_SCRIPT_DIR}/configs/e2_ktransformers_env.json"
      client_python="${KT_PY}"
      case "${tier}" in
        0g) gpu_experts=64; frequency="" ;;
        2g) gpu_experts=59; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_2g.pt" ;;
        4g) gpu_experts=54; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_4g.pt" ;;
        6g) gpu_experts=50; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_6g.pt" ;;
        8g) gpu_experts=45; frequency="${ASSET_ROOT}/kt_placements/kt_frequency_8g.pt" ;;
        *) die "unsupported KTransformers tier: ${tier}" ;;
      esac
      server_command=(
        taskset -c 0-13 "${KT_PY}" -m sglang.launch_server
        --model-path "${MODEL}" --host 127.0.0.1 --port "${port}"
        --tp-size 1 --context-length 8320 --max-running-requests 16
        --mem-fraction-static 0.97 --chunked-prefill-size 131072
        --max-prefill-tokens 131072 --schedule-policy fcfs --stream-interval 1
        --random-seed 20260823 --trust-remote-code --kt-weight-path "${MODEL}"
        --kt-method BF16 --kt-cpuinfer 14 --kt-threadpool-count 1
        --kt-numa-nodes 0 --kt-num-gpu-experts "${gpu_experts}"
        --kt-expert-placement-strategy "$([[ -n "${frequency}" ]] && echo frequency || echo uniform)"
        --kt-max-deferred-experts-per-token 0 --kt-gpu-prefill-token-threshold 0
        --attention-backend flashinfer --cuda-graph-max-bs 16
        --cuda-graph-bs 1 4 8 16 --enable-mixed-chunk
        --disable-shared-experts-fusion --disable-radix-cache --enable-metrics
      )
      if [[ -n "${frequency}" ]]; then
        server_command+=(--init-expert-location "${frequency}")
      fi
      ;;
    moe_infinity)
      port=30203
      env_file="${FORMAL_SCRIPT_DIR}/configs/e2_moe_infinity_env.json"
      client_python="${VLLM_PY}"
      case "${tier}" in
        0g) device_ratio=0.75; kv_ratio=0.01 ;;
        2g) device_ratio=0.68; kv_ratio=0.07 ;;
        4g) device_ratio=0.61; kv_ratio=0.14 ;;
        6g) device_ratio=0.54; kv_ratio=0.21 ;;
        8g) device_ratio=0.47; kv_ratio=0.28 ;;
        *) die "unsupported MoE-Infinity tier: ${tier}" ;;
      esac
      server_command=(
        taskset -c 0-13 "${VLLM_PY}" "${FORMAL_SCRIPT_DIR}/launch_moe_infinity_server.py"
        --model "${MODEL}"
        --offload-path "/home/ubuntu/moe_infinity_offload/deepseek_v2_lite_chat_85864749cd611b4353ce1decdb286193298f64c7"
        --host 127.0.0.1 --port "${port}" --device-memory-ratio "${device_ratio}"
        --kv-cache-ratio "${kv_ratio}" --max-batch-size 16 --num-threads 1
      )
      ;;
    *)
      die "unknown E2 system: ${system}"
      ;;
  esac

  base_url="http://127.0.0.1:${port}"
  runner_command=(
    "${VLLM_PY}" "${BLOCK_RUNNER}"
    --case-root "${case_root}" --protocol-id "${PROTOCOL_ID}"
    --system "${system}" --model "${MODEL}" --base-url "${base_url}"
    --workload-config "${WORKLOADS}" --server-env "${env_file}"
    "${overrides[@]}"
    --python "${client_python}" --streaming-client "${CLIENT}"
    --ready-timeout 1200 --request-timeout 7200 --gpu-clean-timeout 240
    -- "${server_command[@]}"
  )

  set +e
  "${runner_command[@]}" >"${case_root}.runner.log" 2>&1
  local rc=$?
  set -e
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${system}" "${tier}" "${repeat}" \
    "${attempt}" "${case_root}" "${rc}" >>"${EXECUTION_TSV}"
  return "${rc}"
}

run_selected_system() {
  local system="$1" tier repeat failures=0
  IFS=',' read -r -a tiers <<<"${TIER_FILTER}"
  IFS=',' read -r -a repeats <<<"${REPEAT_FILTER}"
  for repeat in "${repeats[@]}"; do
    [[ "${repeat}" =~ ^[1-5]$ ]] || die "invalid repeat: ${repeat}"
    if (( repeat % 2 == 1 )); then
      ordered_tiers=("${tiers[@]}")
    else
      ordered_tiers=()
      for ((index=${#tiers[@]} - 1; index >= 0; --index)); do
        ordered_tiers+=("${tiers[index]}")
      done
    fi
    for tier in "${ordered_tiers[@]}"; do
      if ! run_block "${system}" "${tier}" "${repeat}"; then
        failures=$((failures + 1))
      fi
    done
  done
  return "${failures}"
}

total_failures=0
if contains_csv ours "${SYSTEM_FILTER}"; then
  export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
  export ALLOW_GPU_RUN=1
  export SLOT_BUDGET_BYTES="${SLOT_BUDGET_BYTES:-1207959552}"
  # shellcheck source=/dev/null
  source "${BASE_HARNESS}"
  RUN_ROOT_ACTIVE="${E2_ROOT}"
  runtime_preflight
  module_snapshot "${E2_ROOT}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${E2_ROOT}/module.custom.json" "${CUSTOM_UVM}"
  run_selected_system ours || {
    rc=$?
    total_failures=$((total_failures + rc))
  }
  restore_stock || die "failed to restore stock UVM after E2 ours"
  module_snapshot "${E2_ROOT}/module.after_ours.json" "${STOCK_UVM}"
  restart_exporters
fi

for system in ktransformers moe_infinity; do
  if contains_csv "${system}" "${SYSTEM_FILTER}"; then
    run_selected_system "${system}" || {
      rc=$?
      total_failures=$((total_failures + rc))
    }
  fi
done

echo "E2_ROOT=${E2_ROOT}"
echo "E2_FAILURES=${total_failures}"
((total_failures == 0))
