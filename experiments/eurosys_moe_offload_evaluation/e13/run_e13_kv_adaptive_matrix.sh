#!/usr/bin/env bash
set -euo pipefail

E13_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${E13_DIR}/../../.." && pwd)"
BASE_RUNNER="${REPO_ROOT}/experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh"

export VLLM_PHASE_SWITCH_SOURCE_ONLY=1
export ACCESS_COUNTER_MIGRATION_ENABLE="${ACCESS_COUNTER_MIGRATION_ENABLE:-0}"
export UVM_PREFETCH_ENABLE="${UVM_PREFETCH_ENABLE:-0}"
export UVM_THRASHING_ENABLE="${UVM_THRASHING_ENABLE:-0}"
export STATIC_ALIAS_DUAL_VIEW_ENABLE=1
export STATIC_ALIAS_DUAL_VIEW_REQUIRE_FULL=1
# shellcheck source=/dev/null
source "${BASE_RUNNER}"
unset VLLM_PHASE_SWITCH_SOURCE_ONLY

CONFIG="${E13_CONFIG:-${E13_DIR}/config.json}"
RUNNER="${E13_DIR}/e13_kv_adaptive_runner.py"
MONITOR="${E13_DIR}/monitor_nvml.py"
VERIFY_E13="${E13_DIR}/verify_e13_kv_adaptive.py"
AGGREGATE="${E13_DIR}/aggregate_e13_kv_adaptive.py"
PLOT="${E13_DIR}/plot_e13_kv_adaptive.py"
TESTS="${E13_DIR}/test_e13_tools.py"
PREPARE_TEACHER="${E13_DIR}/prepare_e13_teacher_targets.py"
ANALYZE_TEACHER="${E13_DIR}/analyze_e13_teacher_forced.py"
TEACHER_PROCESSOR="${E13_DIR}/e13_teacher_forced_logits.py"
PHASE_RUNNER="${E13_DIR}/e13_tier_steady_phase_runner.py"
VERIFY_PHASE="${E13_DIR}/verify_e13_tier_steady_phase.py"
DECODE_DIAGNOSTIC="${E13_DIR}/e13_decode_abba_diagnostic.py"
DECODE_NSYS_WRAPPER="${E13_DIR}/e13_decode_nsys_wrapper.py"
ANALYZE_DECODE_DIAGNOSTIC="${E13_DIR}/analyze_e13_decode_abba.py"
ANALYZE_DECODE_BACKING="${E13_DIR}/analyze_e13_decode_backing.py"
ANALYZE_DECODE_NSYS="${E13_DIR}/analyze_e13_decode_nsys.py"
MODEL="${MODEL:-/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782}"
NAMES="${REPO_ROOT}/experiments/vllm_uvm_large_kv_multilevel/inputs/exact_parameter_names.txt"
INPUT_SET="${REPO_ROOT}/experiments/vllm_uvm_e10_multilevel_pressure/inputs/formal_k2_k4_k6_k8_k10"
BASE_PLAN="${INPUT_SET}/cold70.json"
PLAN_CLASSES="${E13_DIR}/plan_classes_k2_k4_k6.json"
TIMEOUT_SEC="${TIMEOUT_SEC:-10800}"
CPUSET="${CPUSET:-0-13}"
CPU_THREADS="${CPU_THREADS:-14}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-0.2}"

KV_INITIAL_BLOCKS=1364
KV_MAX_TARGET_BLOCKS=4095
KV_MAX_BYTES=8589410304
ALLOWED_TARGET_BLOCKS=1364,2730,4095
PLAN_CLASS_MAP='{"1364":"cold70","2730":"cold80","4095":"cold89"}'
BACKUP_RESERVE_MAP='{"1364":16,"2730":16,"4095":16}'

die_e13() {
  echo "E13: $*" >&2
  exit 1
}

static_preflight_e13() {
  local path
  for path in "${CONFIG}" "${RUNNER}" "${MONITOR}" "${VERIFY_E13}" \
    "${AGGREGATE}" "${PLOT}" "${TESTS}" "${PREPARE_TEACHER}" \
    "${ANALYZE_TEACHER}" "${TEACHER_PROCESSOR}" "${PHASE_RUNNER}" \
    "${VERIFY_PHASE}" "${DECODE_DIAGNOSTIC}" "${DECODE_NSYS_WRAPPER}" \
    "${ANALYZE_DECODE_DIAGNOSTIC}" "${ANALYZE_DECODE_BACKING}" \
    "${MODEL}/config.json" \
    "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}"; do
    require_file "${path}"
  done
  [[ "$(wc -l <"${NAMES}")" -eq 96 ]] \
    || die_e13 "expected 96 exact parameter names"
  bash -n "${BASH_SOURCE[0]}"
  "${VLLM_PY}" -m py_compile \
    "${RUNNER}" "${MONITOR}" "${VERIFY_E13}" "${AGGREGATE}" "${PLOT}" \
    "${TESTS}" "${PREPARE_TEACHER}" "${ANALYZE_TEACHER}" \
    "${TEACHER_PROCESSOR}" "${PHASE_RUNNER}" "${VERIFY_PHASE}"
  "${VLLM_PY}" -m py_compile \
    "${DECODE_DIAGNOSTIC}" "${DECODE_NSYS_WRAPPER}" \
    "${ANALYZE_DECODE_DIAGNOSTIC}" \
    "${ANALYZE_DECODE_BACKING}"
  "${VLLM_PY}" -m pytest -q "${TESTS}"
  "${VLLM_PY}" - "${CONFIG}" "${REPO_ROOT}" <<'PY'
import json, pathlib, sys
config = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
p = json.load(open(config, encoding="utf-8"))
assert p["schema"] == "eurosys_e13_kv_adaptive_protocol_v1"
tiers = p["tiers"]
assert [x["target_blocks"] for x in tiers] == [1364, 2730, 4095]
assert [x["plan_class"] for x in tiers] == ["cold70", "cold80", "cold89"]
assert p["runtime"]["backup_reserve_blocks"] == 16
assert p["runtime"]["safety_blocks"] == 16
assert len(p["profiles"]["formal"]["seeds"]) >= 3
assert {x["strategy"] for x in p["arms"]} == {"adaptive", "fixed"}
assert sum(x["strategy"] == "fixed" for x in p["arms"]) >= 3
for value in p["fixtures"].values():
    assert (root / value).is_file(), value
model = pathlib.Path(p["model"])
if not model.is_absolute():
    model = root / model
assert (model / "config.json").is_file(), model
PY
  (
    cd "${VLLM_ROOT}"
    "${VLLM_PY}" -m pytest -q \
      tests/v1/engine/test_uvm_adaptive_kv.py \
      tests/v1/worker/test_uvm_eager_plan_classes_e13.py \
      -k 'autoscale or next_request or plan_class'
  )
  echo "E13 static preflight: OK"
}

profile_values() {
  local profile="$1" field="$2"
  "${VLLM_PY}" - "${CONFIG}" "${profile}" "${field}" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
profile=sys.argv[2]
if sys.argv[3] == "seeds":
    print(" ".join(map(str, p["profiles"][profile]["seeds"])))
elif sys.argv[3] == "arms":
    print(" ".join(x["name"] for x in p["arms"]))
else:
    raise SystemExit("unknown field")
PY
}

run_case_e13() {
  local profile="$1" repeat_index="$2" seed="$3" arm="$4" case_dir="$5"
  local teacher_targets="${6:-}"
  local -a teacher_args=()
  if [[ -n "${teacher_targets}" ]]; then
    teacher_args=(
      --teacher-targets "${teacher_targets}"
      --teacher-logits-output "${case_dir}/teacher_logits.pt"
    )
  fi
  local case_runner="${E13_CASE_RUNNER:-${RUNNER}}"
  local autoscale=0 initial_target=1364
  [[ "${arm}" == adaptive ]] && autoscale=1
  case "${arm}" in
    adaptive|fixed_k4|direct_k4|switched_k4) initial_target=2730 ;;
    fixed_k2|direct_k2|switched_k2) initial_target=1364 ;;
    fixed_k6|direct_k6|switched_k6) initial_target=4095 ;;
    *) die_e13 "unknown arm ${arm}" ;;
  esac

  mkdir -p "${case_dir}"
  : >"${case_dir}/cpu_offload_manifest.jsonl"
  wait_gpu_clean || die_e13 "GPU is not clean before ${arm} repeat ${repeat_index}"
  gpu_state_json "${case_dir}/gpu.before.json"
  cp -- "${STATS_PATH}" "${case_dir}/latency_stats.before.txt"
  local kernel_epoch monitor_pid rc journal_rc
  kernel_epoch="$(date +%s)"

  local -a env_values=(
    "VLLM_USE_UVM=0"
    "VLLM_ENABLE_V1_MULTIPROCESSING=0"
    "VLLM_ATTENTION_BACKEND=TRITON_ATTN"
    "VLLM_CPU_OFFLOAD_BACKEND=copy"
    "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE=${NAMES}"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE=${BASE_PLAN}"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE=${PLAN_CLASSES}"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES=1"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW=1"
    "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1"
    "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW=1"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE=1"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA=1"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH=1"
    "VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE=1"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES=524288"
    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE=auto"
    "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1"
    "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE=0"
    "VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE=0"
    "VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE=0"
    "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1"
    "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_remote_decode"
    "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES=201326592"
    "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ=0"
    "VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1"
    "VLLM_UVM_EXPERT_ENVELOPE_BYTES=524288"
    "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read"
    "VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT=0"
    "VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT=0"
    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=0"
    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release_cpu"
    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup"
    "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE=0"
    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0"
    "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=${E13_DYNAMIC_EXPERT_SWAP_ENABLE:-0}"
    "VLLM_UVM_WEIGHT_EXECUTOR=cpp_batch"
    "VLLM_CPU_OFFLOAD_COPY_TRACE=0"
    "VLLM_CPU_OFFLOAD_COPY_SYNC=0"
    "VLLM_CPU_OFFLOAD_MANIFEST_FILE=${case_dir}/cpu_offload_manifest.jsonl"
    "VLLM_CPU_OFFLOAD_MANIFEST_RESET=1"
    "VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP=1"
    "VLLM_UVM_LOG_FILE=${case_dir}/uvm.log"
    "VLLM_UVM_PHASE_BATCHED_SCHEDULING=1"
    "VLLM_UVM_PHASE_KV_AWARE_COHORTS=1"
    "VLLM_UVM_PHASE_COHORT_KV_RESERVE_BLOCKS=1"
    "VLLM_UVM_KV_CACHE_MANAGED=1"
    "VLLM_UVM_KV_CACHE_LAZY_RESERVE=1"
    "VLLM_UVM_KV_RUNTIME_ENABLE=1"
    "VLLM_UVM_KV_RUNTIME_MODE=enforce"
    "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE=1"
    "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS=${KV_INITIAL_BLOCKS}"
    "VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS=${KV_INITIAL_BLOCKS}"
    "VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS=16"
    "VLLM_UVM_KV_BACKUP_RESERVE_TARGET_MAP=${BACKUP_RESERVE_MAP}"
    "VLLM_UVM_KV_BACKUP_GROW_BLOCKS=1366"
    "VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE=0"
    "VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE=0"
    "VLLM_UVM_KV_AUTO_SCALE_ENABLE=${autoscale}"
    "VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY=0"
    "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP=${PLAN_CLASS_MAP}"
    "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS=${ALLOWED_TARGET_BLOCKS}"
    "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW=1"
    "VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS=1"
    "VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK=3"
    "VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR=2"
    "VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR=2"
    "VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS=${KV_INITIAL_BLOCKS}"
    "VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS=${KV_MAX_TARGET_BLOCKS}"
    "VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS=16"
    "VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK=0.8"
    "VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS=4"
    "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS=1365"
    "VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE=0"
    "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1"
    "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=cpp_batch"
    "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC=1"
    "VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=cpp_batch"
    "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST=1"
    "VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST=1"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
    "TOKENIZERS_PARALLELISM=false"
    "PYTHONUNBUFFERED=1"
    "E13_DECODE_DIAGNOSTIC_HEAVY=${E13_DECODE_DIAGNOSTIC_HEAVY:-0}"
    "E13_DECODE_DIAGNOSTIC_CONDITIONS=${E13_DECODE_DIAGNOSTIC_CONDITIONS:-}"
    "OMP_NUM_THREADS=${CPU_THREADS}"
    "MKL_NUM_THREADS=${CPU_THREADS}"
    "CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
  )
  printf '%s\n' "${env_values[@]}" >"${case_dir}/environment.txt"
  {
    printf 'cd %q && env ' "${VLLM_ROOT}"
    printf '%q ' "${env_values[@]}"
    printf 'timeout --kill-after=30s %qs taskset -c %q %q %q ' \
      "${TIMEOUT_SEC}" "${CPUSET}" "${VLLM_PY}" "${case_runner}"
    printf '%q ' --config "${CONFIG}" --repo-root "${REPO_ROOT}" \
      --profile "${profile}" --arm "${arm}" --seed "${seed}" \
      --repeat-index "${repeat_index}" --output "${case_dir}/result.json" \
      "${teacher_args[@]}"
    printf '\n'
  } >"${case_dir}/command.txt"

  "${VLLM_PY}" "${MONITOR}" --output "${case_dir}/nvml.csv" \
    --gpu "${GPU_INDEX}" --interval "${MONITOR_INTERVAL}" \
    >"${case_dir}/monitor.log" 2>&1 &
  monitor_pid=$!
  set +e
  (
    cd "${VLLM_ROOT}"
    env "${env_values[@]}" \
      /usr/bin/time -v -o "${case_dir}/resource_time.txt" \
      timeout --kill-after=30s "${TIMEOUT_SEC}s" \
      taskset -c "${CPUSET}" \
      "${VLLM_PY}" "${case_runner}" \
        --config "${CONFIG}" \
        --repo-root "${REPO_ROOT}" \
        --profile "${profile}" \
        --arm "${arm}" \
        --seed "${seed}" \
        --repeat-index "${repeat_index}" \
        --output "${case_dir}/result.json" \
        "${teacher_args[@]}"
  ) >"${case_dir}/run.log" 2>&1
  rc=$?
  kill -TERM "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
  sudo journalctl -k --since "@${kernel_epoch}" --no-pager -o short-unix \
    >"${case_dir}/kernel.since.log" 2>"${case_dir}/kernel.since.err"
  journal_rc=$?
  set -e
  printf '%s\n' "${rc}" >"${case_dir}/returncode.txt"
  printf '%s\n' "${journal_rc}" >"${case_dir}/kernel_log_returncode.txt"
  cp -- "${STATS_PATH}" "${case_dir}/latency_stats.after.txt"
  if ! wait_gpu_clean; then
    gpu_state_json "${case_dir}/gpu.after.json"
    return 96
  fi
  gpu_state_json "${case_dir}/gpu.after.json"
  (( journal_rc == 0 )) || return 97
  return "${rc}"
}

run_decode_diagnosis_e13() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-decode-abba-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${DECODE_DIAGNOSTIC}" \
    "${ANALYZE_DECODE_DIAGNOSTIC}" "${TEACHER_PROCESSOR}" \
    "${MODEL}/config.json" "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" \
    "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'mode\ttier\tdirectory\treturncode\n' >"${run_root}/execution_status.tsv"

  E13_CASE_RUNNER="${DECODE_DIAGNOSTIC}"
  local failures=0 mode tier arm case_dir rc
  local saved_interval="${MONITOR_INTERVAL}"
  MONITOR_INTERVAL=0.02
  for mode in performance heavy; do
    if [[ "${mode}" == heavy ]]; then
      E13_DECODE_DIAGNOSTIC_HEAVY=1
      E13_DYNAMIC_EXPERT_SWAP_ENABLE=1
    else
      E13_DECODE_DIAGNOSTIC_HEAVY=0
      E13_DYNAMIC_EXPERT_SWAP_ENABLE=0
    fi
    export E13_DECODE_DIAGNOSTIC_HEAVY E13_DYNAMIC_EXPERT_SWAP_ENABLE
    for tier in k4 k6; do
      arm="direct_${tier}"
      case_dir="${run_root}/${mode}_${tier}"
      if run_case_e13 qualification 0 20260901 "${arm}" "${case_dir}"; then
        rc=0
      else
        rc=$?
        failures=$((failures + 1))
        tail -n 240 "${case_dir}/run.log" >&2 || true
      fi
      printf '%s\t%s\t%s\t%s\n' "${mode}" "${tier}" \
        "${case_dir#${run_root}/}" "${rc}" >>"${run_root}/execution_status.tsv"
    done
  done
  MONITOR_INTERVAL="${saved_interval}"
  unset E13_CASE_RUNNER E13_DECODE_DIAGNOSTIC_HEAVY \
    E13_DYNAMIC_EXPERT_SWAP_ENABLE

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  local analyze_rc=0
  set +e
  "${VLLM_PY}" "${ANALYZE_DECODE_DIAGNOSTIC}" "${run_root}" \
    --output "${run_root}/analysis.json" \
    >"${run_root}/analyze.log" 2>"${run_root}/analyze.err"
  analyze_rc=$?
  set -e
  echo "RUN_ROOT=${run_root}"
  echo "analysis=${run_root}/analysis.json"
  (( failures == 0 && analyze_rc == 0 )) \
    || die_e13 "Decode diagnosis incomplete: cases=${failures}, analyze=${analyze_rc}"
}

run_decode_backing_diagnosis_e13() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-decode-backing-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  rg -q '^cpu_chunk_alloc_successes_2m\s' "${STATS_PATH}" \
    || die_e13 "custom UVM lacks CPU backing chunk counters"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${DECODE_DIAGNOSTIC}" \
    "${MODEL}/config.json" "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" \
    "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'tier\ttrial\tdirectory\treturncode\n' >"${run_root}/execution_status.tsv"

  E13_CASE_RUNNER="${DECODE_DIAGNOSTIC}"
  E13_DECODE_DIAGNOSTIC_HEAVY=0
  E13_DYNAMIC_EXPERT_SWAP_ENABLE=0
  E13_DECODE_DIAGNOSTIC_CONDITIONS=a1_direct
  export E13_DECODE_DIAGNOSTIC_HEAVY E13_DYNAMIC_EXPERT_SWAP_ENABLE \
    E13_DECODE_DIAGNOSTIC_CONDITIONS
  local saved_interval="${MONITOR_INTERVAL}"
  MONITOR_INTERVAL=0.02
  local failures=0 tier trial arm case_dir rc
  for tier in k4 k6; do
    arm="direct_${tier}"
    for trial in 1 2 3; do
      case_dir="${run_root}/${tier}_trial${trial}"
      if run_case_e13 qualification "${trial}" 20260901 "${arm}" "${case_dir}"; then
        rc=0
      else
        rc=$?
        failures=$((failures + 1))
        tail -n 200 "${case_dir}/run.log" >&2 || true
      fi
      printf '%s\t%s\t%s\t%s\n' "${tier}" "${trial}" \
        "${case_dir#${run_root}/}" "${rc}" >>"${run_root}/execution_status.tsv"
    done
  done
  MONITOR_INTERVAL="${saved_interval}"
  unset E13_CASE_RUNNER E13_DECODE_DIAGNOSTIC_HEAVY \
    E13_DYNAMIC_EXPERT_SWAP_ENABLE E13_DECODE_DIAGNOSTIC_CONDITIONS

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  local analyze_rc=0
  set +e
  "${VLLM_PY}" "${ANALYZE_DECODE_BACKING}" "${run_root}" \
    --output "${run_root}/analysis.json" \
    >"${run_root}/analyze.log" 2>"${run_root}/analyze.err"
  analyze_rc=$?
  set -e
  echo "RUN_ROOT=${run_root}"
  echo "analysis=${run_root}/analysis.json"
  (( failures == 0 && analyze_rc == 0 )) \
    || die_e13 "Decode backing diagnosis incomplete: cases=${failures}, analyze=${analyze_rc}"
}

run_decode_nsys_diagnosis_e13() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-decode-nsys-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${DECODE_DIAGNOSTIC}" "${DECODE_NSYS_WRAPPER}" \
    "${ANALYZE_DECODE_NSYS}" \
    "${MODEL}/config.json" "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" \
    "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'tier\tdirectory\treturncode\n' >"${run_root}/execution_status.tsv"

  E13_CASE_RUNNER="${DECODE_NSYS_WRAPPER}"
  E13_DECODE_REAL_RUNNER="${DECODE_DIAGNOSTIC}"
  E13_DECODE_DIAGNOSTIC_HEAVY=0
  E13_DYNAMIC_EXPERT_SWAP_ENABLE=0
  E13_DECODE_DIAGNOSTIC_CONDITIONS=a1_direct
  # Nsight Systems 2024.6 on this host sees PyTorch NVTX annotations but does
  # not start an NVTX-triggered capture. CUDA profiler API calls bound the
  # condition window; nested NVTX ranges still label every Decode step.
  E13_NSYS_CUDA_PROFILER_CAPTURE=1
  export E13_DECODE_REAL_RUNNER E13_DECODE_DIAGNOSTIC_HEAVY \
    E13_DYNAMIC_EXPERT_SWAP_ENABLE E13_DECODE_DIAGNOSTIC_CONDITIONS \
    E13_NSYS_CUDA_PROFILER_CAPTURE
  local saved_interval="${MONITOR_INTERVAL}"
  MONITOR_INTERVAL=0.02
  local case_dir="${run_root}/k4_nsys" rc
  if run_case_e13 qualification 1 20260901 direct_k4 "${case_dir}"; then
    rc=0
  else
    rc=$?
    tail -n 240 "${case_dir}/run.log" >&2 || true
  fi
  printf 'k4\t%s\t%s\n' "${case_dir#${run_root}/}" "${rc}" \
    >>"${run_root}/execution_status.tsv"
  MONITOR_INTERVAL="${saved_interval}"
  unset E13_CASE_RUNNER E13_DECODE_REAL_RUNNER \
    E13_DECODE_DIAGNOSTIC_HEAVY E13_DYNAMIC_EXPERT_SWAP_ENABLE \
    E13_DECODE_DIAGNOSTIC_CONDITIONS E13_NSYS_CUDA_PROFILER_CAPTURE

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  if (( rc == 0 )); then
    [[ -s "${case_dir}/timeline.nsys-rep" ]] \
      || die_e13 "Nsight capture did not produce timeline.nsys-rep"
    nsys stats --force-export=true \
      --report nvtx_gpu_proj_trace,nvtx_pushpop_trace,cuda_gpu_kern_sum,cuda_api_sum \
      "${case_dir}/timeline.nsys-rep" >"${run_root}/nsys_stats.txt" \
      2>"${run_root}/nsys_stats.err"
    "${VLLM_PY}" "${ANALYZE_DECODE_NSYS}" \
      "${case_dir}/timeline.sqlite" "${case_dir}/result.json" \
      --output "${run_root}/analysis.json"
  fi
  echo "RUN_ROOT=${run_root}"
  (( rc == 0 )) || die_e13 "Decode Nsight Systems diagnosis failed: ${rc}"
}

run_phase_once_e13() {
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-tier-phase-once-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${PHASE_RUNNER}" "${VERIFY_PHASE}" \
    "${MODEL}/config.json" "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" \
    "${CUSTOM_UVM}" >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'arm\tdirectory\treturncode\n' >"${run_root}/execution_status.tsv"

  E13_CASE_RUNNER="${PHASE_RUNNER}"
  local failures=0 arm case_dir rc
  for arm in direct_k2 switched_k2 direct_k4 switched_k4 direct_k6 switched_k6; do
    case_dir="${run_root}/${arm}"
    if run_case_e13 qualification 0 20260901 "${arm}" "${case_dir}"; then
      rc=0
    else
      rc=$?
      failures=$((failures + 1))
      tail -n 200 "${case_dir}/run.log" >&2 || true
    fi
    printf '%s\t%s\t%s\n' "${arm}" "${case_dir#${run_root}/}" "${rc}" \
      >>"${run_root}/execution_status.tsv"
  done
  unset E13_CASE_RUNNER

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  local verify_rc=0
  set +e
  "${VLLM_PY}" "${VERIFY_PHASE}" "${run_root}" \
    --output "${run_root}/verification.json" \
    >"${run_root}/verify.log" 2>"${run_root}/verify.err"
  verify_rc=$?
  set -e
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
  echo "summary=${run_root}/summary.md"
  (( failures == 0 && verify_rc == 0 )) \
    || die_e13 "phase-once incomplete: cases=${failures}, verify=${verify_rc}"
}

run_numerical_e13() {
  local source_root="$1" profile="$2"
  [[ "${profile}" == formal || "${profile}" == capacity ]] \
    || die_e13 "numerical profile must be formal or capacity"
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  require_file "${source_root}/verification.json"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01-${profile}-teacher-forced}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  source_root="$(cd "${source_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  "${VLLM_PY}" "${PREPARE_TEACHER}" "${source_root}" "${run_root}" \
    --config "${CONFIG}" --profile "${profile}" \
    >"${run_root}/prepare.log" 2>"${run_root}/prepare.err"

  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${RUNNER}" "${PREPARE_TEACHER}" \
    "${ANALYZE_TEACHER}" "${TEACHER_PROCESSOR}" "${MODEL}/config.json" \
    "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" "${CUSTOM_UVM}" \
    >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'repeat_index\tseed\tarm\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local failures=0 repeat_index seed repeat_name target_path arms_text arm case_dir rc
  while IFS=$'\t' read -r repeat_index seed repeat_name target_path arms_text; do
    [[ -n "${arms_text}" ]] || continue
    for arm in ${arms_text}; do
      case_dir="${run_root}/${repeat_name}/${arm}"
      if run_case_e13 "${profile}" "${repeat_index}" "${seed}" "${arm}" \
        "${case_dir}" "${target_path}"; then
        rc=0
      else
        rc=$?
        failures=$((failures + 1))
        tail -n 200 "${case_dir}/run.log" >&2 || true
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${repeat_index}" "${seed}" "${arm}" \
        "${case_dir#${run_root}/}" "${rc}" >>"${run_root}/execution_status.tsv"
    done
  done < <(
    "${VLLM_PY}" - "${run_root}/numerical_manifest.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
for item in p["repeats"]:
    print("\t".join((str(item["repeat_index"]), str(item["seed"]),
                     item["repeat_name"], item["target_path"],
                     " ".join(item["cases"]))))
PY
  )

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters
  local analyze_rc=0
  set +e
  "${VLLM_PY}" "${ANALYZE_TEACHER}" "${run_root}" \
    --output "${run_root}/teacher_forced_analysis.json" \
    >"${run_root}/analyze.log" 2>"${run_root}/analyze.err"
  analyze_rc=$?
  set -e
  echo "RUN_ROOT=${run_root}"
  echo "analysis=${run_root}/teacher_forced_analysis.json"
  (( failures == 0 && analyze_rc == 0 )) \
    || die_e13 "teacher-forced qualification incomplete: cases=${failures}, analysis=${analyze_rc}"
}

run_matrix_e13() {
  local profile="$1"
  [[ "${ALLOW_GPU_RUN:-0}" == 1 ]] \
    || die_e13 "GPU execution requires ALLOW_GPU_RUN=1"
  static_preflight_e13
  runtime_preflight
  local run_root="${RUN_ROOT:-${REPO_ROOT}/experiments/eurosys_moe_offload_evaluation/artifacts/e13-kv-adaptive-$(date -u +%Y%m%dT%H%M%SZ)-attempt-01-${profile}}"
  [[ ! -e "${run_root}" ]] || die_e13 "RUN_ROOT already exists: ${run_root}"
  mkdir -p "${run_root}"
  run_root="$(cd "${run_root}" && pwd)"
  RUN_ROOT_ACTIVE="${run_root}"
  module_snapshot "${run_root}/module.before.json" "${STOCK_UVM}"
  load_custom_module
  module_snapshot "${run_root}/module.custom.json" "${CUSTOM_UVM}"
  git -C "${REPO_ROOT}" status --short >"${run_root}/git_status.txt"
  sha256sum "${CONFIG}" "${RUNNER}" "${MONITOR}" "${VERIFY_E13}" \
    "${AGGREGATE}" "${PLOT}" "${TESTS}" "${MODEL}/config.json" \
    "${NAMES}" "${BASE_PLAN}" "${PLAN_CLASSES}" "${CUSTOM_UVM}" \
    >"${run_root}/input_sha256.txt"
  cp -- "${CONFIG}" "${run_root}/config.snapshot.json"
  printf 'repeat_index\tseed\tarm\tdirectory\treturncode\n' \
    >"${run_root}/execution_status.tsv"

  local seeds_text arms_text failures=0 repeat_index=0 seed arm case_dir rc
  seeds_text="${E13_SEEDS:-$(profile_values "${profile}" seeds)}"
  arms_text="${E13_ARMS:-$(profile_values "${profile}" arms)}"
  for seed in ${seeds_text}; do
    repeat_dir="${run_root}/$(printf 'repeat-%02d-seed-%s' "${repeat_index}" "${seed}")"
    mkdir -p "${repeat_dir}"
    for arm in ${arms_text}; do
      case_dir="${repeat_dir}/${arm}"
      if run_case_e13 "${profile}" "${repeat_index}" "${seed}" "${arm}" "${case_dir}"; then
        rc=0
      else
        rc=$?
        failures=$((failures + 1))
        tail -n 200 "${case_dir}/run.log" >&2 || true
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' \
        "${repeat_index}" "${seed}" "${arm}" \
        "${case_dir#${run_root}/}" "${rc}" >>"${run_root}/execution_status.tsv"
    done
    repeat_index=$((repeat_index + 1))
  done

  restore_stock || die_e13 "failed to restore stock nvidia_uvm"
  module_snapshot "${run_root}/module.after.json" "${STOCK_UVM}"
  restart_exporters

  if [[ -n "${E13_ARMS:-}" || -n "${E13_SEEDS:-}" ]]; then
    echo "RUN_ROOT=${run_root}"
    echo "partial_arm_filter=${E13_ARMS:-all}"
    echo "partial_seed_filter=${E13_SEEDS:-all}"
    (( failures == 0 )) || die_e13 "filtered pilot failed: cases=${failures}"
    return 0
  fi

  local verify_rc=0 aggregate_rc=0 plot_rc=0
  set +e
  "${VLLM_PY}" "${VERIFY_E13}" "${run_root}" --config "${CONFIG}" \
    --profile "${profile}" --output "${run_root}/verification.json" \
    >"${run_root}/verify.log" 2>"${run_root}/verify.err"
  verify_rc=$?
  "${VLLM_PY}" "${AGGREGATE}" "${run_root}" --config "${CONFIG}" \
    --profile "${profile}" >"${run_root}/aggregate.log" 2>"${run_root}/aggregate.err"
  aggregate_rc=$?
  if (( aggregate_rc == 0 )); then
    "${VLLM_PY}" "${PLOT}" "${run_root}" --config "${CONFIG}" \
      --profile "${profile}" >"${run_root}/plot.log" 2>"${run_root}/plot.err"
    plot_rc=$?
  else
    plot_rc=1
  fi
  set -e
  echo "RUN_ROOT=${run_root}"
  echo "verification=${run_root}/verification.json"
  echo "summary=${run_root}/analysis/e13_summary.md"
  (( failures == 0 && verify_rc == 0 && aggregate_rc == 0 && plot_rc == 0 )) \
    || die_e13 "matrix incomplete: cases=${failures}, verify=${verify_rc}, aggregate=${aggregate_rc}, plot=${plot_rc}"
}

case "${1:-help}" in
  static-preflight)
    static_preflight_e13
    ;;
  qualify|qualification)
    run_matrix_e13 qualification
    ;;
  formal|run)
    run_matrix_e13 formal
    ;;
  capacity)
    run_matrix_e13 capacity
    ;;
  numerical)
    [[ $# -eq 3 ]] \
      || die_e13 "usage: $0 numerical SOURCE_RUN_ROOT {formal|capacity}"
    run_numerical_e13 "$2" "$3"
    ;;
  phase-once)
    run_phase_once_e13
    ;;
  decode-diagnose)
    run_decode_diagnosis_e13
    ;;
  decode-backing-diagnose)
    run_decode_backing_diagnosis_e13
    ;;
  decode-nsys-diagnose)
    run_decode_nsys_diagnosis_e13
    ;;
  verify)
    [[ $# -eq 3 ]] || die_e13 "usage: $0 verify RUN_ROOT PROFILE"
    "${VLLM_PY}" "${VERIFY_E13}" "$2" --config "${CONFIG}" --profile "$3"
    ;;
  aggregate)
    [[ $# -eq 3 ]] || die_e13 "usage: $0 aggregate RUN_ROOT PROFILE"
    "${VLLM_PY}" "${AGGREGATE}" "$2" --config "${CONFIG}" --profile "$3"
    "${VLLM_PY}" "${PLOT}" "$2" --config "${CONFIG}" --profile "$3"
    ;;
  *)
    echo "usage: $0 {static-preflight|qualification|formal|capacity|numerical SOURCE_RUN_ROOT PROFILE|phase-once|decode-diagnose|decode-backing-diagnose|decode-nsys-diagnose|verify RUN_ROOT PROFILE|aggregate RUN_ROOT PROFILE}" >&2
    exit 2
    ;;
esac
