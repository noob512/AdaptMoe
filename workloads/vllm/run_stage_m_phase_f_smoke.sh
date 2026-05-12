#!/usr/bin/env bash
# Minimal Phase F driver executor smoke test.
#
# This script intentionally does not start vLLM. It validates the Phase F
# test ioctl on a tiny cudaMallocManaged allocation so driver failures can be
# isolated before running a full model.
# 说明：
#   这是一个面向 UVM Phase F driver executor 的分层 smoke/诊断脚本。
#   它的核心目标不是启动 vLLM，而是先在最小用户态程序里验证：
#     1. 用户态测试程序能否编译；
#     2. nvidia_uvm 模块是否打开了 builtin test ioctl；
#     3. Phase F 新增的 test ioctl 路径是否能安全进入内核；
#     4. 在需要真实 managed allocation 的 step 中，UVM range 操作是否可用。
#
#   特别注意：--step ping-ioctl 只走 count=0 的 batch executor ioctl，
#   不会分配 cudaMallocManaged buffer。真正分配 managed memory 的 step
#   是 batch-mini1/query/prefetch-gpu/advise-cpu/prefetch-cpu/all。

# 严格 shell 模式：
#   -e: 命令失败时退出；
#   -u: 使用未定义变量时报错；
#   -o pipefail: pipeline 中任一命令失败则整体失败。
set -euo pipefail

# SCRIPT_DIR 是脚本所在目录，允许从任意 cwd 启动脚本。
# 当前脚本位于 workloads/vllm，因此 REPO_ROOT 向上两级回到仓库根目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# UVM_TEST_DIR 存放用户态 smoke 程序和 allocator 测试库：
#   - phase_f_smoke.cpp: smoke 可执行程序入口；
#   - uvm_allocator.cpp: 包含调用 /dev/nvidia-uvm ioctl 的 C/C++ 函数。
UVM_TEST_DIR="$SCRIPT_DIR/vllm/uvm_test"

# 默认使用仓库内编译出的 nvidia-uvm.ko。调用方可以通过环境变量
# UVM_KO_PATH=/path/to/nvidia-uvm.ko 覆盖它。
UVM_KO_PATH="${UVM_KO_PATH:-$REPO_ROOT/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko}"

# 默认把日志留在项目目录下的固定位置，避免写到 /tmp 后因 VM 重启/清理而丢失。
# 调用方仍可以通过环境变量 RUN_ROOT 或 --run-root 覆盖它。
RUN_ROOT="${RUN_ROOT:-}"

# LOG_FILE 为空时，后面会默认写到项目目录下的固定文件名。
LOG_FILE=""

# 默认测试参数：
#   BYTES: managed allocation 大小，仅对需要分配 cudaMallocManaged 的 step 有效；
#   DEVICE: CUDA device id；
#   ASYNC: 传给迁移/prefetch 路径的 async 标志；
#   STEP: 默认跑完整 all 流程。
BYTES="${BYTES:-1048576}"
DEVICE="${DEVICE:-0}"
ASYNC="${ASYNC:-0}"
STEP="all"

# ENABLE_BUILTIN_TESTS=1 时，脚本会尝试通过 sysfs 打开 builtin tests。
# --no-enable-tests 会把它改成 0：脚本不主动写 sysfs，但仍然会检查
# 当前 nvidia_uvm 是否已经启用了 uvm_enable_builtin_tests=1。
ENABLE_BUILTIN_TESTS=1

# FORCE_RELOAD_UVM=1 会 rmmod/insmod nvidia_uvm 并带上 builtin test 参数。
# 这是最后手段：脚本里也明确提示这个路径在当前测试系统上不稳定。
FORCE_RELOAD_UVM=0

usage() {
  cat <<'USAGE'
Usage:
  ./run_stage_m_phase_f_smoke.sh [options]

Options:
  --run-root <path>
  --log-file <path>       default log/phase_f_crash_diag/latest_<step>.log
  --bytes <n>             managed allocation size, default 1048576
  --device <n>            CUDA device, default 0
  --async <0|1>           use async migrate flag for prefetch, default 0
  --step <name>           run only one step: all|ping-open|ping-init|ping-small|batch-mini0|batch-mini1|ping-ioctl|ping|query|prefetch-gpu|advise-cpu|prefetch-cpu
  --no-enable-tests       do not try to enable uvm_enable_builtin_tests
  --force-reload-uvm      last-resort: reload nvidia_uvm with builtin tests
  --reload-uvm            deprecated alias for --force-reload-uvm
  -h, --help

Checks:
  1. Build uvm_allocator.so and phase_f_smoke
  2. Verify /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests
  3. Run Phase F ping ioctl with count=0
  4. Allocate a small cudaMallocManaged buffer
  5. Run QUERY_RESIDENCY
  6. Run PREFETCH_GPU
  7. Run ADVISE_CPU
  8. Run PREFETCH_CPU

Example:
  ./run_stage_m_phase_f_smoke.sh \
    --no-enable-tests \
    --step ping-ioctl
USAGE
}

# 记录用户是否显式指定 run root；未指定时脚本会使用 latest 目录并在每次运行前清理。
RUN_ROOT_EXPLICIT=0

# 命令行参数解析。这里不做复杂校验，值会原样传给后续 make/smoke 程序。
while [ $# -gt 0 ]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; RUN_ROOT_EXPLICIT=1; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --bytes) BYTES="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --async) ASYNC="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --no-enable-tests) ENABLE_BUILTIN_TESTS=0; shift ;;
    --force-reload-uvm) FORCE_RELOAD_UVM=1; shift ;;
    --reload-uvm) FORCE_RELOAD_UVM=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# 初始化输出目录。
# 默认路径全部位于仓库工作目录中，文件名固定明确：
#   主日志: log/phase_f_crash_diag/latest_<step>.log
#   辅助日志目录: log/phase_f_crash_diag/latest_<step>_run/
# 这样缺少数据盘时不会在 /tmp 里积累随机目录，也便于崩溃后直接找最新 trace。
STEP_SAFE="${STEP//[^A-Za-z0-9_.-]/_}"
DIAG_DIR="$SCRIPT_DIR/log/phase_f_crash_diag"
if [ -z "$RUN_ROOT" ]; then
  RUN_ROOT="$DIAG_DIR/latest_${STEP_SAFE}_run"
fi
if [ -z "$LOG_FILE" ]; then
  LOG_FILE="$DIAG_DIR/latest_${STEP_SAFE}.log"
fi

if [ "$RUN_ROOT_EXPLICIT" = "0" ]; then
  rm -rf "$RUN_ROOT"
fi
mkdir -p "$RUN_ROOT"
mkdir -p "$(dirname "$LOG_FILE")"
TRACE_LOG="$RUN_ROOT/smoke_trace.log"

# 写轻量 trace marker。
# 这些 marker 同时写到 stdout 和 $TRACE_LOG；stdout 在 exec 重定向之后会进入 LOG_FILE。
# sync 的目的是让系统崩溃/挂起时尽量保留最后一个已经执行到的阶段。
trace_marker() {
  local msg="$*"
  local ts
  ts="$(date -u '+%F %T.%3N' 2>/dev/null || date -u '+%F %T')"
  printf 'TRACE time_utc=%s pid=%s step=%s msg=%s\n' "$ts" "$$" "$STEP" "$msg" | tee -a "$TRACE_LOG"
  sync "$TRACE_LOG" 2>/dev/null || sync
}

# 这几行会在 stdout 重定向前打印到调用者终端，告诉用户完整日志在哪。
echo "Stage M Phase F smoke test started."
echo "Run root: $RUN_ROOT"
echo "Full shell output: $LOG_FILE"
echo "Follow progress with: tail -f '$LOG_FILE'"
trace_marker "script_start before_stdout_redirect log_file=$LOG_FILE run_root=$RUN_ROOT"

# 从这里开始，脚本 stdout/stderr 都写入 LOG_FILE。
exec >"$LOG_FILE" 2>&1
trace_marker "stdout_redirected"

# 用户态 C++ 代码会读取 VLLM_UVM_PHASE_F_DEBUG_LOG，把 ioctl 前后的细节写到这里。
# journal/dmesg 文件用于保留内核侧诊断信息。
PHASE_F_DEBUG_LOG="$RUN_ROOT/phase_f_user_debug.log"
JOURNAL_BEFORE="$RUN_ROOT/journal_before.log"
JOURNAL_AFTER="$RUN_ROOT/journal_after.log"
DMESG_AFTER="$RUN_ROOT/dmesg_after.log"
trace_marker "paths_initialized phase_f_user_debug_log=$PHASE_F_DEBUG_LOG"

# 打印日志分隔标题，方便在 LOG_FILE 里搜索阶段。
log_section() {
  echo "==========================================================="
  echo " $*"
  echo "==========================================================="
}

# 在执行可能导致系统崩溃/挂起的子命令前，强制把已经生成的诊断文件落盘。
# 这里会先写 marker，再 flush 当前 shell stdout/stderr 对应的 LOG_FILE，
# 最后对已存在的日志文件和所在目录执行 sync，尽量保证崩溃后还能看到
# 子进程启动前的最后状态。
flush_logs_before_child() {
  local name="$1"
  shift

  echo "flush_logs_before_child name=$name cmd=$*"
  date -u '+time_utc=%F %T'
  trace_marker "flush_logs_before_child_begin name=$name"

  touch "$PHASE_F_DEBUG_LOG" "$JOURNAL_BEFORE" "$JOURNAL_AFTER" "$DMESG_AFTER"
  sync "$LOG_FILE" 2>/dev/null || true
  sync "$TRACE_LOG" 2>/dev/null || true
  sync "$PHASE_F_DEBUG_LOG" 2>/dev/null || true
  sync "$JOURNAL_BEFORE" 2>/dev/null || true
  sync "$JOURNAL_AFTER" 2>/dev/null || true
  sync "$DMESG_AFTER" 2>/dev/null || true
  sync "$RUN_ROOT" 2>/dev/null || true
  sync "$(dirname "$LOG_FILE")" 2>/dev/null || true
  sync

  trace_marker "flush_logs_before_child_done name=$name"
}

# 运行一个 smoke 子步骤，并在子进程返回后抓取最近 2 分钟 kernel journal。
# 参数形式：
#   run_step <step-name> <command> [args...]
#
# 返回值就是子命令返回值。失败时额外 dump 用户态 Phase F debug log 尾部。
run_step() {
  local name="$1"
  shift
  trace_marker "run_step_enter name=$name cmd=$*"
  log_section "step: $name"
  date -u '+time_utc=%F %T'
  trace_marker "run_step_before_child name=$name"
  flush_logs_before_child "$name" "$@"
  set +e
  "$@"
  local rc=$?
  set -e
  trace_marker "run_step_after_child name=$name rc=$rc"
  echo "step_result name=$name rc=$rc"
  date -u '+time_utc=%F %T'
  trace_marker "run_step_before_journal_tail name=$name"
  journalctl -k --since '-2 minutes' --no-pager | tail -120 || true
  trace_marker "run_step_after_journal_tail name=$name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: step '$name' failed; dumping Phase F user debug log tail." >&2
    tail -200 "$PHASE_F_DEBUG_LOG" 2>/dev/null || true
    trace_marker "run_step_failed name=$name rc=$rc"
  fi
  trace_marker "run_step_exit name=$name rc=$rc"
  return "$rc"
}

# 打印本次运行配置和内核版本。后续排查 crash 时，这段是日志入口摘要。
trace_marker "before_header"
log_section "Stage M Phase F Smoke"
echo "run_root: $RUN_ROOT"
echo "bytes: $BYTES"
echo "device: $DEVICE"
echo "async: $ASYNC"
echo "step: $STEP"
echo "enable_builtin_tests: $ENABLE_BUILTIN_TESTS"
echo "force_reload_uvm: $FORCE_RELOAD_UVM"
echo "uvm_ko_path: $UVM_KO_PATH"
echo "phase_f_user_debug_log: $PHASE_F_DEBUG_LOG"
uname -a
trace_marker "after_header"

# 在执行测试前保存最近 10 分钟内核日志，便于和测试后的日志做对比。
trace_marker "before_journal_before_capture"
journalctl -k --since '-10 minutes' --no-pager > "$JOURNAL_BEFORE" || true
trace_marker "after_journal_before_capture"

# 强制 rebuild 用户态测试组件：
#   make uvm           -> uvm_allocator.so
#   make phase_f_smoke -> phase_f_smoke 可执行程序，链接 uvm_allocator.so
log_section "Build smoke binary"
trace_marker "before_build"
make -B -C "$UVM_TEST_DIR" uvm phase_f_smoke
trace_marker "after_build"

# 如果没有指定 --no-enable-tests，脚本会尝试通过 sysfs 打开 builtin tests。
# 有些已加载模块的参数不可写，此时脚本只记录提示，真正的硬性检查在后面。
if [ "$ENABLE_BUILTIN_TESTS" = "1" ] && [ "$FORCE_RELOAD_UVM" != "1" ]; then
  log_section "Enable builtin tests through writable sysfs parameter"
  trace_marker "before_enable_builtin_tests"
  if [ -w /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
    echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests >/dev/null
    trace_marker "after_enable_builtin_tests_write"
  else
    echo "uvm_enable_builtin_tests is not writable in the currently loaded module."
    echo "Build and load the updated module once, then re-run without --force-reload-uvm."
    trace_marker "enable_builtin_tests_not_writable"
  fi
fi

# 最后手段：卸载并重新加载 nvidia_uvm，同时带上 Phase F 需要的 builtin test 参数。
# 这个分支可能影响正在使用 GPU/UVM 的进程，也可能触发系统不稳定，所以默认关闭。
if [ "$FORCE_RELOAD_UVM" = "1" ]; then
  log_section "Force reload nvidia_uvm with builtin tests"
  trace_marker "before_force_reload_uvm"
  echo "WARNING: forcing UVM reload is known to be unstable on this test system."
  if lsmod | awk '{print $1}' | grep -qx nvidia_uvm; then
    trace_marker "before_force_rmmod_nvidia_uvm"
    sudo rmmod nvidia_uvm
    trace_marker "after_force_rmmod_nvidia_uvm"
  fi
  trace_marker "before_force_insmod_nvidia_uvm"
  sudo insmod "$UVM_KO_PATH" \
    uvm_enable_debug_procfs=1 \
    uvm_enable_builtin_tests=1 \
    uvm_perf_fault_log_destination=0 \
    uvm_perf_fault_log_counters=0 \
    uvm_perf_fault_log_addresses=0 \
    uvm_perf_fault_kv_range_enable=0
  trace_marker "after_force_insmod_nvidia_uvm"
fi

# 记录当前 nvidia/nvidia_uvm 模块状态，以及 builtin test 参数实际值。
log_section "UVM module state"
trace_marker "before_module_state"
lsmod | grep -E 'nvidia(_uvm)?' || true
if [ -r /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests ]; then
  echo -n "uvm_enable_builtin_tests="
  cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests
else
  echo "uvm_enable_builtin_tests parameter is not readable"
fi
trace_marker "after_module_state"

# Phase F 的 test ioctl 挂在 builtin tests 入口下。
# 如果 uvm_enable_builtin_tests 不是 1，后面 ioctl 会被内核拒绝，因此这里提前失败。
trace_marker "before_tests_enabled_check"
tests_enabled="$(cat /sys/module/nvidia_uvm/parameters/uvm_enable_builtin_tests 2>/dev/null || echo missing)"
if [ "$tests_enabled" != "1" ]; then
  echo "ERROR: Phase F test ioctl requires uvm_enable_builtin_tests=1, got '$tests_enabled'." >&2
  echo "Preferred fix: rebuild/load the updated UVM module once so the parameter is writable." >&2
  echo "Last resort: re-run with --force-reload-uvm, but that path has caused crashes." >&2
  trace_marker "tests_enabled_check_failed value=$tests_enabled"
  exit 3
fi
trace_marker "after_tests_enabled_check value=$tests_enabled"

# 告诉 uvm_allocator.cpp 把用户态 ioctl 诊断写到 $PHASE_F_DEBUG_LOG。
export VLLM_UVM_PHASE_F_DEBUG_LOG="$PHASE_F_DEBUG_LOG"
trace_marker "exported_phase_f_debug_log"

# 按 --step 分发到 phase_f_smoke。
# 单步模式只跑一个 step；all 模式先逐个跑，再跑 phase_f_smoke 自身的 all 流程。
SMOKE="$UVM_TEST_DIR/phase_f_smoke"
trace_marker "before_case_dispatch smoke=$SMOKE"
case "$STEP" in
  all)
    # 从最小风险到更完整路径逐层推进：
    #   ping-open   : 只 open /dev/nvidia-uvm；
    #   ping-init   : open 后执行 UVM_INITIALIZE_IOCTL；
    #   ping-small  : 走最小 Phase F ping ioctl；
    #   batch-mini0 : 走 mini batch ioctl，count=0；
    #   batch-mini1 : 分配 managed memory 后对一个 range 走 mini batch；
    #   ping-ioctl  : 走正式 batch executor ioctl，count=0；
    #   ping        : ping-ioctl 的同类路径；
    #   query/prefetch/advise: 对真实 managed range 执行 Phase F 操作。
    trace_marker "case_all_begin"
    run_step "ping-open" "$SMOKE" --step ping-open --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "ping-init" "$SMOKE" --step ping-init --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "ping-small" "$SMOKE" --step ping-small --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "batch-mini0" "$SMOKE" --step batch-mini0 --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "batch-mini1" "$SMOKE" --step batch-mini1 --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "ping-ioctl" "$SMOKE" --step ping-ioctl --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "ping" "$SMOKE" --step ping --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "query" "$SMOKE" --step query --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "prefetch-gpu" "$SMOKE" --step prefetch-gpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "advise-cpu" "$SMOKE" --step advise-cpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "prefetch-cpu" "$SMOKE" --step prefetch-cpu --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    run_step "all" "$SMOKE" --step all --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    trace_marker "case_all_end"
    ;;
  ping-open|ping-init|ping-small|batch-mini0|batch-mini1|ping-ioctl|ping|query|prefetch-gpu|advise-cpu|prefetch-cpu)
    # 单步诊断模式。你给的命令 --step ping-ioctl 会进入这里，只执行一个子进程。
    trace_marker "case_single_begin selected=$STEP"
    run_step "$STEP" "$SMOKE" --step "$STEP" --bytes "$BYTES" --device "$DEVICE" --async "$ASYNC"
    trace_marker "case_single_end selected=$STEP"
    ;;
  *)
    echo "ERROR: unknown --step '$STEP'" >&2
    trace_marker "case_unknown_step selected=$STEP"
    exit 2
    ;;
esac

# 最终诊断收尾：
#   - 保存测试后 10 分钟 kernel journal；
#   - 保存 dmesg；
#   - 在主日志尾部附上用户态 Phase F debug log 的最后 200 行；
#   - 打印这些文件路径，方便继续排查。
log_section "Final diagnostics"
trace_marker "before_final_diagnostics"
journalctl -k --since '-10 minutes' --no-pager > "$JOURNAL_AFTER" || true
dmesg -T > "$DMESG_AFTER" 2>/dev/null || true
echo "phase_f_user_debug_log=$PHASE_F_DEBUG_LOG"
tail -200 "$PHASE_F_DEBUG_LOG" 2>/dev/null || true
echo "journal_before=$JOURNAL_BEFORE"
echo "journal_after=$JOURNAL_AFTER"
echo "dmesg_after=$DMESG_AFTER"
trace_marker "script_success"
echo "PASS: Phase F smoke test completed."
