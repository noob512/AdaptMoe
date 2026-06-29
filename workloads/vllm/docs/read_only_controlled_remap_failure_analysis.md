# Read-only controlled remap 当前失败原因分析

日期：2026-06-29

本文档总结 tiny vLLM controlled E2E 当前仍失败的现象、已经排除的原因、较可能的深层原因，以及下一轮定位重点。

## 当前最新现象

最新一轮 tiny vLLM controlled E2E 已经越过早期几个卡点：

- `slot_va_block_remap` smoke 通过。
- Stage M 48 层 MoE placement 完成。
- VIDMEM prefill、只读 PTE remap、reset 均返回成功。
- slot residual scan 为 0。
- KV init/warmup skip 已完成。
- `initialize_from_config end` 已打印。
- EngineCore 打印过类似 `init engine took 10.76s` 的初始化完成日志。

但是 API server 仍没有收到 READY，readiness 超时。终止 vLLM 后，退出路径卡在 UVM mm teardown 附近。

关键内核栈/日志特征：

```text
EngineCore main thread:
  __do_sys_brk
  down_write_killable
```

```text
pt_nccl_heartbt or cleanup thread:
  uvm_va_space_mm_shutdown
  uvm_va_space_down_write(va_space)
```

诊断日志显示某个 VA space shutdown 进入：

```text
va_space_mm_shutdown enter retained=2
va_space_mm_shutdown down_write begin
```

但没有看到对应：

```text
va_space_mm_shutdown down_write end
```

这说明当前最新问题更像是：

```text
EngineCore READY 前后的普通进程内存操作 / brk / mmap_lock
        与
UVM va_space_mm_shutdown / VA-space lock / mm retainer

之间出现锁等待或引用未释放
```

而不是 Stage M remap ioctl 自身直接失败。

## 已经排除或降低优先级的原因

### 1. 不是简单的 nvidia-smi 根因

早期实验中看到多个 `nvidia-smi` 进入 D 状态，因此一度怀疑宿主监控或 `qemu-guest-agent` 放大了 NVIDIA RM lock wedge。

后续实验关闭/避开外部查询后，仍能复现 vLLM E2E 卡死。因此当前判断是：

- `nvidia-smi` 可能是受害者或放大器。
- 它不是充分根因。
- 真正问题仍在 controlled remap 与后续 UVM/mm/RM 生命周期交互里。

### 2. 不是 Stage M driver backend 直接失败

多轮 vLLM trace 显示：

```text
records=48
backend_ok=48
remap_failures=0
copy_failed=0
reset_ok=48
max_after_active_slots=0
max_after_residual_va_blocks=0
max_after_residual_pages=0
```

这说明至少从 ioctl 返回值、slot pool counters 和 residual scan 看：

- `gpu_copy_only` prefill 成功。
- `slot_va_block_remap` 成功。
- reset 后 active/backed slot 清零。
- VA block residual scan 没发现 `moe_slot_pte_bits` 残留。

这不能证明完全没有生命周期 bug，但已经排除了“明显 PTE 残留未清”这一类直接错误。

### 3. 不是旧的 managed VA copy_from_user 卡点

早期 `gpu_copy_only` 直接在内核对 managed expert VA 做 `copy_from_user`，会触发 UVM CPU fault，并卡在：

```text
uvm_va_space_cpu_fault
```

当前 vLLM 路径已经改为 host mmap staging：

```text
expert managed VA
  -> ctypes.memmove to anonymous host mmap
  -> kernel copy_from_user(host mmap)
  -> CE copy to slot VIDMEM
```

因此旧的 `uvm_va_space_cpu_fault` 卡点不再是当前最新现象。

### 4. 不是同一 slot 多 range metadata 覆盖这一已知 bug

曾经发现一个明确 bug：同一 expert slot 只有一个 PTE range metadata，导致 gate/up 和 down 多 range 中前一个 range 可能被后一个 range 覆盖，reset 后 VA block 仍有残留 slot PTE。

当前已经修复：

- slot 支持多个 PTE range。
- reset/rollback/release 遍历所有 range 清理。
- query reset 后检查 `pte_residual_*`。

最新 trace 中 residual 为 0，因此该 bug 不再是当前最可能解释。

### 5. KV device-direct / cudaMalloc 不是最新卡点

某一轮曾卡在：

```text
worker initialize_from_config RPC begin
initialize_kv_cache begin allocator=default
phase=initialize_kv_cache
predicted_class=kv_persistent
action=device_direct
device_direct_backend=cuda_malloc
```

后来 controlled remap E2E 默认关闭了 `KV_DEVICE_DIRECT` 和 fixed-weight device-direct，使测试聚焦只读专家权重。最新现象已经越过 KV init 和 warmup skip，因此 KV cudaMalloc/native allocation 不再是最新卡点。

## 当前最可能的原因

### 原因 A：VA-space lock 与 mmap_lock/brk 的锁顺序问题

最新 EngineCore 主线程卡在：

```text
__do_sys_brk -> down_write_killable
```

`brk` 需要拿进程 `mmap_lock` 写锁。与此同时，UVM teardown 线程卡在：

```text
uvm_va_space_mm_shutdown -> uvm_va_space_down_write(va_space)
```

可能的等待结构：

```text
Thread A:
  持有 mmap_lock read 或某个 mm/UVM retainer
  等待 VA-space lock 或 tracker

Thread B:
  持有 VA-space lock 或等待 VA-space write
  退出/teardown 需要等待 retained_count 降到 0

EngineCore main:
  brk 需要 mmap_lock write
  被 mmap_lock read holder 阻塞
```

这种问题不会表现为 ioctl failure，而是在 vLLM 初始化完成后、READY 前后触发普通 Python/heap allocation 或线程启动时卡住。

### 原因 B：mm retainer 未释放或释放顺序不完整

`uvm_va_space_mm_shutdown enter retained=2` 很关键。它说明 shutdown 进入时仍有至少两个 retainer。

当前 controlled remap 新增了多处 retain/release：

- `uvm_va_space_mm_or_current_retain_lock()`
- `uvm_va_space_mm_or_current_release_unlock()`
- fault/migrate/unmap/split/release guards 中的 mm/VA-space 引用
- query reset 中的 mm retain

如果某条错误路径：

- retain 后提前 return；
- rollback 失败后没有 release；
- tracker wait 或 block retry 中断后没有对称 release；
- release path 在 `mm == NULL` 时只 forget metadata 但上游 retainer 状态没有完全配平；

就可能导致 shutdown 看到 `retained_count > 0`，进而等待或卡在 VA-space write lock 附近。

### 原因 C：guard clear 路径在不合适上下文中进入 VA-block uninstall/tracker wait

当前生命周期 guard 的策略是遇到 slot PTE 先 clear/fallback。这个方向是对的，但风险在于 clear 的上下文。

高风险组合：

```text
已经处于 fault/mmu notifier/release/split/teardown 上下文
  -> 持有 mmap_lock、VA block lock、VA-space read lock 或 UVM internal lock
  -> guard 尝试升级到 VA-space write
  -> guard 执行 VA block unmap
  -> guard wait tracker
```

如果 tracker completion 或另一路 teardown 需要当前上下文持有的锁，就会形成间接死锁。

当前已经做过一个缓解：`mm == NULL` 的 release/teardown guard 只 forget slot metadata，不再进入 VA-block uninstall/tracker wait。但最新问题说明还可能存在其他上下文仍会走危险 clear/wait。

### 原因 D：EngineCore READY 前后的 vLLM 线程/IPC 初始化触发 Python heap growth

最新卡点已经在 `initialize_from_config end` 之后，说明 worker 初始化主体已返回或接近返回。READY 前后 vLLM 还会继续建立：

- input socket/queue 线程；
- output socket/queue 线程；
- structured output manager；
- scheduler/cache metadata；
- stats/metrics 组件；
- Python 对象分配和 GC/freeze。

这些路径会触发普通 `brk` 或 mmap 操作。正常情况下这和 UVM 无关；但如果 UVM 里仍有 mmap_lock reader 或 mm retainer，Python heap growth 就会成为第一个暴露卡死的位置。

因此 `__do_sys_brk` 可能不是根因，而是“第一个需要 mmap_lock write 的受害者”。

### 原因 E：NCCL heartbeat / torch distributed teardown 暴露 UVM fd release 问题

终止 vLLM 后看到 `pt_nccl_heartbt` 卡在 UVM shutdown。该线程可能不是根因，但它说明：

- vLLM/torch/NCCL 线程仍持有 CUDA/UVM 相关资源；
- 退出清理会触发 UVM fd release 或 VA-space shutdown；
- shutdown 需要 VA-space write lock 和 retainer draining；
- 一旦 retainer 或锁顺序有问题，退出路径无法完成。

这也是为什么 D 状态进程无法靠 `kill -9` 清掉：线程已经在不可中断内核等待里。

## 与已修复 bug 的关系

历史上已经修过几个真实 bug，它们降低了当前问题空间：

1. `copy_from_user(managed expert VA)` 导致 UVM CPU fault 卡死。
   - 现已通过 host mmap staging 避免。

2. `uvm_va_block_unmap_moe_slot()` 先清 `moe_slot_pte_bits`，导致普通 unmap 把 slot PTE 当成普通 GPU mapping。
   - 现已改成普通 unmap 运行期间保留该 bit，最后由普通 unmap metadata 清理。

3. 一个 slot 只记录一个 PTE range。
   - 现已支持 multi-range，reset/rollback/release 清理所有 range。

4. controlled E2E 混入 KV device-direct/cudaMalloc。
   - 现已默认关闭，使测试聚焦只读专家权重。

当前剩余问题更靠近：

```text
slot PTE 已清理干净
但新增 guard/retain/release/teardown 路径仍可能影响 UVM VA-space/mm 锁生命周期
```

## 下一轮定位重点

当前最需要的是找出谁持有 VA-space lock、mmap_lock reader 或 mm retainer。

已经加入或应继续保留的日志：

- `uvm_va_space_mm_retain/release` 打印 caller `%pS`。
- `uvm_va_space_mm_shutdown` 打印：
  - enter retained count；
  - down_write begin；
  - down_write end；
  - wait retained begin/end。
- EngineCore READY/HELLO/input socket 边界日志。
- `GPUWorker.initialize_from_config()`、`GPUModelRunner.run_uvm_init_time_weight_placement()`、KV init 子步骤日志。

建议下一轮实验只跑最小 case：

```bash
cd /home/ubuntu/nvidia-uvm-gpu

sudo rmmod nvidia_uvm 2>/dev/null || true
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko \
  uvm_moe_slot_pte_remap_enable=1 \
  uvm_moe_slot_pte_remap_max_bytes=4194304 \
  uvm_moe_slot_pool_vidmem_budget_bytes=1073741824 \
  uvm_enable_debug_procfs=1 \
  uvm_enable_builtin_tests=1

RUN_ROOT=log/current_controlled_remap_engine_base_debug \
workloads/vllm/run_stage_m_controlled_remap_skip_warmup_debug.sh
```

运行中避免：

- `nvidia-smi` / NVML 查询；
- `cleanup_gpu.py`；
- 宿主侧 guest-exec 监控；
- 任何额外 CUDA/RM 压力。

卡住时应收集：

```bash
ps -eo pid,ppid,stat,wchan:40,comm,args | \
  egrep 'VLLM|vllm|pt_|python|nvidia|uvm'

sudo cat /proc/<pid>/task/*/stack

dmesg -T | egrep \
  'MoE controlled remap debug|va_space_mm_shutdown|mm_retain|mm_release|moe_slot'
```

重点看：

- 最后一次 `mm_retain caller` 是谁。
- 是否存在 retain 无 release 的 caller。
- 哪个线程在 `uvm_va_space_down_write()` 前持有 VA-space read/write。
- EngineCore 在 READY 前最后一个用户态 marker。

## 可尝试的隔离开关

为了进一步缩小范围，可以做以下 A/B：

1. `gpu_copy_only` only，不执行 `slot_va_block_remap`。
   - 如果 READY 正常，说明问题来自 PTE/guard/teardown 路径。

2. `slot_va_block_remap` 但不 reset。
   - 只在 smoke 中使用，不能长期留在 vLLM。
   - 如果不 reset 反而不挂，说明 reset/clear/uninstall 路径是嫌疑点。

3. `slot_va_block_remap` + reset，但关闭某类 guard。
   - 例如先只保留 explicit reset/release，临时关闭 fault/migrate/access-counter clear。
   - 用于定位，不建议作为最终实现。

4. 跳过 EngineCore cache/structured-output RPC。
   - 如果跳过后 READY，则说明卡点在 EngineCore 初始化完成后的 Python/IPC/heap growth。

5. 单 layer / 单 expert / 单 role。
   - 若单 role 稳定，多 role 不稳定，则回到 multi-range clear 审计。
   - 若单 role 也不稳定，则更像全局 lock/retainer 问题。

## 当前结论

当前 read-only controlled remap 的数据搬运和 PTE 安装路径已经比较清楚：

```text
host staging -> gpu_copy_only VIDMEM backing -> slot_va_block_remap readonly PTE -> reset residual zero
```

但是 tiny vLLM E2E 还不能认为稳定可用。最新失败已经从 RM lock / KV cudaMalloc / PTE residual，推进到 UVM VA-space/mm teardown 和 EngineCore READY 前的 mmap/brk 交互。

最可能的根因是：

```text
controlled remap 新增的 guard/clear/reset/retainer 路径中，
仍有某条路径造成 VA-space lock、mmap_lock 或 mm retainer 生命周期不对称。
```

下一步应优先审计和定位 mm retainer 对称性、VA-space lock 升级路径、以及 guard 中的 tracker wait 位置，而不是继续扩大 vLLM stress。

