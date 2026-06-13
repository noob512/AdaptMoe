# UVM Remote-First Background Promotion 当前实验结论与下一步研究计划

本文档总结截至 2026-06-13 的研究进展。研究目标是为 UVM 中的只读或写后只读数据实现一种 latency-aware promotion 机制：

```text
首次访问尽量走低延迟 remote read；
后台把 CPU/system memory 中的数据复制到 GPU；
copy 完成后再短暂切换 GPU PTE；
后续访问走 GPU local read。
```

核心判断是：对于 LLM inference 中的权重、只读 metadata、写满后进入只读阶段的 KV block，普通 UVM demand migration 的首轮同步迁移成本过高；而纯远程访问首轮低延迟但后续无法本地化。因此更合适的目标不是简单替代 UVM migration，而是把“首次可访问”和“后续本地化”拆成两个阶段。

## 1. 当前实现状态

当前实现已经从最初的 fault-path remote-first 演进为显式 prepare 模式。主要路径如下：

```text
prepare 阶段:
  PREPARE_REMOTE_READ / PREPARE_REMOTE_READ_BG / PREPARE_REMOTE_READ_BG_DEFERRED
  -> 为 ReadMostly managed pages 建 GPU -> CPU remote READ mapping
  -> 可选择 enqueue background read-dup promotion

访问阶段:
  pass0 直接使用提前建好的 remote READ mapping

promotion 阶段:
  background worker CPU -> GPU copy
  unmap GPU old remote mapping
  标记 GPU local read-dup resident
  map GPU local READ PTE
```

已经实现的 driver/user-space 能力：

- ReadMostly remote READ premap：允许 ReadMostly 数据像 `AccessedBy` 一样提前建立 GPU 到 CPU/system memory 的 READ mapping。
- Background read-dup promotion：后台用 `COPY` 而不是 `MOVE`，保留 CPU 副本，增加 GPU 本地只读副本。
- Deferred prepare：`prepare` 只做 remote map 和 pending 标记，不启动 copy。
- After-first start：pass0 完成后通过 `START_BG_PROMOTION` 启动后台 promotion。
- 观测 counter：包括 `remote_prepare_*`、`bg_promote_*`、`pte_switch_pages`、`bg_promote_deferred_pages`、`bg_promote_start_pages`。

关键代码位置：

- ioctl action：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- Phase F dispatch：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- VA block remote premap / bg promotion：`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- benchmark：`tools/uvm_readonly_access_mechanism_bench.cu`
- 实验脚本：`scripts/test_readonly_access_mechanisms.sh`、`scripts/exp_prepare_remote_bg_*.sh`

## 2. 已获得的核心实验结论

### 2.1 本地化收益非常大

早期 256 MiB 只读实验显示：

| 机制 | 关键结果 |
|---|---:|
| `remote_only_ro` | 稳定后约 41.981 ms/pass |
| `local_prefetch_ro` | 稳定后约 1.811 ms/pass |
| `first_touch_readdup_ro` | 首轮约 62.221 ms，之后约 1.815 ms |

结论：

```text
只读数据一旦有 GPU local copy，后续访问收益非常大。
remote-only 稳态比 local/read-dup 慢约 20x 以上。
```

这证明了本研究的后半段价值：如果后台 promotion 能成功完成，后续读取可以接近 GPU local 性能。

### 2.2 用户态 cudaMemPrefetchAsync 不能证明“迁移中仍 remote read”

早期 `cudaMemPrefetchAsync` 实验中：

| 机制 | 关键结果 |
|---|---:|
| `bg_promote_exact_ro` | prefetch submit 13.479 ms，pass0 1.916 ms |
| `bg_promote_over_ro` | prefetch submit 20.354 ms，pass0 1.932 ms |

raw CSV 中 `prefetch_done_before=1`，说明第一次 kernel 前 prefetch 已经完成。因此该实验测到的是：

```text
用户态 prefetch 提交/等待阶段已经完成迁移；
pass0 直接测到本地访问；
没有证明 copy 进行中 GPU 仍能 remote read 同一 VA。
```

结论：

```text
cuda runtime prefetch 不是 driver 内后台 promotion 的等价物。
真正目标必须进入 UVM driver path 实现或验证。
```

### 2.3 fault-path remote-first 方向正确但首轮过重

第一版 remote-first 插在 replayable fault path 中：

```text
GPU replayable READ fault
-> UVM fault service
-> block_try_remote_first_bg_promote()
-> 临时建立 GPU -> CPU remote READ mapping
-> replay
```

实验发现：

| 机制 | 首轮 | 第二轮 |
|---|---:|---:|
| `uvm_pure_remote` | 约 2.95 ms | 约 2.92 ms |
| `hybrid_remote_first_only` | 约 55 ms | 约 2.93 ms |
| `hybrid_remote_first_bg` | 约 12-13 ms | 约 0.03 ms |

结论：

```text
hybrid_remote_first_only 第二轮能回到 remote 速度，说明 remote PTE 建好后访问本身不慢。
首轮慢的主要原因不是 remote read 本身，而是在 fault path 内临时建 remote PTE 太重。
```

因此研究方向从“fault 中快速 remote map”转向“fault 前显式 prepare remote map”。

### 2.4 ReadMostly remote prepare 成功消除了 fault-path 建映射开销

新增 `readmostly_prepare_remote` 后，64 MiB 两轮只读实验显示：

| 机制 | 首轮 | 第二轮 |
|---|---:|---:|
| `uvm_pure_remote` | 2.956 ms | 2.914 ms |
| `uva_pinned_remote` | 2.944 ms | 2.911 ms |
| `readmostly_prepare_remote` | 2.947 ms | 2.908 ms |
| `hybrid_remote_first_only` | 55.425 ms | 2.932 ms |

counter：

```text
remote_prepare_calls            96
remote_prepare_pages            49152
remote_prepare_mapped_pages     49152
```

结论：

```text
ReadMostly 数据提前建立 remote READ mapping 是有效的。
首轮性能可以达到 pure remote / pinned UVA 级别。
```

这证明了最重要的前半段：不是 remote 访问慢，而是 fault-path 临时建映射慢。

### 2.5 prepare + eager background promotion 可以做到首轮 remote、后续 local，但大规模时 pass0 被 copy 污染

`readmostly_prepare_remote_bg` 初版在 prepare 阶段既 premap remote，也 enqueue 并 schedule background worker。64 MiB 实验中效果很好：

| 机制 | prepare | pass0 | pass1/last |
|---|---:|---:|---:|
| `uvm_pure_remote` | 0 | 约 2.957 ms | 约 2.916 ms |
| `readmostly_prepare_remote` | 约 0.302 ms | 约 2.963 ms | 约 2.915 ms |
| `readmostly_prepare_remote_bg` | 约 1.446 ms | 约 2.623 ms | 约 0.032 ms |
| `readonly_dup` | 0 | 约 14.285 ms | 约 0.045 ms |

结论：

```text
prepare_remote_bg 可以达到“首轮接近 remote，后续接近 local/read-dup”的形态。
```

但规模增大后发现 pass0 仍可能被后台 copy 干扰：

| 数据规模 | pure remote pass0 | prepare_remote_bg pass0 | readonly_dup pass0 |
|---:|---:|---:|---:|
| 128 MiB | 2.660 ms | 3.173 ms | 7.793 ms |
| 256 MiB | 5.289 ms | 6.916 ms | 23.997 ms |
| 512 MiB | 10.572 ms | 17.970 ms | 32.727 ms |
| 1 GiB | 21.191 ms | 33.471 ms | 64.577 ms |

结论：

```text
eager background copy 会抢占 PCIe/copy engine/driver lock/tracker/PTE switch 资源。
数据越大，pass0 被污染越明显。
```

### 2.6 deferred prepare + after-first start 保护了 pass0

为解决 pass0 污染，实现了：

```text
PREPARE_REMOTE_READ_BG_DEFERRED:
  premap remote READ
  mark pending promotion
  do not schedule worker

START_BG_PROMOTION:
  pass0 完成后显式启动已有 pending work
```

256 MiB smoke 结果：

| 机制 | pass0 | pass1 | pass2/pass3 |
|---|---:|---:|---:|
| `uvm_pure_remote` | 5.296 ms | 5.278 ms | 5.267 ms |
| `readmostly_prepare_remote_bg_after_first` | 5.300 ms | 8.668 ms | 0.226 / 0.228 ms |
| `readonly_dup` | 16.493 ms | 0.229 ms | 0.229 ms |

counter：

```text
bg_promote_deferred_pages 73728
bg_promote_start_pages    73728
bg_promote_copied_pages   73728
bg_promote_committed_pages 73728
pte_switch_pages          73728
```

结论：

```text
after-first 机制已经成功保护 pass0。
pass0 几乎等于 pure remote。
pass0 后才启动 copy，pass1 进入 copy/PTE switch 干扰阶段。
pass2/pass3 收敛到 local/read-dup 性能。
```

### 2.7 pass1 变慢是预期的迁移窗口

规模扩展实验，`BG_MAX=512`：

| 数据 | pure remote p0/p1 | after_first p0/p1/p2/p3 | readonly_dup p0/p1 |
|---:|---:|---:|---:|
| 16 MiB | 0.358 / 0.340 | 0.367 / 0.509 / 0.060 / 0.023 | 1.063 / 0.026 |
| 64 MiB | 1.344 / 1.327 | 1.352 / 2.065 / 0.242 / 0.025 | 5.005 / 0.028 |
| 256 MiB | 5.296 / 5.277 | 5.300 / 7.703 / 2.047 / 0.226 | 16.554 / 0.230 |
| 512 MiB | 10.569 / 10.536 | 10.588 / 17.506 / 0.477 / 0.479 | 42.831 / 0.476 |
| 1 GiB | 21.193 / 21.169 | 21.201 / 33.366 / 0.958 / 0.952 | 64.516 / 0.953 |

结论：

```text
after_first pass0 始终贴近 pure remote。
pass1 是后台 copy 和 PTE switch 的重叠窗口，因此可能变慢。
后续 pass 收敛到 readonly_dup/local。
```

这说明当前机制把同步迁移成本从 pass0 移到了 pass1，而不是消除了成本。

### 2.8 BG_MAX 控制 local 收敛速度与干扰强度

固定 256 MiB，改变 `BG_MAX_PAGES`：

| BG_MAX | after_first p0 | after_first p1 | after_first p2 | after_first p3 |
|---:|---:|---:|---:|---:|
| 64 | 5.300 | 7.319 | 5.207 | 4.469 |
| 128 | 5.306 | 7.943 | 4.612 | 0.809 |
| 512 | 5.303 | 7.600 | 2.053 | 0.226 |
| 2048 | 5.300 | 8.599 | 0.228 | 0.231 |

结论：

```text
BG_MAX 小:
  每轮 copy 少，单轮干扰较弱，但 local 收敛慢。

BG_MAX 大:
  promotion 更快完成，pass2 可能已经 local；
  但 pass1 更容易被 aggressive copy/PTE switch 干扰。

pass0 对 BG_MAX 不敏感:
  说明 deferred prepare 确实保护了首轮访问。
```

## 3. 机制对比总结

### 3.1 五类访问机制的定位

| 机制 | 首轮目标 | 后续目标 | 问题 |
|---|---|---|---|
| pure remote / UVA pinned remote | 低延迟 remote | 一直 remote | 后续不能本地化 |
| direct migrate | 同步迁移 | GPU local | 首轮最重 |
| readonly_dup | 同步 read-dup | GPU local | 首轮仍重 |
| fault-path remote-first BG | fault 中 remote map + BG | GPU local | 首轮 fault path 太重 |
| prepare_remote_bg_after_first | 提前 remote map，pass0 后 BG | GPU local | pass1 是迁移干扰窗口 |

### 3.2 当前最佳机制

当前最符合研究目标的是：

```text
readmostly_prepare_remote_bg_after_first
```

它证明了：

```text
1. ReadMostly 数据可以提前建 remote READ mapping。
2. pass0 可以达到 pure remote 级别。
3. pass0 之后可以启动后台 read-dup promotion。
4. promotion 完成后后续访问可以达到 GPU local/read-dup 级别。
```

但它尚未解决：

```text
pass1 迁移窗口被后台 copy/PTE switch 干扰。
```

## 4. 已验证的正确性

正确性 smoke test 覆盖了：

- 多轮 GPU read。
- CPU read after promotion。
- CPU write then GPU read。
- GPU write then CPU read。
- 多 stream read。

典型结果全部 PASS：

```text
gpu_read_pass_0              PASS
gpu_read_pass_1              PASS
gpu_read_pass_2              PASS
gpu_read_pass_3              PASS
cpu_read_after_promotion     PASS
cpu_write_then_gpu_read      PASS
gpu_write_then_cpu_read      PASS
multi_stream_reads           PASS
```

结论：

```text
当前 read-dup background promotion 在基本读写一致性测试中未发现错误。
```

但这仍然只是 smoke 级别，不等价于完整正确性证明。

## 5. 当前实现的边界

当前实现仍是保守原型，适用范围包括：

```text
GPU replayable READ fault
显式 ReadMostly / read-dup policy
demand pages 当前 CPU resident
非 HMM
单 GPU 优先
只读或写后只读数据
```

不满足条件时应回退原始 UVM 路径。

尚未完成：

- access-counter 128 次提前 promotion。
- 混合读写数据的自动识别。
- staged resident-but-unmapped promotion。
- 真正低优先级 copy channel。
- copy engine / PCIe 带宽调度。
- 多 GPU 完整一致性。
- HMM read-dup 支持。
- 大规模真实 vLLM end-to-end 评估。

## 6. 对 UVM resident/mapped 的新认识

UVM 中 resident 和 mapped 是两层状态：

```text
resident:
  数据副本在哪里

mapped:
  哪些处理器页表能访问该 VA
```

相关状态：

- `va_block->resident`
- `gpu_state->resident`
- `va_block->cpu.resident`
- `va_block->mapped`
- `gpu_state->pte_bits[]`
- `cpu.pte_bits[]`

但 UVM 有一个重要限制：

```text
UVM 不精确记录每条 PTE 当前指向 CPU 还是 GPU。
它假设 mapping 总是指向离 mapping processor 最近的 resident processor。
```

因此不能简单这样做：

```text
copy CPU -> GPU
把 GPU 标成 resident
保留 GPU old remote PTE 不动
```

否则 driver 状态会认为 GPU READ PTE 已经指向 GPU local，但硬件 PTE 仍可能指向 CPU remote，造成状态不一致。

更合理的下一阶段设计是：

```text
staged_gpu_copy:
  GPU chunk/page 已经 copy 完
  但不加入 gpu_state->resident
  不加入 read_duplicated_pages
  不改变当前 GPU remote PTE

commit 阶段:
  短暂 unmap old remote PTE
  把 staged copy 变成正式 GPU resident/read-dup
  map GPU local READ PTE
```

这可以把大头 copy 与短 PTE switch 进一步拆开。

## 7. 当前问题列表

### 7.1 pass1 迁移窗口干扰

after-first 保护了 pass0，但 pass1 会与后台 copy/PTE switch 重叠。当前表现是：

```text
256 MiB:
  pure remote pass1 ~= 5.277 ms
  after_first pass1 ~= 7.703 ms

1 GiB:
  pure remote pass1 ~= 21.169 ms
  after_first pass1 ~= 33.366 ms
```

问题本质：

```text
copy 和 urgent remote read 共享 PCIe/system memory/copy engine/driver locks。
```

### 7.2 eager BG 与 delayed BG 的 tradeoff

eager BG 可能让 pass0 直接变 local，但 prepare 时间变长，语义更像提前同步搬运：

```text
prepare 时间变大
pass0 可能很快
但首轮请求已经在 prepare 阶段付出了迁移成本
```

after-first 则保护 pass0，但把成本暴露在 pass1。

### 7.3 access-counter 阶段尚未接入

最初设想是：

```text
访问约 128 次时开始后台 promotion
达到 256 次时如果 copy 完成则无缝切 local
未完成则继续 remote 或回退同步迁移
```

当前还没有实现这一阶段。已有 `counter_migrate` 结果显示原生 access-counter 路径没有达到预期 GPU local 效果，因此这是下一阶段重点。

### 7.4 staged resident 需要新状态

如果要进一步降低 PTE switch 对 pass1 的影响，需要引入 staged copy 状态，而不是直接复用 UVM resident mask。

风险点：

- staged copy 生命周期。
- free/munmap/split 时取消 staged copy。
- CPU/GPU 写 fault 时取消 staged copy。
- staged copy 与 read-dup collapse 的交互。
- staged copy 不应被 UVM 当作可映射 resident。

### 7.5 正确性验证仍不完整

还需要系统验证：

- CPU/GPU 并发写。
- GPU write fault 后 read-dup collapse。
- CPU write fault 后 staged/pending copy cancel。
- VA block split。
- munmap/free。
- 多 stream 多 block 并发。
- 多 GPU。
- HMM 不支持路径是否可靠 fallback。

## 8. 下一步研究计划

### 8.1 实现 copy pacing 策略

目标：

```text
pass0 后启动 promotion，但不要让 pass1 被 copy 过度拖慢。
```

建议策略：

```text
阶段 1:
  pass0 后低速 copy
  小 BG_MAX，例如 64/128

阶段 2:
  检测到一次或两次 remote pass 已完成
  或经过 N ms
  加速 BG_MAX 到 512/2048

阶段 3:
  临近下一轮预计访问前完成 commit
```

需要新增参数或策略：

- `uvm_latency_bg_promote_initial_max_pages`
- `uvm_latency_bg_promote_steady_max_pages`
- `uvm_latency_bg_promote_ramp_delay_ms`
- `uvm_latency_bg_promote_reschedule_delay_ms`

评价指标：

- pass0 与 pure remote 的差距。
- pass1 与 pure remote 的差距。
- pass2/pass3 是否收敛 local。
- committed pages / worker rounds / retries。

### 8.2 实现 staged read-dup promotion

目标：

```text
后台 copy 完成后先停在 staged_gpu_copy；
不立即 unmap remote PTE；
在合适时机批量 commit 到 local PTE。
```

建议阶段：

1. 增加 per-GPU staged mask。
2. copy 完成后只记录 staged，不更新 resident/read_duplicated_pages。
3. 增加 `COMMIT_STAGED_PROMOTION` action。
4. commit 时短暂 unmap remote PTE，正式加入 GPU resident/read-dup，map local READ。
5. CPU/GPU write、free、munmap、split 时 cancel staged。

预期收益：

```text
把 CPU->GPU copy 与 PTE switch 拆开；
减少 pass1 中 copy 和 PTE switch 同时发生的干扰；
允许 runtime 选择更合适的 commit 时机。
```

### 8.3 接入 access-counter 提前 promotion

目标：

```text
访问计数未到 UVM 原生迁移阈值前，提前启动 staged/background read-dup promotion。
```

设计：

```text
access count ~= 128:
  remote mapping 保持
  enqueue staged/bg promotion

access count ~= 256:
  如果 staged ready:
    commit local PTE
  否则:
    继续 remote 或回退原始 UVM path
```

需要研究的代码路径：

- `uvm_gpu_access_counters.c`
- `UVM_SERVICE_OPERATION_ACCESS_COUNTERS`
- `uvm_va_block_service_copy()`
- `uvm_va_block_make_resident_read_duplicate()`

关键问题：

- access-counter path 当前为什么没有达到 local 效果。
- access-counter 事件是否能精确区分 READ hot pages。
- 如何避免 counter path 上同步迁移。

### 8.4 增加 CPU-will-not-write / read-mostly runtime hint

目标：

```text
让 runtime 明确告诉 driver 某段 VA 在一段时间内 CPU 不会写。
```

适用对象：

- LLM weights。
- 写满后的 KV block。
- read-only metadata。
- 不再被 CPU 修改的 offload tensor。

可能接口：

```text
PREPARE_REMOTE_READ_BG_DEFERRED + NO_CPU_WRITE_EPOCH
END_NO_CPU_WRITE_EPOCH
```

或者先做实验 ioctl：

```text
UVM_PHASE_F_ACTION_MARK_CPU_READONLY_EPOCH
UVM_PHASE_F_ACTION_UNMARK_CPU_READONLY_EPOCH
```

风险：

- 如果 CPU 真的写了，必须触发 fault/cancel/collapse。
- 不能依赖用户态承诺破坏一致性。
- 第一版仍应使用 PTE write-protect 兜底。

### 8.5 做真实 vLLM end-to-end 验证

微基准已经证明机制形态，但仍需要真实 workload：

指标：

- TTFT。
- TPOT。
- tail latency。
- throughput。
- GPU memory usage。
- PCIe read/write bandwidth。
- UVM fault count。
- background promotion pages。
- local hit ratio。

对比：

- baseline UVM。
- vLLM pinned CPU offload / UVA remote。
- pure remote UVM。
- readonly_dup。
- prepare_remote_bg_after_first。
- staged promotion。

关键问题：

```text
LLM decode 阶段是否有足够 reuse 让 background promotion 回本？
KV block 的写后只读阶段能否安全识别？
weights/expert/page 级热点是否稳定？
```

## 9. 建议的下一轮实验矩阵

### 9.1 pacing sweep

```bash
PAGES=65536
PASSES=6
OPS_PER_PAGE=64
REPEATS=3
BG_MAX_VALUES="32 64 128 256 512 1024 2048"
BG_RESCHEDULE_DELAY_MS="0 1 2 5"
MECHANISMS="uvm_pure_remote readmostly_prepare_remote_bg_after_first readonly_dup"
```

观察：

- pass0 是否仍贴近 pure remote。
- pass1 干扰峰值。
- pass2/pass3 local 收敛速度。

### 9.2 staged commit prototype

实验目标：

```text
copy done 但不立刻 PTE switch；
手动在 pass1 后 commit；
观察 pass1 是否更接近 pure remote。
```

预期对比：

| 机制 | pass0 | pass1 | pass2 |
|---|---:|---:|---:|
| after_first eager commit | remote | copy + switch 干扰 | local |
| staged commit after pass1 | remote | remote + copy 干扰 | switch/local |

### 9.3 access-counter trigger

实验目标：

```text
低 ops/page 不触发原生 threshold；
中等 ops/page 提前 enqueue；
高 ops/page commit/local。
```

观察：

- `remote_first_pages`
- `remote_prepare_mapped_pages`
- `bg_promote_enqueued_pages`
- `bg_promote_deferred_pages`
- `bg_promote_start_pages`
- `bg_promote_committed_pages`
- `pte_switch_pages`

### 9.4 correctness stress

必须覆盖：

- CPU write during pending promotion。
- CPU write during staged copy。
- GPU write after local read-dup。
- free/munmap while worker pending。
- block split while staged/pending。
- 多 stream concurrent read。
- 大小混合 range。

## 10. 当前研究结论

当前已经证明：

```text
1. GPU local read-dup 对只读数据收益很大。
2. 用户态 cudaMemPrefetchAsync 不能代表 driver 内后台 promotion。
3. fault-path 临时建 remote PTE 太重，不适合作为最终方案。
4. ReadMostly remote prepare 可以把首轮访问降到 pure remote / UVA remote 级别。
5. background read-dup promotion 可以把后续访问降到 local/read-dup 级别。
6. deferred prepare + after-first start 可以保护 pass0。
7. pass1 迁移窗口仍会被后台 copy/PTE switch 干扰。
```

当前最准确的论文式表述：

```text
We demonstrate that ReadMostly managed pages can be prepared with CPU-resident remote READ mappings outside the replayable fault path, allowing the first GPU read pass to run at pure-remote latency. A deferred background read-dup promotion can then copy the pages to GPU memory and switch subsequent reads to local GPU mappings. The remaining challenge is not first-touch fault latency, but controlling the interference window where background copy and PTE switching overlap with foreground remote reads.
```

下一步研究核心：

```text
从 "能后台 local 化" 进入 "如何低干扰地后台 local 化"。
```

优先级建议：

1. copy pacing，降低 pass1 干扰。
2. staged read-dup promotion，把 copy 和 PTE switch 分离。
3. access-counter 提前触发 promotion。
4. runtime CPU-will-not-write hint。
5. vLLM end-to-end 验证。
