# E10 整 Parameter 混合块双页表端到端 attempt-10（2026-08-30）

## 结论

按授权只启动了一次完整 E10，没有重试。该轮成功通过 attempt-09 的初始化释放
故障，并验证整 Parameter CPU shadow 和混合块双页表准备/切换可以工作；但在运行时
追加 `cold70→cold80` 专家关系时触发 UVM 内核空指针错误，尚未进入请求测量。
因此本轮没有可报告的 Prefill、Decode、rebalance 或请求 E2E 性能，不能作为成功的
E10 端到端结果。

故障后 UVM 内核锁未被释放，`nvidia-smi` 处于不可中断等待。当前机器必须重启后
才能安全卸载故障模块并恢复 GPU；未擅自重启，也未自动开始下一轮实验。

产物目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-10-full-parameter-partial-dual-qualification`

完整内核 Oops：`/tmp/e10_attempt10_oops.log`

## 本轮协议

- 完整序列：1 次启动；
- 统计重复：0；
- KV：2→4→6 GiB；
- 专家档位：cold70→cold80→cold89；
- 专家策略：cold-frequency；
- workload 预测：关闭；
- CPU shadow：整 Parameter；
- 混合块双页表：启用；
- 父入口切换批次：每批 64 个 2 MiB 块。

## 已成功完成的部分

1. 96 个完整 Parameter、共 28,991,029,248 bytes（27.0 GiB）成功建立 CPU/GPU
   双副本；
2. cold70 初始释放成功，15,854,469,120 bytes 冷专家全部变为 CPU-only；
3. 8 GiB KV 地址池预留成功；
4. Prefill 与 Decode CUDA Graph 捕获成功；
5. 混合块双页表准备 3,360 个 2 MiB 块；
6. CPU/GPU 两个方向各切换一次，共写入 6,720 个父入口；
7. 双表准备耗时 54.700 ms，两次切换合计 13.758 ms。

上述数据只证明优化路径在初始化探测中命中，不能代替端到端请求性能结果。

## 失败位置

Graph 捕获完成后，系统开始把专家档位从 cold70 扩大到 cold80。新增关系需要先撤下
对应地址当前的普通映射；UVM 在处理一个仍标记为 2 MiB 映射、但其中部分 GPU
物理页已由 CPU-shadow cutover 释放的混合块时，继续按“整块 GPU 页面完整存在”
处理。随后 `block_phys_page_address()` 读取了空的 GPU chunk 指针，触发内核
NULL dereference。

调用栈：

```text
block_phys_page_address
block_gpu_pte_write_big
block_unmap_gpu
uvm_va_block_unmap
block_slot_direct_invalidate_locked
uvm_va_block_slot_alias_map_batch
uvm_api_moe_static_alias_map
```

对应源码位置：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:4742`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:9941`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:13517`

## 根因与修复方向

整 Parameter ReadMostly shadow 使参数页在初始化时同时位于 CPU 和 GPU。cold70
cutover 只释放选中的冷专家 GPU 子页。对于没有进入双页表快路径的混合 2 MiB 块，
当前代码可能仍保留“整块使用 2 MiB GPU 映射”的状态标记。之后增加 cold80 关系时，
通用撤映射路径才尝试拆分该 2 MiB 映射；此时部分 GPU chunk 已经不存在，因而崩溃。

应在 immutable CPU shadow 局部 cutover 中：

1. 若目标不是完整 2 MiB 块且当前使用 2 MiB 映射，在任何 GPU chunk 被释放前先
   拆成下层页表并等待完成；
2. 再释放选中的 GPU 子页；
3. 在通用撤映射入口增加 fail-closed 检查：若仍标记 2 MiB 映射但物理 chunk 不完整，
   返回 `NV_ERR_INVALID_STATE`，禁止解引用空指针。

## 后续错误保护（已实现，尚未加载实测）

已在通用的部分 2 MiB 撤映射路径加入 fail-closed 保护。驱动会在分配下层页表、
提交 GPU 工作和调用 `block_gpu_unmap_split_2m()` 之前检查：

- 所有保留映射的页面是否仍指向同一个 CPU/GPU 存储位置；
- 每个保留页面的物理 backing 是否仍然存在；
- 逻辑驻留记录是否缺失。

任一检查失败都会写入 `moe_2m_split_reject` 诊断记录并返回
`NV_ERR_INVALID_STATE`。现有 Python 调用路径看到 `ok=false` 后会抛出 RuntimeError，
使实验失败退出，不再进入空指针路径。该检查发生在任何页表修改之前，因此返回时
当前 ioctl 尚未产生新的页表副作用。

验证结果：

- 保护顺序静态测试：1 passed；
- release/error-path 测试：47 passed；
- static-alias/slot-direct 测试：85 passed；
- `make -j8 modules`：成功；
- 新模块尚未加载，因为 attempt-10 Oops 后的现有 UVM 仍被锁住，必须重启。

## 当前机器状态

- 实验 Python 主进程已退出；
- 实验包装脚本已终止；
- 一个 `nvidia-smi` 进程仍处于内核不可中断等待；
- UVM 无法安全卸载，stock UVM 自动恢复未完成；
- 需要得到重启授权后才能继续修复验证或运行 attempt-11。
