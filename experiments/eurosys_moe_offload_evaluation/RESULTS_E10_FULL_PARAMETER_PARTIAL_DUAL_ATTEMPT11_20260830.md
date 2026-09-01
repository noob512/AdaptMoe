# E10 整 Parameter 混合块双页表 attempt-11（2026-08-30）

## 结论

机器重启后，按授权只启动了一次完整 E10，没有重试。新增的内核错误保护成功把
attempt-10 的空指针崩溃变成了可识别的 `NV_ERR_INVALID_STATE`，本轮没有发生内核
Oops，也没有再次锁死 UVM；包装脚本最终成功恢复 stock UVM，GPU 显存占用为 0 MiB。

但是实验仍在初始化期间的 `cold70→cold80` 档位切换失败，尚未进入 2/4/6 GiB
请求测量，因此没有 Prefill、Decode、rebalance 或请求 E2E 性能结果。本轮不能作为
成功的 E10 端到端结果。

产物目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-11-error-guard-qualification`

原包装脚本的增量内核日志文件为空；本报告另从本轮对应时间窗口的系统日志中提取
Xid 与模块恢复记录，保存为 `kernel.journal.extract.txt`。

## 本轮协议

- 完整序列启动：1 次；
- 统计重复：0；
- KV 目标：2→4→6 GiB；
- 专家档位：cold70→cold80→cold89；
- 专家策略：cold-frequency；
- workload 预测：关闭；
- CPU shadow：整 Parameter；
- 混合块双页表：启用；
- 父入口切换批次：每批 64 个 2 MiB 块。

## 已完成的部分

1. 96 个完整 Parameter、共 28,991,029,248 bytes（27.0 GiB）建立 CPU shadow；
2. cold70 初始释放 15,854,469,120 bytes，驻留检查显示全部目标权重位于 CPU；
3. CUDA Graph 捕获完成；
4. 准备 3,360 个双页表父入口，耗时 54.363 ms；
5. CPU/GPU 两个方向各切换 3,360 个父入口，合计 16.859 ms；
6. PARK 驱动阶段耗时 143.280 ms。

这些是初始化路径数据，不能代替端到端请求性能结果。

## 失败位置和错误保护效果

`cold70→cold80` 需要追加 4,500 条关系、覆盖 2,359,296,000 bytes。驱动在实际修改
前发现目标 2 MiB 块的逻辑状态与现有 GPU 物理页不一致，拒绝整个请求：

- success_count：0；
- failure_count：4,500；
- `rm_status=64` / `NV_ERR_INVALID_STATE`；
- 驱动 ioctl：40.809 ms；
- Python 包装层在内的追加操作：66.534 ms；
- 页表失效次数：0。

这证明 fail-closed 保护命中了预期故障：它没有继续解引用缺失的 GPU chunk，因而没有
重现 attempt-10 的内核空指针和 UVM 锁死。

## 仍然存在的问题

保护只阻止了“继续修改错误页表”，没有修复此前已经形成的不一致状态。失败后的通用
回滚仍尝试从旧 GPU 地址读取权重，触发 CUDA illegal memory access。内核记录了
NVIDIA Xid 31，错误地址为 `0x7c6e_fa900000`，进程最终以返回码 134 退出。

根因仍是：cold70 初始局部释放 GPU 子页之前，没有先把包含这些子页的 2 MiB 大页
映射拆成 64 KiB/4 KiB 结构。物理页已经被释放，但旧的 2 MiB 映射状态仍可能存在。
错误保护能安全拒绝后续操作，却无法使旧状态重新有效。

下一步需要在 immutable CPU shadow cutover 中调整操作顺序：

1. 释放任意 GPU 子页之前，先拆分相应的 2 MiB 映射并等待拆分完成；
2. 再将目标页切换到 CPU shadow 并释放 GPU 物理页；
3. 若追加档位失败，禁止使用通用 GPU 回读回滚，直接销毁会话并退出进程。

## 清理状态

- 实验进程返回码：134；
- 内核日志采集命令返回码：0；
- 驱动会话数：0；
- GPU 显存占用：0 MiB；
- 残留计算进程：0；
- stock UVM 已恢复；
- stock UVM srcversion：`85A79790636BBD99BA3E43B`；
- `nvidia-smi` 正常。

因此本轮属于“安全失败”：没有性能数据，但验证了新保护可以避免再次锁死机器。
