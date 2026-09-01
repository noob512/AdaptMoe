# E10 双页表父入口切换 Qualification（2026-08-30）

## 结论

双页表父入口切换已经实现，但本次唯一一次 GPU qualification 在初始化
SEAL 阶段失败，没有进入 Prefill/Decode，因此不能给出性能提升数字，也不能声称
459.22 ms 的 RESUME 已被消除。本轮没有重试。

失败不是 CPU 副本缺失、显存不足或地址跨 Parameter 分配。根因是第一批几何上
完整的 2 MiB 候选块没有满足初版快路径的全部 UVM 内部状态条件；初版把“单个块
不适用”当成整个会话失败，返回 `NV_ERR_INVALID_STATE`（`rm_status=64`）。

实验后已修正为混合路径：适合的完整 2 MiB 块使用双页表父入口切换；不适合的块
自动保留原有逐页 PARK/RESUME 路径，不再导致 SEAL 失败。该修复已经完成模块编译
和静态测试，但按“不重复实验”约束，尚未进行第二次 GPU 验证。

## 已实现的优化

1. 初始化时为可用的 2 MiB Parameter 块同时保留两种访问方案：CPU backing 视图和
   GPU slot-cache 视图。
2. PARK/RESUME 时不重写块内 512 个 4 KiB 页表项，只切换其父入口。
3. 每 64 个 2 MiB 块组成一次提交，并合并地址缓存失效。
4. 不完整或状态不适合的块继续使用原有叶级页表路径，以保证正确性。
5. REBIND、quiesced remove、DESTROY、GPU VA-space 销毁和异常退出会先安全丢弃双
   视图，避免发布旧方案或泄漏页表资源。
6. 增加 prepare、reuse、skip、双向 switch、父入口写入数、地址缓存清理批次数和
   耗时计数；verifier 要求双向切换实际发生且计数一致。

预期的操作规模是：可覆盖区域从每个 2 MiB 块的 512 个叶入口更新，下降为每块
1 个父入口更新；仍不适合快路径的块保留原开销。实际覆盖率和端到端延迟必须由下
一次获授权的 GPU 运行测量，不能由静态分析代替。

## 唯一一次 GPU 运行结果

运行目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-06-dual-parent-switch-qualification`

| 项目 | 结果 |
|---|---:|
| GPU 运行次数 | 1 |
| 完成序列 | 0 |
| 测量请求 | 0 |
| 统计重复 | 0 |
| 模型加载 | 成功 |
| 初始 SEAL | 失败，`rm_status=64` |
| 双页表 prepare 调用 | 1 |
| 已准备块 | 0 |
| 运行时切换调用 | 0 |
| prepare 时间 | 4.529 µs |
| Prefill/Decode 性能 | 无数据 |

进程最终返回 139；`result.json` 保留了 Python 异常和清理诊断。该返回码发生在
受控清理后的进程退出阶段；实验的首个功能错误仍是 SEAL ioctl 返回
`NV_ERR_INVALID_STATE`。

## 排除项与几何分析

- cold70 权重驻留：CPU 15,854,469,120 bytes，GPU 0，unknown 0。
- immutable CPU shadow：1,824 个范围，4,303,355,904 bytes，状态正常。
- source prefill：1,824 个 slot 已复制，6,720 次复用，状态正常。
- 6,720 个 consumer 范围合并为 3,084 个连续区间，可产生 5,627 个完整 2 MiB
  候选块。
- 5,627 个候选块全部完整位于单个 Parameter 分配内，跨分配块为 0。
- kernel log 中 Xid=0、MMU fault=0。
- cleanup errors 为空；最终恢复 stock UVM，GPU 显存 0 MiB。

因此失败范围收敛到“块的实时页表/UVM 状态不满足快路径假设”，而不是输入几何、
CPU 数据或资源不足。

## 失败后的安全回退修复

初版流程：

```text
完整 2 MiB 几何候选
→ 任一内部状态不满足
→ 整个 SEAL 失败
```

修复后：

```text
完整 2 MiB 几何候选
├─ 状态适合：准备双页表，运行时切父入口
└─ 状态不适合：计入 skipped，继续使用原逐页路径
```

prepare 会原地压缩候选数组，会话只保存真正完成双页表准备的块。后续 PARK、
RESUME 和销毁只切换这部分块，其余块继续由原逻辑处理，因此不会重复修改或漏掉
页表范围。

## 验证

- 内核模块完整编译成功；仅有原有的 1376-byte stack-frame warning。
- `test_uvm_static_alias_ioctl.py`：33 passed。
- E10 定向 static preflight：9 passed，159 deselected。
- `git diff --check`：通过。
- 修复后模块 SHA256：
  `9c87fbccff1e4db4175743b59d14cb1592e5f2c95675714549aac10d272363ed`。
- 修复后模块 srcversion：`C4A6E4BC5006406C20DDE9F`。
- 当前加载的是 stock UVM：`85A79790636BBD99BA3E43B`，GPU 0 MiB。

机器可读证据：

- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-06-dual-parent-switch-qualification/verification.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-06-dual-parent-switch-qualification/dual_view_candidate_geometry.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-06-dual-parent-switch-qualification/sequence/result.json`
