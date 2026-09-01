# E7 workload-shift qualification 错误与修复记录（2026-08-29）

## Attempt-01：动态 heat 使 CUDA Graph capture invalidated

artifact：`artifacts/e7-workload-shift-REPRO-20260829-attempt-01-qualification/`

旧 `_record_uvm_dynamic_expert_heat` 在 Graph capture 内执行布尔索引和 `torch.bincount`，decode
Graph 首次捕获返回 `cudaErrorStreamCaptureInvalidated`，runner 在任何 E7 arm 前失败。失败后
PyTorch 因未完成 capture 在全局析构再次 abort；shell 仍确认 GPU clean 并恢复 stock UVM。

修复包括：

1. 用固定宽度计数器上的 `scatter_add_` 替代布尔索引和 `bincount`；
2. 捕获时标记 `graph_stable_synchronous`，maintenance 在同一稳定地址读取并原地清零；
3. 非 Graph 路径继续使用原双缓冲；
4. 用 int32、int64 两种 top-k id 的独立 CUDA Graph capture/replay/reset probe 验证计数精确；
5. E7 runner/verifier 强制 48 层均报告 `graph_stable_synchronous`。

稳定地址是必要条件，因为 Graph replay 固定写入 capture 时的设备地址；只切换 Python 侧双缓冲
指针不会改变已捕获 kernel 的目的地址。当前修复选择正确的同步语义，没有冒充异步 D2H。

## Attempt-02：跨 arm 输出 hash 键类型不一致

artifact：`artifacts/e7-workload-shift-REPRO-20260829-attempt-02-qualification/`

该 attempt 已成功捕获两个 Graph、建立 192 个动态 slots、完成 32-request calibration 和完整
frequency-cold arm。随后第一 arm 把输出 hash 存为字符串键，第二 arm 以整数 request index
查询，触发 `KeyError(0)`。这不是模型、placement 或 UVM 失败；cleanup 无错误，GPU 0 MiB，
stock UVM 正常恢复。

修复将跨 arm map 统一为 `dict[int, str]`，抽出 `record_output_digest`，并用两臂 synthetic
parity probe 确认 `{0: hash}` 可精确复用。

## Attempt-03：完整成功

artifact：`artifacts/e7-workload-shift-REPRO-20260829-attempt-03-qualification/`

- calibration 和六臂共 224 requests 全部完成；
- 六臂 output/route hash parity 通过；
- adaptive 共执行 221 个真实专家 swaps；
- Graph capture count 2 -> 2；
- runner 和 verifier 均为 `status=ok`；
- kernel 窗口无 fatal signature；
- distributed cleanup、stock UVM 恢复和 GPU 0 MiB 全部通过。

## 额外安全修复：static alias 候选隔离

E7 同时使用 6-GiB static alias base 和 4 个动态 slots/layer。原 EngineCore planner 会在全部
128 个专家中排名，可能选择由 static alias session 拥有的专家作为动态 demotion 目标。

热度快照现在显式返回 `swappable_experts` 与 `static_alias_owned_experts`，EngineCore 只在
swappable 集合中生成 target；worker replicas 的集合不一致或 current remote 不属于 swappable
时 fail closed。新增定向单测证明一个更冷但不可交换的专家不会进入 swap plan。

## 保留原则

三个 artifact 均保留各自 raw result、return code、GPU/module snapshots、kernel window 和日志。
attempt-01/02 未被覆盖或修改为成功，attempt-02 的 partial frequency arm 不进入最终六臂表。
最终结果只引用 attempt-03；正式统计重复尚未启动。
