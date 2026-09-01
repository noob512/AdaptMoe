# 当前 E10 修改下的 E2/E3 兼容性资格实验

> **2026-08-31 更正：** 本文第 5、6 节把 E3 的故障解释为 cold39
> 操作破坏相邻 hot40 页表，这一解释已被后续驻留级诊断推翻；E2 的故障地址
> 也不在任何专家 Parameter VA 内。最终根因、修复和复验见
> `RESULTS_E2_E3_COMPATIBILITY_ROOT_CAUSE_AND_FIX_20260831.md`。本文保留作为
> 首次失败的原始记录，不再作为根因结论引用。

日期：2026-08-30（Asia/Shanghai）

## 1. 结论

当前 E10 修改与旧 E2/E3 路径存在兼容性冲突。

- E2 唯一一轮在 C16 首次正式计算时发生 CUDA illegal memory access；C1、C4、C8 成功。
- E3 唯一一轮在第 16 个 run，即 `long_generation/B8/C` 的初始化 warmup 时发生相同类型故障；此前 15 个 run 成功。
- 两次内核均记录 `Xid 31` 和 `FAULT_PDE ACCESS_TYPE_VIRT_READ`。
- 两项实验均未重试。E3 在首个错误后按 `STOP_ON_CUSTOM_ERROR=1` 停止，未把剩余 run 伪装成已完成。
- 异常清理成功，最终恢复 stock UVM，GPU 占用 0 MiB；本轮没有出现 UVM 锁死。

所以当前 E10 结果本身不受本轮改写，但不能认为当前工作树向后兼容旧 E2/E3。修复共享页表路径并重新完成兼容性门禁前，不应基于当前代码重采正式 E2/E3 数据。

## 2. 使用的冻结复现协议

- E2：`REPRODUCE_E2_DIRECT_BATCH_PURE_PHASE_FENCE_OFF.md`
- E3：`REPRODUCE_E3_GRAPH_SINGLE_ROUND_B1_B8_NO_HTTP.md`
- 每项只启动一轮，失败不重试。
- E10 专属动态 KV、CPU shadow、plan class 与 full dual-view 均未在 E2/E3 中启用。
- 当前 custom UVM SHA-256：`f09ba38db9983e3e225a9d7babd4d241acfd438ac69dff40db98c505db9471ea`。

旧成功参考所用模块不同：

- E2 旧成功模块：`e0fd8ea7a714c36f6759d6de00f0d93a93f1f471a7674e6cc87ee297e9a630ac`
- E3 旧成功模块：`c966620a0148aaa859f64ea1c874d87338fee2d5a64f6f905d7eab70a2e63bef`

fixture、模型和 E2 环境配置的冻结 hash 未改变；runner 与 custom UVM hash 已改变，因此本轮是兼容性检查，不是旧结果的统计重复。

## 3. E2 唯一一轮

Artifact：

`artifacts/e2-direct-batch-pure-phase-fence-off-20260830-current-e10-compat-attempt-01/`

| 并发 | TTFT | TPOT | E2E | 状态 |
|---:|---:|---:|---:|---|
| C1 | 391.25 ms | 22.72 ms | 1095.49 ms | 成功 |
| C4 | 404.24 ms | 44.06 ms | 1770.24 ms | 成功 |
| C8 | 405.42 ms | 57.02 ms | 2173.13 ms | 成功 |
| C16 | — | — | — | 首次正式计算失败 |

结构检查确认 C1/C4/C8 都在第一次 `EngineCore.step()` 前完整入队，未经过 HTTP/client，fence 确实关闭。C16 失败记录为：

```text
Xid 31
MMU Fault ... faulted @ 0x8255_00dc3000
FAULT_PDE ACCESS_TYPE_VIRT_READ
```

`verification.json` 为 `status=error`。`four_cells=false`、`result_status_ok=false`、`no_fatal_log_pattern=false`；不能将前三个 cell 当成完整 E2 复现成功。

运行时记录还确认：

- `static_alias_dual_view_enable=0`
- `static_alias_dual_view_require_full=0`
- PARK/RESUME 返回 `rm_status=0`
- `dual_view_full_coverage=false`

因此不是 E10 双表快路径被误开，而是旧 PARK/RESUME 共享页表路径在当前模块下产生了缺失目录。

## 4. E3 唯一一轮

Artifact：

`artifacts/e3-graph-repro-20260830-current-e10-compat-attempt-01/`

冻结协议期望 10 个逻辑 cell、24 个独立 run。本轮实际状态：

| 项目 | 数量 |
|---|---:|
| 期望 run | 24 |
| 成功 run | 15 |
| 失败 run | 1 |
| 因 fail-fast 未运行 | 8 |
| 完整成功逻辑 cell | 5 |
| 失败逻辑 cell | 1 |
| 未开始逻辑 cell | 4 |

已完整成功：

- short/B1：G、B、C
- short/B8：G、B、C
- ordinary/B1：G、B、C
- ordinary/B8：G、B、C
- long/B1：G、B、C

失败发生于 `long/B8/C`，该 cell 是 `remote_only + FULL_DECODE_ONLY`，在模型加载、静态 alias 建立/销毁、冷专家远程读取关系建立和驻留 reconciliation 后的首次 warmup 计算中失败。尚未进入正式测量，`result.json` 保持 `status=loading`，进程返回 134。

内核记录：

```text
Xid 31
MMU Fault ... faulted @ 0x7d6f_f1800000
FAULT_PDE ACCESS_TYPE_VIRT_READ
```

由于协议设置 `STOP_ON_CUSTOM_ERROR=1`，long/B8 的其余 arm 以及 medium/RAG 组未运行。这符合“失败不重试”，但意味着本轮不能生成 E3 聚合 `status=ok`，也不能报告完整 E3 性能矩阵。

## 5. E3 故障地址定位

本次 E3 提供了明确的相邻页证据。第一层 `w13_weight` Parameter：

```text
Parameter base                 0x7d6f_ea000000
cold expert 39 range          0x7d6f_f1500000 + 0x00300000
cold expert 39 exclusive end  0x7d6f_f1800000
hot expert 40 start           0x7d6f_f1800000
GPU fault address             0x7d6f_f1800000
```

也就是说，GPU 不是在访问已卸载的 cold expert 39 时失败，而是在访问紧邻它的 hot expert 40 的第一个字节时失败。

驱动对 cold range 的 `PREPARE_REMOTE_READ_AUDITED` 操作全部返回成功，驻留检查也只覆盖 cold range；随后 GPU 访问相邻 hot range 时发现上层页目录不存在。这证明当前操作在 2 MiB 边界处破坏了目标范围之外的映射，而现有 audit 没有检查相邻保留页面。

从本轮证据可以确定故障位于共享的以下阶段之一：

1. 销毁 static alias 并恢复普通 Parameter 映射；
2. 将 cold range 改为 CPU 远程读取时拆分/合并大页；
3. 两者之间对同一 2 MiB 页目录的状态交接。

当前日志不足以在 1 和 2 之间唯一归因；不能无证据指定某一条写页表语句。但故障地址、旧版本成功结果和当前模块 hash 变化共同证明这不是请求内容错误，也不是 E3 原协议必然失败，而是当前共享页表实现的回归。

## 6. 与 E10 修改的关系

E10 为 CPU shadow、局部 2 MiB 拆分、禁止重新合并和双页表切换修改了通用 VA-block/PTE 辅助函数。即使 E2/E3 没有打开 full dual-view，这些旧路径仍会调用同一组底层拆分、清除、恢复与远程映射函数。

因此正确的兼容性边界不能只检查环境开关，还必须保证：

- E10 专属的“禁止 2 MiB 合并”和 partial-shadow 规则只影响 shadow transaction；
- legacy static-alias destroy/PARK/RESUME 使用与旧版本相同的普通恢复语义；
- 修改一个 cold 子范围后，验证同一 2 MiB 块中未修改 hot 页仍有有效映射；
- cold-range audit 不能只检查目标范围，还应检查被拆分父目录内的保留页。

## 7. 清理与后续门禁

两轮结束后均满足：

- compute process 为空；
- GPU memory used 为 0 MiB；
- stock `nvidia_uvm.ko` 已恢复；
- 没有内核空指针、UVM 锁死或必须重启的情况。

建议先构造一个不启动 vLLM 的最小回归测试：同一 2 MiB 块内，前半/部分页切换为 CPU 远程读取，后半页保持 GPU；切换后分别读目标页和紧邻保留页。修复并通过该测试后，再分别运行 E2 C16 和 E3 long/B8/C 定向门禁，最后才重跑完整 E2/E3。本报告未授权这些重试，因此没有执行。
