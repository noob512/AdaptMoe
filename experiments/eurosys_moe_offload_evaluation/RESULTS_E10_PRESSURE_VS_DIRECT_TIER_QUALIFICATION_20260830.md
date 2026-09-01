# E10 压力自动切档与请求前直接切档对照（2026-08-30）

## 结论

两个 arm 均使用独立冷模型进程，各完整运行一次 2→4→6 GiB 序列，没有重试，两个
verifier 均为 `status=ok`。

本轮观测中，两种触发方式完成相同档位切换后，Prefill 和 Decode 性能一致到单样本运行噪声
范围：纯 Prefill 最大差异 2.70%，Prefill→Decode 最大差异 1.77%，steady Decode 中位数
最大差异 1.29%。相同的核心切档事务最大差异 1.47%，将完整切档都计入后，请求总时间最大
差异 0.94%。

因此，本轮支持“压力检测只改变切档发生时机，不改变切档后的 Prefill/Decode 执行路径和
性能”。但每个 arm 只有一个样本，不能声称统计等价；正式结论仍需多轮、交换 arm 顺序并
报告置信区间。

## 实验设计

共同配置：Qwen3-30B-A3B-FP8、B1、cold-frequency、cold70/cold80/cold89、固定 8 GiB
KV VA、2/4/6 GiB GPU KV 前缀、当前档位完整双页表、每档 5 个 warmup Decode step 和
16 个 measured Decode step。

唯一自变量是切档调用位置：

- `auto_pressure`：真实请求从 backup 队列取得 4 blocks，达到阈值后在 Prefill schedule 内
  自动调用切档。
- `direct_tier`：在请求进入 scheduler 前、系统空闲时，直接调用同一个目标档位事务。之后
  请求仍真实使用 4 个 backup blocks，但 autoscale 不再次执行切档。

两组均执行相同顺序：恢复上一档 Prefill 双表→追加专家差量→直接使用 CPU shadow 释放专家
显存→调整 KV 物理前缀→重建当前档位 Prefill/Decode 双表。

## 推理性能对比

下表的“纯 Prefill”在自动组中已经扣除请求内切档；direct 组的切档发生在请求前，因此就是
请求内 Prefill。差异为 `(direct-auto)/auto`。

| KV 档 | auto 纯 Prefill | direct 纯 Prefill | 差异 | auto steady Decode | direct steady Decode | 差异 |
|---|---:|---:|---:|---:|---:|---:|
| 2 GiB | 337.582 ms | 337.529 ms | -0.02% | 19.620 ms | 19.376 ms | -1.25% |
| 4 GiB | 394.732 ms | 384.078 ms | -2.70% | 23.621 ms | 23.316 ms | -1.29% |
| 6 GiB | 434.990 ms | 424.203 ms | -2.48% | 26.368 ms | 26.261 ms | -0.41% |

首次 Decode 和双表方向切换：

| KV 档 | auto Prefill→Decode | direct Prefill→Decode | 差异 | auto 首次 Decode E2E | direct 首次 Decode E2E |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 38.788 ms | 38.171 ms | -1.59% | 93.362 ms | 92.681 ms |
| 4 GiB | 42.213 ms | 42.961 ms | +1.77% | 103.637 ms | 104.478 ms |
| 6 GiB | 49.310 ms | 48.796 ms | -1.04% | 117.400 ms | 117.055 ms |

4/6 GiB 切档开始时还需把上一请求的 Decode 视图切回 Prefill 视图。该
Decode→Prefill 时间分别为 auto 21.876/22.977 ms，direct 22.496/23.520 ms，差异
2.84%/2.36%。

## 切档成本对比

`shared tier transaction` 使用两个 arm 共有的两个 RPC 计时边界；direct 的“外层切档”还
包含显式 API 包装、同步和状态读取。因此核心事务更适合判断两种触发方式是否执行同样工作。

| 转换 | auto 核心事务 | direct 核心事务 | 差异 | direct 外层完整调用 |
|---|---:|---:|---:|---:|
| 2→4 GiB | 680.634 ms | 674.826 ms | -0.85% | 682.340 ms |
| 4→6 GiB | 711.311 ms | 700.832 ms | -1.47% | 708.258 ms |

本轮主要分量也相近：

| 分量 | auto 2→4 | direct 2→4 | auto 4→6 | direct 4→6 |
|---|---:|---:|---:|---:|
| Decode→Prefill 双表恢复 | 21.876 ms | 22.496 ms | 22.977 ms | 23.520 ms |
| 专家档更新及状态管理 | 220.503 ms | 219.580 ms | 204.668 ms | 202.644 ms |
| KV 布局及 resize 其他工作 | 91.367 ms | 88.685 ms | 123.188 ms | 117.032 ms |
| 当前档位双表重建 | 346.889 ms | 344.065 ms | 360.478 ms | 357.636 ms |

自动组的请求 E2E 本身包含切档；direct 请求 E2E 不包含请求前切档，不能直接比较。把 direct
外层切档加回后，公平的总时间为：

| KV 档 | auto 请求（含切档） | direct 请求+切档 | 差异 |
|---|---:|---:|---:|
| 2 GiB | 963.957 ms | 958.199 ms | -0.60% |
| 4 GiB | 1760.179 ms | 1749.285 ms | -0.62% |
| 6 GiB | 1920.604 ms | 1902.588 ms | -0.94% |

## 路径和正确性证据

- 两个 arm 的三档目标均为 1364/2730/4095 blocks，专家档均为
  cold70/cold80/cold89。
- 每次请求都真实从 backup 队列取得 4 blocks；auto 组由阈值消费事件触发，direct 组在
  请求前已经提交目标，因此不二次切档。
- 两个 arm 的父入口工作量完全相同：12 次双表切换、125,048 次父入口写入、1,960 个 TLB
  批次、`skipped_blocks=0`，三档覆盖率均为 100%。
- 两组最终输出 hash 完全相同：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- 两组 CUDA Graph capture 均保持 6，最终 replay 均为 69，没有重捕获。
- 无 CPU range-map 回退、Xid、MMU Fault 或内核错误。
- 两个冷进程退出后均恢复 stock UVM；最终 GPU 为 0 MiB。

## 产物

对照根目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
e10-trigger-mode-comparison-20260830-attempt-01-qualification
```

主要文件：

- `comparison.json`：统一机器可读对比。
- `auto_pressure/verification.json`：压力触发 arm 验收。
- `direct_tier/verification.json`：直接切档 arm 验收。
- 两个 arm 的 `sequence/result.json`：逐步原始数据。
- 两个 arm 的 `kernel.since.log`：内核日志。

原始结果 SHA256：

```text
auto_pressure result.json       ec289a276776de506c02e2ebc598639ead3b752e74eb43f7c86a805d115e83aa
direct_tier result.json         bba0074c1caa9e4fb0dc8c820c84fef3f19559f4ffde115e719d3ccf4aaf832f
```

复现步骤见 `REPRODUCE_E10_CURRENT_TIER_DUAL_VIEW_FULL_RUN.md` 第 7 节。
