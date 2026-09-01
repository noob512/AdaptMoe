# E10 单档位完整双页表端到端实验（2026-08-30）

## 结论

按授权重新执行了一次单档位 `cold70` 端到端实验，没有失败后重试，也没有进行统计
重复。本轮关闭只用于 qualification 的热专家 CUDA 采样读取门，同时保留完整双页表
覆盖证书、驱动返回检查、输出检查、Graph 检查、内核日志检查和资源清理检查。

实验成功，`runner status=ok`、`verifier status=ok`。首次 Prefill→Decode 完整阶段
切换为 33.053 ms，Decode→Prefill 为 18.740 ms，再次 Prefill→Decode 为
29.450 ms。结果证明上一轮的 815.486 ms 并非双页表切换本身；去除 qualification
采样后，请求关键路径确实降到了约 19--33 ms。

## 配置

- 模型：Qwen3-30B-A3B-FP8，单 GPU。
- 专家策略：cold-frequency。
- 页表档位：只建立 `cold70` 的 Prefill/Decode 双表，不建立 cold80/cold89。
- cold70：48 层 × 每层 70 个专家，共 3,360 个 layer-expert。
- 冷专家权重：15,854,469,120 bytes。
- 完整双表：9,487 个 2 MiB 父入口，覆盖率 100%。
- KV：固定约 2 GiB，不进行档位切换。
- 热专家 CUDA-read qualification gate：关闭。
- runtime cold-read gate、post-map residency audit、hot residency audit：关闭。
- 自动迁移相关策略保持关闭；驱动错误保护和 fail-closed 检查保留。

运行产物：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e10-single-tier-full-dual-view-20260830-attempt-11-e2e-no-hot-read-gate
```

## 端到端结果

| 指标 | 结果 |
|---|---:|
| Prefill | 333.543 ms |
| 首次 Prefill→Decode 切换 | 33.053 ms |
| 首次 Decode 计算（不含切换） | 30.736 ms |
| 首次 Decode E2E | 63.788 ms |
| steady Decode 中位 | 15.618 ms |
| steady Decode 平均 | 16.008 ms |
| steady Decode 吞吐 | 64.029 token/s |
| 请求 cycle E2E | 785.266 ms |

第二个请求执行一次 Decode→Prefill→Decode roundtrip：

| 阶段 | 阶段切换 | 包含计算的 step |
|---|---:|---:|
| Decode→Prefill | 18.740 ms | 351.916 ms |
| Prefill→Decode | 29.450 ms | 60.158 ms |

扣除切换后，roundtrip Prefill 计算为约 333.175 ms，Decode 计算为约
30.709 ms，与第一个请求一致。roundtrip 总时间为 470.836 ms。

## 双页表切换内部数据

| 方向 | 父入口 | 页表阶段 | ioctl | 完整外层切换 | CPU range-map ioctl |
|---|---:|---:|---:|---:|---:|
| Prefill→Decode（主请求） | 9,487 | 15.739 ms | 32.403 ms | 33.053 ms | 0 |
| Decode→Prefill（roundtrip） | 9,487 | 14.276 ms | 18.421 ms | 18.740 ms | 0 |
| Prefill→Decode（roundtrip） | 9,487 | 13.394 ms | 28.842 ms | 29.450 ms | 0 |

所有切换均为：

- `dual_view_full_coverage=true`；
- `dual_view_block_count=9487`；
- `rm_status=0`；
- 每次报告 149 个地址缓存失效批次；
- `remote_map_operation=prebuilt_full_dual_view`；
- Decode 的 6,720-range CPU mapping ioctl 数为 0；
- Prefill 没有重新安装 30,240 条消费者关系对应的 4 KiB 叶项。

## 与旧路径对比

旧路径取同为 cold70、2 GiB KV、没有档位 rebalance 的 attempt-13 单轮结果。两轮均为
qualification，差值用于验证机制，不代表多 seed 统计置信区间。

| 指标 | 旧路径 attempt-13 | 本轮完整双表 | 下降 |
|---|---:|---:|---:|
| Prefill→Decode 完整切换 | 296.09 ms | 33.053 ms | 88.84% |
| 首次 Decode E2E | 350.10 ms | 63.788 ms | 81.78% |
| 请求 cycle E2E | 1,229.80 ms | 785.266 ms | 36.15% |
| Prefill | 343.45 ms | 333.543 ms | 2.88% |

页表机制的直接对比：

| 操作 | 旧路径 | 本轮 | 下降 |
|---|---:|---:|---:|
| Decode PARK 驱动 ioctl | 139.51 ms | 32.403 ms | 76.77% |
| Prefill RESUME ioctl | 251.01 ms | 18.421 ms | 92.66% |
| Prefill RESUME 页表阶段 | 241.22 ms | 14.276 ms | 94.08% |

## 与上一轮含 qualification gate 的结果对比

上一轮使用相同完整双表，但在每次切换外围按 64 KiB 粒度读取约 20 万个热专家
采样点。本轮关闭该诊断后的变化为：

| 指标 | 含采样门 | 本轮 | 下降 |
|---|---:|---:|---:|
| 首次 Prefill→Decode 外层切换 | 815.486 ms | 33.053 ms | 95.95% |
| 首次 Decode E2E | 854.408 ms | 63.788 ms | 92.53% |
| 请求 cycle E2E | 1,613.370 ms | 785.266 ms | 51.33% |

这部分下降来自移除 qualification 诊断，不应重复算作双页表机制收益。其意义是确认：
此前约 778 ms 额外时间确实完全位于页表切换之外，关闭诊断后没有出现新的隐性等待。

## 初始化成本

双页表把工作提前到模型启动阶段：

| 工作 | 时间 |
|---|---:|
| 模型加载总时间 | 47.471 s |
| 初次 SEAL（包含首次双表准备） | 275.602 ms |
| 四次初始化双表刷新合计 | 1,071.768 ms |
| 驱动累计五次双表准备/刷新内部时间 | 1,337.378 ms |

四次刷新分别为 272.616、264.628、268.624、265.900 ms。它们用于在 KV/Graph
初始化造成普通页物理位置变化后更新预建表，不在正常 Prefill/Decode 请求切换路径中。

## 正确性和清理

- runner 返回码：0；kernel log 收集返回码：0。
- verifier：`status=ok`，`errors=[]`。
- 主请求输出 hash 与上一轮完全一致：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- roundtrip 输出 hash 也与上一轮一致：
  `5d4381e10ea7ad60286860fade0d86c7263054b3b8e4a6c90f611b533f9b1837`。
- CUDA Graph 捕获数从模型加载后到结束均为 2，无重捕获。
- 无 Xid、MMU fault、NULL pointer、general protection fault 或内核 BUG。
- 两次 PARK、一次 RESUME 均成功，没有 range-map 回退。
- 最终恢复 stock UVM，GPU memory used=0 MiB，无计算进程和 UVM client。

## 结论边界

本轮是一次端到端 qualification，不是统计重复实验。它证明单档位完整双页表已经把
真实 Prefill/Decode 阶段切换降至 18.74--33.05 ms；尚未验证多个档位的六套页表，
也没有包含 cold70→cold80→cold89 或 KV 2→4→6 GiB rebalance。

## 产物

- 原始结果：
  `artifacts/e10-single-tier-full-dual-view-20260830-attempt-11-e2e-no-hot-read-gate/sequence/result.json`
- 机器校验：
  `artifacts/e10-single-tier-full-dual-view-20260830-attempt-11-e2e-no-hot-read-gate/verification.json`
- 内核日志：
  `artifacts/e10-single-tier-full-dual-view-20260830-attempt-11-e2e-no-hot-read-gate/sequence/kernel.since.log`
