# E10 当前档位双页表重建 Qualification（2026-08-30）

## 结论

已实现“只保留当前专家档位的 Prefill/Decode 两套页表，并在 KV 档位变化后重建”流程，
并按要求只运行一次完整的 2→4→6 GiB E10，没有重试或统计重复。本轮 runner 和
verifier 均为 `status=ok`。

运行顺序实际为：

```text
恢复上一档 Prefill 表
→ 更新 cold 专家档位，但延后双表刷新
→ 调整 KV 的 GPU/CPU 物理布局
→ 为当前 cold 档位重建两套表并验证完整覆盖
→ 执行 Prefill
→ 首次 Decode 只切换 2 MiB 父入口
```

两次运行时重建分别耗时 350.192 ms 和 362.443 ms。重建完成后，
Decode→Prefill 切换只需 22.237/26.226 ms，Prefill→Decode 切换只需
44.349/51.140 ms；Decode 不再执行逐范围 CPU 映射，`ioctl_calls=0`。

## 实现

1. MAP ABI 新增 `DEFER_DUAL_REFRESH`，只允许与 scheduler 静默点的差量追加一起使用。
2. cold70→cold80/cold80→cold89 差量提交成功后不再立即重建双表，只使旧 recipe 失效。
3. KV 物理迁移和 CUDA 同步成功后，worker 调用 `SEAL(REFRESH_DUAL)`，按最终物理位置
   重建当前档位的 CPU/GPU 两张表。
4. 完整覆盖模式下，refresh 完成前若误入 PARK，驱动返回
   `NV_ERR_INVALID_STATE`，禁止退回旧的逐叶页表路径。
5. refresh 失败会使 resize 失败并 fail-closed，不会继续执行本轮 Prefill。

本方案没有同时保存 cold70/cold80/cold89 六套表；任一时刻只由当前档位持有两套表。

## 实验配置

- 模型：Qwen3-30B-A3B-FP8，单 GPU。
- 专家策略：cold-frequency，档位为 cold70→cold80→cold89。
- KV：初始化保留 8 GiB 稳定 VA，GPU 实际前缀由 backup 队列压力从 2→4→6 GiB。
- 不使用预测；每档一个请求，5 个 Decode warmup step，16 个 measured step。
- CPU immutable shadow、预生成切档指令、quiesced delta、完整双页表均启用。
- access-counter migration、自动 prefetch、thrashing 和运行时全量驻留审计均关闭。
- 完整序列数 1，统计重复数 0。

## 端到端结果

| KV 档 | Prefill E2E | 首次 Decode 页表切换 | 首次 Decode E2E | steady Decode 中位 | 请求 E2E |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 337.679 ms | 39.528 ms | 93.031 ms | 19.719 ms | 962.711 ms |
| 4 GiB | 1077.583 ms | 44.349 ms | 108.265 ms | 23.376 ms | 1769.464 ms |
| 6 GiB | 1150.369 ms | 51.140 ms | 119.050 ms | 26.347 ms | 1926.914 ms |

2 GiB 的 Prefill 不包含档位 rebalance。4/6 GiB Prefill 包含压力触发、专家档位更新、
KV 迁移和当前档位双表重建。

## Rebalance 分解

| 工作 | 2→4 GiB | 4→6 GiB |
|---|---:|---:|
| 恢复上一档 Prefill 双表 | 22.237 ms | 26.226 ms |
| 专家档位更新 | 210.092 ms | 197.273 ms |
| Graph 类别/证书状态切换 | 10.148 ms | 9.453 ms |
| KV 物理布局调整与同步 | 89.860 ms | 119.217 ms |
| 重建当前档位 Prefill/Decode 双表 | 350.192 ms | 362.443 ms |
| 合计（含恢复上一档） | 682.529 ms | 714.611 ms |
| runner 报告的 rebalance（不重复计入恢复） | 660.291 ms | 688.386 ms |

当前最大的 rebalance 开销已经是双表重建，占不含旧表恢复的 rebalance 约 53%。它被
放在 KV 档位变化时支付，不再出现在之后每次 Prefill/Decode 阶段切换中。

专家档位更新内部：

| 工作 | cold70→cold80 | cold80→cold89 |
|---|---:|---:|
| backing 预分配 | 20.684 ms | 19.342 ms |
| 差量追加完整外层 | 121.040 ms | 113.138 ms |
| 其中驱动 ioctl | 95.488 ms | 90.619 ms |
| CPU shadow 切换及状态维护等剩余部分 | 68.368 ms | 64.793 ms |
| 专家档位更新合计 | 210.092 ms | 197.273 ms |

CPU shadow 直接释放 2,160 MiB 和 1,944 MiB GPU 专家页，没有 GPU→CPU 权重复制；
两次差量均明确返回 `dual_view_refresh_deferred=true`，没有在专家切档中提前重复重建。

## 双表重建与阶段切换

| 当前/旧档位 | 操作 | 完整外层 | 驱动 ioctl | 父入口页表阶段 | 父入口数 | CPU range-map ioctl |
|---|---|---:|---:|---:|---:|---:|
| cold70 | Prefill→Decode | 39.528 ms | 38.666 ms | 16.834 ms | 9,487 | 0 |
| cold70 | Decode→Prefill | 22.237 ms | 21.836 ms | 16.147 ms | 9,487 | — |
| cold80 | Prefill→Decode | 44.349 ms | 43.388 ms | 17.769 ms | 10,476 | 0 |
| cold80 | Decode→Prefill | 26.226 ms | 25.769 ms | 18.938 ms | 10,476 | — |
| cold89 | Prefill→Decode | 51.140 ms | 50.311 ms | 19.358 ms | 11,299 | 0 |

每次切换均为 `dual_view_full_coverage=true`，没有安装数万条 4 KiB 叶级页表，也没有
回退到逐范围 CPU remote-map。

## 与完整双表前的 E10 对比

旧结果使用目录检查优化后的单轮 qualification；以下仍是单轮对单轮的机制比较，
不是统计结论。

| KV 档 | 指标 | 旧路径 | 本轮 | 下降 |
|---|---|---:|---:|---:|
| 2 GiB | 首次 Decode 切换 | 383.33 ms | 39.528 ms | 89.69% |
| 4 GiB | 首次 Decode 切换 | 438.13 ms | 44.349 ms | 89.88% |
| 6 GiB | 首次 Decode 切换 | 487.41 ms | 51.140 ms | 89.51% |
| 2 GiB | 请求 E2E | 1236.37 ms | 962.711 ms | 22.13% |
| 4 GiB | 请求 E2E | 2121.81 ms | 1769.464 ms | 16.61% |
| 6 GiB | 请求 E2E | 2344.61 ms | 1926.914 ms | 17.82% |

重建成本使 2→4 和 4→6 rebalance 本身只分别从 687.14/743.64 ms 降到
682.53/714.61 ms；主要收益来自后续 Prefill/Decode 不再全量拆装页表。

## 正确性与清理

- verifier：`status=ok`，完成序列 1，统计重复 0。
- 三档输出 token hash 完全一致：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- CUDA Graph capture 计数保持 6，无重捕获。
- 双表 prepare/refresh 12 次，skipped block 为 0；12 次父入口切换均成功。
- 无 Xid、MMU Fault、NULL pointer、kernel BUG 或 general protection fault。
- 相关 Python 测试 173 项通过；内核模块完整编译通过。
- 实验结束后恢复 stock UVM，GPU memory used=0 MiB，无计算进程。

## 产物

- 原始结果：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-14-current-tier-dual-rebuild-qualification/sequence/result.json`
- 机器校验：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-14-current-tier-dual-rebuild-qualification/verification.json`
- 内核日志：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-14-current-tier-dual-rebuild-qualification/kernel.since.log`
