# E10 当前档位双页表完整单轮实验（2026-08-30）

## 结论

按要求仅运行一轮完整的 2→4→6 GiB E10 序列，没有自动重试或统计重复。runner 与
verifier 均为 `status=ok`。

当前档位双页表流程在 cold70、cold80、cold89 三档均实现 100% 覆盖。两次 KV 档位
变化后只重建当前档位的 Prefill/Decode 双表；之后 Prefill→Decode 切换为
33.582--41.716 ms，Decode→Prefill 为 20.261--23.040 ms，没有逐范围 CPU
remote-map ioctl。

## 实验配置

- 模型：Qwen3-30B-A3B-FP8，单 GPU。
- 专家策略：cold-frequency，cold70→cold80→cold89。
- KV：预留 8 GiB 稳定 VA，GPU 物理前缀由 backup 队列压力从 2→4→6 GiB。
- 不使用预测。
- 每档一个请求；5 个 warmup Decode step，16 个 measured Decode step。
- 完整序列 1，统计重复 0，失败后重试 0。

## 端到端结果

| KV 档 | Prefill E2E | Prefill→Decode | 首次 Decode E2E | steady Decode 中位 | 请求 E2E |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 336.447 ms | 33.582 ms | 77.782 ms | 18.040 ms | 885.638 ms |
| 4 GiB | 941.454 ms | 36.569 ms | 86.311 ms | 22.016 ms | 1547.749 ms |
| 6 GiB | 1013.394 ms | 41.716 ms | 95.439 ms | 24.761 ms | 1699.961 ms |

4/6 GiB Prefill E2E 包含压力触发、专家切档、KV 物理布局更新、当前档位双表重建和
Prefill 计算。2 GiB Prefill 不包含运行时档位转换。

## KV 档位转换分解

| 工作 | 2→4 GiB | 4→6 GiB |
|---|---:|---:|
| 恢复上一档 Prefill 双表 | 20.261 ms | 23.040 ms |
| 专家档位更新 | 168.075 ms | 156.366 ms |
| Graph 类别/证书状态 | 7.953 ms | 7.461 ms |
| KV 物理布局与同步 | 80.782 ms | 113.432 ms |
| 重建当前档位双表 | 274.207 ms | 283.000 ms |
| 完整档位转换合计 | 551.279 ms | 583.299 ms |

双表重建仍是最大单项，占完整档位转换约 49%。纯 KV GPU/CPU 物理布局调整本身为
80.782/113.432 ms。

专家档位更新内部：

| 工作 | cold70→cold80 | cold80→cold89 |
|---|---:|---:|
| backing 预分配 | 16.516 ms | 15.474 ms |
| 差量追加外层 | 95.105 ms | 88.321 ms |
| 其中驱动 ioctl | 75.753 ms | 71.058 ms |
| CPU shadow及状态维护等剩余部分 | 56.454 ms | 52.571 ms |
| 合计 | 168.075 ms | 156.366 ms |

CPU shadow 路径直接释放对应专家显存，没有 GPU→CPU 权重复制；两次差量都返回
`dual_view_refresh_deferred=true`，随后只在 KV 布局完成后重建一次。

## Prefill/Decode 双表切换

| 档位 | 方向 | 完整切换 | 驱动 ioctl | 父入口阶段 | 父入口数 |
|---|---|---:|---:|---:|---:|
| cold70 | Prefill→Decode | 33.582 ms | 32.915 ms | 16.084 ms | 9,487 |
| cold70 | Decode→Prefill | 20.261 ms | 19.934 ms | 14.992 ms | 9,487 |
| cold80 | Prefill→Decode | 36.569 ms | 35.919 ms | 15.794 ms | 10,476 |
| cold80 | Decode→Prefill | 23.040 ms | 22.705 ms | 17.337 ms | 10,476 |
| cold89 | Prefill→Decode | 41.716 ms | 41.070 ms | 17.090 ms | 11,299 |

所有 Prefill→Decode 均为 `remote_map_operation=prebuilt_full_dual_view`、
`dual_view_full_coverage=true`、CPU range-map `ioctl_calls=0`。

## 正确性和清理

- 三档输出 hash 一致：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- CUDA Graph capture 保持 6，档位转换没有重捕获。
- 双表 skipped block 为 0；12 次父入口切换均成功。
- 无 Xid、MMU Fault、NULL pointer、kernel BUG 或 general protection fault。
- 实验结束后恢复 stock UVM，GPU memory used=0 MiB，无计算进程。

## 产物

- 原始结果：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-15-current-tier-dual-rebuild-full/sequence/result.json`
- 机器校验：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-15-current-tier-dual-rebuild-full/verification.json`
- 内核日志：
  `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-15-current-tier-dual-rebuild-full/kernel.since.log`
