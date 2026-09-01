# E10 同档位 CPU/GPU 双页表父入口切换成功 Qualification（2026-08-30）

## 结论

同一专家档位的两套访问方案已经建立并在一次完整 E10 GPU qualification 中实际
切换成功：

- Decode 方案：Parameter 虚拟地址指向 CPU shadow；
- Prefill 方案：同一 Parameter 虚拟地址指向 GPU slot cache；
- 对满足条件的完整 2 MiB 区域，运行时只替换父页表入口，不再重新安装该区域内的
  512 个 4 KiB 叶级入口；
- 每个父入口的发布是原子的。完整模型切换以 64 个父入口为一批，在 scheduler 的
  GPU 静默边界依次发布，并不是用一条指令同时替换整个模型的数千个父入口。

优化有效。cold80 的 34,560 段 Prefill 恢复中，3,840 个 2 MiB 区域命中双页表，
页表阶段由 459.224 ms 降至 276.774 ms，下降 39.73%。对应 6 GiB Prefill E2E
由 1178.88 ms 降至 1031.47 ms，下降 12.50%。

本轮是唯一一次完整成功运行，没有重试和统计重复；结果是 qualification，不是多轮
统计结论。

## 两级验证

### 最小父入口切换探针

先使用独立 2 MiB managed 区域确认两套页表确实读取不同的物理副本：

| 项目 | 结果 |
|---|---:|
| 底层页粒度 | 4 KiB |
| 双页表覆盖 | 1 个完整 2 MiB 区域 |
| GPU→CPU 方案切换 | 0.755 ms |
| CPU→GPU 方案切换 | 0.326 ms |
| 驱动内部父入口写入 | 约 0.013 ms/方向 |
| 父入口写入数 | 2 |
| TLB 清理批次 | 2 |

CPU 方案读到预置值 29，第一次 GPU 方案读到 73，更新 GPU cache 后再次读到 91。
这排除了“计数命中但 GPU 仍读取旧物理页”的假成功。

产物：

`experiments/uvm_static_alias_original_va/artifacts/dual-parent-switch-20260830-attempt-02/result.json`

### 完整 E10

完整实验继续使用 cold-frequency 策略、2→4→6 GiB KV 档位、真实 backup 队列压力，
并保持关闭预测、自动预取、access-counter 迁移和 thrashing 迁移。

运行目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-07-dual-parent-switch-qualification`

| 项目 | 结果 |
|---|---:|
| 完整序列 | 1 |
| 每档请求 | 1 |
| 统计重复 | 0 |
| verifier | `status=ok` |
| 专家档位 | cold70 → cold80 → cold89 |
| KV 档位 | 2 → 4 → 6 GiB |
| CUDA Graph 捕获 | 6，运行期间无重捕获 |
| kernel Xid/MMU fault | 0 / 0 |

## 实际覆盖范围

cold80 有 80 × 48 = 3,840 个被选中的专家。每个专家权重由 9 个 512 KiB 段组成；
当前地址布局下，每个专家只有连续且对齐的 4 段可以组成一个完整 2 MiB 区域：

```text
每个 cold80 专家：
4 × 512 KiB → 1 个双页表父入口
5 × 512 KiB → 继续使用原逐段路径
```

因此 cold80 实际为：

| 路径 | 操作规模 | 覆盖权重 |
|---|---:|---:|
| 双页表父入口切换 | 3,840 个 2 MiB 父入口，60 批 | 7,680 MiB |
| 原逐段路径 | 19,200 个 512 KiB 段 | 9,600 MiB |
| 合计 | 原始 34,560 段 | 17,280 MiB |

双页表覆盖 44.44% 的 cold80 权重字节。因此本轮降低而未消除 459 ms，符合实际
覆盖范围。

全轮驱动计数为：

| 计数 | 数值 |
|---|---:|
| 最终 prepared blocks | 4,272 |
| switch calls | 12 |
| 父入口写入 | 45,888 |
| TLB 清理批次 | 720 |
| prepare 累计时间 | 104.42 ms |
| switch 累计时间 | 95.54 ms |

4,272 等于 cold89 的 89 × 48，说明档位增加后新增专家也成功准备了双页表。

## 页表切换性能

对照是前一轮未启用双页表、但已经完成目录检查优化的 attempt-05。两轮都只运行
一次，下面的差值用于 qualification 定位，不代表统计置信区间。

| 操作 | attempt-05 | 双页表 attempt-07 | 改善 |
|---|---:|---:|---:|
| cold70 Decode PARK 完整切换 | 383.33 ms | 283.45 ms | -26.06% |
| cold70 首次 Decode E2E | 422.42 ms | 321.86 ms | -23.81% |
| cold80 Prefill RESUME 完整切换 | 402.27 ms | 251.74 ms | -37.42% |
| cold80 RESUME ioctl | 401.81 ms | 251.33 ms | -37.45% |
| cold80 RESUME 页表阶段 | 391.26 ms | 241.82 ms | -38.20% |
| cold80 Decode PARK 完整切换 | 438.13 ms | 324.81 ms | -25.86% |
| cold80 PARK ioctl | 328.96 ms | 222.04 ms | -32.50% |
| cold89 前的 cold80 RESUME 完整切换 | 472.24 ms | 288.49 ms | -38.91% |
| cold89 前的 cold80 RESUME ioctl | 471.78 ms | 288.06 ms | -38.94% |
| cold89 前的 cold80 RESUME 页表阶段 | 459.22 ms | 276.77 ms | -39.73% |
| cold89 Decode PARK 完整切换 | 487.41 ms | 361.70 ms | -25.79% |
| cold89 PARK ioctl | 359.53 ms | 240.25 ms | -33.18% |

RESUME 的收益大于完整 Decode 切换的收益，是因为 Decode 切换在 PARK 之外还包含
建立 CPU 直接读取关系；后者没有被双页表完全替代。

## 端到端结果

| KV 档 | Prefill E2E | 首次 Decode 切换 | steady Decode 中位 | 请求 E2E |
|---|---:|---:|---:|---:|
| 2 GiB | 337.29 ms | 283.45 ms | 19.34 ms | 1136.17 ms |
| 4 GiB | 940.71 ms | 324.81 ms | 23.29 ms | 1865.78 ms |
| 6 GiB | 1031.47 ms | 361.70 ms | 26.73 ms | 2071.90 ms |

与 attempt-05 对比：

| 指标 | attempt-05 | 双页表 attempt-07 | 改善 |
|---|---:|---:|---:|
| 4 GiB Prefill E2E | 1082.10 ms | 940.71 ms | -13.07% |
| 6 GiB Prefill E2E | 1178.88 ms | 1031.47 ms | -12.50% |
| 4 GiB 请求 E2E | 2121.81 ms | 1865.78 ms | -12.07% |
| 6 GiB 请求 E2E | 2344.61 ms | 2071.90 ms | -11.63% |

steady Decode 中位数与旧轮基本一致，说明收益来自阶段切换缩短，而不是改变 Decode
kernel 的稳态执行方式。

## 正确性与清理

- 三个档位请求全部完成，输出 token hash 为
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`；
- verifier 为 `status=ok`；
- Graph 捕获数保持为预期的 6，没有因页表切换重捕获；
- 无 Xid、MMU fault 和 kernel fault；
- session 最终为 0，cleanup errors 为空；
- 恢复 stock UVM，最终 GPU 显存为 0 MiB。

机器可读证据：

- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-07-dual-parent-switch-qualification/verification.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-07-dual-parent-switch-qualification/protocol.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-07-dual-parent-switch-qualification/sequence/result.json`

## 当前限制与下一步

这轮已经证明“同一档位预建 CPU/GPU 两套页表并切换父入口”能够在完整模型中生效，
但尚未做到全部专家权重都走父入口切换：

1. 当前只处理完整、对齐且内部状态兼容的 2 MiB 区域；
2. 每个专家剩余 5 个 512 KiB 段仍重新安装叶级页表；
3. 完整档位仍需分几十批发布父入口，并清理地址缓存；
4. 双页表准备累计增加了 104.42 ms 生命周期开销，其中档位新增时的一部分仍位于
   rebalance 路径。

下一步最直接的目标是让剩余非完整 2 MiB 区域也能预建两套子表，或调整专家缓存
布局使每个专家的 4.5 MiB 权重组合成更多完整 2 MiB 区域。这样才能继续削减剩余的
约 277 ms RESUME 页表阶段。
