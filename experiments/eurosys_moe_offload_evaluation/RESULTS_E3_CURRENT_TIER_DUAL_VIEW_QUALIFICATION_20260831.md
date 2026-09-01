# E3 当前固定档位双页表 Qualification（2026-08-31）

## 结论

已只为 E3 的 G/D arm 启用当前固定 cold 档位的 Prefill/Decode 双页表；C
始终远程读取，B 始终使用缓存路径。没有实现或预建多档页表。

唯一一轮完整 E3 成功：24/24 run、10/10 cell，聚合 verifier 为
`status=ok`、`errors=[]`。所有输出一致，G/D 双页表覆盖率均为 100%，没有
CPU range-map 回退、Graph 重捕获、Xid、PTE/PDE fault 或内核错误。实验结束后
恢复 stock UVM，GPU 占用 0 MiB。

正式结果目录：

```text
artifacts/e3-graph-repro-REPRO-20260831-attempt-16-current-tier-dual-view-full/
```

attempt-08 至 attempt-15 是实现和单 cell 调试记录，未拼入本轮正式结果。

## 实现

1. `run_formal_e3.sh` 只为 G/D 注入 full-Parameter CPU shadow、full dual view
   和 PARK/RESUME 环境；B/C 保持原环境。
2. G/D 每个独立冷进程只建立其当前 cold10、cold38 或 cold63 档位的两张表。
3. Prefill Graph 已永久持有 allocator pool 时，D 不再执行第二轮冗余
   all-remote 切换和 eager allocator-reserve warmup。
4. Graph 捕获后恢复普通热页可能生成与冷页不重叠的 64 KiB 映射。驱动刷新现在
   接受这种状态，重新物化两张 4 KiB 子表中的普通页，并重新发布当前 Prefill
   父入口；若 64 KiB 映射碰到冷页仍然 fail-fast。

## G/B/C 结果

单位为 ms。Decode 是 measured steady-step median；E2E 已包含 G 的首次
Prefill→Decode 切换。

| Workload | Batch | G E2E | B E2E | C E2E | G/B/C Decode | G 切换 | Winner |
|---|---:|---:|---:|---:|---:|---:|---|
| Short 128→32 | 1 | 477.41 | 1760.48 | 453.26 | 9.90 / 52.82 / 9.93 | 6.89 | C |
| Short 128→32 | 8 | 655.31 | 1949.81 | 615.22 | 15.34 / 58.34 / 15.23 | 6.46 | C |
| Ordinary 512→128 | 1 | 1422.76 | 6861.06 | 1380.22 | 9.65 / 52.93 / 9.74 | 6.52 | C |
| Ordinary 512→128 | 8 | 2565.57 | 7667.00 | 2570.63 | 18.02 / 58.84 / 18.01 | 7.88 | G |
| Medium 2048→128 | 1 | 1942.06 | 6923.11 | 2103.98 | 14.09 / 53.20 / 14.09 | 6.34 | G |
| Long 512→512 | 1 | 5684.80 | 27252.43 | 5623.16 | 10.50 / 52.92 / 10.48 | 6.49 | C |
| Long 512→512 | 8 | 9044.98 | 30147.56 | 8982.99 | 17.23 / 58.49 / 17.19 | 6.34 | C |

winner 分布为 C×5、G×2、B×0。G 的切换范围为 6.34--7.88 ms，中位数
6.49 ms；每个 G 进程切换 1,501 个 2 MiB 父入口，覆盖率 100%。

## D 大 Prefill 结果

| Workload | Batch | cold/层 | Prefill | P→D 切换 | Decode median | E2E | 父入口数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Medium 2048→128 | 8 | 38 | 649.92 | 23.87 | 41.79 | 6229.61 | 5,674 |
| RAG 8192→128 | 1 | 38 | 587.62 | 23.33 | 15.99 | 2880.52 | 5,674 |
| RAG 8192→128 | 8 | 63 | 3696.22 | 42.33 | 64.22 | 12111.95 | 8,746 |

D 三格均为两张 Graph，捕获计数在服务阶段保持 2→2。首次 Decode 不再安装
数千条 CPU 读取关系，`remote_range_map_ioctl_calls=0`。

## 与最近完整 E3 的对比

最近的同日完整参考是：

```text
artifacts/e3-graph-repro-REPRO-20260831-attempt-07-hybrid-deferred-park/
```

该参考尚未为 G/D 启用当前档位 full dual view。对 10 个相同 G/D 点逐点比较：

| 指标 | 差异中位数 | 最小 | 最大 |
|---|---:|---:|---:|
| Prefill | +0.24% | -9.82% | +6.45% |
| steady Decode median | +0.18% | -1.58% | +3.24% |
| Prefill→Decode 切换 | -96.09% | -96.58% | -95.40% |
| 请求 E2E | -7.03% | -24.74% | -1.56% |

其中：

- G 切换中位改善 96.13%，7 个 G 点的 E2E 全部下降，中位下降 6.91%；
- D 切换中位改善 96.05%，3 个 D 点的 E2E 全部下降，中位下降 7.15%；
- 未启用双页表的 B/C，E2E 相对参考的差异中位数仅 +0.23%，steady Decode
  差异中位数 +0.10%，说明 B/C 路径基本未被改变。

与 2026-08-28 的旧基线相比，G/D 切换差异中位数为 -90.19%。D 的 Prefill
差异为 -1.50% 至 +0.57%，E2E 为 -0.54% 至 +1.32%，基本保持。G 的
steady Decode 在本轮普遍偏慢，导致 G E2E 相对旧基线中位高 8.74%；同轮 C
也有相似 Decode 变慢，不能把这部分单轮波动归因于双页表。

## 修复前失败与本轮验证

D 会在初始 Graph capture 和 post-capture warmup 各执行一次 all-remote。第一轮
建立双页表后，热专家恢复会更新普通热页；旧实现第二次 PARK 或最终 refresh
仍按冻结前状态校验，返回 `NV_ERR_INVALID_STATE / rm_status=64`。

修复后：

- 第二轮冗余 all-remote 不再发生；
- 最终 refresh 能从当前热页驻留状态重建两张表；
- 冷页与普通热页仍保持不重叠约束；
- G/D 只通过父入口切换，CPU range-map ioctl 为 0；
- targeted pytest 5 项和每组 static preflight 8 项通过。

当前自定义 `nvidia-uvm.ko` SHA256：

```text
c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a
```

## 结果边界

本轮每个 arm 只有一个独立冷进程点，是 qualification，不是统计重复。winner
变化和几个百分点的性能差异不能解释为统计显著；正式论文结论仍需冻结模块 hash、
多 seed/多冷进程重复并报告 median、p95 和置信区间。

机器汇总：

```text
artifacts/e3-graph-repro-REPRO-20260831-attempt-16-current-tier-dual-view-full/summary/metrics.csv
artifacts/e3-graph-repro-REPRO-20260831-attempt-16-current-tier-dual-view-full/summary/verification.json
artifacts/e3-graph-repro-REPRO-20260831-attempt-16-current-tier-dual-view-full/summary/winners.csv
```
