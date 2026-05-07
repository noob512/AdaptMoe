# Stage L / Stage C UVM 综合实验报告

实验日期：2026-05-07  
模型：`Qwen/Qwen3-30B-A3B-FP8`  
主要负载：`prompts=4`，`output_len=256`，`request_rate=2`

## 1. 实验目的

本轮综合实验主要回答三个问题：

1. 当前 KV cache 相关优化是否能够降低 KV 缺页，并带来性能收益。
2. runtime scratch / temporary buffer 相关缺页是否是更主要的性能瓶颈。
3. 将 runtime scratch 临时缓冲区直接放到 GPU 上后，缺页和性能是否改善，以及哪些参数组合更有效。

本报告只总结实验结果和结论，不讨论代码实现细节。

## 2. 实验清单

| 实验 | run root | 目的 |
|---|---|---|
| Stage L fault/performance diagnosis | `/tmp/vllm_stage_l_fault_perf_diag_20260507_071018` | 对比 pure UVM、L5、L6、L7 的 KV fault 与性能 |
| Stage C scratch attention-only sweep | `/tmp/vllm_stage_c_scratch_device_direct_sweep_20260507_085908` | 只覆盖 `attention` 阶段，扫描 scratch device-direct 参数 |
| Stage C scratch 8MiB/16MiB confirm | `/tmp/vllm_stage_c_scratch_device_direct_sweep_20260507_133747` | 补测 `max_bytes=8MiB/16MiB` 是否优于 4MiB |
| Stage C scratch attention+moe sweep | `/tmp/vllm_stage_c_scratch_device_direct_sweep_20260507_154109` | 扩展覆盖 `attention + moe` 阶段，验证是否进一步减少 scratch fault |

## 3. Stage L / KV 实验结果

Stage L 诊断实验共 15 次运行，所有请求均成功。pure UVM baseline 的中位性能为：

| 指标 | baseline_pure_uvm |
|---|---:|
| output throughput | 16.60 tok/s |
| mean TPOT | 229.01 ms |
| mean TTFT | 1561.88 ms |

各 Stage L 变体中位结果如下：

| variant | failed | output_tok_s | mean_tpot_ms | mean_ttft_ms | kv_faults | kv_dedup_faults | offload | prefetch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_pure_uvm | 0 | 16.60 | 229.01 | 1561.88 | n/a | n/a | n/a | n/a |
| l5_metadata | 0 | 16.50 | 230.32 | 1572.50 | 11780 | 3009 | n/a | n/a |
| l6_decision_trace | 0 | 16.25 | 234.13 | 1618.96 | 12006 | 2997 | 97 | 94 |
| l6_real_prefetch | 0 | 16.44 | 230.75 | 1644.91 | 5808 | 1482 | 97 | 94 |
| l7_coordinator | 0 | 15.41 | 247.10 | 1644.37 | 5700 | 1424 | 97 | 94 |

### Stage L 结论

L6/L7 对 KV fault 有明确效果：

- `l5_metadata` 的 KV dedup fault 约为 `3009`。
- `l6_real_prefetch` 降至 `1482`。
- `l7_coordinator` 降至 `1424`。

也就是说，KV dedup fault 大约能够减少一半。

但端到端性能没有同步提升：

- `baseline_pure_uvm`：`16.60 tok/s`，`229.01 ms TPOT`
- `l6_real_prefetch`：`16.44 tok/s`，`230.75 ms TPOT`
- `l7_coordinator`：`15.41 tok/s`，`247.10 ms TPOT`

因此，KV fault 优化虽然功能有效，但在当前负载下不是主要性能收益来源。L7 coordinator 还引入了可见开销。

## 4. Stage C Scratch Attention-Only 参数扫描

attention-only sweep 共 15 组参数，所有 device-direct 运行均成功：

| 指标 | 结果 |
|---|---:|
| total_runs | 15 |
| successful_device_direct_runs | 15 |
| scratch fault reduction 中位数 | 25.11% |
| output throughput delta 中位数 | +12.50% |
| mean TPOT delta 中位数 | -12.04% |

最佳 fault 降低组合：

| rank | 参数组合 | scratch fault reduction | unknown fault delta | actual records | budget rejects | output delta | TPOT delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `budget=4MiB, max_bytes=4MiB` | 33.03% | -32.99% | 12384 | 0 | +12.50% | -12.04% |
| 2 | `budget=4MiB, max_bytes=1MiB` | 29.62% | -29.56% | 12336 | 0 | +15.38% | -14.09% |
| 3 | `budget=64MiB, max_bytes=1MiB` | 29.29% | -29.23% | 12336 | 0 | +29.32% | -23.41% |
| 4 | `budget=16MiB, max_bytes=1MiB` | 27.59% | -27.71% | 12432 | 0 | +25.50% | -20.15% |

### Attention-Only 结论

只覆盖 `attention` 阶段时，scratch fault 能下降约 25%-33%，性能通常也会提升。

但 attention-only 存在明显上限：

- device-direct 命中数约 `12336-12432`。
- 仍有约 `14970` 左右记录走 managed 路径。
- 最佳 scratch fault reduction 约 `33%`，没有超过 50% strong reduction 阈值。

参数趋势也很清晰：

- `max_bytes=256KiB` 太小，命中记录只有数百个，部分组合反而使 fault 和性能变差。
- `max_bytes=1MiB/4MiB` 稳定有效。
- `budget=4MiB` 已经足够覆盖 attention-only 的主要 live scratch。

## 5. Stage C Scratch 8MiB / 16MiB 补测

补测实验共 4 组，目标是判断 `max_bytes=8MiB/16MiB` 是否优于此前的 4MiB。

| rank | 参数组合 | scratch fault reduction | unknown fault delta | actual records | budget rejects | output delta | TPOT delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `budget=4MiB, max_bytes=8MiB` | 32.58% | -32.55% | 12384 | 0 | +17.47% | -15.88% |
| 2 | `budget=16MiB, max_bytes=8MiB` | 31.06% | -31.04% | 12384 | 0 | +15.77% | -15.24% |
| 3 | `budget=4MiB, max_bytes=16MiB` | 27.29% | -27.41% | 12432 | 0 | +6.63% | -6.90% |
| 4 | `budget=16MiB, max_bytes=16MiB` | 18.08% | -18.01% | 12384 | 0 | +36.04% | -27.34% |

### 8MiB / 16MiB 结论

虽然 gap2 地址范围约为 `14.5MiB`，但将单个 allocation 上限直接扩大到 `16MiB` 并没有带来更好的 fault 降低。

attention-only 下：

- `max_bytes=4MiB` 的最佳 fault reduction 为 `33.03%`。
- `max_bytes=8MiB` 的最佳 fault reduction 为 `32.58%`。
- `max_bytes=16MiB` 反而降至 `27.29%` 或 `18.08%`。

因此，gap2 大小不应直接等同于单个临时 allocation 的大小。attention-only 下更合理的候选是：

- `budget=4MiB, max_bytes=4MiB`
- `budget=4MiB, max_bytes=8MiB`

## 6. Stage C Scratch Attention + MoE 实验

attention+moe sweep 共 4 组，所有运行均成功，且全部超过 50% strong reduction 阈值。

| rank | 参数组合 | scratch fault reduction | unknown fault delta | actual records | budget rejects | output delta | TPOT delta |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `budget=16MiB, max_bytes=8MiB` | 89.21% | -89.21% | 25865 | 0 | +58.45% | -37.84% |
| 2 | `budget=16MiB, max_bytes=4MiB` | 88.45% | -88.47% | 25817 | 0 | +42.01% | -30.12% |
| 3 | `budget=4MiB, max_bytes=8MiB` | 87.62% | -87.64% | 25772 | 96 | +28.56% | -23.72% |
| 4 | `budget=4MiB, max_bytes=4MiB` | 87.54% | -87.56% | 25769 | 48 | +36.78% | -27.86% |

最佳组合：

```text
target_phases=enabled:attention,enabled:moe
budget=16MiB
max_bytes=8MiB
backend=cuda_malloc_async
```

对应关键指标：

| 指标 | baseline / trace | device-direct | 变化 |
|---|---:|---:|---:|
| gap faults | 4,289,610 | 463,046 | -89.21% |
| unknown faults | 4,295,332 | 463,363 | -89.21% |
| output throughput | 9.82 tok/s | 15.56 tok/s | +58.45% |
| mean TPOT | 390.27 ms | 242.61 ms | -37.84% |
| failed requests | n/a | 0 | 正常 |
| actual device-direct records | n/a | 25,865 | 生效 |
| budget rejects | n/a | 0 | 无拒绝 |

### Attention + MoE 结论

将覆盖阶段从 attention-only 扩展到 `attention + moe` 后，实验结果发生质变：

| 实验 | 最佳 scratch fault reduction | 最佳 actual records | managed 剩余记录 | 典型性能变化 |
|---|---:|---:|---:|---:|
| attention-only | 33.03% | 12,384 | 约 14,970 | 吞吐提升约 12%-33% |
| attention-only 8MiB confirm | 32.58% | 12,384 | 约 14,970 | 吞吐提升约 17% |
| attention+moe | 89.21% | 25,865 | 约 1,590 | 吞吐最高提升 58.45% |

这说明此前剩余的大量 scratch fault 主要来自 MoE 阶段。仅覆盖 attention 时，只能处理约一半主要 scratch 来源；加入 MoE 后，device-direct 命中数翻倍，剩余 managed scratch 大幅减少。

预算方面：

- `budget=4MiB` 已经可以取得约 `87%` fault reduction，但出现 `48-96` 个 budget rejects。
- `budget=16MiB` 没有 budget rejects，且获得最佳结果。
- attention+moe 下观测到的 peak live scratch 最高约 `10MiB`，因此 `16MiB` 比 `4MiB` 更稳妥。

## 7. 综合对比

| 实验方向 | 最佳 fault 改善 | 最佳性能变化 | 主要结论 |
|---|---:|---:|---|
| Stage L / KV | KV dedup fault 约减半 | 无稳定端到端提升，L7 变慢 | KV fault 不是当前主要性能瓶颈 |
| Stage C attention-only | scratch fault 最高 -33.03% | output 最高 +44.93%，但参数敏感 | attention scratch 有收益，但覆盖不足 |
| Stage C attention-only 8MiB/16MiB | scratch fault 最高 -32.58% | output 最高 +36.04% | 16MiB 单 allocation 上限没有更好 |
| Stage C attention+moe | scratch fault 最高 -89.21% | output 最高 +58.45%，TPOT 最多 -37.84% | 当前最有效方向 |

## 8. 最终结论

1. 当前实验中，最主要的性能收益来自 runtime scratch / temporary buffer 缺页减少，而不是 KV cache 缺页减少。

2. Stage L/KV 路径能够显著减少 KV fault，但在当前负载下 KV fault 占总 fault 比例较小，因此端到端性能没有稳定提升。

3. attention-only scratch device-direct 能减少约 25%-33% scratch fault，并带来一定性能收益，但覆盖范围不足。

4. gap2 地址范围约为 `14.5MiB`，但这不意味着单个 allocation 上限应直接设置为 `16MiB`。补测结果显示，attention-only 下 `max_bytes=16MiB` 不优于 `4MiB/8MiB`。

5. 将覆盖阶段扩展到 `attention + moe` 后，scratch fault reduction 提升到约 `87%-89%`，输出吞吐提升约 `28%-58%`，mean TPOT 降低约 `23%-38%`。这是当前最强、最明确的实验结果。

6. 当前最推荐的实验参数是：

```text
target_phases=enabled:attention,enabled:moe
backend=cuda_malloc_async
budget=16777216
max_bytes=8388608
pool_release_threshold=1048576
```

该组合在本轮实验中取得：

```text
scratch fault reduction = 89.21%
unknown fault delta     = -89.21%
output throughput delta = +58.45%
mean TPOT delta         = -37.84%
device_direct records   = 25865
budget rejects          = 0
failed requests         = 0
```

## 9. 后续建议

建议后续围绕最佳组合做稳定性验证：

```text
target_phases=enabled:attention,enabled:moe
budget=16MiB
max_bytes=8MiB
backend=cuda_malloc_async
```

推荐至少重复 3 次，观察：

- scratch fault reduction 是否稳定保持在 85% 以上；
- output throughput 是否稳定提升；
- mean TPOT 是否稳定下降；
- budget rejects 是否持续为 0；
- failed requests 是否持续为 0。

