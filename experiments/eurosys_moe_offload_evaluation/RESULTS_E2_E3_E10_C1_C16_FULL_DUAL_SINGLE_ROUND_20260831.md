# E2、E3、E10 C1/C16 全覆盖双页表单轮结果

日期：2026-08-31（Asia/Shanghai）

## 结论

本轮从 `ours` 开始重新采集，没有重复运行 pinned-UVA、copy-on-forward，也没有运行
MoE-Infinity。E2 只使用 C1/C16；E3 只使用 B1/B16；E10 保留原 2→4→6 GiB
压力切档协议。

三个实验使用同一个 custom UVM 二进制：

```text
SHA256      c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a
srcversion  CD4A27464CD3E907C4EF493
```

| 实验 | 结果 | 双页表结论 |
|---|---|---|
| E2/C1 | 3/4 严格通过 | 4个请求都命中全覆盖双页表；Long/C1仅流式协议检查失败 |
| E2/C16 | 4/4 通过 | 4个场景均为全覆盖双页表，CPU range-map ioctl 为0 |
| E3/B1、B16 | 10/10 功能完成 | cold10/cold38/cold89当前档双页表，全部100%覆盖 |
| E10 | verifier `status=ok` | cold70→cold80→cold89切档后重建当前档双表 |

所有已运行单元均未出现 Xid、PTE/PDE fault、GPU page fault、OOM 或内核崩溃。最终
恢复 stock UVM，GPU 0 MiB。

### ours + Graph 与纯 pinned-UVA remote

本轮 `ours` 没有重复运行已经冻结的 pinned-UVA 基线。对比使用：

```text
baseline  artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-01/e2/
ours      artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-02-full-dual/e2/
```

比较对象的含义如下：

- `ours`：GPU 缓存 Prefill、CUDA Graph、全覆盖 Prefill/Decode 双页表以及阶段切换；
- `native_pinned_uva`：vLLM 原生 pinned-UVA CPU offload，`--enforce-eager`，不使用
  CUDA Graph，卸载参数始终通过 CPU remote 路径访问。

在 workload 和并发一致、且 `ours` 严格 verifier 通过的7个配对点中：

| 场景 | ours E2E | pinned-UVA remote E2E | E2E加速 |
|---|---:|---:|---:|
| Short/C1 | 355.47 ms | 3012.17 ms | 8.47× |
| Ordinary/C1 | 1183.11 ms | 9089.28 ms | 7.68× |
| RAG/C1 | 2066.21 ms | 9401.91 ms | 4.55× |
| Short/C16 | 892.26 ms | 2352.90 ms | 2.64× |
| Ordinary/C16 | 4938.20 ms | 11387.91 ms | 2.31× |
| RAG/C16 | 21207.14 ms | 86402.18 ms | 4.07× |
| Long/C16 | 19822.26 ms | 41269.70 ms | 2.08× |

因此，本轮可以支持以下限定结论：在这7个严格验收的 E2 测试点，完整的
“GPU缓存 Prefill + CUDA Graph + 全覆盖双页表切换”组合比无 Graph、始终 pinned-UVA
remote 的原生路径快 `2.08×–8.47×`。

Long/C1 的数值为 4765.75 ms 对 47566.72 ms，即 9.98×，但 `ours` 的
`grouped_token_events=1` 未通过严格流式协议检查，所以不得计入正式性能范围。

这一对比不能用于拆分单项贡献：它同时改变了 Graph、Prefill 数据位置和页表切换机制；
两种机制实际卸载的权重集合也不完全相同。E3 本轮只采集 D/ours，没有同轮配对 remote
arm。所有点都只有一次冷进程样本，因此这是 qualification 证据，不是统计显著性或普适性
证明。

## E2

| 并发 | 场景 | TTFT | TPOT | 请求 E2E | 严格验收 |
|---:|---|---:|---:|---:|---|
| C1 | Short | 62.41 ms | 9.45 ms | 355.47 ms | 通过 |
| C1 | Ordinary | 78.29 ms | 8.70 ms | 1183.11 ms | 通过 |
| C1 | RAG | 389.22 ms | 13.20 ms | 2066.21 ms | 通过 |
| C1 | Long | 76.81 ms | 9.18 ms | 4765.75 ms | 协议检查失败 |
| C16 | Short | 90.08 ms | 25.88 ms | 892.26 ms | 通过，16/16 |
| C16 | Ordinary | 328.81 ms | 36.29 ms | 4938.20 ms | 通过，16/16 |
| C16 | RAG | 2768.29 ms | 145.19 ms | 21207.14 ms | 通过，16/16 |
| C16 | Long | 330.99 ms | 38.14 ms | 19822.26 ms | 通过，16/16 |

Long/C1 实际成功生成 512/512 token，HTTP、服务退出、GPU 清理和双页表切换都正常。
失败原因是 `grouped_token_events=1`：一次 SSE 事件携带了多个 token，严格 verifier
要求每个事件只对应一个 token，因而把该请求标为协议不通过。这不是工作区挤出、输出
缺失或 GPU 故障。按单轮规则没有重跑。

Long/C1 的 manifest 仍明确记录：

```text
dual_view_full_coverage=true
dual_view_block_count=248
remote_map_operation=prebuilt_full_dual_view
remote range-map ioctl calls=0
```

## E3

表中的 P→D 是 Prefill/Decode 双页表切换时间；Decode 是 steady step 中位数。E2E
包含该场景配置的全部 Decode 步数，不能只由 Prefill、一次切换和一个 Decode step 相加。

| 场景 | Batch | Prefill | P→D | steady Decode | E2E |
|---|---:|---:|---:|---:|---:|
| Short | 1 | 93.03 ms | 7.65 ms | 11.18 ms | 524.73 ms |
| Short | 16 | 163.72 ms | 8.89 ms | 23.49 ms | 983.51 ms |
| Ordinary | 1 | 74.47 ms | 6.52 ms | 9.64 ms | 1381.86 ms |
| Ordinary | 16 | 369.13 ms | 23.94 ms | 55.23 ms | 7522.40 ms |
| Medium | 1 | 135.21 ms | 6.71 ms | 14.17 ms | 1974.15 ms |
| Medium | 16 | 1125.57 ms | 23.78 ms | 56.91 ms | 8503.01 ms |
| RAG | 1 | 787.46 ms | 29.51 ms | 17.23 ms | 3430.35 ms |
| RAG | 16 | 7262.43 ms | 64.73 ms | 161.50 ms | 28019.24 ms |
| Long | 1 | 123.24 ms | 9.02 ms | 12.49 ms | 6821.92 ms |
| Long | 16 | 574.86 ms | 30.12 ms | 61.57 ms | 31883.62 ms |

### RAG/C16 参数修正

第一次 RAG/C16 使用 cold63，在纯 Prefill Graph 捕获后失败。证据为：

```text
需要恢复的热专家             14,722,007,040 B
恢复前可用显存               10,169,876,480 B
实际恢复到GPU的热专家        10,164,895,744 B
缺少                          4,557,111,296 B
```

这证明失败由 12.21 GiB KV、131,072-token Graph/工作区与热专家共同超过显存布局造成，
不是页表错误。请求规模保持不变，只把 RAG/C16 的当前档从 cold63 调为 cold89：

```text
cold bytes                   20,157,825,024 B
需要恢复的热专家              8,833,204,224 B
恢复前可用显存                9,853,206,528 B
恢复后剩余显存                  994,836,480 B
dual_view_block_count               11,299
dual_view_full_coverage               true
remote_map_operation    prebuilt_full_dual_view
```

修正配置只运行一次并成功，没有重跑其余9个已成功单元。

初始失败还暴露了外层 E3 launcher 只看自身退出码、没有核对子单元 receipt 的问题。当前
脚本已改为 `STOP_ON_CUSTOM_ERROR=1`，并强制检查子单元 `status=ok/returncode=0`；以后
相同错误会直接令顶层命令失败。

## E10

| KV 档 | 纯 Prefill | 切档及恢复 | Prefill E2E | P→D页表切换 | steady Decode | 请求 E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | 339.43 ms | 0 | 339.43 ms | 47.87 ms | 21.85 ms | 1034.12 ms |
| 4 GiB | 403.99 ms | 620.14 ms | 1024.13 ms | 53.67 ms | 24.66 ms | 1763.28 ms |
| 6 GiB | 440.98 ms | 474.67 ms | 915.65 ms | 60.71 ms | 27.49 ms | 1729.58 ms |

E10 完成了一条 cold70/2 GiB→cold80/4 GiB→cold89/6 GiB 序列。verifier 记录：

```text
status                       ok
completed_sequences          1
statistical_repetitions      0
dual-view prepare calls      12
dual-view switch calls       12
dual-view prepared blocks    22,598
dual-view skipped blocks     0
```

4/6 GiB 的 Prefill E2E 包含切档；纯计算应读取
`prefill_compute_excluding_rebalance_and_resume_ms`。P→D列仅为页表拓扑 PARK，完整首次
Decode E2E 分别为 104.98、117.27、133.44 ms。

## 产物

```text
artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-02-full-dual/e2/
artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-02-full-dual/e3/
artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-03-rag-c16-cold89/
artifacts/e2-e3-e10-full-c1-c16-uva-single-round-20260831-attempt-02-full-dual/e10/
```

这些是单轮 qualification 数据，不构成统计显著性结论。
