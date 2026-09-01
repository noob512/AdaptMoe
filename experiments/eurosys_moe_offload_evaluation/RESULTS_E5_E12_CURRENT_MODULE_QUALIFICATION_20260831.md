# E5/E12 当前模块补充 qualification（2026-08-31）

## 结论

本轮在与最新 E2/E3/E10 相同的 custom UVM 上补跑了两组此前未完成的当前模块证据：

```text
SHA256      c8fe77c8ecc87f6d724d94ddf6535c934150d4e12bec384bd44562097aec043a
srcversion  CD4A27464CD3E907C4EF493
```

- E12 机制级双页表 soak 完成 1,000/1,000 次 PARK/RESUME，逐轮数据校验和最终
  verifier 均通过；
- E5 stable-VA DESTROY/rebuild 和 non-stable-VA/recapture 各完成 10 个独立冷进程、
  20 个 plan generation，两个 verifier 均为 `status=ok`；
- 两个 E5 臂的 B1/B4 输出 hash 分别一致，fault scan 和资源释放全部通过；
- 退出后恢复 stock UVM `85A79790636BBD99BA3E43B`，GPU 0 MiB。

这些结果补齐的是当前模块的机制 qualification。E12 仍未完成 6/24 小时整模型服务
soak、KV 峰谷和 barrier 故障注入；E5 也未在当前模块上重跑全部 Graph/PTE 消融臂。

## E12：1,000 次双页表 PARK/RESUME

实验覆盖 4 个 2-MiB parent blocks 和 5 个跨块/块内 cold regions。每轮先 PARK 到完整
Decode child，验证 CPU-shadow 视图，再更新 upload 数据、RESUME 到 Prefill child，并
验证 GPU-backing 视图。`verify_every=1`，因此全部 1,000 轮都执行数据检查。

| 操作 | count | mean | p50 | p95 | min | max |
|---|---:|---:|---:|---:|---:|---:|
| PARK | 1,000 | 0.773 ms | 0.769 ms | 0.838 ms | 0.692 ms | 2.682 ms |
| RESUME | 1,000 | 0.559 ms | 0.567 ms | 0.611 ms | 0.496 ms | 2.528 ms |

内核统计增量为 2,000 次 dual-view switch、8,000 次 PDE 写入和 2,000 个 TLB batches。
PARK/RESUME 都保持 4/4 parent-block full coverage；没有 Xid、`FAULT_PDE`、
`FAULT_PTE`、MMU fault、poison 或资源泄漏。

内核日志开头一次 action 22 的 `0x26` 是 FD 候选选择中的
`NV_ERR_INVALID_DEVICE`：首个不匹配 FD 为 0/5，随后正确 FD 为 5/5。它没有产生任何
部分进展，属于用户态按现有 UVM FD 选择策略的预期探测，不是 PARK/RESUME 失败。

本实验循环本身耗时 3.700 s，整个进程墙钟 12.07 s。它满足“至少 1,000 次
PARK/RESUME”的机制次数门，但时间和模型覆盖不足以称为 6 小时服务 soak。

产物：

- `artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification/verification.json`
- `artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification/result.json`
- `artifacts/e12-dual-view-soak-20260831T130345Z-attempt-01-qualification/kernel.since.log`

## E5：stable 与 non-stable VA 当前模块配对

两个臂使用相同 Qwen3-30B-A3B-FP8、layers 0--3、8 个 Parameter、2,415,919,104 B
逻辑 alias 和 134,217,728 B 物理 KV。每个臂运行 10 个独立冷进程，每轮执行
`initial B1 -> B4 -> B1`，产生 B4/B1 各 10 个 rebuild 样本。

| 路径 | B4 rebuild median | B1 rebuild median |
|---|---:|---:|
| stable Parameter VA + DESTROY/rebuild | 8,603.378 ms | 9,167.878 ms |
| non-stable Parameter VA + reallocate/recapture | 11,525.182 ms | 13,884.723 ms |
| non-stable / stable | 1.340x | 1.514x |

stable 臂的 bootstrap 95% median CI 为 B4 `[8,483.248, 8,671.681] ms`、B1
`[9,121.700, 9,230.098] ms`。non-stable verifier 当前只报告 10 个样本的分布，没有
bootstrap CI，因此不为它补造置信区间。

stable 臂验证所有 Parameter VA 保持不变；non-stable 臂验证每一代 8/8 Parameter VA
全部改变且 Graph 全部重捕获。两个臂在 B4 和 B1 分别得到同一输出 hash：

```text
B4  bbbd59a5387d09c821d47674700db95d6f1dc72bdd2c7aa1af354f3ca9b4a5ba
B1  5d4381e10ea7ad60286860fade0d86c7263054b3b8e4a6c90f611b533f9b1837
```

这里的 1.340x/1.514x 只比较两个当前模块 rebuild 臂，不能替代旧 E5 中
PARK/RESUME、grouped PTE 或 eager/Graph 的独立消融，也不能与不同模块的 1.998x/1.811x
直接合并。

产物：

- `artifacts/e5-destroy-rebuild-current-module-20260831T131000Z-attempt-01-formal/verification.json`
- `artifacts/e5-nonstable-va-current-module-20260831T133200Z-attempt-01-formal/verification.json`
