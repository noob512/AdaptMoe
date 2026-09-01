# E11 continuous-arrival 单轮 qualification（2026-08-31）

## 结论

本轮已经用真实 DeepSeek-V2-Lite、真实 EngineCore 和持续到达 trace 证明两项调度行为：

- `phase_epoch` 将新到 Prefill 与旧 Decode 完全分开，32 个 scheduler step 中没有 mixed step；
- `remote_only + default continuous scheduler` 在第 5 步真实形成一次 `4 个新 Prefill + 4 个旧 Decode` 的 mixed step；
- hybrid phase-specific 在关闭 phase epoch 后遇到相同 mixed step，会明确返回
  `requires an unmixed request`，没有错误执行或 GPU/内核故障。

但本轮不能作为正式 E11 correctness 结果：phase-specific 的 pure-Prefill Full Graph 与
remote-only eager 路径生成的 token IDs 不一致，8/8 请求的完整 token hash 都不同。因此
`verification.json` 保持 `status=error`，不把调度功能通过错误写成 formal E11 通过。KT 和
MoE-Infinity 的 continuous-arrival 对照也尚未运行。

## Trace

每个 arm 都是独立冷进程：

1. 在第一次 EngineCore step 前加入 4 个 128-token 请求；
2. 完成 Prefill 并运行到第 4 个 engine step，旧请求仍在 Decode；
3. 再加入 4 个 128-token 请求；
4. 每个请求固定生成 16 token；
5. 包装真实 `scheduler.schedule()`，记录 `scheduled_new_reqs`、
   `scheduled_cached_reqs`、token 数和 phase epoch。

不经过 HTTP，不根据输出时间反推 batch formation。

## 性能结果

| 指标 | phase epoch + phase-specific | default continuous + remote-only |
|---|---:|---:|
| 成功请求 | 8/8 | 8/8 |
| scheduler steps | 32 | 20 |
| mixed steps | 0 | 1（step 5） |
| 后批请求延迟的 scheduler steps | 12 | 0 |
| 后批 queueing p50 | 352.184 ms | 0.495 ms |
| TTFT p50 | 522.061 ms | 175.680 ms |
| TPOT p50 | 57.574 ms | 81.935 ms |
| E2E p50 | 1385.667 ms | 1404.719 ms |
| request goodput | 3.305 req/s | 4.509 req/s |
| generated-token throughput | 52.887 token/s | 72.144 token/s |

在这一个短 trace 中，default continuous 的 request goodput 高 36.4%，且后批请求几乎
立即进入 GPU；代价是 mixed step 之后单 token 间隔更高。phase epoch 避免了任何 mixed
执行，但让后批等待旧 cohort 的剩余 12 个 Decode step。

这些是每个 arm 单轮数据，不能作为统计 speedup 或等价结论。

## Correctness 异常

两个成功 arm 都满足：

- 8 个请求全部生成 16 token；
- 无 Xid、`FAULT_PDE`、`FAULT_PTE`、MMU fault 或内核错误；
- 每个进程退出后 GPU 0 MiB；
- 最后恢复 stock UVM。

但是输出不一致。例如冻结输入 `initial-00`：

```text
phase-specific Full Prefill Graph:
34261, 6301, 40706, 4589, 185, ...

remote-only eager:
1, 185, 58, 17, 60, ...
```

该差异在第一个 token 就出现，早于第二批请求到达，因此不是 mixed step 造成。现有证据把
问题缩小到 pure-Prefill Full Graph 与 eager Prefill 的执行路径差异。下一步必须在相同
remote-only topology 下分别运行 phase-epoch/default scheduler，或增加 teacher-forced
logit 比较，才能把 scheduler 效果与 Prefill Graph 路径分开。

## Fail-closed 结果

`hybrid_fail_closed` 使用 phase-specific topology，但关闭 phase epoch。第 5 步调度器形成
mixed batch 后，worker 返回：

```text
RuntimeError: Original-VA alias-prefill/remote-decode requires an unmixed request
```

进程退出后 GPU 为 0 MiB，内核日志无故障。这证明不支持的 mixed hybrid 路径没有静默执行。

## Attempts

- attempt-01：`phase_epoch` 成功；remote-only 错误使用 `FULL_SEPARATE`，在初始化配置检查
  阶段被拒绝。失败原样保留，未计入结果。
- attempt-02：三种预注册行为均执行；两个成功 arm 完成，fail-closed arm 按预期拒绝。
  formal verifier 仅因跨执行路径 token 不一致保持 error。

## 产物

- `artifacts/e11-continuous-arrival-20260831-attempt-02-qualification/verification.json`
- `artifacts/e11-continuous-arrival-20260831-attempt-02-qualification/phase_epoch/result.json`
- `artifacts/e11-continuous-arrival-20260831-attempt-02-qualification/remote_continuous/result.json`
- `artifacts/e11-continuous-arrival-20260831-attempt-02-qualification/hybrid_fail_closed/result.json`
