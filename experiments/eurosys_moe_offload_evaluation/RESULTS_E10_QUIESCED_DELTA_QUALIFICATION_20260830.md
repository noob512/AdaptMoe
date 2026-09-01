# E10 schedule 静默边界差量切换 qualification（单轮）

## 范围

本次实现并验证以下路径：

```text
上一轮 GPU 工作完成
→ vLLM schedule 检测 backup 队列压力
→ 只提交当前档位的 alias 差量
→ 只迁移发生变化的专家范围
→ execute_model
```

实验继续使用冻结的 cold-frequency `cold29/cold30` 档位；不使用预测，8-block
backup 中消耗 4 blocks 时触发扩容。`baseline` 与 `three_queue` 各使用一个独立冷进程，
各执行一轮，没有统计重复。attempt-01/02 为开发期失败尝试，不计入结果；attempt-03 是
唯一通过完整 verifier 的 qualification。

## 实现

- 新增 scheduler-quiesced sealed-session append 和 remove-only ioctl。
- `cold29 -> cold30` 只追加新档位记录；`cold30 -> cold29` 只删除退出档位记录。
- remove 支持 `remote_parked` session：只缩减保留的 alias/backing registry，使未来
  RESUME 恢复正确的 `cold29` 拓扑，不再提交完整 13,050-entry REBIND。
- 新增和删除均要求完整 backing identity；任一部分失败即 fail-closed/poison，交由
  DESTROY 收敛清理。
- 超过 1,152 entries、混合增删或不满足静默状态时仍回退旧路径。

本次每层改变一个专家，共 48 个专家、96 个参数范围、226,492,416 B（216 MiB）。
由于每个参数范围还会拆成 consumer chunk，并包含 upload alias，驱动 ABI 中的实际差量
为 450 entries；它们在一次 ioctl 中统一提交，而不是 450 次调用。

## 单轮结果

| 指标 | 旧 prefilled/precompiled pilot | 本次差量 qualification | 诊断性变化 |
| --- | ---: | ---: | ---: |
| `cold29 -> cold30` RPC | 1892.124 ms | 1069.805 ms | -822.319 ms（-43.46%） |
| `cold30 -> cold29` RPC | 621.763 ms | 393.978 ms | -227.785 ms（-36.64%） |
| three_queue 高压 Prefill | 2148.325 ms | 1323.617 ms | -824.708 ms（-38.39%） |
| three_queue 高压 cycle E2E | 3010.974 ms | 2163.048 ms | -847.926 ms（-28.16%） |
| three_queue load | 84.490 s | 82.565 s | -1.925 s（-2.28%） |

以上比较来自两个不同的单轮 artifact，只用于确认机制方向，不能作为统计加速结论。

本次同轮 baseline 对比：

| 指标 | baseline | three_queue | three_queue 相对变化 |
| --- | ---: | ---: | ---: |
| 高压 Prefill | 154.092 ms | 1323.617 ms | +758.98% |
| 首次 Decode | 722.106 ms | 745.663 ms | +3.26% |
| steady Decode 中位数 | 10.061 ms | 12.646 ms | +25.69% |
| 高压 cycle E2E | 948.274 ms | 2163.048 ms | +128.10% |
| low-trace step sum | 1344.381 ms | 1951.003 ms | +45.12% |

因此本次证明了差量机制生效并显著缩短了档位切换，但还不能证明性能稳定或无下降。

## 切换内部证据

`cold29 -> cold30`：

- plan switch 1067.739 ms，RPC 1069.805 ms；
- 450-entry append，1 次 ioctl；ioctl 39.208 ms，Python wrapper 393.057 ms；
- PARK=`null`，RESUME=`null`；
- backing preallocate 18 entries / 9 MiB，2.495 ms；
- 96 个范围 / 216 MiB 全部成为 CPU-only；
- 初始化期 CPU source 全部复用，运行时 refill 为 0 B。

`cold30 -> cold29`：

- plan switch 392.133 ms，RPC 393.978 ms；
- parked session 上执行 450-entry remove，1 次 ioctl；ioctl 10.481 ms，wrapper
  23.630 ms；
- PARK=`null`，RESUME=`null`；
- 只失效 18 个仍存在的 upload alias，不再处理已经 PARK 的 consumer PTE；
- 96 个范围 / 216 MiB 全部 prefetch 回 GPU，并通过 GPU-only residency 审计。

释放方向剩余约 672 ms 主要是 216 MiB GPU-to-CPU 驻留准备、同步和审计；另外 append
wrapper 比 ioctl 多约 354 ms，主要来自 Python 中对 450 个新记录与约 13,050 个已有
范围的校验和缓存更新。恢复方向除差量 wrapper 外约 369 ms，主要是 216 MiB
CPU-to-GPU prefetch、同步和驻留审计。

## 与 phase cutover 的边界

本次消除的是“专家档位切换内部”的全量 PARK/REBIND/RESUME。现有
alias-Prefill/remote-Decode 模式在首次 Decode 时仍有独立的 phase cutover，本轮耗时
725.118 ms，其中仍包含对当前整个 alias session 的 PARK。该路径不是档位切换的一部分，
但仍进入 cycle E2E，是总体性能尚未稳定的另一个主要来源。

## 正确性和清理

- verifier `status=ok`；baseline/three_queue 为不同 PID 的独立冷进程；
- backup-only 触发，prediction disabled；KV 24 -> 40 -> 20 blocks；
- CUDA Graph capture count 为 6，运行期保持 6，无重捕获；
- Prefill/Decode Graph replay、输出 token hash 和 autoscale roundtrip 全部通过；
- kernel faults=0，shutdown session count=0；
- Python 回归测试 113/113 通过，内核模块编译通过；
- 真实 GPU parked-remove 小探针通过：单次 remove 0.378 ms，删除后原始 backing、
  retained alias 和后续 RESUME 均正确；
- 最终恢复 stock `nvidia_uvm`，GPU memory used=0 MiB。

## 结论

目标中的核心控制路径已经实现：schedule 静默点不再为档位切换执行全量 PARK、完整
13,500/13,050-entry REBIND 和 RESUME；双向都只提交一次差量 ioctl。与旧单轮 pilot
相比，释放/恢复 RPC 分别缩短 43.46%/36.64%。

当前剩余瓶颈已经转为实际驻留迁移、同步审计、append 的 Python 全量重叠检查，以及
独立的 Prefill-to-Decode phase PARK。端到端 three_queue 仍比同轮 baseline 慢 128.10%，
所以不能声称已经达到性能稳定目标。

## 产物

- 完整 artifact：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-quiesced-delta-pilot-20260830-attempt-03/`
- 最终验证：artifact 下 `verification.json`
- 原始结果：artifact 下 `baseline/result.json` 和 `three_queue/result.json`
- 冻结协议：artifact 下 `protocol.json`
- parked-remove 探针：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-quiesced-delta-pilot-20260830-attempt-02/quiesced_delta_parked_probe_final2.json`
