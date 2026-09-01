# E10 去除 Python 全量重叠检查后重新运行（单轮）

## 运行范围

按相同冻结协议重新完整运行一次 E10。此次请求只执行这一轮，没有失败重跑或统计重复。
baseline 和 three_queue 分别使用独立冷进程；策略为 cold-frequency，不使用预测，
8-block backup 消耗 4 blocks 后触发，KV 目标从 24 増长到 40 blocks。

## 本轮结果

| 指标 | baseline | three_queue | three_queue 相对变化 |
| --- | ---: | ---: | ---: |
| 高压 Prefill | 169.756 ms | 252.402 ms | +48.68% |
| 首次 Decode | 422.338 ms | 434.932 ms | +2.98% |
| steady Decode 中位数 | 10.608 ms | 11.840 ms | +11.61% |
| 高压 cycle E2E | 666.523 ms | 781.336 ms | +17.23% |
| low-trace step sum | 805.811 ms | 936.619 ms | +16.23% |

档位与 KV 切换 RPC：

- `cold29 -> cold30` tier-grow：43.269 ms；
- KV grow：32.606 ms；
- `cold30 -> cold29` tier-restore：95.220 ms；
- KV shrink：15.564 ms。

正式 `cold29 -> cold30` plan switch 为 40.285 ms。其中：

- 450-entry alias append wrapper：26.591 ms；
- 驱动 ioctl：23.827 ms；
- ioctl 外 Python 开销：2.763 ms；
- 18-entry backing preallocate：2.380 ms；
- 其余 shadow cutover、驻留审计和控制开销：约 11.314 ms。

因此原先 358.575 ms 的 Python 全量扫描在本轮仍未出现；wrapper 保持在约 27 ms，
且绝大部分已经是驱动页表操作。高压 Prefill 的同轮差值为 82.646 ms，tier-grow 和
KV-grow 两项显式 RPC 合计 75.875 ms，是当前主要同步差值。

## 正确性与清理

- verifier `status=ok`；baseline/three_queue PID 为 27658/27993；
- 释放并恢复 48 个专家、96 个参数范围、226,492,416 B；
- CPU source 全部复用，`changed_bytes=0`；
- 450-entry quiesced append 成功，450 次 TLB invalidate；
- Prefill/Decode Graph replay 和完整 roundtrip 均通过，Graph capture count 固定为 6；
- `kernel_faults=0`，`shutdown_clean=true`；
- 最终恢复 stock UVM，GPU memory used=0 MiB。

## 结论

本轮再次确认 Python 全量重叠检查瓶颈已经消除，专家释放档位可以在约 43 ms 的 RPC
内完成。但 three_queue 高压 cycle 仍比同轮 baseline 高 17.23%，steady Decode 仍高
11.61%，所以单轮结果仍不能证明端到端性能稳定或无下降。

## 产物

- 完整运行：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-no-python-overlap-REPRO-20260830-attempt-02-qualification/`
- verifier：上述目录下 `verification.json`
- 原始结果：上述目录下 `baseline/result.json` 与 `three_queue/result.json`
- 冻结协议：上述目录下 `protocol.json`
