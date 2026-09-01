# E10 去除 Python 全量重叠检查 qualification（单轮）

## 范围

本次只删除 `driver_static_alias_map()` 在增量 append 前对“新增映射 × 已有映射”
执行的 Python 全量重叠扫描。没有把该扫描移动到初始化阶段。驱动内核仍在持有
static-alias session 锁时检查批内冲突、与已有 alias 的冲突以及 managed VA 所属关系，
冲突请求仍会在页表修改前失败。

实验沿用 immutable CPU shadow qualification 的冻结参数：cold-frequency
`cold29/cold30`，不使用预测，8-block backup 消耗 4 blocks 时触发扩容。baseline 和
three_queue 各使用一个独立冷进程、各执行一轮；没有统计重复，也没有失败后重跑。

## 实现与验证

- 删除 Python `_ranges_overlap_existing()` 及其调用，消除
  `450 × 13,050 = 5,872,500` 次区间比较。
- 保留 450-entry 请求本身的基本格式检查和驱动内核的权威冲突检查。
- 新增回归测试，确认与缓存地址重叠的 append 不再被 Python 拦截，而是提交给驱动。
- 相关 Python 回归：223 passed；E10 static preflight：16 passed。

## 唯一一轮结果

正式 high-pressure 触发的 `cold29 -> cold30`：

| 指标 | 上一版 immutable shadow | 本次 | 单轮诊断变化 |
| --- | ---: | ---: | ---: |
| 450-entry alias append wrapper | 358.575 ms | 27.110 ms | -92.44% |
| 其中驱动 ioctl | 23.584 ms | 24.317 ms | +3.11% |
| ioctl 外用户态时间 | 334.991 ms | 2.794 ms | -99.17% |
| plan switch | 371.904 ms | 40.043 ms | -89.23% |
| tier-grow RPC | 374.865 ms | 42.897 ms | -88.56% |

358 ms 瓶颈已被消除。新的 27.110 ms wrapper 中，24.317 ms 在驱动 ioctl 内，Python
包装、参数打包和缓存更新只剩约 2.794 ms。驱动时间基本不变，说明优化没有通过跳过
页表提交来得到结果。

恢复方向 `cold30 -> cold29` 的 tier-restore RPC 为 72.885 ms；其主要工作仍是把
216 MiB 专家恢复到 GPU。它相对上一轮单次 96.772 ms 有所下降，但两个数均来自不同
的单轮运行，不能解释为统计加速。

## 同轮 baseline 对比

| 指标 | baseline | three_queue | three_queue 相对变化 |
| --- | ---: | ---: | ---: |
| 高压 Prefill | 166.196 ms | 252.479 ms | +51.92% |
| 首次 Decode | 418.633 ms | 427.516 ms | +2.12% |
| steady Decode 中位数 | 10.589 ms | 11.751 ms | +10.98% |
| 高压 cycle E2E | 659.073 ms | 772.594 ms | +17.22% |
| low-trace step sum | 814.540 ms | 907.673 ms | +11.43% |

相对上一版单轮，three_queue cycle E2E 从 1105.696 ms 降到 772.594 ms，数值差
333.102 ms，与移除的 Python 扫描量一致；但这是跨两次单轮运行的诊断比较，不是统计
结论。本次同轮 three_queue 仍比 baseline 高 17.22%，所以仍不能证明性能完全稳定。

当前高压 Prefill 的主要显式切换开销是 tier-grow RPC 42.897 ms 和 KV grow RPC
32.310 ms。绝对延迟中还存在独立的 Prefill-to-Decode phase cutover；它不属于本次
450-entry append 优化。

## 正确性与清理

- verifier `status=ok`，baseline/three_queue PID 分别为 25611/25946。
- prediction disabled；backup 使用 4/8 blocks 后触发，KV 从 24 增到 40 blocks。
- 释放和恢复 48 个专家、96 个参数范围、226,492,416 B。
- 450-entry quiesced append 成功，450 次 TLB invalidate，Graph capture count 保持 6。
- Prefill/Decode Graph replay、输出和 autoscale roundtrip 均通过。
- `kernel_faults=0`，`shutdown_clean=true`；最终恢复 stock UVM，GPU 0 MiB。

## 产物

- 完整 qualification：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-no-python-overlap-REPRO-20260830-attempt-01-qualification/`
- verifier：上述目录下 `verification.json`
- 原始结果：上述目录下 `baseline/result.json` 与 `three_queue/result.json`
- 冻结协议：上述目录下 `protocol.json`
