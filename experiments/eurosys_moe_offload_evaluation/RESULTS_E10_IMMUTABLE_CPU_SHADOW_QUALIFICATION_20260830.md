# E10 immutable CPU shadow qualification（单轮）

## 范围

本次实现并验证了运行期无 GPU→CPU 数据复制的专家释放路径。初始化阶段为未来
`cold30` 档位相对 `cold29` 新增的专家，在同一 Parameter VA 上建立 CPU/GPU
read-duplicate；schedule 静默点触发释放时，驱动保留已有 CPU shadow、撤销 GPU
映射并直接释放 GPU duplicate，随后建立 GPU 到 CPU 的只读远程映射。

实验固定使用 cold-frequency `cold29/cold30` 档位；不使用预测，只在 8-block backup
队列消耗 4 blocks 时触发扩容。`baseline` 与 `three_queue` 各使用一个独立冷进程、各运行
一轮，没有统计重复。

attempt-01/02 是未进入测量或未完成 three-queue 往返的开发失败尝试，不计入实验结果；
attempt-03 是唯一完整通过 verifier 的 qualification。

## 参数与路径

- 模型：Qwen3-30B-A3B-FP8，单卡，batch size 1，seed 0。
- KV 档位：初始 24 blocks，压力后增长到 40，低压后缩到 20。
- backup：保留 8 blocks，消耗 4 blocks 时触发；`prediction_enabled=false`。
- 专家档位：每层 `cold29 -> cold30` 增加 1 个专家，共 48 个专家、96 个参数范围、
  226,492,416 B（216 MiB）。
- 档位拓扑：一次 450-entry quiesced delta ioctl；内部不执行 PARK/RESUME。
- 运行时路径：CPU source 命中，`source_refill.changed_bytes=0`；immutable shadow
  cutover 的 96 个范围和全部字节均成功。

## 单轮结果

正式请求命中预编译指令后的切换如下：

| 指标 | 上一版 quiesced-delta | immutable shadow | 诊断性变化 |
| --- | ---: | ---: | ---: |
| `cold29 -> cold30` RPC | 1069.805 ms | 374.865 ms | -694.940 ms（-64.96%） |
| `cold30 -> cold29` RPC | 393.978 ms | 96.772 ms | -297.206 ms（-75.44%） |
| 双向 RPC 合计 | 1463.783 ms | 471.638 ms | -992.145 ms（-67.78%） |

以上是两个不同单轮 artifact 的机制对比，不能作为统计加速结论。

本次同轮 baseline 对比：

| 指标 | baseline | three_queue | three_queue 相对变化 |
| --- | ---: | ---: | ---: |
| 高压 Prefill | 171.004 ms | 583.047 ms | +240.96% |
| 首次 Decode | 417.587 ms | 429.974 ms | +2.97% |
| steady Decode 中位数 | 10.516 ms | 11.688 ms | +11.15% |
| steady Decode 吞吐 | 95.095 tok/s | 85.555 tok/s | -10.03% |
| 高压 cycle E2E | 662.103 ms | 1105.696 ms | +67.00% |
| low-trace step sum | 790.956 ms | 931.967 ms | +17.83% |

因此，本次证明 CPU shadow 优化和完整多次往返已经生效，但仍不能证明 three_queue
相对 baseline 没有性能下降。

## 切换分解

正式触发、命中指令缓存的 `cold29 -> cold30` plan switch 为 371.904 ms：

- 450-entry alias append wrapper：358.575 ms，其中内核 ioctl 23.584 ms；
- 18-entry / 9 MiB backing preallocate：2.378 ms；
- shadow cutover、同步、驻留审计及其余控制开销合计不超过约 10.951 ms；
- shadow action 完整释放 96 ranges / 226,492,416 B，CPU shadow 和远程映射字节数均
  为 226,492,416 B；没有运行期 GPU→CPU 数据复制。

正式 `cold30 -> cold29` plan switch 为 94.523 ms：

- 450-entry alias remove wrapper：23.918 ms，其中内核 ioctl 10.010 ms；
- 其余约 70.606 ms 为 216 MiB CPU→GPU 恢复、同步和驻留审计；
- 恢复后对应范围 CPU/GPU 各驻留 226,492,416 B，可以供下一次无复制 cutover 复用。

释放方向的剩余主要瓶颈已不再是 216 MiB 数据迁移，而是 Python alias append wrapper
对完整注册表的校验和缓存维护。cycle 中还包含独立的 Prefill→Decode phase cutover，
它不属于专家档位切换，也尚未由本次优化消除。

## 第二次 cutover 修复

开发期 attempt-02 的第一次往返成功，第二次释放被错误拒绝。驱动诊断表明 CPU/GPU
双驻留和 `read_duplicated_pages` 均完整，但 static-alias remove 在 GPU 恢复后保守地
留下 WRITE/ATOMIC PTE 权限标志。

修复后以 UVM 的 `read_duplicated_pages` 作为内容一致性的权威条件；实际 GPU 写入会先
撤销该状态。切换仍要求 schedule 静默、CPU/GPU 全驻留、完整 GPU read mapping、没有
slot PTE、没有 discarded page，并在释放前撤销 GPU 映射。完整模型诊断随后成功完成两次
释放和两次恢复，拒绝数为 0。

## 正确性与清理

- verifier `status=ok`；baseline/three_queue PID 分别为 21856/22208，确为独立冷进程。
- baseline 与 three_queue 的高压输出 token 完全一致；低压 autoscale roundtrip 输出也
  完全一致。
- Prefill/Decode Graph replay 均通过，CUDA Graph capture count 固定为 6，无重捕获。
- `kernel_faults=0`，96 个 shadow ranges 每个字节均通过释放/驻留审计。
- 双循环 2 MiB 小探针连续两轮通过；第二轮 cutover 0.551 ms，恢复 0.406 ms，远程读取
  和恢复后内容均正确。
- 负向安全探针在第一次恢复后真实写入 GPU 页；UVM 随即撤销 CPU residency 和
  read-duplicate，第二次 cutover 按预期 fail-closed，未释放任何字节。
- Python 回归测试 181 passed；E10 queue/plan-class 测试 16 passed；内核模块编译成功。
- verifier `shutdown_clean=true`，最终恢复 stock `nvidia_uvm`，GPU memory used=0 MiB。

## 结论

immutable CPU shadow 已达到本轮实现目标：初始化时准备 Parameter VA 的 CPU shadow，
运行期释放 216 MiB GPU duplicate 时不再复制数据，并且支持同一进程内多次
`release -> restore -> release` 往返。相对上一版单轮，正式释放和恢复 RPC 分别缩短
64.96% 和 75.44%。

端到端性能稳定目标尚未达到：three_queue 高压 cycle 仍比同轮 baseline 高 67.00%。
下一主要优化对象是 358.575 ms 的 alias append wrapper 全量校验，以及独立的
Prefill→Decode phase cutover；不是再次优化 GPU→CPU 数据复制。

## 产物

- 完整 qualification：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-immutable-shadow-REPRO-20260830-attempt-03-qualification/`
- verifier：上述目录下 `verification.json`
- 原始结果：上述目录下 `baseline/result.json` 与 `three_queue/result.json`
- 冻结协议：上述目录下 `protocol.json`
- 双循环小探针：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-immutable-shadow-probe-20260830-attempt-09/`
- 完整模型双往返诊断：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-immutable-shadow-diagnostic-20260830-attempt-03/`
- GPU 写入负向安全探针：
  `experiments/eurosys_moe_offload_evaluation/artifacts/e10-immutable-shadow-probe-20260830-attempt-10-negative-write/`
