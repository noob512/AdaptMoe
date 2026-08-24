# GPU/free、GPU/backup、CPU/offline 双向 E2E qualification

该实验使用真实 Qwen3-30B-A3B-FP8 模型，在同一实验组进程中验证一次完整的
三队列升压与自动降压往返，而不是只验证 Python 队列单测。

第一轮请求使用 314-token Prefill 和 6 个输出 token。实验组从 24-block GPU KV
prefix 启动，其中 8 blocks 是 GPU backup；请求需要 20 blocks，先耗尽 16 个普通
free blocks，再消费 4 个 backup blocks，并在同一调度步触发：

```text
GPU/free -> GPU/backup -> pressure event
         -> 每层卸载 1 个专家 -> KV 24 -> 40
         -> CPU/offline blocks 迁入 GPU -> backup 补回 8
```

第二轮请求使用 16-token Prefill 和 5 个输出 token。统一 autoscale 控制器在前三个
有效低压调度轮依次产生：

```text
observe_low_pressure -> observe_low_pressure -> shrink_pending
```

第三轮只生成缩容决定；活动请求尚未到安全边界时不会迁移其 KV。最终 Decode 完成后，
控制器在空闲边界执行 `40 -> 20`，把 KV tail 迁回 CPU/offline，再按层恢复此前动态
卸载的 48 个专家。

baseline 从固定 40-block target 启动，并关闭 backup、增量专家 wave 和 autoscale。
两组都顺序执行：

```text
314-token Prefill Graph -> Decode Graph
16-token Prefill Graph  -> Decode Graph
```

CUDA Graph 使用 `FULL_SEPARATE`，预捕获大小为 `1,16,314`。验证器检查两轮输出 token
逐项一致、Prefill/Decode replay、`remote_parked -> sealed -> remote_parked` 往返、
三队列守恒/互斥、GPU/CPU 物理驻留、KV 与 Parameter VA 稳定、每层对称专家
卸载/恢复、Graph capture 计数不增加、内核日志以及 shutdown 清理。

历史 qualification artifact 已按清理策略移除。本目录保留 runner、验证器和复现方法；
任何新结果必须使用新的 run ID 重新生成，不再引用旧数值。

```bash
./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh static-preflight

ALLOW_GPU_RUN=1 \
  ./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh qualify
```

新结果保存在 `artifacts/<UTC>_qualification/verification.json`。可对已有产物重新验证：

```bash
./experiments/vllm_uvm_three_queue_e2e/run_experiment.sh verify <RUN_ROOT>
```
