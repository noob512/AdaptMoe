# Qwen3 E2 pure-phase 修复与重跑报告

日期：2026-08-26（Asia/Shanghai）

## 结论

项目已经接通 pure-Prefill CUDA Graph、uniform-Decode CUDA Graph 和
phase-batched scheduler。正式组合保持：

```text
CompilationMode.NONE
CUDAGraphMode.FULL_SEPARATE
```

运行时 mixed Prefill/Decode step 会 fail-closed，不会静默退回 eager。

Qwen3 E2 的 2 GiB 单次重跑仍未进入正式 workload，因此没有可用的 TTFT、
TPOT、goodput 或系统排名数据。失败已明确归类为 pure-Prefill Graph 捕获时的
临时/native 显存需求挤占热门 managed 专家页，不是 KV cache 初始化挤占。

## 实现修改

- 新增真正的 `capture_pure_prefill_graphs()`，使用零上下文、多请求纯 Prefill
  shape；不再借用 mixed capture 语义。
- Decode 继续使用 `capture_uniform_decode_graphs()`。
- 捕获顺序为 remote Decode Graph -> residency reconciliation -> sealed
  pure-Prefill Graph。
- Decode Graph 捕获后增加第二次 reconciliation，并保留 allocator cache。
- phase-batched scheduler 保证一个 step 只包含 Prefill 或 Decode；Decode epoch
  中到达的新请求进入下一 cohort。
- `FULL_SEPARATE` 正式 phase 请求禁止 eager fallback。
- E2 runner 支持独立配置 Qwen 模型、fixture、计划、Graph sizes 和诊断门。

针对性验证共 12 个用例通过：8 个 worker/Graph phase 用例、2 个 scheduler
用例、2 个 FULL_SEPARATE/dispatcher 用例。`bash -n` 与 `git diff --check`
也通过。

## 实验配置

- 模型：Qwen3-30B-A3B-FP8，snapshot
  `d206ba732169f29bb77fbf80fc2c4b81d4d30782`
- 系统：ours，budget 2 GiB，repeat 1
- KV cache：2,147,483,648 B
- cold expert plan：2,264,924,160 B（2,160 MiB）
- hot expert complement：26,726,105,088 B（25,488 MiB）
- Decode Graph：1/4/8/16
- pure-Prefill Graph：128/512/1024/2048/4096/8192/16384
- `max_num_batched_tokens=16384`，`max_num_seqs=16`
- attention：TRITON_ATTN
- 自动 CPU -> GPU fault migration：关闭
- hot-read gate、同步 CUDA 报错和 workspace/allocator 诊断：开启

## 启动与诊断 attempts

| attempt | 结果 | 首个失败边界 | 归因 |
|---|---|---|---|
| 1 | error，0 请求 | Decode Graph 后、Prefill RESUME 前 | Decode Graph native/persistent 分配驱逐 20 MiB hot 页 |
| 2 | error，0 请求 | 首个 16,384-token pure-Prefill dummy forward | pure-Prefill 临时/native 分配驱逐 hot 页 |
| 3 | error，0 请求 | 持有 2 GiB native sentinel 时的 pre-capture hot reconciliation | hot set 仍有 738 MiB 留在 CPU，尚未进入 Graph capture |
| 4 | error，0 请求 | 严格 pre-KV 2 GiB sentinel 下的 hot reconciliation | hot set 仍有 678 MiB 留在 CPU，尚未进入 Graph capture |

attempt 1 中，KV 与 kernel warmup 后 reconciliation 证明全部 25,488 MiB hot
权重在 GPU；Decode Graph 捕获后却有 20 MiB hot 页在 CPU。因此加入了 Decode
Graph 后二次 reconciliation。

attempt 2 中，该修复生效：

- Decode Graph 捕获后二次 reconciliation 成功；
- hot GPU = 26,726,105,088 B，hot CPU = 0，unknown = 0；
- cold CPU = 2,264,924,160 B，unknown = 0；
- 保留 Decode Graph allocator cache；
- free HBM = 491,520,000 B（468.75 MiB）；
- RESUME 前、后 hot CUDA-read gate 都通过。

随后首个 16,384-token pure-Prefill capture 在 layer 0 fused MoE 触发：

```text
Xid 31: FAULT_PDE ACCESS_TYPE_VIRT_READ
fault VA: 0x73f549800000
parameter: model.layers.0.mlp.experts.w13_weight
expert: 40 (hot)
```

layer 0 的计划内 cold experts 是
`10/15/19/25/35/39/55/58/88/90`，所以 expert 40 不是 remote cold expert。

## KV 与临时缓冲区判定

不是 KV cache 导致本次失败。理由是 KV 分配完成以后，先后两次完整 residency
审计都能把全部 hot 权重放在 GPU；紧邻 pure-Prefill capture 的 RESUME 前后
真实 CUDA 读取也通过。

故障只在 16,384-token pure-Prefill 开始创建 attention/MoE workspace 与
Graph-private native storage 后出现，且首个 MoE 对 hot expert 40 的读取立即
失败。因此失败类别是 pure-Prefill 临时/native 分配边界。由于 capture 期间
不能插入会扰动 Graph 的完整 post-workspace 审计，本次没有故障瞬间的精确
evicted-byte 数；但故障时间边界、hot VA 和 KV 后的可读性均已确认。

attempt 3 进一步验证了 2 GiB native-hole 方法。sentinel 的
`cudaMalloc + cudaMemset` 成功并精确占用 2 GiB；但在 sentinel 仍存活时，
26,726,105,088 B hot set 只能恢复 25,952,256,000 B，另有 773,849,088 B
（738 MiB）留在 CPU，free HBM 仅剩 2.75 MiB。fail-closed gate 因此在任何
Decode/Prefill Graph 捕获前终止。cleanup 精确归还 2 GiB，未产生 Xid。

所以在冻结的 2 GiB cold plan 下，2 GiB sentinel 不能与完整 hot set 同时驻留，
还无法检验释放后的 extent 是否足以承载 16,384-token capture。若继续该方法，
至少还要减少 738 MiB hot set，并额外满足 192 MiB 安全余量；这会改变 E2 memory
tier。完整账目见 attempt 3 的 `RESULT.md`。

attempt 4 按严格顺序复测：先分配并触碰 2 GiB sentinel，再实际初始化 2 GiB
KV，保持 sentinel 完成 kernel warmup，并在最终 hot/cold reconciliation 后才准备
释放。结果 hot set 仍有 710,934,528 B（678 MiB）留在 CPU，free HBM 仅
2.75 MiB，fail-closed gate 在所有 Graph capture 前终止。失败 cleanup 精确归还
2 GiB，未产生 Xid，也未发送请求。严格结果将额外 headroom 的下界修正为
678 MiB + 192 MiB safety margin = 870 MiB；完整账目见 attempt 4 的
`RESULT.md`。

## E2 有效性

- server ready：false
- 完成请求：0
- 有效 performance records：0
- verifier：error
- equal-HBM/速度提升/系统排名：不可得出
- GPU 清理：0 MiB、无 compute process
- UVM：已恢复 stock，无需重启

## 后续条件

正式 2 GiB E2 要继续，需要在不关闭 CUDA Graph、不恢复 mixed scheduling 的
前提下解决 16,384-token pure-Prefill 的 HBM headroom。可行方向是增加 cold
expert 容量，或把最大 pure-Prefill shape/KV budget 限制到经过资格验证且能保持
hot 页可读的范围。任何改变都会改变当前 E2 memory tier，必须重新冻结配置后再
生成正式性能结果。

详细逐事件取证见本轮 `attempt_2/ROOT_CAUSE.md`。
