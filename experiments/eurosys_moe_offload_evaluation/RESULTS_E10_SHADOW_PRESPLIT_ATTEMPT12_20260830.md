# E10 immutable-shadow 预拆分 attempt-12（2026-08-30）

## 结论

按要求只启动了一次完整 E10 qualification，没有重试。本轮在初始化阶段失败，尚未进入
2→4→6 GiB KV 请求序列，因此没有可报告的 Prefill、首次 Decode、steady Decode、
rebalance 或请求 E2E 性能数据，不能作为成功的 E10 端到端结果。

新增的“局部 CPU-shadow 释放前强制拆分 2 MiB 映射”路径已经实际执行，但首批 64 个
3 MiB 范围中只有 47 个拆分并释放成功，另有 17 个因内部状态不一致返回
`NV_ERR_INVALID_STATE`（状态 64）。程序按 fail-closed 规则终止，没有继续使用不完整映射。

本轮没有内核 Oops、NVIDIA Xid 或 GPU MMU fault，也没有再次锁死 UVM。实验结束后已
恢复 stock UVM，GPU 显存占用为 0 MiB，`nvidia-smi` 正常。

产物目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-12-shadow-presplit-qualification`

## 实验协议

- 完整序列启动：1 次；
- 自动重试：0；
- 统计重复：0；
- 计划 KV 档位：2→4→6 GiB；
- 计划专家档位：cold70→cold80→cold89；
- 专家策略：cold-frequency；
- workload 预测：关闭；
- CPU shadow：整 Parameter；
- 混合块双页表：启用；
- 自动漫游、prefetch、access-counter 和 thrashing：关闭。

## 实际执行结果

1. E10 专项静态预检 9 项通过，通用静态预检 8 项通过；
2. 自定义模块成功加载，SHA256 为
   `5bc2849bb361c1583649b0f61c0a33bd8266c1a707b68b2996686e8463879b7e`；
3. 模型 7 个 checkpoint shard 加载完成，权重加载耗时 13.91 秒；
4. 96 个完整 Parameter、28,991,029,248 bytes 的 CPU/GPU read-duplicate shadow
   建立成功；
5. cold70 初始 immutable CPU-shadow cutover 在首批 64 个范围失败；
6. KV 地址池、CUDA Graph 和请求测量均未开始。

### 首批 shadow cutover

| 指标 | 结果 |
|---|---:|
| 送入驱动的首批范围 | 64 × 3 MiB |
| 成功范围 | 47 |
| 失败范围 | 17 |
| 成功释放显存 | 147,849,216 bytes（141 MiB） |
| 成功保留 CPU shadow | 147,849,216 bytes（141 MiB） |
| 成功建立 CPU remote mapping | 147,849,216 bytes（141 MiB） |
| 最终状态 | 64 / `NV_ERR_INVALID_STATE` |

### 新增预拆分计数

| 计数 | 结果 |
|---|---:|
| `immutable_shadow_presplit_calls` | 64 |
| `immutable_shadow_presplit_successes` | 47 |
| `immutable_shadow_presplit_already_lower` | 0 |
| `immutable_shadow_presplit_failures` | 17 |
| `immutable_shadow_presplit_ns` | 1,950,848 ns（1.951 ms） |

这说明新机制已经命中，且自身耗时很小；失败原因不是预拆分太慢，而是 17 个目标块没有
满足当前拆分函数要求的完整、一致 GPU backing/页表状态。

## 失败位置与含义

失败发生在模型初始化的 cold70 初始释放，而不是 `cold70→cold80` 运行时切档。
驱动准备把某个 2 MiB 大映射改为下级页表时，会先确认该块中仍要保留的每一页都具有
一致的逻辑位置和真实物理页。17 个块没有通过该检查，因此驱动没有继续拆分或释放这些
块，最终上层检测到“没有释放全部 cold70 专家”并抛出 RuntimeError。

目前的预拆分修复避免了“先释放物理页、以后再拆大页”的危险顺序，但它仍假设整个
2 MiB 映射在进入拆分前具有单一且完整的 backing。真实模型中部分混合块在 full-parameter
shadow 建立后不满足该假设。下一步需要为失败检查增加分项原因计数，并让预拆分按照当前
2 MiB PTE 的真实目标逐页构造下级页表；在全部下级条目发布并等待完成后，才允许释放
目标 64 KiB 页。

## 错误处理与清理

- 结果文件成功记录原始 RuntimeError；
- `cleanup_errors=[]`；
- 驱动会话数：0；
- NVIDIA Xid：0；
- GPU MMU fault：0；
- 内核 Oops/NULL dereference：0；
- runner 返回码：139。该返回码来自错误记录和清理完成后的 controlled hard exit，
  内核日志只记录用户态 Python 在空地址退出，不是 UVM 内核空指针；
- stock UVM srcversion：`85A79790636BBD99BA3E43B`；
- 最终 GPU 显存：0 MiB；
- 残留计算进程：0。

标准 verifier 输出为 `status=failed`、`completed_sequences=0`、`requests_measured=0`、
`prefill_timing=null`、`decode_timing=null`。
