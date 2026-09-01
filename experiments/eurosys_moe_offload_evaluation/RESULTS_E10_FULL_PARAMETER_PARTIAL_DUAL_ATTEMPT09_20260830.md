# E10 整 Parameter 混合块双页表端到端 attempt-09（2026-08-30）

## 结论

按最终配置只启动了一次完整 E10，但该轮在模型初始化的首个冷专家释放阶段失败，
尚未进入 Prefill、Decode 或 2→4→6 GiB KV 档位切换。因此本轮没有可报告的
端到端性能和完整模型切换延迟，不能作为成功 E10 结果。

失败不是 CPU shadow 建立失败：96 个完整 Parameter、共 28,991,029,248 bytes
（27.0 GiB）已成功建立 CPU/GPU 双副本并通过驻留检查。失败原因是初始化代码随后
仍调用旧的“GPU→CPU 迁移后释放”接口；该接口不接受已经同时存在 CPU/GPU 副本的
页面状态，首个 3 MiB 冷专家因此返回 `NV_ERR_INVALID_STATE`（`rm_status=64`）。

该路径已修复为调用 immutable CPU shadow cutover：直接保留已有 CPU 副本、释放
GPU 副本并安装只读 CPU 映射，不再发起 GPU→CPU 数据迁移。修复已通过单元测试和
E10 静态预检，但遵守单轮要求，没有自动运行第二次完整 E10。

## 本轮协议

- 完整序列：1；
- 统计重复：0；
- KV：2→4→6 GiB；
- 专家档位：cold70→cold80→cold89；
- 专家策略：cold-frequency；
- workload 预测：关闭；
- CPU shadow：整 Parameter；
- 混合块双页表：启用；
- 父入口切换批次：每批 64 个 2 MiB 块。

产物目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-09-full-parameter-partial-dual-qualification`

## 失败位置

成功完成：

1. 加载自定义 UVM；
2. 加载 Qwen3-30B-A3B-FP8 的 7 个权重分片；
3. 建立 96 个完整 Parameter 的 ReadMostly CPU shadow；
4. 查询确认全部 27.0 GiB 同时存在 CPU/GPU 副本。

随后在 `finalize_cpu_offload_graph_static_alias()` 中处理第一个 cold70 专家：

```text
Parameter: model.layers.0.mlp.experts.w13_weight
expert: 3
range: 0x7b61ba900000 + 0x300000（3 MiB）
attempted_blocks: 2
success_blocks: 0
failure_blocks: 2
rm_status: 64
```

双页表准备和切换尚未发生，相关统计均为 0：

- `slot_alias_dual_prepare_calls=0`；
- `slot_alias_dual_switch_calls=0`；
- `slot_alias_dual_pde_writes=0`。

因此不存在本轮 Prefill、Decode、rebalance 或请求 E2E 数据。

## 根因

旧初始化流程原本假设冷专家只有 GPU 副本：

```text
GPU 权重 → 搬到 CPU → 释放 GPU 页面
```

整 Parameter shadow 优化提前形成的状态是：

```text
CPU 副本 + GPU 副本同时存在
```

原有 `release_uvm_regions()` 接口用于前一种状态，不是为第二种重复驻留状态设计的，
所以虽然 shadow 证书正确，旧释放操作仍拒绝该范围。这是初始化调用路径选择错误，
不是显存不足、地址不对齐或双页表父入口实现错误。

进程在抛出上述 RuntimeError 后的解释器清理阶段又发生一次 CPU 侧空指针段错误；
内核日志没有 NVIDIA Xid 或 GPU MMU fault。主失败仍是更早、已有完整结构化记录的
`rm_status=64`。

## 修复

`_release_uvm_original_alias_consumer_gpu_chunks()` 现在检查整 Parameter shadow 证书：

- 新冷专家：调用 `driver_cutover_ranges_to_immutable_cpu_shadow()`；
- 已属于旧冷档的范围：只验证其仍为 CPU-only，不重复 cutover；
- 完整校验释放字节数、CPU shadow 保留字节数和远程映射字节数；
- 任一范围不完整仍 fail-closed。

修复位置：

- `workloads/vllm/vllm/vllm/model_executor/models/utils.py`
- `workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py`

修复后验证：

- 相关定向单元测试：4 passed；
- E10 static preflight：9 passed，161 deselected；
- `git diff --check`：通过。

## 清理状态

- alias session：0；
- cleanup errors：0；
- GPU：0 MiB、无计算进程；
- 已恢复 stock UVM，srcversion `85A79790636BBD99BA3E43B`。
