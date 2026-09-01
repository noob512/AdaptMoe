# E10 目录检查与 RESUME 优化 Qualification（2026-08-30）

## 结论

本次按 qualification 协议只完成一轮 2→4→6 GiB E10 序列，没有统计重复，也没有失败重试。实验完整成功，`verification.json` 为 `status=ok`。

优化对差量追加非常有效：

- cold70→cold80 的 4,500-entry 驱动 ioctl：370.704 ms → 77.572 ms，下降 79.07%。
- cold80→cold89 的 4,050-entry 驱动 ioctl：569.302 ms → 73.628 ms，下降 87.07%，即约 7.73 倍加速。
- cold80→cold89 完整专家档切换：653.894 ms → 175.870 ms，下降 73.10%。
- 4→6 GiB 完整 rebalance（含提前 RESUME、专家档切换和 KV 调整）：1172.637 ms → 743.637 ms，下降 36.58%。

RESUME 的目录检查也已消除，但本轮 RESUME 总时间没有下降：34,560-entry RESUME 为 471.785 ms，旧结果为 376.219 ms。新计时证明目录版本检查只占 0.000415 ms，而真实页表安装占 459.224 ms（97.34%）。因此剩余瓶颈已经不是目录检查，而是逐范围安装页表和地址缓存失效。

## 实现

1. MAP 差量地址检查改为新旧目录排序后线性相邻检查，替代新增关系之间的二次扫描和新×旧扫描。
2. 每个 backing slot 的范围计数和目录所有权改为 8,192-entry 计数/位图。
3. 每次 MAP 只扫描一次 8,192-entry slot 池并生成排序索引；4,050/4,500 条输入随后用二分查找，替代每条输入重新扫描 slot 池。
4. static-alias session 增加 topology generation/certificate。拓扑建立或改变时完整验证；PARK/RESUME 运行时只验证状态和 generation。
5. session 缓存 consumer/upload 数量和字节数，transition snapshot 变为常数时间。
6. RESUME 成功路径不再逐 block 重复汇入 tracker；批内提交后统一等待，失败路径仍保留保守清理。
7. transition ABI 增加 validation、整理/排序、preflight、metadata、CPU unmap、PTE map、wait 和 bookkeeping 分段计时。

主要实现位置：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h`
- `workloads/vllm/vllm/vllm/device_allocator/uvm_static_alias.py`
- `workloads/vllm/vllm/tests/test_uvm_static_alias_ioctl.py`

## 实验协议

- 模型：Qwen3-30B-A3B-FP8。
- 专家策略：cold-frequency，固定 cold70、cold80、cold89 三档。
- KV：初始化保留 8 GiB VA，实际 GPU 前缀按 backup 队列压力从 2→4→6 GiB 调整。
- 压力来源：真实 scheduler backup 队列；未使用预测。
- 每档一个请求，5 个 warmup Decode step，16 个 measured Decode step。
- CPU immutable shadow、预生成档位指令和 quiesced delta 均启用。
- 自动 access-counter migration、prefetch 和 thrashing 均关闭；manual fault mode=2。
- 旧基准：`e10-kv-2-4-6-pressure-20260830-attempt-04-post-audit-removal-qualification`。
- 新结果：`e10-kv-2-4-6-pressure-20260830-attempt-05-indexed-map-cached-resume-qualification`。

## 端到端结果

| KV 档 | Prefill E2E | 首次 Decode 切换 | steady Decode 中位 | 请求 E2E |
|---|---:|---:|---:|---:|
| 2 GiB | 337.48 ms | 383.33 ms | 19.62 ms | 1236.37 ms |
| 4 GiB | 1082.10 ms | 438.13 ms | 23.22 ms | 2121.81 ms |
| 6 GiB | 1178.88 ms | 487.41 ms | 26.34 ms | 2344.61 ms |

与 attempt-04 单轮结果相比：

| KV 档 | Prefill E2E 变化 | steady Decode 变化 | 请求 E2E 变化 |
|---|---:|---:|---:|
| 2 GiB | +0.56% | +9.69% | +10.64% |
| 4 GiB | -14.40% | +6.67% | -1.72% |
| 6 GiB | -26.64% | +0.14% | -15.31% |

这些是各一次运行之间的直接比较，不是统计结论。2 GiB 没有发生本次要优化的档位追加；它的退化以及 steady Decode 的变化不能归因于检查算法，需要重复实验才可区分运行波动。

## Rebalance 分解

| 切换 | 部分 | attempt-04 | 本轮 | 变化 |
|---|---|---:|---:|---:|
| 2→4 GiB | 提前 RESUME（30,240 entries，ioctl） | 327.323 ms | 401.812 ms | +22.76% |
| 2→4 GiB | 差量追加（4,500 entries，ioctl） | 370.704 ms | 77.572 ms | -79.07% |
| 2→4 GiB | 完整专家档切换 | 458.607 ms | 186.395 ms | -59.36% |
| 2→4 GiB | 完整 rebalance（含 RESUME） | 874.322 ms | 687.141 ms | -21.41% |
| 4→6 GiB | 提前 RESUME（34,560 entries，ioctl） | 376.219 ms | 471.785 ms | +25.40% |
| 4→6 GiB | 差量追加（4,050 entries，ioctl） | 569.302 ms | 73.628 ms | -87.07% |
| 4→6 GiB | 完整专家档切换 | 653.894 ms | 175.870 ms | -73.10% |
| 4→6 GiB | 完整 rebalance（含 RESUME） | 1172.637 ms | 743.637 ms | -36.58% |

两次档位切换均命中预生成指令和 immutable CPU shadow：

- 2→4 GiB 直接释放并保留 2,160 MiB CPU shadow，source refill 为 0。
- 4→6 GiB 直接释放并保留 1,944 MiB CPU shadow，source refill 为 0。
- 专家切换内部没有 PARK/RESUME，也没有 GPU→CPU 权重复制。

## 34,560-entry RESUME 的新分解

| 驱动阶段 | 时间 | 占 471.785 ms ioctl |
|---|---:|---:|
| generation/状态检查 | 0.000415 ms | <0.001% |
| 构造批次与排序 | 5.331 ms | 1.13% |
| 目标状态预检 | 2.721 ms | 0.58% |
| 恢复 slot 元数据 | 0.769 ms | 0.16% |
| 撤下 CPU 直接访问关系 | 3.410 ms | 0.72% |
| 安装 GPU 缓存页表关系 | 459.224 ms | 97.34% |
| 最终 tracker wait | 0.0036 ms | <0.001% |
| bookkeeping | 0.0024 ms | <0.001% |

这说明原先的前后目录全量检查已经不再是 RESUME 瓶颈。即使继续删除检查，也不会显著降低 471.785 ms；下一步必须减少实际写入的 34,560 条页表关系，例如让 PARK 保留一套不可达的已构建页表，在 RESUME 时进行更粗粒度的发布，而不是重新逐范围安装。

## 正确性与清理

- verifier：`status=ok`，完成序列数 1，统计重复数 0。
- 三档输出 token hash 完全一致：`d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- CUDA Graph capture 计数始终为 6，档位切换没有重捕获。
- 所有差量和 RESUME 均 `rm_status=0`，无失败条目。
- kernel log 无 Xid、MMU Fault、BUG 或 general protection fault。
- 实验结束后恢复 stock UVM，无 GPU/UVM client，GPU 显存 0 MiB。

## 验证

- `git diff --check`：通过。
- `test_uvm_static_alias_ioctl.py`：33 passed。
- E10 定向静态预检：9 passed，159 deselected。
- 内核模块完整构建：通过；仅保留原有 frame-size warning。

完整机器可读结果：

- `experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-05-indexed-map-cached-resume-qualification/verification.json`
- `experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-05-indexed-map-cached-resume-qualification/sequence/result.json`
