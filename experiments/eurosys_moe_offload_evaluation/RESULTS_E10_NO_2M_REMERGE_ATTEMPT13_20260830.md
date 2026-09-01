# E10 禁止局部 shadow 映射重新合并 2 MiB：attempt-13（2026-08-30）

## 结论

已修复“局部 immutable CPU-shadow 预拆分完成后，CPU 读取映射又把下级页表合并成
2 MiB”的错误，并按要求只运行了一次完整 E10 GPU qualification，没有重试和统计重复。

本轮成功完成 cold70→cold80→cold89 与 2→4→6 GiB KV 全序列，verifier 为
`status=ok`。预拆分共调用 10,482 次，其中实际拆分 7,027 次、已有下级页表 3,455 次，
失败为 0；CPU 映射后的 10,482 次结构检查也全部通过。attempt-12 中的 17 个
`NV_ERR_INVALID_STATE` 已消除。

这轮证明修复后的页表状态转换能够正确完成一次端到端序列。由于每档只有一个请求，
下列延迟是单轮 qualification 数据，不是统计性能结论。

产物目录：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-13-no-2m-remerge-qualification`

## 修复内容

错误的旧顺序是：

```text
拆开 2 MiB 映射
→ 释放其中一部分 GPU 页
→ 建立 CPU 读取关系时重新合并成 2 MiB
→ 同块下一位专家再释放时遇到混合驻留并失败
```

修复后的顺序是：

```text
局部释放前拆开 2 MiB 映射并等待生效
→ 更新驻留关系
→ 若中间路径重新合并，则在 GPU 页仍可安全引用时再次拆开
→ 建立 CPU 读取关系时强制禁止 2 MiB 合并
→ 映射完成后确认下级页表仍存在
→ 只有全部检查通过后才允许释放目标 GPU 页
```

具体实现包括：

- 局部 immutable-shadow CPU 映射传入 `forbid_2m_merge=true`；
- 通用映射逻辑在该模式下强制使用下级页表，不发布新的 2 MiB 大页入口；
- CPU 映射完成后检查 `pte_is_2m=false`、4 KiB 子表存在且已经启用；
- 任一步失败都会在释放更多 GPU 页之前安全终止；
- 新增预拆分、强制下级映射和映射后检查计数。

修改文件：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`
- `experiments/uvm_static_alias_original_va/test_kernel_error_guards.py`

修复后模块编译成功；4 项新增错误保护测试与 131 项既有 UVM 相关回归测试全部通过，
合计 135 项通过。

## 实验协议

- 完整序列：1 次；
- 自动重试：0；
- 统计重复：0；
- 每个 KV 档位请求：1；
- KV 档位：2→4→6 GiB；
- 专家档位：cold70→cold80→cold89；
- 专家选择：cold-frequency；
- 压力来源：真实 scheduler backup 队列消耗；
- workload 预测：关闭；
- KV 预留虚拟地址池：8 GiB；
- CPU shadow 与同档位 CPU/GPU 双页表：启用；
- 自动预取、access-counter、thrashing 和缺页自动迁移：关闭。

自定义模块 SHA256：
`28b93ecf272e29143dcde1fdb70445d804916ea5b1d6898b28587b6f90980489`。

## 修复路径实测计数

| 指标 | 结果 |
|---|---:|
| `immutable_shadow_presplit_calls` | 10,482 |
| 实际从 2 MiB 拆成下级页表 | 7,027 |
| 调用时已经是下级页表 | 3,455 |
| 预拆分失败 | 0 |
| 预拆分累计时间 | 237.156 ms |
| 强制禁止重新合并的映射 | 10,188 |
| CPU 映射后结构检查 | 10,482 |
| 映射后检查失败 | 0 |

`7,027 + 3,455 = 10,482`，说明所有局部块都以有效的下级页表进入释放和 CPU
读取阶段，没有留下未处理空洞。

## 端到端结果

| KV 档 | Prefill E2E | 首次 Decode 切换 | 首次 Decode E2E | steady Decode 中位 | 请求 E2E |
|---|---:|---:|---:|---:|---:|
| 2 GiB | 343.45 ms | 296.09 ms | 350.10 ms | 19.85 ms | 1229.80 ms |
| 4 GiB | 977.07 ms | 322.54 ms | 384.06 ms | 23.41 ms | 1940.25 ms |
| 6 GiB | 1045.84 ms | 351.09 ms | 422.99 ms | 26.97 ms | 2172.23 ms |

2 GiB 是初始档位，没有档位 rebalance。4 GiB 和 6 GiB 的 Prefill E2E 包含进入
Prefill 前恢复旧目录、增加下一档冷专家以及 KV 物理布局扩容。

## Rebalance 分解

| 操作 | 2→4 GiB | 4→6 GiB |
|---|---:|---:|
| 恢复上一档 Prefill 目录 | 251.01 ms | 269.51 ms |
| 专家档位切换及 RPC 管理 | 242.87 ms | 223.31 ms |
| KV 物理布局调整 | 88.03 ms | 117.75 ms |
| 合计 | 581.91 ms | 610.57 ms |

专家档位切换继续命中预生成指令和 CPU shadow：

| 切换 | 新增关系 | 差量驱动 ioctl | 直接释放 GPU 显存 |
|---|---:|---:|---:|
| cold70→cold80 | 4,500 | 116.24 ms | 2,160 MiB |
| cold80→cold89 | 4,050 | 108.01 ms | 1,944 MiB |

这里没有执行 GPU→CPU 权重复制；CPU shadow 已经存在，释放的是对应 GPU 物理页。

## Prefill/Decode 页表切换

Prefill 恢复：

| KV 档 | 恢复关系数 | 恢复总时间 | 其中页表阶段 |
|---|---:|---:|---:|
| 4 GiB 前的 cold70 | 30,240 | 251.01 ms | 241.22 ms |
| 6 GiB 前的 cold80 | 34,560 | 269.51 ms | 258.61 ms |

首次 Decode 切换：

| KV 档 | PARK/切到 CPU 方案 | 建立 CPU 读取关系 | 切换合计 |
|---|---:|---:|---:|
| 2 GiB | 139.51 ms | 156.16 ms | 296.09 ms |
| 4 GiB | 149.90 ms | 172.27 ms | 322.54 ms |
| 6 GiB | 160.74 ms | 189.96 ms | 351.09 ms |

双页表父入口切换保持启用。本轮累计准备 10,344 个块、复用 15,418 个块，执行
51,524 次父入口写入；运行时切换累计 102.99 ms。剩余 Prefill/Decode 切换时间仍主要
来自未被父入口快路径完全替代的页表工作和分批地址缓存清理。

## 正确性、错误保护与清理

- cold70、cold80、cold89 输出 token hash 完全一致：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`；
- CUDA Graph 捕获数从模型加载后到实验结束均为 6，无运行时重捕获；
- NVIDIA Xid：0；GPU MMU fault：0；内核错误：0；
- runner 返回码：0；`cleanup_errors=[]`；驱动 session 最终为 0；
- 已恢复 stock UVM，srcversion 为 `85A79790636BBD99BA3E43B`；
- 最终 GPU 显存为 0 MiB，残留计算进程为 0。

机器可读结果：

- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-13-no-2m-remerge-qualification/verification.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-13-no-2m-remerge-qualification/protocol.json`
- `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-13-no-2m-remerge-qualification/sequence/result.json`

## 结论边界

本轮可以证明：禁止重新合并的修复消除了已知 mixed-residency 失败，并使完整 E10
序列正确跑通。它不能单独证明性能稳定性或显著加速，因为没有重复实验、随机化顺序或
置信区间；本轮数值只能用于功能 qualification 和后续瓶颈定位。
