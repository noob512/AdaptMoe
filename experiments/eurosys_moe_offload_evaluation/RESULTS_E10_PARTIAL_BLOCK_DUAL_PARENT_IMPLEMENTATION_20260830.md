# E10 混合 2 MiB 块双页表实现与验证（2026-08-30）

## 结论

已实现混合 2 MiB 区域的 CPU/GPU 双页表。一个 2 MiB 块不再要求全部由冷专家
占据：驱动为完整块保留两张 4 KiB 子页表，并用位图只标记需要切换的冷专家
页面。普通页和热专家页在两张表中保持相同，PARK/RESUME 仅切换父入口。

最终方案完成了一次独立 GPU 探针，结果正确：同一 2 MiB 块内的 512 KiB 冷页
在 CPU 与两个 GPU slot 之间正确切换，另外 1.5 MiB 普通/热页始终保持原值。
PARK 为 0.716 ms，RESUME 为 0.341 ms，未出现 GPU fault 或输出错误。

本轮没有重复运行完整 E10。此前 attempt-08 是本次开发期间唯一一轮完整 E10，
但它暴露并帮助定位了 UVM managed range 被按专家切碎的问题；最终的整 Parameter
策略是在该轮之后完成，并由小探针验证。因此 attempt-08 不能作为最终方案的完整模型
性能数据。

## 实现

### 1. 混合块双页表

每个完整 2 MiB UVM 块新增：

- 冷专家 4 KiB 页位图；
- 完整 GPU 子页表；
- 完整 CPU 子页表；
- 当前启用 CPU 视图还是 GPU 视图的状态。

初始化时先复制整个 GPU 子页表，再仅将位图标记的冷页改为 CPU shadow 地址。
因此两张表中只有冷页不同，其他页面完全相同。运行时每个块只写一个父入口，
并按 64 个父入口一批统一清除旧地址缓存。

档位扩大时只将新冷页补入位图和 CPU 表；销毁时安全恢复可见 GPU 表，并释放
隐藏的 CPU 表。无法满足父入口快路径条件的特殊块仍自动退回旧的逐页路径，不会
导致模型初始化失败。

主要内核位置：

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c`
- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`

### 2. 保持完整 Parameter 的 UVM 几何

attempt-08 中混合块覆盖率没有提高。原因不是位图实现失效，而是用户态此前对每个
专家分别设置 ReadMostly，UVM 因此在专家边界切分内部 managed range。一个地址上
完整的 2 MiB 区域会跨越多个较小 UVM 块，无法挂接一张完整的父入口双表。

新增环境变量：

```text
VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW=1
```

启用后，对每个完整专家 Parameter 一次性设置 ReadMostly 并建立 CPU shadow，避免
按专家切碎 managed range；随后仅将当前冷专家页切为 CPU-only，热页仍保留 CPU/GPU
双副本。E10 脚本已启用该模式，并在协议中记录：

```text
full_parameter_shadow=true
managed_policy_granularity=whole_parameter
```

该方案不增加显存，但为本模型保留约 28,991,029,248 bytes（27.0 GiB）的 CPU
Parameter shadow。

主要用户态位置：

- `workloads/vllm/vllm/vllm/model_executor/models/utils.py`
- `experiments/vllm_uvm_e10_multilevel_pressure/run_experiment.sh`

## 最终针对性 GPU 探针

产物：

`experiments/uvm_static_alias_original_va/artifacts/dual-parent-switch-20260830-attempt-05-full-policy-partial-block/result.json`

探针仅完整执行一次，未做统计重复。

| 项目 | 结果 |
|---|---:|
| 2 MiB 块内冷页 | 512 KiB |
| 同块普通/热页 | 1.5 MiB |
| 准备成功块 | 1 |
| 跳过块 | 0 |
| PARK 父入口切换 | 0.716 ms |
| RESUME 父入口切换 | 0.341 ms |

数值正确性：

- 冷页：原 CPU 值 29，GPU slot 值依次为 73 和 91，切换后均正确；
- 热页前缀：始终为 41；
- 热页后缀：始终为 53；
- 冷页驻留：CPU-only；
- 热页驻留：CPU/GPU 双驻留；
- `rm_status=0`，无 GPU 错误；
- 清理后恢复 stock UVM，GPU 0 MiB。

## attempt-08 完整 E10（定位用，不代表最终策略性能）

产物：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-kv-2-4-6-pressure-20260830-attempt-08-partial-dual-parent-qualification/verification.json`

该轮 verifier 为 `status=ok`，输出 hash 正确、Graph 无重捕获、无 GPU 页表故障。
由于运行时仍使用按专家切分的 managed range，实际双表切换块数与 attempt-07 相同，
所以没有测到新增混合块覆盖带来的性能提升：

| KV 档 | Prefill E2E | 首次 Decode 切换 | 请求 E2E |
|---|---:|---:|---:|
| 2 GiB | 337.34 ms | — | 1136.15 ms |
| 4 GiB | 943.82 ms | 320.06 ms | 1859.06 ms |
| 6 GiB | 1037.91 ms | 360.21 ms | 2075.42 ms |

最终整 Parameter 策略尚未进行第二轮完整 E10，因此不能把小探针的亚毫秒结果直接
外推为完整模型的最终 PARK/RESUME 延迟。

## 静态验证

- 新增 full-Parameter shadow 单元测试：1 passed；
- `test_uvm_static_alias_ioctl.py`：33 passed；
- E10 定向 static preflight：9 passed，160 deselected；
- 内核模块此前已完整编译成功，仅有原有 stack-frame warning；
- `git diff --check`：通过。
