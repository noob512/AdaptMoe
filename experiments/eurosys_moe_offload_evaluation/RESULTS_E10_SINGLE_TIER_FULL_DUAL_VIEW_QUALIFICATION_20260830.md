# E10 单档位完整双页表切换 Qualification（2026-08-30）

## 结论

已按要求只为 `cold70` 建立一组 Prefill/Decode 页表，没有建立 `cold80` 或
`cold89` 页表。唯一一次完整成功 qualification 表明：完整双页表方案已经覆盖
`cold70` 涉及的全部 9,487 个 2 MiB 父入口；Prefill/Decode 切换不再重装
30,240 条消费者关系对应的 4 KiB 叶项，也不再为 Decode 建立 6,720 条 CPU
读取关系。

真正的父入口页表切换耗时为 14.73--15.55 ms；包含一次系统调用、锁和状态维护后，
三次实测切换分别为 36.59 ms、20.33 ms 和 34.94 ms。输出、CUDA Graph、GPU
页表和清理校验全部通过。

本轮外层首次 Decode 切换仍记录为 815.49 ms，但其中约 778 ms 是 qualification
启用的三轮 64 KiB 粒度热专家 CUDA 读取校验，并不是页表切换。因而本报告同时给出
“真实双页表切换时间”和“包含 qualification 校验的外层时间”，不把后者当作机制
本身的开销。

## 实验范围

- 模型：Qwen3-30B-A3B-FP8，单 GPU。
- 专家策略：cold-frequency，只注册 `cold70`。
- cold70：48 层、每层 70 个冷专家，共 3,360 个 layer-expert。
- 冷专家权重：15,854,469,120 bytes（14.765625 GiB）。
- 专家 Parameter 总量：28,991,029,248 bytes（约 27.0 GiB）。
- 只测 Prefill/Decode 阶段切换；不切换专家档位，不调整 KV 档位。
- 运行次数：一次完整成功 qualification，不做统计重复。

单档位清单：

```text
classes = { cold70 }
initial_class = cold70
```

运行目录：

```text
experiments/eurosys_moe_offload_evaluation/artifacts/
  e10-single-tier-full-dual-view-20260830-attempt-10-refresh-qualification
```

## 实现内容

### 1. 为所有相关 2 MiB 区域冻结两张完整子页表

每个 cold70 权重涉及的完整 2 MiB 区域都保留两张子页表：

- Prefill 表：冷专家页指向 GPU 缓存；
- Decode 表：冷专家页指向固定 CPU shadow；
- 普通页和热专家页：两张表保持同一映射。

运行时不修改块内 4 KiB 叶项，只替换 2 MiB 父入口。混合区域不再被跳过，所以
候选块数和成功准备块数相同，覆盖率为 100%。驱动把块列表和拓扑 generation
绑定；版本不匹配、状态不匹配或覆盖不完整时 fail-closed。

### 2. 只使用单个 cold70 档位

本轮计划类注册表只有 `cold70`，没有 cold70→cold80/cold89 指令，也没有运行时
专家增减。这样测到的是同一专家档位下 Prefill/Decode 两个视图的切换成本。

### 3. 消除 Decode 的逐范围 CPU 映射

PARK 成功后，9,487 个父入口已经发布预建 CPU 表。用户态直接使用该驱动证书，
不再调用 6,720-range remote-map 接口：

```text
remote_map_operation = prebuilt_full_dual_view
remote_map.ioctl_calls = 0
```

### 4. 消除 Prefill 的叶级页表重装

RESUME 只把相同 9,487 个父入口切回预建 GPU 表，不再为 30,240 条消费者关系逐段
重新安装 4 KiB 叶项。CPU/GPU 两套表在整个 session 生命周期中保留。

### 5. 分配变化后刷新双表

KV 地址池和 CUDA Graph 建立过程中，普通/热页的物理位置可能变化。为防止初始化时
冻结的表过期，新增了 sealed、parked 和 original 三种状态下的双表刷新；刷新后才
允许进入请求阶段。完整模型本轮四次刷新分别为 340.23--343.15 ms，均发生在正常
阶段切换计时之外。

初始化成本如下：

| 工作 | 时间 |
|---|---:|
| 初次 SEAL（包含首次双表准备） | 275.339 ms |
| 四次双表刷新合计 | 1,367.518 ms |
| 驱动统计的五次双表准备/刷新内部时间 | 1,631.950 ms |

这是把运行时工作提前到初始化的代价；本轮刷新次数还包含 Graph/KV 初始化过程中的
qualification 刷新，不会在每次 Prefill/Decode 切换时重新执行。

## 页表切换结果

| 方向 | 父入口数 | 驱动页表阶段 | ioctl | CPU range-map ioctl |
|---|---:|---:|---:|---:|
| Prefill→Decode（主请求） | 9,487 | 15.548 ms | 36.591 ms | 0 |
| Decode→Prefill（roundtrip） | 9,487 | 14.989 ms | 20.332 ms | 0 |
| Prefill→Decode（roundtrip） | 9,487 | 14.734 ms | 34.937 ms | 0 |

三次操作均为 `rm_status=0`，每次对父入口按 64 个一组失效旧地址缓存，共报告
149 个 TLB 批次。页表阶段的变化范围只有 0.81 ms，说明同一档位的双向切换稳定在
约 15 ms；ioctl 的 20--37 ms 还包含进入驱动、加锁、状态更新和返回结果。

与此前 cold70 仍有叶级处理的单轮 attempt-13 对照：

| 操作 | 旧路径 | 本轮完整双表 | 下降 |
|---|---:|---:|---:|
| Decode PARK 驱动操作 | 139.51 ms | 36.59 ms | 73.77% |
| Prefill RESUME ioctl | 251.01 ms | 20.33 ms | 91.90% |
| Prefill RESUME 页表阶段 | 241.22 ms | 14.99 ms | 93.79% |

这是不同单轮 qualification 之间的机制对照，不是多 seed 统计结果。

## 为什么外层仍显示数百毫秒

主请求 Prefill→Decode 的 815.486 ms 分解为：

| 工作 | 时间 | 是否属于双页表切换 |
|---|---:|---|
| GPU 静默同步 | 0.166 ms | 阶段边界固定成本 |
| PARK 区间 | 557.967 ms | 其中 ioctl 36.591 ms；其余主要是两轮热页读取校验 |
| 使用预建 CPU 表并生成结果 | 0.981 ms | 是，但没有 range-map ioctl |
| PARK 后第三轮热页读取校验 | 256.372 ms | 否，qualification 诊断 |
| 合计 | 815.486 ms | — |

`alias_detach_ns` 的 557.967 ms 不是驱动 PARK 时间。该计时窗口依次包含：

1. PARK 前读取并校验热专家；
2. 36.591 ms PARK ioctl；
3. PARK 后再次读取并校验热专家。

每轮校验按 64 KiB 粒度读取 13,136,560,128 bytes 热专家中的 200,448 个采样点。
remote-map 后又做第三轮相同校验。因此三轮资格校验合计约：

```text
(557.967 - 36.591) + 256.372 = 777.748 ms
```

脚本已关闭运行时全量驻留审计，但继承的热专家 CUDA-read gate 仍开启；字段名
`physical_audit_ns` 在这里也覆盖了这轮 CUDA-read gate，不能把 256.372 ms 理解为
重新安装页表。

Decode→Prefill 的外层 RESUME 为 539.635 ms，其中 ioctl 只有 20.332 ms；其余同样
主要是 RESUME 前后两轮热页读取校验。因而本轮能够证明父入口切换已经降到
20--37 ms ioctl / 约 15 ms 驱动页表时间，但不能用含 qualification gate 的
815 ms/540 ms 外层数字代表生产关键路径。

## 请求阶段数据

| 项目 | 时间 |
|---|---:|
| 主请求 Prefill | 334.526 ms |
| 主请求首次 Decode（含 815.486 ms qualification 切换） | 854.408 ms |
| 首次 Decode 纯计算部分 | 38.921 ms |
| steady Decode 中位 | 16.375 ms |
| steady Decode 吞吐 | 61.068 token/s |
| 主请求 cycle E2E | 1,613.370 ms |
| roundtrip Prefill（含 539.635 ms qualification RESUME） | 874.448 ms |
| roundtrip Decode（含 812.874 ms qualification PARK） | 850.428 ms |

roundtrip Prefill 减去 RESUME 后为约 334.812 ms，与主请求 Prefill 的 334.526 ms
一致；roundtrip Decode 减去 PARK 后为约 37.554 ms。这进一步说明数百毫秒增量来自
切换外围的资格检查，而不是模型计算或 CUDA Graph 重捕获。

## 正确性与清理

- runner：`status=ok`；离线 verifier：`status=ok`，errors 为空。
- 两次 PARK、一次 RESUME 均为 `rm_status=0`。
- `dual_view_full_coverage=true`，`dual_view_block_count=9487`。
- 主请求输出 token hash：
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`。
- CUDA Graph 捕获数切换前后均为 2，无重捕获。
- 无 Xid、MMU fault、NULL pointer 或内核 BUG。
- 38 项用户态/错误保护测试和 11 项 plan-class 定向测试通过；驱动模块编译成功。
- 最终恢复 stock UVM，GPU memory used 为 0 MiB，无残留 client。

## 结论边界

本轮已经证明单档位完整双页表的核心目标成立：全部相关混合 2 MiB 区域可以在
初始化阶段同时保存 CPU/GPU 子表，运行时通过父入口切换完成 Prefill/Decode 变更，
不再建立数千条 CPU 读取关系，也不再重装数万条 4 KiB 叶项。

本轮没有建立多个专家档位页表，也没有测 cold70→cold80→cold89。由于只运行一次，
当前结果是 qualification，不是统计性能结论。生产关键路径的下一项独立工作是关闭
或移出 64 KiB 热页 CUDA-read qualification gate；本轮不重复 GPU 实验，也不据此
宣称外层请求切换已经实测达到 20--37 ms。

## 产物

- 机器校验：
  `artifacts/e10-single-tier-full-dual-view-20260830-attempt-10-refresh-qualification/verification.json`
- 原始结果：
  `artifacts/e10-single-tier-full-dual-view-20260830-attempt-10-refresh-qualification/sequence/result.json`
- 单档位运行脚本：
  `../vllm_uvm_e10_multilevel_pressure/run_single_tier_full_dual_view.sh`
- 单档位计划：
  `../vllm_uvm_e10_multilevel_pressure/inputs/plan_classes_cold70_only.json`
- 混合块/刷新探针：
  `../uvm_static_alias_original_va/artifacts/full-dual-view-geometry-20260830-attempt-06-refresh/result.json`
