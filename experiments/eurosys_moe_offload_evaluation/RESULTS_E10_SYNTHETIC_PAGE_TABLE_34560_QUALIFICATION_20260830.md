# E10 34,560 段独立页表微实验（2026-08-30）

## 结论

已完成一次不加载 vLLM 模型的独立 GPU 微实验，没有统计重复。实验直接分配
`cudaMallocManaged` 数据，构造与 E10 cold80 相同规模的静态访问目录，然后只测一次
完整 RESUME 安装和一次完整 PARK 撤下。

结果确认：34,560 个 512 KiB 范围使用 4 KiB 页表重新安装时，完整 RESUME 为
802.04 ms，其中驱动记录的纯页表安装为 680.32 ms。即使没有模型、scheduler、
CUDA Graph 或推理计算，页表安装本身仍然需要数百毫秒。因此 E10 中 459 ms 的历史
测量不是 vLLM 调度产生的假开销。

本轮是唯一一次正式 GPU 测量，结果为 qualification，不代表多轮统计分布。

产物：

`experiments/eurosys_moe_offload_evaluation/artifacts/e10-synthetic-page-table-34560-20260830-attempt-01-qualification`

## 实验规模

| 项目 | 数值 |
|---|---:|
| vLLM/模型 | 未加载 |
| Managed consumer 数据 | 18,119,393,280 bytes（17.28 GiB） |
| GPU slot backing | 754,974,720 bytes（720 MiB） |
| consumer 范围 | 34,560 × 512 KiB |
| upload/backing 范围 | 1,440 × 512 KiB |
| 总目录记录 | 36,000 |
| 每个 backing 对应 consumer | 24 |
| 页表模式 | 4 KiB |
| 双页表父入口快路径 | 关闭 |
| 统计重复 | 0 |

34,560 × 512 KiB 与 E10 cold80 的 18,119,393,280 bytes 完全一致。按 4 KiB
粒度计算，需要描述 4,423,680 个叶级页面。

为了只测运行时安装，Managed 内存分配、CPU 驻留、slot 预分配和 36,000 条目录构建
都在计时开始前完成。目录以 parked 状态进入唯一一次 RESUME。

## RESUME 结果

| 工作 | 时间 | 占驱动总时间 |
|---|---:|---:|
| 写入 4 KiB 页表并清理旧地址缓存 | 680.315 ms | 84.83% |
| 撤下原 CPU 读取映射 | 115.374 ms | 14.39% |
| 构造批次和排序 | 2.491 ms | 0.31% |
| 前置准备 | 2.524 ms | 0.31% |
| 元数据 | 0.719 ms | 0.09% |
| 等待、记账及未单列部分 | 约 0.580 ms | 0.07% |
| 驱动总时间 | 802.004 ms | 100% |
| Python 外层墙钟 | 802.037 ms | — |

其他计数：

- 处理 consumer：34,560；
- 处理数据：18,119,393,280 bytes；
- transition batches：8；
- 地址缓存失效范围：34,560；
- `rm_status=0`。

Python 外层时间与驱动时间只差约 0.034 ms，证明本轮 802 ms 几乎全部位于驱动，
不是 Python 包装开销。

## PARK 结果

同一目录随后只执行一次 PARK：

| 指标 | 结果 |
|---|---:|
| Python 外层墙钟 | 95.188 ms |
| 驱动总时间 | 95.163 ms |
| ioctl | 95.060 ms |
| transition batches | 8 |
| 地址缓存失效范围 | 8,640 |

撤下比安装快得多，因为撤下可以按 2 MiB 覆盖范围使入口失效；重新安装则需要重新写出
每个 4 KiB 页面对应的目标地址和权限。

## 未计入页表测量的初始化

| 初始化工作 | 时间 |
|---|---:|
| Managed 虚拟地址分配 | 4.04 ms |
| 17.28 GiB consumer 与 720 MiB upload 的 CPU 驻留准备 | 3878.67 ms |
| 1,440 个 GPU slot 预分配 | 104.76 ms |
| 初始 2,880 条 MAP | 86.94 ms |
| parked 状态构建完整 34,560-consumer 目录 | 1330.51 ms |
| RESUME 前全部准备 | 5421.11 ms |

这些时间均在 RESUME 计时窗口之外，不能用于解释 802 ms 的页表安装结果。

## 与已有 E10 数据的关系

| 场景 | 34,560 段页表阶段 |
|---|---:|
| E10 attempt-05，无双页表 | 459.22 ms |
| 本轮纯合成数据，无双页表 | 680.32 ms |
| E10 attempt-13，部分双页表 | 258.61 ms |

三次结果的布局和运行状态不同，不能把差值当成统计加速比；但两次关闭双页表的独立
结果都明确位于数百毫秒量级。合成实验甚至更慢，主要额外开销是撤下 17.28 GiB
consumer 当前的 CPU 读取映射，本轮为 115.37 ms，而成功 E10 中该步骤只有约
1.7～3.4 ms。

已有 attempt-10 的双页表探针则表明，预建完成后切换 3,360 个 2 MiB 父入口，CPU/GPU
双向合计只需 13.76 ms。这说明瓶颈不是“修改任意页表都很慢”，而是运行时重新安装
442 万个 4 KiB 叶级页面很慢。

## 正确性与清理

- 分别抽查低、中、高三个 backing identity，对应 consumer 均读到正确值；
- runner 返回码为 0；
- verifier 为 `status=ok`；
- NVIDIA Xid、MMU fault、kernel Oops/NULL dereference 均为 0；
- session 和 slot backing 正常销毁；
- stock UVM 已恢复，srcversion 为 `85A79790636BBD99BA3E43B`；
- 最终 GPU 显存为 0 MiB，无残留计算进程。

正式 GPU 测量完成后，最初的日志验证表达式把正常日志中的 `debug:` 后缀误认为
`BUG:`。该纯文本验证错误已修复，并直接使用本轮已有产物重新验证，没有再次运行 GPU
实验。

机器可读结果：

- `artifacts/e10-synthetic-page-table-34560-20260830-attempt-01-qualification/result.json`
- `artifacts/e10-synthetic-page-table-34560-20260830-attempt-01-qualification/verification.json`
- `artifacts/e10-synthetic-page-table-34560-20260830-attempt-01-qualification/kernel.since.log`

## 最终判断

本轮可以回答“459 ms 是否可能是真实页表开销”：是。独立于 vLLM 的同规模实验中，
纯页表安装实际测得 680.32 ms。

要把阶段切换降到几十毫秒，不能继续运行时重装数百万个 4 KiB 叶级页面；必须提前保留
CPU/GPU 两套完整子页表，并将运行时操作缩减为少量 2 MiB 或更高层父入口切换。
