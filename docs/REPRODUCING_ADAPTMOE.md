# 构建、验证与复现 AdaptMoE

本页描述当前代码的入口，不自动运行模型、不下载模型，也不重新生成论文中的全部图表。
[项目机制与边界](ADAPTMOE_PROJECT_GUIDE.md) 给出了论文到源码的对应关系。

## 环境与源码

已使用的环境是单张 RTX 5090（32,607 MiB）、NVIDIA 595.71.05、Linux
6.13.12、Python 3.12。vLLM 构建依赖见
[`pyproject.toml`](../workloads/vllm/vllm/pyproject.toml) 与
[`requirements/`](../workloads/vllm/vllm/requirements/)，其中当前 build-system
固定 PyTorch 2.9.0。CUDA toolkit、PyTorch 和 vLLM 扩展需相互兼容。
这不是对其他 GPU/驱动/多 GPU 配置的兼容承诺。

```bash
git clone https://github.com/noob512/AdaptMoe.git
cd AdaptMoe
```

vLLM 和驱动源码已放在树内。只运行四机制比较不需要初始化 baseline 子模块；
KTransformers/MoE-Infinity 复现才需要 `git submodule update --init --recursive`。

在 `workloads/vllm/vllm/` 建立虚拟环境，并按该目录上游构建说明安装兼容 CUDA/PyTorch
和 build dependencies。使用此树的 editable/source build，例如在依赖齐备后：

```bash
cd workloads/vllm/vllm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/build.txt
python -m pip install --no-build-isolation -e .
cd ../../..
```

不要让另一份 PyPI vLLM 遮蔽修改过的模块。可用下面的 CPU import 检查路径：

```bash
workloads/vllm/vllm/.venv/bin/python -c 'import vllm; print(vllm.__file__)'
```

## 驱动与 allocator 构建

须安装匹配的 NVIDIA 595.71.05 基础驱动、运行内核的 headers 和 build tools。
本仓库修改的 UVM 与基础驱动之间存在 ABI 依赖。

```bash
make -C kernel-module/nvidia-module modules -j4
make -C workloads/vllm/vllm/uvm_test uvm libuvm_slot_cuda_fence_kernels.so
```

内核 headers 不在默认位置时，给第一个命令传入 `SYSSRC=/path/to/kernel/headers`。
CUDA 不在默认位置时，给第二个命令传入 `CUDA_HOME` 与 `NVCC`。
完整模块构建会生成 NVIDIA 导出符号；直接只构建 `nvidia-uvm` 而不提供基础驱动的
符号表可能在 modpost 阶段出现未定义的 `nvUvmInterface*`。

本次整理执行了完整模块构建，未安装/加载新模块。模型实验 wrapper 会在 GPU 无
compute/UVM client 时临时加载 custom UVM，结束时恢复 stock UVM 和原先运行的
exporter 服务。该操作需要 sudo 和独占 GPU，不能与其他 GPU 服务并行运行。

## 不运行模型的检查

```bash
workloads/vllm/vllm/.venv/bin/python -m pytest -q \
  workloads/vllm/vllm/tests/test_cpu_offload_exact_static.py \
  workloads/vllm/vllm/tests/test_uvm_static_alias_ioctl.py \
  workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py \
  workloads/vllm/vllm/tests/v1/worker/test_uvm_eager_plan_classes_e13.py \
  workloads/vllm/vllm/tests/v1/engine/test_uvm_adaptive_kv.py \
  workloads/vllm/vllm/tests/v1/core/test_uvm_dynamic_pool_queues.py \
  workloads/vllm/vllm/tests/v1/core/test_uvm_dynamic_pool_policy.py
```

本次上述集合结果为 **366 passed**。这些 mock/contract 测试验证阈值、状态拒绝、ABI、
差集批处理、KV 队列和容量边界；不能替代真实模型的输出一致性与性能验证。

四机制输入可以在无模型、无数据集、无 GPU 的情况下准备：

```bash
PREPARE_ONLY=1 bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/prepared-inputs 1 20
```

`PREPARE_ONLY=1` 的目录用于检查输入，不是一次完成的实验。正式运行须另选新目录。

## 当前 mixed-scheduling 四机制实验

准备本地 DeepSeek-V2-Lite-Chat snapshot 与 ShareGPT V3 数据集。
冻结 trace 已包含 prompt token IDs，因此四臂使用同一请求内容和顺序；完整数据集
留在本地，用于 runner 的来源一致性检查。

```bash
export MODEL=/absolute/path/to/DeepSeek-V2-Lite-Chat
export SHAREGPT_DATASET=/absolute/path/to/ShareGPT_V3_unfiltered_cleaned_split.json

ALLOW_GPU_RUN=1 SAMPLES=1 \
bash experiments/adaptmoe_token_threshold_sharegpt_compare/run_kv1_compare.sh \
  /absolute/new/artifacts/kv1-qps1 1
```

四个进程依次为 Native UVA、Native Copy、AdaptMoE fixed K1、AdaptMoE dynamic
K1↔K2。默认 `SAMPLES=1`，每臂只运行一次，不自动复测。
需要扫多个到达率时，在同一命令末尾传 `1 2 4 8 10 20 30`；这些 rate 都是全新运行，
不复用旧结果。`SAMPLES=2` 才会显式请求第二遍反序运行。

| 参数 | 固定值 |
|---|---|
| 请求 | 128 条 ShareGPT，1024 输入 / 16 输出 |
| 到达 | 第 i 个请求在 `i / QPS` 到达；QPS 是 offered rate |
| 模型 | DeepSeek-V2-Lite-Chat BF16 |
| 调度 | C128，8192-token budget，max_model_len=1088 |
| Attention / Graph | TRITON_MLA，eager，CUDA Graph `NONE` |
| Prefix cache / 采样 | 关闭；greedy，ignore EOS |
| Access thresholds | PARK≤16，RESUME≥128，区间内保持原状态 |
| 固定 KV | 2155 usable blocks，34,480 tokens，约 1 GiB |
| 动态 KV | 2155↔4313 usable blocks，K1↔K2 |
| AdaptMoE 专家 | K1/cold8，K2/cold11，逐层 cold slices |
| Native 专家 | UVA/Copy 使用同 7 个 whole Parameters，较 cold8 多卸载 2.56% |

`CPUSET`、`VLLM_PY`、`CUSTOM_UVM`、`TIMEOUT_SEC` 可按环境覆盖；默认 CPUSET 为
0–13。模型和 dataset 路径必须显式设置。不要照抄历史 frozen env 中的绝对路径直接
启动服务；wrapper 会覆盖当前实验需要的 plan、phase、Graph 与路径配置。

## 结果与正确性

新输出目录保留 `sweep_protocol.json`、每档输入/执行状态、逐臂 `result.json`、
manifest、日志、内核快照、清理证明，以及汇总 `summary.json`、`summary.csv`、
`output_checks.csv` 和 `REPORT.md`。不覆写已有成功或失败目录。

actual QPS = 128 / 从首个计划到达到最后请求完成的时长，包含尾部 drain。
TTFT 包含到达、admission 和排队等待；TPOT 来自后续 token 的间隔。
计时包含真实 resize，未减去扩容时间；无额外真实 conditioning，请勿标为纯稳态。

校验器检查完整请求数、trace hash、到达时间、卸载字节、Graph 禁用、copy/remote
路径、KV 队列/驻留、容量事件、内核 fault 和最终清理，并比较完整输出与首 token。
出现输出差异时保留计数并输出 `ok_with_output_drift`；这不是严格数值等价通过。

本次公开入口复用已运行过的底层 runner，并补充无历史 artifact 依赖的启动包装。
包装脚本通过语法、CPU 输入准备和已有 artifact 的离线校验；本次整理没有重新执行
GPU 性能实验，也没有把历史速度倍率写成新的测量结果。

## 旧实验与数据保留

较早的双 Graph、placement、KV round-trip 协议从
[`experiments/reproduce/README.md`](../experiments/reproduce/README.md) 进入。
其数据口径与本页的 mixed/eager/1024-token trace 不同。

模型、完整 ShareGPT、raw artifact、性能报告快照、PDF 草稿、profile 和本地构建文件
不随本次增量上传，但全部保留在本地。本仓库历史已经受控的 vendor/source 内容保留，
没有通过清理或重写 Git 历史来缩减它们。
