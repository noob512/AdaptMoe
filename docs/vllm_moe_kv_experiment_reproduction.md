# vLLM MoE 专家卸载与 KV Cache 空间复用实验复现指南

> **文档快照日期：2026 年 8 月 3 日**
> **适用代码基线：`23c4cc939515a1a1a49c045b1de1e5c20321de7e`（`feat: 修复bug`）**
> **实验目的：** 固化当前 Qwen3-30B-A3B-FP8、UVM 驱动和 vLLM 修改版上的 MoE 冷专家卸载实验，使后续精简代码时仍可准确恢复实验环境、操作和判定方法。

本文记录四类相互关联的实验：

1. **空间复用正确性验证**：KV 物理页卸载后，专家权重能否复用释放的 GPU 物理空间，且不驱逐热点数据。
2. **三种卸载机制对比**：原生 UVA offload、warm-up 后静态卸载冷专家、随 KV 压力逐波卸载冷专家。
3. **固定 KV 容量下的静态 4/7/10 GB 卸载扫描**：检验当 KV cache 能容纳目标并发时，少卸载专家是否更快。
4. **一次性 10 GB 与阶梯式 4→7→10 GB 对比**：检验较小的中间卸载量是否能改善对应并发区间的 decode 性能。

本文的“4/7/10 GB”均指**按专家权重估算的卸载量**，不是 KV cache 的容量。静态扫描和阶梯实验中 KV cache 均固定为 **4 GiB（`4294967296` bytes）**。

---

## 1. 实验边界与术语

### 1.1 测试对象

- 模型：`Qwen/Qwen3-30B-A3B-FP8`
- 模型结构假设：48 层、每层 128 个专家。
- 每个“冷专家”估算大小：`4718592` bytes。
- 冷专家计划以每层路由频率从低到高排序；同一条 pilot routing trace 生成的 `k18`、`k31`、`k45` 计划是嵌套的。
- 推理采用单卡、`tensor_parallel_size=1`、`enforce_eager=True`、关闭 prefix caching、`temperature=0.0`、`ignore_eos=True`、随机种子 `0`。

### 1.2 三种机制的定义

| 名称 | runner case | 行为 | 用途 |
|---|---|---|---|
| 原生 UVA | `native_uva` | 使用 `VLLM_CPU_OFFLOAD_BACKEND=uva` 与 `cpu_offload_gb`；不启用本项目的 UVM 专家/KV 管理 | 原生 CPU offload 参照组 |
| 静态冷专家卸载 | `static_custom` | warm-up 后依据显式 cold plan 一次释放冷专家，并在第一业务 phase 前把 KV 物理池调到目标大小 | 固定最终卸载量的对照组 |
| 渐进/自适应卸载 | `adaptive` | KV 目标块数逐 wave 增长；每个 wave 最多每层释放一个冷专家 | 将专家释放与 KV 空间压力同步 |

“自适应”在这里描述的是**释放时机与 KV 动态池联动**。本指南的 4→7→10 实验使用已固定的 pilot cold plan，不把运行时重新选择专家集合（expert swap）作为变量。

### 1.3 关键指标与比较口径

`vllm_generate_phased.py` 每个 phase 的 `result.json` 中记录：

- `output_tokens_per_second`：该 phase 实际完成输出 token 数除以 phase 的 `elapsed_seconds`。
- `mean_tpot_ms`：各请求从第一个输出 token 到最后一个输出 token 的平均每输出 token 时间；只在输出 token 数大于 1 时计算。
- `mean_ttft_ms` / `p99_ttft_ms`：首 token 延迟。
- `maintenance[].elapsed_seconds`：phase 之间执行的同步 KV/专家维护耗时。

比较时必须区分两种口径：

1. **phase-local decode 性能**：只比较业务 phase 的 `output_tokens_per_second` 或 `mean_tpot_ms`；不把 phase 之后的 maintenance wall time 加到该 phase 中。
2. **端到端运行成本**：加载时间、所有 phase 时间和全部 `maintenance` 时间均计入。该口径反映在线系统真实代价。

因此，“不考虑逐步释放开销”只表示在第一种口径下排除 `maintenance[].elapsed_seconds`；它**不等于**消除了先前内存布局、页驻留状态和已发生迁移的后效。

---

## 2. 精确版本与运行环境

### 2.1 Git 基线与当前工作树要求

必须包含以下提交中的内核修复：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
git rev-parse HEAD
git show --stat --oneline 23c4cc93
```

基线提交为：

```text
23c4cc939515a1a1a49c045b1de1e5c20321de7e
```

该提交在 `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c` 中移除了 `uvm_va_block_evict_chunks()` 内过早的 `goto out`。这是空间复用实验的**必要条件**：当待驱逐页面被 `moe_slot_pte_bits` 过滤为空时，函数仍需继续执行后续逻辑，才能使释放的 KV 物理 chunk 可被后来的专家分配复用。

截至本文档快照，工作树还包含两个**未提交但必须保留**的分阶段维护接口修改：

```text
workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh
workloads/vllm/vllm/uvm_test/vllm_generate_phased.py
```

它们新增：

```text
环境变量：MAINTENANCE_WAVES_PER_PHASE_LIST
CLI 参数：--maintenance-waves-per-phase-list
```

该接口使每个 phase 可消耗不同数量的 maintenance wave，是精确复现 4→7→10 GB 阶梯实验的必要条件。后续精简代码前应将这两个改动提交，或至少保存二进制 patch：

```bash
git diff --binary -- \
  workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh \
  workloads/vllm/vllm/uvm_test/vllm_generate_phased.py \
  > /safe/archive/maintenance-waves-per-phase-list.patch
```

并在新工作树恢复后验证：

```bash
rg -n 'MAINTENANCE_WAVES_PER_PHASE_LIST|maintenance-waves-per-phase-list' \
  workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh \
  workloads/vllm/vllm/uvm_test/vllm_generate_phased.py
```

### 2.2 已验证的软硬件快照

| 项目 | 值 |
|---|---|
| OS / kernel | Linux `6.13.12-061312-generic` |
| GPU | NVIDIA GeForce RTX 5090，`32607 MiB` |
| NVIDIA driver | `595.71.05` |
| Python | `3.12.3` |
| PyTorch | `2.9.0+cu128` |
| CUDA runtime | `12.8` |
| vLLM | `0.1.dev11683+gd9d342d21.d20260330` |
| vLLM 虚拟环境 | `workloads/vllm/vllm/.venv` |

复现实验前保存实际机器快照，而不是假定上述版本始终可用：

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
cd "$REPO"
{
  date --iso-8601=seconds
  git rev-parse HEAD
  git status --short
  uname -a
  nvidia-smi
  "$REPO/workloads/vllm/vllm/.venv/bin/python" --version
  "$REPO/workloads/vllm/vllm/.venv/bin/python" - <<'PY'
import torch
import vllm
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"vllm={getattr(vllm, '__version__', 'unknown')}")
PY
} | tee /tmp/moe_kv_environment_$(date +%Y%m%d_%H%M%S).txt
```

不同 GPU、driver、CUDA、vLLM commit、模型 snapshot 或页面缓存状态都可能改变绝对吞吐；本指南保证的是**实验定义和比较方法**可复现，而非保证逐 token 时间完全相同。

---

## 3. 文件、模型和数据集

### 3.1 必须保留的最小实现集合

后续精简项目时，至少保留下列文件或等价实现：

| 作用 | 路径 |
|---|---|
| 空间复用内核修复 | `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c` |
| 统一实验 runner | `workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh` |
| 单进程多 phase 生成器及指标写入 | `workloads/vllm/vllm/uvm_test/vllm_generate_phased.py` |
| routing trace 到 cold plan 的确定性构建器 | `workloads/vllm/vllm/uvm_test/build_hot_cold_expert_plan.py` |
| UVM allocator 与复用 probe 构建规则 | `workloads/vllm/vllm/uvm_test/Makefile` |
| UVM-to-UVM 空间复用 probe | `workloads/vllm/vllm/uvm_test/uvm_to_uvm_reuse_probe.cu` |
| vLLM 侧 UVM 动态池 / 专家释放实现 | 与上述 runner 使用的 vLLM 修改代码 |

不应把 `/tmp/<run-root>` 中的 JSON、trace 和 CSV 当作实现代码删除前的唯一证据；它们应另行归档。

### 3.2 默认模型和数据路径

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
VLLM_ROOT="$REPO/workloads/vllm/vllm"
MODEL=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
DATASET="$REPO/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
VLLM_PY=.venv/bin/python

test -d "$MODEL"
test -f "$DATASET"
test -x "$VLLM_ROOT/$VLLM_PY"
sha256sum "$DATASET"
```

runner 会用模型 tokenizer 从 ShareGPT 中按原始顺序选择首批 prompt token 数在指定区间内的请求，并写入固定 fixture。模型 snapshot 或 tokenizer 改变时，即使数据集不变，fixture 也可能改变；必须归档 `fixture_metadata.json` 和 fixture 本身。

---

## 4. 驱动、allocator 和空间复用前置检查

### 4.1 编译自定义内核模块

仅当需要加载本仓库的 UVM 驱动实验时执行。编译不等于已加载；不要使用 `make modules_install`，避免覆盖系统安装的驱动模块。

```bash
cd /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module
make modules -j"$(nproc)"
```

### 4.2 临时重载驱动模块（高风险）

此操作会停止图形栈。只可在 GPU 空闲、没有 SSH/桌面会话依赖 GPU、且已确认可以恢复显示服务时执行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo systemctl stop gdm3 2>/dev/null || true
sleep 2
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true

sudo insmod kernel-module/nvidia-module/kernel-open/nvidia.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-modeset.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-drm.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

sudo systemctl start gdm3 2>/dev/null || true
nvidia-smi
```

若只替换 `nvidia-uvm.ko`，也必须确保没有 CUDA 进程或依赖 UVM 的进程仍在运行。完整操作背景见 `docs/uvm_gpu_only_cpu_unmap_executable_plan.md`。

### 4.3 编译 allocator 与空间复用 probe

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
make -C "$REPO/workloads/vllm/vllm/uvm_test" uvm uvm_to_uvm_reuse_probe
```

### 4.4 先验证“KV 释放可被专家复用”

此 probe 是性能实验之前的正确性门槛：热点区保持 GPU 驻留；KV 被 offload；新专家分配提升到 GPU；且专家所需空间超过初始报告的 free space，从而证明复用的不是原本空闲空间。

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test
./uvm_to_uvm_reuse_probe --device 0 --hot-gib 2 --kv-gib 2 --expert-gib 2 \
  | tee /tmp/uvm_to_uvm_reuse_probe.log
```

以下标记都必须为 `1`：

```text
reuse_without_hot_eviction=1
probe_result success=1
hot_preserved=1
kv_offloaded=1
expert_promoted=1
```

若任一标记为 `0`，停止后续性能解释，依次确认：正在加载的 `nvidia-uvm.ko` 是否为本仓库编译版本、`23c4cc93` 的修复是否存在、GPU 是否被其他进程占用，以及 probe 的显存大小是否适配当前 GPU。

---

## 5. 统一负载与固定参数

### 5.1 静态 4/7/10 GB 扫描的标准配置

| 参数 | 值 | 说明 |
|---|---:|---|
| `REQUEST_COUNT` | `96` | fixture 总请求数 |
| `PROMPT_MIN_TOKENS` | `300` | prompt 筛选下界 |
| `PROMPT_MAX_TOKENS` | `320` | prompt 筛选上界 |
| `OUTPUT_TOKENS` | `64` | 每请求输出 token 数 |
| `LARGE_KV_BYTES` | `4294967296` | 固定 4 GiB KV cache |
| `INITIAL_KV_BLOCKS` | `818` | 动态池初始目标 |
| `FINAL_KV_BLOCKS` | `2730` | 动态池最终目标 |
| `MAX_MODEL_LEN` | `512` | 上下文上限 |
| `MAX_NUM_SEQS` | `96` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | batch token 上限 |
| `GPU_MEMORY_UTILIZATION` | `0.97` | vLLM 显存利用率 |
| `TIMEOUT_SEC` | `1800` | 每个 case 超时 |

统一 phase 顺序：

```text
setup4,c4,c8,c16,c24,c32,c40,c48,c56,c64,c72,c80,c88,c96
```

统一请求数：

```text
4,4,8,16,24,32,40,48,56,64,72,80,88,96
```

统一输出长度：

```text
64,64,64,64,64,64,64,64,64,64,64,64,64,64
```

`setup4` 用于预热；不得纳入汇总吞吐、TPOT 或不同策略的统计比较。每个 phase 从 fixture 的开头取前 `N` 个请求；这是有意控制变量，使同一并发 `N` 在不同 case 中使用相同 prompt 集合。

### 5.2 专家计划与估算卸载量

| 计划 | 冷专家数 / 层 | cold plan 条目数 | 估算释放字节数 | 十进制显示 |
|---|---:|---:|---:|---:|
| `k18` | 18 | 864 | 4,076,863,488 | 4.08 GB |
| `k31` | 31 | 1488 | 7,021,264,896 | 7.02 GB |
| `k45` | 45 | 2160 | 10,192,158,720 | 10.19 GB |

本表的“GB”是十进制显示；对应的 GiB 分别约为 3.80、6.54、9.49 GiB。不要把它们误写为 KV cache 大小。

由于每个专家涉及 `moe_gate_up` 和 `moe_down` 两个目标 role，运行时 `remote_prepare_records` 预期为 cold plan 条目数的两倍：

| 计划 | 预期 `remote_prepare_records` | 预期 `remote_prepare_bytes` |
|---|---:|---:|
| `k18` | 1728 | 4,076,863,488 |
| `k31` | 2976 | 7,021,264,896 |
| `k45` | 4320 | 10,192,158,720 |

---

## 6. 统一实验准备

### 6.1 清理 GPU 并建立归档根目录

每个 case 开始与结束前 GPU 应接近空闲。runner 会等待显存使用量不高于 1024 MiB，但应在运行前人工检查，排除其他 CUDA 进程和前一次崩溃残留。

```bash
nvidia-smi
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits

REPO=/home/ubuntu/nvidia-uvm-gpu
STAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE=/safe/archive/moe_kv_$STAMP
mkdir -p "$ARCHIVE"

{
  git -C "$REPO" rev-parse HEAD
  git -C "$REPO" status --short
  git -C "$REPO" diff --binary
  nvidia-smi
  sha256sum "$REPO/workloads/vllm/datasets/ShareGPT_V3_unfiltered_cleaned_split.json"
} > "$ARCHIVE/provenance.txt"
```

`/safe/archive` 只是示例路径，应替换为不会随 `/tmp` 清理的持久化存储位置。

### 6.2 先运行一次 pilot，并从同一 trace 构建嵌套计划

所有 4/7/10 GB 对比必须使用**同一** pilot routing trace。不能分别运行 pilot 后再分别构建计划，否则“卸载量”之外还改变了专家集合。

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
RUNNER="$REPO/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
PILOT_ROOT=/tmp/moe_kv_pilot_$(date +%Y%m%d_%H%M%S)

RUN_ROOT="$PILOT_ROOT" \
REQUEST_COUNT=96 PILOT_REQUEST_COUNT=8 PILOT_OUTPUT_TOKENS=1 \
PROMPT_MIN_TOKENS=300 PROMPT_MAX_TOKENS=320 OUTPUT_TOKENS=64 \
SMALL_KV_BYTES=4294967296 LARGE_KV_BYTES=4294967296 \
INITIAL_KV_BLOCKS=818 FINAL_KV_BLOCKS=2730 \
COLD_EXPERTS_PER_LAYER=45 EXPERT_EST_BYTES=4718592 \
MAX_MODEL_LEN=512 MAX_NUM_SEQS=96 MAX_NUM_BATCHED_TOKENS=2048 \
GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=1800 \
RUN_CASES=pilot \
bash "$RUNNER"

test -s "$PILOT_ROOT/pilot/routing_trace.jsonl"
test -s "$PILOT_ROOT/sharegpt_256_prompt320_360.json"
```

从该 trace 构建三个计划：

```bash
VLLM_ROOT="$REPO/workloads/vllm/vllm"
PLAN_DIR="$ARCHIVE/plans"
mkdir -p "$PLAN_DIR"

for experts_per_layer in 18 31 45; do
  "$VLLM_ROOT/.venv/bin/python" \
    "$VLLM_ROOT/uvm_test/build_hot_cold_expert_plan.py" \
    --routing-trace "$PILOT_ROOT/pilot/routing_trace.jsonl" \
    --output "$PLAN_DIR/plan-k${experts_per_layer}.json" \
    --num-layers 48 \
    --num-experts 128 \
    --cold-experts-per-layer "$experts_per_layer" \
    --expert-est-bytes 4718592
done

sha256sum "$PLAN_DIR"/*.json | tee "$ARCHIVE/plan-sha256.txt"
```

计划构建器按 `(pilot_token_count, expert_id)` 升序选择每层前 `K` 个专家。因此在同一输入 trace 下，理论上每层满足：

```text
cold(k18) ⊆ cold(k31) ⊆ cold(k45)
```

用以下检查确认嵌套性与估算字节数：

```bash
"$VLLM_ROOT/.venv/bin/python" - \
  "$PLAN_DIR/plan-k18.json" "$PLAN_DIR/plan-k31.json" "$PLAN_DIR/plan-k45.json" <<'PY'
import json
import sys

plans = [json.load(open(path, encoding="utf-8")) for path in sys.argv[1:]]
selected = []
for plan in plans:
    by_layer = plan["metadata"]["selected_by_layer"]
    selected.append({layer: set(ids) for layer, ids in by_layer.items()})
    print(plan["metadata"]["cold_experts_per_layer"],
          plan["metadata"]["cold_records"],
          plan["metadata"]["estimated_cold_bytes"])
for layer in selected[0]:
    assert selected[0][layer] <= selected[1][layer] <= selected[2][layer], layer
print("nested_plans=ok")
PY
```

### 6.3 runner 的 plan 文件命名约束

runner 当前固定从以下路径读取计划：

```text
$RUN_ROOT/hot_cold_plan_k45.json
```

名称中的 `k45` 是历史遗留命名，不会自动根据文件内容判断实际 K。即使跑 `k18` 或 `k31`，也必须把对应计划复制为该文件名：

```bash
cp "$PLAN_DIR/plan-k18.json" "$RUN_ROOT/hot_cold_plan_k45.json"
```

当 `RUN_CASES=static_custom` 或 `RUN_CASES=adaptive` 而不跑 `pilot` 时，先复制该 plan；否则 runner 会尝试从 `$RUN_ROOT/pilot/routing_trace.jsonl` 构建计划并失败。

---

## 7. 实验 A：固定 4 GiB KV 下静态卸载 4/7/10 GB

### 7.1 实验问题与控制变量

问题：在最大 96 并发、64 output tokens、固定 4 GiB KV cache 的可接受容量范围内，是否“少卸载专家、更多专家留在 GPU”能改善性能？

控制变量：模型、driver、vLLM commit、fixture、phase、KV cache、并发、输出长度、cold-plan 生成 trace 均固定；唯一主动变量是每层 cold expert 数 `18 / 31 / 45`。

### 7.2 可直接执行的静态扫描脚本

在 shell 中定义函数并依次运行三个 case。每个 case 独立进程和独立 run root；开始前复制从同一 pilot 得到的 plan。

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
RUNNER="$REPO/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
PHASE_NAMES='setup4,c4,c8,c16,c24,c32,c40,c48,c56,c64,c72,c80,c88,c96'
PHASE_COUNTS='4,4,8,16,24,32,40,48,56,64,72,80,88,96'
PHASE_OUTPUTS='64,64,64,64,64,64,64,64,64,64,64,64,64,64'

run_static_case() {
  local label="$1"
  local experts_per_layer="$2"
  local plan="$3"
  local run_root="/tmp/moe_kv_static_${label}_$(date +%Y%m%d_%H%M%S)"

  mkdir -p "$run_root"
  cp "$plan" "$run_root/hot_cold_plan_k45.json"

  RUN_ROOT="$run_root" \
  REQUEST_COUNT=96 PROMPT_MIN_TOKENS=300 PROMPT_MAX_TOKENS=320 OUTPUT_TOKENS=64 \
  LARGE_KV_BYTES=4294967296 INITIAL_KV_BLOCKS=818 FINAL_KV_BLOCKS=2730 \
  COLD_EXPERTS_PER_LAYER="$experts_per_layer" EXPERT_EST_BYTES=4718592 \
  MAX_MODEL_LEN=512 MAX_NUM_SEQS=96 MAX_NUM_BATCHED_TOKENS=2048 \
  GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=1800 \
  RUN_CASES=static_custom \
  STATIC_PHASE_NAMES="$PHASE_NAMES" \
  STATIC_PHASE_COUNTS="$PHASE_COUNTS" \
  STATIC_PHASE_OUTPUTS="$PHASE_OUTPUTS" \
  bash "$RUNNER"

  echo "run_root=$run_root"
}

run_static_case k18_4g 18 "$PLAN_DIR/plan-k18.json"
run_static_case k31_7g 31 "$PLAN_DIR/plan-k31.json"
run_static_case k45_10g 45 "$PLAN_DIR/plan-k45.json"
```

静态机制在业务 phase 前调用 `--pre-phase-target-blocks 2730`，因此完整 cold plan 的释放和 KV 物理池调整发生在计时 phase 之前。其 `load_seconds` 会受 OS page cache、模型首次加载和首次释放操作影响，**不可**跨 case 作为 decode 性能指标比较。

### 7.3 每个 case 的成功条件

```bash
RUN_ROOT=/tmp/moe_kv_static_k18_4g_YYYYMMDD_HHMMSS
cat "$RUN_ROOT/static_custom/returncode.txt"
"$REPO/workloads/vllm/vllm/.venv/bin/python" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
case = summary["cases"]["static_custom"]
print("summary_status=", case.get("status"))
print("fatal_markers=", case.get("fatal_markers"))
result = json.loads((root / "static_custom" / "result.json").read_text())
print("result_status=", result.get("status"))
print("phases=", [phase["phase"] for phase in result.get("phases", [])])
print("maintenance_waves=", len(result.get("maintenance", [])))
PY
```

必须同时满足：

- `returncode.txt` 为 `0`；
- `summary.json` 对应 case 的 `status == "ok"`；
- `fatal_markers == []`；
- `result.json` 的 `status == "ok"`；
- 包含 `setup4` 至 `c96` 的全部 14 个 phase；
- 对应计划的 `remote_prepare_bytes` 和 records 符合第 5.2 节；
- 可比较 case 的每个 phase `completed_requests`、实际 `output_tokens` 一致；
- 进程结束后 GPU 回到近空闲状态。

### 7.4 汇总吞吐的确定义与提取命令

汇总吞吐只使用 `c4` 到 `c96`：

```text
aggregate_tok_s = sum(各业务 phase 的 output_tokens) / sum(各业务 phase 的 elapsed_seconds)
```

下面命令打印每个 phase、汇总吞吐、c96 TPOT 与 maintenance 总时长：

```bash
"$REPO/workloads/vllm/vllm/.venv/bin/python" - \
  /tmp/moe_kv_static_k18_4g_YYYYMMDD_HHMMSS \
  /tmp/moe_kv_static_k31_7g_YYYYMMDD_HHMMSS \
  /tmp/moe_kv_static_k45_10g_YYYYMMDD_HHMMSS <<'PY'
import json
import sys
from pathlib import Path

for raw_root in sys.argv[1:]:
    root = Path(raw_root)
    result = json.loads((root / "static_custom" / "result.json").read_text())
    phases = [item for item in result["phases"] if item["phase"] != "setup4"]
    tokens = sum(item["output_tokens"] for item in phases)
    elapsed = sum(item["elapsed_seconds"] for item in phases)
    c96 = next(item for item in phases if item["phase"] == "c96")
    maintenance = sum(item["elapsed_seconds"] for item in result["maintenance"])
    print(f"{root.name}: aggregate_tok_s={tokens / elapsed:.3f}, "
          f"c96_tpot_ms={c96['mean_tpot_ms']:.2f}, "
          f"maintenance_s={maintenance:.3f}")
    for phase in phases:
        print(f"  {phase['phase']:>3} {phase['output_tokens_per_second']:.3f} tok/s")
PY
```

### 7.5 已获得的静态扫描结果（单次历史运行）

以下为已记录的同配置结果；它们是复现目标的量级和趋势参考，不应用作不同机器上的严格 golden value。

| 并发 | 4.08 GB (`k18`) | 7.02 GB (`k31`) | 10.19 GB (`k45`) |
|---:|---:|---:|---:|
| 4 | 59.314 | 64.536 | 66.056 |
| 8 | 108.295 | 127.124 | 127.222 |
| 16 | 247.020 | 236.152 | 228.587 |
| 24 | 364.572 | 344.836 | 322.219 |
| 32 | 458.022 | 421.638 | 341.966 |
| 40 | 539.713 | 503.102 | 367.959 |
| 48 | 631.089 | 562.887 | 389.300 |
| 56 | 712.176 | 627.106 | 426.903 |
| 64 | 775.409 | 652.257 | 439.514 |
| 72 | 835.677 | 696.742 | 464.419 |
| 80 | 909.673 | 729.003 | 482.737 |
| 88 | 969.571 | 737.349 | 493.338 |
| 96 | 1026.238 | 770.525 | 511.589 |

单位均为 output tokens/s。排除 `setup4` 后：

| 冷专家计划 | 汇总吞吐 | c96 Mean TPOT |
|---|---:|---:|
| `k18`（4.08 GB） | 620.842 tok/s | 69.96 ms |
| `k31`（7.02 GB） | 547.337 tok/s | 92.38 ms |
| `k45`（10.19 GB） | 403.321 tok/s | 142.25 ms |

**结论：** 在并发不低于 16 时，少卸载专家的性能呈单调优势；保留更多专家在 GPU 上值得。`c4/c8` 的 10 GB 略快不应反推高并发规律，它更可能来自低负载下的预热、页驻留、CPU page cache 与测量噪声。要量化不确定性，应以不同 case 交错顺序运行至少三轮，并报告中位数与分位范围。

---

## 8. 实验 B：一次性 10 GB 与阶梯式 4→7→10 GB

### 8.1 问题和正确解释

此实验不比较“同一最终卸载量下渐进释放操作本身一定更快”。它检验的是：当低并发只需较少释放量、高并发再逐步扩大 KV/专家释放时，是否能在**中间并发区间**避免一开始就承受 10 GB 卸载对应的远端专家访问代价。

最终到 `c72` 后，两方案均达到 `k45`（10.19 GB）。因此 `c72`–`c96` 应接近；真正的证据窗口是 `c4`–`c64`，尤其是阶梯方案仍处于 `k31` 而一次性方案已处于 `k45` 的 `c40`–`c64`。

### 8.2 maintenance wave 的精确映射

`COLD_EXPERTS_PER_LAYER=45` 时，runner 生成 45 个 KV grow target。每个带 `--release-expert-wave` 的 wave 最多每层释放一个专家：

| phase 后 | wave 数 | 累计 wave | 对应每层冷专家数 | 估算卸载量 |
|---|---:|---:|---:|---:|
| `setup4` | 18 | 18 | 18 | 4.08 GB |
| `c32_4g` | 13 | 31 | 31 | 7.02 GB |
| `c64_7g` | 14 | 45 | 45 | 10.19 GB |

对应的完整 wave list 必须与 14 个 phase 对齐：

```text
18,0,0,0,0,13,0,0,0,14,0,0,0,0
```

### 8.3 运行直接一次性 10 GB 对照组

该对照用第 7 节的静态函数，使用 `k45` 计划，且 phase 序列与阶梯方案保持一致：

```bash
PHASE_NAMES='setup4,c4_4g,c8_4g,c16_4g,c24_4g,c32_4g,c40_7g,c48_7g,c56_7g,c64_7g,c72_10g,c80_10g,c88_10g,c96_10g'
PHASE_COUNTS='4,4,8,16,24,32,40,48,56,64,72,80,88,96'
PHASE_OUTPUTS='64,64,64,64,64,64,64,64,64,64,64,64,64,64'

run_static_case direct_k45_10g 45 "$PLAN_DIR/plan-k45.json"
```

`run_static_case` 定义见第 7.2 节。名称中的 `_4g/_7g/_10g` 只标注供比较的阶梯目标；直接对照在全部业务 phase 中都已经是 `k45`。

### 8.4 运行单进程阶梯方案

`adaptive` 必须在一个 vLLM 进程内跑完所有 phase，才能保留渐进增长后的运行状态。提前复制 `k45` plan，并显式使用可变 wave list：

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
RUNNER="$REPO/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
STAGED_ROOT=/tmp/moe_kv_staged_4_7_10_$(date +%Y%m%d_%H%M%S)

mkdir -p "$STAGED_ROOT"
cp "$PLAN_DIR/plan-k45.json" "$STAGED_ROOT/hot_cold_plan_k45.json"

RUN_ROOT="$STAGED_ROOT" \
REQUEST_COUNT=96 PROMPT_MIN_TOKENS=300 PROMPT_MAX_TOKENS=320 OUTPUT_TOKENS=64 \
LARGE_KV_BYTES=4294967296 INITIAL_KV_BLOCKS=818 FINAL_KV_BLOCKS=2730 \
COLD_EXPERTS_PER_LAYER=45 EXPERT_EST_BYTES=4718592 \
MAX_MODEL_LEN=512 MAX_NUM_SEQS=96 MAX_NUM_BATCHED_TOKENS=2048 \
GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=1800 \
RUN_CASES=adaptive \
PHASE_NAMES='setup4,c4_4g,c8_4g,c16_4g,c24_4g,c32_4g,c40_7g,c48_7g,c56_7g,c64_7g,c72_10g,c80_10g,c88_10g,c96_10g' \
PHASE_COUNTS='4,4,8,16,24,32,40,48,56,64,72,80,88,96' \
PHASE_OUTPUTS='64,64,64,64,64,64,64,64,64,64,64,64,64,64' \
MAINTENANCE_WAVES_PER_PHASE_LIST='18,0,0,0,0,13,0,0,0,14,0,0,0,0' \
bash "$RUNNER"
```

运行后确认共有 45 条 maintenance 记录，且 wave 累计位置符合第 8.2 节：

```bash
"$REPO/workloads/vllm/vllm/.venv/bin/python" - "$STAGED_ROOT/adaptive/result.json" <<'PY'
import json
import sys
from collections import Counter

result = json.load(open(sys.argv[1], encoding="utf-8"))
maintenance = result["maintenance"]
print("maintenance_waves=", len(maintenance))
print("waves_by_after_phase=", Counter(item["after_phase"] for item in maintenance))
print("maintenance_seconds=", sum(item["elapsed_seconds"] for item in maintenance))
assert len(maintenance) == 45
assert Counter(item["after_phase"] for item in maintenance) == {0: 18, 5: 13, 9: 14}
PY
```

### 8.5 比较命令

```bash
"$REPO/workloads/vllm/vllm/.venv/bin/python" - \
  /tmp/moe_kv_static_direct_k45_10g_YYYYMMDD_HHMMSS/static_custom/result.json \
  "$STAGED_ROOT/adaptive/result.json" <<'PY'
import json
import sys

labels = ["direct_10g", "staged_4_7_10"]
all_phases = []
for path in sys.argv[1:]:
    result = json.load(open(path, encoding="utf-8"))
    all_phases.append({item["phase"]: item for item in result["phases"]})

for phase in all_phases[0]:
    if phase == "setup4":
        continue
    direct = all_phases[0][phase]["output_tokens_per_second"]
    staged = all_phases[1][phase]["output_tokens_per_second"]
    change = (staged / direct - 1.0) * 100.0
    print(f"{phase:>7} direct={direct:.3f} staged={staged:.3f} delta={change:+.2f}%")
PY
```

phase 名称必须完全匹配。若使用了不同的 phase 名称，按并发和阶段对应关系比较，不要只按数组位置猜测。

### 8.6 已获得的阶梯实验结果（单次历史运行）

| 并发 | 阶梯时的专家卸载量 | 直接 10 GB tok/s | 阶梯 tok/s | 阶梯相对变化 |
|---:|---:|---:|---:|---:|
| 4 | 4 GB | 65.18 | 54.85 | -15.8% |
| 8 | 4 GB | 124.65 | 103.00 | -17.4% |
| 16 | 4 GB | 223.47 | 186.27 | -16.7% |
| 24 | 4 GB | 324.48 | 315.25 | -2.9% |
| 32 | 4 GB | 342.45 | 320.30 | -6.5% |
| 40 | 7 GB | 368.38 | 351.22 | -4.7% |
| 48 | 7 GB | 389.44 | 499.86 | +28.4% |
| 56 | 7 GB | 427.06 | 460.11 | +7.7% |
| 64 | 7 GB | 440.21 | 514.33 | +16.8% |
| 72 | 10 GB | 464.56 | 464.24 | -0.1% |
| 80 | 10 GB | 483.04 | 477.13 | -1.2% |
| 88 | 10 GB | 494.21 | 488.70 | -1.1% |
| 96 | 10 GB | 511.96 | 510.84 | -0.2% |

解释：

- 在 `c48`、`c56`、`c64`，阶梯方案只释放约 7 GB，而直接方案已释放约 10 GB，阶梯方案出现明显 phase-local decode 优势。
- 到 `c72` 后两者同为 10 GB，吞吐回到约 ±1.2% 范围，符合“最终专家集合相同，steady-state 性能应接近”的预期。
- `c4`–`c40` 阶梯较慢说明渐进过程、先前 phase 的布局/热度、以及低并发噪声仍会影响结果；不能声称“逐步释放在所有并发下都更快”。
- 阶梯方案曾记录 45 次同步 maintenance，累计约 `48.565 s`。这部分不在每个业务 phase 的 `elapsed_seconds` 中，但端到端比较必须把它加回；因此本实验只证明“避免过早达到 10 GB 可能改善相应阶段性能”，不证明动态维护本身降低总运行成本。

---

## 9. 实验 C：三种机制的历史对比

### 9.1 复现命令模板

以下模板使用 runner 的默认 phase 设计：adaptive 经过多个 phase 分批增长，static 与 native 为 warm/final 两 phase。为完整公平比较，应从新的独立 run root 开始并让同一个 pilot trace 构建 `k45` plan。

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
RUNNER="$REPO/workloads/vllm/vllm/uvm_test/run_vllm_online_hot_cold_kv10g_scale.sh"
RUN_ROOT=/tmp/moe_kv_three_mechanisms_$(date +%Y%m%d_%H%M%S)

RUN_ROOT="$RUN_ROOT" \
REQUEST_COUNT=256 PILOT_REQUEST_COUNT=256 PILOT_OUTPUT_TOKENS=64 \
PROMPT_MIN_TOKENS=320 PROMPT_MAX_TOKENS=360 OUTPUT_TOKENS=64 \
LARGE_KV_BYTES=10737418240 INITIAL_KV_BLOCKS=818 FINAL_KV_BLOCKS=6825 \
COLD_EXPERTS_PER_LAYER=45 EXPERT_EST_BYTES=4718592 NATIVE_CPU_OFFLOAD_GB=10.0 \
MAX_MODEL_LEN=512 MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=2048 \
GPU_MEMORY_UTILIZATION=0.97 TIMEOUT_SEC=3600 \
RUN_CASES=pilot,adaptive,static_custom,native_uva \
bash "$RUNNER"
```

若只运行某一个 case，按第 6.3 节先准备 `$RUN_ROOT/hot_cold_plan_k45.json`。`native_uva` 不使用 UVM 的 expert/KV manager；不要将其 `load_seconds` 与 UVM case 直接混合比较。

### 9.2 已记录的历史结论

在“每层最终释放 45 个专家、共 2160 个专家、静态与动态最终集合重合率 100%”的原始三机制实验中：

1. **decode 阶段（Mean TPOT）**：动态卸载优于静态卸载，静态卸载优于原生 UVA。动态的 Mean TPOT 相比静态低约 `39.5%`，相比原生 UVA 低约 `46.8%`。
2. **端到端吞吐、总耗时和 TTFT**：静态卸载优于动态卸载，动态卸载略优于原生 UVA。静态吞吐为 `233.115 tok/s`，比动态高约 `28.3%`；动态只比原生 UVA 高约 `2.15%`。
3. **原因**：动态方案同步执行了 45 次 maintenance wave，累计约 `46.54 s`；静态方案只需一次维护，约 `1.73 s`。同步维护阻塞请求调度和 prefill，使端到端收益被抵消。

这组实验的正确结论是：**按 KV 压力逐步释放可改善某些 decode phase 的稳态性能，但若把维护操作放在在线请求关键路径上，端到端性能未必优于一次性静态释放。**

---

## 10. 相关历史消融及其正确结论

### 10.1 相同最终释放量：每层 22 个专家

曾比较“warm-up 后直接释放 22 个/层”和“逐步释放、最终同为 22 个/层”。历史结果显示直接静态方案更快：

- 最终 decode 的历史均值约为 direct `660 tok/s`、gradual `555 tok/s`；
- 加入 settle phase 后约为 direct `979.4 tok/s`、gradual `850.7 tok/s`。

因此，**在最终卸载量相同的条件下，不能从现有数据得出‘逐步释放本身比一次性释放更快’的结论**。渐进方案的价值主要是避免在低 KV 压力时过早释放过多专家。

### 10.2 不同最终释放量：22/层与全部 128/层

历史对比中，逐步 22/层约 `569.7 tok/s`，静态全部 128/层约 `111.5 tok/s`。这只能说明“最终卸载全部专家过于激进”；两组最终专家集合和最终远端访问量不同，因此不能用来证明释放时机的优劣。

---

## 11. 结果归档、完整性检查与重复实验

### 11.1 每个 run root 必须保留的产物

```text
sharegpt_256_prompt320_360.json
fixture_metadata.json
hot_cold_plan_k45.json
kv_grow_targets.txt
summary.json
<case>/result.json
<case>/progress.jsonl
<case>/run.log
<case>/returncode.txt
<case>/expert_trace.jsonl
<case>/kv_runtime_trace.jsonl
<case>/kv_physical_trace.jsonl
<case>/residency_trace.jsonl
<case>/dmon_pcie.csv
```

`native_uva` 还应保存：

```text
native_uva/cpu_offload_manifest.jsonl
```

立即归档，而不是只保存在 `/tmp`：

```bash
RUN_ROOT=/tmp/moe_kv_static_k18_4g_YYYYMMDD_HHMMSS
ARCHIVE=/safe/archive/moe_kv_YYYYMMDD_HHMMSS
mkdir -p "$ARCHIVE"
rsync -a --info=progress2 "$RUN_ROOT/" "$ARCHIVE/$(basename "$RUN_ROOT")/"
```

### 11.2 建议的摘要提取与失败检查

```bash
REPO=/home/ubuntu/nvidia-uvm-gpu
"$REPO/workloads/vllm/vllm/.venv/bin/python" - /tmp/moe_kv_static_k18_4g_YYYYMMDD_HHMMSS <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
for name, case in summary["cases"].items():
    print(name, "status=", case.get("status"),
          "fatal_markers=", case.get("fatal_markers"))
    result_path = root / name / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        print("  result_status=", result.get("status"),
              "maintenance_waves=", len(result.get("maintenance", [])))
PY

nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
```

### 11.3 重复实验规则

为将趋势升级为可统计结论：

1. 每个策略至少执行 3 次；每次使用新 `RUN_ROOT`，但复用同一 fixture 和同一 plan 文件。
2. 轮转执行顺序，例如 `k18 → k31 → k45`、`k45 → k18 → k31`、`k31 → k45 → k18`，减少 OS page cache 和热状态造成的系统性偏差。
3. 报告每个并发的中位数、最小/最大值或 p25/p75；汇总吞吐按第 7.4 节的 token 加权定义计算，不平均各 phase 的 tok/s。
4. `setup4` 始终仅预热，不参与统计。
5. 维护端到端性能时，把 `load_seconds`、所有 phase `elapsed_seconds` 和 maintenance 总时间分开报告；不要用一个未定义的“总时间”混淆它们。

---

## 12. 常见错误与排查清单

| 现象 | 首先检查 | 处理 |
|---|---|---|
| 复用 probe 不成功 | 内核模块版本、修复是否存在、GPU 是否空闲 | 重新编译/加载本仓库 UVM，确认 `23c4cc93` 的修改 |
| `routing_trace.jsonl` 不存在 | pilot case 是否成功、trace 环境变量 | 重跑 `RUN_CASES=pilot`，确认 `returncode=0` |
| runner 在 static/adaptive 前构建 plan 失败 | `$RUN_ROOT/hot_cold_plan_k45.json` 不存在 | 先复制对应 `plan-k18/k31/k45` 为固定文件名 |
| phase 参数长度不匹配 | names/counts/outputs/list 的逗号数 | 确保四个序列等长；wave list 必须有 14 项 |
| `unused KV grow targets` | wave list 累计数不足 45 | 4→7→10 实验必须为 `18+13+14=45` |
| 输出 token 数不一致 | `ignore_eos` 或参数被改动 | 保持 `temperature=0.0`、`ignore_eos=True`、每请求 64 tokens |
| `load_seconds` 差异巨大 | OS page cache / 首次初始化 | 不以此衡量 decode 优劣；按计划轮转和重复 |
| 动态 phase 更快但总吞吐更慢 | maintenance 被排除在 phase 计时外 | 同时报 phase-local 与含 maintenance 的端到端结果 |
| 执行后 GPU 未释放 | vLLM 子进程/driver 状态残留 | 用 `nvidia-smi` 查 PID，清理后再启动下一 case |

---

## 13. 最终可复现结论边界

在当前 RTX 5090、4 GiB KV cache、最高 96 并发和每请求 64 输出 token 的设置下，现有数据支持以下结论：

1. 内核修复后，KV cache 物理空间释放可以被专家分配复用；这一点先由 UVM-to-UVM probe 验证，再解释 vLLM 实验。
2. 当 KV cache 容量足以承载目标并发时，在中高并发下少卸载专家（4 GB）明显优于多卸载专家（7 GB、10 GB）；更多热/常用专家留在 GPU 是有价值的。
3. 阶梯 4→7→10 GB 在它仍只需 7 GB 的中间并发区间可优于一开始即 10 GB；当两者都达到 10 GB 后性能接近。
4. 不应把上述结论扩大为“逐步卸载操作本身总是更快”：最终释放量相同的 22/层对照中，静态直接卸载更快；动态 maintenance 的同步成本还会明显伤害端到端性能。

后续代码精简必须优先保留第 3.1 节文件、固定 fixture/plan 的构建流程、内核修复及第 11.1 节运行产物；否则将无法区分“策略差异”与“专家集合、KV 容量、运行状态或驱动行为变化”造成的性能差异。
