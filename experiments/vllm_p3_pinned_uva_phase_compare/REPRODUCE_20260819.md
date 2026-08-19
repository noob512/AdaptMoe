# 同步 P3、vLLM pinned 与 managed-UVA 阶段性能实验复现指南

本文档复现以下正式实验：

| 阶段 | 负载 | P3 | pinned | 自定义 managed-UVA |
|---|---:|---:|---:|---:|
| Prefill | 314 tokens | 109.057 ms | 63.741 ms | 59.941 ms |
| Prefill | 1217 tokens | 108.844 ms | 78.769 ms | 121.612 ms |
| Decode step | batch 1 | 110.427 ms | 53.662 ms | 51.328 ms |
| Decode step | batch 4 | 110.565 ms | 55.370 ms | 51.635 ms |

参考 artifact：

```text
experiments/vllm_p3_pinned_uva_phase_compare/artifacts/
  20260819T_sync_p3_phase_formal_locked_v1/
```

正式运行入口只有
`run_with_module_cycle.sh`。它负责临时加载 workspace 中的自定义
`nvidia-uvm.ko`、运行九个隔离进程、执行 verifier，并在所有正常或异常退出路径上尝试恢复 stock UVM。不要直接把
`run_phase_compare.sh run` 当作正式入口。

## 1. 实验精确定义

### 1.1 三种机制

| 符号 | 机制 | 热路径语义 |
|---|---|---|
| `P` | 同步 P3 | fenced cmd96、registered 512 KiB plan、64 KiB PTE、双 bank、shared CE push、copy/metadata cache、CUDA compute/completion fence、O(1) bank commit、epoch validation、compact trace |
| `C` | vLLM pinned | exact-name pinned-host `copy_`，`non_blocking=True`，不做逐层 host synchronize |
| `U` | 自定义 managed-UVA | 对同一权重范围采用项目的 `remote_read` placement |

host-async cmd97/cmd98、ticket/kthread、P3-to-UVA 混合切换均不属于本实验。静态门禁会拒绝残留的 cmd97/cmd98 集成。

### 1.2 权重与搬运范围

- 模型：`Qwen3-30B-A3B-FP8`。
- tensor parallel size：1。
- 选择 layer 0–3 的全部 128 个 packed experts。
- 每层只选择 `w13_weight` 和 `w2_weight`。
- 每层 `603,979,776` bytes，即 576 MiB。
- 每层 1,152 个 512 KiB region。
- 四层共 `2,415,919,104` bytes，即 2.25 GiB。
- P3 和 pinned 每次 forward 都搬运完整的 2.25 GiB。
- managed-UVA 映射相同地址范围，但 PCIe 实际读取量由路由按需访问决定。

因此，这是相同模型、参数和地址覆盖下的端到端机制比较，不是三种机制实际 PCIe 字节数完全相等的纯带宽比较。

### 1.3 阶段测量

默认固定 fixture 产生：

- batch 1：314 个 prompt tokens；
- batch 4：1,217 个 prompt tokens。

每个独立进程对每个 batch 执行：

1. 1 个 conditioning cycle，不进入正式阶段统计；
2. 3 个 measurement cycles；
3. 每个 cycle 含 1 个完整 prefill step；
4. 随后执行 5 个不计入 decode 主指标的 warmup decode steps；
5. 再执行 16 个 measured decode steps；
6. 每个 decode step 对每个请求恰好调度 1 个 token。

每个 engine step 前后均调用 `torch.cuda.synchronize()`。报告的是数据和计算均已完成的 GPU wall time，而不是异步 host-submit 时间。

### 1.4 进程与统计单位

正式顺序为：

```text
PCU / CUP / UPC
```

这表示每种机制运行三个独立进程。第 1、3 个三进程 block 使用 batch 顺序 `1,4`，第 2 个 block 使用 `4,1`，用于平衡进程顺序和 batch 顺序影响。

统计分两级：

1. 每个进程先对三个 measurement repeats 取中位数；
2. 再对同一机制的三个独立进程中位数取中位数。

所以独立样本是 process median，不是单个请求、单次 transition 或单个 decode step。

## 2. 参考环境与可复现性边界

正式结果来自以下环境：

| 项目 | 参考值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Driver | 595.71.05 |
| Kernel | 6.13.12-061312-generic |
| CPU | Intel Xeon Gold 6530，14 个在线 CPU，`0-13` |
| Python | 3.12.3 |
| PyTorch | 2.9.0+cu128 |
| CUDA toolkit | 12.8，nvcc 12.8.93 |
| vLLM | 0.1.dev11683+gd9d342d21.d20260330 |
| 自定义 UVM srcversion | `A2526DA8952DDBBD48BC8AC` |
| stock UVM srcversion | `85A79790636BBD99BA3E43B` |
| GPU clock 目标 | SM 2400 MHz、memory 14001 MHz |

正式 artifact 保存了以下关键 SHA-256：

```text
custom nvidia-uvm.ko
26939860c040c1f4f2269817164b65a3d6c438e85dc808def819d716662262ec

model config.json
d33c579e6a0f520344712e22dfb9d6fa49ed23e43ce3f32f12daea43597eddae

evaluation fixture
0f28e34b410774573c3f8bdcddaffddd0ac29d50daad09d5d28e0e09c14930a4

generated exact_l0_l3_plan.json
a0e5695409eb1da7b99993a0590aa2e33f62d8637e1bf255e7a1406d4b67fddf

generated exact_parameter_names.txt
473d3f28f1f70788fa6dc08e333108950d8d89a423973daa7dfb74751ab46eb9
```

需要明确以下限制：

- 正式 artifact 的 `git_status.txt` 显示实验基于包含大量未提交和未跟踪内容的 workspace，不是一个仅凭 clean commit 即可重建的发布快照。
- 正式 artifact 没有记录完整模型权重哈希。
- 正式 artifact 没有记录 CUDA fence `.so` 的哈希；当前 workspace 文件的 SHA-256 是 `b4f4824e088abfe9ef99390d6ab8a8de06fdfcfa062e36b1b04b61db6d0b4471`，但这只能作为当前状态补充，不能倒推为原 artifact 已验证的 provenance。
- artifact 记录了 P0 和 14001 MHz memory clock，但没有保存外部锁频命令。锁频是人工运行环境条件，不是 verifier 可重建的事实。

因此，在当前 workspace/同一机器上可以复现功能契约和近似性能；要在另一台机器或 clean checkout 上做严格二进制复现，需要先额外归档完整 source snapshot、Python environment、模型权重、fence `.so` 和锁频命令。

## 3. 安全要求

该实验会卸载并临时替换 `nvidia_uvm`。执行前必须满足：

- 独占整台机器上的 GPU 计算环境；wrapper 对目标 GPU 检查 0 MiB，同时要求全机没有任何 compute PID。
- `/dev/nvidia-uvm` 没有 client。
- 不运行其他 UVM/module-cycle 实验。
- 具有无交互或可交互的 `sudo` 权限。
- `node-exporter.service` 和 `gpu-exporter.service` 存在；参考运行开始时二者均为 active。wrapper 会保存其初始 active 状态并临时停止服务。
- 不使用 `rmmod -f`，也不在 GPU 非空或 UVM 仍有 client 时强行卸载模块。

wrapper 会 fail closed：若退出后 GPU 仍不干净或 `/dev/nvidia-uvm` 仍被占用，它会拒绝危险的模块卸载。SIGKILL、主机掉电或内核挂死无法由 shell trap 自动恢复，因此仍需准备本文第 11 节的人工恢复流程。

## 4. 准备路径和依赖

以下命令假设仓库仍位于正式实验所用路径。若改变路径，应显式设置对应变量，并保留新路径到 artifact 中。

```bash
PHASE_REPO_ROOT=/home/ubuntu/nvidia-uvm-gpu
PHASE_VLLM_ROOT=/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm
PHASE_VLLM_PY=/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/.venv/bin/python
PHASE_CUSTOM_UVM=/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
PHASE_FENCE_LIB=/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/libuvm_slot_cuda_fence_kernels.so
PHASE_MODEL=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782
PHASE_FIXTURE=/home/ubuntu/nvidia-uvm-gpu/experiments/broad_offload_compare/artifacts/common_fixtures_20260813/evaluation_fixture_token_ids.json
```

运行脚本不会创建 vLLM virtual environment，也不会下载模型。运行期间设置了 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，所以模型必须已完整存在。

检查基础依赖：

```bash
test -x "${PHASE_VLLM_PY}"
test -f "${PHASE_MODEL}/config.json"
test -f "${PHASE_FIXTURE}"
command -v nvidia-smi
command -v nvcc
command -v make
command -v modinfo
command -v modprobe
command -v insmod
command -v systemctl
command -v taskset
command -v timeout
command -v flock
command -v fuser
command -v jq
command -v column
```

确认 Python/CUDA 版本：

```bash
"${PHASE_VLLM_PY}" - <<'PY'
import sys
import torch
import vllm

print("python", sys.version.replace("\n", " "))
print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("vLLM", getattr(vllm, "__version__", "unknown"))
PY

nvidia-smi --query-gpu=driver_version,name --format=csv,noheader
nvcc --version
```

## 5. 构建实验产物

### 5.1 构建自定义 UVM

自定义模块必须与当前 kernel ABI 以及 595.71.05 userspace/GSP 组件匹配。

```bash
cd "${PHASE_REPO_ROOT}/kernel-module/nvidia-module"
make modules -j"$(nproc)"

test -f "${PHASE_CUSTOM_UVM}"
test "$(modinfo -F vermagic "${PHASE_CUSTOM_UVM}" | awk '{print $1}')" = "$(uname -r)"
modinfo -F version "${PHASE_CUSTOM_UVM}"
modinfo -F srcversion "${PHASE_CUSTOM_UVM}"
sha256sum "${PHASE_CUSTOM_UVM}"
```

本实验只由 wrapper 临时 `insmod` workspace 中的 `nvidia-uvm.ko`。不要运行 `make modules_install`，否则可能覆盖 wrapper 准备恢复的 installed stock UVM。

### 5.2 构建 CUDA fence helper

```bash
cd "${PHASE_REPO_ROOT}"
make -C workloads/vllm/vllm/uvm_test \
  libuvm_slot_cuda_fence_kernels.so

test -f "${PHASE_FENCE_LIB}"
sha256sum "${PHASE_FENCE_LIB}"
```

## 6. 静态门禁

该步骤不初始化 CUDA、不加载模型，也不切换内核模块：

```bash
cd "${PHASE_REPO_ROOT}"
bash experiments/vllm_p3_pinned_uva_phase_compare/run_phase_compare.sh \
  static-preflight
```

预期输出包括：

```text
6 passed
static preflight: OK
```

门禁会检查：

- vLLM Python、benchmark、verifier、自定义 `.ko` 和 fence `.so` 存在；
- Python 文件可编译；
- 当前集成使用 synchronous fenced cmd96；
- vLLM 源码中没有 cmd97/cmd98/async drain 残留；
- registered P3 和 managed-UVA remote-read hook 存在；
- 6 个 CPU-only tests 通过。

## 7. GPU、模块和时钟前置检查

记录 installed stock UVM：

```bash
modinfo -F filename nvidia-uvm
modinfo -F srcversion nvidia-uvm
tr -d '[:space:]' < /sys/module/nvidia_uvm/srcversion
```

确认 GPU 严格空闲：

```bash
PHASE_USED_MIB="$(nvidia-smi -i 0 --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
PHASE_COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ {print $1}')"

test "${PHASE_USED_MIB}" = 0
test -z "${PHASE_COMPUTE_PIDS}"
sudo fuser /dev/nvidia-uvm || true
```

最后一条命令必须没有 client 输出。若有输出，应先正常停止拥有该 fd 的服务或进程；不要强卸模块。

### 7.1 可选但推荐的锁频

正式目标是 SM 2400 MHz、memory 14001 MHz。只能在 GPU/驱动支持且符合部署规范时使用相应的 `nvidia-smi` 锁频命令。例如：

```bash
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 -lgc 2400,2400
sudo nvidia-smi -i 0 -lmc 14001,14001
```

不同驱动或 GPU 可能不支持 memory lock。若不能锁频：

- 保留 conditioning repeat；
- 保留 `PCU/CUP/UPC` 平衡顺序；
- 检查每进程的 `gpu.before.csv`、`gpu.after.csv`；
- 若 P3 出现明显升频双峰，不要把冷态和稳态混成一个正式结果。

无论是否成功锁频，都应把实际执行的 lock/reset 命令另外保存到 run root，因为现有 runner 不会替你记录这些外部命令。

## 8. Qualification smoke

先运行一组 `PCU` 可以验证模块周期、三种机制和 verifier，但每种机制只有一个独立进程，不能作为正式性能统计：

```bash
cd "${PHASE_REPO_ROOT}"

PHASE_SMOKE_ROOT="${PHASE_REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_smoke"

RUN_ROOT="${PHASE_SMOKE_ROOT}" \
ORDER=PCU \
BATCH_SIZES=1,4 \
MODEL="${PHASE_MODEL}" \
EVAL_FIXTURE="${PHASE_FIXTURE}" \
CUSTOM_MODULE="${PHASE_CUSTOM_UVM}" \
VLLM_PY="${PHASE_VLLM_PY}" \
GPU_INDEX=0 \
CPUSET=0-13 \
CPU_THREADS=14 \
TIMEOUT_SEC=1800 \
bash experiments/vllm_p3_pinned_uva_phase_compare/run_with_module_cycle.sh
```

`RUN_ROOT` 必须是一个尚不存在的新目录。不要删除旧 artifact 来复用名字。

## 9. 正式九进程运行

正式运行约耗时 17 分钟，但模型加载、磁盘缓存、CPU 和 GPU 状态不同会改变总时长。每个独立进程的 timeout 在参考运行中设为 1,800 秒。

```bash
cd "${PHASE_REPO_ROOT}"

PHASE_RUN_ROOT="${PHASE_REPO_ROOT}/experiments/vllm_p3_pinned_uva_phase_compare/artifacts/$(date -u +%Y%m%dT%H%M%SZ)_formal"

RUN_ROOT="${PHASE_RUN_ROOT}" \
ORDER=PCUCUPUPC \
BATCH_SIZES=1,4 \
MODEL="${PHASE_MODEL}" \
EVAL_FIXTURE="${PHASE_FIXTURE}" \
CUSTOM_MODULE="${PHASE_CUSTOM_UVM}" \
VLLM_PY="${PHASE_VLLM_PY}" \
GPU_INDEX=0 \
CPUSET=0-13 \
CPU_THREADS=14 \
TIMEOUT_SEC=1800 \
CONDITIONING_REPEATS=1 \
MEASUREMENT_REPEATS=3 \
WARMUP_DECODE_STEPS=5 \
MEASURED_DECODE_STEPS=16 \
KV_CACHE_BYTES=268435456 \
MAX_MODEL_LEN=512 \
MAX_NUM_SEQS=4 \
MAX_NUM_BATCHED_TOKENS=2048 \
GPU_MEMORY_UTILIZATION=0.97 \
POOL_BUDGET_BYTES=1207959552 \
bash experiments/vllm_p3_pinned_uva_phase_compare/run_with_module_cycle.sh
```

wrapper 的执行顺序为：

1. 记录 stock module path/srcversion 和 exporter 初始状态；
2. 检查 GPU 0 MiB、无 compute PID、无 UVM client；
3. 停止两个 exporter；
4. 卸载 stock `nvidia_uvm`；
5. 加载 workspace custom UVM 及完整 P3 参数；
6. 校验 loaded srcversion 和所有 module parameters；
7. 自动生成 exact L0–L3 plan 和 pinned parameter-name list；
8. 按 `PCU/CUP/UPC` 运行九个独立进程；
9. 每个进程前后均要求 GPU 返回 0 MiB；
10. 运行 verifier；
11. 恢复 installed stock UVM；
12. 恢复实验前处于 active 的 exporter。

不要在 wrapper 运行期间手工加载/卸载 UVM，也不要并行运行其他 GPU 模型或 UVM 实验。

## 10. Artifact、验证与结果提取

### 10.1 目录结构

run root 顶层应包含：

```text
inputs/exact_l0_l3_plan.json
inputs/exact_parameter_names.txt
inputs/input_summary.json
prepare_inputs.stdout.json
input_sha256.txt
run_config.json
execution_status.tsv
verification.json
verify.log
verify.err
host.txt
gpu.initial.csv
gpu.final.csv
git_status.txt
module_cycle.log
```

每个 `NN_arm/` 目录应包含：

```text
command.txt
environment.txt
execution.env
run.log
resource_time.txt
returncode.txt
result.json
slot_trace.jsonl
cpu_offload_manifest.jsonl
latency_stats.before.txt
latency_stats.after.txt
gpu.before.csv
gpu.after.csv
```

UVA arm 还应有 `runtime_remote_placement_manifest.json`。

### 10.2 离线重新验证

该命令不需要再次加载模型或 custom UVM：

```bash
cd "${PHASE_REPO_ROOT}"
bash experiments/vllm_p3_pinned_uva_phase_compare/run_phase_compare.sh \
  verify "${PHASE_RUN_ROOT}" \
  > "${PHASE_RUN_ROOT}/verification.recheck.json"

jq -e '.status == "pass"' \
  "${PHASE_RUN_ROOT}/verification.recheck.json"
```

注意：verifier 只验证 artifact 的实验契约，不替代运行结束后的 live stock-module、GPU 和 exporter 检查。

### 10.3 提取四组阶段中位数

```bash
jq -r '
  def m($phase;$batch;$arm):
    .phase_metrics_ms[$phase][$batch][$arm].median;
  [
    ["phase", "batch", "P3_ms", "pinned_ms", "managed_UVA_ms"],
    ["prefill", "1", m("prefill_step_ms";"1";"p3"), m("prefill_step_ms";"1";"pinned"), m("prefill_step_ms";"1";"uva")],
    ["prefill", "4", m("prefill_step_ms";"4";"p3"), m("prefill_step_ms";"4";"pinned"), m("prefill_step_ms";"4";"uva")],
    ["decode", "1", m("decode_step_ms";"1";"p3"), m("decode_step_ms";"1";"pinned"), m("decode_step_ms";"1";"uva")],
    ["decode", "4", m("decode_step_ms";"4";"p3"), m("decode_step_ms";"4";"pinned"), m("decode_step_ms";"4";"uva")]
  ][] | @tsv
' "${PHASE_RUN_ROOT}/verification.json" | column -t
```

### 10.4 查看每个 P3 进程的完整性证据

```bash
jq '.cases[] | select(.arm == "p3") |
  {directory, evidence}' \
  "${PHASE_RUN_ROOT}/verification.json"
```

### 10.5 查看输出 hash

```bash
jq '.output_token_ids_sha256' \
  "${PHASE_RUN_ROOT}/verification.json"
```

## 11. 验收标准

### 11.1 硬性正确性标准

- `execution_status.tsv` 有 9 行 case，全部 return code 为 0。
- `verification.json` 的 `status` 为 `pass`。
- 同一 batch 的三种机制 output-token hash 完全相同。
- pinned manifest 精确包含 8 个 selected packed parameters，共 `2,415,919,104` bytes，且 CPU backing 为 pinned。
- UVA placement 精确覆盖 512 个 experts、4,608 个 regions，共 `2,415,919,104` mapped bytes。
- P3 不得有 fallback、error、conflict 或 poisoned session。

默认正式配置下，每个 P3 进程的预期计数为：

| 项目 | 每 P3 进程 |
|---|---:|
| cycles | `2 × (1 + 3) = 8` |
| forward steps/cycle | `1 + 5 + 16 = 22` |
| transitions | `8 × 22 × 4 = 704` |
| bytes/transition | 603,979,776 |
| regions/transition | 1,152 |
| shared copy pushes/transition | 1 |
| 64 KiB blocks/TLB invalidates per transition | 288 |
| copy-run cache hits | 704 |
| target-recipe hits | 704 |
| clear-recipe hits | 702，前两次为空 bank |
| registered plans | 5 |
| 显式 destroy | 4 |
| session abort | 1 次成功 |
| abort released slots | 2,304 |
| abort released bytes | 1,207,959,552 |
| abort remaining slots | 0 |

三个 P3 进程合计应有 2,112 次 transition 和 2,106 次 clear-recipe hit。每次 transition 都必须满足：cmd96 registered plan、compute fence accepted、completion signaled、主 CUDA stream wait 已入队、bank-level commit、`userspace_commit_slots=0`。

参考 output-token hash：

```text
batch 1
83dd1cdf2e151863a154ba5ee4cd1dab1b0db5e02ac0e1cd6c97c5a0c7eb7cbf

batch 4
659c552b293f3060934ae3c63414bc4c7d75717617cb89d348f9117e5f9783e0
```

如果 fixture、模型或代码有意变化，hash 可以变化，但同一 run 内三种机制仍必须一致；此时不能再声称逐字复现 2026-08-19 的冻结实验。

### 11.2 系统恢复标准

实验完成后另行检查：

```bash
PHASE_LIVE_SRCVERSION="$(tr -d '[:space:]' \
  < /sys/module/nvidia_uvm/srcversion)"
PHASE_INSTALLED_SRCVERSION="$(modinfo -F srcversion nvidia-uvm)"
PHASE_FINAL_USED_MIB="$(nvidia-smi -i 0 --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d '[:space:]')"
PHASE_FINAL_COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk '$1 ~ /^[0-9]+$/ {print $1}')"

test "${PHASE_LIVE_SRCVERSION}" = "${PHASE_INSTALLED_SRCVERSION}"
test "${PHASE_FINAL_USED_MIB}" = 0
test -z "${PHASE_FINAL_COMPUTE_PIDS}"
systemctl is-active node-exporter.service
systemctl is-active gpu-exporter.service
```

严格复现参考环境时，最后两条都应输出 `active`。如果运行前有意让某个 exporter 处于 inactive，则应与运行前保存的状态比较，而不是强制要求 active。

还应检查 `module_cycle.log` 中存在类似：

```text
recovery: loaded_srcversion=<stock srcversion> gpu_clean=yes rc=0
```

### 11.3 性能判断

verifier 没有设置“必须等于参考毫秒数”的性能阈值。硬性 pass 只代表实验契约、正确性、计数和 artifact 完整性通过。

性能复现应同时报告：

- 每个机制的三个独立 process-median 样本；
- 跨进程中位数和 IQR；
- 三个 balanced block 的 paired delta/ratio；
- GPU/CPU 时钟、P-state、功耗及是否锁频；
- 与参考硬件、软件、模型和 source snapshot 的差异。

在相同机器和稳态时，预期机制排序应与参考结果一致。不要把不同硬件上的绝对毫秒数差异直接判为功能失败。

## 12. 异常后的人工恢复

只有在确认 GPU 0 MiB、无任何 compute PID、且 `/dev/nvidia-uvm` 无 client 后才能执行模块恢复：

```bash
nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits
sudo fuser /dev/nvidia-uvm || true
```

三项确认干净后：

```bash
sudo modprobe -r nvidia_uvm
sudo modprobe nvidia
sudo modprobe nvidia_uvm
sudo systemctl start node-exporter.service gpu-exporter.service
```

然后执行第 11.2 节的 live/installed srcversion、GPU 和 exporter 检查。如果 GPU 或 UVM client 无法清理，应先定位并正常停止持有者；不要使用强制卸载。

若实验前执行过锁频，使用部署批准的恢复命令。例如：

```bash
sudo nvidia-smi -i 0 -rgc
sudo nvidia-smi -i 0 -rmc
```

## 13. 常见失败原因

### `static-preflight` 报缺少 `.ko` 或 fence `.so`

重新执行第 5 节构建步骤。不要通过删除检查或指向旧版本二进制绕过。

### wrapper 报 GPU 不干净

检查目标 GPU 显存、全机 compute PID 和 `/dev/nvidia-uvm` client。wrapper 不会替用户杀进程。

### runtime preflight 报 module parameter 不匹配

说明绕过了 module-cycle wrapper、加载了错误 `.ko`，或 custom 模块不包含当前 P3 参数。恢复 stock 后重新从第 6 节开始，不要直接修改 sysfs 来凑参数。

### P3 延迟呈明显双峰

首先检查 SM/memory clock 和 P-state。不要把升频阶段样本与稳态样本混合；重新锁频或在相同 balanced protocol 下重新运行整个九进程序列。

### verifier 报 output hash 不一致

将该 run 判为 correctness failure。优先检查模型/fixture hash、各 arm 的 `environment.txt`、P3 fallback/error、pinned exact-name manifest 和 UVA placement manifest。

### `verification.json` 已 pass，但 stock UVM 或 exporter 未恢复

verifier 不检查 live module 和 service 状态。按第 11.2、12 节单独恢复和验收。

## 14. 适用边界

- 只覆盖 eager、单 GPU、单进程、tensor parallel size 1。
- CUDA Graph 当前 fail closed，不在资格范围内。
- 未验证多 GPU、NCCL 或其他通信流。
- 这是 packed 128-expert 全量搬运，不能外推到 sparse-12 pinned slicing。
- managed-UVA 没有测量实际 PCIe demand-read 字节数。
- 每个机制只有三个独立进程，不是生产尾延迟研究。
- P3 的 compact trace 开销包含在结果中，但 benchmark hot loop 内没有文件写入、NVML 查询或 subprocess。
- decode 测的是每个 token step 都完整搬运四层 2.25 GiB，不代表未来稀疏 decode plan 的上限。
- 本实验的三种机制是 process-isolated arms，不验证同一进程内 P3 prefill 后切换 managed-UVA decode 的混合路径。

结果解释和正式结论见 `RESULTS_20260819.md`；较短的入口说明见本目录 `README.md`。
