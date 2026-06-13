# UVM GPU-only / CPU-no-access 可执行研究与实现文档

本文档面向一个具体目标：把 vLLM 中显存充足时的 weights 和 active KV cache 作为 GPU 主访问数据处理，尽量避免 CPU 用户态 PTE 长期存在，从而减少迁移过程中 CPU unmap / TLB shootdown 的同步成本。

目标不是删除 CPU/system memory backing，而是把两件事拆开：

```text
CPU resident copy:
  system memory 里有物理副本，可作为 backing、offload storage 或 DMA source/destination。

CPU mapping:
  用户态 CPU VA 上存在可 load/store 的 PTE。
```

对 weights / active KV cache，推荐先实现：

```text
weights:
  GPU local READ
  CPU resident 可保留
  CPU user mapping 可撤销
  CPU fault 先计数并 soft fallback

active KV cache:
  GPU local READ_WRITE
  CPU resident 不强制需要
  CPU user mapping 可撤销
  CPU fault 先计数并 soft fallback
```

## 1. 代码证据

### 1.1 UVM 已经区分 resident 和 mapping

`kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h`

关键语义：

1. `va_block->resident` 表示处理器在 block 中有 coherent physical copy，但不代表有虚拟映射。
2. `va_block->cpu.resident` 是 CPU cumulative resident mask。
3. `va_block->cpu.pte_bits[UVM_PTE_BITS_CPU_READ/WRITE]` 才代表 CPU user VA 是否有有效 PTE。
4. 头文件注释明确说明：`CPU_READ` bit 清掉时，CPU 没有 valid virtual mapping，CPU 访问会 fault。

重点位置：

```text
uvm_va_block.h:267-285
uvm_va_block.h:342-376
uvm_va_block.h:416-443
```

这说明“保留 CPU resident storage，但撤销 CPU mapping”是符合 UVM 状态模型的，不需要发明新语义。

### 1.2 迁移时 CPU unmap 的重成本来自 Linux MM

普通 GPU fault / migrate 路径会进入：

```text
uvm_va_block_service_locked()
-> uvm_va_block_service_copy()
-> uvm_va_block_make_resident_copy()
-> uvm_va_block_unmap_mask()
-> uvm_va_block_unmap(UVM_ID_CPU)
-> block_unmap_cpu()
-> unmap_mapping_range()
```

关键代码：

```text
uvm_va_block.c:4900-5005
uvm_va_block.c:4939-4949
uvm_va_block.c:5983-6044
uvm_va_block.c:6019-6021
```

`block_unmap_cpu()` 不只是清 UVM mask。它会调用：

```c
unmap_mapping_range(va_space->mapping,
                    uvm_va_block_region_start(block, subregion),
                    uvm_va_block_region_size(subregion), 1);
```

随后才清：

```text
block->cpu.pte_bits[CPU_READ/CPU_WRITE]
block->mapped 中的 UVM_ID_CPU
```

`uvm_migrate.c` 对这个成本有原生注释：

```text
uvm_migrate.c:281-320
uvm_migrate.c:441-442
```

注释说明 CPU unmap 在 P9/NVLINK/ATS 场景下会成为 `UvmMigrate` 瓶颈，带宽可比只创建 CPU mappings 低到 3 倍，主要来自 TLB shootdown 和 ATS/MMU notifier invalidation。

### 1.3 CPU mapping 创建也很重

CPU fault 入口：

```text
uvm.c:2726-2742
uvm_va_space.c:2748-2751
uvm_va_block.c:13650-13805
```

CPU mapping 最终由 `vm_insert_page()` 建立：

```text
uvm_va_block.c:8044-8096
uvm_va_block.c:8176-8267
uvm_va_block.c:8276-8363
uvm_va_block.c:8545-8655
```

`block_map_cpu_page_to()` 在升级已有 CPU PTE 时还会先 `unmap_mapping_range(..., PAGE_SIZE, 1)`，然后等 block tracker，再 `vm_insert_page()`。

因此优化方向不应只盯住迁移时的 unmap，还要减少不必要 CPU PTE 的建立。

### 1.4 当前已有可复用机制

当前仓库已经有 Phase F 和后台 read-dup promotion：

```text
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED
UVM_PHASE_F_ACTION_START_BG_PROMOTION
```

定义：

```text
uvm_ioctl.h:1015-1023
```

执行入口：

```text
uvm.c:3331-3479
uvm.c:3481-3601
uvm.c:3603-3785
uvm.c:3958-4025
```

VA block helper：

```text
uvm_va_block.c:8806-8874
uvm_va_block.c:8876-8945
uvm_va_block.c:8947-9171
uvm_va_block.c:9173-9550
```

后台 promotion 已经实现了需要的核心动作：

```text
CPU resident -> GPU local COPY
保留 CPU resident
设置 read_duplicated_pages
unmap GPU old remote mapping
map GPU local READ
```

见：

```text
uvm_va_block.c:9296-9302
uvm_va_block.c:9311-9315
uvm_va_block.c:9325-9335
uvm_va_block.c:9344-9352
```

这可以作为 weights GPU-only READ 的基础。

## 2. 设计目标

### 2.1 Stage A：观测 CPU 是否真的 touch

先不改变默认行为，只记录被标记为 GPU-only 候选的 range 是否发生 CPU fault。

验收目标：

```text
vLLM 推理期间：
  weights region CPU fault 数接近 0
  active KV region CPU fault 数接近 0

若 CPU fault 非 0：
  能记录 region_id / pool / fault_addr / read_or_write / fallback 次数
```

第一步实现只做 trace-only observation：

```text
不撤销 CPU PTE。
不改变 fault residency selection。
不阻止 CPU fault 正常 fallback。
只在 CPU fault path 中按 Phase F registered region 做 VA lookup。
```

对应实现点：

```text
uvm_va_block.c:
  block_cpu_fault_locked()
  -> block_stage_a_observe_cpu_fault()

uvm_ioctl.h:
  UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT
  UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING
  UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY
  UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE
```

运行时开关：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_stage_a_cpu_fault_observe_enable
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_stage_a_cpu_fault_log_enable
```

观测输出：

```bash
cat /proc/driver/nvidia-uvm/latency_stats | grep stage_a
cat /proc/driver/nvidia-uvm/phase_f_route_log | grep stage_a_cpu_fault | tail -100
```

关键 counters：

```text
stage_a_cpu_faults_weight:
  命中 weight region 的 CPU fault 数。

stage_a_cpu_faults_kv:
  命中 KV region 的 CPU fault 数。

stage_a_cpu_faults_candidate:
  命中 GPU-only candidate region 的 CPU fault 数。

stage_a_cpu_fault_fallbacks:
  trace-only 模式下等于 candidate CPU faults，含义是“如果启用 soft GPU-only，会进入 fallback 的次数”。
```

注意：现有 `analyze_uvm_fault_addresses.py` 主要分类 UVM replayable/GPU fault 地址，不能单独证明 CPU 是否 touch。Stage A 必须看 CPU fault path 的 `stage_a_*` counters 和 `stage_a_cpu_fault` route log。

### 2.2 Stage B：soft GPU-only

显式对 region 做：

```text
CPU PTE revoke/unmap
GPU local READ 或 READ_WRITE map
CPU fault 时计数，并自动解除 GPU-only policy 回退普通 UVM
```

soft 模式用于 vLLM 真实运行，不应该 fatal。

### 2.3 Stage C：hard GPU-only

CPU fault 时直接返回 fatal / SIGBUS，用于实验确认 runtime 绝不会 CPU touch。

hard 模式只用于 microbench 和受控 vLLM 实验。

## 3. 推荐 API 形态

复用现有 Phase F / Stage N region 注册体系，不新增独立设备文件。

### 3.1 新增 Phase F action

在 `uvm_ioctl.h` 扩展：

```c
typedef enum
{
    UVM_PHASE_F_ACTION_PREFETCH_CPU = 1,
    UVM_PHASE_F_ACTION_PREFETCH_GPU = 2,
    UVM_PHASE_F_ACTION_ADVISE_CPU = 3,
    UVM_PHASE_F_ACTION_QUERY_RESIDENCY = 4,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6,
    UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7,
    UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8,
    UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT = 9,
    UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT = 10,
    UVM_PHASE_F_ACTION_GPU_ONLY_READ_HARD = 11,
    UVM_PHASE_F_ACTION_GPU_ONLY_RW_HARD = 12,
    UVM_PHASE_F_ACTION_CLEAR_GPU_ONLY = 13,
} UVM_PHASE_F_ACTION;
```

### 3.2 新增 region flag

在 `uvm_ioctl.h` 的 Phase N flags 附近添加：

```c
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_SOFT       (1U << 6)
#define UVM_PHASE_N_REGION_FLAG_GPU_ONLY_HARD       (1U << 7)
#define UVM_PHASE_N_REGION_FLAG_CPU_NO_USER_MAPPING (1U << 8)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_ONLY       (1U << 9)
#define UVM_PHASE_N_REGION_FLAG_GPU_READ_WRITE      (1U << 10)
```

用途：

```text
WEIGHT region:
  pool = UVM_PHASE_N_POOL_WEIGHT
  flags = GPU_ONLY_SOFT | CPU_NO_USER_MAPPING | GPU_READ_ONLY

KV region:
  pool = UVM_PHASE_N_POOL_KV
  flags = GPU_ONLY_SOFT | CPU_NO_USER_MAPPING | GPU_READ_WRITE
```

### 3.3 为什么走 Phase F

Phase F 已有：

1. range batch executor。
2. region register / unregister。
3. batch by region id。
4. GPU UUID 参数。
5. vLLM 侧按 semantic range 上报 KV/Weight 的设计基础。

关键路径：

```text
uvm.c:3858-3928  register region
uvm.c:3958-4025  batch regions
uvm_ioctl.h:1033-1041 UVM_PHASE_F_REGION
uvm_ioctl.h:1084-1096 UVM_PHASE_F_BATCH_REGIONS
```

## 4. Driver 实现步骤

### 4.1 新增 counters

在 `uvm_va_block.c` 靠近现有 `g_uvm_latency_*` counter 添加：

```c
static atomic64_t g_uvm_gpu_only_apply_calls = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_apply_pages = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_cpu_unmap_pages = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_cpu_faults = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_cpu_write_faults = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_soft_breaks = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_hard_faults = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_map_gpu_pages = ATOMIC64_INIT(0);
static atomic64_t g_uvm_gpu_only_failed_pages = ATOMIC64_INIT(0);
```

同时更新：

```text
uvm_va_block_print_latency_stats()
uvm_va_block_reset_latency_stats()
```

位置：

```text
uvm_va_block.c:148-210
uvm_va_block.c:212-244
uvm_va_block.h:2329-2333
```

### 4.2 VA block 增加 per-page policy mask

在 `uvm_va_block_gpu_state_t` 或 `uvm_va_block_t` 中新增 mask。

推荐放在 `uvm_va_block_t`，因为 CPU fault 时不一定想先选 GPU state：

```c
uvm_page_mask_t gpu_only_soft_pages;
uvm_page_mask_t gpu_only_hard_pages;
uvm_page_mask_t gpu_only_read_pages;
uvm_page_mask_t gpu_only_rw_pages;
```

位置：

```text
uvm_va_block.h:492 附近
```

需要同步处理 block split：

```text
uvm_va_block.c:11745-11757 附近
```

把四个 mask 和 `read_duplicated_pages` 一样 split。

### 4.3 新增 helper：撤销 CPU user mapping

在 `uvm_va_block.c` 新增：

```c
NV_STATUS uvm_va_block_gpu_only_unmap_cpu(uvm_va_block_t *block,
                                          uvm_va_block_context_t *block_context,
                                          uvm_va_block_region_t region,
                                          const uvm_page_mask_t *page_mask,
                                          NvU32 *unmapped_pages_out);
```

内部逻辑：

```text
1. page_mask 限制在 region 内。
2. intersect CPU_READ pte_bits，得到真正需要 unmap 的 pages。
3. 调 uvm_va_block_unmap(block, ..., UVM_ID_CPU, region, mask, NULL)。
4. counter 加实际 CPU mapped pages。
```

注意：

`uvm_va_block_unmap(UVM_ID_CPU)` 本身返回 `NV_OK`，但会同步调用 `unmap_mapping_range()`，这正是要计量的成本。

### 4.4 新增 helper：应用 GPU-only policy

新增：

```c
NV_STATUS uvm_va_block_apply_gpu_only_policy(uvm_va_block_t *block,
                                             uvm_va_block_retry_t *retry,
                                             uvm_va_block_context_t *block_context,
                                             uvm_gpu_id_t gpu_id,
                                             uvm_va_block_region_t region,
                                             const uvm_page_mask_t *page_mask,
                                             bool read_only,
                                             bool hard,
                                             NvU32 *applied_pages_out);
```

READ-only weights 路线：

```text
1. 确保目标页 GPU resident。
   a. 若已经 GPU resident，跳过 copy。
   b. 若 CPU resident 且 ReadMostly，可复用 read-dup COPY 方式。
   c. 若普通页，先用 uvm_va_block_make_resident_copy() 迁移到 GPU。

2. 建 GPU mapping。
   READ: uvm_va_block_map(..., gpu_id, ..., UVM_PROT_READ_ONLY)
   RW:   uvm_va_block_map(..., gpu_id, ..., UVM_PROT_READ_WRITE_ATOMIC)

3. 撤销 CPU mapping。
   uvm_va_block_gpu_only_unmap_cpu()

4. 标记 policy mask。
   soft/hard
   read/rw

5. 更新 counters。
```

建议第一版约束：

```text
只支持 managed allocation。
不支持 HMM。
不支持 pageable ATS。
只允许 range 是 PAGE_SIZE 对齐。
先不支持多 GPU 同时 GPU-only。
```

### 4.5 Phase F action 入口

仿照：

```text
uvm_phase_f_prepare_remote_read()
uvm_phase_f_start_bg_promotion()
```

新增：

```c
static NV_STATUS uvm_phase_f_apply_gpu_only(const NvProcessorUuid *gpu_uuid,
                                            NvBool async,
                                            NvU64 base,
                                            NvU64 length,
                                            bool read_only,
                                            bool hard,
                                            UVM_PHASE_F_RANGE_RESULT *result,
                                            uvm_va_space_t *va_space)
```

步骤：

```text
1. retain mm。
2. va_space down_read。
3. 找 gpu。
4. range type 必须是 MANAGED。
5. 遍历 managed_range 和 va_block。
6. UVM_VA_BLOCK_LOCK_RETRY 调 uvm_va_block_apply_gpu_only_policy()。
7. result->resident_gpu_bytes = applied_pages * PAGE_SIZE。
8. route log 输出 action/gpu/base/length/applied/unmapped/status。
```

在 `uvm_phase_f_execute_range()` 增加 action 分支。

### 4.6 CPU fault policy break

在 `block_cpu_fault_locked()` 的 logical permission 后、正常 residency selection 前插入：

```text
uvm_va_block.c:13693-13709 附近
```

逻辑：

```c
if (page is in gpu_only_soft/hard mask) {
    atomic64_inc(&g_uvm_gpu_only_cpu_faults);
    if (fault_access_type > UVM_FAULT_ACCESS_TYPE_READ)
        atomic64_inc(&g_uvm_gpu_only_cpu_write_faults);

    if (page is hard) {
        atomic64_inc(&g_uvm_gpu_only_hard_faults);
        return NV_ERR_INVALID_ACCESS_TYPE;
    }

    clear gpu_only policy bits for this page;
    atomic64_inc(&g_uvm_gpu_only_soft_breaks);
    continue normal UVM CPU fault service;
}
```

soft break 后普通 UVM 会：

```text
select new residency
possibly migrate/copy back to CPU
map CPU PTE with vm_insert_page()
```

这能保证真实 vLLM 不因为偶发 CPU touch 崩溃。

### 4.7 清理路径

需要在以下路径清 policy mask：

```text
discard:
  uvm_va_block_discard / va_block_discard_locked

unregister/free:
  block destroy 路径随 struct 释放即可

split:
  block_split_page_mask()

read-dup break / write fault:
  soft break 时清对应 page
```

最小版只要保证 split 和 CPU fault soft break 正确即可；discard 可作为第二个补丁补齐。

## 5. vLLM 接入步骤

### 5.1 上报 Weight region

在模型加载完成后遍历 parameters/buffers：

```python
for p in model.parameters():
    ptr = p.data_ptr()
    nbytes = p.numel() * p.element_size()
    register_phase_f_region(
        region_id=next_id(),
        base=align_down(ptr, PAGE_SIZE),
        length=align_up(ptr + nbytes, PAGE_SIZE) - align_down(ptr, PAGE_SIZE),
        pool=UVM_PHASE_N_POOL_WEIGHT,
        flags=GPU_ONLY_SOFT | CPU_NO_USER_MAPPING | GPU_READ_ONLY,
    )
```

然后批量执行：

```text
UVM_PHASE_F_ACTION_GPU_ONLY_READ_SOFT
```

### 5.2 上报 active KV cache region

KV cache tensor 分配完成后：

```python
for t in kv_caches.values():
    ptr = t.data_ptr()
    nbytes = t.numel() * t.element_size()
    register_phase_f_region(
        region_id=next_id(),
        base=align_down(ptr, PAGE_SIZE),
        length=align_up(ptr + nbytes, PAGE_SIZE) - align_down(ptr, PAGE_SIZE),
        pool=UVM_PHASE_N_POOL_KV,
        flags=GPU_ONLY_SOFT | CPU_NO_USER_MAPPING | GPU_READ_WRITE,
    )
```

然后批量执行：

```text
UVM_PHASE_F_ACTION_GPU_ONLY_RW_SOFT
```

### 5.3 不适用对象

不要对以下对象启用 CPU-no-access：

```text
vLLM cpu_offload_gb 的 pinned CPU tensor/UVA view
KV offload CPU backend 的 pinned CPU tensor
需要 torch CPU op 直接读写的 tensor
debug/checksum 会在 CPU 侧读 unified pointer 的实验 buffer
```

这些场景 CPU mapping 仍然是对象语义的一部分。可以只做观测，不做 revoke。

## 6. Microbench 验证计划

### 6.1 新增 bench 模式

扩展：

```text
tools/uvm_readonly_access_mechanism_bench.cu
scripts/test_readonly_access_mechanisms.sh
```

新增机制：

```text
gpu_only_read_soft
gpu_only_read_hard
gpu_only_rw_soft
gpu_only_rw_hard
gpu_only_cpu_touch_soft
gpu_only_cpu_touch_hard
```

### 6.2 READ-only weights 模拟

流程：

```text
cudaMallocManaged
CPU 初始化
cudaMemAdviseSetReadMostly
cudaMemPrefetchAsync 到 GPU 或 Phase F GPU_ONLY_READ_SOFT
GPU read kernel 多轮
检查 CPU fault counter = 0
检查 second pass 接近 local/read-dup
```

预期：

```text
gpu_only_read_soft:
  gpu_only_apply_pages > 0
  gpu_only_cpu_unmap_pages > 0 或等于先前 CPU mapped pages
  gpu_only_cpu_faults = 0

gpu_only_cpu_touch_soft:
  CPU touch 后 gpu_only_cpu_faults > 0
  soft_breaks > 0
  程序继续运行

gpu_only_cpu_touch_hard:
  CPU touch 后返回 SIGBUS / CUDA 后续报错
```

### 6.3 RW KV 模拟

流程：

```text
cudaMallocManaged
GPU-only RW soft
GPU kernel 写入每页
GPU kernel 读取校验
CPU 不 touch
```

预期：

```text
GPU 写/读正确
CPU fault counter = 0
CPU PTE 不被重新建立
```

### 6.4 unmap 成本对比

对比四组：

```text
direct_migrate:
  普通迁移，迁移时会撤 CPU mapping

readonly_dup:
  ReadMostly read duplication

readmostly_prepare_remote_bg_after_first:
  remote first + bg promotion

gpu_only_read_soft:
  初始化后主动撤 CPU mapping，后续迁移/访问不反复创建/撤销 CPU PTE
```

采集：

```text
/sys/module/nvidia_uvm/parameters/uvm_perf_migrate_cpu_preunmap_enable
UVM latency counters
perf stat -e tlb:* 或 irq/ipis 相关事件
bench pass0/pass1/pass2 latency
```

## 7. 端到端 vLLM 验收

### 7.1 trace-only

先只注册 region，不执行 GPU-only action。

目标：

```text
Weight/KV region 覆盖率 > 95%
推理正常
CPU fault 统计可按 region/pool 输出
```

### 7.2 soft mode

执行：

```text
weights -> GPU_ONLY_READ_SOFT
active KV -> GPU_ONLY_RW_SOFT
```

验收：

```text
吞吐/latency 不退化
GPU-only CPU faults = 0 或极低
soft_breaks = 0 或极低
无 SIGBUS
无 Xid / UVM fatal
```

### 7.3 hard mode

仅小模型、小 batch、固定 prompt：

```text
weights -> GPU_ONLY_READ_HARD
active KV -> GPU_ONLY_RW_HARD
```

验收：

```text
正常推理可完成
若手工 CPU touch 对应 pointer，进程按预期失败
```

## 8. 风险与边界

### 8.1 CPU mapping 撤销本身仍然贵

GPU-only 第一次 apply 如果 CPU PTE 已经存在，仍会调用 `unmap_mapping_range()`。收益来自：

```text
只撤一次
后续不再反复建 CPU PTE / 撤 CPU PTE
迁移和 GPU-side promotion 不再让 CPU mapping 参与热路径
```

### 8.2 CPU resident 不等于 CPU 可访问

不要清 `cpu.resident`，除非明确要释放 system memory copy。第一阶段只清 CPU PTE。

### 8.3 ReadMostly 和 RW 要分开

weights 可以依托 read-dup 语义；KV active RW 不应标成 read-dup。KV 的 GPU-only RW 第一版应更接近“GPU preferred + CPU PTE absent + CPU fault fallback”。

### 8.4 pinned CPU / UVA offload 不适用

vLLM `cpu_offload_gb` 和 KV offload 的 pinned CPU tensor 依赖 CPU VA / CUDA host pointer 语义。不能直接撤 CPU mapping。

## 9. 推荐补丁顺序

1. `uvm_va_block`: counters + policy masks + split。
2. `uvm_va_block`: CPU fault soft/hard break。
3. `uvm_va_block`: `uvm_va_block_gpu_only_unmap_cpu()`。
4. `uvm_va_block`: `uvm_va_block_apply_gpu_only_policy()`，先只支持 already GPU resident。
5. `uvm.c/uvm_ioctl.h`: Phase F action 接入。
6. `tools/uvm_readonly_access_mechanism_bench.cu`: microbench action 调用。
7. 扩展 apply helper 支持 CPU resident -> GPU copy。
8. vLLM region 注册接入 weights/KV。
9. soft mode 端到端。
10. hard mode 小规模验证。

## 10. 最小成功标准

最小可接受实现不是一开始就提升 vLLM 性能，而是证明下面三件事：

```text
1. UVM 状态上能稳定存在：
   CPU resident = true
   CPU pte_bits READ/WRITE = false
   GPU resident/mapped = true

2. 对 GPU-only region：
   GPU 正常读写
   CPU 不 touch 时没有 CPU fault
   CPU touch 时 soft/hard 行为符合预期

3. 对 weights/KV：
   vLLM 正常推理期间 CPU 不反复 load/store 大块数据本体
   CPU mapping 可从长期必须存在降级为初始化/调试/显式 touch 才需要
```

做到这一步后，再继续比较迁移窗口、TLB shootdown、后台 promotion 和真实 vLLM tail latency，才有清晰因果链。

## 11. 命令级执行手册

本节把上面的设计拆成可以直接执行的检查、基线实验和实现后验收步骤。分清两类命令：

```text
当前可执行:
  不需要 GPU-only patch，用现有 remote-first / Phase F / vLLM region 能力复现实验背景。

GPU-only patch 后执行:
  需要先合入第 4 节的 action、mask、CPU fault policy break 和用户态 action wrapper。
```

所有命令默认从仓库根目录运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu
```

### 11.1 快速核对当前工作树

先确认自己没有混入无关变更：

```bash
git status --short
```

当前文档工作只需要新增/修改：

```text
docs/uvm_gpu_only_cpu_unmap_executable_plan.md
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
scripts/test_stage_a_vllm_cpu_touch_observation.sh
scripts/test_stage_a_cpu_touch_probe.sh
tools/uvm_stage_a_cpu_touch_probe.cu
```

如果工作树里已有别人的删除或未跟踪文件，不要顺手恢复或清理；只关注本实验相关文件。

### 11.2 复查 UVM 代码锚点

用下面命令复核 CPU resident / CPU mapping 的语义：

```bash
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h | sed -n '250,295p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h | sed -n '330,385p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h | sed -n '410,450p'
```

预期能看到：

```text
resident mask 描述 physical copy。
cpu.resident 描述 CPU cumulative residency。
cpu.pte_bits[CPU_READ/CPU_WRITE] 描述 CPU user virtual mapping。
CPU_READ 清掉时，CPU 没有 valid virtual mapping，CPU touch 会 fault。
```

复核迁移时 CPU unmap 的重路径：

```bash
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c | sed -n '4900,5010p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c | sed -n '5980,6048p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_migrate.c | sed -n '275,322p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_migrate.c | sed -n '435,446p'
```

重点检查：

```text
uvm_va_block_make_resident_copy() 会在 copy 前 unmap 旧 mapping。
block_unmap_cpu() 调 unmap_mapping_range()。
uvm_migrate.c 注释明确说明 CPU unmap / TLB shootdown 会成为瓶颈。
```

复核 CPU fault 重新建 CPU PTE 的路径：

```bash
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c | sed -n '2718,2748p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_space.c | sed -n '2738,2760p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c | sed -n '8040,8105p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c | sed -n '8170,8370p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c | sed -n '13640,13730p'
```

预期能看到：

```text
CPU page fault 最终进入 block_cpu_fault_locked()。
CPU mapping 最终通过 vm_insert_page() 建立。
已有 CPU PTE 需要升级时，路径里也会出现 per-page unmap_mapping_range()。
```

复核 Phase F 可复用入口：

```bash
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_ioctl.h | sed -n '1010,1098p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c | sed -n '3330,3608p'
nl -ba kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c | sed -n '3608,4045p'
```

预期能看到：

```text
UVM_PHASE_F_BATCH_EXECUTOR
UVM_PHASE_F_REGISTER_REGION
UVM_PHASE_F_BATCH_REGIONS
uvm_phase_f_execute_range()
uvm_phase_f_prepare_remote_read()
uvm_phase_f_start_bg_promotion()
```

### 11.3 构建并临时加载自定义 UVM 模块

只有需要跑 driver 实验时才执行。不要 `make modules_install`。

```bash
cd /home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module
make modules -j"$(nproc)"
cd /home/ubuntu/nvidia-uvm-gpu
```

临时加载：

```bash
sudo systemctl stop nvidia-persistenced 2>/dev/null || true
sudo systemctl stop gdm3 2>/dev/null || true
sleep 2
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia 2>/dev/null || true

sudo insmod kernel-module/nvidia-module/kernel-open/nvidia.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-modeset.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-drm.ko
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko

sudo systemctl start gdm3 2>/dev/null || true
lsmod | grep nvidia
```

如果只重载 `nvidia_uvm`，且 `nvidia` 主模块已经是本仓库版本，可用：

```bash
sudo modprobe -r nvidia_uvm
sudo insmod kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko
```

回到系统模块：

```bash
sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia
sudo modprobe nvidia
sudo modprobe nvidia_uvm
```

### 11.4 检查 procfs counters 和 route log

加载模块后确认 counter 节点存在：

```bash
cat /proc/driver/nvidia-uvm/latency_stats
cat /proc/driver/nvidia-uvm/phase_f_route_log | tail -80
```

重置 counters：

```bash
echo 1 | sudo tee /proc/driver/nvidia-uvm/latency_stats_reset >/dev/null
cat /proc/driver/nvidia-uvm/latency_stats
```

如果节点不存在，说明当前加载的不是带 procfs 扩展的 `nvidia-uvm.ko`，先回到 11.3 重载。

### 11.5 Stage A driver 观测开关

Stage A 只做 trace-only CPU fault 观测，不撤销 CPU PTE，不改变 UVM 默认迁移和 fault 行为。

启用观测：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_stage_a_cpu_fault_observe_enable
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_stage_a_cpu_fault_log_enable
echo 1 | sudo tee /proc/driver/nvidia-uvm/latency_stats_reset >/dev/null
```

确认 counter 存在：

```bash
cat /proc/driver/nvidia-uvm/latency_stats | grep stage_a
```

必须看到这些字段：

```text
stage_a_cpu_faults_total
stage_a_cpu_faults_registered
stage_a_cpu_faults_candidate
stage_a_cpu_read_faults
stage_a_cpu_write_faults
stage_a_cpu_faults_weight
stage_a_cpu_faults_kv
stage_a_cpu_fault_fallbacks
stage_a_last_region_id
stage_a_last_pool
stage_a_last_fault_addr
stage_a_last_is_write
```

边界要讲清楚：

```text
Stage A 观测的是 UVM CPU fault path。
如果某段 VA 仍有 valid CPU PTE，CPU load/store 可能不会进入 CPU fault path。
所以 Stage A 的结论必须配合正负样本：
  无 CPU touch 时 candidate fault 为 0。
  手动 CPU touch 已注册 candidate range 时，weight/KV counter 会增长。
```

### 11.6 Stage A vLLM CPU touch 观测

这一步回答第一个问题：vLLM 推理期间，已标记的 weight / active KV region 是否真的发生 CPU fault。

运行：

```bash
RUN_ROOT=workloads/vllm/log/stage_a_vllm_cpu_touch_observation \
PROMPTS=4 \
REQUEST_RATE=1 \
OUTPUT_LEN=32 \
MAX_MODEL_LEN=2048 \
SERVER_MAX_NUM_SEQS=2 \
SERVER_GPU_MEMORY_UTILIZATION=0.80 \
DEVICE=0 \
scripts/test_stage_a_vllm_cpu_touch_observation.sh
```

输出文件：

```text
workloads/vllm/log/stage_a_vllm_cpu_touch_observation/stage_a_cpu_touch_summary.txt
workloads/vllm/log/stage_a_vllm_cpu_touch_observation/stage_a_cpu_fault_events.log
workloads/vllm/log/stage_a_vllm_cpu_touch_observation/latency_stats_before.txt
workloads/vllm/log/stage_a_vllm_cpu_touch_observation/latency_stats_after.txt
workloads/vllm/log/stage_a_vllm_cpu_touch_observation/vllm_smoke/
```

通过条件：

```text
stage_a_hypothesis=passes_for_weight_and_kv
stage_a_cpu_faults_weight=0
stage_a_cpu_faults_kv=0
```

如果不通过，先看：

```bash
cat workloads/vllm/log/stage_a_vllm_cpu_touch_observation/stage_a_cpu_touch_summary.txt
tail -100 workloads/vllm/log/stage_a_vllm_cpu_touch_observation/stage_a_cpu_fault_events.log
```

route log 单行会包含：

```text
region_id / pool / pool_name / fault_addr / access=read|write / is_candidate / fallback=normal
```

解释：

```text
weight 非 0:
  推理期间存在 CPU 读 weight region，不能直接 hard GPU-only。

KV 非 0:
  active KV region 存在 CPU 读写，优先定位是否是 debug/checksum/offload/swap 路径。

fallbacks 非 0:
  trace-only 模式下表示“如果现在启用 soft GPU-only，这些 CPU fault 会触发 policy break”。
```

### 11.7 Stage A 手动分配链路实验

如果 11.6 的假设成立，不先接 vLLM GPU-only policy，而是先用手动 `cudaMallocManaged` 数据跑通链路：

```text
分配 weight managed range。
分配 KV managed range。
CPU 初始化。
prefetch 到 GPU。
Phase F register_region 标记为 GPU-only candidate。
GPU kernel 读取 weight、写入 KV。
按 case 选择 CPU 不 touch / touch weight / touch KV / touch both。
观察 Stage A counters 是否按预期增长。
```

运行：

```bash
RUN_ROOT=results/stage_a_cpu_touch_probe \
PAGES=1024 \
DEVICE=0 \
CASES="none weight kv both" \
scripts/test_stage_a_cpu_touch_probe.sh
```

输出文件：

```text
results/stage_a_cpu_touch_probe/summary.csv
results/stage_a_cpu_touch_probe/metadata.txt
results/stage_a_cpu_touch_probe/none/probe_stdout.txt
results/stage_a_cpu_touch_probe/weight/probe_stdout.txt
results/stage_a_cpu_touch_probe/kv/probe_stdout.txt
results/stage_a_cpu_touch_probe/both/probe_stdout.txt
```

快速看结果：

```bash
column -s, -t results/stage_a_cpu_touch_probe/summary.csv
cat results/stage_a_cpu_touch_probe/*/probe_stdout.txt
grep stage_a_cpu_fault results/stage_a_cpu_touch_probe/*/phase_f_route_log_after.txt
```

预期 counter delta：

| case | 预期 |
|---|---|
| `none` | `stage_a_cpu_faults_candidate_delta=0`，weight/KV 都为 0 |
| `weight` | `stage_a_cpu_faults_weight_delta>0`，`stage_a_cpu_read_faults_delta>0` |
| `kv` | `stage_a_cpu_faults_kv_delta>0`，`stage_a_cpu_write_faults_delta>0` |
| `both` | weight/KV 都 > 0，read/write 都 > 0 |

`probe_stdout.txt` 中必须看到：

```text
adopted_existing_uvm_fd=1
```

如果不是 1，本次结果不能用于证明 CPU fault 与 registered region 的关系。原因是 Phase F region tree 挂在 UVM VA space 上，probe 必须复用 CUDA runtime 已经打开的 `/dev/nvidia-uvm` fd；另开 fd 只适合 ioctl 调试，不适合作为 CPU touch 观测结论。

### 11.8 当前可执行的 microbench 基线

这一步不需要 GPU-only patch，用来建立 CPU unmap / remote read / read-dup / background promotion 的基线。

单次小规模：

```bash
RUN_ROOT=results/gpu_only_baseline_readonly_small \
PAGES=8192 \
PASSES=2 \
OPS_PER_PAGE=128 \
REPEATS=1 \
DEVICE=0 \
MECHANISMS="uvm_pure_remote readmostly_prepare_remote readmostly_prepare_remote_bg readmostly_prepare_remote_bg_after_first readonly_dup direct_migrate" \
scripts/test_readonly_access_mechanisms.sh
```

输出文件：

```text
results/gpu_only_baseline_readonly_small/readonly_access_mechanisms_raw.csv
results/gpu_only_baseline_readonly_small/readonly_access_mechanisms_summary.md
results/gpu_only_baseline_readonly_small/metadata.txt
```

查看摘要：

```bash
sed -n '1,220p' results/gpu_only_baseline_readonly_small/readonly_access_mechanisms_summary.md
```

扩展规模 sweep：

```bash
RUN_ROOT=results/gpu_only_baseline_scale_sweep \
PAGES_VALUES="4096 16384 65536 262144" \
PASSES=2 \
OPS_PER_PAGE=64 \
REPEATS=2 \
DEVICE=0 \
scripts/exp_prepare_remote_bg_scale_sweep.sh
```

重点看：

```text
direct_migrate:
  first pass 是否明显慢，是否对应迁移和 CPU unmap。

readonly_dup:
  first pass 支付 read-dup/migration，second pass 是否变快。

readmostly_prepare_remote_bg_after_first:
  pass0 是否接近 remote read，后续是否向 local/read-dup 收敛。
```

抓 counters：

```bash
cat /proc/driver/nvidia-uvm/latency_stats
cat /proc/driver/nvidia-uvm/phase_f_route_log | tail -120
```

### 11.9 当前可执行的 remote-first counter 验证

确认 remote-first 和 background promotion counters 确实工作：

```bash
RUN_ROOT=results/gpu_only_remote_first_counter_probe \
RESET_COUNTERS=1 \
RELOAD_UVM=0 \
PAGES=16384 \
PASSES=4 \
OPS_PER_PAGE=128 \
REPEATS=1 \
DEVICE=0 \
scripts/test_remote_first_bg_promotion_counters.sh
```

输出文件：

```text
results/gpu_only_remote_first_counter_probe/probe_raw.csv
results/gpu_only_remote_first_counter_probe/counter_delta_by_case.csv
results/gpu_only_remote_first_counter_probe/probe_summary.md
results/gpu_only_remote_first_counter_probe/latency_stats_before.txt
results/gpu_only_remote_first_counter_probe/latency_stats_after.txt
```

验收：

```text
remote_first_bg case:
  remote_first_pages > 0
  bg_promote_enqueued_pages > 0
  bg_promote_committed_pages > 0

remote_first_only case:
  remote_first_pages > 0
  bg_promote_* 基本为 0
```

### 11.10 当前可执行的 Phase F smoke

先验证用户态 ioctl 路径、region 注册、query/prefetch 都能跑通。

policy ioctl 不要求 builtin tests：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
./run_stage_m_phase_f_smoke.sh \
  --no-enable-tests \
  --step query \
  --bytes 1048576 \
  --device 0
cd /home/ubuntu/nvidia-uvm-gpu
```

region 注册/query：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
VLLM_UVM_PHASE_F_IOCTL_MODE=policy \
./run_stage_m_phase_f_smoke.sh \
  --no-enable-tests \
  --step query-region \
  --bytes 1048576 \
  --device 0
cd /home/ubuntu/nvidia-uvm-gpu
```

查看日志：

```bash
tail -200 workloads/vllm/log/phase_f_crash_diag/latest_query.log
tail -200 workloads/vllm/log/phase_f_crash_diag/latest_query-region.log
cat /proc/driver/nvidia-uvm/phase_f_route_log | tail -120
```

### 11.11 当前可执行的 vLLM region/tag-only 验证

这一步验证 vLLM 已经能把 KV/weight/scratch semantic region 上报到 driver，后续 GPU-only 直接复用这些 region id。

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/gpu_only_region_tag_only_smoke \
RESET_COUNTERS=1 \
PROMPTS=1 \
REQUEST_RATE=1 \
OUTPUT_LEN=16 \
MAX_MODEL_LEN=2048 \
SERVER_MAX_NUM_SEQS=1 \
SERVER_GPU_MEMORY_UTILIZATION=0.70 \
VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
./run_stage_n_tag_only_three_pool_smoke.sh
cd /home/ubuntu/nvidia-uvm-gpu
```

验收文件：

```text
workloads/vllm/log/gpu_only_region_tag_only_smoke/vllm_uvm_address_regions.log
workloads/vllm/log/gpu_only_region_tag_only_smoke/vllm_uvm_kv_pointer_map_stage_n.jsonl
workloads/vllm/log/gpu_only_region_tag_only_smoke/vllm_uvm_pool_residency_stage_n.jsonl
workloads/vllm/log/gpu_only_region_tag_only_smoke/phase_f_route_log_snapshot.log
workloads/vllm/log/gpu_only_region_tag_only_smoke/stage_n_acceptance_summary.txt
```

快速检查：

```bash
grep -E "phase_f_register_region|phase_n_tagged_range_register" \
  workloads/vllm/log/gpu_only_region_tag_only_smoke/phase_f_route_log_snapshot.log | head -40

grep -E "pool|kv|weight|scratch" \
  workloads/vllm/log/gpu_only_region_tag_only_smoke/stage_n_acceptance_summary.txt
```

通过条件：

```text
driver region register 成功。
KV pointer map 非空。
weight / KV / scratch 三类至少能在 trace 中区分。
没有 Xid / UVM fatal。
```

## 12. GPU-only patch 后的执行步骤

本节命令在第 4 节补丁实现后执行。

### 12.1 用户态 action wrapper 同步

实现 driver action 后，必须同步更新 vLLM allocator wrapper：

```text
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/uvm_test/phase_f_smoke.cpp
```

需要增加的 action 常量：

```text
GPU_ONLY_READ_SOFT = 9
GPU_ONLY_RW_SOFT = 10
GPU_ONLY_READ_HARD = 11
GPU_ONLY_RW_HARD = 12
CLEAR_GPU_ONLY = 13
```

需要增加的 Python 封装：

```text
driver_batch_gpu_only_read_soft(ptrs, sizes, device=0, async_=False)
driver_batch_gpu_only_rw_soft(ptrs, sizes, device=0, async_=False)
driver_batch_gpu_only_read_hard(ptrs, sizes, device=0, async_=False)
driver_batch_gpu_only_rw_hard(ptrs, sizes, device=0, async_=False)
driver_batch_clear_gpu_only(ptrs, sizes, device=0, async_=False)

driver_batch_gpu_only_read_soft_regions(region_ids, device=0, async_=False)
driver_batch_gpu_only_rw_soft_regions(region_ids, device=0, async_=False)
```

### 12.2 GPU-only smoke case 扩展

扩展：

```text
tools/uvm_readonly_access_mechanism_bench.cu
scripts/test_readonly_access_mechanisms.sh
```

新增机制：

```text
gpu_only_read_soft
gpu_only_read_hard
gpu_only_rw_soft
gpu_only_rw_hard
gpu_only_cpu_touch_soft
gpu_only_cpu_touch_hard
```

实现后跑：

```bash
RUN_ROOT=results/gpu_only_smoke_read \
RESET_COUNTERS=1 \
PAGES=8192 \
PASSES=3 \
OPS_PER_PAGE=128 \
REPEATS=1 \
DEVICE=0 \
MECHANISMS="gpu_only_read_soft gpu_only_cpu_touch_soft" \
scripts/test_readonly_access_mechanisms.sh
```

预期：

```text
gpu_only_read_soft:
  gpu_only_apply_pages > 0
  gpu_only_map_gpu_pages > 0
  gpu_only_cpu_faults = 0
  程序正确结束

gpu_only_cpu_touch_soft:
  gpu_only_cpu_faults > 0
  gpu_only_soft_breaks > 0
  程序正确结束
```

hard 模式单独跑，避免影响长实验：

```bash
RUN_ROOT=results/gpu_only_smoke_hard \
RESET_COUNTERS=1 \
PAGES=1024 \
PASSES=1 \
OPS_PER_PAGE=16 \
REPEATS=1 \
DEVICE=0 \
MECHANISMS="gpu_only_read_hard gpu_only_cpu_touch_hard" \
scripts/test_readonly_access_mechanisms.sh || true
```

预期：

```text
gpu_only_read_hard:
  正常结束。

gpu_only_cpu_touch_hard:
  CPU touch 后进程失败或收到 SIGBUS。
  latency_stats 中 gpu_only_hard_faults > 0。
```

查看 GPU-only counters：

```bash
cat /proc/driver/nvidia-uvm/latency_stats | grep -E "gpu_only|remote|bg_promote|pte_switch"
cat /proc/driver/nvidia-uvm/phase_f_route_log | grep -E "gpu_only|phase_f" | tail -120
```

### 12.3 GPU-only READ/RW 状态验收

对每个 GPU-only range，Phase F result 至少要能证明：

```text
resident_gpu_bytes == length 或接近 length
GPU kernel read/write 正确
CPU fault counter 在不 touch CPU 时为 0
```

建议增加一个 tiny probe，流程固定：

```text
1. cudaMallocManaged N pages。
2. CPU 初始化。
3. GPU_ONLY_READ_SOFT。
4. GPU read kernel 校验。
5. QUERY_RESIDENCY。
6. 读取 latency_stats。
7. CPU touch 一页。
8. 再次读取 latency_stats，验证 soft_breaks。
```

如果只做 RW KV 模拟：

```text
1. cudaMallocManaged N pages。
2. GPU_ONLY_RW_SOFT。
3. GPU write kernel 每页写固定 pattern。
4. GPU read kernel 校验 pattern。
5. CPU 不 touch。
6. gpu_only_cpu_faults 必须为 0。
```

### 12.4 vLLM trace-only 验收

先只注册 semantic region，不执行 revoke：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/gpu_only_trace_only_vllm \
RESET_COUNTERS=1 \
PROMPTS=4 \
REQUEST_RATE=1 \
OUTPUT_LEN=32 \
MAX_MODEL_LEN=2048 \
SERVER_MAX_NUM_SEQS=2 \
SERVER_GPU_MEMORY_UTILIZATION=0.80 \
VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
./run_stage_n_tag_only_three_pool_smoke.sh
cd /home/ubuntu/nvidia-uvm-gpu
```

通过条件：

```text
vLLM 推理完成。
weight/KV region 注册成功。
CPU fault address 分析中，大块 weight/KV 数据本体没有反复 CPU touch。
```

### 12.5 vLLM soft GPU-only 验收

在 vLLM allocator/residency manager 中增加一个开关，例如：

```text
VLLM_UVM_GPU_ONLY_ENABLE=1
VLLM_UVM_GPU_ONLY_MODE=soft
VLLM_UVM_GPU_ONLY_WEIGHT_ACTION=read
VLLM_UVM_GPU_ONLY_KV_ACTION=rw
```

接入点优先复用：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

推荐策略：

```text
weight region:
  register 后 batch_regions(GPU_ONLY_READ_SOFT)

active KV region:
  KV cache tensor ready 后 batch_regions(GPU_ONLY_RW_SOFT)

cpu_offload_gb / KV offload CPU backend:
  不执行 GPU-only，只保留 trace。
```

运行：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/gpu_only_soft_vllm \
RESET_COUNTERS=1 \
PROMPTS=8 \
REQUEST_RATE=2 \
OUTPUT_LEN=64 \
MAX_MODEL_LEN=2048 \
SERVER_MAX_NUM_SEQS=4 \
SERVER_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_UVM_GPU_ONLY_ENABLE=1 \
VLLM_UVM_GPU_ONLY_MODE=soft \
VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
./run_stage_n_tag_only_three_pool_smoke.sh
cd /home/ubuntu/nvidia-uvm-gpu
```

通过条件：

```text
vLLM 正常完成。
gpu_only_apply_pages > 0。
gpu_only_cpu_faults = 0 或极低。
gpu_only_soft_breaks = 0 或极低。
没有 SIGBUS。
没有 Xid / UVM fatal。
```

检查：

```bash
cat /proc/driver/nvidia-uvm/latency_stats | grep gpu_only
grep -E "gpu_only|phase_f_policy_batch_regions|phase_f_register_region" \
  workloads/vllm/log/gpu_only_soft_vllm/phase_f_route_log_snapshot.log | tail -200
tail -120 workloads/vllm/log/gpu_only_soft_vllm/vllm_server.log
tail -120 workloads/vllm/log/gpu_only_soft_vllm/vllm_bench.log
```

### 12.6 vLLM hard GPU-only 验收

hard 模式只用于小模型、小 batch、固定 prompt：

```bash
cd /home/ubuntu/nvidia-uvm-gpu/workloads/vllm
RUN_ROOT=log/gpu_only_hard_vllm \
RESET_COUNTERS=1 \
PROMPTS=1 \
REQUEST_RATE=1 \
OUTPUT_LEN=8 \
MAX_MODEL_LEN=1024 \
SERVER_MAX_NUM_SEQS=1 \
SERVER_GPU_MEMORY_UTILIZATION=0.80 \
VLLM_UVM_GPU_ONLY_ENABLE=1 \
VLLM_UVM_GPU_ONLY_MODE=hard \
VLLM_UVM_DRIVER_KV_TAG_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_ENABLE=1 \
UVM_PHASE_N_BLOCK_QUOTA_TRACE_ONLY=1 \
UVM_PHASE_N_NATIVE_BLOCK_ENABLE=1 \
./run_stage_n_tag_only_three_pool_smoke.sh
cd /home/ubuntu/nvidia-uvm-gpu
```

通过条件：

```text
正常推理时不触发 hard fault。
手工 CPU touch 对应 pointer 的专用 probe 会触发 hard fault。
```

如果 hard 模式正常推理失败，第一优先级不是调性能，而是定位谁在 CPU 侧 touch：

```text
1. route log 中找最后一个 gpu_only action。
2. latency_stats 中看 gpu_only_cpu_write_faults / gpu_only_hard_faults。
3. vLLM address region log 映射 fault_addr 到 weight/KV/scratch。
4. 对对应对象降级到 soft 或 trace-only。
```

## 13. 验收矩阵

| 阶段 | 命令入口 | 必须通过的条件 |
|---|---|---|
| UVM 代码证据 | `nl -ba ... sed -n ...` | resident 与 pte_bits 语义清楚，unmap_mapping_range 路径确认 |
| 模块加载 | `make modules` + `insmod` | `/proc/driver/nvidia-uvm/latency_stats` 存在 |
| 基线 microbench | `scripts/test_readonly_access_mechanisms.sh` | remote/read-dup/direct_migrate 行为可区分 |
| remote-first counter | `scripts/test_remote_first_bg_promotion_counters.sh` | remote_first 和 bg_promote counters 按 case 增长 |
| Phase F smoke | `run_stage_m_phase_f_smoke.sh` | query/query-region 成功，route log 有记录 |
| vLLM tag-only | `run_stage_n_tag_only_three_pool_smoke.sh` | weight/KV/scratch region 能注册和区分 |
| Stage A vLLM CPU touch | `scripts/test_stage_a_vllm_cpu_touch_observation.sh` | weight/KV 的 Stage A CPU fault delta 为 0 或可解释 |
| Stage A 手动链路 | `scripts/test_stage_a_cpu_touch_probe.sh` | none 无 candidate fault，weight/KV/both 正样本分别命中对应 counter |
| GPU-only microbench | 扩展后的 `test_readonly_access_mechanisms.sh` | GPU read/RW 正确，CPU 不 touch 时 fault 为 0 |
| soft vLLM | `VLLM_UVM_GPU_ONLY_MODE=soft` | 推理完成，soft_breaks 为 0 或极低 |
| hard vLLM | `VLLM_UVM_GPU_ONLY_MODE=hard` | 小规模推理完成，专用 CPU touch probe 能失败 |

最终可以宣称 GPU-only / CPU-no-access 成立的条件是：

```text
1. UVM VA block 状态能稳定表达 CPU resident 但 CPU PTE absent。
2. GPU 对 weight READ、KV READ_WRITE 正确。
3. vLLM 主推理路径没有反复 CPU load/store 大块 weight/KV 数据。
4. 偶发 CPU touch 在 soft 模式下可回退，在 hard 模式下可被明确抓住。
5. CPU unmap 成本从热路径中的反复成本，变成初始化/策略切换时的一次性成本。
```
