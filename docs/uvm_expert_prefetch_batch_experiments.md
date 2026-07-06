# UVM MoE Expert Prefetch Batch 实现与实验结果

本文档记录当前 MoE expert weight targeted prefetch 的实现路径、关键实验结果和结论。目标场景是：

```text
CPU 中有多个 expert weight region。
每层 router/topk 已经知道本次会访问哪些 expert。
这些 expert 通常不连续。
希望一次性把选中的 expert 权重预取到 GPU，减少 page fault 和串行 prefetch 开销。
```

当前代码已经证明：

```text
小 expert / 单 VA block expert:
  CUDA 原生 cudaMemPrefetchAsync 路径更快。

大 expert / 多 VA block expert:
  region-id expert_fast_regions batch 路径更快。

expert_fast_regions 使用 async=true 后，对 4MiB expert 有明显收益。
```

## 1. 主要实现路径

### 1.1 cuda_prefetch_serial

benchmark 名称：

```text
cuda_prefetch_serial
```

调用链：

```text
bench_uvm_expert_batch_prefetch.py
  -> batch_prefetch_ranges_to_device()
  -> uvm_batch_prefetch()
  -> for each range:
       cudaMemPrefetchAsync(ptr, size, device, stream)
```

关键文件：

```text
workloads/vllm/bench_uvm_expert_batch_prefetch.py
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

这是用户态循环调用 CUDA Runtime API。它不是 UVM ioctl batch，只是把 Python 调用开销合并进 C++ shim。

### 1.2 expert_fast_native_regions

benchmark 名称：

```text
expert_fast_native_regions
```

当前实现已经改成：

```text
Python region_id list
  -> C++ shim 查询用户态 region_id -> base/length 缓存
  -> for each region:
       cudaMemPrefetchAsync(base, length, device, stream)
```

关键文件：

```text
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

它的意义是验证：

```text
如果 vLLM 只想用 region_id 表达 expert，
但底层仍然走 CUDA 原生 prefetch，
性能是否接近 cuda_prefetch_serial。
```

实测结果是：几乎一致。

### 1.3 expert_fast_batch

benchmark 名称：

```text
expert_fast_batch
```

调用链：

```text
Python base/length list
  -> C++ UVM ioctl action 18
  -> uvm_api_phase_f_batch_executor()
  -> uvm_phase_f_expert_read_dup_fast_batch()
  -> collect base/length ranges into va_block entries
  -> sort/dedup entries
  -> uvm_va_block_expert_promote_read_duplicate_batch()
```

关键文件：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.h
workloads/vllm/vllm/vllm/device_allocator/uvm.py
workloads/vllm/vllm/uvm_test/uvm_allocator.cpp
```

这是 base/length 版本的 batch fastpath。

### 1.4 expert_fast_regions

benchmark 名称：

```text
expert_fast_regions
```

调用链：

```text
Python region_id list
  -> C++ UVM batch-regions ioctl
  -> uvm_api_phase_f_batch_regions()
  -> uvm_phase_f_expert_read_dup_fast_batch_regions()
  -> region_id -> registered expert region
  -> collect all selected region va_block entries
  -> sort/dedup entries
  -> uvm_va_block_expert_promote_read_duplicate_batch()
```

关键代码：

```text
kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c
  uvm_phase_f_expert_read_dup_fast_batch_regions()
  uvm_phase_f_expert_batch_collect_region()

kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c
  uvm_va_block_expert_promote_read_duplicate_batch()
  block_expert_promote_read_duplicate_batch_common()
```

这是当前最重要的自定义 batch 路径。它利用已注册 expert region，避免每次从 base/length 重新判断 expert weight 语义。

### 1.5 async=true

`expert_fast_regions` 和 `expert_fast_batch` 已经有 `async` 参数。

内核中关键逻辑：

```c
if (!async) {
    status = uvm_tracker_wait(&batch_tracker);
}
```

因此：

```text
async=false:
  ioctl 内等待 copy/map tracker 完成。

async=true:
  ioctl 提交 copy/map push 后不等待 batch_tracker。
```

注意：当前 async=true 不是完整 CUDA event/token 模型。它仍然在 ioctl 内同步完成：

```text
region/block 收集
prepare/populate
PTE allocation
copy/map push 提交
metadata 更新
```

只是跳过最后的 tracker wait。

### 1.6 expert_fast_regions async 使用方法

这是当前推荐的 multi-block expert 权重搬运方法。

适用条件：

```text
1. expert weight 已经通过 Phase-F region registry 注册。
2. 用户态知道要搬运的 expert region_id 列表。
3. 每个 expert 至少跨多个 UVM VA block 时收益更明显。
4. 调用方能接受 async=true 语义，即 ioctl 返回时 copy/map push 已提交，但不等待 batch_tracker。
```

内核参数建议：

```bash
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_range_tag_enable
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_migrate_fastpath_enable
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_migrate_batch_enable
echo 1 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_migrate_fastpath_align_blocks_enable
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_migrate_batch_skip_revalidate_enable
echo 0 | sudo tee /sys/module/nvidia_uvm/parameters/uvm_phase_n_expert_migrate_single_va_range_batch_enable
```

Python 直接调用：

```python
from vllm.device_allocator.uvm import driver_expert_read_dup_fast_regions

region_ids = [
    # 已注册 expert weight region ids
]

successes = driver_expert_read_dup_fast_regions(
    region_ids,
    device=0,
    async_=True,
)
```

返回值：

```text
successes >= 0:
  成功处理的 region 数。

successes < 0:
  ioctl / fd / driver 路径失败。
```

对应 C++ shim：

```text
uvm_driver_batch_expert_read_dup_fast_regions(region_ids, count, device, async)
  -> uvm_driver_phase_f_batch_regions_internal(...)
  -> UVM_PHASE_F_BATCH_REGIONS ioctl
```

对应内核路径：

```text
UVM_PHASE_F_BATCH_REGIONS
  -> uvm_api_phase_f_batch_regions()
  -> uvm_phase_f_expert_read_dup_fast_batch_regions()
  -> uvm_va_block_expert_promote_read_duplicate_batch()
```

benchmark 命令示例：

```bash
workloads/vllm/.venv/bin/python workloads/vllm/bench_uvm_expert_batch_prefetch.py \
  --expert-size 6MiB \
  --experts 32 \
  --select-count 10 \
  --selected 1,4,7,10,13,16,19,22,25,28 \
  --repeats 10 \
  --cases expert_fast_regions \
  --async-ioctl \
  --configure-kernel \
  --output workloads/vllm/results/uvm_expert_10x6mib_regions_async_10r.json \
  --no-samples
```

语义注意：

```text
async=true 并不表示 ioctl 只做 O(1) enqueue。
它仍会同步完成 region collect / block prepare / populate / PTE allocation / copy-map push。
它跳过的是最后的 uvm_tracker_wait。
```

因此实际使用时建议：

```text
1. 在 router/topk 得到 expert id 后尽早调用。
2. 后续 compute 可以和 copy/map completion 部分重叠。
3. 在真正访问 expert weight 前，需要由 CUDA stream 同步、GPU fault replay、
   或显式 query/touch 确保搬运完成。
```

当前 vLLM 接入建议：

```text
selected expert ids
  -> expert weight region ids
  -> driver_expert_read_dup_fast_regions(region_ids, async_=True)
```

最终可封装成：

```text
VLLM_UVM_WEIGHT_EXECUTOR=expert_fast_regions_async
```

## 2. Batch Helper 内部流程

核心 helper：

```text
uvm_va_block_expert_promote_read_duplicate_batch()
```

主要阶段：

```text
1. 分配 per-entry state
2. 分配 uvm_va_block_context
3. prepare 阶段:
   - 逐 block lock
   - 判断 CPU resident pages
   - populate GPU pages
   - prepare map / PTE allocation
   - unlock
4. 批量 lock 所有 block
5. revalidate
6. batch copy push
7. 更新 residency/read_duplicated_pages metadata
8. batch map push
9. 合并 tracker
10. 如果外层 async=false，则外层 wait tracker
```

当前性能大头不是 sort 或 revalidate，而是：

```text
prepare_populate_map_alloc
block_populate_pages
PTE allocation / map preparation
per-block metadata
tracker wait / DMA completion
```

## 3. 实验设置

主要 benchmark：

```text
workloads/vllm/bench_uvm_expert_batch_prefetch.py
```

固定选中的非连续 expert：

```text
1,4,7,10,13,16,19,22,25,28
```

默认：

```text
32 个 expert
选 10 个 expert
repeats = 10
device = 0
```

正确性通过 residency query 验证：

```text
selected expert GPU resident bytes == requested bytes
unselected expert GPU resident bytes == 0
```

## 4. 核心实验结果

### 4.1 10 x 2MiB，总 20MiB

结果文件：

```text
workloads/vllm/results/uvm_expert_native_cuda_regions_10x2mib_10r.json
```

| 路径 | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|
| `cuda_prefetch_serial` | 1.614 ms | 1.555 ms | 12.10 GiB/s | 10/10 |
| `expert_fast_native_regions` | 1.562 ms | 1.516 ms | 12.50 GiB/s | 10/10 |
| `expert_fast_regions` | 2.487 ms | 2.436 ms | 7.85 GiB/s | 10/10 |
| `expert_fast_batch` | 2.764 ms | 2.700 ms | 7.07 GiB/s | 10/10 |

结论：

```text
2MiB expert 基本是 1 个 UVM VA block。
此时 CUDA 原生 prefetch 路径最快。
region batch 的固定管理成本无法摊薄。
```

### 4.2 10 x 4MiB，总 40MiB

结果文件：

```text
workloads/vllm/results/uvm_expert_native_cuda_regions_10x4mib_10r.json
```

| 路径 | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|
| `cuda_prefetch_serial` | 5.274 ms | 0.330 ms | 7.41 GiB/s | 10/10 |
| `expert_fast_native_regions` | 5.168 ms | 0.333 ms | 7.56 GiB/s | 10/10 |
| `expert_fast_regions` | 3.099 ms | 3.021 ms | 12.60 GiB/s | 10/10 |
| `expert_fast_batch` | 3.307 ms | 3.248 ms | 11.81 GiB/s | 10/10 |

结论：

```text
4MiB expert = 2 个 UVM VA block。
CUDA 原生路径 submit 很快，但 complete 变慢。
expert_fast_regions 能一次聚合 20 个 VA block，
因此端到端明显更快。
```

### 4.3 expert_fast_regions async=true

4MiB 结果文件：

```text
workloads/vllm/results/uvm_expert_regions_async_probe_sync_10x4mib_10r.json
workloads/vllm/results/uvm_expert_regions_async_probe_async_10x4mib_10r.json
```

| 路径 | async | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|
| `expert_fast_regions` | false | 3.046 ms | 2.990 ms | 12.82 GiB/s | 10/10 |
| `expert_fast_regions` | true | 2.314 ms | 2.253 ms | 16.88 GiB/s | 10/10 |
| `cuda_prefetch_serial` | - | 5.311 ms | 0.421 ms | 7.35 GiB/s | 10/10 |

2MiB async 结果文件：

```text
workloads/vllm/results/uvm_expert_regions_async_probe_async_10x2mib_10r.json
```

| 路径 | async | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|
| `expert_fast_regions` | true | 2.106 ms | 2.048 ms | 9.27 GiB/s | 10/10 |
| `cuda_prefetch_serial` | - | 1.634 ms | 1.584 ms | 11.96 GiB/s | 10/10 |

结论：

```text
async=true 对 4MiB multi-block expert 有明显收益。
但对 2MiB single-block expert 仍然追不上 CUDA 原生路径。
```

### 4.4 10 x 6MiB，总 60MiB

结果文件：

```text
workloads/vllm/results/uvm_expert_10x6mib_sync_10r.json
workloads/vllm/results/uvm_expert_10x6mib_regions_async_10r.json
```

| 路径 | async | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|
| `cuda_prefetch_serial` | - | 6.305 ms | 0.369 ms | 9.29 GiB/s | 10/10 |
| `expert_fast_regions` | false | 3.576 ms | 3.512 ms | 16.39 GiB/s | 10/10 |
| `expert_fast_batch` | false | 3.875 ms | 3.823 ms | 15.12 GiB/s | 10/10 |
| `expert_fast_regions` | true | **2.525 ms** | **2.457 ms** | **23.21 GiB/s** | 10/10 |

结论：

```text
60MiB multi-block expert 搬运时，expert_fast_regions async 是当前最快路径。
这也是 vLLM 中大 expert targeted prefetch 的推荐实现。
```

### 4.5 10 x 6MiB 分段流水线

结果文件：

```text
workloads/vllm/results/uvm_expert_batch_pipeline_90x6mib_10missing_5r.json
```

实验设置：

```text
90 个 expert
每个 expert = 6MiB
10 个 missing expert = 80..89
80 个 resident expert 先放 GPU
missing expert 先放 CPU
compute window = 约 3ms
compute_windows = 2
最后触发 GPU page touch，确保如果 prefetch 没完成会真实等待
```

| 方法 | p50 total | p50 submit | 说明 |
|---|---:|---:|---|
| `full_regions_sync` | 10.868 ms | 4.253 ms | 完整 10 x 6MiB region batch，同步等待后计算 |
| `full_regions_async` | 9.979 ms | 3.284 ms | 完整 10 x 6MiB region batch，async 后计算 |
| `split_batch_inline` | 10.122 ms | 6.153 ms | 主线程：4MiB batch -> compute -> 2MiB batch -> compute |
| `split_batch_threaded` | **9.322 ms** | 8.731 ms | 后台线程提交 4MiB/2MiB，主线程直接计算 |

结论：

```text
完整 expert_fast_regions async 已经很强。
分段 inline 没有收益，因为主线程仍被两次 batch 提交阻塞。
分段 threaded 有收益，说明“更早重叠 prefetch submit 与 compute”方向有效。
```

但当前 threaded 版本只是 Python 实验，不建议直接作为生产实现。
真正接入 vLLM 时应改成 C++/CUDA 侧调度，避免 Python 线程抖动。

### 4.6 相同总量、相同 VA block 数

20MiB 总量：

```text
10 x 2MiB expert:
  10 ranges
  10 VA blocks

5 x 4MiB expert:
  5 ranges
  10 VA blocks
```

结果文件：

```text
workloads/vllm/results/uvm_expert_fast_batch_10x2mib_total20mib_10r.json
workloads/vllm/results/uvm_expert_fast_batch_5x4mib_total20mib_10r.json
```

| 形态 | range 数 | VA block 数 | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|---:|
| `10 x 2MiB` | 10 | 10 | 2.799 ms | 2.745 ms | 6.98 GiB/s | 10/10 |
| `5 x 4MiB` | 5 | 10 | 2.786 ms | 2.708 ms | 7.01 GiB/s | 10/10 |

40MiB 总量：

```text
20 x 2MiB expert:
  20 ranges
  20 VA blocks

10 x 4MiB expert:
  10 ranges
  20 VA blocks
```

结果文件：

```text
workloads/vllm/results/uvm_expert_fast_batch_20x2mib_total40mib_10r.json
workloads/vllm/results/uvm_expert_fast_batch_10x4mib_total40mib_10r.json
```

| 形态 | range 数 | VA block 数 | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|---:|
| `20 x 2MiB` | 20 | 20 | 3.379 ms | 3.321 ms | 11.56 GiB/s | 10/10 |
| `10 x 4MiB` | 10 | 20 | 3.469 ms | 3.410 ms | 11.26 GiB/s | 10/10 |

结论：

```text
expert_fast_batch / expert_fast_regions 的成本主要按 VA block 数计费，
不是按 expert/range 数计费。
```

## 5. 已尝试但不推荐的优化

### 5.1 skip revalidate

实现：

```text
uvm_phase_n_expert_migrate_batch_skip_revalidate_enable
uvm_va_block_expert_promote_read_duplicate_batch_trusted()
```

结果：

```text
收益很小，约 0% - 1% 量级，噪声内。
```

原因：

```text
lock_revalidate 不是主瓶颈。
```

### 5.2 single-va-range ordered batch / single-lock helper

实现：

```text
uvm_phase_n_expert_migrate_single_va_range_batch_enable
uvm_va_block_expert_promote_read_duplicate_batch_single_lock()
```

实验假设：

```text
同一 layer 的 expert 一起分配，落在同一个 managed va_range。
因此可以按 block index 去重，跳过 sort，并减少 lock/unlock。
```

20 repeats 结果：

```text
workloads/vllm/results/uvm_expert_regions_single_va_baseline_10x2mib_20r.json
workloads/vllm/results/uvm_expert_regions_single_va_fast_10x2mib_20r.json
```

| 路径 | 开关 | p50 complete | p50 submit | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|
| `expert_fast_regions` | off | 2.595 ms | 2.534 ms | 7.53 GiB/s | 20/20 |
| `expert_fast_regions` | on | 2.756 ms | 2.695 ms | 7.09 GiB/s | 20/20 |

结论：

```text
正确性 OK，但没有稳定性能收益，甚至略慢。
不建议作为主线优化。
当前运行时开关保持默认 0。
```

原因：

```text
sort / duplicate check / 第二次 lock 不是主要成本。
prepare_populate_map_alloc 才是主成本。
```

## 6. 当前推荐策略

推荐 hybrid policy：

```text
如果每个 selected expert <= 1 个 VA block:
  使用 cudaMemPrefetchAsync 或 expert_fast_native_regions。

如果每个 selected expert > 1 个 VA block:
  使用 expert_fast_regions，并传 async=true。
```

当前经验拐点：

```text
2MiB 和 4MiB 之间。
```

因为当前 UVM VA block 基本是：

```text
2MiB
```

所以：

```text
1MiB / 2MiB expert:
  CUDA prefetch 更适合。

4MiB / 8MiB expert:
  expert_fast_regions 更适合。
```

## 7. vLLM 接入状态

vLLM 已经具备 expert id 获取位置：

```text
FusedMoE.select_experts()
  -> topk_ids
  -> _maybe_prefetch_uvm_expert_weights(topk_ids)
```

但当前通用 weight prefetch executor 主要还是：

```text
ptr/size ranges
  -> batch_prefetch_ranges_to_device_with_executor()
  -> cpp_batch / driver_batch / auto
```

要真正使用最佳路径，需要新增 region-id executor，例如：

```text
VLLM_UVM_WEIGHT_EXECUTOR=expert_fast_regions_async
```

其行为应是：

```text
selected expert ids
  -> expert weight region ids
  -> driver_expert_read_dup_fast_regions(region_ids, async_=True)
```

同时保留 hybrid 策略：

```text
single-block expert:
  native CUDA prefetch

multi-block expert:
  expert_fast_regions async
```

## 8. 当前结论

1. `expert_fast_regions` 已经实现了“10 个不连续 expert 一起 batch”的核心目标。
2. 它可以跨多个 expert、多个 VA block，一次提交 batch copy/map。
3. 它的性能主要随 VA block 数增长，而不是随 expert/range 数增长。
4. `cudaMemPrefetchAsync` 在 single-block expert 上仍然最快。
5. `expert_fast_regions async=true` 在 multi-block expert 上是当前最佳自定义路径。
6. 继续优化时，应优先降低 `prepare_populate_map_alloc`，而不是继续削 sort、duplicate check、revalidate 或 lock/unlock。
