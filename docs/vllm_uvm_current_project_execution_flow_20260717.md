# vLLM UVM 当前项目完整执行流程

下面按原格式完整重给一版，链接更新为当前本地仓库可跳转地址，并在每个关键函数后补上“它干什么”。

当前默认主时序是：

```text
启用 UVM allocator
  -> load model / 分配 expert weights
  -> determine_available_memory / profile_run 可选
  -> 生成并分配 KV cache
  -> 保存 KV tensor 对象 / 注册 KV physical ranges / 记录地址审计
  -> compile / kernel warmup / CUDA graph / sampler warmup
  -> post-warmup 下调 KV target 并物理迁移 tail KV
  -> Stage-M：全 local experts 预注册 + hot/cold_reserved/cold_regular 分类
  -> hot expert prefetch/protect + cold expert driver release
  -> 预分配 reserved slots
  -> reserved runtime gate 打开
  -> READY 前同步 scheduler dynamic-pool target
  -> serving：scheduler 只分配 target 内 KV blocks
  -> router top-k -> reserved prefetch/remap -> fused MoE kernel
```

## 1. 进程启动：启用 UVM allocator

入口：  
[env_override.py (line 381)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/env_override.py:381)  
作用：模块 import 时读取 `VLLM_USE_UVM`，决定是否启用 UVM allocator。

[uvm.py (line 1179)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:1179)  
函数：`enable_uvm_allocator()`。  
作用：把 PyTorch CUDA allocator 切到自定义 UVM allocator，使后续 CUDA tensor 分配可以走 managed allocation。

[uvm.py (line 1136)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:1136)  
函数：`uvm_local_memory_pool()`。  
作用：为局部作用域启用 managed allocation，用于只包住某一段初始化/分配流程。

[uvm_allocator.cpp (line 6694)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:6694)  
函数：`uvm_malloc()`。  
作用：底层 managed allocation 入口，分配可被 UVM 迁移/查询/driver 管理的 VA。

[uvm_allocator.cpp (line 7818)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:7818)  
函数：`uvm_free()`。  
作用：释放 managed allocation。

含义：expert weights 和 managed KV tensor 获得 managed VA，后续才能做 CPU/GPU residency 调整。

## 2. EngineCore 初始化 executor / worker / model runner

主入口：  
[core.py (line 79)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:79)  
函数：`EngineCore.__init__()`。  
作用：创建 executor、初始化 KV cache、创建 scheduler，并进入服务准备流程。

executor 初始化：  
[uniproc_executor.py (line 26)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/uniproc_executor.py:26)  
类：`UniProcExecutor`。  
作用：单进程模式下创建本地 worker 执行环境。

[ray_executor.py (line 62)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/ray_executor.py:62)  
类：`RayDistributedExecutor`。  
作用：Ray 分布式模式下创建远端 worker。

[multiproc_executor.py (line 95)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/multiproc_executor.py:95)  
函数：`MultiprocExecutor.__init__()`。  
作用：多进程模式下拉起 worker process 和通信通道。

GPU worker：  
[gpu_worker.py (line 208)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:208)  
函数：`Worker.init_device()`。  
作用：初始化 CUDA device、分布式环境和 worker 运行状态。

[gpu_worker.py (line 299)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:299)  
动作：构造 `GPUModelRunner`。  
作用：后续 load/profile/forward/KV 初始化都由 model runner 执行。

## 3. load model / expert 参数分配

worker 入口：  
[gpu_worker.py (line 313)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:313)  
函数：`Worker.load_model()`。  
作用：调用 model runner 加载模型。

model runner 入口：  
[gpu_model_runner.py (line 5371)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5371)  
函数：`GPUModelRunner.load_model()`。  
作用：执行 `get_model_loader(...).load_model(...)`，创建模型参数。

含义：MoE expert 参数在此阶段创建；启用 UVM allocator 后，expert weights 通常是 managed allocation，load 后初始倾向 GPU resident。

## 4. profile 可用显存：不执行最终 expert placement

EngineCore 入口：  
[core.py (line 269)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:269)  
函数：`EngineCore._initialize_kv_caches()`。  
作用：决定 KV cache 可用容量并生成 KV config。

executor RPC：  
[abstract.py (line 129)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/abstract.py:129)  
函数：`Executor.determine_available_memory()`。  
作用：向 worker 查询可用于 KV 的显存。

worker：  
[gpu_worker.py (line 337)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:337)  
函数：`Worker.determine_available_memory()`。  
作用：普通路径会 profile，UVM fast path 可直接使用指定 KV size。

[gpu_worker.py (line 394)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:394)  
动作：读取 `VLLM_UVM_KV_CACHE_SIZE_GB`。  
作用：UVM fast path 生效时，用该值作为 KV cache size。

[gpu_worker.py (line 413)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:413)  
调用：`self.model_runner.profile_run()`。  
作用：非 fast path 时通过 dummy forward 探测可用显存。

model runner：  
[gpu_model_runner.py (line 6264)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6264)  
函数：`GPUModelRunner.profile_run()`。  
作用：做 dummy forward、profiling、compile 相关准备。

[gpu_model_runner.py (line 6343)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6343)  
调用：`_dummy_run(..., is_profile=True)`。  
作用：用正常 forward 流程探测运行内存。

关键点：当前 `profile_run()` 不执行最终 `run_uvm_init_time_weight_placement()`，expert hot/cold placement 延后到 post-warmup。

## 5. 生成 KV cache config

KV spec RPC：  
[abstract.py (line 132)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/abstract.py:132)  
函数：`Executor.get_kv_cache_specs()`。  
作用：从 worker 收集每层 KV spec。

[gpu_worker.py (line 472)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:472)  
函数：`Worker.get_kv_cache_spec()`。  
作用：转发 model runner 的 KV spec。

[gpu_model_runner.py (line 8200)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:8200)  
函数：`GPUModelRunner.get_kv_cache_spec()`。  
作用：按 attention backend 生成每层 KV cache 规格。

生成 config：  
[core.py (line 299)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:299)  
调用：`get_kv_cache_configs(...)`。  
作用：把 worker KV spec 和可用内存转成 worker/scheduler configs。

[kv_cache_utils.py (line 1314)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:1314)  
函数：`get_kv_cache_configs(...)`。  
作用：根据可用内存和 KV spec 生成 KV cache 配置。

[kv_cache_utils.py (line 1248)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_utils.py:1248)  
函数：`generate_scheduler_kv_cache_config(...)`。  
作用：生成 scheduler 侧只需要的 block-level KV config。

## 6. 分配 KV cache tensor

executor 入口：  
[abstract.py (line 110)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/abstract.py:110)  
函数：`Executor.initialize_from_config()`。  
作用：把 KV config 下发给 worker。

worker：  
[gpu_worker.py (line 475)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:475)  
函数：`Worker.initialize_from_config()`。  
作用：初始化 worker KV cache。

[gpu_worker.py (line 169)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:169)  
函数：`_maybe_get_kv_cache_uvm_memory_pool_context()`。  
作用：决定 KV tensor 是否用 UVM memory pool 分配。

[gpu_worker.py (line 173)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:173)  
配置：`VLLM_UVM_KV_CACHE_MANAGED`。  
作用：为 KV cache 单独打开 managed allocation。

model runner：  
[gpu_model_runner.py (line 8027)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:8027)  
函数：`initialize_kv_cache()`。  
作用：保存 config 并初始化 KV tensors。

[gpu_model_runner.py (line 7866)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:7866)  
函数：`initialize_kv_cache_tensors()`。  
作用：分配、reshape、bind KV cache。

[gpu_model_runner.py (line 7618)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:7618)  
函数：`_allocate_kv_cache_tensors()`。  
作用：为 KV 分配 raw tensor。

[gpu_model_runner.py (line 7644)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:7644)  
动作：`torch.zeros(..., device=self.device)`。  
作用：实际触碰并初始化 KV pages。

含义：KV 在 expert 之后分配并清零；KV/expert 都是 UVM managed 时，KV 可能把部分 expert pages 挤出 GPU。

## 7. 记录 weight / KV 地址和 residency metadata

weight 地址：  
[gpu_model_runner.py (line 4659)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:4659)  
函数：`_log_model_weight_addresses()`。  
作用：扫描模型权重地址、写日志、注册非 MoE weight residency。

[gpu_model_runner.py (line 4148)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:4148)  
函数：`_collect_weight_address_rows()`。  
作用：遍历 parameters/buffers 收集 VA range 和语义信息。

[gpu_model_runner.py (line 4073)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:4073)  
函数：`_register_uvm_weight_residency_tags()`。  
作用：当前跳过 `is_moe_expert`，只注册非 MoE/fixed weight。

KV 地址审计：  
[gpu_model_runner.py (line 5073)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5073)  
函数：`_log_kv_cache_addresses()`。  
作用：记录 KV tensor 的整体 VA range 到审计日志。

[gpu_model_runner.py (line 4541)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:4541)  
函数：`_collect_kv_cache_address_rows()`。  
作用：收集 KV tensor start/end/size。

[gpu_model_runner.py (line 7961)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:7961)  
动作：保存 `_uvm_kv_caches_by_layer = kv_caches`。  
作用：保留真实 KV tensor 对象，供 post-warmup 和动态 KV 迁移计算 raw ranges。

KV physical range 注册：  
[gpu_model_runner.py (line 5092)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5092)  
函数：`_register_uvm_kv_cache_physical_ranges()`。  
作用：把 contiguous KV tensor span 注册成可按 block_id 推导 VA 的物理 range map。

[gpu_model_runner.py (line 5155)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5155)  
调用：`register_uvm_kv_cache_physical_ranges(...)`。  
作用：把 model runner 收集到的 tensor span 写入 policy 侧 registry。

[uvm_kv_runtime_policy.py (line 65)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:65)  
函数：`register_uvm_kv_cache_physical_ranges()`。  
作用：保存 KV physical span registry，并可注册 `kv_phys:*` residency region。

[uvm_kv_runtime_policy.py (line 162)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:162)  
函数：`collect_uvm_kv_cache_physical_ranges_for_blocks()`。  
作用：把 block ids 转成真实 `ptr/bytes` ranges。

region manager：  
[uvm_pool_residency.py (line 1983)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py:1983)  
函数：`register_uvm_residency_region()`。  
作用：统一注册 residency region。

[uvm_pool_residency.py (line 788)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py:788)  
函数：`UvmPoolResidencyManager.register_region()`。  
作用：维护 region 状态和 metadata。

说明：Stage-L6 pointer map 已删除；当前保留地址审计，并新增 KV physical range registry，用于 target 变化后的自动物理搬运。

## 8. compile / kernel warmup / CUDA graph / sampler warmup

executor 入口：  
[abstract.py (line 119)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/executor/abstract.py:119)  
调用：`collective_rpc("compile_or_warm_up_model")`。  
作用：让所有 worker 执行 warmup。

worker：  
[gpu_worker.py (line 794)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:794)  
函数：`Worker.compile_or_warm_up_model()`。  
作用：统一编排 compile、kernel warmup、CUDA graph、sampler warmup。

[gpu_worker.py (line 829)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:829)  
动作：compile dummy run。  
作用：用 dummy input 触发编译路径。

[gpu_worker.py (line 840)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:840)  
动作：`kernel_warmup(self)`。  
作用：触发 kernel autotune / warmup。

[gpu_worker.py (line 847)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:847)  
动作：`capture_model()`。  
作用：执行 CUDA graph capture。

[gpu_worker.py (line 921)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:921)  
动作：sampler warmup。  
作用：预热 logits/sampling buffer。

含义：允许 warmup 正常 forward；最终 KV/expert residency 调整放在 warmup 后，避免 warmup 污染最终 placement。

## 9. post-warmup 总编排入口

正常 warmup 后：  
[gpu_worker.py (line 936)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:936)  
调用：`self._run_uvm_post_warmup_rebalance()`。  
作用：warmup 完成后进入 UVM 最终重平衡。

跳过 warmup 兼容路径：  
[gpu_worker.py (line 811)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:811)  
调用同一函数。  
作用：保证 no-warmup 场景也能做 UVM rebalance。

真正编排：  
[gpu_worker.py (line 702)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:702)  
函数：`Worker._run_uvm_post_warmup_rebalance()`。  
作用：固定执行：发布 KV target -> 物理迁移 KV tail -> Stage-M expert placement -> reserved slot prealloc -> 打开 runtime gate。

兼容分支：  
[gpu_worker.py (line 616)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:616)  
动作：可提前执行 `run_uvm_init_time_weight_placement()`。  
作用：显式设置 `VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=pre_warmup` 时，可提前执行 Stage-M；默认是 post-warmup。

## 10. post-warmup 下调 KV GPU resident target

target 来源：`VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS`，兼容 `VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS`。

worker 发布 target：  
[gpu_worker.py (line 672)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:672)  
函数：`_write_uvm_post_warmup_kv_target()`。  
作用：把 target 写入 dynamic pool control file。

KV tail range 计算：  
[gpu_model_runner.py (line 5181)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5181)  
函数：`_collect_uvm_post_warmup_kv_tail_ranges()`。  
作用：按 `target_blocks` 找出 target 后面的 KV tensor VA ranges。

[gpu_model_runner.py (line 207)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:207)  
函数：`_uvm_kv_tail_range_for_allocation()`。  
作用：根据 `ptr + block_id * page_size` 算 tail range。

物理迁移：  
[gpu_model_runner.py (line 5283)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5283)  
函数：`rebalance_uvm_kv_cache_after_warmup()`。  
作用：执行一次 post-warmup KV tail CPU offload。

[gpu_model_runner.py (line 5322)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:5322)  
调用：`batch_offload_ranges_to_cpu_with_executor(..., sync=True)`。  
作用：把 tail ranges 批量迁到 CPU，并同步等待完成。

[uvm.py (line 4205)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:4205)  
函数：`batch_offload_ranges_to_cpu_with_executor()`。  
作用：优先 driver batch，失败时 fallback 到 C++ batch。

[uvm_allocator.cpp (line 8495)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:8495)  
函数：`uvm_batch_prefetch()`。  
作用：底层批量 `cudaMemPrefetchAsync` 到 CPU/GPU。

含义：KV tensor 不释放，只把 target 外 managed pages 迁到 CPU，让 GPU 物理页留给 hot experts 和 reserved slots。

## 11. 后续 target 改变时自动搬运 KV blocks

target 刷新：  
[uvm_kv_runtime_policy.py (line 599)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:599)  
函数：`apply_unified_kv_target_blocks()`。  
作用：统一应用外部 target，并触发 dynamic pool 更新。

[uvm_kv_runtime_policy.py (line 1183)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1183)  
函数：`_ensure_dynamic_pool_applied()`。  
作用：检测 grow/shrink，并只处理 `ref_cnt == 0` 的 safe free blocks。

物理搬运：  
[uvm_kv_runtime_policy.py (line 1004)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1004)  
函数：`_migrate_dynamic_pool_physical_blocks()`。  
作用：shrink 时 CPU offload，grow 时 GPU prefetch。

[uvm_kv_runtime_policy.py (line 1041)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1041)  
调用：`collect_uvm_kv_cache_physical_ranges_for_blocks(...)`。  
作用：把 block ids 转成可提交给 prefetch/offload executor 的真实 ranges。

[uvm_kv_runtime_policy.py (line 1240)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1240)  
动作：target shrink 后迁移被移出 target 的 free blocks。  
作用：把 `old_target > block_id > new_target` 且 safe 的 block 迁到 CPU。

[uvm_kv_runtime_policy.py (line 1270)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1270)  
动作：target grow 后 prefetch 新纳入 target 的 free blocks。  
作用：把 `old_target < block_id <= new_target` 且 safe 的 block 迁回 GPU。

[uvm_kv_runtime_policy.py (line 1321)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1321)  
函数：`_mark_dynamic_pool_block_cpu()`。  
作用：更新 block lifecycle 和 residency metadata。

含义：当前实现不再只是 metadata-only；只要 target 改变且 KV physical ranges 可见，就会对 safe free KV blocks 发起真实物理搬运。

## 12. Stage-M：全 local experts 覆盖 + hot/cold placement

worker 调用：  
[gpu_worker.py (line 752)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:752)  
调用：`run_uvm_init_time_weight_placement(prealloc_reserved=False)`。  
作用：post-warmup 阶段执行最终 expert placement，但暂不在该函数内部预分配 reserved slots。

model runner：  
[gpu_model_runner.py (line 6363)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6363)  
函数：`GPUModelRunner.run_uvm_init_time_weight_placement()`。  
作用：遍历 MoE module 并执行 init-time placement。

MoE module：  
[layer.py (line 6652)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6652)  
函数：`FusedMoE.run_uvm_init_weight_offload()`。  
作用：执行本层 hot/cold 注册、hot prefetch、cold driver release。

全 local expert 覆盖：  
[layer.py (line 1265)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1265)  
配置：`VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS`。  
作用：决定是否覆盖注册本 rank 本层所有 local experts。

[layer.py (line 1269)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1269)  
配置：`VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN`。  
作用：决定 cold plan 是否从“计划中 expert”扩展为“所有非 hot local experts”。

[layer.py (line 2834)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2834)  
函数：`_uvm_all_local_expert_pairs()`。  
作用：列出本 rank 本层所有 `(global_expert_id, local_expert_id)`。

[layer.py (line 2896)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2896)  
函数：`_uvm_init_weight_expert_class()`。  
作用：把 expert 分类为 `hot` / `cold_reserved` / `cold_regular`。

[layer.py (line 5279)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:5279)  
函数：`_register_uvm_init_all_local_weight_regions()`。  
作用：预注册全部 local expert logical regions。

hot expert：  
[layer.py (line 5437)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:5437)  
函数：`_register_uvm_init_hot_weight_regions()`。  
作用：把 hot experts 标为 GPU/protected/non-evictable。

[layer.py (line 5590)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:5590)  
调用：`batch_prefetch_ranges_to_device_with_executor()`。  
作用：把 hot expert 物理拉回 GPU。

cold expert：  
[layer.py (line 6058)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6058)  
调用：`driver_expert_offload_release_regions(...)`。  
作用：非 batch 模式下通过自定义 driver release 释放 cold expert GPU PMM backing。

[layer.py (line 6201)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6201)  
调用：`driver_expert_offload_release_regions(...)`。  
作用：batch 模式下统一对 cold expert region ids 执行 driver release。

含义：Stage-M 后本层 local experts 不再有未知 expert；hot 留 GPU，cold_reserved 可走 reserved prefetch，cold_regular 是普通 CPU offload/release target。当前最新实验的 Stage-M cold 路径只保留 driver release 系列，旧的 `advise_cpu/prefetch_cpu/offload_cpu` init 分支已经删除。

## 13. Stage-M 后预分配 reserved slots

入口：  
[gpu_model_runner.py (line 6497)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6497)  
函数：`_maybe_prealloc_uvm_expert_reserved_slots_after_stage_m()`。  
作用：按环境变量预分配 reserved slot pool。

[gpu_model_runner.py (line 6576)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6576)  
调用：`driver_expert_reserved_pool_query(...)`。  
作用：先查询已有 slot backing。

[gpu_model_runner.py (line 6577)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:6577)  
调用：`driver_expert_reserved_pool_prealloc(...)`。  
作用：向 driver 请求预分配 slot backing。

[uvm.py (line 2538)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:2538)  
函数：`driver_expert_reserved_pool_prealloc()`。  
作用：Python ioctl wrapper。

kernel ioctl：  
[uvm.c (line 11392)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:11392)  
函数：`uvm_api_expert_reserved_pool()`。  
作用：处理 reserved pool query/prealloc/reset。

[uvm.c (line 828)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:828)  
函数：`uvm_expert_reserved_pool_prealloc_entry()`。  
作用：为单个 slot 分配 backing。

[uvm.c (line 902)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:902)  
调用：`uvm_pmm_gpu_alloc_user()`。  
作用：创建 GPU PMM chunks。

含义：serving 前预先开辟真实 GPU backing，不在 expert miss 关键路径懒分配。

## 14. reserved runtime gate

配置读取：  
[layer.py (line 1109)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:1109)  
字段：`uvm_expert_reserved_enable_after_warmup`。  
作用：控制 reserved path 是否等 post-warmup 才开启。

初始化关闭：  
[gpu_worker.py (line 610)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:610)  
设置：`VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY=0`。  
作用：worker 初始化阶段先关闭 runtime reserved path。

prealloc 成功后：  
[gpu_worker.py (line 781)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:781)  
设置：`VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY=1`。  
作用：slot backing 准备好后打开 runtime gate。

runtime 检查：  
[layer.py (line 6418)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6418)  
作用：gate 未打开时 reserved prefetch path 不进入。

含义：slot prealloc 不完整时 gate 保持 `0`，默认 fail-fast，避免无 backing 时进入 reserved path。

## 15. EngineCore 创建 scheduler / KVCacheManager / BlockPool

scheduler：  
[core.py (line 161)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:161)  
动作：`Scheduler = vllm_config.scheduler_config.get_scheduler_cls()`。  
作用：根据配置选择 scheduler 类。

[scheduler.py (line 56)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py:56)  
类：`Scheduler`。  
作用：负责每 step 选择请求、分配 tokens/KV blocks。

KV manager：  
[kv_cache_manager.py (line 94)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py:94)  
类：`KVCacheManager`。  
作用：维护 request 到 KV blocks 的映射。

[kv_cache_manager.py (line 117)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py:117)  
构造：`UvmKvRuntimePolicy(kv_cache_config)`。  
作用：创建 UVM KV runtime policy。

[uvm_kv_runtime_policy.py (line 274)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:274)  
类：`UvmKvRuntimePolicy`。  
作用：控制 dynamic pool、KV metadata、后续 target 改变时的 safe block 迁移。

block pool：  
[kv_cache_coordinator.py (line 526)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_coordinator.py:526)  
函数：`get_kv_cache_coordinator()`。  
作用：创建 KV coordinator 和 BlockPool。

[block_pool.py (line 128)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:128)  
类：`BlockPool`。  
作用：管理 free/used KV blocks。

## 16. READY 前同步 scheduler KV dynamic-pool target

入口：  
[core.py (line 252)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:252)  
调用：`self._maybe_apply_pre_service_uvm_kv_dynamic_pool()`。  
作用：READY 前同步 scheduler 侧 KV dynamic pool。

[core.py (line 319)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:319)  
函数：`EngineCore._maybe_apply_pre_service_uvm_kv_dynamic_pool()`。  
作用：在 READY 前把 scheduler 的 target 和 worker post-warmup target 对齐。

[uvm_kv_runtime_policy.py (line 599)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:599)  
函数：`apply_unified_kv_target_blocks()`。  
作用：应用 unified target。

[uvm_kv_runtime_policy.py (line 1183)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1183)  
函数：`_ensure_dynamic_pool_applied()`。  
作用：更新 target 外 block lifecycle。

含义：worker 已迁移 KV tail；scheduler 侧同步后，target 外 block 不会在 READY 后被当作普通 GPU-free block 分配。

## 17. 接收请求 / scheduler 分配 KV blocks

服务期主循环：  
[core.py (line 566)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:566)  
函数：`EngineCore.step()`。  
作用：schedule -> execute -> update。

[core.py (line 601)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:601)  
函数：`EngineCore.step_with_batch_queue()`。  
作用：batch queue 模式下的 schedule/execute 流程。

调度：  
[scheduler.py (line 197)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py:197)  
函数：`Scheduler.schedule()`。  
作用：计算本 step 每个请求推进多少 token。

[scheduler.py (line 286)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py:286)  
调用：`kv_cache_manager.allocate_slots(...)`。  
作用：为 running request 分配 KV blocks。

[scheduler.py (line 560)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/sched/scheduler.py:560)  
调用：`kv_cache_manager.allocate_slots(...)`。  
作用：为 waiting request 分配 KV blocks。

[kv_cache_manager.py (line 207)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/kv_cache_manager.py:207)  
函数：`KVCacheManager.allocate_slots()`。  
作用：计算需要新增的 block 数，并实际从 BlockPool 取 block。

[uvm_kv_runtime_policy.py (line 1550)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:1550)  
函数：`should_allow_allocation()`。  
作用：记录 pressure，并在 enforce 模式下可拒绝分配。

[block_pool.py (line 483)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/block_pool.py:483)  
函数：`BlockPool.get_num_free_blocks()`。  
作用：dynamic pool 开启时不再返回全局 free blocks。

[uvm_kv_runtime_policy.py (line 575)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/core/uvm_kv_runtime_policy.py:575)  
函数：`get_num_allocatable_free_blocks()`。  
作用：只统计 `0 < block_id <= target_blocks` 的 free blocks。

含义：target 外 KV blocks 即使在 free queue 里，也不会被 scheduler 当作可分配 GPU KV 容量。

## 18. prefill / decode model forward

EngineCore：  
[core.py (line 578)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/engine/core.py:578)  
调用：`model_executor.execute_model(...)`。  
作用：把 scheduler output 下发给 executor。

worker：  
[gpu_worker.py (line 975)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py:975)  
函数：`Worker.execute_model()`。  
作用：接收 scheduler output 并调用 model runner。

model runner：  
[gpu_model_runner.py (line 2996)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:2996)  
函数：`GPUModelRunner.execute_model()`。  
作用：准备 input batch、执行 forward、采样输出。

[gpu_model_runner.py (line 2699)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:2699)  
函数：`_uvm_enabled_step_phase()`。  
作用：标记当前是 prefill/decode 等 UVM phase。

[gpu_model_runner.py (line 2863)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py:2863)  
函数：`_model_forward()`。  
作用：真正调用模型 forward。

## 19. MoE router 计算 top-k expert

以 Qwen3-MoE 为例：  
[qwen3_moe.py (line 586)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py:586)  
调用：`router_logits, _ = self.gate(hidden_states)`。  
作用：计算每个 token 对 expert 的 logits。

[qwen3_moe.py (line 611)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py:611)  
调用：`self.experts(...)`。  
作用：进入 FusedMoE。

FusedMoE top-k：  
[layer.py (line 2472)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2472)  
函数：`FusedMoE.select_experts()`。  
作用：根据 router logits 选择 top-k expert。

[layer.py (line 2563)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2563)  
调用：`fused_topk(...)`。  
作用：输出 `topk_weights` 和 `topk_ids`。

含义：dummy/profile/warmup/serving 都走正常 router；没有强制遍历所有 expert 的隐藏路径。

## 20. top-k 后执行 reserved fast-gate expert hook

[layer.py (line 2664)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:2664)  
调用：`_maybe_restore_uvm_reserved_expert_weights(...)`。  
作用：仅当当前 top-k 命中 cold_reserved/offloaded expert 时，才进入 reserved slot restore/remap。

[layer.py (line 6403)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6403)  
函数：`_maybe_restore_uvm_reserved_expert_weights()`。  
作用：reserved-only runtime 入口；检查 runtime gate、筛选 active reserved experts、注册 active region，并提交 reserved batch。

说明：当前主线已经不再从 top-k 后触发 ordinary active top-k prefetch、cold streaming，或 runtime cold offload。cold expert 的 CPU release 在 Stage-M 完成；serving 时只对 router 当前命中的 cold_reserved expert 做 reserved fast-gate restore/remap。

## 21. 构造 reserved expert prefetch batch

入口：  
[layer.py (line 3650)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3650)  
函数：`_run_uvm_expert_reserved_prefetch_batch()`。  
作用：筛选 reserved-offloaded candidate 并构造 batch。

[layer.py (line 3542)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3542)  
函数：`_uvm_expand_reserved_items_to_chunk_envelopes()`。  
作用：把 logical region 展开到配置粒度 chunk envelope；当前 512KiB 主线会在 driver 侧按 64KiB subchunk release/restore。

[layer.py (line 3693)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3693)  
调用：`get_expert_reserved_regions_executor(...)`。  
作用：获取 reserved executor。

[layer.py (line 3704)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:3704)  
调用：`executor.prefetch_for_layer(...)`。  
作用：真正提交本层 reserved prefetch。

含义：这里把 router 选中的 cold_reserved expert 转成 slot prefetch/remap 请求。

## 22. reserved executor 校验 slot、选择/复用 slot

executor 入口：  
[uvm_expert_reserved_executor.py (line 588)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:588)  
函数：`prefetch_for_layer()`。  
作用：处理本层 reserved prefetch 请求。

预分配校验：  
[uvm_expert_reserved_executor.py (line 334)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:334)  
函数：`_ensure_preallocated_locked()`。  
作用：只验证已有 slot backing，不在 miss 路径懒分配。

slot 复用：  
[uvm_expert_reserved_executor.py (line 450)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:450)  
函数：`_invalidate_slots_for_reuse_locked()`。  
作用：等待旧 event、invalidate 旧 owner mapping，然后复用 slot。

含义：reserved slot 的 PMM chunks 长期存在，只在不同 expert 之间复用映射。

## 23. Python reserved prefetch ioctl

executor 发起：  
[uvm_expert_reserved_executor.py (line 935)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:935)  
调用：`driver_expert_reserved_prefetch_regions(...)`。  
作用：把 reserved prefetch 请求交给 Python driver wrapper。

Python ioctl 封装：  
[uvm.py (line 3134)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm.py:3134)  
函数：`driver_expert_reserved_prefetch_regions()`。  
作用：把 region/slot 参数打包给 C++ helper。

C++ helper：  
[uvm_allocator.cpp (line 8995)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/uvm_test/uvm_allocator.cpp:8995)  
函数：`uvm_driver_expert_reserved_prefetch_regions()`。  
作用：调用 UVM driver ioctl。

含义：传入 `driver_region_id + pool_id + slot_id + slot_offset + bytes_override`，mode 为 reserved prefetch。

## 24. kernel ioctl：解析 region、释放旧 slot mapping、安装新 mapping

ioctl 入口：  
[uvm.c (line 11530)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:11530)  
函数：`uvm_api_expert_reserved_prefetch_batch()`。  
作用：处理 reserved prefetch batch。

解析 region/slot：  
[uvm.c (line 590)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:590)  
函数：`uvm_expert_reserved_prefetch_resolve_entry_locked()`。  
作用：解析 VA region、slot、offset、大小。

安装入口：  
[uvm.c (line 9735)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9735)  
函数：`uvm_expert_reserved_prefetch_install_entry_locked()`。  
作用：执行 release old mapping + install new mapping。

释放旧 owner mapping：  
[uvm.c (line 9497)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9497)  
函数：`uvm_expert_reserved_pool_release_slot_mapping_locked()`。  
作用：清除 slot 当前 owner 的 VA/PTE 映射。

安装新 owner mapping：  
[uvm.c (line 9589)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9589)  
函数：`uvm_expert_reserved_pool_install_entry_mapping_locked()`。  
作用：把新 owner expert VA 映射到 reserved slot backing。

含义：slot 物理 chunks 不重新申请，只替换 owner expert 和 owner VA 映射。

## 25. 实际搬运：CPU expert pages -> reserved GPU PMM chunks

跨 VA block 映射入口：  
[uvm.c (line 9556)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c:9556)  
函数：`uvm_expert_reserved_pool_map_va_block_reserved_locked()`。  
作用：对单个 VA block 执行 reserved map。

[uvm_va_block.c (line 11151)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11151)  
函数：`uvm_va_block_copy_and_map_reserved_readonly()`。  
作用：copy CPU pages 并安装 readonly GPU mapping。

真实 copy：  
[uvm_va_block.c (line 10966)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:10966)  
函数：`block_copy_cpu_pages_to_reserved_locked()`。  
作用：逐页从 CPU 拷贝到 reserved slot PMM chunks。

含义：通过 `UVM_CHANNEL_TYPE_CPU_TO_GPU` 和 CE `memcopy()` 搬运 4KiB pages；source 必须完整 CPU resident，目标 slot range 必须足够。

## 26. PTE remap：原 expert VA -> reserved slot backing

PTE 安装：  
[uvm_va_block.c (line 11042)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11042)  
函数：`block_map_reserved_readonly_locked()`。  
作用：把原 VA 映射到 reserved chunks。

[uvm_va_block.c (line 11111)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:11111)  
调用：`block_gpu_pte_write_reserved_chunk_4k()`。  
作用：写 GPU PTE。

[uvm_va_block.c (line 7180)](/home/ubuntu/nvidia-uvm-gpu/kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm_va_block.c:7180)  
函数：`block_gpu_pte_write_reserved_chunk_4k()`。  
作用：把 GPU PTE 指向 reserved chunk，并设置 readonly 权限。

含义：tensor `data_ptr()` 不变，只是该 VA 的 GPU PTE backing 变成 reserved slot chunks；权限为 GPU READ，清除 WRITE/ATOMIC。

## 27. FusedMoE kernel 访问 expert weights

普通路径：  
[layer.py (line 6969)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6969)  
函数：`FusedMoE.forward_impl()`。  
作用：普通 MoE forward。

[layer.py (line 7051)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:7051)  
调用：`self.quant_method.apply(...)`。  
作用：执行 fused MoE kernel。

chunked 路径：  
[layer.py (line 6809)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6809)  
函数：`forward_impl_chunked()`。  
作用：分 chunk 执行大 batch MoE。

[layer.py (line 6869)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:6869)  
调用：`self.quant_method.apply(...)`。  
作用：在 chunked 路径执行 fused MoE kernel。

含义：UVM hook 在 kernel 前完成 residency/remap；kernel 仍使用原 `w13_weight` / `w2_weight` tensor pointer。

## 28. layer 完成后的 reserved cleanup 和下一层复用

FusedMoE cleanup wrapper：  
[layer.py (line 4239)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py:4239)  
函数：`_maybe_cleanup_uvm_expert_reserved_slots()`。  
作用：在 layer 后触发 slot cleanup。

executor cleanup：  
[uvm_expert_reserved_executor.py (line 1071)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/vllm/device_allocator/uvm_expert_reserved_executor.py:1071)  
函数：`cleanup_layer()`。  
作用：等待 compute event、invalidate owner PTE、标记 region 回 CPU。

含义：slot PMM chunks 保留，下一层或下一 step 可以重新绑定其他 expert，实现跨层物理空间复用。

## 29. 当前方案的关键配置

基础 managed allocation：`VLLM_USE_UVM=1`。  
KV managed allocation：`VLLM_UVM_KV_CACHE_MANAGED=1`。  
KV cache size fast path：`VLLM_UVM_KV_CACHE_SIZE_GB=<GiB>`。  
post-warmup KV target：`VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS=<blocks>`。  
后续 dynamic target control file：`VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE=<path>`。  
target 改变后物理搬运：`VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE=1`。  
物理搬运 executor：`VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=auto|cpp_batch|driver_batch`。  
placement 时机：`VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE=post_warmup`。  
全 expert 覆盖：`VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS=1`、`VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN=1`。  
hot/cold plan：`VLLM_UVM_INIT_WEIGHT_PLAN_FILE=<plan.json>`。  
hot prefetch：`VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE=1`。  
cold release：`VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE=1`、`VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=driver_release`。  
expert envelope：`VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1`、`VLLM_UVM_EXPERT_ENVELOPE_BYTES=<bytes>`。  
reserved executor：`VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async`。  
slot 参数：`VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT`、`VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES`、`VLLM_UVM_EXPERT_RESERVED_POOL_ID`。  
trace：`VLLM_UVM_POST_WARMUP_REBALANCE_TRACE_FILE=<trace.jsonl>`、`VLLM_UVM_KV_RUNTIME_TRACE_FILE=<trace.jsonl>`、`VLLM_UVM_INIT_WEIGHT_TRACE_FILE=<trace.jsonl>`。

## 30. 一句话总结

当前项目不是在 fused MoE kernel 中替换 expert tensor，也不是给每个 cold expert 永久分配 GPU 空间。它先在 warmup 后把 KV tail 和 cold experts 迁到 CPU/释放 GPU PMM，把 hot experts 放回 GPU，并显式注册/分类全部 local experts；同时预分配少量可复用 reserved PMM slots。serving 时，scheduler 只使用 KV target 内 blocks；router 当前选中的 cold_reserved expert 会从 CPU copy 到 reserved slots，并把原 expert VA 临时 remap 到 slot backing；kernel 继续使用原始 tensor pointer。

## 31. 当前实现验证

[test_uvm_post_warmup_rebalance.py (line 15)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py:15)  
验证：KV tail range 严格按 block 边界计算。

[test_uvm_post_warmup_rebalance.py (line 39)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py:39)  
验证：固定编排顺序是 KV migration -> expert placement -> slot prealloc -> runtime ready。

[test_uvm_post_warmup_rebalance.py (line 89)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py:89)  
验证：post-warmup target 会自动启用 scheduler dynamic pool。

[test_uvm_post_warmup_rebalance.py (line 110)](/home/ubuntu/nvidia-uvm-gpu/workloads/vllm/vllm/tests/v1/worker/test_uvm_post_warmup_rebalance.py:110)  
验证：reserved slot 预分配失败时 runtime gate 保持关闭。
