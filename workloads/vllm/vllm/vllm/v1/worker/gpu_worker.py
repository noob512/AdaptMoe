# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A GPU worker class."""

import gc
import os
import time
from contextlib import AbstractContextManager, nullcontext
from types import NoneType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.device_allocator.uvm import uvm_allocation_phase
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
)
from vllm.distributed.ec_transfer import ensure_ec_transfer_initialized
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    get_kv_transfer_group,
    has_kv_transfer_group,
)
from vllm.distributed.parallel_state import (
    get_pcp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.model_executor.warmup.kernel_warmup import kernel_warmup
from vllm.platforms import current_platform
from vllm.profiler.gpu_profiler import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.sequence import IntermediateTensors
from vllm.tasks import SupportedTask
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.engine import ReconfigureDistributedRequest, ReconfigureRankType
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.utils import is_residual_scattered_for_sp
from vllm.v1.worker.worker_base import WorkerBase

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig


class Worker(WorkerBase):
    def __init__(
            self,
            vllm_config: VllmConfig,          # vLLM 的全局配置对象
            local_rank: int,                  # 当前进程在单台机器（节点）内的 GPU 编号
            rank: int,                        # 当前进程在整个分布式集群中的全局编号
            distributed_init_method: str,     # 分布式后端的初始化方式（如 tcp:// 等）
            is_driver_worker: bool = False,   # 标记当前 worker 是否是“主控/驱动” worker
        ):
            # 1. 调用父类初始化方法，完成基础的分布式状态和环境配置
            super().__init__(
                vllm_config=vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=is_driver_worker,
            )

            # 2. 处理 Hugging Face 模型的自定义代码 (trust_remote_code)
            if self.model_config.trust_remote_code:
                # 【设计考量】采用延迟加载 (lazy import)。
                # 在多进程或分布式环境中，过早地 `import torch` 可能会导致 CUDA 上下文
                # 在错误的 GPU 设备上被提前初始化，或者引发多进程 fork() 时的死锁问题。
                from vllm.utils.import_utils import init_cached_hf_modules

                # 初始化并缓存 Hugging Face 模块，确保安全性或后续加载速度
                init_cached_hf_modules()

            # 3. 休眠状态管理
            # 用于保存当系统/Worker进入休眠(sleep)状态时需要暂存的张量(Tensor)或缓冲区。
            # 这通常用于弹性计算或挂起/恢复机制，避免显存资源一直被闲置占用。
            self._sleep_saved_buffers: dict[str, torch.Tensor] = {}

            # 4. 性能分析工具 (Profiler) 初始化
            # 允许开发者通过环境变量无侵入式地开启 PyTorch 或 CUDA 的底层性能追踪，用于排查性能瓶颈。
            self.profiler: Any | None = None
            if envs.VLLM_TORCH_PROFILER_DIR:
                # 如果配置了 PyTorch Profiler 导出目录，则实例化 TorchProfilerWrapper。
                # 为了区分分布式环境下的不同文件，将 instance_id 和 rank 拼接到文件名中。
                worker_name = f"{vllm_config.instance_id}-rank-{self.rank}"
                self.profiler = TorchProfilerWrapper(
                    worker_name=worker_name, local_rank=self.local_rank
                )
            elif envs.VLLM_TORCH_CUDA_PROFILE:
                # 如果仅需轻量级的纯 CUDA 层面 Profiler（如 Nsight Systems），则实例化该 Wrapper
                self.profiler = CudaProfilerWrapper()
            else:
                self.profiler = None

            # 5. 模型运行器架构选择
            # vLLM V2 架构引入了更高效的调度器和模型执行流程（如基于 CUDA Graph 的优化增强等）。
            self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER

    def sleep(self, level: int = 1) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            model = self.model_runner.model
            self._sleep_saved_buffers = {
                name: buffer.cpu().clone() for name, buffer in model.named_buffers()
            }

        allocator = CuMemAllocator.get_instance()
        allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
            "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
            freed_bytes / GiB_bytes,
            used_bytes / GiB_bytes,
        )

    def wake_up(self, tags: list[str] | None = None) -> None:
        from vllm.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)

        # Restore the buffers after level 2 sleep
        if len(self._sleep_saved_buffers):
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            self._sleep_saved_buffers = {}

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        if self.vllm_config.model_config.enable_sleep_mode:
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()
            if tag == "weights":
                assert allocator.get_current_usage() == 0, (
                    "Sleep mode can only be used for one instance per process."
                )
            return allocator.use_memory_pool(tag=tag)
        else:
            return nullcontext()

    def _maybe_get_kv_cache_uvm_memory_pool_context(
        self,
    ) -> AbstractContextManager:
        enabled = os.environ.get(
            "VLLM_UVM_KV_CACHE_MANAGED",
            "0",
        ).strip().lower() in ("1", "true", "yes", "on")
        if not enabled:
            return nullcontext()
        if self.vllm_config.model_config.enable_sleep_mode:
            logger.warning(
                "VLLM_UVM_KV_CACHE_MANAGED is ignored when sleep mode is enabled."
            )
            return nullcontext()

        from vllm.device_allocator.uvm import uvm_local_memory_pool

        return uvm_local_memory_pool(enable_prefetch=True)

    def _kv_cache_uvm_allocations_enabled(self) -> bool:
        return (
            os.environ.get(
                "VLLM_UVM_KV_CACHE_MANAGED",
                "0",
            ).strip().lower() in ("1", "true", "yes", "on")
            and not self.vllm_config.model_config.enable_sleep_mode
        )

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        logger.warning(
            "UVM controlled remap debug: worker initialize_cache entered "
            "num_gpu_blocks=%s num_cpu_blocks=%s",
            num_gpu_blocks,
            num_cpu_blocks,
        )
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        logger.warning("UVM controlled remap debug: worker initialize_cache returning")

    def init_device(self):
        device = self.device_config.device
        if isinstance(device, torch.device) and device.type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            if (
                self.parallel_config.data_parallel_size > 1
                and self.parallel_config.data_parallel_size_local > 0
                and self.parallel_config.distributed_executor_backend
                not in ["ray", "external_launcher"]
                and self.vllm_config.parallel_config.data_parallel_backend != "ray"
                and self.vllm_config.parallel_config.nnodes_within_dp == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = self.parallel_config.data_parallel_rank_local
                if dp_local_rank is None:
                    dp_local_rank = self.parallel_config.data_parallel_rank

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size
                    * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.cuda.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                )
                assert self.parallel_config.local_world_size <= visible_device_count, (
                    f"local_world_size ({self.parallel_config.local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
            current_platform.set_device(self.device)

            current_platform.check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            torch.cuda.empty_cache()

            # take current memory snapshot
            self.init_snapshot = MemorySnapshot()
            self.requested_memory = (
                self.init_snapshot.total_memory
                * self.cache_config.gpu_memory_utilization
            )
            if self.init_snapshot.free_memory < self.requested_memory:
                GiB = lambda b: round(b / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device "
                    f"({GiB(self.init_snapshot.free_memory)}/"
                    f"{GiB(self.init_snapshot.total_memory)} GiB) on startup "
                    f"is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{GiB(self.requested_memory)} GiB). Decrease GPU memory "
                    f"utilization or reduce GPU memory used by other processes."
                )
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Construct the model runner
        if self.use_v2_model_runner:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GPUModelRunnerV2,
            )

            # HACK(woosuk): This is a temporary fix to avoid type errors.
            self.model_runner: GPUModelRunner = GPUModelRunnerV2(  # type: ignore
                self.vllm_config, self.device
            )
        else:
            self.model_runner = GPUModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            report_usage_stats(self.vllm_config)

    # 【开发者备忘/技术债标记】
    # FIXME(youkaichao & ywang96): Use TorchDispatchMode instead of memory pool
    # to hijack tensor allocation.
    # 解释：目前 vLLM 使用“内存池 (memory pool) 上下文”来劫持/接管 PyTorch 的底层张量分配，
    # 以便将模型权重分配到连续或特定的内存区域。
    # 开发者计划未来使用 PyTorch 原生的 `TorchDispatchMode` 来替换现有的内存池机制。
    # `TorchDispatchMode` 是 PyTorch 提供的一种强大机制，允许在 Python 层面对所有的
    # 张量操作和内存分配进行拦截和重写，这种方式比当前的内存池更加干净和灵活。
    def load_model(self) -> None:
        """
        加载模型权重到 GPU/计算设备中，并完成底层内存特征的标记。
        """
        # 1. 读取弹性扩展 (Elastic Scale Up) 环境变量
        # EP 通常指 Expert Parallelism (专家并行，常用于 MoE 混合专家模型)。
        # 这个标志位用于判断当前是否处于动态扩容（Scale Up）启动阶段。
        eep_scale_up = os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1"
        
        # 2. 获取特定的内存池上下文
        # 这里传入了 tag="weights"，意味着在这个 `with` 代码块内，
        # 所有的内存分配操作（即加载模型权重产生的 Tensor）都会被引导到
        # 专门为“模型权重”预留的内存池中，这有助于减少内存碎片，或者为了 CUDAGraph 的兼容性。
        with self._maybe_get_memory_pool_context(tag="weights"):
            # 真正的模型加载动作交由底层的 model_runner 去执行
            self.model_runner.load_model(eep_scale_up=eep_scale_up)

    def update_config(self, overrides: dict[str, Any]) -> None:
        self.model_runner.update_config(overrides)

    def reload_weights(self) -> None:
        self.model_runner.reload_weights()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """Profiles the peak memory usage of the model to determine how much
        memory can be used for KV cache without OOMs.

        The engine will first conduct a profiling of the existing memory usage.
        Then, it calculates the free memory that can be used for KV cache in
        bytes.

        Tip:
            You may limit the usage of GPU memory
            by adjusting the `gpu_memory_utilization` parameter.
        """
        GiB = lambda b: b / GiB_bytes
        if kv_cache_memory_bytes := self.cache_config.kv_cache_memory_bytes:
            # still need a profile run which compiles the model for
            # max_num_batched_tokens
            self.model_runner.profile_run()
            log_weights = getattr(self.model_runner, "_log_model_weight_addresses", None)
            weight_map_enabled = getattr(
                self.model_runner, "_is_uvm_weight_map_enabled", None
            )
            address_log_enabled = getattr(
                self.model_runner, "_is_uvm_address_logging_enabled", None
            )
            try:
                should_log_weights = (
                    callable(log_weights)
                    and (
                        callable(weight_map_enabled)
                        and bool(weight_map_enabled())
                        or callable(address_log_enabled)
                        and bool(address_log_enabled())
                    )
                )
            except Exception:
                should_log_weights = False
            if should_log_weights:
                log_weights("after_determine_available_memory_profile_run")

            msg = (
                f"Initial free memory {GiB(self.init_snapshot.free_memory):.2f} "
                f"GiB, reserved {GiB(kv_cache_memory_bytes):.2f} GiB memory for "
                "KV Cache as specified by kv_cache_memory_bytes config and "
                "skipped memory profiling. This does not respect the "
                "gpu_memory_utilization config. Only use kv_cache_memory_bytes "
                "config when you want manual control of KV cache memory "
                "size. If OOM'ed, check the difference of initial free "
                "memory between the current run and the previous run "
                "where kv_cache_memory_bytes is suggested and update it "
                "correspondingly."
            )
            logger.info(msg)
            return kv_cache_memory_bytes

        try:
            from vllm.device_allocator.uvm import is_uvm_enabled
            if is_uvm_enabled():
                uvm_kv_size_gb = float(os.environ.get("VLLM_UVM_KV_CACHE_SIZE_GB", "6"))
                uvm_kv_size_bytes = int(uvm_kv_size_gb * GiB_bytes)
                logger.info("UVM enabled: setting KV cache size to %.1f GiB", uvm_kv_size_gb)
                return uvm_kv_size_bytes
        except ImportError:
            pass

        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            pass

        # Execute a forward pass with dummy inputs to profile the memory usage
        # of the model.
        with memory_profiling(
            self.init_snapshot,
            weights_memory=int(self.model_runner.model_memory_usage),
        ) as profile_result:
            self.model_runner.profile_run()

        self.non_torch_memory = profile_result.non_torch_increase
        self.peak_activation_memory = profile_result.torch_peak_increase

        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory > free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {GiB(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {GiB(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
        self.available_kv_cache_memory_bytes = (
            self.requested_memory - profile_result.non_kv_cache_memory
        )

        unrequested_memory = self.init_snapshot.free_memory - self.requested_memory
        logger.debug(
            "Initial free memory: %.2f GiB; Requested memory: %.2f (util), %.2f GiB",
            GiB(self.init_snapshot.free_memory),
            self.cache_config.gpu_memory_utilization,
            GiB(self.requested_memory),
        )
        logger.debug(
            "Free memory after profiling: %.2f GiB (total), "
            "%.2f GiB (within requested)",
            GiB(free_gpu_memory),
            GiB(free_gpu_memory - unrequested_memory),
        )
        logger.debug(profile_result)
        logger.info_once(
            "Available KV cache memory: %.2f GiB",
            GiB(self.available_kv_cache_memory_bytes),
            scope="local",
        )
        gc.collect()

        return int(self.available_kv_cache_memory_bytes)

    def resize_uvm_kv_cache_target(
        self,
        old_target_blocks: int,
        new_target_blocks: int,
        reason: str = "adaptive_runtime",
    ) -> dict[str, Any]:
        return self.model_runner.resize_uvm_kv_cache_target(
            old_target_blocks,
            new_target_blocks,
            reason,
        )

    def get_uvm_incremental_expert_offload_candidates(
        self,
        expert_class: str,
    ) -> dict[str, Any]:
        return self.model_runner.get_uvm_incremental_expert_offload_candidates(
            expert_class
        )

    def run_uvm_incremental_expert_offload(
        self,
        allow_warm: bool | None = None,
        selected_experts_by_layer: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        return self.model_runner.run_uvm_incremental_expert_offload(
            allow_warm,
            selected_experts_by_layer,
        )

    def get_uvm_dynamic_expert_heat(
        self,
        reset: bool = False,
    ) -> dict[str, Any]:
        return self.model_runner.get_uvm_dynamic_expert_heat(reset)

    def run_uvm_dynamic_expert_swaps(
        self,
        swaps_by_layer: dict[str, list[dict[str, object]]],
    ) -> dict[str, Any]:
        return self.model_runner.run_uvm_dynamic_expert_swaps(swaps_by_layer)

    def get_cuda_memory_info(self) -> dict[str, int]:
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "used_bytes": int(total_bytes - free_bytes),
            "free_mib": int(free_bytes // (1024 * 1024)),
            "total_mib": int(total_bytes // (1024 * 1024)),
            "used_mib": int((total_bytes - free_bytes) // (1024 * 1024)),
        }

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        根据指定的 kv_cache_config 配置，在 GPU 上分配 KV Cache 内存。
        这是大模型推理引擎初始化的关键步骤，因为 KV Cache 通常会占据可用显存的绝大部分。
        """
        init_start = time.monotonic()
        logger.warning(
            "UVM controlled remap debug: worker initialize_from_config entered "
            "sleep_mode=%s kv_cache_config_type=%s",
            self.vllm_config.model_config.enable_sleep_mode,
            type(kv_cache_config).__name__,
        )

        # 1. 初始化 KV Cache 连接器 (KV Cache Connector/Transfer)
        # 这里的 Connector 通常用于分布式环境下的 KV Cache 传输（例如：在不同的 GPU、节点
        # 或者分离的 Prefill/Decode 实例之间转移 KV Cache）。
        #
        # NOTE(Kuntai): 为什么必须在这里（初始化真正的 KV Cache 之前）执行？
        # 因为后续的 `initialize_kv_cache` 方法可能会注入一些与连接器无关的、特殊的 
        # KV Cache 组（例如为了实现注意力共享层 PagedAttention 的层级共享机制）。
        # 如果不提前初始化连接器，这些复杂的内部共享层可能会破坏连接器对连续内存或特定结构的假设。
        section_start = time.monotonic()
        logger.warning(
            "UVM controlled remap debug: ensure_kv_transfer_initialized begin")
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        logger.warning(
            "UVM controlled remap debug: ensure_kv_transfer_initialized end "
            "elapsed=%.3fs",
            time.monotonic() - section_start,
        )

        stage_m_stage = (
            os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                "post_warmup",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        reserved_executor_enabled = (
            os.environ.get("VLLM_UVM_WEIGHT_EXECUTOR", "").strip().lower()
            == "expert_reserved_regions_async"
        )
        reserved_after_warmup = os.environ.get(
            "VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP",
            "1" if reserved_executor_enabled else "0",
        ).strip().lower() in ("1", "true", "yes", "on")
        if reserved_after_warmup:
            os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] = "0"

        if stage_m_stage == "pre_kv":
            logger.info(
                "UVM controlled remap boundary: pre-KV Stage M placement begin"
            )
            section_start = time.monotonic()
            self.model_runner.run_uvm_init_time_weight_placement(
                run_hot_repair=False,
            )
            logger.warning(
                "UVM controlled remap debug: pre-KV Stage M placement end "
                "elapsed=%.3fs",
                time.monotonic() - section_start,
            )

        # 2. 判断是否启用了“休眠模式 (Sleep Mode)”
        # 休眠模式允许 vLLM 实例在闲置时释放物理计算资源，这是弹性计算（Elasticity）的重要特性。
        if self.vllm_config.model_config.enable_sleep_mode:
            # 如果启用了休眠模式，必须使用底层基于 CUDA Virtual Memory (cuMem) 的分配器。
            # 为什么？因为常规的 PyTorch 缓存分配器无法轻易地将物理显存交还给操作系统同时保留虚拟地址。
            # CuMem API 允许我们将虚拟显存地址与物理显存解绑（Unmap），这样在休眠时就能真正释放物理显存，
            # 等唤醒时再重新分配物理显存并映射回原虚拟地址，上层 PyTorch 代码对此无感知。
            from vllm.device_allocator.cumem import CuMemAllocator

            allocator = CuMemAllocator.get_instance()

            # 使用上下文管理器接管内存分配。
            # 在这个 `with` 块内部，vLLM 分配的 KV Cache 张量会被打上 "kv_cache" 的标签，
            # 并被路由到专门支持解绑/休眠的 CuMem 内存池中。
            section_start = time.monotonic()
            logger.warning(
                "UVM controlled remap debug: initialize_kv_cache begin "
                "allocator=cumem")
            with allocator.use_memory_pool(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
            logger.warning(
                "UVM controlled remap debug: initialize_kv_cache end "
                "allocator=cumem elapsed=%.3fs",
                time.monotonic() - section_start,
            )
        else:
            # 如果没有启用休眠模式，则走常规的 PyTorch 默认内存分配路径分配 KV Cache。
            section_start = time.monotonic()
            allocator_name = (
                "kv_cache_uvm"
                if self._kv_cache_uvm_allocations_enabled()
                else "default"
            )
            logger.warning(
                "UVM controlled remap debug: initialize_kv_cache begin "
                "allocator=%s",
                allocator_name,
            )
            with self._maybe_get_kv_cache_uvm_memory_pool_context():
                self.model_runner.initialize_kv_cache(kv_cache_config)
            logger.warning(
                "UVM controlled remap debug: initialize_kv_cache end "
                "allocator=%s elapsed=%.3fs",
                allocator_name,
                time.monotonic() - section_start,
            )

        if stage_m_stage == "pre_kv":
            section_start = time.monotonic()
            logger.info(
                "UVM controlled remap boundary: post-KV hot residency repair begin"
            )
            self.model_runner.run_uvm_init_hot_weight_repair()
            logger.warning(
                "UVM controlled remap debug: post-KV hot residency repair end "
                "elapsed=%.3fs",
                time.monotonic() - section_start,
            )

        log_weights = getattr(self.model_runner, "_log_model_weight_addresses", None)
        weight_map_enabled = getattr(
            self.model_runner, "_is_uvm_weight_map_enabled", None
        )
        address_log_enabled = getattr(
            self.model_runner, "_is_uvm_address_logging_enabled", None
        )
        try:
            should_log_weights = (
                callable(log_weights)
                and (
                    callable(weight_map_enabled)
                    and bool(weight_map_enabled())
                    or callable(address_log_enabled)
                    and bool(address_log_enabled())
                )
            )
        except Exception:
            should_log_weights = False
        force_weight_log_after_kv = os.environ.get(
            "VLLM_UVM_FORCE_WEIGHT_MAP_AFTER_KV", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        logger.warning(
            "UVM controlled remap debug: after_initialize_kv_cache "
            "weight map probe should_log=%s force=%s",
            should_log_weights,
            force_weight_log_after_kv,
        )
        if callable(log_weights) and (should_log_weights or force_weight_log_after_kv):
            if force_weight_log_after_kv:
                logged_ptrs = getattr(self.model_runner, "_logged_weight_ptrs", None)
                if isinstance(logged_ptrs, set):
                    logger.warning(
                        "UVM controlled remap debug: clearing %d logged weight "
                        "ptrs before after_initialize_kv_cache probe",
                        len(logged_ptrs),
                    )
                    logged_ptrs.clear()
            log_weights("after_initialize_kv_cache")

        # Stage M init-time placement must run on the UVM fast path too.
        # determine_available_memory() skips profile_run() when UVM is enabled.
        # By default keep the original pre-warmup placement point, but allow
        # experiments to defer CPU migration until kernel/sampler warmup has
        # finished so warmup/autotune does not see CPU-prefetched weights.
        if stage_m_stage == "pre_kv":
            logger.info(
                "UVM controlled remap boundary: Stage M placement already "
                "completed before KV allocation"
            )
        elif stage_m_stage not in ("post_warmup", "after_warmup", "deferred"):
            logger.info("UVM controlled remap boundary: Stage M init-time placement begin")
            section_start = time.monotonic()
            logger.warning(
                "UVM controlled remap debug: run_uvm_init_time_weight_placement begin")
            self.model_runner.run_uvm_init_time_weight_placement()
            logger.warning(
                "UVM controlled remap debug: run_uvm_init_time_weight_placement end "
                "elapsed=%.3fs",
                time.monotonic() - section_start,
            )
            logger.info("UVM controlled remap boundary: Stage M init-time placement end")
        else:
            logger.info(
                "UVM controlled remap boundary: Stage M init-time placement "
                "deferred until post warmup"
            )
        logger.warning(
            "UVM controlled remap debug: worker initialize_from_config returning "
            "elapsed=%.3fs",
            time.monotonic() - init_start,
        )

    @staticmethod
    def _uvm_env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _uvm_post_warmup_stage_enabled() -> bool:
        stage = (
            os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                "post_warmup",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        return stage in ("post_warmup", "after_warmup", "deferred")

    @staticmethod
    def _uvm_post_warmup_kv_target_blocks() -> int | None:
        value = (
            os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS", "")
        ).strip()
        if not value:
            return None
        try:
            return max(int(value), 0)
        except ValueError:
            logger.warning(
                "Ignoring invalid post-warmup KV target blocks value: %r",
                value,
            )
            return None

    def _write_uvm_post_warmup_kv_target(self, target_blocks: int) -> None:
        control_file = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE", ""
        ).strip()
        if not control_file:
            return
        try:
            os.makedirs(os.path.dirname(control_file) or ".", exist_ok=True)
            temporary_file = f"{control_file}.{os.getpid()}.tmp"
            with open(temporary_file, "w", encoding="utf-8") as handle:
                handle.write(f"target_blocks={int(target_blocks)}\n")
            os.replace(temporary_file, control_file)
            logger.info(
                "UVM post-warmup KV target published target_blocks=%d control_file=%s",
                target_blocks,
                control_file,
            )
        except Exception:
            logger.exception(
                "Failed to publish post-warmup KV target target_blocks=%d "
                "control_file=%s",
                target_blocks,
                control_file,
            )
            if self._uvm_env_flag(
                "VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST",
                default=True,
            ):
                raise

    def _run_uvm_post_warmup_rebalance(self) -> None:
        if not self._uvm_post_warmup_stage_enabled():
            return
        if getattr(self, "_uvm_post_warmup_rebalance_done", False):
            return

        target_blocks = self._uvm_post_warmup_kv_target_blocks()
        reserved_executor_enabled = (
            os.environ.get("VLLM_UVM_WEIGHT_EXECUTOR", "").strip().lower()
            == "expert_reserved_regions_async"
        )
        remote_read_mode = (
            os.environ.get(
                "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE",
                "restore",
            )
            .strip()
            .lower()
            == "remote_read"
        )
        placement_requested = any(
            self._uvm_env_flag(name, default=False)
            for name in (
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
                "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE",
                "VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE",
                "VLLM_UVM_EXPERT_RESERVED_PREALLOC_AFTER_STAGE_M",
            )
        )
        if target_blocks is None and not (
            placement_requested or reserved_executor_enabled
        ):
            return

        start = time.monotonic()
        logger.info("UVM controlled remap boundary: post-warmup rebalance begin")
        if target_blocks is not None:
            self._write_uvm_post_warmup_kv_target(target_blocks)
            managed_kv = self._kv_cache_uvm_allocations_enabled() or self._uvm_env_flag(
                "VLLM_USE_UVM", default=False
            )
            if managed_kv:
                result = self.model_runner.rebalance_uvm_kv_cache_after_warmup(
                    target_blocks
                )
                if not bool(result.get("offload_complete")) and self._uvm_env_flag(
                    "VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST",
                    default=True,
                ):
                    raise RuntimeError(
                        f"post-warmup KV physical rebalance failed: result={result!r}"
                    )
            else:
                logger.warning(
                    "Post-warmup KV target was updated logically, but physical "
                    "KV offload was skipped because KV allocations are not UVM"
                )

        logger.info("UVM controlled remap boundary: post-warmup expert placement begin")
        self.model_runner.run_uvm_init_time_weight_placement(prealloc_reserved=False)
        logger.info("UVM controlled remap boundary: post-warmup expert placement end")

        if remote_read_mode:
            prealloc_result = {
                "requested": False,
                "ok": True,
                "reason": "remote_read_does_not_use_reserved_slots",
            }
            reserved_after_warmup = False
            os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] = "0"
        else:
            prealloc_reserved = (
                self.model_runner._maybe_prealloc_uvm_expert_reserved_slots_after_stage_m
            )
            prealloc_result = prealloc_reserved()
            reserved_after_warmup = self._uvm_env_flag(
                "VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP",
                default=reserved_executor_enabled,
            )
        if reserved_after_warmup:
            prealloc_ready = (
                not reserved_executor_enabled
                or bool(prealloc_result.get("requested"))
                and bool(prealloc_result.get("ok"))
            )
            if not prealloc_ready:
                os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] = "0"
                message = (
                    "reserved expert runtime remains disabled because post-warmup "
                    f"slot preallocation was not ready: {prealloc_result!r}"
                )
                if self._uvm_env_flag(
                    "VLLM_UVM_EXPERT_RESERVED_FAIL_FAST_ON_PREALLOC_FAILURE",
                    default=True,
                ):
                    raise RuntimeError(message)
                logger.error(message)
            else:
                os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] = "1"
                logger.info(
                    "UVM controlled remap boundary: expert reserved runtime "
                    "enabled after post-warmup preallocation"
                )

        self._uvm_post_warmup_rebalance_done = True
        logger.info(
            "UVM controlled remap boundary: post-warmup rebalance end "
            "elapsed=%.3fs",
            time.monotonic() - start,
        )

    def compile_or_warm_up_model(self) -> None:
        driver_mode = os.environ.get(
            "VLLM_UVM_COLD_SLOT_POOL_DRIVER_MODE",
            "",
        ).strip().lower().replace("-", "_")
        skip_uvm_controlled_warmup = (
            driver_mode == "slot_va_block_remap"
            and os.environ.get(
                "VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP",
                "0",
            ).strip().lower() in ("1", "true", "yes", "on")
        )
        if skip_uvm_controlled_warmup:
            logger.warning(
                "Skipping post-init compile/kernel/sampler warmup for "
                "VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1"
            )
            self._run_uvm_post_warmup_rebalance()
            self.model_runner.run_forward_token_benchmark()
            set_random_seed(self.model_config.seed)
            return

        logger.info("UVM controlled remap boundary: compile_or_warm_up_model begin")
        # warm up sizes that are not in cudagraph capture sizes,
        # but users still want to compile for better performance,
        # e.g. for the max-num-batched token size in chunked prefill.
        compile_sizes = self.vllm_config.compilation_config.compile_sizes
        warmup_sizes = compile_sizes.copy() if compile_sizes is not None else []
        if not self.model_config.enforce_eager:
            capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            if capture_sizes is not None:
                warmup_sizes = [x for x in warmup_sizes if x not in capture_sizes]
        # We skip EPLB here since we don't want to record dummy metrics
        for size in sorted(warmup_sizes, reverse=True):
            logger.info("Compile and warming up model for size %d", size)
            with uvm_allocation_phase(f"compile_warmup:size={size}"):
                self.model_runner._dummy_run(
                    size,
                    skip_eplb=True,
                    remove_lora=False,
                )
        self.model_runner.maybe_remove_all_loras(self.model_runner.lora_config)

        # Warmup and tune the kernels used during model execution before
        # cuda graph capture.
        logger.info("UVM controlled remap boundary: kernel_warmup begin")
        with uvm_allocation_phase("kernel_warmup"):
            kernel_warmup(self)
        logger.info("UVM controlled remap boundary: kernel_warmup end")

        cuda_graph_memory_bytes = 0
        if not self.model_config.enforce_eager:
            logger.info("UVM controlled remap boundary: capture_model begin")
            with uvm_allocation_phase("capture_model"):
                cuda_graph_memory_bytes = self.model_runner.capture_model()
            logger.info("UVM controlled remap boundary: capture_model end")

        if self.cache_config.kv_cache_memory_bytes is None and hasattr(
            self, "peak_activation_memory"
        ):
            # Suggests optimal kv cache memory size if we rely on
            # memory_profiling to guess the kv cache memory size which
            # provides peak_activation_memory and a few other memory
            # consumption. `memory_profiling` does not consider
            # CUDAGraph memory size and may not utilize all gpu memory.
            # Users may want fine-grained control to specify kv cache
            # memory size.
            GiB = lambda b: round(b / GiB_bytes, 2)

            # empirically observed that the memory profiling may
            # slightly underestimate the memory consumption.
            # So leave a small buffer (=150MiB) to avoid OOM.
            redundancy_buffer_memory = 150 * (1 << 20)
            non_kv_cache_memory = (
                self.model_runner.model_memory_usage
                + self.peak_activation_memory
                + self.non_torch_memory
                + cuda_graph_memory_bytes
            )
            kv_cache_memory_bytes_to_gpu_limit = (
                self.init_snapshot.free_memory
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )
            kv_cache_memory_bytes_to_requested_limit = (
                int(self.requested_memory)
                - non_kv_cache_memory
                - redundancy_buffer_memory
            )

            msg = (
                f"Free memory on device "
                f"({GiB(self.init_snapshot.free_memory)}/"
                f"{GiB(self.init_snapshot.total_memory)} GiB) on startup. "
                f"Desired GPU memory utilization is "
                f"({self.cache_config.gpu_memory_utilization}, "
                f"{GiB(self.requested_memory)} GiB). "
                f"Actual usage is {GiB(self.model_runner.model_memory_usage)} "
                f"GiB for weight, {GiB(self.peak_activation_memory)} GiB "
                f"for peak activation, {GiB(self.non_torch_memory)} GiB "
                f"for non-torch memory, and {GiB(cuda_graph_memory_bytes)} "
                f"GiB for CUDAGraph memory. Replace gpu_memory_utilization "
                f"config with `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_requested_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_requested_limit)} GiB) to fit "
                f"into requested memory, or `--kv-cache-memory="
                f"{kv_cache_memory_bytes_to_gpu_limit}` "
                f"({GiB(kv_cache_memory_bytes_to_gpu_limit)} GiB) to fully "
                f"utilize gpu memory. Current kv cache memory in use is "
                f"{GiB(self.available_kv_cache_memory_bytes)} GiB."
            )

            logger.debug(msg)

        # Warm up sampler and preallocate memory buffer for logits and other
        # sampling related tensors of max possible shape to avoid memory
        # fragmentation issue.
        # NOTE: This is called after `capture_model` on purpose to prevent
        # memory buffers from being cleared by `torch.cuda.empty_cache`.
        if get_pp_group().is_last_rank:
            max_num_reqs = min(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
            )

            # We skip EPLB here since we don't want to record dummy metrics.
            # Keep this sampler warmup out of enabled:* serving phases.
            logger.info("UVM controlled remap boundary: sampler_warmup begin")
            with uvm_allocation_phase("sampler_warmup"):
                hidden_states, last_hidden_states = self.model_runner._dummy_run(
                    num_tokens=max_num_reqs,
                    skip_eplb=True,
                )
                if self.model_runner.is_pooling_model:
                    self.model_runner._dummy_pooler_run(hidden_states)
                else:
                    self.model_runner._dummy_sampler_run(
                        hidden_states=last_hidden_states
                    )
            logger.info("UVM controlled remap boundary: sampler_warmup end")

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        self._run_uvm_post_warmup_rebalance()
        self.model_runner.run_forward_token_benchmark()

        set_random_seed(self.model_config.seed)
        logger.info("UVM controlled remap boundary: compile_or_warm_up_model end")

    def reset_mm_cache(self) -> None:
        self.model_runner.reset_mm_cache()

    def get_model(self) -> nn.Module:
        return self.model_runner.get_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def annotate_profile(self, scheduler_output):
        # add trace annotation so that we can easily distinguish
        # new/cached request numbers in each iteration
        if not self.profiler:
            return nullcontext()

        self.profiler.step()

        num_new = len(scheduler_output.scheduled_new_reqs)
        num_cached = len(scheduler_output.scheduled_cached_reqs.req_ids)

        return self.profiler.annotate_context_manager(
            f"execute_new_{num_new}_cached_{num_cached}"
        )

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    # @torch.inference_mode() 是 PyTorch 推理时的标配。
    # 它会全局关闭 Autograd 引擎，不记录反向传播的梯度图，
    # 从而大幅减少显存消耗并提升运行速度。
    @torch.inference_mode()
    def execute_model(
        self, scheduler_output: "SchedulerOutput"
    ) -> ModelRunnerOutput | None:
        """
        执行单步模型推理计算。
        在分布式环境（尤其是流水线并行）中，该方法会处理张量的接收、计算和发送。
        """

        # =====================================================================
        # 1. 分布式并行通信：前置数据接收 (Pipeline Parallelism - Recv)
        # =====================================================================
        intermediate_tensors = None
        # 判断调度器是否实际分配了需要计算的 Token (可能当前步该 Worker 在空转)
        forward_pass = scheduler_output.total_num_scheduled_tokens > 0
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        
        # 获取输入 Token 的数量，用于后续判断是否需要特定的通信优化
        num_input_tokens = self.model_runner._get_num_input_tokens(num_scheduled_tokens)
        
        # 序列并行 (Sequence Parallelism, SP) 优化标志位。
        # 如果启用了 SP，残差 (residual) 可能被打散在不同的 GPU 上，不需要额外做 all_gather。
        all_gather_tensors = {
            "residual": not is_residual_scattered_for_sp(
                self.vllm_config, num_input_tokens
            )
        }
        
        # 如果当前有前向计算任务，并且【当前进程不是流水线并行的第一阶段(First Rank)】
        # 说明这是一个中间层或输出层，它需要从上一台机器/GPU接收前一层的隐藏状态(Hidden States)。
        if forward_pass and not get_pp_group().is_first_rank:
            # 通过 NCCL 阻塞接收上一个 Pipeline 阶段传来的 Tensor 字典
            tensor_dict = get_pp_group().recv_tensor_dict(
                all_gather_group=get_tp_group(),
                all_gather_tensors=all_gather_tensors,
            )
            assert tensor_dict is not None
            # 将接收到的裸数据包装成中间态张量对象
            intermediate_tensors = IntermediateTensors(tensor_dict)

        # =====================================================================
        # 3. 核心计算：执行模型前向传播
        # =====================================================================
        # annotate_profile 是一个上下文管理器，用于在 Nsight 或 PyTorch Profiler 中
        # 打上时间轴标签（标记这段时间的计算属于哪个 Scheduler Step），方便性能调优。
        with self.annotate_profile(scheduler_output):
            # 将调度指令和（可能来自上一阶段的）中间张量送入底层的 model_runner 
            # 进行实际的 CUDA 算子调用和矩阵乘法。
            output = self.model_runner.execute_model(
                scheduler_output, intermediate_tensors
            )
            
            # 如果输出是 ModelRunnerOutput（包含了最终的 Logits/采样 Token），
            # 或者干脆是 None（当前步完全没做任何事），则直接返回给上层引擎。
            # 这通常发生在使用单卡、纯张量并行(纯TP) 或 流水线并行的最后一层(Last Rank)时。
            if isinstance(output, (ModelRunnerOutput, NoneType)):
                return output

        # =====================================================================
        # 4. 分布式并行通信：后置数据发送 (Pipeline Parallelism - Send)
        # =====================================================================
        # 如果代码运行到这里，说明 output 是 IntermediateTensors（中间隐藏状态），
        # 这意味着当前进程是流水线的一个【中间阶段】，它尚未生成最终的文本。
        assert isinstance(output, IntermediateTensors)
        parallel_config = self.vllm_config.parallel_config
        
        # 再次确保逻辑正确性：能发中间状态的，绝对不可能是流水线的最后一步。
        assert (
            parallel_config.distributed_executor_backend != "external_launcher"
            and not get_pp_group().is_last_rank
        )

        # 通过 NCCL 将当前层计算完毕的隐藏状态发送给流水线的【下一阶段】。
        get_pp_group().send_tensor_dict(
            output.tensors,
            all_gather_group=get_tp_group(),
            all_gather_tensors=all_gather_tensors,
        )

        # 中间流水线阶段不返回具体的采样结果给 vLLM 的主引擎，返回 None。
        return None

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.model_runner.take_draft_token_ids()

    def profile(self, is_start: bool = True):
        if self.profiler is None:
            raise RuntimeError("Profiling is not enabled.")
        if is_start:
            self.profiler.start()
        else:
            self.profiler.stop()

    def execute_dummy_batch(self) -> None:
        if self.use_v2_model_runner:
            self.model_runner.execute_model(
                SchedulerOutput.make_empty(), dummy_run=True
            )
        else:
            self.model_runner._dummy_run(1, uniform_decode=True)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_runner.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_runner.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_runner.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_runner.pin_lora(lora_id)

    def check_health(self) -> None:
        # worker will always be healthy as long as it's running.
        return

    def _eplb_before_scale_down(self, old_ep_size: int, new_ep_size: int) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info(
                "[Elastic EP] Starting expert resharding before scaling down..."
            )
        rank_mapping = {
            old_ep_rank: old_ep_rank if old_ep_rank < new_ep_size else -1
            for old_ep_rank in range(old_ep_size)
        }
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_loads=None,
            rank_mapping=rank_mapping,
        )
        torch.cuda.synchronize()
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _eplb_after_scale_up(
        self,
        old_ep_size: int,
        new_ep_size: int,
        global_expert_loads: list[torch.Tensor] | None,
    ) -> None:
        from vllm.distributed.parallel_state import get_ep_group

        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Starting expert resharding after scaling up...")
        rank_mapping = {old_ep_rank: old_ep_rank for old_ep_rank in range(old_ep_size)}
        assert self.model_runner.eplb_state is not None
        self.model_runner.eplb_state.rearrange(
            execute_shuffle=True,
            global_expert_loads=global_expert_loads,
            rank_mapping=rank_mapping,
        )
        if get_ep_group().rank == 0:
            logger.info("[Elastic EP] Expert resharding completed!")

    def _reconfigure_parallel_config(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        """
        Update parallel config with provided reconfig_request
        """
        parallel_config = self.vllm_config.parallel_config
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if (
            reconfig_request.new_data_parallel_rank
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        if (
            reconfig_request.new_data_parallel_rank_local
            != ReconfigureRankType.KEEP_CURRENT_RANK
        ):
            parallel_config.data_parallel_rank_local = (
                reconfig_request.new_data_parallel_rank_local
            )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )

    def _reconfigure_moe(
        self, old_ep_size: int, new_ep_size: int
    ) -> list[torch.Tensor] | None:
        """
        Reconfigure MoE modules with provided reconfig_request

        Return the global expert load if new_ep_size > old_ep_size,
        otherwise None
        """
        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_ep_group,
            prepare_communication_buffer_for_model,
        )
        from vllm.model_executor.layers.fused_moe.layer import (
            FusedMoE,
            FusedMoEParallelConfig,
        )

        parallel_config = self.vllm_config.parallel_config

        def get_moe_modules(model: torch.nn.Module) -> list[FusedMoE]:
            return [
                module
                for module in model.modules()
                if (
                    module.__class__.__name__ == "FusedMoE"
                    or module.__class__.__name__ == "SharedFusedMoE"
                )
            ]

        def update_moe_modules(moe_modules: list[FusedMoE], num_local_experts: int):
            assert all(
                module.moe_config.num_local_experts == num_local_experts
                for module in moe_modules
            ), "All MoE modules must have the same number of experts"
            for module in moe_modules:
                module.moe_config.num_experts = num_local_experts * new_ep_size
                module.global_num_experts = module.moe_config.num_experts
                module.moe_parallel_config = FusedMoEParallelConfig.make(
                    tp_size_=get_tp_group().world_size,
                    pcp_size_=get_pcp_group().world_size,
                    dp_size_=get_dp_group().world_size,
                    vllm_parallel_config=parallel_config,
                )
                module.moe_config.moe_parallel_config = module.moe_parallel_config
            return moe_modules

        model_moe_modules = get_moe_modules(self.model_runner.model)
        num_local_experts = model_moe_modules[0].moe_config.num_local_experts

        update_moe_modules(model_moe_modules, num_local_experts)
        drafter_model = None
        if hasattr(self.model_runner, "drafter") and hasattr(
            self.model_runner.drafter, "model"
        ):
            drafter_model = self.model_runner.drafter.model
        if drafter_model is not None and is_mixture_of_experts(drafter_model):
            drafter_moe_modules = get_moe_modules(drafter_model)
            # Check if drafter and model have matching configs
            assert (
                drafter_moe_modules[0].moe_config.num_local_experts == num_local_experts
            ), "Drafter and model configs should be the same"
            update_moe_modules(drafter_moe_modules, num_local_experts)

        if new_ep_size < old_ep_size:
            num_local_physical_experts = num_local_experts
            assert self.model_runner.eplb_state is not None
            new_physical_experts = (
                self.model_runner.eplb_state.physical_to_logical_map.shape[1]  # type: ignore[attr-defined]
            )
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts
                - self.model_runner.eplb_state.logical_replica_count.shape[1]  # type: ignore[attr-defined]
            )
            global_expert_loads = None
        else:
            num_local_physical_experts_tensor = torch.tensor(
                [num_local_experts], dtype=torch.int32, device="cpu"
            )
            torch.distributed.broadcast(
                num_local_physical_experts_tensor,
                group=get_ep_group().cpu_group,
                group_src=0,
            )
            num_local_physical_experts = int(num_local_physical_experts_tensor.item())
            new_physical_experts = num_local_physical_experts * new_ep_size
            assert self.model_runner.eplb_state is not None
            global_expert_loads_any = self.model_runner.eplb_state.rearrange(
                execute_shuffle=False
            )
            global_expert_loads = cast(list[torch.Tensor], global_expert_loads_any)
            parallel_config.eplb_config.num_redundant_experts = (
                new_physical_experts - global_expert_loads[0].shape[1]
            )
        prepare_communication_buffer_for_model(self.model_runner.model)
        if drafter_model is not None:
            prepare_communication_buffer_for_model(drafter_model)
        self.model_runner.model.update_physical_experts_metadata(
            num_physical_experts=new_physical_experts,
            num_local_physical_experts=num_local_physical_experts,
        )
        return global_expert_loads

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            cleanup_dist_env_and_memory,
            get_ep_group,
        )

        old_ep_size = get_ep_group().world_size
        old_ep_rank = get_ep_group().rank
        new_ep_size = (
            reconfig_request.new_data_parallel_size
            * get_tp_group().world_size
            * get_pp_group().world_size
        )
        if new_ep_size < old_ep_size:
            self._eplb_before_scale_down(old_ep_size, new_ep_size)

        cleanup_dist_env_and_memory()

        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            assert old_ep_rank >= new_ep_size
            # shutdown
            return

        self._reconfigure_parallel_config(reconfig_request)

        with set_current_vllm_config(self.vllm_config):
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
            )

        global_expert_loads = self._reconfigure_moe(old_ep_size, new_ep_size)

        if new_ep_size > old_ep_size:
            assert global_expert_loads is not None
            self._eplb_after_scale_up(old_ep_size, new_ep_size, global_expert_loads)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        from vllm.model_executor.model_loader import ShardedStateLoader

        ShardedStateLoader.save_model(
            self.model_runner.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        self.model_runner.save_tensorized_model(
            tensorizer_config=tensorizer_config,
        )

    def shutdown(self) -> None:
        if runner := getattr(self, "model_runner", None):
            runner.ensure_kv_transfer_shutdown()

        if self.profiler is not None:
            self.profiler.shutdown()


def init_worker_distributed_environment(
    vllm_config: VllmConfig,
    rank: int,
    distributed_init_method: str | None = None,
    local_rank: int = -1,
    backend: str = "nccl",
) -> None:
    """Initialize the distributed environment."""
    parallel_config = vllm_config.parallel_config
    from vllm.model_executor.layers.batch_invariant import init_batch_invariance

    init_batch_invariance()
    set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)

    init_method = distributed_init_method or "env://"
    init_distributed_environment(
        parallel_config.world_size, rank, init_method, local_rank, backend
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size,
        parallel_config.pipeline_parallel_size,
        parallel_config.prefill_context_parallel_size,
        parallel_config.decode_context_parallel_size,
    )

    # Init ec connector here before KV caches caches init
    # NOTE: We do not init KV caches for Encoder-only instance in EPD disagg mode
    ensure_ec_transfer_initialized(vllm_config)
