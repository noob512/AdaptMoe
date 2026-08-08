# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.structured_output_manager = None
        if os.environ.get(
            "VLLM_UVM_CONTROLLED_REMAP_EARLY_STRUCTURED_OUTPUT_MANAGER"
        ) == "1":
            logger.debug(
                "UVM controlled remap boundary: "
                "early structured_output_manager begin"
            )
            self.structured_output_manager = StructuredOutputManager(vllm_config)
            logger.debug(
                "UVM controlled remap boundary: "
                "early structured_output_manager end"
            )

        logger.debug("UVM controlled remap boundary: executor_class init begin")
        self.model_executor = executor_class(vllm_config)
        logger.debug("UVM controlled remap boundary: executor_class init end")
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        logger.debug("UVM controlled remap boundary: _initialize_kv_caches call begin")
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )
        logger.debug(
            "UVM controlled remap boundary: _initialize_kv_caches call end "
            "num_gpu_blocks=%s num_cpu_blocks=%s",
            num_gpu_blocks,
            num_cpu_blocks,
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        if os.environ.get("VLLM_UVM_CONTROLLED_REMAP_SKIP_ENGINE_CACHE_RPC") == "1":
            logger.warning(
                "Skipping EngineCore initialize_cache RPC for "
                "VLLM_UVM_CONTROLLED_REMAP_SKIP_ENGINE_CACHE_RPC=1"
            )
        else:
            logger.debug("UVM controlled remap boundary: initialize_cache RPC begin")
            self.collective_rpc(
                "initialize_cache", args=(num_gpu_blocks, num_cpu_blocks)
            )
            logger.debug("UVM controlled remap boundary: initialize_cache RPC end")

        if self.structured_output_manager is None:
            logger.debug("UVM controlled remap boundary: structured_output_manager begin")
            self.structured_output_manager = StructuredOutputManager(vllm_config)
            logger.debug("UVM controlled remap boundary: structured_output_manager end")
        else:
            logger.debug(
                "UVM controlled remap boundary: "
                "structured_output_manager already initialized"
            )

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        logger.debug("UVM controlled remap boundary: scheduler init begin")
        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        logger.debug("UVM controlled remap boundary: scheduler init end")
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            logger.debug("UVM controlled remap boundary: init_kv_output_aggregator begin")
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore
            logger.debug("UVM controlled remap boundary: init_kv_output_aggregator end")

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        logger.debug("UVM controlled remap boundary: mm_receiver_cache begin")
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )
        logger.debug("UVM controlled remap boundary: mm_receiver_cache end")

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            logger.debug("UVM controlled remap boundary: kv connector metadata begin")
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )
            logger.debug("UVM controlled remap boundary: kv connector metadata end")

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)
        self._uvm_pending_kv_grow_target_blocks: int | None = None
        self._uvm_pending_kv_grow_reason = ""

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            logger.debug("UVM controlled remap boundary: request_block_hasher begin")
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )
            logger.debug("UVM controlled remap boundary: request_block_hasher end")

        self._maybe_apply_pre_service_uvm_kv_dynamic_pool()

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        logger.debug("UVM controlled remap boundary: freeze_gc_heap begin")
        freeze_gc_heap()
        logger.debug("UVM controlled remap boundary: freeze_gc_heap end")
        # If enable, attach GC debugger after static variable freeze.
        logger.debug("UVM controlled remap boundary: maybe_attach_gc_debug_callback begin")
        maybe_attach_gc_debug_callback()
        logger.debug("UVM controlled remap boundary: maybe_attach_gc_debug_callback end")

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        logger.debug("UVM controlled remap boundary: initialize_from_config begin")
        self.model_executor.initialize_from_config(kv_cache_configs)
        logger.debug("UVM controlled remap boundary: initialize_from_config end")

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def _maybe_apply_pre_service_uvm_kv_dynamic_pool(self) -> None:
        stage = (
            os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                "post_warmup",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        post_warmup_target = (
            os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
        ).strip()
        initial_target = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS",
            "",
        ).strip()
        explicit_apply = (
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY", "0") == "1"
        )
        automatic_post_warmup_apply = bool(
            post_warmup_target or initial_target
        ) and stage in ("post_warmup", "after_warmup", "deferred")
        if not (explicit_apply or automatic_post_warmup_apply):
            return

        target_text = (
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS", "")
        ).strip()
        if not target_text:
            logger.warning(
                "UVM KV pre-service dynamic pool apply requested, but no target "
                "blocks env was provided"
            )
            return

        try:
            target_blocks = int(target_text)
        except ValueError:
            logger.warning(
                "Ignoring invalid UVM KV pre-service target blocks value: %r",
                target_text,
            )
            return

        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None:
            logger.warning(
                "UVM KV pre-service dynamic pool apply skipped: "
                "policy or block pool unavailable"
            )
            return

        logger.debug(
            "UVM controlled remap boundary: pre-service KV dynamic pool apply "
            "begin target_blocks=%d",
            target_blocks,
        )
        result = self._apply_uvm_kv_dynamic_pool_target(
            target_blocks,
            policy=policy,
            block_pool=block_pool,
            reason="pre_service_before_ready",
            metadata={
                "source": "engine_core_pre_ready",
                "ready_handshake_sent": False,
                "physical_migration_complete": True,
            },
        )
        logger.debug(
            "UVM controlled remap boundary: pre-service KV dynamic pool apply "
            "end result=%s",
            result,
        )

        grow_text = (
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_GROW_TARGET_BLOCKS", "")
        ).strip()
        if not grow_text:
            return

        try:
            grow_target_blocks = int(grow_text)
        except ValueError:
            logger.warning(
                "Ignoring invalid UVM KV pre-service grow target blocks value: %r",
                grow_text,
            )
            return

        if grow_target_blocks <= 0:
            return

        if grow_target_blocks == target_blocks:
            return

        logger.debug(
            "UVM controlled remap boundary: pre-service KV dynamic pool grow "
            "begin target_blocks=%d",
            grow_target_blocks,
        )
        grow_result = self._resize_uvm_kv_target_with_workers(
            grow_target_blocks,
            reason="pre_service_before_ready_grow",
            release_expert_wave=False,
        )
        logger.debug(
            "UVM controlled remap boundary: pre-service KV dynamic pool grow "
            "end result=%s",
            grow_result,
        )

    def _apply_uvm_kv_dynamic_pool_target(
        self,
        target_blocks: int,
        *,
        policy: Any,
        block_pool: Any,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return policy.apply_unified_kv_target_blocks(
            target_blocks,
            block_pool=block_pool,
            reason=reason,
            metadata=metadata,
        )

    @staticmethod
    def _uvm_env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _uvm_env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)) or default)
        except ValueError:
            return default

    def _uvm_dynamic_pool_components(self) -> tuple[Any, Any] | None:
        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None or not policy.dynamic_pool_active():
            return None
        return policy, block_pool

    def _resize_uvm_kv_target_with_workers(
        self,
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, Any]:
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, block_pool = components
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        target = min(max(int(target_blocks), 0), int(policy.total_blocks))
        if target == old_target:
            return {
                "ok": True,
                "reason": "already_at_target",
                "applied_target_blocks": old_target,
            }
        if target < old_target and self.scheduler.has_unfinished_requests():
            return {
                "ok": False,
                "reason": "shrink_requires_idle_scheduler",
                "old_target_blocks": old_target,
                "new_target_blocks": target,
            }

        expert_results: list[Any] = []
        expert_candidate_results: dict[str, list[Any]] = {}
        selected_experts_by_layer: dict[str, int] = {}
        selected_expert_class: str | None = None
        expert_release_missing_layers: list[str] = []
        if release_expert_wave and target > old_target:
            selected_expert_class = "cold"
            candidate_results = self.collective_rpc(
                "get_uvm_incremental_expert_offload_candidates",
                args=(selected_expert_class,),
            )
            expert_candidate_results[selected_expert_class] = candidate_results
            candidate_query_ok = bool(candidate_results) and all(
                isinstance(result, dict) and bool(result.get("ok"))
                for result in candidate_results
            )
            if candidate_query_ok:
                selected_experts_by_layer = (
                    self._select_uvm_incremental_experts_by_layer(
                        candidate_results
                    )
                )
            if candidate_query_ok and not selected_experts_by_layer:
                selected_expert_class = "warm"
                candidate_results = self.collective_rpc(
                    "get_uvm_incremental_expert_offload_candidates",
                    args=(selected_expert_class,),
                )
                expert_candidate_results[selected_expert_class] = candidate_results
                candidate_query_ok = bool(candidate_results) and all(
                    isinstance(result, dict) and bool(result.get("ok"))
                    for result in candidate_results
                )
                if candidate_query_ok:
                    selected_experts_by_layer = (
                        self._select_uvm_incremental_experts_by_layer(
                            candidate_results
                        )
                    )
            if candidate_query_ok and selected_experts_by_layer:
                expert_results = self.collective_rpc(
                    "run_uvm_incremental_expert_offload",
                    args=(
                        selected_expert_class == "warm",
                        selected_experts_by_layer,
                    ),
                )
                expert_release_missing_layers = (
                    self._uvm_incremental_release_missing_layers(
                        selected_experts_by_layer,
                        expert_results,
                    )
                )
            expert_execution_ok = bool(expert_results) and all(
                isinstance(result, dict) and bool(result.get("ok"))
                for result in expert_results
            )
            expert_wave_ok = candidate_query_ok and (
                not selected_experts_by_layer
                or (expert_execution_ok and not expert_release_missing_layers)
            )
            if not expert_wave_ok:
                result = {
                    "ok": False,
                    "reason": "incremental_expert_wave_failed",
                    "old_target_blocks": old_target,
                    "new_target_blocks": target,
                    "expert_candidate_results": expert_candidate_results,
                    "selected_expert_class": selected_expert_class,
                    "selected_experts_by_layer": selected_experts_by_layer,
                    "expert_release_missing_layers": (
                        expert_release_missing_layers
                    ),
                    "expert_results": expert_results,
                }
                if self._uvm_env_flag(
                    "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST",
                    default=True,
                ):
                    raise RuntimeError(
                        f"Adaptive expert offload wave failed: {result!r}"
                    )
                logger.error("Adaptive expert offload wave failed: %s", result)
                return result

        worker_results = self.collective_rpc(
            "resize_uvm_kv_cache_target",
            args=(old_target, target, reason),
        )
        physical_ok = bool(worker_results) and all(
            isinstance(result, dict) and bool(result.get("ok"))
            for result in worker_results
        )
        if not physical_ok:
            result = {
                "ok": False,
                "reason": "worker_physical_migration_failed",
                "old_target_blocks": old_target,
                "new_target_blocks": target,
                "worker_results": worker_results,
                "expert_candidate_results": expert_candidate_results,
                "selected_expert_class": selected_expert_class,
                "selected_experts_by_layer": selected_experts_by_layer,
                "expert_release_missing_layers": expert_release_missing_layers,
                "expert_results": expert_results,
            }
            if self._uvm_env_flag(
                "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST",
                default=True,
            ):
                raise RuntimeError(f"Adaptive KV resize failed: {result!r}")
            logger.error("Adaptive KV resize failed: %s", result)
            return result

        apply_result = self._apply_uvm_kv_dynamic_pool_target(
            target,
            policy=policy,
            block_pool=block_pool,
            reason=reason,
            metadata={
                "source": "engine_core_worker_coordinated_resize",
                "physical_migration_complete": True,
                "worker_results": worker_results,
            },
        )
        return {
            **apply_result,
            "worker_results": worker_results,
            "expert_candidate_results": expert_candidate_results,
            "selected_expert_class": selected_expert_class,
            "selected_experts_by_layer": selected_experts_by_layer,
            "expert_release_missing_layers": expert_release_missing_layers,
            "expert_results": expert_results,
        }

    @staticmethod
    def _select_uvm_incremental_experts_by_layer(
        candidate_results: list[Any],
    ) -> dict[str, int]:
        """Choose one deterministic global expert id per logical layer."""
        selected: dict[str, int] = {}
        for result in candidate_results:
            if not isinstance(result, dict):
                continue
            candidates = result.get("candidates")
            if not isinstance(candidates, dict):
                continue
            for layer_key, candidate in candidates.items():
                if not isinstance(candidate, dict):
                    continue
                try:
                    global_expert_id = int(candidate["global_expert_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                normalized_key = str(layer_key)
                previous = selected.get(normalized_key)
                if previous is None or global_expert_id < previous:
                    selected[normalized_key] = global_expert_id
        return selected

    @staticmethod
    def _uvm_incremental_release_missing_layers(
        selected_experts_by_layer: dict[str, int],
        worker_results: list[Any],
    ) -> list[str]:
        released_layers: set[str] = set()
        for worker_result in worker_results:
            if not isinstance(worker_result, dict):
                continue
            layer_results = worker_result.get("results")
            if not isinstance(layer_results, list):
                continue
            for layer_result in layer_results:
                if not isinstance(layer_result, dict):
                    continue
                layer_key = str(layer_result.get("layer_key") or "")
                selected_expert = selected_experts_by_layer.get(layer_key)
                if selected_expert is None:
                    continue
                released_experts = layer_result.get("released_experts")
                if not isinstance(released_experts, list):
                    continue
                try:
                    released_ids = {int(value) for value in released_experts}
                except (TypeError, ValueError):
                    continue
                if selected_expert in released_ids:
                    released_layers.add(layer_key)
        return sorted(set(selected_experts_by_layer) - released_layers)

    def _maybe_handle_uvm_kv_preemption_pressure(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if not self._uvm_env_flag(
            "VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE",
            default=False,
        ):
            return
        if not scheduler_output.preempted_req_ids:
            return
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return
        policy, _ = components
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        grow_blocks = max(
            self._uvm_env_int(
                "VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION",
                128,
            ),
            1,
        )
        preemption_count = max(len(scheduler_output.preempted_req_ids), 1)
        new_target = min(
            old_target + grow_blocks * preemption_count,
            int(policy.total_blocks),
        )
        if new_target <= old_target:
            logger.warning(
                "KV preemption pressure detected at maximum target=%d",
                old_target,
            )
            return
        batch_queue = getattr(self, "batch_queue", None)
        if batch_queue is not None and batch_queue:
            pending_target = int(
                getattr(self, "_uvm_pending_kv_grow_target_blocks", 0) or 0
            )
            self._uvm_pending_kv_grow_target_blocks = max(
                pending_target,
                new_target,
            )
            self._uvm_pending_kv_grow_reason = (
                "scheduler_preemption_pressure_after_batch_queue_drain"
            )
            logger.info(
                "Deferring UVM KV grow until %d in-flight batches drain: "
                "old_target=%d pending_target=%d preemptions=%d",
                len(batch_queue),
                old_target,
                self._uvm_pending_kv_grow_target_blocks,
                preemption_count,
            )
            return
        self._resize_uvm_kv_target_with_workers(
            new_target,
            reason="scheduler_preemption_pressure",
            release_expert_wave=self._uvm_env_flag(
                "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                default=False,
            ),
        )

    def _apply_pending_uvm_kv_growth_if_ready(self) -> dict[str, Any] | None:
        target = getattr(self, "_uvm_pending_kv_grow_target_blocks", None)
        if target is None:
            return None
        batch_queue = getattr(self, "batch_queue", None)
        if batch_queue is not None and batch_queue:
            return None

        reason = (
            getattr(self, "_uvm_pending_kv_grow_reason", "")
            or "scheduler_preemption_pressure_after_batch_queue_drain"
        )
        self._uvm_pending_kv_grow_target_blocks = None
        self._uvm_pending_kv_grow_reason = ""
        result = self._resize_uvm_kv_target_with_workers(
            int(target),
            reason=reason,
            release_expert_wave=self._uvm_env_flag(
                "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                default=False,
            ),
        )
        if not bool(result.get("ok")):
            self._uvm_pending_kv_grow_target_blocks = int(target)
            self._uvm_pending_kv_grow_reason = reason
        return result

    def apply_uvm_kv_dynamic_pool_target(
        self,
        target_blocks: int,
        reason: str = "manual_utility",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None:
            return {
                "ok": False,
                "reason": "policy_or_block_pool_unavailable",
                "requested_target_blocks": int(target_blocks),
            }

        logger.debug(
            "UVM controlled remap boundary: utility KV dynamic pool apply "
            "begin target_blocks=%d reason=%s",
            int(target_blocks),
            reason,
        )
        result = self._resize_uvm_kv_target_with_workers(
            int(target_blocks),
            reason=reason,
            release_expert_wave=bool(
                (metadata or {}).get("release_expert_wave", False)
            ),
        )
        logger.debug(
            "UVM controlled remap boundary: utility KV dynamic pool apply "
            "end result=%s",
            result,
        )
        return result

    @staticmethod
    def _plan_uvm_dynamic_expert_swaps(
        snapshot_results: list[Any],
        *,
        max_swaps_per_layer: int,
        min_count_delta: int,
    ) -> dict[str, Any]:
        aggregated: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for worker_index, worker_result in enumerate(snapshot_results):
            if not isinstance(worker_result, dict) or not bool(
                worker_result.get("ok")
            ):
                errors.append(f"worker:{worker_index}:snapshot_failed")
                continue
            layers = worker_result.get("layers")
            if not isinstance(layers, dict):
                errors.append(f"worker:{worker_index}:layers_missing")
                continue
            for layer_key, layer_value in layers.items():
                if not isinstance(layer_value, dict):
                    errors.append(f"worker:{worker_index}:{layer_key}:invalid")
                    continue
                try:
                    global_count = int(layer_value["global_num_experts"])
                    local_count = int(layer_value["local_num_experts"])
                    counts = [
                        int(value)
                        for value in layer_value["expert_token_counts"]
                    ]
                    remote = {
                        int(value) for value in layer_value["remote_experts"]
                    }
                except (KeyError, TypeError, ValueError):
                    errors.append(f"worker:{worker_index}:{layer_key}:malformed")
                    continue
                if local_count != global_count:
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:"
                        "distributed_expert_placement_unsupported"
                    )
                    continue
                if len(counts) != global_count:
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:count_width_mismatch"
                    )
                    continue
                normalized_key = str(layer_key)
                current = aggregated.get(normalized_key)
                if current is None:
                    aggregated[normalized_key] = {
                        "global_num_experts": global_count,
                        "counts": counts,
                        "remote": remote,
                    }
                    continue
                if (
                    int(current["global_num_experts"]) != global_count
                    or set(current["remote"]) != remote
                ):
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:replica_state_mismatch"
                    )
                    continue
                current["counts"] = [
                    left + right
                    for left, right in zip(current["counts"], counts)
                ]

        swaps_by_layer: dict[str, list[dict[str, object]]] = {}
        layer_summaries: dict[str, dict[str, object]] = {}
        limit = max(int(max_swaps_per_layer), 0)
        minimum_delta = max(int(min_count_delta), 0)
        for layer_key, layer in sorted(aggregated.items()):
            counts = list(layer["counts"])
            remote = set(layer["remote"])
            target_count = len(remote)
            ranked = sorted(
                range(len(counts)),
                key=lambda expert: (counts[expert], expert),
            )
            target_remote = set(ranked[:target_count])
            promote = sorted(
                remote - target_remote,
                key=lambda expert: (-counts[expert], expert),
            )
            demote = sorted(
                target_remote - remote,
                key=lambda expert: (counts[expert], expert),
            )
            selected: list[dict[str, object]] = []
            for promote_expert, demote_expert in zip(promote, demote):
                count_delta = counts[promote_expert] - counts[demote_expert]
                if count_delta < minimum_delta or len(selected) >= limit:
                    continue
                selected.append(
                    {
                        "promote_expert": promote_expert,
                        "demote_expert": demote_expert,
                        "promote_count": counts[promote_expert],
                        "demote_count": counts[demote_expert],
                        "count_delta": count_delta,
                    }
                )
            if selected:
                swaps_by_layer[layer_key] = selected
            layer_summaries[layer_key] = {
                "remote_count": target_count,
                "current_remote_experts": sorted(remote),
                "target_remote_experts": sorted(target_remote),
                "eligible_swap_count": min(len(promote), len(demote)),
                "selected_swap_count": len(selected),
                "selected_swaps": selected,
                "observed_assignments": sum(counts),
            }
        return {
            "ok": not errors,
            "errors": errors,
            "swaps_by_layer": swaps_by_layer,
            "selected_layers": len(swaps_by_layer),
            "selected_swaps": sum(len(value) for value in swaps_by_layer.values()),
            "layers": layer_summaries,
        }

    def reset_uvm_dynamic_expert_heat(self) -> dict[str, Any]:
        """Reset routing windows while preserving expert placement."""
        if self.scheduler.has_unfinished_requests():
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        if getattr(self, "batch_queue", None):
            return {"ok": False, "reason": "maintenance_requires_drained_batches"}
        snapshots = self.collective_rpc(
            "get_uvm_dynamic_expert_heat",
            args=(True,),
        )
        return {
            "ok": bool(snapshots)
            and all(
                isinstance(result, dict) and bool(result.get("ok"))
                for result in snapshots
            ),
            "reason": "heat_reset",
            "worker_results": snapshots,
        }

    def rebalance_uvm_remote_experts(
        self,
        max_swaps_per_layer: int = 1,
        min_count_delta: int = 1,
        reset_heat: bool = True,
    ) -> dict[str, Any]:
        """Recompute cold experts from recent routing and exchange placement."""
        if not self._uvm_env_flag(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        ):
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if self.scheduler.has_unfinished_requests():
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        if getattr(self, "batch_queue", None):
            return {"ok": False, "reason": "maintenance_requires_drained_batches"}

        snapshots = self.collective_rpc(
            "get_uvm_dynamic_expert_heat",
            args=(bool(reset_heat),),
        )
        plan = self._plan_uvm_dynamic_expert_swaps(
            snapshots,
            max_swaps_per_layer=max_swaps_per_layer,
            min_count_delta=min_count_delta,
        )
        if not bool(plan.get("ok")):
            return {
                "ok": False,
                "reason": "heat_snapshot_invalid",
                "snapshot_results": snapshots,
                "plan": plan,
            }
        swaps_by_layer = plan.get("swaps_by_layer") or {}
        if not swaps_by_layer:
            return {
                "ok": True,
                "reason": "placement_already_matches_heat",
                "snapshot_results": snapshots,
                "plan": plan,
                "worker_results": [],
            }
        worker_results = self.collective_rpc(
            "run_uvm_dynamic_expert_swaps",
            args=(swaps_by_layer,),
        )
        execution_ok = bool(worker_results) and all(
            isinstance(result, dict) and bool(result.get("ok"))
            for result in worker_results
        )
        return {
            "ok": execution_ok,
            "reason": (
                "expert_swaps_committed" if execution_ok else "expert_swap_failed"
            ),
            "snapshot_results": snapshots,
            "plan": plan,
            "worker_results": worker_results,
        }

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        self._maybe_handle_uvm_kv_preemption_pressure(scheduler_output)
        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with self.log_error_detail(scheduler_output):
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None
        self._apply_pending_uvm_kv_growth_if_ready()

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if (
            self.scheduler.has_requests()
            and self._uvm_pending_kv_grow_target_blocks is None
        ):
            scheduler_output = self.scheduler.schedule()
            self._maybe_handle_uvm_kv_preemption_pressure(scheduler_output)
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        self._apply_pending_uvm_kv_growth_if_ready()

        return engine_core_outputs, model_executed

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        if self._uvm_env_flag(
            "VLLM_UVM_KV_ADAPTIVE_SHRINK_ON_SLEEP",
            default=False,
        ):
            components = self._uvm_dynamic_pool_components()
            if components is not None:
                policy, _ = components
                if self.scheduler.has_unfinished_requests():
                    raise RuntimeError(
                        "Cannot shrink the UVM KV pool for sleep while requests "
                        "are still active"
                    )
                default_sleep_target = (
                    int(policy.dynamic_pool_initial_blocks)
                    if int(policy.dynamic_pool_initial_blocks) > 0
                    else int(policy._dynamic_pool_applied_target_blocks)
                )
                sleep_target = max(
                    self._uvm_env_int(
                        "VLLM_UVM_KV_ADAPTIVE_SLEEP_TARGET_BLOCKS",
                        default_sleep_target,
                    ),
                    0,
                )
                if sleep_target < int(policy._dynamic_pool_applied_target_blocks):
                    self._resize_uvm_kv_target_with_workers(
                        sleep_target,
                        reason="engine_sleep_shrink",
                        release_expert_wave=False,
                    )
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            logger.debug("UVM controlled remap boundary: engine handshake HELLO/INIT complete")
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)

            logger.debug("UVM controlled remap boundary: EngineCore super init begin")
            super().__init__(
                vllm_config, executor_class, log_stats, executor_fail_callback
            )
            logger.debug("UVM controlled remap boundary: EngineCore super init end")

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            logger.debug("UVM controlled remap boundary: EngineCore input thread start begin")
            input_thread.start()
            logger.debug("UVM controlled remap boundary: EngineCore input thread started")

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            logger.debug("UVM controlled remap boundary: EngineCore output thread start begin")
            self.output_thread.start()
            logger.debug("UVM controlled remap boundary: EngineCore output thread started")

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

        logger.debug("UVM controlled remap boundary: EngineCore READY handshake sent")

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            logger.debug("UVM controlled remap boundary: EngineCore startup HELLO begin")
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            logger.debug("UVM controlled remap boundary: EngineCore startup HELLO/INIT end")
            yield addresses

            # Send ready message.
            logger.debug("UVM controlled remap boundary: EngineCore READY send begin")
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))
            logger.debug("UVM controlled remap boundary: EngineCore READY send end")

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            logger.debug("UVM controlled remap boundary: input socket thread enter")
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                logger.debug("UVM controlled remap boundary: input socket initial send begin")
                input_socket.send(b"")
                logger.debug("UVM controlled remap boundary: input socket initial send end")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                logger.debug("UVM controlled remap boundary: coordinator READY recv begin")
                assert coord_socket.recv() == b"READY"
                logger.debug("UVM controlled remap boundary: coordinator READY recv end")
                poller.register(coord_socket, zmq.POLLIN)

            logger.debug("UVM controlled remap boundary: input socket ready_event set")
            ready_event.set()
            del ready_event
            logger.debug("UVM controlled remap boundary: input socket poll loop begin")
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            logger.debug("UVM controlled remap boundary: output socket thread enter")
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            logger.debug("UVM controlled remap boundary: output socket loop begin")
            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
