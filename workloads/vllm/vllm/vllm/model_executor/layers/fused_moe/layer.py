# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Literal, cast, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.device_allocator.uvm import (
    batch_advise_ranges_preferred_location,
    batch_prefetch_ranges_to_device,
    batch_prefetch_ranges_to_device_with_executor,
    batch_prepare_ranges_for_native_remote_access_with_executor,
    driver_batch_query_ranges_residency_bytes,
    driver_expert_offload_release_regions,
    driver_prefetch_cpu_release_remote_512k_regions,
    probe_driver_capable_range,
    uvm_enabled_allocation_phase,
    uvm_local_memory_pool,
)
from vllm.device_allocator.uvm_expert_reserved_executor import (
    ReservedRegionItem,
    get_expert_reserved_regions_executor,
)
from vllm.device_allocator.uvm_pool_residency import (
    ensure_uvm_residency_driver_region_registered,
    get_uvm_residency_region_snapshot,
    mark_uvm_residency_state,
    record_uvm_residency_access,
    register_uvm_residency_region,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,
)
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    is_flashinfer_supporting_global_sf,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike():
    from .fused_moe import eplb_map_to_physical_and_record, fused_experts
else:
    fused_experts = None  # type: ignore
    FusedMoEPermuteExpertsUnpermute = object  # type: ignore
    FusedMoEPrepareAndFinalize = object  # type: ignore

    def _eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record
from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
    rocm_aiter_grouped_topk,
)

if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

logger = init_logger(__name__)

_UVM_PRE_EXPERT_TIMING_LOCK = threading.Lock()
_UVM_PRE_EXPERT_TIMING_STARTS: dict[str, dict[str, object]] = {}
_UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES = (
    "driver_release",
    "driver_release_cpu",
    "offload_release",
    "release_cpu",
)
_UVM_EXPERT_512K_ENVELOPE_BYTES = 512 * 1024
_UVM_EXPERT_2MB_ENVELOPE_BYTES = 2 * 1024 * 1024


def _uvm_pre_expert_timing_enabled() -> bool:
    return os.environ.get("VLLM_UVM_PRE_EXPERT_TIMING_ENABLE", "0").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _uvm_pre_expert_timing_file() -> str:
    return os.environ.get("VLLM_UVM_PRE_EXPERT_TIMING_FILE", "")


def record_uvm_pre_expert_timing_layer_start(
    *,
    layer_name: str,
    layer_id: int | None,
    num_tokens: int,
    positions: torch.Tensor,
) -> None:
    if not _uvm_pre_expert_timing_enabled() or not torch.cuda.is_available():
        return

    try:
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream())
        position_min = None
        position_max = None
        if positions.numel() > 0:
            position_min = int(positions.detach().min().item())
            position_max = int(positions.detach().max().item())

        with _UVM_PRE_EXPERT_TIMING_LOCK:
            _UVM_PRE_EXPERT_TIMING_STARTS[layer_name] = {
                "event": start_event,
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": layer_name,
                "layer_id": layer_id,
                "num_tokens": int(num_tokens),
                "position_min": position_min,
                "position_max": position_max,
            }
    except Exception:
        logger.debug("Failed to record UVM pre-expert timing start", exc_info=True)


def _record_uvm_pre_expert_timing_ids_ready(
    *,
    layer_name: str,
    topk_ids: torch.Tensor,
) -> None:
    trace_file = _uvm_pre_expert_timing_file()
    if (
        not _uvm_pre_expert_timing_enabled()
        or not trace_file
        or not torch.cuda.is_available()
    ):
        return

    try:
        with _UVM_PRE_EXPERT_TIMING_LOCK:
            start_record = _UVM_PRE_EXPERT_TIMING_STARTS.pop(layer_name, None)
        if start_record is None:
            return

        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record(torch.cuda.current_stream())
        end_event.synchronize()
        elapsed_ms = float(start_record["event"].elapsed_time(end_event))  # type: ignore[union-attr]

        ids = topk_ids.detach()
        valid_ids = ids[ids >= 0].to(dtype=torch.long)
        active_experts = 0
        if valid_ids.numel() > 0:
            active_experts = int(torch.unique(valid_ids).numel())

        record = {
            "event": "uvm_pre_expert_timing",
            "timestamp": time.time(),
            "pid": int(start_record["pid"]),
            "layer_name": start_record["layer_name"],
            "layer_id": start_record["layer_id"],
            "num_tokens": start_record["num_tokens"],
            "position_min": start_record["position_min"],
            "position_max": start_record["position_max"],
            "kv_len_approx": (
                int(start_record["position_max"]) + 1
                if start_record["position_max"] is not None
                else None
            ),
            "topk_shape": [int(value) for value in topk_ids.shape],
            "active_experts": active_experts,
            "elapsed_ms": elapsed_ms,
        }
        os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
        with open(trace_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        logger.debug("Failed to record UVM pre-expert timing end", exc_info=True)


_UVM_WARM_PREFETCH_REGISTRY_LOCK = threading.Lock()
_UVM_WARM_PREFETCH_REGISTRY: weakref.WeakValueDictionary[
    tuple[int, str, int], object
] = weakref.WeakValueDictionary()
_UVM_WARM_PREFETCH_EXECUTOR_LOCK = threading.Lock()
_UVM_WARM_PREFETCH_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_UVM_WARM_PREFETCH_TIMING_STREAM_LOCK = threading.Lock()
_UVM_WARM_PREFETCH_TIMING_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _uvm_warm_prefetch_executor() -> ThreadPoolExecutor:
    pid = os.getpid()
    with _UVM_WARM_PREFETCH_EXECUTOR_LOCK:
        executor = _UVM_WARM_PREFETCH_EXECUTORS.get(pid)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="uvm-warm-prefetch",
            )
            _UVM_WARM_PREFETCH_EXECUTORS[pid] = executor
        return executor


def _uvm_warm_prefetch_timing_stream(device: int) -> torch.cuda.Stream:
    key = (os.getpid(), int(device))
    with _UVM_WARM_PREFETCH_TIMING_STREAM_LOCK:
        stream = _UVM_WARM_PREFETCH_TIMING_STREAMS.get(key)
        if stream is None:
            stream = torch.cuda.Stream(device=device)
            _UVM_WARM_PREFETCH_TIMING_STREAMS[key] = stream
        return stream


def _uvm_layer_sequence_key(layer_name: str) -> str:
    parts = str(layer_name).split(".")
    for index, part in enumerate(parts[:-1]):
        if part in ("layers", "layer"):
            parts[index + 1] = "*"
            break
    return ".".join(parts)


def _env_flag_enabled(env_name: str, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    return env_value.lower() in ("1", "true", "yes", "on")


def _env_int(env_name: str, default: int) -> int:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    try:
        return int(env_value)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%s; using %s",
            env_name,
            env_value,
            default,
        )
        return default


@triton.jit
def _uvm_topk_mask_hit_kernel(
    topk_ids_ptr,
    mask_ptr,
    hit_flag_ptr,
    topk_numel,
    width: tl.constexpr,
    ASSUME_VALID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    active = offsets < topk_numel
    expert_ids = tl.load(topk_ids_ptr + offsets, mask=active, other=0)

    if ASSUME_VALID:
        mask_offsets = expert_ids.to(tl.int64)
        mask_values = tl.load(mask_ptr + mask_offsets, mask=active, other=0)
    else:
        valid = active & (expert_ids >= 0) & (expert_ids < width)
        safe_ids = tl.where(valid, expert_ids, 0).to(tl.int64)
        mask_values = tl.load(mask_ptr + safe_ids, mask=valid, other=0)

    hit = tl.sum(mask_values.to(tl.int32), axis=0) > 0
    tl.store(hit_flag_ptr, hit.to(tl.int32))


@triton.jit
def _uvm_topk_mask_simple_kernel(hit_flag_ptr):
    tl.store(hit_flag_ptr, tl.full((), 0, tl.int32))


def _uvm_next_power_of_2(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _align_down(value: int, alignment: int) -> int:
    alignment = max(int(alignment), 1)
    return int(value) - (int(value) % alignment)


def _align_up(value: int, alignment: int) -> int:
    alignment = max(int(alignment), 1)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _parse_int_set_env(env_name: str) -> set[int]:
    env_value = os.environ.get(env_name, "")
    values: set[int] = set()
    for item in env_value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            logger.warning("Ignoring invalid integer in %s=%s", env_name, item)
    return values


def _parse_plan_expert_keys(path: str, plan_key: str) -> set[tuple[int, int]]:
    if not path:
        return set()
    try:
        with open(path, encoding="utf-8") as handle:
            plan = json.load(handle)
    except OSError:
        logger.warning("Failed to read Stage I plan file: %s", path, exc_info=True)
        return set()
    except json.JSONDecodeError:
        logger.warning("Failed to parse Stage I plan file: %s", path, exc_info=True)
        return set()

    records = plan.get(plan_key)
    if not isinstance(records, list):
        return set()

    keys: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            layer_id = int(record["layer_id"])
            expert_id = int(record["expert_id"])
        except (KeyError, TypeError, ValueError):
            continue
        keys.add((layer_id, expert_id))
    return keys


def _load_uvm_weight_plan(path: str) -> dict[str, object]:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            plan = json.load(handle)
    except OSError:
        logger.warning("Failed to read UVM weight plan file: %s", path, exc_info=True)
        return {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse UVM weight plan file: %s", path, exc_info=True)
        return {}
    return plan if isinstance(plan, dict) else {}


def _uvm_weight_plan_policy(path: str) -> str:
    plan = _load_uvm_weight_plan(path)
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    policy = metadata.get("policy")
    return str(policy) if policy is not None else ""


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)
            and num_redundant_experts == 0
            and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_deepep_ll_kernels
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    quant_config: QuantizationConfig | None,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)
        if (
            current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        elif (
            current_platform.is_rocm()
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
        routing_method_type: int | None = None,
    ):
        super().__init__()

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.info_once("Disabling MoE shared_experts cuda stream")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.info_once("Enabled separate cuda stream for MoE shared_experts")

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.logical_num_experts = num_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size,
            moe_in_dtype,
            quant_config,
            self.moe_parallel_config,
            is_lora_enabled=self.vllm_config.lora_config is not None,
        )

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: torch.Tensor | None = None
        self.logical_to_physical_map: torch.Tensor | None = None
        self.logical_replica_count: torch.Tensor | None = None
        self.expert_placement_strategy: ExpertPlacementStrategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )

        # ROCm aiter shared experts fusion
        self.rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled
            else 0
        )
        if (
            not self.aiter_fmoe_shared_expert_enabled
            and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            self.expert_placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=self.expert_placement_strategy,
                moe_parallel_config=self.moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=self.enable_eplb,
            )

            self.expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.top_k = top_k

        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config, dp_size=dp_size_
        )

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        # ToDo: Better logic to determine the routing method type
        if routing_method_type is not None:
            self.routing_method_type = routing_method_type
        else:
            if scoring_func == "sigmoid":
                if self.use_grouped_topk:
                    self.routing_method_type = RoutingMethodType.DeepSeekV3
                elif self.top_k == 1:
                    self.routing_method_type = RoutingMethodType.Llama4
            elif self.scoring_func == "softmax":
                self.routing_method_type = (
                    RoutingMethodType.Renormalize
                    if not self.renormalize
                    else RoutingMethodType.RenormalizeNaive
                )
            else:
                self.routing_method_type = RoutingMethodType.TopK

        self.moe_config: FusedMoEConfig = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
        )
        self.moe_config_use_flashinfer_cutlass_kernels = (
            self.moe_config.use_flashinfer_cutlass_kernels
        )
        self.uvm_moe_trace_enabled = _env_flag_enabled(
            "VLLM_UVM_MOE_ROUTING_TRACE_ENABLE", default=False
        )
        self.uvm_moe_trace_file = os.environ.get(
            "VLLM_UVM_MOE_ROUTING_TRACE_FILE",
            "vllm_uvm_moe_routing_trace.jsonl",
        )
        self._uvm_moe_trace_step = 0
        self.uvm_moe_address_map_enabled = _env_flag_enabled(
            "VLLM_UVM_MOE_ADDRESS_MAP_ENABLE", default=False
        )
        self.uvm_moe_address_map_file = os.environ.get(
            "VLLM_UVM_MOE_ADDRESS_MAP_FILE",
            "vllm_uvm_moe_address_map.jsonl",
        )
        self.uvm_moe_address_map_driver_probe = _env_flag_enabled(
            "VLLM_UVM_MOE_ADDRESS_MAP_DRIVER_PROBE", default=True
        )
        self.uvm_weight_prefetch_trace_enabled = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE", default=True
        )
        self.uvm_weight_prefetch_enabled = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_ENABLE", default=False
        )
        self.uvm_weight_prefetch_mode = os.environ.get(
            "VLLM_UVM_WEIGHT_PREFETCH_MODE", "trace_only"
        )
        if self.uvm_weight_prefetch_mode not in ("trace_only", "prefetch"):
            logger.warning(
                "Unknown VLLM_UVM_WEIGHT_PREFETCH_MODE=%s; using trace_only",
                self.uvm_weight_prefetch_mode,
            )
            self.uvm_weight_prefetch_mode = "trace_only"
        self.uvm_weight_prefetch_trace_file = os.environ.get(
            "VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE",
            "vllm_uvm_weight_prefetch_stage_i.jsonl",
        )
        self.uvm_moe_stream_trace = _env_flag_enabled(
            "VLLM_UVM_MOE_STREAM_TRACE",
            default=False,
        )
        self.uvm_moe_stream_trace_file = os.environ.get(
            "VLLM_UVM_MOE_STREAM_TRACE_FILE",
            self.uvm_weight_prefetch_trace_file,
        )
        self.uvm_moe_stream_trace_marker = _env_flag_enabled(
            "VLLM_UVM_MOE_STREAM_TRACE_MARKER",
            default=False,
        )
        self._uvm_moe_stream_trace_marker_disabled = False
        self.uvm_moe_topk_item_probe_mode = (
            os.environ.get(
                "VLLM_UVM_MOE_TOPK_ITEM_PROBE_MODE",
                "off",
            )
            .strip()
            .lower()
        )
        if self.uvm_moe_topk_item_probe_mode == "gpu_only":
            self.uvm_moe_topk_item_probe_mode = "any"
        elif self.uvm_moe_topk_item_probe_mode == "with_item":
            self.uvm_moe_topk_item_probe_mode = "item"
        if self.uvm_moe_topk_item_probe_mode not in (
            "off",
            "cast",
            "valid",
            "clamp",
            "gather",
            "and",
            "any",
            "item",
            "kernel",
            "kernel_item",
        ):
            logger.warning(
                "Unknown VLLM_UVM_MOE_TOPK_ITEM_PROBE_MODE=%s; using off",
                self.uvm_moe_topk_item_probe_mode,
            )
            self.uvm_moe_topk_item_probe_mode = "off"
        self.uvm_moe_topk_item_probe_width = max(
            _env_int("VLLM_UVM_MOE_TOPK_ITEM_PROBE_WIDTH", default=0),
            0,
        )
        self._uvm_moe_topk_item_probe_mask_cache: dict[
            tuple[str, int, int], torch.Tensor
        ] = {}
        self._uvm_moe_topk_item_probe_last: torch.Tensor | None = None
        self._uvm_moe_topk_item_probe_last_item = False
        self.uvm_expert_reserved_topk_gate_impl = (
            os.environ.get(
                "VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_IMPL",
                "triton",
            )
            .strip()
            .lower()
        )
        if self.uvm_expert_reserved_topk_gate_impl not in (
            "triton",
            "triton_async",
            "torch",
            "none",
            "simple",
            "fused_select",
            "fused_select_async",
        ):
            logger.warning(
                "Unknown VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_IMPL=%s; using triton",
                self.uvm_expert_reserved_topk_gate_impl,
            )
            self.uvm_expert_reserved_topk_gate_impl = "triton"
        self.uvm_expert_offload_access_mode = (
            os.environ.get(
                "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE",
                "restore",
            )
            .strip()
            .lower()
        )
        if self.uvm_expert_offload_access_mode not in (
            "restore",
            "remote_read",
        ):
            logger.warning(
                "Unknown VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=%s; using restore",
                self.uvm_expert_offload_access_mode,
            )
            self.uvm_expert_offload_access_mode = "restore"
        self.uvm_remote_prepare_mode = (
            os.environ.get(
                "VLLM_UVM_REMOTE_PREPARE_MODE",
                "native",
            )
            .strip()
            .lower()
        )
        if self.uvm_remote_prepare_mode not in ("native", "release_remote"):
            logger.warning(
                "Unknown VLLM_UVM_REMOTE_PREPARE_MODE=%s; using native",
                self.uvm_remote_prepare_mode,
            )
            self.uvm_remote_prepare_mode = "native"
        self.uvm_expert_reserved_topk_gate_assume_valid = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_ASSUME_VALID",
            default=True,
        )
        self.uvm_expert_reserved_topk_gate_max_block = max(
            _env_int(
                "VLLM_UVM_EXPERT_RESERVED_TOPK_GATE_MAX_BLOCK",
                default=65536,
            ),
            1,
        )
        self.uvm_weight_router_topk_every_n_steps = _env_int(
            "VLLM_UVM_WEIGHT_ROUTER_TOPK_EVERY_N_STEPS",
            default=1,
        )
        if self.uvm_weight_router_topk_every_n_steps <= 0:
            self.uvm_weight_router_topk_every_n_steps = 1
        self._uvm_weight_prefetch_issue_info: dict[str, dict[str, object]] = {}
        self.uvm_weight_prefetch_expert_cache_enabled = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_EXPERT_CACHE_ENABLE",
            default=True,
        )
        self._uvm_weight_expert_role_states: dict[tuple[str, int], dict[str, str]] = {}
        self._uvm_weight_full_gpu_experts: set[tuple[str, int]] = set()
        self.uvm_weight_prefetch_target_roles = {
            role.strip()
            for role in os.environ.get(
                "VLLM_UVM_WEIGHT_PREFETCH_TARGET_ROLES",
                "moe_gate_up,moe_down",
            ).split(",")
            if role.strip()
        }
        self.uvm_weight_prefetch_device = _env_int(
            "VLLM_UVM_WEIGHT_PREFETCH_DEVICE",
            default=-1,
        )
        self.uvm_weight_prefetch_plan_file = os.environ.get(
            "VLLM_UVM_WEIGHT_PREFETCH_PLAN_FILE",
            "",
        )
        self.uvm_weight_prefetch_hit_trace_enabled = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_HIT_TRACE_ENABLE",
            default=False,
        )
        self.uvm_weight_prefetch_hit_driver_query_enabled = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_HIT_DRIVER_QUERY_ENABLE",
            default=False,
        )
        self.uvm_weight_prefetch_hit_max_regions_per_step = _env_int(
            "VLLM_UVM_WEIGHT_PREFETCH_HIT_MAX_REGIONS_PER_STEP",
            default=64,
        )
        if self.uvm_weight_prefetch_hit_max_regions_per_step < 0:
            self.uvm_weight_prefetch_hit_max_regions_per_step = 0
        self.uvm_weight_prefetch_plan_keys = _parse_plan_expert_keys(
            self.uvm_weight_prefetch_plan_file,
            "prefetch_plan",
        )
        self.uvm_weight_batch_executor_enable = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_BATCH_EXECUTOR_ENABLE",
            default=True,
        )
        self.uvm_weight_executor = os.environ.get(
            "VLLM_UVM_WEIGHT_EXECUTOR",
            "cpp_batch",
        )
        if self.uvm_weight_executor not in (
            "cpp_batch",
            "driver_batch",
            "auto",
            "expert_reserved_regions_async",
        ):
            logger.warning(
                "Unknown VLLM_UVM_WEIGHT_EXECUTOR=%s; using cpp_batch",
                self.uvm_weight_executor,
            )
            self.uvm_weight_executor = "cpp_batch"
        self.uvm_expert_reserved_enable = (
            self.uvm_weight_executor == "expert_reserved_regions_async"
        )
        self.uvm_expert_reserved_pool_id = _env_int(
            "VLLM_UVM_EXPERT_RESERVED_POOL_ID",
            default=0,
        )
        self.uvm_expert_reserved_slot_count = _env_int(
            "VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT",
            default=6,
        )
        if self.uvm_expert_reserved_slot_count < 0:
            self.uvm_expert_reserved_slot_count = 0
        self.uvm_expert_reserved_slot_bytes = _env_int(
            "VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES",
            default=0,
        )
        if self.uvm_expert_reserved_slot_bytes < 0:
            self.uvm_expert_reserved_slot_bytes = 0
        legacy_2mb_envelope_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_2MB_ENVELOPE_ENABLE",
            default=False,
        )
        legacy_2mb_envelope_bytes = _env_int(
            "VLLM_UVM_EXPERT_2MB_ENVELOPE_BYTES",
            default=_UVM_EXPERT_2MB_ENVELOPE_BYTES,
        )
        if legacy_2mb_envelope_bytes <= 0:
            legacy_2mb_envelope_bytes = _UVM_EXPERT_2MB_ENVELOPE_BYTES
        legacy_2mb_envelope_bytes = _align_up(
            legacy_2mb_envelope_bytes,
            4096,
        )
        generic_envelope_env_present = (
            "VLLM_UVM_EXPERT_ENVELOPE_ENABLE" in os.environ
            or "VLLM_UVM_EXPERT_ENVELOPE_BYTES" in os.environ
        )
        self.uvm_expert_envelope_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_ENVELOPE_ENABLE",
            default=legacy_2mb_envelope_enable,
        )
        self.uvm_expert_envelope_bytes = _env_int(
            "VLLM_UVM_EXPERT_ENVELOPE_BYTES",
            default=(
                _UVM_EXPERT_512K_ENVELOPE_BYTES
                if generic_envelope_env_present
                else legacy_2mb_envelope_bytes
            ),
        )
        if self.uvm_expert_envelope_bytes <= 0:
            self.uvm_expert_envelope_bytes = (
                _UVM_EXPERT_512K_ENVELOPE_BYTES
                if generic_envelope_env_present
                else _UVM_EXPERT_2MB_ENVELOPE_BYTES
            )
        self.uvm_expert_envelope_bytes = _align_up(
            self.uvm_expert_envelope_bytes,
            4096,
        )
        self.uvm_expert_envelope_strict_logical_alignment = _env_flag_enabled(
            "VLLM_UVM_EXPERT_ENVELOPE_STRICT_LOGICAL_ALIGNMENT",
            default=True,
        )
        self.uvm_expert_2mb_envelope_enable = self.uvm_expert_envelope_enable
        self.uvm_expert_2mb_envelope_bytes = self.uvm_expert_envelope_bytes
        self.uvm_expert_reserved_offloaded_ids = _parse_int_set_env(
            "VLLM_UVM_EXPERT_RESERVED_OFFLOADED_EXPERT_IDS"
        )
        self.uvm_expert_warm_prefetch_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WARM_PREFETCH_ENABLE",
            default=self.uvm_expert_reserved_enable,
        )
        self.uvm_expert_warm_prefetch_distance = max(
            _env_int("VLLM_UVM_EXPERT_WARM_PREFETCH_DISTANCE", default=1),
            1,
        )
        self.uvm_expert_warm_prefetch_max_experts_per_layer = max(
            _env_int(
                "VLLM_UVM_EXPERT_WARM_PREFETCH_MAX_EXPERTS_PER_LAYER",
                default=0,
            ),
            0,
        )
        self.uvm_expert_warm_async_min_tokens = max(
            _env_int("VLLM_UVM_EXPERT_WARM_ASYNC_MIN_TOKENS", default=0),
            0,
        )
        self.uvm_expert_warm_background_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WARM_BACKGROUND_ENABLE",
            default=True,
        )
        self.uvm_expert_warm_fail_closed = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WARM_FAIL_CLOSED",
            default=True,
        )
        self.uvm_expert_reserved_cleanup_on_layer_done = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_CLEANUP_ON_LAYER_DONE",
            default=True,
        )
        self.uvm_expert_reserved_cleanup_delay_steps = _env_int(
            "VLLM_UVM_EXPERT_RESERVED_CLEANUP_DELAY_STEPS",
            default=0,
        )
        if self.uvm_expert_reserved_cleanup_delay_steps < 0:
            self.uvm_expert_reserved_cleanup_delay_steps = 0
        self.uvm_expert_reserved_invalidate_on_layer_done = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_INVALIDATE_ON_LAYER_DONE",
            default=True,
        )
        self.uvm_expert_reserved_strict_footprint = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT",
            default=True,
        )
        self.uvm_expert_reserved_async = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_ASYNC",
            default=False,
        )
        self.uvm_expert_reserved_wait_on_reuse = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_WAIT_ON_REUSE",
            default=True,
        )
        self.uvm_expert_reserved_sync_before_compute = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE",
            default=True,
        )
        self.uvm_expert_reserved_cleanup_synchronize = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE",
            default=True,
        )
        self.uvm_expert_reserved_debug_sync_compute = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_DEBUG_SYNC_COMPUTE",
            default=False,
        )
        self.uvm_expert_reserved_stream_trace = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_STREAM_TRACE",
            default=False,
        )
        self.uvm_expert_reserved_fail_fast = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_FAIL_FAST",
            default=True,
        )
        self.uvm_expert_reserved_skip_active_topk = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_SKIP_ACTIVE_TOPK",
            default=False,
        )
        self.uvm_expert_reserved_enable_after_warmup = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_ENABLE_AFTER_WARMUP",
            default=(
                self.uvm_expert_reserved_enable
                and os.environ.get(
                    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                    "post_warmup",
                )
                .strip()
                .lower()
                .replace("-", "_")
                in ("post_warmup", "after_warmup", "deferred")
            ),
        )
        self.uvm_expert_reserved_coverage_trace_enabled = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_ENABLE",
            default=False,
        )
        self.uvm_expert_reserved_coverage_trace_file = os.environ.get(
            "VLLM_UVM_EXPERT_RESERVED_COVERAGE_TRACE_FILE",
            self.uvm_weight_prefetch_trace_file,
        )
        self.uvm_expert_reserved_coverage_residency_query = _env_flag_enabled(
            "VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_QUERY",
            default=False,
        )
        self.uvm_expert_reserved_coverage_residency_max_experts = _env_int(
            "VLLM_UVM_EXPERT_RESERVED_COVERAGE_RESIDENCY_MAX_EXPERTS",
            default=0,
        )
        self._uvm_expert_reserved_active_steps: set[int] = set()
        self.uvm_weight_offload_plan_file = os.environ.get(
            "VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE",
            self.uvm_weight_prefetch_plan_file,
        )
        self.uvm_weight_offload_plan_keys = _parse_plan_expert_keys(
            self.uvm_weight_offload_plan_file,
            "offload_plan",
        )
        self._uvm_weight_prefetch_plan_loaded = bool(
            self.uvm_weight_prefetch_plan_file
            and os.path.isfile(self.uvm_weight_prefetch_plan_file)
        )
        self._uvm_weight_offload_plan_loaded = bool(
            self.uvm_weight_offload_plan_file
            and os.path.isfile(self.uvm_weight_offload_plan_file)
        )
        self.uvm_init_weight_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_init_weight_offload_mode = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE",
            "trace_only",
        )
        if self.uvm_init_weight_offload_mode not in (
            "trace_only",
            *_UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES,
        ):
            logger.warning(
                "Unknown VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=%s; using trace_only",
                self.uvm_init_weight_offload_mode,
            )
            self.uvm_init_weight_offload_mode = "trace_only"
        self.uvm_init_weight_offload_plan_file = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_PLAN_FILE",
            self.uvm_weight_offload_plan_file,
        )
        self.uvm_init_weight_require_routing_plan = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_REQUIRE_ROUTING_PLAN",
            default=False,
        )
        self.uvm_init_weight_offload_target_roles = {
            role.strip()
            for role in os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_TARGET_ROLES",
                ",".join(sorted(self.uvm_weight_prefetch_target_roles)),
            ).split(",")
            if role.strip()
        }
        self.uvm_init_weight_offload_max_bytes = _env_int(
            "VLLM_UVM_INIT_WEIGHT_MAX_BYTES",
            default=0,
        )
        self.uvm_init_weight_offload_max_experts_per_layer = _env_int(
            "VLLM_UVM_INIT_WEIGHT_MAX_EXPERTS_PER_LAYER",
            default=0,
        )
        self.uvm_init_weight_offload_sync = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_SYNC",
            default=True,
        )
        self.uvm_init_weight_hot_prefetch_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_ENABLE",
            default=False,
        )
        self.uvm_init_weight_hot_prefetch_sync = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_SYNC",
            default=True,
        )
        self.uvm_init_weight_hot_prefetch_max_bytes = _env_int(
            "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_MAX_BYTES",
            default=0,
        )
        self.uvm_init_weight_hot_prefetch_device = _env_int(
            "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_DEVICE",
            default=-1,
        )
        self.uvm_init_weight_hot_prefetch_real_query_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_HOT_PREFETCH_REAL_QUERY_ENABLE",
            default=False,
        )
        self.uvm_init_weight_hot_repair_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_HOT_REPAIR_ENABLE",
            default=False,
        )
        self.uvm_init_weight_hot_repair_preferred_gpu = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_HOT_REPAIR_PREFERRED_GPU",
            default=True,
        )
        self.uvm_init_weight_hot_repair_require_full_residency = (
            _env_flag_enabled(
                "VLLM_UVM_INIT_WEIGHT_HOT_REPAIR_REQUIRE_FULL_RESIDENCY",
                default=True,
            )
        )
        self.uvm_hot_residency_stage_trace_enable = _env_flag_enabled(
            "VLLM_UVM_HOT_RESIDENCY_STAGE_TRACE_ENABLE",
            default=False,
        )
        self.uvm_init_weight_plan_residency_trace_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_ENABLE",
            default=False,
        )
        self.uvm_init_weight_plan_residency_trace_max_experts = _env_int(
            "VLLM_UVM_INIT_WEIGHT_PLAN_RESIDENCY_TRACE_MAX_EXPERTS",
            default=0,
        )
        self.uvm_hot_residency_stage_max_num_tokens = _env_int(
            "VLLM_UVM_HOT_RESIDENCY_STAGE_MAX_NUM_TOKENS",
            default=1,
        )
        self.uvm_hot_residency_stage_device = _env_int(
            "VLLM_UVM_HOT_RESIDENCY_STAGE_DEVICE",
            default=-1,
        )
        self._uvm_hot_residency_stage_markers_seen: set[str] = set()
        self.uvm_init_weight_offload_trace_file = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_TRACE_FILE",
            self.uvm_weight_prefetch_trace_file,
        )
        self.uvm_init_weight_offload_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "offload_plan",
        )
        self.uvm_init_weight_hot_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "hot_plan",
        ) | (
            _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "prefetch_plan",
            )
            - self.uvm_init_weight_offload_plan_keys
        )
        self.uvm_init_weight_warm_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "warm_plan",
        )
        self.uvm_init_weight_plan_policy = _uvm_weight_plan_policy(
            self.uvm_init_weight_offload_plan_file,
        )
        self._uvm_init_weight_offload_plan_loaded = bool(
            self.uvm_init_weight_offload_plan_file
            and os.path.isfile(self.uvm_init_weight_offload_plan_file)
        )
        register_all_default = (
            self.uvm_expert_reserved_enable
            or self.uvm_init_weight_offload_enable
            or self.uvm_init_weight_hot_prefetch_enable
            or self.uvm_init_weight_hot_repair_enable
        )
        self.uvm_init_weight_register_all_local_experts = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_REGISTER_ALL_LOCAL_EXPERTS",
            default=register_all_default,
        )
        self.uvm_init_weight_complete_expert_plan = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN",
            default=self.uvm_init_weight_register_all_local_experts,
        )
        self.uvm_init_weight_complete_plan_ignore_limits = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_COMPLETE_PLAN_IGNORE_LIMITS",
            default=False,
        )
        self.uvm_init_weight_warm_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_incremental_expert_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_dynamic_expert_swap_enable = _env_flag_enabled(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        )
        self._uvm_dynamic_expert_heat_counts: torch.Tensor | None = None
        self.uvm_incremental_experts_per_layer_per_round = max(
            _env_int(
                "VLLM_UVM_INCREMENTAL_EXPERTS_PER_LAYER_PER_ROUND",
                default=1,
            ),
            1,
        )
        self._uvm_incremental_offload_in_progress = False
        self._uvm_incremental_allow_warm = False
        self._uvm_incremental_selected_global_expert_id: int | None = None
        self._uvm_plan_expert_bitmask_width = int(self.global_num_experts)
        self._uvm_reserved_offload_mask_tensor_cache: dict[
            tuple[int | None, str, int], torch.Tensor
        ] = {}
        self._uvm_reserved_offload_mask_values_cache: (
            tuple[frozenset[int], int, list[bool]] | None
        ) = None
        self._uvm_topk_gate_hit_flag_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._uvm_slot_managed_global_expert_ids: set[int] = set()
        self._uvm_remote_read_global_expert_ids: set[int] = set()
        self._uvm_offloaded_warm_global_expert_ids: set[int] = set()
        self._uvm_warm_prefetch_generation = 0
        self._uvm_warm_prefetch_ticket: dict[str, object] | None = None
        self._uvm_warm_prefetch_ready_step = -1
        self._uvm_warm_prefetch_ready_global_expert_ids: set[int] = set()
        self._uvm_warm_prefetch_source_tickets: list[dict[str, object]] = []
        self._uvm_expert_envelope_region_cache: dict[
            tuple[int | None, int, str], list[ReservedRegionItem]
        ] = {}
        self._uvm_init_weight_offload_done = False
        self._uvm_init_weight_hot_repair_done = False
        uvm_layer_id = self._uvm_layer_id()
        if uvm_layer_id is not None:
            self._uvm_register_warm_prefetch_module(uvm_layer_id)
        self.uvm_residency_weight_initial_state = os.environ.get(
            "VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE", "cpu"
        )
        if self.uvm_residency_weight_initial_state not in (
            "unknown",
            "cpu",
            "gpu",
        ):
            logger.warning(
                "Unknown VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE=%s; using cpu",
                self.uvm_residency_weight_initial_state,
            )
            self.uvm_residency_weight_initial_state = "cpu"
        self.uvm_expert_weights_managed = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED",
            default=False,
        )
        self.quant_config = quant_config

        def _get_quant_method() -> FusedMoEMethodBase:
            """
            Helper method to ensure self.quant_method is never None and
            of the proper type.
            """
            quant_method = None
            if self.quant_config is not None:
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                quant_method = UnquantizedFusedMoEMethod(self.moe_config)
            assert isinstance(quant_method, FusedMoEMethodBase)
            return quant_method

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        self.quant_method: FusedMoEMethodBase = _get_quant_method()

        if not self.moe_config.is_act_and_mul:
            # Avoid circular import
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8MoEMethod,
            )

            if not isinstance(
                self.quant_method, (UnquantizedFusedMoEMethod, ModelOptFp8MoEMethod)
            ):
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for unquantized "
                    "and ModelOpt FP8 moe for now"
                )
            if not current_platform.is_cuda():
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for CUDA for now"
                )

        if self.enable_eplb and not self.quant_method.supports_eplb:
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}. "
                "EPLB is only supported for FP8 quantization for now."
            )

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": self.global_num_experts,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    def uvm_expert_weight_allocation_scope(self):
        if not self.uvm_expert_weights_managed:
            return nullcontext()
        return uvm_local_memory_pool()

    def _uvm_managed_expert_weight_parameter_records(
        self,
    ) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        local_num_experts = int(self.local_num_experts)
        if local_num_experts <= 0:
            return records

        for name, param in self.named_parameters(recurse=False):
            if name == "e_score_correction_bias":
                continue
            if not isinstance(param, torch.Tensor):
                continue
            if not param.is_cuda or param.ndim == 0 or param.numel() <= 0:
                continue
            if int(param.shape[0]) != local_num_experts:
                continue

            size_bytes = int(param.numel()) * int(param.element_size())
            ptr = int(param.data_ptr())
            capable, reason, code = probe_driver_capable_range(ptr, size_bytes)
            if name.startswith(("w13", "w1", "w3")):
                role = "moe_gate_up"
            elif name.startswith("w2"):
                role = "moe_down"
            else:
                role = "moe_expert_aux"
            records.append(
                {
                    "name": name,
                    "role": role,
                    "ptr": ptr,
                    "ptr_hex": f"0x{ptr:x}",
                    "bytes": size_bytes,
                    "dtype": str(param.dtype),
                    "shape": list(param.shape),
                    "driver_capable": bool(capable),
                    "driver_capability_reason": reason,
                    "driver_capability_code": int(code),
                }
            )
        return records

    def prepare_uvm_managed_expert_weights_for_request(
        self,
        marker: str,
    ) -> dict[str, object] | None:
        if not self.uvm_expert_weights_managed:
            return None

        layer_id = self._uvm_layer_id()
        try:
            device = _env_int("VLLM_UVM_EXPERT_WEIGHTS_DEVICE", default=-1)
            if device < 0:
                device = int(torch.cuda.current_device())
        except Exception:
            device = 0

        records = self._uvm_managed_expert_weight_parameter_records()
        managed_records = [record for record in records if record["driver_capable"]]
        ranges = [
            (int(record["ptr"]), int(record["bytes"]))
            for record in managed_records
            if int(record["ptr"]) > 0 and int(record["bytes"]) > 0
        ]

        prefetch_enabled = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WEIGHTS_PREFETCH_BEFORE_REQUEST",
            default=True,
        )
        prefetch_successes = 0
        prefetch_elapsed_ms = 0.0
        if prefetch_enabled and ranges:
            prefetch_start_ns = time.perf_counter_ns()
            prefetch_successes = batch_prefetch_ranges_to_device(
                ranges,
                device=device,
            )
            try:
                torch.cuda.synchronize(device)
            except Exception:
                logger.debug(
                    "Failed to synchronize after expert-only UVM prefetch",
                    exc_info=True,
                )
            prefetch_elapsed_ms = (
                time.perf_counter_ns() - prefetch_start_ns
            ) / 1_000_000.0

        residency_query = {
            "ok": True,
            "status": 0,
            "queried_bytes": 0,
            "gpu_bytes": 0,
            "cpu_bytes": 0,
            "any_bytes": 0,
            "unknown_bytes": 0,
        }
        if ranges and _env_flag_enabled(
            "VLLM_UVM_EXPERT_WEIGHTS_RESIDENCY_QUERY_ENABLE",
            default=True,
        ):
            try:
                residency_query = driver_batch_query_ranges_residency_bytes(
                    ranges,
                    device,
                )
            except Exception:
                logger.debug("Failed to query expert-only UVM residency", exc_info=True)
                residency_query = {
                    "ok": False,
                    "status": -1,
                    "queried_bytes": 0,
                    "gpu_bytes": 0,
                    "cpu_bytes": 0,
                    "any_bytes": 0,
                    "unknown_bytes": 0,
                }

        queried_bytes = int(residency_query.get("queried_bytes") or 0)
        gpu_bytes = int(residency_query.get("gpu_bytes") or 0)
        cpu_bytes = int(residency_query.get("cpu_bytes") or 0)
        unknown_bytes = int(residency_query.get("unknown_bytes") or 0)
        non_managed = [
            {
                "name": str(record["name"]),
                "role": str(record["role"]),
                "ptr": str(record["ptr_hex"]),
                "bytes": int(record["bytes"]),
                "driver_capability_reason": str(record["driver_capability_reason"]),
                "driver_capability_code": int(record["driver_capability_code"]),
            }
            for record in records
            if not bool(record["driver_capable"])
        ]
        gpu_full = (
            bool(residency_query.get("ok"))
            and len(records) > 0
            and len(non_managed) == 0
            and queried_bytes > 0
            and gpu_bytes == queried_bytes
            and cpu_bytes == 0
            and unknown_bytes == 0
        )

        result: dict[str, object] = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "action": "uvm_expert_weights_prepare_for_request",
            "marker": marker,
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "device": device,
            "local_num_experts": int(self.local_num_experts),
            "parameter_records": len(records),
            "managed_parameter_records": len(managed_records),
            "non_managed_parameter_records": len(non_managed),
            "managed_bytes": sum(int(record["bytes"]) for record in managed_records),
            "prefetch_enabled": prefetch_enabled,
            "prefetch_successes": int(prefetch_successes),
            "prefetch_elapsed_ms": prefetch_elapsed_ms,
            "residency_query": residency_query,
            "queried_bytes": queried_bytes,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "unknown_bytes": unknown_bytes,
            "gpu_full": gpu_full,
            "non_managed_parameters": non_managed[:8],
            "sample_parameters": [
                {
                    "name": str(record["name"]),
                    "role": str(record["role"]),
                    "ptr": str(record["ptr_hex"]),
                    "bytes": int(record["bytes"]),
                    "driver_capable": bool(record["driver_capable"]),
                    "dtype": str(record["dtype"]),
                    "shape": record["shape"],
                }
                for record in records[:8]
            ],
        }

        trace_file = os.environ.get(
            "VLLM_UVM_EXPERT_WEIGHTS_RESIDENCY_TRACE_FILE",
            "",
        )
        if trace_file:
            try:
                trace_dir = os.path.dirname(trace_file)
                if trace_dir:
                    os.makedirs(trace_dir, exist_ok=True)
                with open(trace_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, sort_keys=True) + "\n")
            except OSError:
                logger.debug(
                    "Failed to write expert-only UVM residency trace",
                    exc_info=True,
                )

        if (
            _env_flag_enabled(
                "VLLM_UVM_EXPERT_WEIGHTS_REQUIRE_GPU_FULL",
                default=False,
            )
            and not gpu_full
        ):
            raise RuntimeError(
                "expert-only UVM managed weights are not fully GPU resident: "
                f"layer={self.layer_name} result={result!r}"
            )
        return result

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        self.ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self.quant_method = FusedMoEModularMethod.make(
                self, self.quant_method, prepare_finalize, self.shared_experts
            )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return None

    @property
    def gate(self) -> torch.nn.Module | None:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def pcp_size(self):
        return self.moe_parallel_config.pcp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def pcp_rank(self):
        return self.moe_parallel_config.pcp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (
            self.moe_quant_config is not None
            and self.moe_quant_config.quant_dtype == "nvfp4"
            and self.moe_config_use_flashinfer_cutlass_kernels
        )

    @property
    def use_marlin_kernels(self):
        return getattr(self.quant_method, "use_marlin", False)

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        )

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return False

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # Currently routing_tables only needed for round-robin expert placement
        # with DeepEP-ll all2all backend.
        if (
            self.expert_placement_strategy != "round_robin"
            or not self.use_deepep_ll_kernels
        ):
            return None

        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )

        if self.expert_map is None:
            return None

        routing_tables = self.ensure_round_robin_expert_routing_tables(
            global_num_experts=self.global_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            local_num_experts=self.local_num_experts,
            device=self.expert_map.device,
        )

        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    @staticmethod
    def ensure_round_robin_expert_routing_tables(
        global_num_experts: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_kwargs = {"device": device} if device is not None else {}
        global_indices = torch.arange(
            global_num_experts, dtype=torch.long, **device_kwargs
        )
        owner = torch.remainder(global_indices, ep_size)
        local_index = torch.div(global_indices, ep_size, rounding_mode="floor")
        base = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        physical_offset = owner * base
        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            ep_rank,
            global_num_experts,
            ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != local_num_experts:
            local_global = local_global[:local_num_experts]

        return (global_to_physical, physical_to_global, local_global)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self.expert_map is not None
        with self.expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            if self.aiter_fmoe_shared_expert_enabled:
                self._init_aiter_shared_experts_topK_buffer(
                    vllm_config=get_current_vllm_config(),
                    dp_size=get_dp_group().world_size,
                )

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = hidden_states.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We dont need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())
            self._append_uvm_reserved_debug_trace(
                "reserved_stream_shared_experts_wait_current",
                hidden_shape=list(hidden_states.shape),
                use_chunked_impl=use_chunked_impl,
            )

        return use_shared_experts_stream, hidden_states_clone

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
        self, vllm_config: VllmConfig, dp_size: int
    ):
        if self.num_fused_shared_experts > 0:
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
                * dp_size,
                is_EP=self.use_ep,
            )
        self.local_num_experts += self.num_fused_shared_experts

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        allow_flashinfer = getattr(self.quant_method, "allow_flashinfer", False)
        moe_backend = getattr(self.quant_method, "flashinfer_moe_backend", None)

        use_global_sf = (
            allow_flashinfer
            and is_flashinfer_supporting_global_sf(moe_backend)
            and "input_scale" in weight_name
            and quant_method_name == "ModelOptNvFp4FusedMoE"
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=param,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded %s for expert %d into %s",
                        param_name,
                        expert_id,
                        self.layer_name,
                    )
                    yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer paramater as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith("_shared_experts.")
        )

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_shared_experts.")
            # exclude parameters from non-expert submodules (e.g. gate/shared)
            and not name.startswith("_gate.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None:
        self.ensure_moe_quant_config_init()
        return self.quant_method.moe_quant_config

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.vllm_config.parallel_config.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.hidden_size)
            logits_shape = (2, moe.max_num_tokens, self.logical_num_experts)
        else:
            states_shape = (moe.max_num_tokens, self.hidden_size)
            logits_shape = (moe.max_num_tokens, self.logical_num_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids, zero_expert_result)
                (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                The weights, expert ids, and zero expert computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        if self.enable_eplb:
            if self.quant_method.supports_eplb:
                if self.expert_load_view is None:
                    raise ValueError(
                        "enable_eplb=True requiere expert_load_view != None"
                    )
                if self.logical_to_physical_map is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_to_physical_map != None"
                    )
                if self.logical_replica_count is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_replica_count != None"
                    )
            else:
                raise NotImplementedError(
                    f"EPLB is not supported for {self.quant_method.method_name}."
                )

        indices_type = self.quant_method.topk_indices_dtype
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        fused_select_plans_refreshed = False
        fused_select_mask: torch.Tensor | None = None
        fused_select_hit_flag: torch.Tensor | None = None
        self._uvm_fused_select_hit_flag = None
        self._uvm_fused_select_hit_layer_id = None
        run_reserved_topk_gate = (
            self.uvm_expert_offload_access_mode == "restore"
            and self.uvm_expert_reserved_enable
            and self.uvm_weight_prefetch_enabled
            and not self.uvm_expert_reserved_skip_active_topk
            and (
                self.uvm_weight_router_topk_every_n_steps <= 1
                or (
                    self._uvm_moe_trace_step % self.uvm_weight_router_topk_every_n_steps
                )
                == 0
            )
        )
        if (
            run_reserved_topk_gate
            and self.uvm_expert_reserved_topk_gate_impl
            in ("fused_select", "fused_select_async")
            and not self.enable_eplb
            and routing_strategy == ""
            and not self.use_grouped_topk
            and self.e_score_correction_bias is None
            and self.custom_routing_function is None
            and router_logits.is_cuda
        ):
            self._refresh_uvm_weight_action_plans()
            fused_select_plans_refreshed = True
            fused_layer_id = self._uvm_layer_id()
            mask_values = self._uvm_reserved_offload_mask_for_layer(fused_layer_id)
            if mask_values:
                fused_select_mask = self._uvm_reserved_offload_mask_tensor(
                    layer_id=fused_layer_id,
                    device=router_logits.device,
                    mask_values=mask_values,
                )
                fused_select_hit_flag = self._uvm_topk_gate_hit_flag_tensor(
                    router_logits.device
                )
                self._uvm_fused_select_hit_flag = fused_select_hit_flag
                self._uvm_fused_select_hit_layer_id = fused_layer_id

        # Check if we should use a routing simulation strategy
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=self.top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif self.use_grouped_topk:
            assert self.topk_group is not None
            assert self.num_expert_group is not None
            if rocm_aiter_ops.is_fused_moe_enabled():
                if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                    assert self.num_fused_shared_experts == 0
                grouped_topk_impl = partial(
                    rocm_aiter_grouped_topk,
                    num_fused_shared_experts=self.num_fused_shared_experts,
                )
            else:
                grouped_topk_impl = grouped_topk

            topk_weights, topk_ids = grouped_topk_impl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                expert_mask=fused_select_mask,
                hit_flag=fused_select_hit_flag,
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

        self._record_uvm_dynamic_expert_heat(topk_ids)

        if self.enable_eplb:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None
        _record_uvm_pre_expert_timing_ids_ready(
            layer_name=self.layer_name,
            topk_ids=topk_ids,
        )

        # Compute zero expert result if needed
        if (
            self.zero_expert_num is not None
            and self.zero_expert_num > 0
            and self.zero_expert_type is not None
            and self.global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.global_num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        self._log_uvm_moe_routing_trace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        self._record_uvm_moe_address_map(
            marker="select_experts_after_topk",
            extra_tensors={
                "hidden_states": hidden_states,
                "router_logits": router_logits,
                "topk_ids": topk_ids,
                "topk_weights": topk_weights,
            },
        )
        self._run_uvm_moe_topk_item_probe(topk_ids)
        if (
            self.uvm_expert_reserved_enable
            and self.uvm_expert_offload_access_mode == "restore"
        ):
            if (
                self.uvm_weight_prefetch_enabled
                and self.uvm_expert_reserved_cleanup_delay_steps > 0
                and self._uvm_expert_reserved_active_steps
            ):
                self._maybe_cleanup_uvm_expert_reserved_slots(
                    step=(
                        self._uvm_moe_trace_step
                        - self.uvm_expert_reserved_cleanup_delay_steps
                    )
                )
            if run_reserved_topk_gate:
                if not fused_select_plans_refreshed:
                    self._refresh_uvm_weight_action_plans()
                reserved_layer_id = self._uvm_layer_id()
                if self._uvm_topk_has_reserved_offloaded_expert(
                    topk_ids,
                    reserved_layer_id,
                ):
                    active_expert_pairs = self._uvm_active_expert_pairs(topk_ids)
                    reserved_expert_pairs = (
                        self._uvm_active_reserved_offload_expert_pairs(
                            reserved_layer_id,
                            active_expert_pairs,
                        )
                    )
                    if reserved_expert_pairs:
                        self._maybe_restore_uvm_reserved_expert_weights(
                            topk_ids,
                            num_tokens=int(hidden_states.shape[0]),
                            layer_id=reserved_layer_id,
                            active_expert_pairs=active_expert_pairs,
                            reserved_expert_pairs=reserved_expert_pairs,
                            plans_refreshed=True,
                            cleanup_already_done=True,
                        )
                elif self.uvm_weight_prefetch_trace_enabled:
                    self._append_uvm_weight_prefetch_trace(
                        {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "layer_name": self.layer_name,
                            "step": self._uvm_moe_trace_step,
                            "mode": self.uvm_weight_prefetch_mode,
                            "prefetch_source": "reserved_fast_gate",
                            "action": "reserved_prefetch_no_offloaded_active",
                            "active_global_experts": [],
                            "candidate_expert_records": 0,
                            "reserved_offload_mask_hit_records": 0,
                            "caller_fast_gate": True,
                            "gpu_mask_gate": True,
                        }
                    )
            self._maybe_prefetch_next_layer_uvm_warm_expert_weights(
                num_tokens=int(hidden_states.shape[0]),
            )
        self._uvm_moe_trace_step += 1
        return topk_weights, topk_ids, zero_expert_result

    def _log_uvm_moe_routing_trace(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        num_tokens = int(hidden_states.shape[0]) if hidden_states.ndim > 0 else 1
        if (
            self.uvm_hot_residency_stage_trace_enable
            and num_tokens <= self.uvm_hot_residency_stage_max_num_tokens
            and "before_first_decode" not in self._uvm_hot_residency_stage_markers_seen
        ):
            self._uvm_hot_residency_stage_markers_seen.add("before_first_decode")
            self._query_uvm_init_hot_plan_residency(
                marker="before_first_decode",
                step=self._uvm_moe_trace_step,
                num_tokens=num_tokens,
            )
        if not self.uvm_moe_trace_enabled:
            return
        try:
            ids = topk_ids.detach()
            valid_ids = ids[ids >= 0].to(dtype=torch.long)
            counts = torch.bincount(
                valid_ids.flatten(),
                minlength=max(int(self.global_num_experts), 0),
            ).cpu()
            weight_sums: dict[str, float] = {}
            try:
                weights = topk_weights.detach()
                if weights.shape == ids.shape:
                    valid_weights = weights[ids >= 0].to(dtype=torch.float32)
                    weighted_counts = torch.bincount(
                        valid_ids.flatten(),
                        weights=valid_weights.flatten(),
                        minlength=max(int(self.global_num_experts), 0),
                    ).cpu()
                    weight_sums = {
                        str(int(expert_id)): float(weighted_counts[expert_id].item())
                        for expert_id in torch.nonzero(weighted_counts, as_tuple=False)
                        .flatten()
                        .tolist()
                        if weighted_counts[expert_id].item() > 0
                    }
            except Exception:
                logger.debug(
                    "Failed to compute weighted UVM MoE routing trace",
                    exc_info=True,
                )
            active = torch.nonzero(counts, as_tuple=False).flatten().tolist()
            active_counts = {
                str(int(expert_id)): int(counts[expert_id].item())
                for expert_id in active
                if counts[expert_id].item() > 0
            }
            record = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "step": self._uvm_moe_trace_step,
                "num_tokens": num_tokens,
                "top_k": int(self.top_k),
                "global_num_experts": int(self.global_num_experts),
                "logical_num_experts": int(self.logical_num_experts),
                "local_num_experts": int(self.local_num_experts),
                "topk_shape": list(topk_ids.shape),
                "router_logits_shape": list(router_logits.shape),
                "expert_token_counts": active_counts,
                "expert_weight_sums": weight_sums,
                "active_experts": [int(expert_id) for expert_id in active],
                "topk_weight_sum": float(topk_weights.detach().sum().cpu().item()),
                "enable_eplb": bool(self.enable_eplb),
            }
            with open(self.uvm_moe_trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write UVM MoE routing trace", exc_info=True)

    def _record_uvm_dynamic_expert_heat(self, topk_ids: torch.Tensor) -> None:
        """Accumulate logical expert assignments on GPU until maintenance."""
        if not self.uvm_dynamic_expert_swap_enable:
            return
        try:
            ids = topk_ids.detach()
            valid_ids = ids[(ids >= 0) & (ids < int(self.global_num_experts))]
            if valid_ids.numel() == 0:
                return
            counts = torch.bincount(
                valid_ids.flatten().to(dtype=torch.long),
                minlength=int(self.global_num_experts),
            )
            if (
                self._uvm_dynamic_expert_heat_counts is None
                or self._uvm_dynamic_expert_heat_counts.device != counts.device
            ):
                self._uvm_dynamic_expert_heat_counts = torch.zeros(
                    int(self.global_num_experts),
                    dtype=torch.int64,
                    device=counts.device,
                )
            self._uvm_dynamic_expert_heat_counts.add_(counts)
        except Exception:
            logger.debug("Failed to record dynamic expert heat", exc_info=True)

    def get_uvm_dynamic_expert_heat(
        self,
        *,
        reset: bool = False,
    ) -> dict[str, object]:
        """Materialize one layer's routing window at a maintenance boundary."""
        counts_tensor = self._uvm_dynamic_expert_heat_counts
        if counts_tensor is None:
            counts = [0] * int(self.global_num_experts)
        else:
            # The counter is created during model inference and is therefore an
            # inference tensor. Maintenance RPCs run outside that context.
            with torch.inference_mode():
                counts = [int(value) for value in counts_tensor.detach().cpu().tolist()]
                if reset:
                    counts_tensor.zero_()
        return {
            "ok": True,
            "layer_name": self.layer_name,
            "layer_id": self._uvm_layer_id(),
            "global_num_experts": int(self.global_num_experts),
            "local_num_experts": int(self.local_num_experts),
            "expert_token_counts": counts,
            "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
            "reset": bool(reset),
        }

    def _uvm_layer_id(self) -> int | None:
        parts = self.layer_name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part in ("layers", "layer"):
                try:
                    return int(parts[index + 1])
                except ValueError:
                    return None
        return None

    def _uvm_global_to_local_expert(self, global_expert_id: int) -> int:
        if self.expert_map is None:
            return global_expert_id
        if 0 <= global_expert_id < int(self.expert_map.numel()):
            return int(self.expert_map[global_expert_id].item())
        return -1

    def _uvm_active_expert_pairs(
        self,
        topk_ids: torch.Tensor,
    ) -> list[tuple[int, int]]:
        ids = topk_ids.detach()
        valid_ids = ids[ids >= 0].to(dtype=torch.long)
        if valid_ids.numel() == 0:
            return []
        unique_ids = torch.unique(valid_ids).cpu().tolist()
        expert_pairs: list[tuple[int, int]] = []
        for expert_id in unique_ids:
            global_expert_id = int(expert_id)
            local_expert_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= local_expert_id < int(self.local_num_experts):
                expert_pairs.append((global_expert_id, local_expert_id))
        return sorted(set(expert_pairs))

    def _uvm_active_local_experts(self, topk_ids: torch.Tensor) -> list[int]:
        return sorted(
            {
                local_expert_id
                for _, local_expert_id in self._uvm_active_expert_pairs(topk_ids)
            }
        )

    def _uvm_plan_local_experts(
        self,
        plan_keys: set[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return []
        expert_pairs: list[tuple[int, int]] = []
        for plan_layer_id, global_expert_id in sorted(plan_keys):
            if plan_layer_id != layer_id:
                continue
            local_expert_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= local_expert_id < int(self.local_num_experts):
                expert_pairs.append((global_expert_id, local_expert_id))
        return sorted(set(expert_pairs))

    def _uvm_all_local_expert_pairs(self) -> list[tuple[int, int]]:
        expert_pairs: set[tuple[int, int]] = set()
        mapped_local_ids: set[int] = set()
        for global_expert_id in range(max(int(self.global_num_experts), 0)):
            local_expert_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= local_expert_id < int(self.local_num_experts):
                expert_pairs.add((global_expert_id, local_expert_id))
                mapped_local_ids.add(local_expert_id)

        for local_expert_id in range(max(int(self.local_num_experts), 0)):
            if local_expert_id in mapped_local_ids:
                continue
            if local_expert_id < int(self.global_num_experts):
                expert_pairs.add((local_expert_id, local_expert_id))
        return sorted(expert_pairs)

    def _uvm_register_warm_prefetch_module(self, layer_id: int) -> None:
        registry_key = (
            os.getpid(),
            _uvm_layer_sequence_key(self.layer_name),
            int(layer_id),
        )
        with _UVM_WARM_PREFETCH_REGISTRY_LOCK:
            _UVM_WARM_PREFETCH_REGISTRY[registry_key] = self

    def _uvm_next_warm_prefetch_module(self) -> "FusedMoE | None":
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None
        sequence_key = _uvm_layer_sequence_key(self.layer_name)
        with _UVM_WARM_PREFETCH_REGISTRY_LOCK:
            candidates = sorted(
                (
                    candidate_layer_id,
                    module,
                )
                for (
                    candidate_pid,
                    candidate_sequence_key,
                    candidate_layer_id,
                ), module in list(_UVM_WARM_PREFETCH_REGISTRY.items())
                if candidate_pid == os.getpid()
                and candidate_sequence_key == sequence_key
                and candidate_layer_id > layer_id
                and isinstance(module, FusedMoE)
            )
        distance = self.uvm_expert_warm_prefetch_distance
        if len(candidates) < distance:
            return None
        return cast(FusedMoE, candidates[distance - 1][1])

    def _uvm_init_weight_expert_is_hot(
        self,
        layer_id: int | None,
        global_expert_id: int,
    ) -> bool:
        if layer_id is None:
            return False
        plan_key = (layer_id, global_expert_id)
        if plan_key in self.uvm_init_weight_offload_plan_keys:
            return False
        return (
            plan_key in self.uvm_init_weight_hot_plan_keys
            or plan_key in self.uvm_weight_prefetch_plan_keys
        )

    def _uvm_init_weight_expert_class(
        self,
        *,
        layer_id: int | None,
        global_expert_id: int,
    ) -> str:
        if self._uvm_init_weight_expert_is_hot(layer_id, global_expert_id):
            return "hot"
        if (
            layer_id is not None
            and (
                layer_id,
                global_expert_id,
            )
            in self.uvm_init_weight_offload_plan_keys
        ):
            return "cold"
        return "warm"

    def _uvm_stage_m_offload_expert_pairs(
        self,
        layer_id: int,
        all_local_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if not self.uvm_init_weight_offload_enable:
            return []

        cold_pairs: list[tuple[int, int]] = []
        warm_pairs: list[tuple[int, int]] = []
        layer_warm_plan_ids = {
            expert_id
            for plan_layer_id, expert_id in self.uvm_init_weight_warm_plan_keys
            if plan_layer_id == layer_id
        }
        for global_expert_id, local_expert_id in all_local_pairs:
            if global_expert_id in (
                self._uvm_slot_managed_global_expert_ids
                | getattr(self, "_uvm_remote_read_global_expert_ids", set())
            ):
                continue
            expert_class = self._uvm_init_weight_expert_class(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
            )
            if expert_class == "cold":
                cold_pairs.append((global_expert_id, local_expert_id))
                continue
            if expert_class != "warm":
                continue
            if layer_warm_plan_ids:
                if global_expert_id in layer_warm_plan_ids:
                    warm_pairs.append((global_expert_id, local_expert_id))
                continue
            if (
                self.uvm_init_weight_warm_offload_enable
                and self.uvm_init_weight_complete_expert_plan
            ):
                warm_pairs.append((global_expert_id, local_expert_id))

        # Limits consume this list from the front, so warm experts cannot be
        # released until all selected cold experts have been considered.
        return sorted(cold_pairs) + sorted(warm_pairs)

    def _uvm_expert_reserved_envelope_count(self, local_expert_id: int) -> int:
        slices = self._uvm_expert_weight_slices(
            local_expert_id,
            self.uvm_init_weight_offload_target_roles,
        )
        if not self.uvm_expert_envelope_enable:
            return len(slices)
        block_bytes = max(int(self.uvm_expert_envelope_bytes), 1)
        return sum(
            max(
                (
                    _align_up(int(ptr) + int(size_bytes), block_bytes)
                    - _align_down(int(ptr), block_bytes)
                )
                // block_bytes,
                1,
            )
            for _, ptr, size_bytes in slices
        )

    def uvm_incremental_reserved_slot_requirement(self) -> int:
        """Return slots needed for one incremental expert wave per bank."""
        max_envelopes = max(
            (
                self._uvm_expert_reserved_envelope_count(local_expert_id)
                for _, local_expert_id in self._uvm_all_local_expert_pairs()
            ),
            default=0,
        )
        bank_count = max(
            _env_int("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", default=2),
            1,
        )
        return (
            max_envelopes
            * self.uvm_incremental_experts_per_layer_per_round
            * bank_count
        )

    def _uvm_incremental_slot_capacity_allows(
        self,
        local_expert_id: int,
    ) -> bool:
        if self._uvm_stage_m_uses_remote_read():
            # Remote-read experts keep their managed VA and use CPU backing.
            # They do not consume a reserved restore slot.
            return True
        bank_count = min(
            max(
                _env_int("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", default=2),
                1,
            ),
            max(self.uvm_expert_reserved_slot_count, 1),
        )
        bank_slots = self.uvm_expert_reserved_slot_count // bank_count
        if bank_slots <= 0:
            return False
        existing_envelopes = 0
        for global_expert_id in self._uvm_slot_managed_global_expert_ids:
            existing_local_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= existing_local_id < int(self.local_num_experts):
                existing_envelopes += self._uvm_expert_reserved_envelope_count(
                    existing_local_id
                )
        return (
            existing_envelopes
            + self._uvm_expert_reserved_envelope_count(local_expert_id)
            <= bank_slots
        )

    def has_remaining_uvm_incremental_experts(self, expert_class: str) -> bool:
        return self.uvm_next_incremental_expert(expert_class) is not None

    def uvm_next_incremental_expert(self, expert_class: str) -> int | None:
        """Return this layer's next capacity-safe global expert candidate."""
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None
        pairs = self._uvm_stage_m_offload_expert_pairs(
            layer_id,
            self._uvm_all_local_expert_pairs(),
        )
        for global_expert_id, local_expert_id in pairs:
            if (
                self._uvm_init_weight_expert_class(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                )
                != expert_class
            ):
                continue
            if self._uvm_incremental_slot_capacity_allows(local_expert_id):
                return int(global_expert_id)
        return None

    def run_uvm_incremental_weight_offload(
        self,
        *,
        allow_warm: bool,
        selected_global_expert_id: int | None = None,
    ) -> dict[str, object]:
        if not self.uvm_incremental_expert_offload_enable:
            return {"ok": True, "reason": "disabled", "released_experts": []}
        remote_read = self._uvm_stage_m_uses_remote_read()
        tracked_experts = (
            self._uvm_remote_read_global_expert_ids
            if remote_read
            else self._uvm_slot_managed_global_expert_ids
        )
        before = set(tracked_experts)
        self._uvm_incremental_offload_in_progress = True
        self._uvm_incremental_allow_warm = bool(allow_warm)
        self._uvm_incremental_selected_global_expert_id = (
            None
            if selected_global_expert_id is None
            else int(selected_global_expert_id)
        )
        self._uvm_init_weight_offload_done = False
        try:
            self.run_uvm_init_weight_offload()
        finally:
            self._uvm_incremental_offload_in_progress = False
            self._uvm_incremental_allow_warm = False
            self._uvm_incremental_selected_global_expert_id = None
        released = sorted(set(tracked_experts) - before)
        return {
            "ok": True,
            "reason": "released" if released else "no_capacity_or_candidates",
            "layer_id": self._uvm_layer_id(),
            "released_experts": released,
            "released_count": len(released),
            "allow_warm": bool(allow_warm),
            "selected_global_expert_id": selected_global_expert_id,
            "access_mode": "remote_read" if remote_read else "restore",
        }

    def _uvm_mark_slot_managed_expert(
        self,
        *,
        global_expert_id: int,
        expert_class: str,
    ) -> None:
        self._uvm_slot_managed_global_expert_ids.add(int(global_expert_id))
        if expert_class == "warm":
            self._uvm_offloaded_warm_global_expert_ids.add(int(global_expert_id))
        self._uvm_reserved_offload_mask_values_cache = None
        self._uvm_reserved_offload_mask_tensor_cache.clear()

    def _uvm_mark_remote_read_expert(self, global_expert_id: int) -> None:
        self._uvm_remote_read_global_expert_ids.add(int(global_expert_id))

    def _uvm_record_release_remote_envelope_states(
        self,
        envelope_records: object,
        entries_by_region: dict[int, dict[str, object]],
    ) -> None:
        if not isinstance(envelope_records, list):
            return
        for record in envelope_records:
            if isinstance(record, ReservedRegionItem):
                region_id = str(record.region_id)
                driver_region_id = int(record.driver_region_id)
            elif isinstance(record, dict):
                region_id = str(record.get("region_id") or "")
                driver_region_id = int(record.get("driver_region_id") or 0)
            else:
                continue
            entry = entries_by_region.get(driver_region_id)
            if entry is None:
                continue
            if not region_id:
                continue
            snapshot = get_uvm_residency_region_snapshot(region_id) or {}
            metadata = dict(
                cast(dict[str, object], snapshot.get("metadata") or {})
            )
            complete = bool(entry.get("ok"))
            changed = (
                int(entry.get("released_bytes") or 0) > 0
                or int(entry.get("remote_mapped_bytes") or 0) > 0
            )
            metadata.update(
                {
                    "remote_prepare_mode": "release_remote",
                    "driver_release_remote": True,
                    "driver_release_remote_complete": complete,
                    "driver_release_remote_entry": entry,
                    "placement": (
                        "CPU_REMOTE_READ"
                        if complete
                        else "MIXED_UNCOMMITTED"
                        if changed
                        else "GPU_RESIDENT"
                    ),
                    "cpu_remote_read": complete,
                }
            )
            mark_uvm_residency_state(
                region_id=region_id,
                state="cpu" if complete else "unknown" if changed else "gpu",
                metadata=metadata,
            )

    def _uvm_mark_dynamic_expert_placement(
        self,
        *,
        layer_id: int,
        global_expert_id: int,
        local_expert_id: int,
        slices: list[tuple[str, int, int, dict[str, object] | None]],
        state: str,
        action: str,
        transaction_id: str,
    ) -> None:
        cpu_remote = state == "cpu"
        placement = "CPU_REMOTE_READ" if cpu_remote else "GPU_RESIDENT"
        for role, ptr, size_bytes, _ in slices:
            region_id = self._uvm_expert_weight_region_id(
                layer_id,
                global_expert_id,
                role,
            )
            metadata = self._uvm_expert_weight_residency_metadata(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                role=role,
                ptr=ptr,
                size_bytes=size_bytes,
                active_this_step=False,
                source_action=action,
            )
            metadata.update(
                {
                    "source": "dynamic_expert_swap",
                    "placement": placement,
                    "target_placement": placement,
                    "cpu_remote_read": cpu_remote,
                    "dynamic_expert_swap": True,
                    "transaction_id": transaction_id,
                }
            )
            mark_uvm_residency_state(
                region_id=region_id,
                state=state,
                metadata=metadata,
            )
            self._uvm_update_weight_expert_role_cache(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                role=role,
                state=state,
                estimated_gpu_resident=not cpu_remote,
            )
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "layer_id": layer_id,
                    "action": action,
                    "transaction_id": transaction_id,
                    "global_expert_id": global_expert_id,
                    "local_expert_id": local_expert_id,
                    "role": role,
                    "ptr": f"0x{ptr:x}",
                    "bytes": size_bytes,
                    "state": state,
                    "placement": placement,
                    "success": True,
                }
            )

            cache = getattr(self, "_uvm_expert_envelope_region_cache", None)
            if not isinstance(cache, dict):
                continue
            cache_key = self._uvm_expert_envelope_cache_key(
                layer_id,
                global_expert_id,
                role,
            )
            cached_items = cache.get(cache_key)
            if not isinstance(cached_items, list):
                continue
            updated_items: list[ReservedRegionItem] = []
            for item in cached_items:
                snapshot = get_uvm_residency_region_snapshot(item.region_id) or {}
                envelope_metadata = {
                    **cast(dict[str, object], snapshot.get("metadata") or {}),
                    **item.metadata,
                    "source": "dynamic_expert_swap",
                    "placement": placement,
                    "target_placement": placement,
                    "cpu_remote_read": cpu_remote,
                    "driver_release_remote_active": cpu_remote,
                    "dynamic_expert_swap": True,
                    "transaction_id": transaction_id,
                }
                mark_uvm_residency_state(
                    region_id=item.region_id,
                    state=state,
                    metadata=envelope_metadata,
                )
                updated_items.append(
                    ReservedRegionItem(
                        region_id=item.region_id,
                        driver_region_id=int(item.driver_region_id),
                        ptr=int(item.ptr),
                        bytes=int(item.bytes),
                        layer_id=item.layer_id,
                        global_expert_id=int(item.global_expert_id),
                        local_expert_id=int(item.local_expert_id),
                        role=item.role,
                        metadata=envelope_metadata,
                    )
                )
            cache[cache_key] = updated_items

    def _uvm_release_remote_expert_slices(
        self,
        *,
        layer_id: int,
        global_expert_id: int,
        local_expert_id: int,
        slices: list[tuple[str, int, int, dict[str, object] | None]],
        device: int,
        source_action: str,
    ) -> dict[str, object]:
        envelope_bytes = _UVM_EXPERT_512K_ENVELOPE_BYTES
        if (
            not self.uvm_expert_envelope_enable
            or int(self.uvm_expert_envelope_bytes) != envelope_bytes
        ):
            return {
                "ok": False,
                "reason": "release_remote_512k_envelopes_required",
                "envelope_enable": bool(self.uvm_expert_envelope_enable),
                "envelope_bytes": int(self.uvm_expert_envelope_bytes),
            }

        envelope_items: list[ReservedRegionItem] = []
        logical_region_ids: list[str] = []
        for role, ptr, size_bytes, existing in slices:
            region_id = self._uvm_expert_weight_region_id(
                layer_id,
                global_expert_id,
                role,
            )
            snapshot = existing or get_uvm_residency_region_snapshot(region_id) or {}
            if ptr % envelope_bytes != 0 or size_bytes % envelope_bytes != 0:
                return {
                    "ok": False,
                    "reason": "unaligned_release_remote_logical_region",
                    "role": role,
                    "ptr": ptr,
                    "bytes": size_bytes,
                    "envelope_bytes": envelope_bytes,
                }
            base_metadata = self._uvm_expert_weight_residency_metadata(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                role=role,
                ptr=ptr,
                size_bytes=size_bytes,
                active_this_step=False,
                source_action=source_action,
            )
            metadata = {
                **cast(dict[str, object], snapshot.get("metadata") or {}),
                **base_metadata,
                "source": source_action,
                "placement": "GPU_RESIDENT",
                "target_placement": "CPU_REMOTE_READ",
                "cpu_remote_read": False,
                "dynamic_expert_swap": True,
            }
            role_envelopes = self._uvm_get_or_build_expert_envelope_items(
                source_region_id=region_id,
                logical_driver_region_id=int(
                    snapshot.get("driver_region_id") or 0
                ),
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                role=role,
                ptr=ptr,
                size_bytes=size_bytes,
                metadata=metadata,
                state=str(snapshot.get("state") or "gpu"),
                source_action=source_action,
            )
            if (
                not role_envelopes
                or sum(int(item.bytes) for item in role_envelopes) != size_bytes
            ):
                return {
                    "ok": False,
                    "reason": "release_remote_envelope_coverage_incomplete",
                    "role": role,
                    "region_id": region_id,
                    "logical_bytes": size_bytes,
                    "envelope_count": len(role_envelopes),
                    "envelope_covered_bytes": sum(
                        int(item.bytes) for item in role_envelopes
                    ),
                }
            for item in role_envelopes:
                if (
                    int(item.ptr) % envelope_bytes != 0
                    or int(item.bytes) != envelope_bytes
                    or int(item.driver_region_id) <= 0
                    or not bool(item.metadata.get("driver_registered"))
                ):
                    return {
                        "ok": False,
                        "reason": "invalid_release_remote_envelope",
                        "role": role,
                        "region_id": item.region_id,
                        "ptr": int(item.ptr),
                        "bytes": int(item.bytes),
                        "driver_region_id": int(item.driver_region_id),
                        "driver_registered": bool(
                            item.metadata.get("driver_registered")
                        ),
                    }
            logical_region_ids.append(region_id)
            envelope_items.extend(role_envelopes)

        unique_envelopes: dict[int, ReservedRegionItem] = {}
        for item in envelope_items:
            driver_region_id = int(item.driver_region_id)
            existing_item = unique_envelopes.get(driver_region_id)
            if existing_item is not None and (
                existing_item.region_id != item.region_id
                or int(existing_item.ptr) != int(item.ptr)
                or int(existing_item.bytes) != int(item.bytes)
            ):
                return {
                    "ok": False,
                    "reason": "conflicting_release_remote_envelope_identity",
                    "driver_region_id": driver_region_id,
                    "first_region_id": existing_item.region_id,
                    "second_region_id": item.region_id,
                }
            unique_envelopes[driver_region_id] = item

        driver_region_ids = list(unique_envelopes)
        result = driver_prefetch_cpu_release_remote_512k_regions(
            driver_region_ids,
            device=device,
            prefer_existing=True,
        )
        entries_by_region = {
            int(entry.get("region_id") or 0): entry
            for entry in result.get("entries") or []
            if isinstance(entry, dict)
        }
        self._uvm_record_release_remote_envelope_states(
            list(unique_envelopes.values()),
            entries_by_region,
        )
        complete_entries = [
            entries_by_region.get(driver_region_id)
            for driver_region_id in driver_region_ids
        ]
        exact_complete = bool(driver_region_ids) and all(
            entry is not None
            and bool(entry.get("ok"))
            and bool(entry.get("prefetch_release_complete"))
            and int(entry.get("length") or 0) == envelope_bytes
            and int(entry.get("released_bytes") or 0) == envelope_bytes
            and int(entry.get("cpu_resident_bytes") or 0) == envelope_bytes
            and int(entry.get("remote_mapped_bytes") or 0) == envelope_bytes
            and int(entry.get("remote_map_failure_blocks") or 0) == 0
            for entry in complete_entries
        )
        expected_bytes = len(driver_region_ids) * envelope_bytes
        released_bytes = sum(
            int(entry.get("released_bytes") or 0)
            for entry in complete_entries
            if entry is not None
        )
        cpu_resident_bytes = sum(
            int(entry.get("cpu_resident_bytes") or 0)
            for entry in complete_entries
            if entry is not None
        )
        remote_mapped_bytes = sum(
            int(entry.get("remote_mapped_bytes") or 0)
            for entry in complete_entries
            if entry is not None
        )
        return {
            **result,
            "ok": bool(result.get("ok")) and exact_complete,
            "reason": (
                "released_remote_512k"
                if bool(result.get("ok")) and exact_complete
                else "release_remote_512k_validation_failed"
            ),
            "logical_region_ids": logical_region_ids,
            "logical_region_count": len(logical_region_ids),
            "logical_range_count": len(slices),
            "envelope_region_count": len(driver_region_ids),
            "expected_bytes": expected_bytes,
            "released_bytes": released_bytes,
            "cpu_resident_bytes": cpu_resident_bytes,
            "remote_mapped_bytes": remote_mapped_bytes,
            "source_action": source_action,
        }

    def run_uvm_dynamic_remote_read_swaps(
        self,
        swaps: list[dict[str, object]],
    ) -> dict[str, object]:
        """Swap newly cold GPU experts with newly hot remote-read experts."""
        if not self.uvm_dynamic_expert_swap_enable:
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if not self._uvm_stage_m_uses_remote_read():
            return {"ok": False, "reason": "remote_read_mode_required"}
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return {"ok": False, "reason": "layer_id_unavailable"}

        device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        results: list[dict[str, object]] = []
        for swap_index, swap in enumerate(swaps):
            try:
                promote_expert = int(swap["promote_expert"])
                demote_expert = int(swap["demote_expert"])
            except (KeyError, TypeError, ValueError):
                return {
                    "ok": False,
                    "reason": "invalid_swap_record",
                    "swap_index": swap_index,
                    "results": results,
                }
            transaction_id = (
                f"layer{layer_id}:swap{swap_index}:"
                f"{promote_expert}->{demote_expert}:{time.time_ns()}"
            )
            if (
                promote_expert not in self._uvm_remote_read_global_expert_ids
                or demote_expert in self._uvm_remote_read_global_expert_ids
                or promote_expert == demote_expert
            ):
                return {
                    "ok": False,
                    "reason": "stale_or_invalid_swap",
                    "transaction_id": transaction_id,
                    "promote_expert": promote_expert,
                    "demote_expert": demote_expert,
                    "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
                    "results": results,
                }

            promote_local = self._uvm_global_to_local_expert(promote_expert)
            demote_local = self._uvm_global_to_local_expert(demote_expert)
            if not (
                0 <= promote_local < int(self.local_num_experts)
                and 0 <= demote_local < int(self.local_num_experts)
            ):
                return {
                    "ok": False,
                    "reason": "nonlocal_expert_swap_unsupported",
                    "transaction_id": transaction_id,
                    "promote_expert": promote_expert,
                    "demote_expert": demote_expert,
                    "results": results,
                }

            promote_slices = self._uvm_stage_m_expert_weight_slices(
                layer_id=layer_id,
                global_expert_id=promote_expert,
                local_expert_id=promote_local,
            )
            demote_slices = self._uvm_stage_m_expert_weight_slices(
                layer_id=layer_id,
                global_expert_id=demote_expert,
                local_expert_id=demote_local,
            )
            promote_ranges = [
                (ptr, size_bytes) for _, ptr, size_bytes, _ in promote_slices
            ]
            demote_ranges = [
                (ptr, size_bytes) for _, ptr, size_bytes, _ in demote_slices
            ]
            if not promote_ranges or len(promote_ranges) != len(demote_ranges):
                return {
                    "ok": False,
                    "reason": "incomplete_expert_weight_slices",
                    "transaction_id": transaction_id,
                    "promote_range_count": len(promote_ranges),
                    "demote_range_count": len(demote_ranges),
                    "results": results,
                }

            demote_driver_release_result: dict[str, object] | None = None
            if self._uvm_remote_prepare_uses_release():
                demote_driver_release_result = self._uvm_release_remote_expert_slices(
                    layer_id=layer_id,
                    global_expert_id=demote_expert,
                    local_expert_id=demote_local,
                    slices=demote_slices,
                    device=device,
                    source_action="dynamic_expert_swap_release_remote_envelope",
                )
                demote_successes = (
                    len(demote_ranges)
                    if bool(demote_driver_release_result.get("ok"))
                    else 0
                )
                demote_executor = "driver_release_remote"
                demote_accessed_by = 0
                demote_prefetch_cpu = 0
                demote_preferred_cpu = 0
            else:
                (
                    demote_successes,
                    demote_executor,
                    demote_accessed_by,
                    demote_prefetch_cpu,
                    demote_preferred_cpu,
                ) = batch_prepare_ranges_for_native_remote_access_with_executor(
                    demote_ranges,
                    sync=True,
                    executor=self.uvm_weight_executor,
                    device=device,
                )
            if demote_successes != len(demote_ranges):
                return {
                    "ok": False,
                    "reason": "demote_remote_prepare_failed",
                    "transaction_id": transaction_id,
                    "demote_expert": demote_expert,
                    "demote_successes": demote_successes,
                    "demote_range_count": len(demote_ranges),
                    "demote_driver_release_result": demote_driver_release_result,
                    "results": results,
                }

            # Commit the capacity-producing half first. If promotion fails, the
            # conservative state is two remote experts rather than an OOM.
            self._uvm_remote_read_global_expert_ids.add(demote_expert)
            self._uvm_mark_dynamic_expert_placement(
                layer_id=layer_id,
                global_expert_id=demote_expert,
                local_expert_id=demote_local,
                slices=demote_slices,
                state="cpu",
                action="dynamic_expert_swap_demote_issued",
                transaction_id=transaction_id,
            )

            promote_preferred_gpu = batch_advise_ranges_preferred_location(
                promote_ranges,
                device,
            )
            promote_successes = 0
            promote_executor = self.uvm_weight_executor
            promote_driver_result = 0
            promote_residency: dict[str, int | bool] = {
                "ok": False,
                "status": -1,
                "queried_bytes": 0,
                "gpu_bytes": 0,
                "cpu_bytes": 0,
                "any_bytes": 0,
                "unknown_bytes": 0,
            }
            if promote_preferred_gpu == len(promote_ranges):
                (
                    promote_successes,
                    promote_executor,
                    promote_driver_result,
                ) = batch_prefetch_ranges_to_device_with_executor(
                    promote_ranges,
                    device=device,
                    executor=self.uvm_weight_executor,
                )
                if promote_successes > 0 and torch.cuda.is_available():
                    torch.cuda.synchronize(device)
                if promote_successes == len(promote_ranges):
                    try:
                        promote_residency = (
                            driver_batch_query_ranges_residency_bytes(
                                promote_ranges,
                                device,
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Dynamic expert promotion residency query failed: "
                            "layer=%s expert=%s",
                            layer_id,
                            promote_expert,
                        )

            promote_bytes = sum(size for _, size in promote_ranges)
            promote_residency_complete = (
                bool(promote_residency.get("ok"))
                and int(promote_residency.get("queried_bytes") or 0)
                == promote_bytes
                and int(promote_residency.get("gpu_bytes") or 0)
                == promote_bytes
                and int(promote_residency.get("cpu_bytes") or 0) == 0
                and int(promote_residency.get("any_bytes") or 0)
                == promote_bytes
                and int(promote_residency.get("unknown_bytes") or 0) == 0
            )

            pair_result = {
                "transaction_id": transaction_id,
                "promote_expert": promote_expert,
                "demote_expert": demote_expert,
                "range_count": len(promote_ranges),
                "bytes": sum(size for _, size in promote_ranges),
                "demote_executor": demote_executor,
                "demote_accessed_by_result": demote_accessed_by,
                "demote_prefetch_cpu_successes": demote_prefetch_cpu,
                "demote_preferred_cpu_result": demote_preferred_cpu,
                "demote_driver_release_result": demote_driver_release_result,
                "promote_executor": promote_executor,
                "promote_driver_result": promote_driver_result,
                "promote_preferred_gpu_result": promote_preferred_gpu,
                "promote_prefetch_gpu_successes": promote_successes,
                "promote_residency": promote_residency,
                "promote_residency_complete": promote_residency_complete,
            }
            if promote_preferred_gpu != len(promote_ranges) or promote_successes != len(
                promote_ranges
            ):
                results.append({**pair_result, "ok": False})
                return {
                    "ok": False,
                    "reason": "promote_gpu_prefetch_failed",
                    "partial_demote_committed": True,
                    "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
                    "results": results,
                }

            if not promote_residency_complete:
                results.append({**pair_result, "ok": False})
                return {
                    "ok": False,
                    "reason": "promote_gpu_residency_incomplete",
                    "partial_demote_committed": True,
                    "remote_experts": sorted(
                        self._uvm_remote_read_global_expert_ids
                    ),
                    "results": results,
                }

            self._uvm_remote_read_global_expert_ids.remove(promote_expert)
            self._uvm_mark_dynamic_expert_placement(
                layer_id=layer_id,
                global_expert_id=promote_expert,
                local_expert_id=promote_local,
                slices=promote_slices,
                state="gpu",
                action="dynamic_expert_swap_promote_issued",
                transaction_id=transaction_id,
            )
            results.append({**pair_result, "ok": True})

        return {
            "ok": True,
            "reason": "swapped" if results else "no_swaps",
            "layer_id": layer_id,
            "swapped_count": len(results),
            "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
            "results": results,
        }

    def _uvm_stage_m_uses_remote_read(self) -> bool:
        return (
            getattr(self, "uvm_expert_offload_access_mode", "restore") == "remote_read"
            and self.uvm_init_weight_offload_mode
            in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
        )

    def _uvm_remote_prepare_uses_release(self) -> bool:
        return (
            self._uvm_stage_m_uses_remote_read()
            and getattr(self, "uvm_remote_prepare_mode", "native")
            == "release_remote"
        )

    def _uvm_stage_m_requires_release_envelopes(self) -> bool:
        return (
            self.uvm_init_weight_offload_mode
            in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
            and (
                not self._uvm_stage_m_uses_remote_read()
                or self._uvm_remote_prepare_uses_release()
            )
        )

    def _uvm_stage_m_requires_logical_release_registration(self) -> bool:
        return self._uvm_remote_prepare_uses_release() and not (
            self._uvm_stage_m_requires_release_envelopes()
        )

    def _uvm_stage_m_sync_enabled(self) -> bool:
        return self._uvm_stage_m_uses_remote_read() or bool(
            self.uvm_init_weight_offload_sync
        )

    def _uvm_init_weight_class_metadata(
        self,
        *,
        metadata: dict[str, object],
        expert_class: str,
        driver_tag_skip: bool = False,
        cpu_offload_placement: bool = True,
    ) -> dict[str, object]:
        class_metadata = {
            **metadata,
            "source": "stage_m",
            "stage": "stage_m",
            "expert_init_class": expert_class,
            "plan_policy": self.uvm_init_weight_plan_policy,
            "driver_tag_skip": bool(driver_tag_skip),
            "driver_tag_skip_reason": (
                self._uvm_expert_envelope_skip_reason() if driver_tag_skip else None
            ),
        }
        if expert_class == "hot":
            return {
                **class_metadata,
                "pool_class": "hot",
                "driver_subpool": "hot",
                "placement": "GPU_RESIDENT",
                "target_placement": "GPU_RESIDENT",
                "init_hot": True,
                "init_cold": False,
                "protected": True,
                "pinned": True,
                "prefetchable": True,
                "priority": 100,
            }
        reserved_metadata = {
            **class_metadata,
            "pool_class": expert_class,
            "driver_subpool": f"{expert_class}_reserved",
            "target_placement": "CPU_OFFLOADED_RESERVED",
            "cpu_offloaded_reserved": False,
            "reserved_prefetch_eligible": False,
            "init_hot": False,
            "init_warm": expert_class == "warm",
            "init_cold": expert_class == "cold",
            "protected": False,
            "pinned": False,
            "prefetchable": True,
            "priority": 20 if expert_class == "warm" else 10,
        }
        if cpu_offload_placement:
            if self._uvm_stage_m_uses_remote_read():
                reserved_metadata.update(
                    {
                        "driver_subpool": f"{expert_class}_remote_read",
                        "placement": "CPU_REMOTE_READ",
                        "target_placement": "CPU_REMOTE_READ",
                        "cpu_remote_read": True,
                        "cpu_offloaded_reserved": False,
                        "reserved_prefetch_eligible": False,
                        "prefetchable": False,
                    }
                )
            else:
                reserved_metadata.update(
                    {
                        "placement": "CPU_OFFLOADED_RESERVED",
                        "cpu_offloaded_reserved": True,
                        "reserved_prefetch_eligible": True,
                    }
                )
        else:
            reserved_metadata["placement"] = "GPU_RESIDENT"
        return reserved_metadata

    def _uvm_expert_in_reserved_offload_mask(
        self,
        layer_id: int | None,
        global_expert_id: int,
    ) -> bool:
        del layer_id
        return (
            global_expert_id in self._uvm_slot_managed_global_expert_ids
            or global_expert_id in self.uvm_expert_reserved_offloaded_ids
        )

    def _uvm_reserved_offload_mask_for_layer(
        self,
        layer_id: int | None,
    ) -> list[bool] | None:
        del layer_id
        managed_ids = (
            self._uvm_slot_managed_global_expert_ids
            | self.uvm_expert_reserved_offloaded_ids
        )
        if not managed_ids:
            return None
        frozen_ids = frozenset(int(expert_id) for expert_id in managed_ids)
        max_global_id = max(managed_ids, default=-1)
        width = max(
            self._uvm_plan_expert_bitmask_width,
            max_global_id + 1,
        )
        if width <= 0:
            return None
        cached = self._uvm_reserved_offload_mask_values_cache
        if cached is not None and cached[0] == frozen_ids and cached[1] == width:
            return cached[2]
        mask = [False] * width
        for expert_id in frozen_ids:
            if 0 <= expert_id < width:
                mask[expert_id] = True
        self._uvm_reserved_offload_mask_values_cache = (frozen_ids, width, mask)
        return mask

    def _uvm_reserved_offload_mask_tensor(
        self,
        *,
        layer_id: int | None,
        device: torch.device,
        mask_values: list[bool],
    ) -> torch.Tensor:
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (layer_id, device.type, device_index)
        cached = self._uvm_reserved_offload_mask_tensor_cache.get(cache_key)
        if cached is not None and cached.device == device:
            return cached
        mask_tensor = torch.as_tensor(mask_values, dtype=torch.bool, device=device)
        self._uvm_reserved_offload_mask_tensor_cache[cache_key] = mask_tensor
        return mask_tensor

    def _uvm_topk_gate_hit_flag_tensor(self, device: torch.device) -> torch.Tensor:
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (device.type, device_index)
        cached = self._uvm_topk_gate_hit_flag_cache.get(cache_key)
        if cached is not None and cached.device == device:
            return cached
        flag_tensor = torch.zeros((), dtype=torch.int32, device=device)
        self._uvm_topk_gate_hit_flag_cache[cache_key] = flag_tensor
        return flag_tensor

    def _uvm_topk_mask_hit_triton(
        self,
        *,
        ids: torch.Tensor,
        mask_tensor: torch.Tensor,
        width: int,
        assume_valid: bool,
        sync: bool = True,
    ) -> bool | torch.Tensor | None:
        if not HAS_TRITON or not ids.is_cuda or not ids.is_contiguous():
            return None
        topk_numel = int(ids.numel())
        if topk_numel <= 0 or topk_numel > self.uvm_expert_reserved_topk_gate_max_block:
            return None
        block_size = _uvm_next_power_of_2(topk_numel)
        if block_size > self.uvm_expert_reserved_topk_gate_max_block:
            return None
        hit_flag = self._uvm_topk_gate_hit_flag_tensor(ids.device)
        _uvm_topk_mask_hit_kernel[(1,)](
            ids,
            mask_tensor,
            hit_flag,
            topk_numel,
            width=width,
            ASSUME_VALID=bool(assume_valid),
            BLOCK_SIZE=block_size,
        )
        if not sync:
            return hit_flag
        return bool(hit_flag.item())

    def _uvm_topk_mask_simple_triton(
        self,
        *,
        ids: torch.Tensor,
        sync: bool = True,
    ) -> bool | torch.Tensor | None:
        if not HAS_TRITON or not ids.is_cuda:
            return None
        hit_flag = self._uvm_topk_gate_hit_flag_tensor(ids.device)
        _uvm_topk_mask_simple_kernel[(1,)](hit_flag)
        if not sync:
            return hit_flag
        return bool(hit_flag.item())

    def _run_uvm_moe_topk_item_probe(self, topk_ids: torch.Tensor) -> None:
        mode = self.uvm_moe_topk_item_probe_mode
        if mode == "off":
            return

        ids = topk_ids.detach()
        if ids.numel() == 0 or not ids.is_cuda:
            return

        width = self.uvm_moe_topk_item_probe_width
        if width <= 0:
            width = max(
                int(self.global_num_experts or 0),
                int(self.logical_num_experts or 0),
                1,
            )

        device = ids.device
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (device.type, device_index, width)
        mask_tensor = self._uvm_moe_topk_item_probe_mask_cache.get(cache_key)
        if mask_tensor is None or mask_tensor.device != device:
            mask_tensor = torch.zeros(width, dtype=torch.bool, device=device)
            self._uvm_moe_topk_item_probe_mask_cache[cache_key] = mask_tensor

        if mode in ("kernel", "kernel_item"):
            hit = self._uvm_topk_mask_hit_triton(
                ids=ids,
                mask_tensor=mask_tensor,
                width=width,
                assume_valid=self.uvm_expert_reserved_topk_gate_assume_valid,
                sync=mode == "kernel_item",
            )
            if hit is not None:
                if isinstance(hit, torch.Tensor):
                    self._uvm_moe_topk_item_probe_last = hit
                else:
                    self._uvm_moe_topk_item_probe_last_item = bool(hit)
                return

        index_ids = ids.to(dtype=torch.long)
        if mode == "cast":
            self._uvm_moe_topk_item_probe_last = index_ids
            return

        valid = (index_ids >= 0) & (index_ids < width)
        if mode == "valid":
            self._uvm_moe_topk_item_probe_last = valid
            return

        clamped_ids = index_ids.clamp(min=0, max=width - 1)
        if mode == "clamp":
            self._uvm_moe_topk_item_probe_last = clamped_ids
            return

        hits = mask_tensor[clamped_ids]
        if mode == "gather":
            self._uvm_moe_topk_item_probe_last = hits
            return

        hits = hits & valid
        if mode == "and":
            self._uvm_moe_topk_item_probe_last = hits
            return

        has_hit = hits.any()
        if mode == "item":
            self._uvm_moe_topk_item_probe_last_item = bool(has_hit.item())
        else:
            self._uvm_moe_topk_item_probe_last = has_hit

    def _uvm_topk_has_reserved_offloaded_expert(
        self,
        topk_ids: torch.Tensor,
        layer_id: int | None,
    ) -> bool:
        mask_values = self._uvm_reserved_offload_mask_for_layer(layer_id)
        if not mask_values:
            return False
        width = len(mask_values)
        if width <= 0:
            return False

        ids = topk_ids.detach()
        if ids.numel() == 0:
            return False

        if ids.is_cuda:
            mask_tensor = self._uvm_reserved_offload_mask_tensor(
                layer_id=layer_id,
                device=ids.device,
                mask_values=mask_values,
            )
            if self.uvm_expert_reserved_topk_gate_impl == "none":
                return False
            if self.uvm_expert_reserved_topk_gate_impl in (
                "fused_select",
                "fused_select_async",
            ):
                fused_flag = getattr(self, "_uvm_fused_select_hit_flag", None)
                fused_layer_id = getattr(self, "_uvm_fused_select_hit_layer_id", None)
                if fused_flag is not None and fused_layer_id == layer_id:
                    if self.uvm_expert_reserved_topk_gate_impl == "fused_select_async":
                        return False
                    hit = bool(fused_flag.item())
                    if hit:
                        fused_flag.zero_()
                    return hit
            if self.uvm_expert_reserved_topk_gate_impl == "simple":
                simple_result = self._uvm_topk_mask_simple_triton(ids=ids)
                if simple_result is not None:
                    return bool(simple_result)
                return False
            if self.uvm_expert_reserved_topk_gate_impl == "triton_async":
                self._uvm_topk_mask_hit_triton(
                    ids=ids,
                    mask_tensor=mask_tensor,
                    width=width,
                    assume_valid=self.uvm_expert_reserved_topk_gate_assume_valid,
                    sync=False,
                )
                return False
            if self.uvm_expert_reserved_topk_gate_impl == "triton":
                triton_result = self._uvm_topk_mask_hit_triton(
                    ids=ids,
                    mask_tensor=mask_tensor,
                    width=width,
                    assume_valid=self.uvm_expert_reserved_topk_gate_assume_valid,
                )
                if triton_result is not None:
                    return bool(triton_result)
            index_ids = ids.to(dtype=torch.long)
            if self.uvm_expert_reserved_topk_gate_assume_valid:
                hits = mask_tensor[index_ids]
            else:
                valid = (index_ids >= 0) & (index_ids < width)
                hits = mask_tensor[index_ids.clamp(min=0, max=width - 1)] & valid
            return bool(hits.any().item())

        flat_ids = ids.flatten().to(dtype=torch.long).tolist()
        if self.uvm_expert_reserved_topk_gate_assume_valid:
            return any(mask_values[int(expert_id)] for expert_id in flat_ids)
        return any(
            0 <= int(expert_id) < width and mask_values[int(expert_id)]
            for expert_id in flat_ids
        )

    def _uvm_active_reserved_offload_expert_pairs(
        self,
        layer_id: int | None,
        active_expert_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        return [
            (global_expert_id, local_expert_id)
            for global_expert_id, local_expert_id in active_expert_pairs
            if self._uvm_expert_in_reserved_offload_mask(
                layer_id,
                global_expert_id,
            )
        ]

    def _refresh_uvm_weight_action_plans(self) -> None:
        if (
            not self._uvm_weight_prefetch_plan_loaded
            and self.uvm_weight_prefetch_plan_file
            and os.path.isfile(self.uvm_weight_prefetch_plan_file)
        ):
            self.uvm_weight_prefetch_plan_keys = _parse_plan_expert_keys(
                self.uvm_weight_prefetch_plan_file,
                "prefetch_plan",
            )
            self._uvm_weight_prefetch_plan_loaded = True
        if (
            not self._uvm_weight_offload_plan_loaded
            and self.uvm_weight_offload_plan_file
            and os.path.isfile(self.uvm_weight_offload_plan_file)
        ):
            self.uvm_weight_offload_plan_keys = _parse_plan_expert_keys(
                self.uvm_weight_offload_plan_file,
                "offload_plan",
            )
            self._uvm_weight_offload_plan_loaded = True
        if (
            not self._uvm_init_weight_offload_plan_loaded
            and self.uvm_init_weight_offload_plan_file
            and os.path.isfile(self.uvm_init_weight_offload_plan_file)
        ):
            self.uvm_init_weight_offload_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "offload_plan",
            )
            self.uvm_init_weight_hot_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "hot_plan",
            ) | (
                _parse_plan_expert_keys(
                    self.uvm_init_weight_offload_plan_file,
                    "prefetch_plan",
                )
                - self.uvm_init_weight_offload_plan_keys
            )
            self.uvm_init_weight_warm_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "warm_plan",
            )
            self.uvm_init_weight_plan_policy = _uvm_weight_plan_policy(
                self.uvm_init_weight_offload_plan_file,
            )
            self._uvm_init_weight_offload_plan_loaded = True

    def _uvm_expert_weight_slices(
        self,
        local_expert_id: int,
        target_roles: set[str] | None = None,
    ) -> list[tuple[str, int, int]]:
        slices: list[tuple[str, int, int]] = []
        roles = target_roles or self.uvm_weight_prefetch_target_roles
        for attr_name, role in (
            ("w13_weight", "moe_gate_up"),
            ("w2_weight", "moe_down"),
        ):
            if role not in roles:
                continue
            tensor = getattr(self, attr_name, None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            if tensor.ndim == 0 or local_expert_id >= int(tensor.shape[0]):
                continue
            total_bytes = tensor.numel() * tensor.element_size()
            if int(tensor.shape[0]) <= 0 or total_bytes <= 0:
                continue
            slice_bytes = total_bytes // int(tensor.shape[0])
            if slice_bytes <= 0:
                continue
            ptr = tensor.data_ptr() + local_expert_id * slice_bytes
            slices.append((role, ptr, slice_bytes))
        return slices

    def _uvm_stage_m_expert_weight_slices(
        self,
        *,
        layer_id: int,
        global_expert_id: int,
        local_expert_id: int,
    ) -> list[tuple[str, int, int, dict[str, object] | None]]:
        """Resolve Stage-M slices from the logical registry when possible."""
        resolved: list[tuple[str, int, int, dict[str, object] | None]] = []
        missing_roles: list[str] = []
        roles = self.uvm_init_weight_offload_target_roles

        for role in ("moe_gate_up", "moe_down"):
            if role not in roles:
                continue
            region_id = self._uvm_expert_weight_region_id(
                layer_id,
                global_expert_id,
                role,
            )
            existing = get_uvm_residency_region_snapshot(region_id)
            if existing is None:
                missing_roles.append(role)
                continue
            ptr = int(existing.get("ptr") or 0)
            size_bytes = int(existing.get("bytes") or 0)
            if ptr <= 0 or size_bytes <= 0:
                raise RuntimeError(
                    "Stage-M logical weight region has invalid geometry: "
                    f"region_id={region_id}, ptr={ptr}, bytes={size_bytes}"
                )
            resolved.append((role, ptr, size_bytes, existing))

        if not missing_roles:
            return resolved
        if self.uvm_init_weight_register_all_local_experts:
            raise RuntimeError(
                "Stage-M all-local logical preregistration is incomplete: "
                f"layer_id={layer_id}, global_expert_id={global_expert_id}, "
                f"missing_roles={','.join(missing_roles)}"
            )

        tensor_slices = {
            role: (ptr, size_bytes)
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                set(missing_roles),
            )
        }
        for role in missing_roles:
            geometry = tensor_slices.get(role)
            if geometry is None:
                continue
            ptr, size_bytes = geometry
            resolved.append((role, ptr, size_bytes, None))

        role_order = {"moe_gate_up": 0, "moe_down": 1}
        resolved.sort(key=lambda item: role_order.get(item[0], len(role_order)))
        return resolved

    def _uvm_expert_weight_region_id(
        self,
        layer_id: int | None,
        global_expert_id: int,
        role: str,
    ) -> str:
        layer_label = "unknown" if layer_id is None else str(layer_id)
        return f"weights:{layer_label}:{global_expert_id}:{role}"

    def _uvm_expert_weight_envelope_region_id(
        self,
        layer_id: int | None,
        role: str,
        envelope_base: int,
        envelope_bytes: int,
    ) -> str:
        layer_label = "unknown" if layer_id is None else str(layer_id)
        if int(envelope_bytes) == _UVM_EXPERT_2MB_ENVELOPE_BYTES:
            envelope_kind = "uvm2m"
        else:
            envelope_kind = f"uvmchunk:{int(envelope_bytes):x}"
        return (
            f"weights:{layer_label}:{envelope_kind}:{role}:"
            f"{int(envelope_base):x}:{int(envelope_bytes):x}"
        )

    def _uvm_expert_envelope_stage(self) -> str:
        if int(self.uvm_expert_envelope_bytes) == _UVM_EXPERT_2MB_ENVELOPE_BYTES:
            return "uvm_2mb_envelope"
        return "uvm_chunk_envelope"

    def _uvm_expert_envelope_skip_reason(self) -> str:
        if int(self.uvm_expert_envelope_bytes) == _UVM_EXPERT_2MB_ENVELOPE_BYTES:
            return "logical_slice_covered_by_2mb_envelope"
        return (
            f"logical_slice_covered_by_{int(self.uvm_expert_envelope_bytes)}b_envelope"
        )

    def _uvm_expert_release_uses_subchunk_64k(self) -> bool:
        return (
            bool(self.uvm_expert_envelope_enable)
            and int(self.uvm_expert_envelope_bytes) == _UVM_EXPERT_512K_ENVELOPE_BYTES
        )

    def _uvm_role_tensor_bounds(self, role: str) -> tuple[int, int] | None:
        attr_name = None
        if role == "moe_gate_up":
            attr_name = "w13_weight"
        elif role == "moe_down":
            attr_name = "w2_weight"
        if attr_name is None:
            return None
        tensor = getattr(self, attr_name, None)
        if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
            return None
        total_bytes = int(tensor.numel()) * int(tensor.element_size())
        if total_bytes <= 0:
            return None
        base = int(tensor.data_ptr())
        return base, base + total_bytes

    def _uvm_build_expert_chunk_envelope_records(
        self,
        *,
        source_region_id: str,
        layer_id: int | None,
        global_expert_id: int,
        local_expert_id: int,
        role: str,
        ptr: int,
        size_bytes: int,
        metadata: dict[str, object],
        state: str,
        source_action: str,
    ) -> list[dict[str, object]]:
        if not self.uvm_expert_envelope_enable:
            return []
        block_bytes = int(self.uvm_expert_envelope_bytes)
        if block_bytes <= 0 or int(ptr) <= 0 or int(size_bytes) <= 0:
            return []

        slice_start = int(ptr)
        slice_end = slice_start + int(size_bytes)
        if self.uvm_expert_envelope_strict_logical_alignment and (
            slice_start % block_bytes != 0 or int(size_bytes) % block_bytes != 0
        ):
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "action": "expert_chunk_envelope_unaligned_logical_slice",
                    "source_action": source_action,
                    "source_region_id": source_region_id,
                    "role": role,
                    "global_expert_id": global_expert_id,
                    "local_expert_id": local_expert_id,
                    "ptr": f"0x{int(ptr):x}",
                    "bytes": int(size_bytes),
                    "envelope_bytes": block_bytes,
                    "strict_logical_alignment": True,
                }
            )
            return []
        envelope_start = _align_down(slice_start, block_bytes)
        envelope_end = _align_up(slice_end, block_bytes)
        tensor_bounds = self._uvm_role_tensor_bounds(role)
        if tensor_bounds is not None:
            tensor_base, tensor_end = tensor_bounds
            # The release path requires complete envelopes. If a rounded
            # envelope falls outside the tensor allocation, skip it instead of
            # registering a partial range that the driver will reject.
            if envelope_start < tensor_base or envelope_end > tensor_end:
                self._append_uvm_init_weight_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "action": "expert_chunk_envelope_out_of_tensor_bounds",
                        "source_action": source_action,
                        "source_region_id": source_region_id,
                        "role": role,
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "ptr": f"0x{int(ptr):x}",
                        "bytes": int(size_bytes),
                        "envelope_base": f"0x{envelope_start:x}",
                        "envelope_bytes": envelope_end - envelope_start,
                        "envelope_granularity_bytes": block_bytes,
                        "tensor_base": f"0x{tensor_base:x}",
                        "tensor_bytes": tensor_end - tensor_base,
                    }
                )
                return []

        records: list[dict[str, object]] = []
        for envelope_base in range(envelope_start, envelope_end, block_bytes):
            envelope_id = self._uvm_expert_weight_envelope_region_id(
                layer_id,
                role,
                envelope_base,
                block_bytes,
            )
            envelope_metadata = {
                **metadata,
                "source": source_action,
                "stage": self._uvm_expert_envelope_stage(),
                "expert_envelope": True,
                "driver_tag_skip": False,
                "driver_tag_skip_reason": None,
                "logical_region_id": source_region_id,
                "logical_ptr": int(ptr),
                "logical_bytes": int(size_bytes),
                "envelope_region_id": envelope_id,
                "envelope_base": int(envelope_base),
                "envelope_bytes": int(block_bytes),
                "envelope_granularity_bytes": block_bytes,
                "envelope_policy": (
                    "2mb_aligned"
                    if block_bytes == _UVM_EXPERT_2MB_ENVELOPE_BYTES
                    else "chunk_aligned"
                ),
                "envelope_role": role,
                "envelope_global_expert_id": int(global_expert_id),
                "envelope_local_expert_id": int(local_expert_id),
                "driver_release_subchunk_64k": (
                    block_bytes == _UVM_EXPERT_512K_ENVELOPE_BYTES
                ),
            }
            existing = get_uvm_residency_region_snapshot(envelope_id)
            if existing is not None and bool(existing.get("driver_registered")):
                # The chunk envelope has a stable driver identity across
                # Stage-M release and runtime restore. Keep that registration
                # instead of forcing a CPU_OFFLOADED_RESERVED re-register that
                # can collide with the same VA range in the driver tree.
                envelope_metadata["driver_registered"] = True
                envelope_metadata["driver_register_status"] = existing.get(
                    "driver_register_status"
                )
                envelope_metadata["driver_cpu_offloaded_reserved_registered"] = True
            registered = register_uvm_residency_region(
                pool="weights",
                region_id=envelope_id,
                ptr=int(envelope_base),
                bytes=int(block_bytes),
                state=(str(existing.get("state")) if existing is not None else state),
                evictable=True,
                active=False,
                metadata=envelope_metadata,
            )
            driver_registered = bool(getattr(registered, "driver_registered", False))
            driver_register_status = getattr(
                registered,
                "driver_register_status",
                None,
            )
            envelope_metadata = {
                **envelope_metadata,
                "driver_registered": driver_registered,
                "driver_register_status": driver_register_status,
            }
            records.append(
                {
                    "region_id": envelope_id,
                    "driver_region_id": int(
                        getattr(registered, "driver_region_id", 0) or 0
                    ),
                    "ptr": int(envelope_base),
                    "bytes": int(block_bytes),
                    "metadata": envelope_metadata,
                    "driver_registered": driver_registered,
                    "driver_register_status": driver_register_status,
                }
            )
        return records

    @staticmethod
    def _uvm_expert_envelope_cache_key(
        layer_id: int | None,
        global_expert_id: int,
        role: str,
    ) -> tuple[int | None, int, str]:
        return (layer_id, int(global_expert_id), str(role))

    def _uvm_cached_expert_envelope_items_match(
        self,
        items: list[ReservedRegionItem],
        *,
        source_region_id: str,
        layer_id: int | None,
        global_expert_id: int,
        local_expert_id: int,
        role: str,
        ptr: int,
        size_bytes: int,
    ) -> bool:
        if not items:
            return False
        block_bytes = int(self.uvm_expert_envelope_bytes)
        try:
            return all(
                item.layer_id == layer_id
                and int(item.global_expert_id) == int(global_expert_id)
                and int(item.local_expert_id) == int(local_expert_id)
                and item.role == role
                and str(item.metadata.get("logical_region_id") or "")
                == source_region_id
                and int(item.metadata.get("logical_ptr") or 0) == int(ptr)
                and int(item.metadata.get("logical_bytes") or 0) == int(size_bytes)
                and int(item.metadata.get("envelope_granularity_bytes") or 0)
                == block_bytes
                for item in items
            )
        except (TypeError, ValueError):
            return False

    def _uvm_bind_cached_expert_envelope_items(
        self,
        items: list[ReservedRegionItem],
        *,
        source_region_id: str,
        logical_driver_region_id: int,
        ptr: int,
        size_bytes: int,
        metadata: dict[str, object],
        source_action: str,
    ) -> list[ReservedRegionItem]:
        block_bytes = int(self.uvm_expert_envelope_bytes)
        bound_items: list[ReservedRegionItem] = []
        for item in items:
            bound_items.append(
                ReservedRegionItem(
                    region_id=item.region_id,
                    driver_region_id=int(item.driver_region_id),
                    ptr=int(item.ptr),
                    bytes=int(item.bytes),
                    layer_id=item.layer_id,
                    global_expert_id=int(item.global_expert_id),
                    local_expert_id=int(item.local_expert_id),
                    role=item.role,
                    metadata={
                        **item.metadata,
                        **metadata,
                        "source": source_action,
                        "stage": self._uvm_expert_envelope_stage(),
                        "driver_tag_skip": False,
                        "driver_tag_skip_reason": None,
                        "driver_registered": bool(
                            item.metadata.get("driver_registered")
                        ),
                        "driver_register_status": item.metadata.get(
                            "driver_register_status"
                        ),
                        "driver_cpu_offloaded_reserved_registered": bool(
                            item.metadata.get(
                                "driver_cpu_offloaded_reserved_registered"
                            )
                        ),
                        "logical_region_id": source_region_id,
                        "logical_driver_region_id": int(logical_driver_region_id),
                        "logical_ptr": int(ptr),
                        "logical_bytes": int(size_bytes),
                        "envelope_region_id": item.region_id,
                        "envelope_base": int(item.ptr),
                        "envelope_bytes": int(item.bytes),
                        "envelope_granularity_bytes": block_bytes,
                        "envelope_role": item.role,
                        "envelope_global_expert_id": int(item.global_expert_id),
                        "envelope_local_expert_id": int(item.local_expert_id),
                        "driver_release_subchunk_64k": (
                            block_bytes == _UVM_EXPERT_512K_ENVELOPE_BYTES
                        ),
                    },
                )
            )
        return bound_items

    def _uvm_get_or_build_expert_envelope_items(
        self,
        *,
        source_region_id: str,
        logical_driver_region_id: int,
        layer_id: int | None,
        global_expert_id: int,
        local_expert_id: int,
        role: str,
        ptr: int,
        size_bytes: int,
        metadata: dict[str, object],
        state: str,
        source_action: str,
    ) -> list[ReservedRegionItem]:
        if not self.uvm_expert_envelope_enable:
            return []

        cache = getattr(self, "_uvm_expert_envelope_region_cache", None)
        if cache is None:
            cache = {}
            self._uvm_expert_envelope_region_cache = cache
        cache_key = self._uvm_expert_envelope_cache_key(
            layer_id,
            global_expert_id,
            role,
        )
        cached_items = cache.get(cache_key)
        if cached_items is not None and not (
            self._uvm_cached_expert_envelope_items_match(
                cached_items,
                source_region_id=source_region_id,
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                role=role,
                ptr=ptr,
                size_bytes=size_bytes,
            )
        ):
            cache.pop(cache_key, None)
            cached_items = None

        if cached_items is None:
            envelope_records = self._uvm_build_expert_chunk_envelope_records(
                source_region_id=source_region_id,
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
                role=role,
                ptr=ptr,
                size_bytes=size_bytes,
                metadata=metadata,
                state=state,
                source_action=source_action,
            )
            cached_items = [
                ReservedRegionItem(
                    region_id=str(record["region_id"]),
                    driver_region_id=int(record.get("driver_region_id") or 0),
                    ptr=int(record["ptr"]),
                    bytes=int(record["bytes"]),
                    layer_id=layer_id,
                    global_expert_id=int(global_expert_id),
                    local_expert_id=int(local_expert_id),
                    role=role,
                    metadata=dict(
                        cast(dict[str, object], record.get("metadata") or {})
                    ),
                )
                for record in envelope_records
            ]
            if cached_items:
                cache[cache_key] = cached_items

        return self._uvm_bind_cached_expert_envelope_items(
            cached_items,
            source_region_id=source_region_id,
            logical_driver_region_id=logical_driver_region_id,
            ptr=ptr,
            size_bytes=size_bytes,
            metadata=metadata,
            source_action=source_action,
        )

    def _uvm_weight_expert_cache_key(
        self,
        layer_id: int | None,
        global_expert_id: int,
    ) -> tuple[str, int]:
        layer_label = self.layer_name if layer_id is None else str(layer_id)
        return (layer_label, int(global_expert_id))

    def _uvm_update_weight_expert_role_cache(
        self,
        *,
        layer_id: int | None,
        global_expert_id: int,
        role: str,
        state: str,
        estimated_gpu_resident: bool | None = None,
    ) -> None:
        if not self.uvm_weight_prefetch_expert_cache_enabled:
            return
        key = self._uvm_weight_expert_cache_key(layer_id, global_expert_id)
        role_states = self._uvm_weight_expert_role_states.setdefault(key, {})
        normalized_state = str(state or "unknown")
        if estimated_gpu_resident is True:
            normalized_state = "gpu"
        role_states[role] = normalized_state
        target_roles = self.uvm_weight_prefetch_target_roles
        if target_roles and all(
            role_states.get(target_role) in ("gpu", "prefetching_gpu")
            for target_role in target_roles
        ):
            self._uvm_weight_full_gpu_experts.add(key)
        else:
            self._uvm_weight_full_gpu_experts.discard(key)

    def _uvm_expert_weight_residency_metadata(
        self,
        *,
        layer_id: int | None,
        global_expert_id: int,
        local_expert_id: int,
        role: str,
        ptr: int,
        size_bytes: int,
        active_this_step: bool,
        source_action: str,
    ) -> dict[str, object]:
        (
            driver_capable,
            driver_capability_reason,
            driver_capability_code,
        ) = probe_driver_capable_range(ptr, size_bytes)
        return {
            "source": "stage_i",
            "stage": "stage_l1",
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "global_expert_id": global_expert_id,
            "local_expert_id": local_expert_id,
            "role": role,
            "ptr": f"0x{ptr:x}",
            "bytes": size_bytes,
            "driver_capable": driver_capable,
            "driver_capability_reason": driver_capability_reason,
            "driver_capability_code": driver_capability_code,
            "plan_hot": (
                layer_id is not None
                and (
                    (layer_id, global_expert_id) in self.uvm_weight_prefetch_plan_keys
                    or (layer_id, global_expert_id)
                    in self.uvm_init_weight_hot_plan_keys
                )
            ),
            "plan_cold": (
                layer_id is not None
                and (
                    (layer_id, global_expert_id) in self.uvm_weight_offload_plan_keys
                    or (layer_id, global_expert_id)
                    in self.uvm_init_weight_offload_plan_keys
                )
            ),
            "plan_warm": (
                layer_id is not None
                and (layer_id, global_expert_id) in self.uvm_init_weight_warm_plan_keys
            ),
            "active_this_step": active_this_step,
            "source_action": source_action,
        }

    def _uvm_expert_reserved_candidate(
        self,
        *,
        layer_id: int | None,
        global_expert_id: int,
        metadata: dict[str, object],
        existing_state: str,
    ) -> bool:
        if not self.uvm_expert_reserved_enable:
            return False
        if self._uvm_expert_in_reserved_offload_mask(layer_id, global_expert_id):
            return True
        placement = str(metadata.get("placement") or "").strip().upper()
        if placement == "CPU_OFFLOADED_RESERVED":
            return True
        if bool(metadata.get("cpu_offloaded_reserved")):
            return True
        return existing_state in ("cpu_offloaded_reserved", "reserved_cpu")

    def _uvm_make_reserved_region_item(
        self,
        *,
        item: dict[str, object],
    ) -> ReservedRegionItem | None:
        region_id = str(item["residency_region_id"])
        snapshot = get_uvm_residency_region_snapshot(region_id)
        if snapshot is None:
            return None
        try:
            driver_region_id = int(snapshot.get("driver_region_id") or 0)
            ptr = int(item["ptr"])
            size_bytes = int(item["bytes"])
            global_expert_id = int(item["global_expert_id"])
            local_expert_id = int(item["local_expert_id"])
        except (TypeError, ValueError, KeyError):
            return None
        metadata = cast(dict[str, object], item.get("residency_metadata") or {})
        return ReservedRegionItem(
            region_id=region_id,
            driver_region_id=driver_region_id,
            ptr=ptr,
            bytes=size_bytes,
            layer_id=cast(int | None, item.get("layer_id")),
            global_expert_id=global_expert_id,
            local_expert_id=local_expert_id,
            role=str(item["role"]),
            metadata={
                **metadata,
                "driver_registered": bool(snapshot.get("driver_registered")),
                "driver_register_status": snapshot.get("driver_register_status"),
                "driver_flags": int(metadata.get("driver_flags") or 0),
            },
        )

    def _uvm_expand_reserved_items_to_chunk_envelopes(
        self,
        items: list[ReservedRegionItem],
    ) -> tuple[list[ReservedRegionItem], dict[str, list[str]]]:
        if not self.uvm_expert_envelope_enable:
            return items, {}

        expanded: list[ReservedRegionItem] = []
        envelope_ids_by_logical: dict[str, list[str]] = {}
        seen_envelopes: set[str] = set()
        for item in items:
            envelope_items = self._uvm_get_or_build_expert_envelope_items(
                source_region_id=item.region_id,
                logical_driver_region_id=item.driver_region_id,
                layer_id=item.layer_id,
                global_expert_id=item.global_expert_id,
                local_expert_id=item.local_expert_id,
                role=item.role,
                ptr=item.ptr,
                size_bytes=item.bytes,
                metadata=item.metadata,
                state="cpu_offloaded_reserved",
                source_action="reserved_prefetch_chunk_envelope",
            )
            if not envelope_items:
                expanded.append(item)
                continue

            logical_envelope_ids: list[str] = []
            for envelope_item in envelope_items:
                envelope_region_id = envelope_item.region_id
                logical_envelope_ids.append(envelope_region_id)
                if envelope_region_id in seen_envelopes:
                    continue
                seen_envelopes.add(envelope_region_id)
                expanded.append(envelope_item)
            envelope_ids_by_logical[item.region_id] = logical_envelope_ids

        return expanded, envelope_ids_by_logical

    @staticmethod
    def _uvm_reduce_envelope_prefetch_result(
        result: dict[str, object],
        envelope_ids_by_logical: dict[str, list[str]],
    ) -> dict[str, object]:
        if not envelope_ids_by_logical:
            return result

        envelope_success = {
            str(region_id) for region_id in (result.get("successful_region_ids") or [])
        }
        envelope_failed = {
            str(region_id) for region_id in (result.get("failed_region_ids") or [])
        }
        logical_success: list[str] = []
        logical_failed: list[str] = []
        for logical_region_id, envelope_ids in envelope_ids_by_logical.items():
            if envelope_ids and all(
                envelope_id in envelope_success for envelope_id in envelope_ids
            ):
                logical_success.append(logical_region_id)
            elif envelope_ids:
                logical_failed.append(logical_region_id)

        reduced = dict(result)
        reduced["envelope_successful_region_ids"] = sorted(envelope_success)
        reduced["envelope_failed_region_ids"] = sorted(envelope_failed)
        reduced["envelope_ids_by_logical_region"] = {
            key: list(value) for key, value in envelope_ids_by_logical.items()
        }
        reduced["successful_region_ids"] = sorted(set(logical_success))
        reduced["failed_region_ids"] = sorted(set(logical_failed))
        reduced["success_count"] = len(set(logical_success))
        reduced["failure_count"] = len(set(logical_failed))
        flattened_envelope_ids = [
            envelope_id
            for envelope_ids in envelope_ids_by_logical.values()
            for envelope_id in envelope_ids
        ]
        reduced["envelope_mode"] = (
            "2mb"
            if flattened_envelope_ids
            and all(":uvm2m:" in envelope_id for envelope_id in flattened_envelope_ids)
            else "chunk"
        )
        return reduced

    def _run_uvm_expert_reserved_prefetch_batch(
        self,
        *,
        batch_prefetch_items: list[dict[str, object]],
        device: int,
        step: int,
        fail_fast: bool | None = None,
        async_: bool | None = None,
        sync_before_compute: bool | None = None,
        before_driver_ioctl: Callable[[], None] | None = None,
        after_driver_ioctl: Callable[[], None] | None = None,
    ) -> dict[str, object]:
        candidate_items: list[dict[str, object]] = []
        skipped_items: list[dict[str, object]] = []
        selected_experts: set[int] = set()
        for item in batch_prefetch_items:
            metadata = cast(dict[str, object], item.get("residency_metadata") or {})
            global_expert_id = int(item["global_expert_id"])
            existing_state = str(metadata.get("existing_state") or "")
            if not self._uvm_expert_reserved_candidate(
                layer_id=cast(int | None, item.get("layer_id")),
                global_expert_id=global_expert_id,
                metadata=metadata,
                existing_state=existing_state,
            ):
                skipped_items.append(item)
                continue
            selected_experts.add(global_expert_id)
            candidate_items.append(item)

        reserved_items: list[ReservedRegionItem] = []
        malformed_records = 0
        for item in candidate_items:
            region_item = self._uvm_make_reserved_region_item(item=item)
            if region_item is None:
                malformed_records += 1
                continue
            reserved_items.append(region_item)

        executor_items, envelope_ids_by_logical = (
            self._uvm_expand_reserved_items_to_chunk_envelopes(reserved_items)
        )

        executor = get_expert_reserved_regions_executor(
            device=device,
            pool_id=self.uvm_expert_reserved_pool_id,
            slot_count=self.uvm_expert_reserved_slot_count,
            slot_bytes=self.uvm_expert_reserved_slot_bytes,
            strict_footprint=self.uvm_expert_reserved_strict_footprint,
            async_=self.uvm_expert_reserved_async,
            wait_on_reuse=self.uvm_expert_reserved_wait_on_reuse,
            sync_before_compute=self.uvm_expert_reserved_sync_before_compute,
            cleanup_synchronize=self.uvm_expert_reserved_cleanup_synchronize,
        )
        result = executor.prefetch_for_layer(
            layer_name=self.layer_name,
            step=step,
            items=executor_items,
            async_=async_,
            sync_before_compute=sync_before_compute,
            before_driver_ioctl=before_driver_ioctl,
            after_driver_ioctl=after_driver_ioctl,
        )
        result = self._uvm_reduce_envelope_prefetch_result(
            result,
            envelope_ids_by_logical,
        )
        success_ids = set(result.get("successful_region_ids") or [])
        failed_ids = set(result.get("failed_region_ids") or [])
        if success_ids:
            self._uvm_expert_reserved_active_steps.add(step)

        for item in candidate_items:
            region_id = str(item["residency_region_id"])
            success = region_id in success_ids
            failed = region_id in failed_ids
            metadata = cast(dict[str, object], item.get("residency_metadata") or {})
            trace_record = {
                **{
                    key: value
                    for key, value in item.items()
                    if key != "residency_metadata"
                },
                "action": (
                    "reserved_prefetch_issued"
                    if success
                    else "reserved_prefetch_failed"
                    if failed
                    else "reserved_prefetch_skipped"
                ),
                "ptr": f"0x{int(item['ptr']):x}",
                "success": success,
                "executor": "expert_reserved_regions_async",
                "requested_executor": self.uvm_weight_executor,
                "driver_result": result.get("driver_result"),
                "reserved_executor_result": {
                    key: value
                    for key, value in result.items()
                    if key != "driver_result"
                },
            }
            self._append_uvm_weight_prefetch_trace(trace_record)
            if not success:
                should_fail_fast = (
                    self.uvm_expert_reserved_fail_fast
                    if fail_fast is None
                    else bool(fail_fast)
                )
                if should_fail_fast:
                    raise RuntimeError(
                        "reserved expert prefetch failed before MoE compute: "
                        f"layer={self.layer_name} step={step} "
                        f"region_id={region_id} role={item.get('role')} "
                        f"expert={item.get('global_expert_id')} "
                        f"result={result!r}"
                    )
                continue
            mark_uvm_residency_state(
                region_id=region_id,
                state="prefetching_gpu",
                metadata={
                    **metadata,
                    "executor": "expert_reserved_regions_async",
                    "placement": "CPU_OFFLOADED_RESERVED",
                    "cpu_offloaded_reserved": True,
                },
            )
            self._uvm_update_weight_expert_role_cache(
                layer_id=cast(int | None, item.get("layer_id")),
                global_expert_id=int(item["global_expert_id"]),
                role=str(item["role"]),
                state="prefetching_gpu",
                estimated_gpu_resident=True,
            )
            self._record_uvm_weight_prefetch_issue(
                region_id=region_id,
                layer_id=cast(int | None, item.get("layer_id")),
                global_expert_id=int(item["global_expert_id"]),
                local_expert_id=int(item["local_expert_id"]),
                role=str(item["role"]),
                ptr=int(item["ptr"]),
                size_bytes=int(item["bytes"]),
                device=device,
                executor="expert_reserved_regions_async",
            )

        for item in skipped_items:
            self._append_uvm_weight_prefetch_trace(
                {
                    **{
                        key: value
                        for key, value in item.items()
                        if key != "residency_metadata"
                    },
                    "action": "reserved_prefetch_not_offloaded",
                    "ptr": f"0x{int(item['ptr']):x}",
                    "success": False,
                    "executor": "expert_reserved_regions_async",
                    "requested_executor": self.uvm_weight_executor,
                }
            )

        return {
            "reserved_candidate_records": len(candidate_items),
            "reserved_skipped_records": len(skipped_items),
            "reserved_malformed_records": malformed_records,
            "reserved_success_records": len(success_ids),
            "reserved_failed_records": len(failed_ids),
            "reserved_selected_experts": sorted(selected_experts),
            "reserved_success_region_ids": sorted(success_ids),
            "reserved_failed_region_ids": sorted(failed_ids),
            "reserved_expected_physical_region_ids": sorted(
                {
                    envelope_id
                    for envelope_ids in envelope_ids_by_logical.values()
                    for envelope_id in envelope_ids
                }
            ),
            "reserved_success_physical_region_ids": sorted(
                result.get("envelope_successful_region_ids") or []
            ),
            "reserved_assigned_physical_region_ids": sorted(
                {
                    str(record.get("region_id"))
                    for record in (result.get("assigned_slots") or [])
                    if record.get("region_id") is not None
                }
            ),
            "reserved_driver_requested_bytes": int(
                (result.get("driver_result") or {}).get("requested_bytes", 0) or 0
            ),
            "reserved_driver_accepted_bytes": int(
                (result.get("driver_result") or {}).get("accepted_bytes", 0) or 0
            ),
            "reserved_driver_success_count": int(
                (result.get("driver_result") or {}).get("success_count", 0) or 0
            ),
            "reserved_driver_failure_count": int(
                (result.get("driver_result") or {}).get("failure_count", 0) or 0
            ),
            "reserved_driver_ordinary_resident_bytes": int(
                (result.get("driver_result") or {}).get("ordinary_resident_bytes", 0)
                or 0
            ),
            "reserved_driver_footprint_contract_broken": int(
                (result.get("driver_result") or {}).get("footprint_contract_broken", 0)
                or 0
            ),
            "reserved_executor_result": {
                key: value for key, value in result.items() if key != "driver_result"
            },
        }

    def _maybe_cleanup_uvm_expert_reserved_slots(self, step: int | None = None) -> None:
        if not self.uvm_expert_reserved_enable:
            return
        if not self.uvm_expert_reserved_cleanup_on_layer_done:
            return
        if step is None:
            latest_cleanup_step = (
                self._uvm_moe_trace_step
                - 1
                - self.uvm_expert_reserved_cleanup_delay_steps
            )
            cleanup_steps = [
                active_step
                for active_step in sorted(self._uvm_expert_reserved_active_steps)
                if active_step <= latest_cleanup_step
            ]
        else:
            cleanup_steps = [int(step)]

        if not cleanup_steps:
            return

        for cleanup_step in cleanup_steps:
            if cleanup_step not in self._uvm_expert_reserved_active_steps:
                continue
            cleanup_start_ns = time.perf_counter_ns()
            self._append_uvm_reserved_debug_trace(
                "reserved_cleanup_wrapper_enter",
                cleanup_step=cleanup_step,
                cleanup_delay_steps=self.uvm_expert_reserved_cleanup_delay_steps,
                cleanup_lag_steps=self._uvm_moe_trace_step - cleanup_step,
                active_steps=sorted(self._uvm_expert_reserved_active_steps),
            )
            try:
                device = (
                    int(torch.cuda.current_device())
                    if self.uvm_weight_prefetch_device < 0
                    else self.uvm_weight_prefetch_device
                )
                executor = get_expert_reserved_regions_executor(
                    device=device,
                    pool_id=self.uvm_expert_reserved_pool_id,
                    slot_count=self.uvm_expert_reserved_slot_count,
                    slot_bytes=self.uvm_expert_reserved_slot_bytes,
                    strict_footprint=self.uvm_expert_reserved_strict_footprint,
                    async_=self.uvm_expert_reserved_async,
                    wait_on_reuse=self.uvm_expert_reserved_wait_on_reuse,
                    sync_before_compute=self.uvm_expert_reserved_sync_before_compute,
                    cleanup_synchronize=self.uvm_expert_reserved_cleanup_synchronize,
                )
                result = executor.cleanup_layer(
                    layer_name=self.layer_name,
                    step=cleanup_step,
                    record_event=True,
                    invalidate=self.uvm_expert_reserved_invalidate_on_layer_done,
                )
                for region in result.get("regions") or []:
                    try:
                        layer_id = cast(int | None, region.get("layer_id"))
                        global_expert_id = int(region["global_expert_id"])
                        role = str(region["role"])
                    except (TypeError, ValueError, KeyError, AttributeError):
                        continue
                    self._uvm_update_weight_expert_role_cache(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        role=role,
                        state="cpu",
                        estimated_gpu_resident=False,
                    )
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": cleanup_step,
                        "current_step": self._uvm_moe_trace_step,
                        "cleanup_delay_steps": (
                            self.uvm_expert_reserved_cleanup_delay_steps
                        ),
                        "mode": "reserved_cleanup",
                        "prefetch_source": "reserved_fast_gate",
                        "action": "reserved_layer_done_cleanup",
                        "executor": "expert_reserved_regions_async",
                        "success": bool(result.get("ok")),
                        "cleanup_result": result,
                    }
                )
                self._append_uvm_reserved_debug_trace(
                    "reserved_cleanup_wrapper_exit",
                    cleanup_step=cleanup_step,
                    success=bool(result.get("ok")),
                    elapsed_us=(time.perf_counter_ns() - cleanup_start_ns) / 1000.0,
                )
            except Exception:
                logger.debug("Failed to cleanup reserved expert slots", exc_info=True)
                self._append_uvm_reserved_debug_trace(
                    "reserved_cleanup_wrapper_exception",
                    cleanup_step=cleanup_step,
                    elapsed_us=(time.perf_counter_ns() - cleanup_start_ns) / 1000.0,
                )
            finally:
                self._uvm_expert_reserved_active_steps.discard(cleanup_step)

    def _uvm_driver_capability_trace_fields(
        self, metadata: dict[str, object]
    ) -> dict[str, object]:
        return {
            "driver_capable": bool(metadata.get("driver_capable", False)),
            "driver_capability_reason": str(
                metadata.get("driver_capability_reason") or ""
            ),
            "driver_capability_code": int(metadata.get("driver_capability_code") or 0),
        }

    def _record_uvm_weight_prefetch_issue(
        self,
        *,
        region_id: str,
        layer_id: int | None,
        global_expert_id: int,
        local_expert_id: int,
        role: str,
        ptr: int,
        size_bytes: int,
        device: int,
        executor: str | None = None,
    ) -> None:
        self._uvm_weight_prefetch_issue_info[region_id] = {
            "step": self._uvm_moe_trace_step,
            "timestamp": time.time(),
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "global_expert_id": global_expert_id,
            "local_expert_id": local_expert_id,
            "role": role,
            "ptr": ptr,
            "bytes": size_bytes,
            "device": device,
            "executor": executor or self.uvm_weight_executor,
            "prefetch_source": "reserved_fast_gate",
        }

    def _probe_uvm_weight_prefetch_hit_before_access(
        self,
        *,
        layer_id: int | None,
        active_expert_pairs: list[tuple[int, int]],
        device: int,
    ) -> dict[str, object]:
        if not self.uvm_weight_prefetch_hit_trace_enabled:
            return {}

        stats: dict[str, object] = {
            "prefetch_access_probe_records": 1,
            "prefetch_access_active_expert_records": len(active_expert_pairs),
        }
        probe_records = 0
        before_access_candidates = 0
        hit_estimated = 0
        miss_estimated = 0
        same_step_access = 0
        no_prior_prefetch = 0
        lag_sum = 0
        lag_min: int | None = None
        lag_max = 0
        driver_ranges: list[tuple[int, int]] = []
        sample_records: list[dict[str, object]] = []
        max_regions = self.uvm_weight_prefetch_hit_max_regions_per_step

        for global_expert_id, local_expert_id in active_expert_pairs:
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_weight_prefetch_target_roles,
            ):
                region_id = self._uvm_expert_weight_region_id(
                    layer_id,
                    global_expert_id,
                    role,
                )
                probe_records += 1
                issue_info = self._uvm_weight_prefetch_issue_info.get(region_id)
                if issue_info is None:
                    no_prior_prefetch += 1
                    continue
                issue_step = issue_info.get("step")
                if not isinstance(issue_step, int):
                    no_prior_prefetch += 1
                    continue
                if issue_step >= self._uvm_moe_trace_step:
                    same_step_access += 1
                    continue

                before_access_candidates += 1
                lag = self._uvm_moe_trace_step - issue_step
                lag_sum += lag
                lag_min = lag if lag_min is None else min(lag_min, lag)
                lag_max = max(lag_max, lag)
                snapshot = get_uvm_residency_region_snapshot(region_id)
                state = (
                    str(snapshot.get("state"))
                    if snapshot is not None
                    else self.uvm_residency_weight_initial_state
                )
                estimated_gpu = bool(
                    snapshot is not None and snapshot.get("estimated_gpu_resident")
                )
                estimated_hit = estimated_gpu or state in (
                    "gpu",
                    "prefetching_gpu",
                )
                if estimated_hit:
                    hit_estimated += 1
                else:
                    miss_estimated += 1
                if self.uvm_weight_prefetch_hit_driver_query_enabled and (
                    max_regions == 0 or len(driver_ranges) < max_regions
                ):
                    driver_ranges.append((ptr, size_bytes))
                if len(sample_records) < 8:
                    sample_records.append(
                        {
                            "residency_region_id": region_id,
                            "issue_step": issue_step,
                            "access_step": self._uvm_moe_trace_step,
                            "step_lag": lag,
                            "state": state,
                            "estimated_gpu_resident": estimated_gpu,
                            "estimated_hit": estimated_hit,
                            "bytes": size_bytes,
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "role": role,
                        }
                    )

        stats.update(
            {
                "prefetch_access_region_records": probe_records,
                "prefetch_before_access_candidate_records": (before_access_candidates),
                "prefetch_hit_before_access_estimated_records": hit_estimated,
                "prefetch_miss_before_access_estimated_records": miss_estimated,
                "prefetch_same_step_access_records": same_step_access,
                "prefetch_access_no_prior_prefetch_records": no_prior_prefetch,
                "prefetch_hit_step_lag_sum": lag_sum,
                "prefetch_hit_step_lag_min": lag_min or 0,
                "prefetch_hit_step_lag_max": lag_max,
            }
        )

        if driver_ranges:
            try:
                driver_result = driver_batch_query_ranges_residency_bytes(
                    driver_ranges,
                    device=device,
                )
            except Exception:
                driver_result = {
                    "ok": False,
                    "status": -1,
                    "queried_bytes": 0,
                    "gpu_bytes": 0,
                    "cpu_bytes": 0,
                    "any_bytes": 0,
                    "unknown_bytes": 0,
                }
                logger.debug(
                    "Failed to query prefetch hit-before-access residency",
                    exc_info=True,
                )
            queried_bytes = int(driver_result.get("queried_bytes") or 0)
            gpu_bytes = int(driver_result.get("gpu_bytes") or 0)
            stats.update(
                {
                    "prefetch_hit_before_access_real_query_records": 1,
                    "prefetch_hit_before_access_real_query_ok": bool(
                        driver_result.get("ok")
                    ),
                    "prefetch_hit_before_access_real_query_status": int(
                        driver_result.get("status") or -1
                    ),
                    "prefetch_hit_before_access_real_query_ranges": (
                        len(driver_ranges)
                    ),
                    "prefetch_hit_before_access_real_queried_bytes": (queried_bytes),
                    "prefetch_hit_before_access_real_gpu_bytes": gpu_bytes,
                    "prefetch_hit_before_access_real_cpu_bytes": int(
                        driver_result.get("cpu_bytes") or 0
                    ),
                    "prefetch_hit_before_access_real_any_bytes": int(
                        driver_result.get("any_bytes") or 0
                    ),
                    "prefetch_hit_before_access_real_unknown_bytes": int(
                        driver_result.get("unknown_bytes") or 0
                    ),
                    "prefetch_hit_before_access_real_gpu_ratio": (
                        (gpu_bytes / queried_bytes) if queried_bytes else 0.0
                    ),
                }
            )

        if (
            before_access_candidates
            or same_step_access
            or no_prior_prefetch
            or driver_ranges
        ):
            self._append_uvm_weight_prefetch_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "step": self._uvm_moe_trace_step,
                    "mode": self.uvm_weight_prefetch_mode,
                    "prefetch_source": "reserved_fast_gate",
                    "action": "prefetch_hit_before_access_probe",
                    **stats,
                    "sample_records": sample_records,
                }
            )
        return stats

    def _uvm_hot_residency_query_device(self) -> int:
        device = self.uvm_hot_residency_stage_device
        if device < 0:
            device = self.uvm_init_weight_hot_prefetch_device
        if device < 0:
            try:
                device = torch.cuda.current_device()
            except Exception:
                device = 0
        return device

    def _query_uvm_init_hot_plan_residency(
        self,
        *,
        marker: str,
        step: int | None = None,
        num_tokens: int | None = None,
    ) -> dict[str, int | float]:
        layer_id = self._uvm_layer_id()
        ranges: list[tuple[int, int]] = []
        hot_pairs = self._uvm_plan_local_experts(self.uvm_init_weight_hot_plan_keys)
        for _, local_expert_id in hot_pairs:
            for _, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_init_weight_offload_target_roles,
            ):
                ranges.append((ptr, size_bytes))

        device = self._uvm_hot_residency_query_device()
        try:
            result = driver_batch_query_ranges_residency_bytes(ranges, device)
        except Exception:
            logger.debug(
                "Failed to query hot plan residency for marker=%s",
                marker,
                exc_info=True,
            )
            result = {
                "queried_bytes": 0,
                "gpu_bytes": 0,
                "cpu_bytes": 0,
                "any_bytes": 0,
                "unknown_bytes": 0,
            }

        queried_bytes = int(result.get("queried_bytes") or 0)
        gpu_bytes = int(result.get("gpu_bytes") or 0)
        cpu_bytes = int(result.get("cpu_bytes") or 0)
        any_bytes = int(result.get("any_bytes") or 0)
        unknown_bytes = int(result.get("unknown_bytes") or 0)
        gpu_ratio = gpu_bytes / queried_bytes if queried_bytes else 0.0
        payload: dict[str, int | float] = {
            "hot_residency_query_records": 1,
            "hot_residency_query_ranges": len(ranges),
            "hot_residency_queried_bytes": queried_bytes,
            "hot_residency_gpu_bytes": gpu_bytes,
            "hot_residency_cpu_bytes": cpu_bytes,
            "hot_residency_any_bytes": any_bytes,
            "hot_residency_unknown_bytes": unknown_bytes,
            "hot_residency_gpu_ratio": gpu_ratio,
        }
        self._append_uvm_init_weight_trace(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "layer_id": layer_id,
                "step": step,
                "num_tokens": num_tokens,
                "action": "init_hot_residency_stage_query",
                "marker": marker,
                "hot_plan_records": len(self.uvm_init_weight_hot_plan_keys),
                "local_hot_experts": len(hot_pairs),
                "device": device,
                "requested_stage_device": self.uvm_hot_residency_stage_device,
                "requested_prefetch_device": (self.uvm_init_weight_hot_prefetch_device),
                "plan_policy": self.uvm_init_weight_plan_policy,
                **payload,
            }
        )
        return payload

    def record_uvm_init_weight_plan_residency(
        self,
        marker: str,
    ) -> dict[str, object] | None:
        if not self.uvm_init_weight_plan_residency_trace_enable:
            return None

        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None

        self._refresh_uvm_weight_action_plans()
        plan_specs = (
            ("hot", self.uvm_init_weight_hot_plan_keys),
            ("cold", self.uvm_init_weight_offload_plan_keys),
        )
        max_experts = self.uvm_init_weight_plan_residency_trace_max_experts
        device = self._uvm_hot_residency_query_device()
        summary: dict[str, object] = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "action": "init_weight_plan_residency_snapshot",
            "marker": marker,
            "device": device,
            "plan_policy": self.uvm_init_weight_plan_policy,
        }

        for plan_name, plan_keys in plan_specs:
            pairs = self._uvm_plan_local_experts(plan_keys)
            if max_experts > 0:
                pairs = pairs[:max_experts]

            ranges: list[tuple[int, int]] = []
            managed_records = 0
            non_managed_records = 0
            invalid_records = 0
            sample_records: list[dict[str, object]] = []

            for global_expert_id, local_expert_id in pairs:
                for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_init_weight_offload_target_roles,
                ):
                    capable, reason, code = probe_driver_capable_range(
                        ptr,
                        size_bytes,
                    )
                    if capable:
                        managed_records += 1
                        ranges.append((ptr, size_bytes))
                    elif ptr > 0 and size_bytes > 0:
                        non_managed_records += 1
                    else:
                        invalid_records += 1

                    if len(sample_records) < 8:
                        sample_records.append(
                            {
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "role": role,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "driver_capable": bool(capable),
                                "driver_capability_reason": reason,
                                "driver_capability_code": int(code),
                            }
                        )

            result: dict[str, object]
            if ranges:
                try:
                    result = driver_batch_query_ranges_residency_bytes(
                        ranges,
                        device,
                    )
                except Exception:
                    logger.debug(
                        "Failed to query %s plan residency for marker=%s",
                        plan_name,
                        marker,
                        exc_info=True,
                    )
                    result = {
                        "ok": False,
                        "status": -1,
                        "queried_bytes": 0,
                        "gpu_bytes": 0,
                        "cpu_bytes": 0,
                        "any_bytes": 0,
                        "unknown_bytes": 0,
                    }
            else:
                result = {
                    "ok": True,
                    "status": 0,
                    "queried_bytes": 0,
                    "gpu_bytes": 0,
                    "cpu_bytes": 0,
                    "any_bytes": 0,
                    "unknown_bytes": 0,
                }

            queried_bytes = int(result.get("queried_bytes") or 0)
            gpu_bytes = int(result.get("gpu_bytes") or 0)
            cpu_bytes = int(result.get("cpu_bytes") or 0)
            unknown_bytes = int(result.get("unknown_bytes") or 0)
            prefix = f"{plan_name}_plan"
            summary.update(
                {
                    f"{prefix}_expert_records": len(pairs),
                    f"{prefix}_range_records": len(ranges),
                    f"{prefix}_managed_records": managed_records,
                    f"{prefix}_non_managed_records": non_managed_records,
                    f"{prefix}_invalid_records": invalid_records,
                    f"{prefix}_query_ok": bool(result.get("ok")),
                    f"{prefix}_query_status": int(result.get("status") or 0),
                    f"{prefix}_queried_bytes": queried_bytes,
                    f"{prefix}_gpu_bytes": gpu_bytes,
                    f"{prefix}_cpu_bytes": cpu_bytes,
                    f"{prefix}_any_bytes": int(result.get("any_bytes") or 0),
                    f"{prefix}_unknown_bytes": unknown_bytes,
                    f"{prefix}_gpu_ratio": (
                        float(gpu_bytes) / float(queried_bytes)
                        if queried_bytes
                        else 0.0
                    ),
                    f"{prefix}_cpu_ratio": (
                        float(cpu_bytes) / float(queried_bytes)
                        if queried_bytes
                        else 0.0
                    ),
                    f"{prefix}_unknown_ratio": (
                        float(unknown_bytes) / float(queried_bytes)
                        if queried_bytes
                        else 0.0
                    ),
                    f"{prefix}_samples": sample_records,
                }
            )

        self._append_uvm_init_weight_trace(summary)
        return summary

    def _append_uvm_weight_prefetch_trace(self, record: dict[str, object]) -> None:
        if not self.uvm_weight_prefetch_trace_enabled:
            return
        try:
            with open(
                self.uvm_weight_prefetch_trace_file,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write UVM weight prefetch trace", exc_info=True)

    def _append_uvm_moe_address_map(self, record: dict[str, object]) -> None:
        if not self.uvm_moe_address_map_enabled:
            return
        try:
            with open(self.uvm_moe_address_map_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write UVM MoE address map", exc_info=True)

    def _uvm_tensor_address_record(
        self,
        *,
        marker: str,
        name: str,
        tensor: torch.Tensor,
        source: str,
        layer_id: int | None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        if tensor.numel() <= 0:
            return None
        try:
            ptr = int(tensor.data_ptr())
        except Exception:
            return None
        if ptr == 0:
            return None

        element_size = int(tensor.element_size())
        data_bytes = int(tensor.numel()) * element_size
        storage_ptr = ptr
        storage_bytes = data_bytes
        try:
            storage = tensor.untyped_storage()
            storage_ptr = int(storage.data_ptr())
            storage_bytes = int(storage.nbytes())
        except Exception:
            pass

        record: dict[str, object] = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "action": "moe_address_map",
            "marker": marker,
            "source": source,
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "step": self._uvm_moe_trace_step,
            "name": name,
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "is_cuda": bool(tensor.is_cuda),
            "shape": [int(dim) for dim in tensor.shape],
            "stride": [int(value) for value in tensor.stride()],
            "storage_offset": int(tensor.storage_offset()),
            "element_size": element_size,
            "numel": int(tensor.numel()),
            "ptr": f"0x{ptr:x}",
            "bytes": data_bytes,
            "end": f"0x{ptr + data_bytes:x}",
            "storage_ptr": f"0x{storage_ptr:x}",
            "storage_bytes": storage_bytes,
            "storage_end": f"0x{storage_ptr + storage_bytes:x}",
            "is_contiguous": bool(tensor.is_contiguous()),
        }
        if extra:
            record.update(extra)

        if self.uvm_moe_address_map_driver_probe and tensor.is_cuda and data_bytes > 0:
            try:
                (
                    driver_capable,
                    driver_capability_reason,
                    driver_capability_code,
                ) = probe_driver_capable_range(ptr, data_bytes)
                record.update(
                    {
                        "driver_capable": driver_capable,
                        "driver_capability_reason": driver_capability_reason,
                        "driver_capability_code": driver_capability_code,
                    }
                )
            except Exception as exc:
                record.update(
                    {
                        "driver_capable": False,
                        "driver_capability_reason": f"probe_exception:{exc!r}",
                        "driver_capability_code": -1,
                    }
                )
        return record

    def _record_uvm_moe_address_map(
        self,
        *,
        marker: str,
        extra_tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if not self.uvm_moe_address_map_enabled:
            return

        try:
            layer_id = self._uvm_layer_id()
        except Exception:
            layer_id = None

        records = 0
        seen: set[tuple[str, int, int]] = set()

        def emit(name: str, tensor: torch.Tensor, source: str) -> None:
            nonlocal records
            if not isinstance(tensor, torch.Tensor):
                return
            try:
                ptr = int(tensor.data_ptr())
                bytes_ = int(tensor.numel()) * int(tensor.element_size())
            except Exception:
                return
            key = (name, ptr, bytes_)
            if key in seen:
                return
            seen.add(key)
            record = self._uvm_tensor_address_record(
                marker=marker,
                name=name,
                tensor=tensor,
                source=source,
                layer_id=layer_id,
            )
            if record is not None:
                self._append_uvm_moe_address_map(record)
                records += 1

        for name, tensor in self.named_parameters(recurse=False):
            emit(f"parameter:{name}", tensor, "named_parameter")
        for name, tensor in self.named_buffers(recurse=False):
            emit(f"buffer:{name}", tensor, "named_buffer")

        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                emit(f"attr:{name}", value, "module_attr")

        quant_method = getattr(self, "quant_method", None)
        if quant_method is not None:
            for name, value in vars(quant_method).items():
                if isinstance(value, torch.Tensor):
                    emit(f"quant_method:{name}", value, "quant_method_attr")

        if extra_tensors:
            for name, tensor in extra_tensors.items():
                emit(f"runtime:{name}", tensor, "runtime_tensor")

        self._append_uvm_moe_address_map(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "action": "moe_address_map_summary",
                "marker": marker,
                "layer_name": self.layer_name,
                "layer_id": layer_id,
                "step": self._uvm_moe_trace_step,
                "records": records,
            }
        )

    def record_uvm_active_expert_residency_probe(
        self,
        marker: str,
    ) -> dict[str, object] | None:
        if os.environ.get(
            "VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_ENABLE", "0"
        ).strip().lower() not in ("1", "true", "yes", "on"):
            return None

        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None

        probe_file = os.environ.get("VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_PROBE_FILE", "")
        output_file = os.environ.get("VLLM_UVM_ACTIVE_EXPERT_RESIDENCY_TRACE_FILE", "")
        if not probe_file or not output_file:
            return None

        try:
            with open(probe_file, encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            logger.debug(
                "Failed to read active expert residency probe file", exc_info=True
            )
            return None

        layers = payload.get("layers") if isinstance(payload, dict) else payload
        if not isinstance(layers, dict):
            return None
        raw_experts = layers.get(str(layer_id), layers.get(layer_id))
        if not isinstance(raw_experts, list):
            return None

        active_pairs = sorted(
            {
                (int(global_expert_id), int(global_expert_id))
                for global_expert_id in raw_experts
            }
        )
        if not active_pairs:
            return None

        # Reuse the same low-level range query used by coverage tracing, but
        # force it on for this explicit no-forward probe.
        old_enabled = self.uvm_expert_reserved_coverage_residency_query
        self.uvm_expert_reserved_coverage_residency_query = True
        try:
            residency = self._query_uvm_expert_reserved_active_residency(
                layer_id=layer_id,
                all_active_expert_pairs=active_pairs,
            )
        finally:
            self.uvm_expert_reserved_coverage_residency_query = old_enabled

        record: dict[str, object] = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "action": "active_expert_residency_probe",
            "marker": marker,
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "active_global_experts": [int(global_id) for global_id, _ in active_pairs],
            **residency,
        }
        try:
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write active expert residency trace", exc_info=True)
        return record

    def record_uvm_post_offload_slot_use_probe(
        self,
        marker: str,
    ) -> dict[str, object] | None:
        if os.environ.get(
            "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_ENABLE", "0"
        ).strip().lower() not in ("1", "true", "yes", "on"):
            return None

        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None

        target_layer = _env_int(
            "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_LAYER_ID",
            default=layer_id,
        )
        if int(layer_id) != int(target_layer):
            return None

        output_file = os.environ.get(
            "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_TRACE_FILE",
            "",
        )
        if not output_file:
            return None

        expert_ids = sorted(
            _parse_int_set_env("VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_EXPERTS")
        )
        if not expert_ids:
            expert_ids = sorted(
                int(expert_id)
                for plan_layer_id, expert_id in self.uvm_init_weight_offload_plan_keys
                if int(plan_layer_id) == int(layer_id)
            )
        max_experts = _env_int(
            "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_MAX_EXPERTS",
            default=4,
        )
        if max_experts > 0:
            expert_ids = expert_ids[:max_experts]
        if not expert_ids:
            return None

        try:
            device = (
                int(torch.cuda.current_device())
                if self.uvm_weight_prefetch_device < 0
                else self.uvm_weight_prefetch_device
            )
        except Exception:
            device = 0

        items: list[ReservedRegionItem] = []
        source_ranges: list[tuple[int, int]] = []
        item_records: list[dict[str, object]] = []
        for global_expert_id in expert_ids:
            local_expert_id = int(global_expert_id)
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_init_weight_offload_target_roles,
            ):
                region_id = self._uvm_expert_weight_region_id(
                    layer_id,
                    global_expert_id,
                    role,
                )
                metadata = self._uvm_expert_weight_residency_metadata(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    local_expert_id=local_expert_id,
                    role=role,
                    ptr=ptr,
                    size_bytes=size_bytes,
                    active_this_step=False,
                    source_action="post_offload_slot_use_probe",
                )
                metadata = {
                    **metadata,
                    "source": "post_offload_slot_use_probe",
                    "stage": "post_stage_m_slot_probe",
                    "driver_tag_skip": bool(self.uvm_expert_envelope_enable),
                    "driver_tag_skip_reason": (
                        self._uvm_expert_envelope_skip_reason()
                        if self.uvm_expert_envelope_enable
                        else None
                    ),
                    "pool_class": "cold",
                    "driver_subpool": "cold",
                    "init_cold": True,
                    "protected": False,
                    "prefetchable": True,
                    "cpu_offloaded_reserved": True,
                }
                existing = get_uvm_residency_region_snapshot(region_id)
                region = register_uvm_residency_region(
                    pool="weights",
                    region_id=region_id,
                    ptr=ptr,
                    bytes=size_bytes,
                    state=(
                        str(existing.get("state"))
                        if existing is not None
                        else "cpu_offloaded_reserved"
                    ),
                    evictable=True,
                    active=False,
                    metadata=metadata,
                )
                driver_region_id = int(getattr(region, "driver_region_id", 0) or 0)
                item = ReservedRegionItem(
                    region_id=region_id,
                    driver_region_id=driver_region_id,
                    ptr=int(ptr),
                    bytes=int(size_bytes),
                    layer_id=layer_id,
                    global_expert_id=int(global_expert_id),
                    local_expert_id=int(local_expert_id),
                    role=role,
                    metadata=metadata,
                )
                items.append(item)
                source_ranges.append((int(ptr), int(size_bytes)))
                item_records.append(
                    {
                        "region_id": region_id,
                        "driver_region_id": driver_region_id,
                        "global_expert_id": int(global_expert_id),
                        "local_expert_id": int(local_expert_id),
                        "role": role,
                        "ptr": f"0x{int(ptr):x}",
                        "bytes": int(size_bytes),
                    }
                )

        residency_before: dict[str, object]
        try:
            residency_before = driver_batch_query_ranges_residency_bytes(
                source_ranges,
                device,
            )
        except Exception as exc:
            residency_before = {"ok": False, "error": repr(exc)}

        executor = get_expert_reserved_regions_executor(
            device=device,
            slot_count=_env_int(
                "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_COUNT",
                self.uvm_expert_reserved_slot_count,
            ),
            slot_bytes=_env_int(
                "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_SLOT_BYTES",
                self.uvm_expert_reserved_slot_bytes,
            ),
            strict_footprint=_env_flag_enabled(
                "VLLM_UVM_POST_OFFLOAD_SLOT_USE_PROBE_STRICT_FOOTPRINT",
                self.uvm_expert_reserved_strict_footprint,
            ),
            async_=False,
            sync_before_compute=True,
            cleanup_synchronize=True,
        )
        prefetch_result = executor.prefetch_for_layer(
            layer_name=self.layer_name,
            step=-1000 - int(layer_id),
            items=items,
        )

        residency_after: dict[str, object]
        try:
            residency_after = driver_batch_query_ranges_residency_bytes(
                source_ranges,
                device,
            )
        except Exception as exc:
            residency_after = {"ok": False, "error": repr(exc)}

        record: dict[str, object] = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "action": "post_offload_slot_use_probe",
            "marker": marker,
            "layer_name": self.layer_name,
            "layer_id": int(layer_id),
            "expert_ids": expert_ids,
            "item_count": len(items),
            "items": item_records,
            "residency_before": residency_before,
            "prefetch_result": prefetch_result,
            "residency_after": residency_after,
        }
        try:
            with open(output_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug(
                "Failed to write post-offload slot use probe trace", exc_info=True
            )
        return record

    def _query_uvm_expert_reserved_active_residency(
        self,
        *,
        layer_id: int | None,
        all_active_expert_pairs: list[tuple[int, int]],
    ) -> dict[str, object]:
        if not self.uvm_expert_reserved_coverage_residency_query or layer_id is None:
            return {}

        unique_pairs = sorted(
            {
                int(global_expert_id): int(local_expert_id)
                for global_expert_id, local_expert_id in all_active_expert_pairs
            }.items()
        )
        max_experts = self.uvm_expert_reserved_coverage_residency_max_experts
        if max_experts > 0:
            unique_pairs = unique_pairs[:max_experts]

        try:
            device = (
                int(torch.cuda.current_device())
                if self.uvm_weight_prefetch_device < 0
                else self.uvm_weight_prefetch_device
            )
        except Exception:
            device = 0

        expert_records: list[dict[str, object]] = []
        total_queried_bytes = 0
        total_gpu_bytes = 0
        total_cpu_bytes = 0
        total_unknown_bytes = 0
        query_failed_experts: list[int] = []
        non_full_gpu_experts: list[int] = []
        cpu_present_experts: list[int] = []
        unknown_present_experts: list[int] = []

        for global_expert_id, local_expert_id in unique_pairs:
            ranges: list[tuple[int, int]] = []
            roles: list[str] = []
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_weight_prefetch_target_roles,
            ):
                if ptr > 0 and size_bytes > 0:
                    ranges.append((ptr, size_bytes))
                    roles.append(role)

            if not ranges:
                expert_records.append(
                    {
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "roles": roles,
                        "range_records": 0,
                        "state": "no_ranges",
                        "query_ok": False,
                    }
                )
                query_failed_experts.append(global_expert_id)
                continue

            try:
                result = driver_batch_query_ranges_residency_bytes(
                    ranges,
                    device,
                )
            except Exception:
                logger.debug(
                    "Failed to query active expert residency for coverage",
                    exc_info=True,
                )
                result = {
                    "ok": False,
                    "status": -1,
                    "queried_bytes": 0,
                    "gpu_bytes": 0,
                    "cpu_bytes": 0,
                    "unknown_bytes": 0,
                }

            query_ok = bool(result.get("ok"))
            queried_bytes = int(result.get("queried_bytes") or 0)
            gpu_bytes = int(result.get("gpu_bytes") or 0)
            cpu_bytes = int(result.get("cpu_bytes") or 0)
            unknown_bytes = int(result.get("unknown_bytes") or 0)
            total_queried_bytes += queried_bytes
            total_gpu_bytes += gpu_bytes
            total_cpu_bytes += cpu_bytes
            total_unknown_bytes += unknown_bytes

            if not query_ok:
                state = "query_failed"
                query_failed_experts.append(global_expert_id)
            elif (
                queried_bytes > 0 and gpu_bytes == queried_bytes and unknown_bytes == 0
            ):
                state = "gpu_full"
            elif cpu_bytes > 0:
                state = "cpu_present"
                non_full_gpu_experts.append(global_expert_id)
                cpu_present_experts.append(global_expert_id)
            elif unknown_bytes > 0:
                state = "unknown_present"
                non_full_gpu_experts.append(global_expert_id)
                unknown_present_experts.append(global_expert_id)
            else:
                state = "not_full_gpu"
                non_full_gpu_experts.append(global_expert_id)

            expert_records.append(
                {
                    "global_expert_id": global_expert_id,
                    "local_expert_id": local_expert_id,
                    "roles": roles,
                    "range_records": len(ranges),
                    "query_ok": query_ok,
                    "query_status": result.get("status"),
                    "queried_bytes": queried_bytes,
                    "gpu_bytes": gpu_bytes,
                    "cpu_bytes": cpu_bytes,
                    "unknown_bytes": unknown_bytes,
                    "state": state,
                }
            )

        return {
            "active_residency_query_enabled": True,
            "active_residency_device": device,
            "active_residency_expert_records": len(expert_records),
            "active_residency_queried_bytes": total_queried_bytes,
            "active_residency_gpu_bytes": total_gpu_bytes,
            "active_residency_cpu_bytes": total_cpu_bytes,
            "active_residency_unknown_bytes": total_unknown_bytes,
            "active_residency_non_full_gpu_experts": non_full_gpu_experts,
            "active_residency_cpu_present_experts": cpu_present_experts,
            "active_residency_unknown_present_experts": unknown_present_experts,
            "active_residency_query_failed_experts": query_failed_experts,
            "active_residency_records": expert_records,
        }

    def _append_uvm_expert_reserved_coverage_trace(
        self,
        *,
        layer_id: int | None,
        all_active_expert_pairs: list[tuple[int, int]],
        requested_expert_pairs: list[tuple[int, int]],
        reserved_batch_stats: dict[str, object],
        batch_prefetch_items: list[dict[str, object]],
    ) -> None:
        if not self.uvm_expert_reserved_coverage_trace_enabled:
            return
        try:
            active_reserved_pairs = self._uvm_active_reserved_offload_expert_pairs(
                layer_id,
                all_active_expert_pairs,
            )
            requested_experts = {
                int(global_expert_id) for global_expert_id, _ in requested_expert_pairs
            }
            candidate_experts = {
                int(item["global_expert_id"]) for item in batch_prefetch_items
            }
            success_region_ids = {
                str(region_id)
                for region_id in (
                    reserved_batch_stats.get("reserved_success_region_ids") or []
                )
            }
            failed_region_ids = {
                str(region_id)
                for region_id in (
                    reserved_batch_stats.get("reserved_failed_region_ids") or []
                )
            }
            expected_region_ids_by_expert: dict[int, list[str]] = {}
            for global_expert_id, local_expert_id in active_reserved_pairs:
                expected_region_ids_by_expert[int(global_expert_id)] = [
                    self._uvm_expert_weight_region_id(
                        layer_id,
                        int(global_expert_id),
                        role,
                    )
                    for role, _, _ in self._uvm_expert_weight_slices(
                        int(local_expert_id),
                        self.uvm_weight_prefetch_target_roles,
                    )
                ]

            successful_experts: list[int] = []
            incomplete_experts: list[int] = []
            missing_experts: list[int] = []
            failed_experts: list[int] = []
            coverage_details: dict[str, dict[str, object]] = {}
            for global_expert_id, expected_region_ids in sorted(
                expected_region_ids_by_expert.items()
            ):
                success_count = sum(
                    1
                    for region_id in expected_region_ids
                    if region_id in success_region_ids
                )
                failed_count = sum(
                    1
                    for region_id in expected_region_ids
                    if region_id in failed_region_ids
                )
                requested = global_expert_id in requested_experts
                candidate = global_expert_id in candidate_experts
                if success_count == len(expected_region_ids):
                    successful_experts.append(global_expert_id)
                else:
                    incomplete_experts.append(global_expert_id)
                    if not requested or not candidate or success_count == 0:
                        missing_experts.append(global_expert_id)
                if failed_count:
                    failed_experts.append(global_expert_id)
                coverage_details[str(global_expert_id)] = {
                    "requested": requested,
                    "candidate": candidate,
                    "expected_regions": len(expected_region_ids),
                    "success_regions": success_count,
                    "failed_regions": failed_count,
                }

            active_residency = self._query_uvm_expert_reserved_active_residency(
                layer_id=layer_id,
                all_active_expert_pairs=all_active_expert_pairs,
            )
            record: dict[str, object] = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "action": "reserved_prefetch_coverage",
                "layer_name": self.layer_name,
                "layer_id": layer_id,
                "step": self._uvm_moe_trace_step,
                "active_global_experts": sorted(
                    {
                        int(global_expert_id)
                        for global_expert_id, _ in all_active_expert_pairs
                    }
                ),
                "active_reserved_experts": sorted(
                    {
                        int(global_expert_id)
                        for global_expert_id, _ in active_reserved_pairs
                    }
                ),
                "requested_reserved_experts": sorted(requested_experts),
                "candidate_reserved_experts": sorted(candidate_experts),
                "successful_reserved_experts": successful_experts,
                "incomplete_reserved_experts": incomplete_experts,
                "missing_reserved_experts": sorted(set(missing_experts)),
                "failed_reserved_experts": sorted(set(failed_experts)),
                "active_reserved_records": len(active_reserved_pairs),
                "requested_reserved_records": len(requested_expert_pairs),
                "batch_prefetch_records": len(batch_prefetch_items),
                "success_region_records": len(success_region_ids),
                "failed_region_records": len(failed_region_ids),
                "coverage_ok": not missing_experts and not failed_experts,
                "coverage_details": coverage_details,
                "reserved_candidate_records": (
                    reserved_batch_stats.get("reserved_candidate_records")
                ),
                "reserved_skipped_records": (
                    reserved_batch_stats.get("reserved_skipped_records")
                ),
                "reserved_malformed_records": (
                    reserved_batch_stats.get("reserved_malformed_records")
                ),
                **active_residency,
            }
            with open(
                self.uvm_expert_reserved_coverage_trace_file,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            logger.debug(
                "Failed to write UVM expert reserved coverage trace",
                exc_info=True,
            )

    def _append_uvm_reserved_debug_trace(
        self,
        action: str,
        **fields: object,
    ) -> None:
        self._append_uvm_moe_stream_trace(action, **fields)
        if not self.uvm_expert_reserved_enable:
            return
        if self.uvm_expert_reserved_stream_trace:
            fields = {
                **self._uvm_reserved_stream_trace_fields(),
                **fields,
            }
        self._append_uvm_weight_prefetch_trace(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "step": self._uvm_moe_trace_step,
                "mode": "reserved_debug",
                "prefetch_source": "reserved_fast_gate",
                "executor": "expert_reserved_regions_async",
                "action": action,
                **fields,
            }
        )

    def _append_uvm_moe_stream_trace(
        self,
        action: str,
        **fields: object,
    ) -> None:
        if not self.uvm_moe_stream_trace:
            return
        try:
            record = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "step": self._uvm_moe_trace_step,
                "mode": "moe_stream_trace",
                "action": action,
                **self._uvm_reserved_stream_trace_fields(),
                **fields,
            }
            trace_dir = os.path.dirname(self.uvm_moe_stream_trace_file)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(
                self.uvm_moe_stream_trace_file,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._write_uvm_moe_stream_trace_marker(record)
        except OSError:
            logger.debug("Failed to write UVM MoE stream trace", exc_info=True)

    def _write_uvm_moe_stream_trace_marker(
        self,
        record: dict[str, object],
    ) -> None:
        if (
            not self.uvm_moe_stream_trace_marker
            or self._uvm_moe_stream_trace_marker_disabled
        ):
            return
        marker_record = {
            key: record.get(key)
            for key in (
                "pid",
                "layer_name",
                "step",
                "action",
                "stage",
                "current_stream_cuda",
                "shared_experts_stream_cuda",
                "shared_experts_stream_enabled",
            )
            if key in record
        }
        try:
            marker = "vllm_moe_stream " + json.dumps(
                marker_record,
                sort_keys=True,
                separators=(",", ":"),
            )
            for path in (
                "/sys/kernel/debug/tracing/trace_marker",
                "/sys/kernel/tracing/trace_marker",
            ):
                if os.path.exists(path):
                    with open(path, "w", encoding="utf-8") as handle:
                        handle.write(marker + "\n")
                    return
        except OSError:
            self._uvm_moe_stream_trace_marker_disabled = True
            logger.debug("Failed to write UVM MoE stream trace marker", exc_info=True)

    def _uvm_reserved_stream_trace_fields(self) -> dict[str, object]:
        fields: dict[str, object] = {}
        try:
            stream = torch.cuda.current_stream()
            fields["current_stream_cuda"] = int(stream.cuda_stream)
            fields["current_stream_device"] = int(stream.device.index or 0)
        except Exception:
            fields["current_stream_cuda"] = -1
        try:
            shared_stream = self.shared_experts_stream
        except AttributeError:
            shared_stream = None
        if shared_stream is not None:
            try:
                fields["shared_experts_stream_cuda"] = int(shared_stream.cuda_stream)
                fields["shared_experts_stream_device"] = int(
                    shared_stream.device.index or 0
                )
            except Exception:
                fields["shared_experts_stream_cuda"] = -1
        else:
            fields["shared_experts_stream_cuda"] = 0
        fields["shared_experts_stream_enabled"] = shared_stream is not None
        return fields

    def _maybe_sync_uvm_reserved_debug_compute(self, stage: str) -> None:
        if not (
            self.uvm_expert_reserved_enable
            and self.uvm_expert_reserved_debug_sync_compute
        ):
            return
        device = (
            int(torch.cuda.current_device())
            if self.uvm_weight_prefetch_device < 0
            else self.uvm_weight_prefetch_device
        )
        start_ns = time.perf_counter_ns()
        self._append_uvm_reserved_debug_trace(
            "moe_compute_sync_enter",
            stage=stage,
            device=device,
        )
        torch.cuda.synchronize(device)
        self._append_uvm_reserved_debug_trace(
            "moe_compute_sync_exit",
            stage=stage,
            device=device,
            elapsed_us=(time.perf_counter_ns() - start_ns) / 1000.0,
        )

    def _append_uvm_init_weight_trace(self, record: dict[str, object]) -> None:
        try:
            trace_dir = os.path.dirname(self.uvm_init_weight_offload_trace_file)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
            with open(
                self.uvm_init_weight_offload_trace_file,
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write Stage M weight offload trace", exc_info=True)

    def _register_uvm_init_all_local_weight_regions(
        self,
        layer_id: int,
    ) -> dict[str, object]:
        if not self.uvm_init_weight_register_all_local_experts:
            return {
                "all_local_preregister_enabled": False,
                "all_local_experts": 0,
                "all_local_registered_records": 0,
            }

        expert_pairs = self._uvm_all_local_expert_pairs()
        expert_class_by_global = {
            global_expert_id: self._uvm_init_weight_expert_class(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
            )
            for global_expert_id, _ in expert_pairs
        }
        expert_counts = {
            "hot": 0,
            "warm": 0,
            "cold": 0,
        }
        for expert_class in expert_class_by_global.values():
            expert_counts[expert_class] = expert_counts.get(expert_class, 0) + 1

        registered_records = 0
        registered_bytes = 0
        records_by_class = {
            "hot": 0,
            "warm": 0,
            "cold": 0,
        }
        bytes_by_class = {
            "hot": 0,
            "warm": 0,
            "cold": 0,
        }
        for global_expert_id, local_expert_id in expert_pairs:
            expert_class = expert_class_by_global[global_expert_id]
            logical_driver_tag_skip = bool(self.uvm_expert_envelope_enable)
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_init_weight_offload_target_roles,
            ):
                region_id = self._uvm_expert_weight_region_id(
                    layer_id,
                    global_expert_id,
                    role,
                )
                base_metadata = self._uvm_expert_weight_residency_metadata(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    local_expert_id=local_expert_id,
                    role=role,
                    ptr=ptr,
                    size_bytes=size_bytes,
                    active_this_step=False,
                    source_action="stage_m_all_local_preregister",
                )
                metadata = self._uvm_init_weight_class_metadata(
                    metadata=base_metadata,
                    expert_class=expert_class,
                    driver_tag_skip=logical_driver_tag_skip,
                    cpu_offload_placement=False,
                )
                existing_region = get_uvm_residency_region_snapshot(region_id)
                state = (
                    str(existing_region.get("state"))
                    if existing_region is not None
                    else "gpu"
                )
                if existing_region is None:
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state=state,
                        evictable=expert_class != "hot",
                        active=False,
                        metadata=metadata,
                    )
                else:
                    mark_uvm_residency_state(
                        region_id=region_id,
                        state=state,
                        metadata={
                            **metadata,
                            "evictable": expert_class != "hot",
                        },
                    )
                self._uvm_update_weight_expert_role_cache(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    role=role,
                    state="gpu" if expert_class == "hot" else state,
                    estimated_gpu_resident=(expert_class == "hot"),
                )
                registered_records += 1
                registered_bytes += size_bytes
                records_by_class[expert_class] = (
                    records_by_class.get(expert_class, 0) + 1
                )
                bytes_by_class[expert_class] = (
                    bytes_by_class.get(expert_class, 0) + size_bytes
                )

        stats: dict[str, object] = {
            "all_local_preregister_enabled": True,
            "all_local_experts": len(expert_pairs),
            "all_local_registered_records": registered_records,
            "all_local_registered_bytes": registered_bytes,
            "all_local_hot_experts": expert_counts.get("hot", 0),
            "all_local_warm_experts": expert_counts.get("warm", 0),
            "all_local_cold_experts": expert_counts.get("cold", 0),
            "all_local_hot_records": records_by_class.get("hot", 0),
            "all_local_warm_records": records_by_class.get("warm", 0),
            "all_local_cold_records": records_by_class.get("cold", 0),
            "all_local_hot_bytes": bytes_by_class.get("hot", 0),
            "all_local_warm_bytes": bytes_by_class.get("warm", 0),
            "all_local_cold_bytes": bytes_by_class.get("cold", 0),
            "complete_expert_plan": self.uvm_init_weight_complete_expert_plan,
            "complete_plan_ignore_limits": (
                self.uvm_init_weight_complete_plan_ignore_limits
            ),
        }
        if registered_records:
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "layer_id": layer_id,
                    "action": "init_weight_all_local_preregister_summary",
                    **stats,
                    "plan_policy": self.uvm_init_weight_plan_policy,
                }
            )
        return stats

    def _register_uvm_init_hot_weight_regions(self, layer_id: int) -> dict[str, object]:
        """Register routing-plan hot experts as GPU/protected before offload."""
        prefetch_device = self.uvm_init_weight_hot_prefetch_device
        if prefetch_device < 0:
            try:
                prefetch_device = int(torch.cuda.current_device())
            except Exception:
                prefetch_device = 0
        hot_pairs = self._uvm_plan_local_experts(self.uvm_init_weight_hot_plan_keys)
        registered_records = 0
        prefetch_items: list[dict[str, object]] = []
        prefetch_issued_records = 0
        prefetch_issued_bytes = 0
        prefetch_rejected_records = 0
        prefetch_attempted_bytes = 0
        prefetch_requested_bytes = 0
        for global_expert_id, local_expert_id in hot_pairs:
            for (
                role,
                ptr,
                size_bytes,
                existing_region,
            ) in self._uvm_stage_m_expert_weight_slices(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
            ):
                region_id = self._uvm_expert_weight_region_id(
                    layer_id,
                    global_expert_id,
                    role,
                )
                metadata = self._uvm_expert_weight_residency_metadata(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    local_expert_id=local_expert_id,
                    role=role,
                    ptr=ptr,
                    size_bytes=size_bytes,
                    active_this_step=False,
                    source_action="stage_m_init_hot_register",
                )
                logical_driver_tag_skip = bool(self.uvm_expert_envelope_enable)
                hot_metadata = {
                    **metadata,
                    "source": "stage_m",
                    "stage": "stage_m",
                    "driver_tag_skip": logical_driver_tag_skip,
                    "driver_tag_skip_reason": (
                        self._uvm_expert_envelope_skip_reason()
                        if logical_driver_tag_skip
                        else None
                    ),
                    "pool_class": "hot",
                    "driver_subpool": "hot",
                    "expert_init_class": "hot",
                    "placement": "GPU_RESIDENT",
                    "target_placement": "GPU_RESIDENT",
                    "init_hot": True,
                    "init_cold": False,
                    "protected": True,
                    "pinned": True,
                    "prefetchable": True,
                    "priority": 100,
                    "plan_policy": self.uvm_init_weight_plan_policy,
                }
                if existing_region is None:
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state="gpu",
                        evictable=False,
                        active=False,
                        metadata=hot_metadata,
                    )
                mark_uvm_residency_state(
                    region_id=region_id,
                    state="gpu",
                    metadata=hot_metadata,
                )
                self._uvm_update_weight_expert_role_cache(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    role=role,
                    state="gpu",
                    estimated_gpu_resident=True,
                )
                registered_records += 1
                if not self.uvm_init_weight_hot_prefetch_enable:
                    continue
                prefetch_attempted_bytes += size_bytes
                if (
                    self.uvm_init_weight_hot_prefetch_max_bytes > 0
                    and prefetch_requested_bytes + size_bytes
                    > self.uvm_init_weight_hot_prefetch_max_bytes
                ):
                    prefetch_rejected_records += 1
                    self._append_uvm_init_weight_trace(
                        {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "layer_name": self.layer_name,
                            "action": "init_weight_hot_prefetch_budget_reject",
                            "role": role,
                            "residency_region_id": region_id,
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "ptr": f"0x{ptr:x}",
                            "bytes": size_bytes,
                            "requested_bytes_before": prefetch_requested_bytes,
                            "max_bytes": self.uvm_init_weight_hot_prefetch_max_bytes,
                            "plan_policy": self.uvm_init_weight_plan_policy,
                        }
                    )
                    continue
                prefetch_requested_bytes += size_bytes
                prefetch_items.append(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "role": role,
                        "residency_region_id": region_id,
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "ptr": ptr,
                        "bytes": size_bytes,
                        "device": prefetch_device,
                        "sync_prefetch": self.uvm_init_weight_hot_prefetch_sync,
                        "plan_policy": self.uvm_init_weight_plan_policy,
                        "residency_metadata": hot_metadata,
                    }
                )

        executor_used = ""
        driver_result = 0
        hot_real_query_records = 0
        hot_real_query_ranges = 0
        hot_real_queried_bytes = 0
        hot_real_gpu_bytes = 0
        hot_real_cpu_bytes = 0
        hot_real_any_bytes = 0
        hot_real_unknown_bytes = 0
        hot_real_gpu_ratio: float | None = None
        if prefetch_items:
            ranges = [(int(item["ptr"]), int(item["bytes"])) for item in prefetch_items]
            (
                batch_successes,
                executor_used,
                driver_result,
            ) = batch_prefetch_ranges_to_device_with_executor(
                ranges,
                prefetch_device,
                executor=self.uvm_weight_executor,
            )
            if self.uvm_init_weight_hot_prefetch_sync and batch_successes > 0:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    logger.debug(
                        "Failed to synchronize after init hot weight prefetch",
                        exc_info=True,
                    )
            if (
                self.uvm_init_weight_hot_prefetch_real_query_enable
                and batch_successes > 0
            ):
                successful_ranges = ranges[:batch_successes]
                query_device = prefetch_device
                try:
                    real_result = driver_batch_query_ranges_residency_bytes(
                        successful_ranges,
                        query_device,
                    )
                except Exception:
                    logger.debug(
                        "Failed to query init hot weight residency",
                        exc_info=True,
                    )
                    real_result = {
                        "queried_bytes": 0,
                        "gpu_bytes": 0,
                        "cpu_bytes": 0,
                        "any_bytes": 0,
                        "unknown_bytes": 0,
                    }
                hot_real_query_records = 1
                hot_real_query_ranges = len(successful_ranges)
                hot_real_queried_bytes = int(real_result.get("queried_bytes") or 0)
                hot_real_gpu_bytes = int(real_result.get("gpu_bytes") or 0)
                hot_real_cpu_bytes = int(real_result.get("cpu_bytes") or 0)
                hot_real_any_bytes = int(real_result.get("any_bytes") or 0)
                hot_real_unknown_bytes = int(real_result.get("unknown_bytes") or 0)
                hot_real_gpu_ratio = (
                    hot_real_gpu_bytes / hot_real_queried_bytes
                    if hot_real_queried_bytes
                    else 0.0
                )
                self._append_uvm_init_weight_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "action": "init_weight_hot_prefetch_real_query",
                        "query_records": hot_real_query_records,
                        "query_ranges": hot_real_query_ranges,
                        "queried_bytes": hot_real_queried_bytes,
                        "gpu_bytes": hot_real_gpu_bytes,
                        "cpu_bytes": hot_real_cpu_bytes,
                        "any_bytes": hot_real_any_bytes,
                        "unknown_bytes": hot_real_unknown_bytes,
                        "gpu_ratio": hot_real_gpu_ratio,
                        "device": query_device,
                        "requested_device": self.uvm_init_weight_hot_prefetch_device,
                        "plan_policy": self.uvm_init_weight_plan_policy,
                    }
                )
            for batch_index, item in enumerate(prefetch_items):
                success = batch_index < batch_successes
                action = (
                    "init_weight_hot_prefetch_issued"
                    if success
                    else "init_weight_hot_prefetch_skipped"
                )
                if success:
                    prefetch_issued_records += 1
                    prefetch_issued_bytes += int(item["bytes"])
                    mark_uvm_residency_state(
                        region_id=str(item["residency_region_id"]),
                        state="gpu",
                        metadata={
                            **cast(dict[str, object], item["residency_metadata"]),
                            "source": "stage_m",
                            "stage": "stage_m",
                            "pool_class": "hot",
                            "driver_subpool": "hot",
                            "expert_init_class": "hot",
                            "placement": "GPU_RESIDENT",
                            "target_placement": "GPU_RESIDENT",
                            "init_hot": True,
                            "init_cold": False,
                            "protected": True,
                            "pinned": True,
                            "prefetchable": True,
                            "priority": 100,
                            "plan_policy": self.uvm_init_weight_plan_policy,
                            "executor": executor_used,
                            "sync_prefetch": self.uvm_init_weight_hot_prefetch_sync,
                        },
                    )
                    self._uvm_update_weight_expert_role_cache(
                        layer_id=layer_id,
                        global_expert_id=int(item["global_expert_id"]),
                        role=str(item["role"]),
                        state="gpu",
                        estimated_gpu_resident=True,
                    )
                self._append_uvm_init_weight_trace(
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if key != "residency_metadata"
                        },
                        "action": action,
                        "ptr": f"0x{int(item['ptr']):x}",
                        "success": success,
                        "executor": executor_used,
                        "requested_executor": self.uvm_weight_executor,
                        "driver_result": driver_result,
                        "batch_index": batch_index,
                        "batch_size": len(prefetch_items),
                        "batch_successes": batch_successes,
                    }
                )

        return {
            "hot_registered_records": registered_records,
            "hot_prefetch_candidate_records": len(prefetch_items),
            "hot_prefetch_attempted_bytes": prefetch_attempted_bytes,
            "hot_prefetch_requested_bytes": prefetch_requested_bytes,
            "hot_prefetch_issued_records": prefetch_issued_records,
            "hot_prefetch_issued_bytes": prefetch_issued_bytes,
            "hot_prefetch_rejected_records": prefetch_rejected_records,
            "hot_prefetch_executor": executor_used,
            "hot_prefetch_driver_result": driver_result,
            "hot_prefetch_real_query_records": hot_real_query_records,
            "hot_prefetch_real_query_ranges": hot_real_query_ranges,
            "hot_prefetch_real_queried_bytes": hot_real_queried_bytes,
            "hot_prefetch_real_gpu_bytes": hot_real_gpu_bytes,
            "hot_prefetch_real_cpu_bytes": hot_real_cpu_bytes,
            "hot_prefetch_real_any_bytes": hot_real_any_bytes,
            "hot_prefetch_real_unknown_bytes": hot_real_unknown_bytes,
            "hot_prefetch_real_gpu_ratio": hot_real_gpu_ratio,
        }

    def _uvm_init_hot_repair_expert_pairs(
        self,
        layer_id: int,
    ) -> list[tuple[int, int]]:
        """Return local experts which must remain GPU resident after Stage M."""
        offloaded_ids = (
            self._uvm_slot_managed_global_expert_ids
            | self._uvm_remote_read_global_expert_ids
        )
        return [
            (global_expert_id, local_expert_id)
            for global_expert_id, local_expert_id in (
                self._uvm_all_local_expert_pairs()
            )
            if (layer_id, global_expert_id)
            not in self.uvm_init_weight_offload_plan_keys
            and global_expert_id not in offloaded_ids
        ]

    def _uvm_init_hot_repair_layer_weight_ranges(
        self,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for role in ("moe_gate_up", "moe_down"):
            if role not in self.uvm_init_weight_offload_target_roles:
                continue
            bounds = self._uvm_role_tensor_bounds(role)
            if bounds is None:
                continue
            start, end = bounds
            if end > start:
                ranges.append((start, end - start))
        return ranges

    def _uvm_init_hot_repair_excluded_ranges(
        self,
        *,
        layer_id: int,
        hot_expert_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        hot_pairs = set(hot_expert_pairs)
        ranges: list[tuple[int, int]] = []
        for global_expert_id, local_expert_id in self._uvm_all_local_expert_pairs():
            if (global_expert_id, local_expert_id) in hot_pairs:
                continue
            ranges.extend(
                (ptr, size_bytes)
                for _, ptr, size_bytes, _ in (
                    self._uvm_stage_m_expert_weight_slices(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        local_expert_id=local_expert_id,
                    )
                )
            )
        return ranges

    @staticmethod
    def _uvm_init_hot_residency_from_aggregate_queries(
        *,
        layer_residency: dict[str, int | bool],
        excluded_residency: dict[str, int | bool],
        layer_requested_bytes: int,
        excluded_requested_bytes: int,
        hot_requested_bytes: int,
    ) -> dict[str, int | bool]:
        def subtract(field: str) -> int:
            return max(
                int(layer_residency.get(field) or 0)
                - int(excluded_residency.get(field) or 0),
                0,
            )

        excluded_ok = bool(excluded_residency.get("ok"))
        if excluded_requested_bytes == 0:
            excluded_ok = True
        queried_bytes = subtract("queried_bytes")
        gpu_bytes = subtract("gpu_bytes")
        cpu_bytes = subtract("cpu_bytes")
        any_bytes = subtract("any_bytes")
        unknown_bytes = subtract("unknown_bytes")
        ok = bool(
            layer_residency.get("ok")
            and excluded_ok
            and int(layer_residency.get("queried_bytes") or 0)
            == layer_requested_bytes
            and int(excluded_residency.get("queried_bytes") or 0)
            == excluded_requested_bytes
            and queried_bytes == hot_requested_bytes
        )
        return {
            "ok": ok,
            "status": (
                int(layer_residency.get("status") or 0)
                if not excluded_requested_bytes
                else min(
                    int(layer_residency.get("status") or 0),
                    int(excluded_residency.get("status") or 0),
                )
            ),
            "queried_bytes": queried_bytes,
            "gpu_bytes": gpu_bytes,
            "cpu_bytes": cpu_bytes,
            "any_bytes": any_bytes,
            "unknown_bytes": unknown_bytes,
            "layer_queried_bytes": int(
                layer_residency.get("queried_bytes") or 0
            ),
            "excluded_queried_bytes": int(
                excluded_residency.get("queried_bytes") or 0
            ),
        }

    def _uvm_query_init_hot_weight_residency(
        self,
        *,
        layer_ranges: list[tuple[int, int]],
        excluded_ranges: list[tuple[int, int]],
        hot_requested_bytes: int,
        device: int,
    ) -> dict[str, int | bool]:
        layer_residency = driver_batch_query_ranges_residency_bytes(
            layer_ranges,
            device,
            wait_for_trackers=False,
        )
        excluded_residency: dict[str, int | bool]
        if excluded_ranges:
            excluded_residency = driver_batch_query_ranges_residency_bytes(
                excluded_ranges,
                device,
                wait_for_trackers=False,
            )
        else:
            excluded_residency = {
                "ok": True,
                "status": 0,
                "queried_bytes": 0,
                "gpu_bytes": 0,
                "cpu_bytes": 0,
                "any_bytes": 0,
                "unknown_bytes": 0,
            }
        result = self._uvm_init_hot_residency_from_aggregate_queries(
            layer_residency=layer_residency,
            excluded_residency=excluded_residency,
            layer_requested_bytes=sum(size for _, size in layer_ranges),
            excluded_requested_bytes=sum(size for _, size in excluded_ranges),
            hot_requested_bytes=hot_requested_bytes,
        )
        result["layer_query_ranges"] = len(layer_ranges)
        result["excluded_query_ranges"] = len(excluded_ranges)
        return result

    def repair_uvm_init_hot_weight_residency(self) -> dict[str, object]:
        """Repair and verify hot expert residency after all cold release work.

        The model runner invokes this in a second global module pass. This is
        intentionally separate from per-layer Stage M so cold ranges across all
        layers have released their HBM capacity before hot pages are restored.
        """
        if not self.uvm_init_weight_hot_repair_enable:
            return {"hot_repair_enabled": False}
        if self._uvm_init_weight_hot_repair_done:
            return {
                "hot_repair_enabled": True,
                "hot_repair_already_done": True,
            }

        self._refresh_uvm_weight_action_plans()
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return {
                "hot_repair_enabled": True,
                "hot_repair_skipped_no_layer": True,
            }

        device = self.uvm_init_weight_hot_prefetch_device
        if device < 0:
            device = int(torch.cuda.current_device())
        expert_pairs = self._uvm_init_hot_repair_expert_pairs(layer_id)
        repair_items: list[dict[str, object]] = []
        for global_expert_id, local_expert_id in expert_pairs:
            for role, ptr, size_bytes, _ in self._uvm_stage_m_expert_weight_slices(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                local_expert_id=local_expert_id,
            ):
                repair_items.append(
                    {
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "role": role,
                        "ptr": ptr,
                        "bytes": size_bytes,
                        "region_id": self._uvm_expert_weight_region_id(
                            layer_id,
                            global_expert_id,
                            role,
                        ),
                    }
                )

        ranges = [
            (int(item["ptr"]), int(item["bytes"])) for item in repair_items
        ]
        requested_bytes = sum(size_bytes for _, size_bytes in ranges)
        layer_ranges = self._uvm_init_hot_repair_layer_weight_ranges()
        excluded_ranges = self._uvm_init_hot_repair_excluded_ranges(
            layer_id=layer_id,
            hot_expert_pairs=expert_pairs,
        )
        if not layer_ranges:
            layer_ranges = ranges
            excluded_ranges = []
        torch.cuda.synchronize()
        initial_residency = self._uvm_query_init_hot_weight_residency(
            layer_ranges=layer_ranges,
            excluded_ranges=excluded_ranges,
            hot_requested_bytes=requested_bytes,
            device=device,
        )
        initial_queried_bytes = int(initial_residency.get("queried_bytes") or 0)
        initial_gpu_bytes = int(initial_residency.get("gpu_bytes") or 0)
        initial_cpu_bytes = int(initial_residency.get("cpu_bytes") or 0)
        initial_unknown_bytes = int(initial_residency.get("unknown_bytes") or 0)
        initially_fully_gpu_resident = bool(
            initial_residency.get("ok")
            and initial_queried_bytes == requested_bytes
            and initial_gpu_bytes == requested_bytes
            and initial_cpu_bytes == 0
            and initial_unknown_bytes == 0
        )
        if (
            self.uvm_init_weight_hot_repair_require_full_residency
            and not initial_residency.get("ok")
        ):
            raise RuntimeError(
                "Stage-M hot repair initial residency query failed: "
                f"layer_id={layer_id}, requested_bytes={requested_bytes}, "
                f"status={initial_residency.get('status')}"
            )

        preferred_result = len(ranges)
        if self.uvm_init_weight_hot_repair_preferred_gpu:
            preferred_result = batch_advise_ranges_preferred_location(
                ranges,
                device,
            )
            if preferred_result != len(ranges):
                raise RuntimeError(
                    "Stage-M hot repair failed to set PreferredLocation(GPU): "
                    f"layer_id={layer_id}, expected={len(ranges)}, "
                    f"actual={preferred_result}"
                )

        prefetch_successes = 0
        executor_used = "query_skip"
        driver_result = 0
        if not initially_fully_gpu_resident:
            prefetch_successes, executor_used, driver_result = (
                batch_prefetch_ranges_to_device_with_executor(
                    ranges,
                    device,
                    executor=self.uvm_weight_executor,
                )
            )
            if prefetch_successes != len(ranges):
                raise RuntimeError(
                    "Stage-M hot repair GPU prefetch was incomplete: "
                    f"layer_id={layer_id}, expected={len(ranges)}, "
                    f"actual={prefetch_successes}, executor={executor_used}, "
                    f"driver_result={driver_result}"
                )
            torch.cuda.synchronize()
            residency = self._uvm_query_init_hot_weight_residency(
                layer_ranges=layer_ranges,
                excluded_ranges=excluded_ranges,
                hot_requested_bytes=requested_bytes,
                device=device,
            )
        else:
            residency = initial_residency
        queried_bytes = int(residency.get("queried_bytes") or 0)
        gpu_bytes = int(residency.get("gpu_bytes") or 0)
        cpu_bytes = int(residency.get("cpu_bytes") or 0)
        unknown_bytes = int(residency.get("unknown_bytes") or 0)
        fully_gpu_resident = bool(
            residency.get("ok")
            and queried_bytes == requested_bytes
            and gpu_bytes == requested_bytes
            and cpu_bytes == 0
            and unknown_bytes == 0
        )
        stats: dict[str, object] = {
            "hot_repair_enabled": True,
            "hot_repair_layer_id": layer_id,
            "hot_repair_experts": len(expert_pairs),
            "hot_repair_ranges": len(ranges),
            "hot_repair_requested_bytes": requested_bytes,
            "hot_repair_layer_query_ranges": int(
                residency.get("layer_query_ranges") or 0
            ),
            "hot_repair_excluded_query_ranges": int(
                residency.get("excluded_query_ranges") or 0
            ),
            "hot_repair_initial_residency_ok": bool(
                initial_residency.get("ok")
            ),
            "hot_repair_initial_residency_status": int(
                initial_residency.get("status") or 0
            ),
            "hot_repair_initial_queried_bytes": initial_queried_bytes,
            "hot_repair_initial_gpu_bytes": initial_gpu_bytes,
            "hot_repair_initial_cpu_bytes": initial_cpu_bytes,
            "hot_repair_initial_unknown_bytes": initial_unknown_bytes,
            "hot_repair_initially_fully_gpu_resident": (
                initially_fully_gpu_resident
            ),
            "hot_repair_preferred_gpu": (
                self.uvm_init_weight_hot_repair_preferred_gpu
            ),
            "hot_repair_preferred_result": preferred_result,
            "hot_repair_prefetch_successes": prefetch_successes,
            "hot_repair_executor": executor_used,
            "hot_repair_driver_result": driver_result,
            "hot_repair_residency_ok": bool(residency.get("ok")),
            "hot_repair_residency_status": int(residency.get("status") or 0),
            "hot_repair_queried_bytes": queried_bytes,
            "hot_repair_gpu_bytes": gpu_bytes,
            "hot_repair_cpu_bytes": cpu_bytes,
            "hot_repair_any_bytes": int(residency.get("any_bytes") or 0),
            "hot_repair_unknown_bytes": unknown_bytes,
            "hot_repair_gpu_ratio": (
                gpu_bytes / queried_bytes if queried_bytes else 0.0
            ),
            "hot_repair_fully_gpu_resident": fully_gpu_resident,
        }
        self._append_uvm_init_weight_trace(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "layer_id": layer_id,
                "action": "init_weight_hot_repair_summary",
                "plan_policy": self.uvm_init_weight_plan_policy,
                **stats,
            }
        )
        if (
            self.uvm_init_weight_hot_repair_require_full_residency
            and not fully_gpu_resident
        ):
            raise RuntimeError(
                "Stage-M hot repair residency verification failed: "
                f"layer_id={layer_id}, requested_bytes={requested_bytes}, "
                f"queried_bytes={queried_bytes}, gpu_bytes={gpu_bytes}, "
                f"cpu_bytes={cpu_bytes}, unknown_bytes={unknown_bytes}, "
                f"status={residency.get('status')}"
            )

        for item in repair_items:
            region_id = str(item["region_id"])
            mark_uvm_residency_state(
                region_id=region_id,
                state="gpu",
                metadata={
                    "placement": "GPU_RESIDENT",
                    "target_placement": "GPU_RESIDENT",
                    "hot_repair_verified": fully_gpu_resident,
                    "hot_repair_device": device,
                },
            )
            self._uvm_update_weight_expert_role_cache(
                layer_id=layer_id,
                global_expert_id=int(item["global_expert_id"]),
                role=str(item["role"]),
                state="gpu",
                estimated_gpu_resident=fully_gpu_resident,
            )
        self._uvm_init_weight_hot_repair_done = True
        return stats

    def run_uvm_init_weight_offload(self) -> None:
        """Stage M: apply hot/warm/cold MoE placement before serving.

        Cold experts consume the offload budget first. Warm experts are only
        considered after cold experts and use the same reserved-slot restore
        path if they are released. Hot experts remain protected on the GPU.
        """
        if not (
            self.uvm_init_weight_offload_enable
            or self.uvm_init_weight_hot_prefetch_enable
            or self.uvm_init_weight_register_all_local_experts
        ):
            return
        if (
            self.uvm_init_weight_offload_enable
            and self.uvm_init_weight_offload_mode
            in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
            and not self._uvm_stage_m_uses_remote_read()
            and (
                not self.uvm_expert_reserved_enable
                or not self.uvm_weight_prefetch_enabled
            )
        ):
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "action": "init_weight_offload_reject_without_slot_restore",
                    "mode": self.uvm_init_weight_offload_mode,
                    "access_mode": getattr(
                        self, "uvm_expert_offload_access_mode", "restore"
                    ),
                    "reserved_executor_enabled": self.uvm_expert_reserved_enable,
                    "runtime_prefetch_enabled": self.uvm_weight_prefetch_enabled,
                }
            )
            raise RuntimeError(
                "Stage-M expert release requires reserved slot restore: set "
                "VLLM_UVM_WEIGHT_EXECUTOR=expert_reserved_regions_async and "
                "VLLM_UVM_WEIGHT_PREFETCH_ENABLE=1"
            )
        if self._uvm_init_weight_offload_done:
            return
        self._uvm_init_weight_offload_done = True
        self._refresh_uvm_weight_action_plans()
        accepted_init_plan_policies = {
            "routing_heatmap_v1",
            "exact_used_oracle_v1",
            "train_test_static_hotset_v1",
        }
        if (
            self.uvm_init_weight_require_routing_plan
            and self.uvm_init_weight_plan_policy not in accepted_init_plan_policies
        ):
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "action": "init_weight_offload_reject_non_routing_plan",
                    "mode": self.uvm_init_weight_offload_mode,
                    "plan_file": self.uvm_init_weight_offload_plan_file,
                    "plan_policy": self.uvm_init_weight_plan_policy,
                    "required_policy": ",".join(sorted(accepted_init_plan_policies)),
                }
            )
            return
        if (
            not self.uvm_init_weight_offload_plan_keys
            and not self.uvm_init_weight_hot_plan_keys
            and not self.uvm_init_weight_warm_plan_keys
        ):
            self._append_uvm_init_weight_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "action": "init_weight_offload_no_plan",
                    "mode": self.uvm_init_weight_offload_mode,
                    "plan_file": self.uvm_init_weight_offload_plan_file,
                    "register_all_local_experts": (
                        self.uvm_init_weight_register_all_local_experts
                    ),
                    "complete_expert_plan": (self.uvm_init_weight_complete_expert_plan),
                }
            )
            if not self.uvm_init_weight_register_all_local_experts:
                return

        require_complete_remote_prepare = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_REQUIRE_COMPLETE_REMOTE_PREPARE",
            default=False,
        )
        try:
            layer_id = self._uvm_layer_id()
            if layer_id is None:
                return
            self._uvm_register_warm_prefetch_module(layer_id)
            self._record_uvm_moe_address_map(
                marker="stage_m_init_offload_begin",
            )
            preregister_stats = self._register_uvm_init_all_local_weight_regions(
                layer_id
            )
            hot_stats = self._register_uvm_init_hot_weight_regions(layer_id)
            all_local_pairs = self._uvm_all_local_expert_pairs()
            expert_pairs = self._uvm_stage_m_offload_expert_pairs(
                layer_id,
                all_local_pairs,
            )
            incremental_preregister_only = False
            if self.uvm_incremental_expert_offload_enable:
                if not self._uvm_incremental_offload_in_progress:
                    # release_remote submits stable envelope region IDs to the
                    # driver. Register planned envelopes before serving, but
                    # leave their backing and residency unchanged until a KV
                    # pressure wave selects the expert for release.
                    incremental_preregister_only = (
                        self._uvm_stage_m_requires_release_envelopes()
                    )
                    if not incremental_preregister_only:
                        expert_pairs = []
                else:
                    expert_pairs = [
                        pair
                        for pair in expert_pairs
                        if (
                            self._uvm_incremental_allow_warm
                            or self._uvm_init_weight_expert_class(
                                layer_id=layer_id,
                                global_expert_id=pair[0],
                            )
                            == "cold"
                        )
                        and (
                            self._uvm_incremental_selected_global_expert_id is None
                            or pair[0]
                            == self._uvm_incremental_selected_global_expert_id
                        )
                        and self._uvm_incremental_slot_capacity_allows(pair[1])
                    ][: self.uvm_incremental_experts_per_layer_per_round]
            ignore_limits = bool(
                self.uvm_init_weight_complete_expert_plan
                and self.uvm_init_weight_complete_plan_ignore_limits
            )
            if (
                self.uvm_init_weight_offload_max_experts_per_layer > 0
                and not ignore_limits
            ):
                expert_pairs = expert_pairs[
                    : self.uvm_init_weight_offload_max_experts_per_layer
                ]
            if self.uvm_init_weight_offload_max_bytes > 0 and not ignore_limits:
                byte_limited_pairs: list[tuple[int, int]] = []
                selected_bytes = 0
                for global_expert_id, local_expert_id in expert_pairs:
                    expert_bytes = sum(
                        size_bytes
                        for _, _, size_bytes, _ in (
                            self._uvm_stage_m_expert_weight_slices(
                                layer_id=layer_id,
                                global_expert_id=global_expert_id,
                                local_expert_id=local_expert_id,
                            )
                        )
                    )
                    if (
                        selected_bytes + expert_bytes
                        > self.uvm_init_weight_offload_max_bytes
                    ):
                        break
                    byte_limited_pairs.append((global_expert_id, local_expert_id))
                    selected_bytes += expert_bytes
                expert_pairs = byte_limited_pairs

            attempted_bytes = 0
            issued_bytes = 0
            issued_records = 0
            scheduled_bytes = 0
            rejected_records = 0
            batch_items: list[dict[str, object]] = []
            for global_expert_id, local_expert_id in expert_pairs:
                for (
                    role,
                    ptr,
                    size_bytes,
                    existing_region,
                ) in self._uvm_stage_m_expert_weight_slices(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    local_expert_id=local_expert_id,
                ):
                    region_id = self._uvm_expert_weight_region_id(
                        layer_id,
                        global_expert_id,
                        role,
                    )
                    residency_metadata = self._uvm_expert_weight_residency_metadata(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        local_expert_id=local_expert_id,
                        role=role,
                        ptr=ptr,
                        size_bytes=size_bytes,
                        active_this_step=False,
                        source_action="stage_m_init_offload",
                    )
                    expert_class = self._uvm_init_weight_expert_class(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                    )
                    logical_driver_tag_skip = bool(
                        self.uvm_expert_envelope_enable
                        and self._uvm_stage_m_requires_release_envelopes()
                    )
                    offload_metadata = self._uvm_init_weight_class_metadata(
                        metadata=residency_metadata,
                        expert_class=expert_class,
                        driver_tag_skip=logical_driver_tag_skip,
                        cpu_offload_placement=(
                            not incremental_preregister_only
                            and self.uvm_init_weight_offload_mode != "trace_only"
                        ),
                    )
                    if existing_region is None:
                        registered_region = register_uvm_residency_region(
                            pool="weights",
                            region_id=region_id,
                            ptr=ptr,
                            bytes=size_bytes,
                            state="gpu",
                            evictable=True,
                            active=False,
                            metadata=offload_metadata,
                        )
                        driver_region_id = int(
                            getattr(registered_region, "driver_region_id", 0) or 0
                        )
                    else:
                        driver_region_id = int(
                            existing_region.get("driver_region_id") or 0
                        )
                        mark_uvm_residency_state(
                            region_id=region_id,
                            state=str(existing_region.get("state") or "gpu"),
                            metadata=offload_metadata,
                        )
                    targeted_driver_registration: dict[str, object] | None = None
                    if self._uvm_stage_m_requires_logical_release_registration():
                        targeted_driver_registration = (
                            ensure_uvm_residency_driver_region_registered(region_id)
                        )
                        refreshed_region = get_uvm_residency_region_snapshot(region_id)
                        if refreshed_region is not None:
                            driver_region_id = int(
                                refreshed_region.get("driver_region_id") or 0
                            )
                        if not bool(
                            targeted_driver_registration.get("registered")
                        ):
                            raise RuntimeError(
                                "Stage-M release_remote driver registration failed "
                                f"for {region_id}: {targeted_driver_registration}"
                            )
                    envelope_records: list[dict[str, object]] = []
                    if (
                        self._uvm_stage_m_requires_release_envelopes()
                    ):
                        envelope_items = self._uvm_get_or_build_expert_envelope_items(
                            source_region_id=region_id,
                            logical_driver_region_id=driver_region_id,
                            layer_id=layer_id,
                            global_expert_id=global_expert_id,
                            local_expert_id=local_expert_id,
                            role=role,
                            ptr=ptr,
                            size_bytes=size_bytes,
                            metadata=offload_metadata,
                            state=(
                                str(existing_region.get("state"))
                                if existing_region is not None
                                else "gpu"
                            ),
                            source_action=("stage_m_driver_release_chunk_envelope"),
                        )
                        envelope_records = [
                            {
                                "region_id": envelope_item.region_id,
                                "driver_region_id": envelope_item.driver_region_id,
                                "ptr": envelope_item.ptr,
                                "bytes": envelope_item.bytes,
                                "metadata": envelope_item.metadata,
                            }
                            for envelope_item in envelope_items
                        ]
                    release_driver_region_ids = [
                        int(record.get("driver_region_id") or 0)
                        for record in envelope_records
                        if int(record.get("driver_region_id") or 0) > 0
                        and bool(
                            cast(dict[str, object], record.get("metadata") or {}).get(
                                "driver_registered"
                            )
                        )
                    ]
                    if (
                        self._uvm_stage_m_requires_release_envelopes()
                        and self.uvm_expert_envelope_enable
                        and (
                            not envelope_records
                            or len(release_driver_region_ids) != len(envelope_records)
                        )
                    ):
                        raise RuntimeError(
                            "Stage-M envelope driver registration is incomplete "
                            f"for {region_id}: registered="
                            f"{len(release_driver_region_ids)}/"
                            f"{len(envelope_records)}"
                        )
                    if not release_driver_region_ids and driver_region_id > 0:
                        release_driver_region_ids = [driver_region_id]

                    if incremental_preregister_only:
                        self._append_uvm_init_weight_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "layer_id": layer_id,
                                "action": (
                                    "init_weight_incremental_envelope_preregistered"
                                ),
                                "mode": self.uvm_init_weight_offload_mode,
                                "access_mode": self.uvm_expert_offload_access_mode,
                                "remote_prepare_mode": self.uvm_remote_prepare_mode,
                                "role": role,
                                "residency_region_id": region_id,
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "expert_init_class": expert_class,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "release_driver_region_ids": (
                                    release_driver_region_ids
                                ),
                                "envelope_count": len(envelope_records),
                                "envelope_bytes": sum(
                                    int(record.get("bytes") or 0)
                                    for record in envelope_records
                                ),
                            }
                        )
                        continue

                    attempted_bytes += size_bytes
                    if (
                        self.uvm_init_weight_offload_max_bytes > 0
                        and not (
                            self.uvm_init_weight_complete_expert_plan
                            and self.uvm_init_weight_complete_plan_ignore_limits
                        )
                        and issued_bytes + scheduled_bytes + size_bytes
                        > self.uvm_init_weight_offload_max_bytes
                    ):
                        rejected_records += 1
                        self._append_uvm_init_weight_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "action": "init_weight_offload_budget_reject",
                                "mode": self.uvm_init_weight_offload_mode,
                                "role": role,
                                "residency_region_id": region_id,
                                "driver_region_id": driver_region_id,
                                "targeted_driver_registration": (
                                    targeted_driver_registration
                                ),
                                "release_driver_region_ids": (
                                    release_driver_region_ids
                                ),
                                "envelope_records": [
                                    {
                                        "region_id": str(record.get("region_id")),
                                        "driver_region_id": int(
                                            record.get("driver_region_id") or 0
                                        ),
                                        "ptr": f"0x{int(record.get('ptr') or 0):x}",
                                        "bytes": int(record.get("bytes") or 0),
                                    }
                                    for record in envelope_records
                                ],
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "expert_init_class": expert_class,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                **self._uvm_driver_capability_trace_fields(
                                    residency_metadata
                                ),
                                "issued_bytes_before": issued_bytes,
                                "max_bytes": self.uvm_init_weight_offload_max_bytes,
                            }
                        )
                        continue

                    if (
                        self.uvm_weight_batch_executor_enable
                        and self.uvm_init_weight_offload_mode
                        in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                    ):
                        scheduled_bytes += size_bytes
                        batch_items.append(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "mode": self.uvm_init_weight_offload_mode,
                                "role": role,
                                "residency_region_id": region_id,
                                "driver_region_id": driver_region_id,
                                "targeted_driver_registration": (
                                    targeted_driver_registration
                                ),
                                "release_driver_region_ids": (
                                    release_driver_region_ids
                                ),
                                "envelope_records": [
                                    {
                                        "region_id": str(record.get("region_id")),
                                        "driver_region_id": int(
                                            record.get("driver_region_id") or 0
                                        ),
                                        "ptr": int(record.get("ptr") or 0),
                                        "bytes": int(record.get("bytes") or 0),
                                    }
                                    for record in envelope_records
                                ],
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": ptr,
                                "bytes": size_bytes,
                                **self._uvm_driver_capability_trace_fields(
                                    residency_metadata
                                ),
                                "device": -1,
                                "sync_offload": self._uvm_stage_m_sync_enabled(),
                                "plan_policy": self.uvm_init_weight_plan_policy,
                                "expert_init_class": expert_class,
                                "residency_metadata": offload_metadata,
                            }
                        )
                        continue

                    action = "init_weight_offload_trace_candidate"
                    success = False
                    advise_success = False
                    prefetch_success = False
                    driver_release_success = False
                    driver_release_result: dict[str, object] | None = None
                    remote_accessed_by_result = 0
                    remote_prefetch_cpu_successes = 0
                    remote_preferred_cpu_result = 0
                    remote_executor = ""
                    if self._uvm_stage_m_uses_remote_read():
                        device = (
                            int(torch.cuda.current_device())
                            if torch.cuda.is_available()
                            else 0
                        )
                        if self._uvm_remote_prepare_uses_release():
                            driver_release_result = (
                                driver_prefetch_cpu_release_remote_512k_regions(
                                    release_driver_region_ids,
                                    device=device,
                                    prefer_existing=True,
                                )
                            )
                            release_entries_by_id = {
                                int(entry.get("region_id") or 0): entry
                                for entry in (
                                    driver_release_result.get("entries") or []
                                )
                                if isinstance(entry, dict)
                            }
                            self._uvm_record_release_remote_envelope_states(
                                envelope_records,
                                release_entries_by_id,
                            )
                            success = bool(driver_release_result.get("ok")) and all(
                                bool(
                                    release_entries_by_id.get(driver_id, {}).get(
                                        "ok"
                                    )
                                )
                                for driver_id in release_driver_region_ids
                            )
                            driver_release_success = success
                            remote_executor = "driver_release_remote"
                        else:
                            remote_prepare_result = (
                                batch_prepare_ranges_for_native_remote_access_with_executor(
                                    [(ptr, size_bytes)],
                                    sync=self._uvm_stage_m_sync_enabled(),
                                    executor=self.uvm_weight_executor,
                                    device=device,
                                )
                            )
                            (
                                remote_successes,
                                remote_executor,
                                remote_accessed_by_result,
                                remote_prefetch_cpu_successes,
                                remote_preferred_cpu_result,
                            ) = remote_prepare_result
                            success = remote_successes == 1
                            advise_success = remote_preferred_cpu_result == 1
                            prefetch_success = remote_prefetch_cpu_successes == 1
                        action = (
                            "init_weight_remote_read_prepare_issued"
                            if success
                            else "init_weight_remote_read_prepare_failed"
                        )
                    elif (
                        self.uvm_init_weight_offload_mode
                        in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                    ):
                        device = (
                            int(torch.cuda.current_device())
                            if torch.cuda.is_available()
                            else 0
                        )
                        driver_release_result = driver_expert_offload_release_regions(
                            release_driver_region_ids,
                            device=device,
                            subchunk_64k=(self._uvm_expert_release_uses_subchunk_64k()),
                            prefer_existing=True,
                            return_entries=True,
                        )
                        release_entries_by_id: dict[int, dict[str, object]] = {
                            int(entry.get("region_id") or 0): entry
                            for entry in (driver_release_result.get("entries") or [])
                            if isinstance(entry, dict)
                        }
                        success = bool(driver_release_result.get("ok")) and all(
                            bool(release_entries_by_id.get(driver_id, {}).get("ok"))
                            for driver_id in release_driver_region_ids
                        )
                        driver_release_success = success
                        action = (
                            "init_weight_offload_driver_release_issued"
                            if success
                            else "init_weight_offload_skipped"
                        )

                    if success:
                        issued_records += 1
                        issued_bytes += size_bytes
                        next_state = (
                            "cpu"
                            if self._uvm_stage_m_sync_enabled()
                            else "prefetching_cpu"
                        )
                        mark_uvm_residency_state(
                            region_id=region_id,
                            state=next_state,
                            metadata={
                                **offload_metadata,
                                "sync_offload": self._uvm_stage_m_sync_enabled(),
                                "driver_release": driver_release_success,
                                "access_mode": self.uvm_expert_offload_access_mode,
                                "remote_prepare_mode": self.uvm_remote_prepare_mode,
                                "executor": remote_executor or None,
                                "accessed_by_result": remote_accessed_by_result,
                                "prefetch_cpu_successes": (
                                    remote_prefetch_cpu_successes
                                ),
                                "preferred_cpu_result": (remote_preferred_cpu_result),
                                "driver_release_result": driver_release_result,
                            },
                        )
                        self._uvm_update_weight_expert_role_cache(
                            layer_id=layer_id,
                            global_expert_id=global_expert_id,
                            role=role,
                            state=next_state,
                            estimated_gpu_resident=False,
                        )
                        if self._uvm_stage_m_uses_remote_read():
                            self._uvm_mark_remote_read_expert(global_expert_id)
                        else:
                            self._uvm_mark_slot_managed_expert(
                                global_expert_id=global_expert_id,
                                expert_class=expert_class,
                            )
                    elif self._uvm_stage_m_uses_remote_read():
                        partial_release = bool(
                            self._uvm_remote_prepare_uses_release()
                            and driver_release_result
                            and (
                                int(driver_release_result.get("released_bytes") or 0)
                                > 0
                                or int(
                                    driver_release_result.get(
                                        "remote_mapped_bytes"
                                    )
                                    or 0
                                )
                                > 0
                            )
                        )
                        mark_uvm_residency_state(
                            region_id=region_id,
                            state="unknown" if partial_release else "gpu",
                            metadata={
                                **offload_metadata,
                                "placement": (
                                    "MIXED_UNCOMMITTED"
                                    if partial_release
                                    else "GPU_RESIDENT"
                                ),
                                "cpu_remote_read": False,
                                "remote_prepare_failed": True,
                                "remote_prepare_mode": self.uvm_remote_prepare_mode,
                                "driver_release_result": driver_release_result,
                            },
                        )
                    elif self.uvm_init_weight_offload_mode == "trace_only":
                        issued_records += 1
                        issued_bytes += size_bytes

                    self._append_uvm_init_weight_trace(
                        {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "layer_name": self.layer_name,
                            "action": action,
                            "mode": self.uvm_init_weight_offload_mode,
                            "access_mode": self.uvm_expert_offload_access_mode,
                            "remote_prepare_mode": self.uvm_remote_prepare_mode,
                            "role": role,
                            "residency_region_id": region_id,
                                "driver_region_id": driver_region_id,
                                "targeted_driver_registration": (
                                    targeted_driver_registration
                                ),
                                "release_driver_region_ids": release_driver_region_ids,
                            "envelope_records": [
                                {
                                    "region_id": str(record.get("region_id")),
                                    "driver_region_id": int(
                                        record.get("driver_region_id") or 0
                                    ),
                                    "ptr": f"0x{int(record.get('ptr') or 0):x}",
                                    "bytes": int(record.get("bytes") or 0),
                                }
                                for record in envelope_records
                            ],
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "expert_init_class": expert_class,
                            "ptr": f"0x{ptr:x}",
                            "bytes": size_bytes,
                            **self._uvm_driver_capability_trace_fields(
                                residency_metadata
                            ),
                            "device": -1,
                            "success": success,
                            "advise_cpu_success": (
                                advise_success
                                if self._uvm_stage_m_uses_remote_read()
                                or self.uvm_init_weight_offload_mode
                                in ("advise_cpu", "prefetch_cpu", "offload_cpu")
                                else False
                            ),
                            "prefetch_cpu_success": (
                                prefetch_success
                                if self._uvm_stage_m_uses_remote_read()
                                or self.uvm_init_weight_offload_mode
                                in ("prefetch_cpu", "offload_cpu")
                                else False
                            ),
                            "driver_release_success": driver_release_success,
                            "driver_release_result": driver_release_result,
                            "remote_executor": remote_executor,
                            "accessed_by_result": remote_accessed_by_result,
                            "prefetch_cpu_successes": (remote_prefetch_cpu_successes),
                            "preferred_cpu_result": remote_preferred_cpu_result,
                            "sync_offload": self._uvm_stage_m_sync_enabled(),
                            "plan_policy": self.uvm_init_weight_plan_policy,
                        }
                    )

            if batch_items:
                driver_release_result: dict[str, object] | None = None
                driver_release_summary: dict[str, object] | None = None
                driver_release_entries_by_region: dict[int, dict[str, object]] = {}
                driver_advise_result = 0
                driver_prefetch_result = 0
                remote_accessed_by_result = 0
                remote_prefetch_cpu_successes = 0
                remote_preferred_cpu_result = 0
                if self._uvm_stage_m_uses_remote_read():
                    device = (
                        int(torch.cuda.current_device())
                        if torch.cuda.is_available()
                        else 0
                    )
                    if self._uvm_remote_prepare_uses_release():
                        driver_region_ids: list[int] = []
                        seen_driver_region_ids: set[int] = set()
                        for item in batch_items:
                            release_ids = item.get("release_driver_region_ids")
                            if not isinstance(release_ids, list):
                                release_ids = []
                            for driver_id_value in release_ids:
                                try:
                                    driver_id = int(driver_id_value)
                                except (TypeError, ValueError):
                                    continue
                                if (
                                    driver_id <= 0
                                    or driver_id in seen_driver_region_ids
                                ):
                                    continue
                                seen_driver_region_ids.add(driver_id)
                                driver_region_ids.append(driver_id)
                        driver_release_result = (
                            driver_prefetch_cpu_release_remote_512k_regions(
                                driver_region_ids,
                                device=device,
                                prefer_existing=True,
                            )
                        )
                        driver_release_summary = {
                            key: value
                            for key, value in driver_release_result.items()
                            if key != "entries"
                        }
                        for entry in driver_release_result.get("entries") or []:
                            if isinstance(entry, dict):
                                driver_release_entries_by_region[
                                    int(entry.get("region_id") or 0)
                                ] = entry
                        batch_successes = int(
                            driver_release_result.get("success_count") or 0
                        )
                        executor_used = "driver_release_remote"
                    else:
                        ranges = [
                            (int(item["ptr"]), int(item["bytes"]))
                            for item in batch_items
                        ]
                        (
                            batch_successes,
                            executor_used,
                            remote_accessed_by_result,
                            remote_prefetch_cpu_successes,
                            remote_preferred_cpu_result,
                        ) = batch_prepare_ranges_for_native_remote_access_with_executor(
                            ranges,
                            sync=self._uvm_stage_m_sync_enabled(),
                            executor=self.uvm_weight_executor,
                            device=device,
                        )
                elif (
                    self.uvm_init_weight_offload_mode
                    in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                ):
                    device = (
                        int(torch.cuda.current_device())
                        if torch.cuda.is_available()
                        else 0
                    )
                    driver_region_ids = []
                    seen_driver_region_ids: set[int] = set()
                    for item in batch_items:
                        release_ids = item.get("release_driver_region_ids")
                        if not isinstance(release_ids, list) or not release_ids:
                            release_ids = [int(item.get("driver_region_id") or 0)]
                        for driver_id_value in release_ids:
                            try:
                                driver_id = int(driver_id_value)
                            except (TypeError, ValueError):
                                continue
                            if driver_id <= 0 or driver_id in seen_driver_region_ids:
                                continue
                            seen_driver_region_ids.add(driver_id)
                            driver_region_ids.append(driver_id)
                    driver_release_result = driver_expert_offload_release_regions(
                        driver_region_ids,
                        device=device,
                        subchunk_64k=(self._uvm_expert_release_uses_subchunk_64k()),
                        prefer_existing=True,
                        return_entries=True,
                    )
                    driver_release_summary = {
                        key: value
                        for key, value in driver_release_result.items()
                        if key != "entries"
                    }
                    for entry in driver_release_result.get("entries") or []:
                        if isinstance(entry, dict):
                            driver_release_entries_by_region[
                                int(entry.get("region_id") or 0)
                            ] = entry
                    batch_successes = int(
                        driver_release_result.get("success_count") or 0
                    )
                    executor_used = "driver_offload_release"
                else:
                    batch_successes = 0
                    executor_used = "unsupported_init_weight_offload_mode"
                for batch_index, item in enumerate(batch_items):
                    driver_release_entry: dict[str, object] | None = None
                    if (
                        self._uvm_stage_m_uses_remote_read()
                        and not self._uvm_remote_prepare_uses_release()
                    ):
                        success = batch_index < batch_successes
                    elif self._uvm_remote_prepare_uses_release() or (
                        self.uvm_init_weight_offload_mode
                        in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                    ):
                        release_ids = item.get("release_driver_region_ids")
                        if not isinstance(release_ids, list) or not release_ids:
                            release_ids = [int(item.get("driver_region_id") or 0)]
                        release_driver_ids: list[int] = []
                        for driver_id_value in release_ids:
                            try:
                                driver_id = int(driver_id_value)
                            except (TypeError, ValueError):
                                continue
                            if driver_id > 0:
                                release_driver_ids.append(driver_id)
                        release_entries = [
                            driver_release_entries_by_region.get(driver_id)
                            for driver_id in release_driver_ids
                        ]
                        if self._uvm_remote_prepare_uses_release():
                            self._uvm_record_release_remote_envelope_states(
                                item.get("envelope_records"),
                                driver_release_entries_by_region,
                            )
                        success = bool(release_entries) and all(
                            bool(entry and entry.get("ok")) for entry in release_entries
                        )
                        driver_release_entry = {
                            "release_driver_region_ids": release_driver_ids,
                            "entries": release_entries,
                            "envelope_records": item.get("envelope_records"),
                        }
                    else:
                        success = batch_index < batch_successes
                    driver_release_success = (
                        success
                        if self.uvm_init_weight_offload_mode
                        in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                        and (
                            not self._uvm_stage_m_uses_remote_read()
                            or self._uvm_remote_prepare_uses_release()
                        )
                        else False
                    )
                    if self._uvm_stage_m_uses_remote_read():
                        action = (
                            "init_weight_remote_read_prepare_issued"
                            if success
                            else "init_weight_remote_read_prepare_failed"
                        )
                        advise_success = (
                            not self._uvm_remote_prepare_uses_release()
                            and batch_index < remote_preferred_cpu_result
                        )
                        prefetch_success = (
                            not self._uvm_remote_prepare_uses_release()
                            and batch_index < remote_prefetch_cpu_successes
                        )
                    elif (
                        self.uvm_init_weight_offload_mode
                        in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
                    ):
                        action = (
                            "init_weight_offload_driver_release_issued"
                            if success
                            else "init_weight_offload_skipped"
                        )
                        advise_success = False
                        prefetch_success = False
                    else:
                        action = "init_weight_offload_skipped"
                        advise_success = False
                        prefetch_success = False
                    if success:
                        issued_records += 1
                        issued_bytes += int(item["bytes"])
                        next_state = (
                            "cpu"
                            if self._uvm_stage_m_sync_enabled()
                            else "prefetching_cpu"
                        )
                        mark_uvm_residency_state(
                            region_id=str(item["residency_region_id"]),
                            state=next_state,
                            metadata={
                                **cast(dict[str, object], item["residency_metadata"]),
                                "sync_offload": self._uvm_stage_m_sync_enabled(),
                                "executor": executor_used,
                                "driver_release": driver_release_success,
                                "access_mode": self.uvm_expert_offload_access_mode,
                                "remote_prepare_mode": self.uvm_remote_prepare_mode,
                                "accessed_by_result": remote_accessed_by_result,
                                "prefetch_cpu_successes": (
                                    remote_prefetch_cpu_successes
                                ),
                                "preferred_cpu_result": (remote_preferred_cpu_result),
                                "driver_release_entry": driver_release_entry,
                            },
                        )
                        self._uvm_update_weight_expert_role_cache(
                            layer_id=layer_id,
                            global_expert_id=int(item["global_expert_id"]),
                            role=str(item["role"]),
                            state=next_state,
                            estimated_gpu_resident=False,
                        )
                        if self._uvm_stage_m_uses_remote_read():
                            self._uvm_mark_remote_read_expert(
                                int(item["global_expert_id"])
                            )
                        else:
                            self._uvm_mark_slot_managed_expert(
                                global_expert_id=int(item["global_expert_id"]),
                                expert_class=str(item["expert_init_class"]),
                            )
                    elif self._uvm_stage_m_uses_remote_read():
                        partial_release = bool(
                            self._uvm_remote_prepare_uses_release()
                            and driver_release_entry
                            and any(
                                int(entry.get("released_bytes") or 0) > 0
                                or int(entry.get("remote_mapped_bytes") or 0) > 0
                                for entry in (
                                    driver_release_entry.get("entries") or []
                                )
                                if isinstance(entry, dict)
                            )
                        )
                        mark_uvm_residency_state(
                            region_id=str(item["residency_region_id"]),
                            state="unknown" if partial_release else "gpu",
                            metadata={
                                **cast(dict[str, object], item["residency_metadata"]),
                                "placement": (
                                    "MIXED_UNCOMMITTED"
                                    if partial_release
                                    else "GPU_RESIDENT"
                                ),
                                "cpu_remote_read": False,
                                "remote_prepare_failed": True,
                                "remote_prepare_mode": self.uvm_remote_prepare_mode,
                                "driver_release_entry": driver_release_entry,
                            },
                        )
                    self._append_uvm_init_weight_trace(
                        {
                            **{
                                key: value
                                for key, value in item.items()
                                if key != "residency_metadata"
                            },
                            "action": action,
                            "ptr": f"0x{int(item['ptr']):x}",
                            "success": success,
                            "advise_cpu_success": advise_success,
                            "prefetch_cpu_success": prefetch_success,
                            "executor": executor_used,
                            "requested_executor": self.uvm_weight_executor,
                            "access_mode": self.uvm_expert_offload_access_mode,
                            "remote_prepare_mode": self.uvm_remote_prepare_mode,
                            "accessed_by_result": remote_accessed_by_result,
                            "prefetch_cpu_successes": (remote_prefetch_cpu_successes),
                            "preferred_cpu_result": remote_preferred_cpu_result,
                            "driver_advise_result": driver_advise_result,
                            "driver_prefetch_result": driver_prefetch_result,
                            "driver_release_success": driver_release_success,
                            "driver_release_entry": driver_release_entry,
                            "driver_release_result": driver_release_summary,
                            "batch_index": batch_index,
                            "batch_size": len(batch_items),
                            "batch_successes": batch_successes,
                        }
                    )

            if (
                issued_records
                or rejected_records
                or hot_stats.get("hot_registered_records")
                or hot_stats.get("hot_prefetch_candidate_records")
                or preregister_stats.get("all_local_registered_records")
            ):
                stage_stats: dict[str, int | float] = {}
                if self.uvm_hot_residency_stage_trace_enable:
                    stage_stats = self._query_uvm_init_hot_plan_residency(
                        marker="after_expert_offload",
                    )
                self._append_uvm_init_weight_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "action": "init_weight_offload_summary",
                        "mode": self.uvm_init_weight_offload_mode,
                        "access_mode": self.uvm_expert_offload_access_mode,
                        "remote_prepare_mode": self.uvm_remote_prepare_mode,
                        "incremental_preregister_only": (
                            incremental_preregister_only
                        ),
                        "planned_experts": [
                            global_expert_id for global_expert_id, _ in expert_pairs
                        ],
                        "planned_expert_classes": {
                            str(global_expert_id): self._uvm_init_weight_expert_class(
                                layer_id=layer_id,
                                global_expert_id=global_expert_id,
                            )
                            for global_expert_id, _ in expert_pairs
                        },
                        "hot_plan_records": len(self.uvm_init_weight_hot_plan_keys),
                        "warm_plan_records": len(self.uvm_init_weight_warm_plan_keys),
                        "slot_managed_global_experts": sorted(
                            self._uvm_slot_managed_global_expert_ids
                        ),
                        "remote_read_global_experts": sorted(
                            self._uvm_remote_read_global_expert_ids
                        ),
                        "offloaded_warm_global_experts": sorted(
                            self._uvm_offloaded_warm_global_expert_ids
                        ),
                        **preregister_stats,
                        **hot_stats,
                        **{
                            f"after_expert_offload_{key}": value
                            for key, value in stage_stats.items()
                        },
                        "plan_policy": self.uvm_init_weight_plan_policy,
                        "issued_records": issued_records,
                        "issued_bytes": issued_bytes,
                        "attempted_bytes": attempted_bytes,
                        "budget_reject_records": rejected_records,
                        "max_bytes": self.uvm_init_weight_offload_max_bytes,
                        "max_experts_per_layer": (
                            self.uvm_init_weight_offload_max_experts_per_layer
                        ),
                        "complete_expert_plan": (
                            self.uvm_init_weight_complete_expert_plan
                        ),
                        "complete_plan_ignore_limits": (
                            self.uvm_init_weight_complete_plan_ignore_limits
                        ),
                    }
                )
                self._uvm_init_weight_offload_last_stats = {
                    "attempted_bytes": attempted_bytes,
                    "issued_bytes": issued_bytes,
                    "issued_records": issued_records,
                    "rejected_records": rejected_records,
                }
                if (
                    require_complete_remote_prepare
                    and self._uvm_stage_m_uses_remote_read()
                    and (
                        issued_bytes != attempted_bytes
                        or rejected_records != 0
                    )
                ):
                    raise RuntimeError(
                        "Stage-M remote prepare was incomplete for "
                        f"{self.layer_name}: issued={issued_bytes}/"
                        f"{attempted_bytes} bytes, rejected_records="
                        f"{rejected_records}"
                    )
        except Exception as exc:
            logger.warning("Failed to run Stage M init weight offload", exc_info=True)
            try:
                self._append_uvm_init_weight_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "action": "init_weight_offload_exception",
                        "mode": self.uvm_init_weight_offload_mode,
                        "plan_file": self.uvm_init_weight_offload_plan_file,
                        "plan_policy": self.uvm_init_weight_plan_policy,
                        "error_type": type(exc).__name__,
                        "error": repr(exc),
                    }
                )
            except Exception:
                logger.debug("Failed to write Stage M exception trace", exc_info=True)
            if require_complete_remote_prepare:
                raise

    def _uvm_warm_prefetch_expert_pairs(
        self,
    ) -> list[tuple[int, int]]:
        pairs = [
            (global_expert_id, local_expert_id)
            for global_expert_id, local_expert_id in self._uvm_all_local_expert_pairs()
            if global_expert_id in self._uvm_offloaded_warm_global_expert_ids
        ]
        configured_max = self.uvm_expert_warm_prefetch_max_experts_per_layer
        bank_count = min(
            max(
                _env_int("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", default=2),
                1,
            ),
            max(self.uvm_expert_reserved_slot_count, 1),
        )
        bank_slots = self.uvm_expert_reserved_slot_count // bank_count
        if bank_slots <= 0:
            return []
        selected: list[tuple[int, int]] = []
        selected_slots = 0
        for pair in pairs:
            if configured_max > 0 and len(selected) >= configured_max:
                break
            required_slots = self._uvm_expert_reserved_envelope_count(pair[1])
            if selected_slots + required_slots > bank_slots:
                break
            selected.append(pair)
            selected_slots += required_slots
        return selected

    def _prefetch_uvm_offloaded_warm_expert_weights(
        self,
        *,
        source_layer_name: str,
        source_layer_id: int | None,
        force_sync: bool = False,
        source_compute_start_event: torch.cuda.Event | None = None,
    ) -> dict[str, object] | None:
        if not self.uvm_expert_warm_prefetch_enable:
            return None
        if not self.uvm_expert_reserved_enable or not self.uvm_weight_prefetch_enabled:
            return None
        if (
            self.uvm_expert_reserved_enable_after_warmup
            and os.environ.get("VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY", "0") != "1"
        ):
            return None

        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None
        expert_pairs = self._uvm_warm_prefetch_expert_pairs()
        if not expert_pairs:
            return None

        step = self._uvm_moe_trace_step
        with _UVM_WARM_PREFETCH_REGISTRY_LOCK:
            existing_ticket = getattr(self, "_uvm_warm_prefetch_ticket", None)
            if (
                existing_ticket is not None
                and int(existing_ticket.get("target_step", -1)) == step
            ):
                return existing_ticket
            existing_future = (
                existing_ticket.get("future") if existing_ticket is not None else None
            )
            if isinstance(existing_future, Future) and not existing_future.done():
                raise RuntimeError(
                    "warm expert prefetch pipeline overrun: previous target layer "
                    f"ticket is still pending for {self.layer_name}"
                )

        device = (
            int(torch.cuda.current_device())
            if self.uvm_weight_prefetch_device < 0
            else self.uvm_weight_prefetch_device
        )
        batch_prefetch_items: list[dict[str, object]] = []
        attempted_bytes = 0
        for global_expert_id, local_expert_id in expert_pairs:
            for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                local_expert_id,
                self.uvm_weight_prefetch_target_roles,
            ):
                attempted_bytes += size_bytes
                region_id = self._uvm_expert_weight_region_id(
                    layer_id,
                    global_expert_id,
                    role,
                )
                existing_region = get_uvm_residency_region_snapshot(region_id)
                existing_state = (
                    str(existing_region.get("state"))
                    if existing_region is not None
                    else self.uvm_residency_weight_initial_state
                )
                residency_metadata = self._uvm_expert_weight_residency_metadata(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    local_expert_id=local_expert_id,
                    role=role,
                    ptr=ptr,
                    size_bytes=size_bytes,
                    active_this_step=False,
                    source_action="next_layer_warm_slot_prefetch",
                )
                residency_metadata = {
                    **residency_metadata,
                    "expert_init_class": "warm",
                    "placement": "CPU_OFFLOADED_RESERVED",
                    "cpu_offloaded_reserved": True,
                    "reserved_prefetch_eligible": True,
                    "driver_tag_skip": bool(self.uvm_expert_envelope_enable),
                    "driver_tag_skip_reason": (
                        self._uvm_expert_envelope_skip_reason()
                        if self.uvm_expert_envelope_enable
                        else None
                    ),
                    "driver_flags": (
                        int(residency_metadata.get("driver_flags") or 0) | (1 << 14)
                    ),
                    "existing_state": existing_state,
                    "warm_prefetch_source_layer_name": source_layer_name,
                    "warm_prefetch_source_layer_id": source_layer_id,
                }
                register_uvm_residency_region(
                    pool="weights",
                    region_id=region_id,
                    ptr=ptr,
                    bytes=size_bytes,
                    state=existing_state,
                    evictable=False,
                    active=False,
                    metadata=residency_metadata,
                )
                batch_prefetch_items.append(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": step,
                        "mode": "prefetch",
                        "prefetch_source": "next_layer_warm_pipeline",
                        "role": role,
                        "layer_id": layer_id,
                        "residency_region_id": region_id,
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "ptr": ptr,
                        "bytes": size_bytes,
                        **self._uvm_driver_capability_trace_fields(residency_metadata),
                        "device": device,
                        "residency_metadata": residency_metadata,
                    }
                )

        roles_by_expert: dict[int, set[str]] = {}
        for item in batch_prefetch_items:
            roles_by_expert.setdefault(int(item["global_expert_id"]), set()).add(
                str(item["role"])
            )
        required_roles = {"moe_gate_up", "moe_down"}
        incomplete_experts = {
            expert_id: sorted(required_roles - roles)
            for expert_id, roles in roles_by_expert.items()
            if not required_roles.issubset(roles)
        }
        if incomplete_experts:
            raise RuntimeError(
                "warm expert pipeline requires complete w13/w2 coverage: "
                f"layer={self.layer_name} missing={incomplete_experts}"
            )

        self._uvm_warm_prefetch_generation = (
            getattr(self, "_uvm_warm_prefetch_generation", 0) + 1
        )
        generation = self._uvm_warm_prefetch_generation
        expected_region_ids = sorted(
            str(item["residency_region_id"]) for item in batch_prefetch_items
        )
        ticket: dict[str, object] = {
            "generation": generation,
            "target_step": step,
            "target_layer_name": self.layer_name,
            "target_layer_id": layer_id,
            "source_layer_name": source_layer_name,
            "source_layer_id": source_layer_id,
            "enqueue_ns": time.perf_counter_ns(),
            "expected_region_ids": expected_region_ids,
            "expected_records": len(expected_region_ids),
            "expected_bytes": attempted_bytes,
            "expected_global_experts": [pair[0] for pair in expert_pairs],
            "items": batch_prefetch_items,
            "force_sync": bool(force_sync),
            "source_compute_start_event": source_compute_start_event,
        }

        def run_prefetch() -> dict[str, object]:
            ticket["worker_start_ns"] = time.perf_counter_ns()
            timing_stream: torch.cuda.Stream | None = None

            def before_driver_ioctl() -> None:
                nonlocal timing_stream
                if (
                    not ticket.get("background")
                    or source_compute_start_event is None
                    or not torch.cuda.is_available()
                ):
                    return
                try:
                    source_compute_start_event.synchronize()
                    timing_stream = _uvm_warm_prefetch_timing_stream(device)
                    ioctl_start_event = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(timing_stream):
                        ioctl_start_event.record(timing_stream)
                    ioctl_start_event.synchronize()
                    ticket["ioctl_cuda_start_event"] = ioctl_start_event
                    ticket["ioctl_start_ns"] = time.perf_counter_ns()
                except Exception:
                    timing_stream = None
                    ticket["ioctl_cuda_timing_error"] = True
                    logger.debug(
                        "Failed to record warm prefetch ioctl start",
                        exc_info=True,
                    )

            def after_driver_ioctl() -> None:
                if timing_stream is None:
                    return
                try:
                    ioctl_end_event = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(timing_stream):
                        ioctl_end_event.record(timing_stream)
                    ioctl_end_event.synchronize()
                    ticket["ioctl_cuda_end_event"] = ioctl_end_event
                    ticket["ioctl_end_ns"] = time.perf_counter_ns()
                except Exception:
                    ticket["ioctl_cuda_timing_error"] = True
                    logger.debug(
                        "Failed to record warm prefetch ioctl end",
                        exc_info=True,
                    )

            try:
                result = self._run_uvm_expert_reserved_prefetch_batch(
                    batch_prefetch_items=batch_prefetch_items,
                    device=device,
                    step=step,
                    fail_fast=False,
                    # The driver waits for its copy tracker inside this ioctl.
                    # Running it off-thread creates the cross-layer overlap.
                    async_=False,
                    sync_before_compute=False,
                    before_driver_ioctl=before_driver_ioctl,
                    after_driver_ioctl=after_driver_ioctl,
                )
                ticket["result"] = result
                return result
            except BaseException as exc:
                ticket["worker_error"] = repr(exc)
                raise
            finally:
                ticket["worker_end_ns"] = time.perf_counter_ns()

        background = bool(self.uvm_expert_warm_background_enable and not force_sync)
        ticket["background"] = background
        if background:
            future = _uvm_warm_prefetch_executor().submit(run_prefetch)
        else:
            future = Future()
            try:
                future.set_result(run_prefetch())
            except BaseException as exc:
                future.set_exception(exc)
        ticket["future"] = future
        with _UVM_WARM_PREFETCH_REGISTRY_LOCK:
            self._uvm_warm_prefetch_ticket = ticket
        self._append_uvm_weight_prefetch_trace(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "layer_id": layer_id,
                "step": step,
                "mode": "prefetch",
                "prefetch_source": "next_layer_warm_pipeline",
                "action": "next_layer_warm_prefetch_enqueued",
                "source_layer_name": source_layer_name,
                "source_layer_id": source_layer_id,
                "generation": generation,
                "target_step": step,
                "background": background,
                "warm_global_experts": [pair[0] for pair in expert_pairs],
                "candidate_records": len(batch_prefetch_items),
                "attempted_bytes": attempted_bytes,
                "expected_region_ids": expected_region_ids,
            }
        )
        return ticket

    def _wait_for_uvm_warm_prefetch_before_compute(self) -> None:
        if not self.uvm_expert_warm_prefetch_enable:
            return
        expert_pairs = self._uvm_warm_prefetch_expert_pairs()
        if not expert_pairs:
            return

        step = self._uvm_moe_trace_step
        with _UVM_WARM_PREFETCH_REGISTRY_LOCK:
            ticket = getattr(self, "_uvm_warm_prefetch_ticket", None)
        if ticket is None or int(ticket.get("target_step", -1)) != step:
            ticket = self._prefetch_uvm_offloaded_warm_expert_weights(
                source_layer_name=self.layer_name,
                source_layer_id=self._uvm_layer_id(),
                force_sync=True,
            )
        if ticket is None:
            return

        future = ticket.get("future")
        if not isinstance(future, Future):
            raise RuntimeError(
                f"warm expert prefetch ticket has no future: {self.layer_name}"
            )
        source_end_event = ticket.get("source_compute_end_event")
        prefetch_pending_at_barrier = not future.done()
        source_compute_pending_at_barrier = False
        if source_end_event is not None:
            try:
                source_compute_pending_at_barrier = not bool(source_end_event.query())
            except Exception:
                logger.debug("Failed to query warm source compute event", exc_info=True)

        wait_start_ns = time.perf_counter_ns()
        result = future.result()
        wait_end_ns = time.perf_counter_ns()
        expected_ids = set(cast(list[str], ticket["expected_region_ids"]))
        successful_ids = set(result.get("reserved_success_region_ids") or [])
        failed_ids = set(result.get("reserved_failed_region_ids") or [])
        missing_ids = sorted(expected_ids - successful_ids)
        unexpected_ids = sorted(successful_ids - expected_ids)
        expected_physical_ids = set(
            result.get("reserved_expected_physical_region_ids") or []
        )
        successful_physical_ids = set(
            result.get("reserved_success_physical_region_ids") or []
        )
        assigned_physical_ids = set(
            result.get("reserved_assigned_physical_region_ids") or []
        )
        missing_physical_ids = sorted(expected_physical_ids - successful_physical_ids)
        unassigned_physical_ids = sorted(expected_physical_ids - assigned_physical_ids)
        driver_requested_bytes = int(
            result.get("reserved_driver_requested_bytes", 0) or 0
        )
        driver_accepted_bytes = int(
            result.get("reserved_driver_accepted_bytes", 0) or 0
        )
        physical_coverage_ok = bool(expected_physical_ids) and (
            expected_physical_ids == successful_physical_ids == assigned_physical_ids
            and int(result.get("reserved_driver_failure_count", 0) or 0) == 0
            and driver_accepted_bytes == driver_requested_bytes
            and int(result.get("reserved_driver_ordinary_resident_bytes", 0) or 0) == 0
            and int(result.get("reserved_driver_footprint_contract_broken", 0) or 0)
            == 0
        )
        logical_coverage_ok = (
            not missing_ids
            and not unexpected_ids
            and not failed_ids
            and int(result.get("reserved_success_records", 0)) == len(expected_ids)
        )
        coverage_ok = logical_coverage_ok and physical_coverage_ok
        worker_start_ns = int(ticket.get("worker_start_ns", wait_start_ns))
        worker_end_ns = int(ticket.get("worker_end_ns", wait_end_ns))
        worker_us = max(worker_end_ns - worker_start_ns, 0) / 1000.0
        wait_us = max(wait_end_ns - wait_start_ns, 0) / 1000.0
        background = bool(ticket.get("background"))
        hidden_us = max(worker_us - wait_us, 0.0) if background else 0.0
        overlap_ratio = hidden_us / worker_us if worker_us > 0 else 0.0

        source_compute_ms = None
        ioctl_start_offset_ms = None
        ioctl_end_offset_ms = None
        ioctl_duration_ms = None
        direct_overlap_ms = 0.0
        direct_overlap_ratio = 0.0
        source_start_event = ticket.get("source_compute_start_event")
        ioctl_start_event = ticket.get("ioctl_cuda_start_event")
        ioctl_end_event = ticket.get("ioctl_cuda_end_event")
        if source_start_event is not None and source_end_event is not None:
            try:
                source_end_event.synchronize()
                source_compute_ms = float(
                    source_start_event.elapsed_time(source_end_event)
                )
                if ioctl_start_event is not None and ioctl_end_event is not None:
                    ioctl_end_event.synchronize()
                    ioctl_start_offset_ms = float(
                        source_start_event.elapsed_time(ioctl_start_event)
                    )
                    ioctl_end_offset_ms = float(
                        source_start_event.elapsed_time(ioctl_end_event)
                    )
                    ioctl_duration_ms = float(
                        ioctl_start_event.elapsed_time(ioctl_end_event)
                    )
                    direct_overlap_ms = max(
                        min(source_compute_ms, ioctl_end_offset_ms)
                        - max(0.0, ioctl_start_offset_ms),
                        0.0,
                    )
                    if ioctl_duration_ms > 0:
                        direct_overlap_ratio = direct_overlap_ms / ioctl_duration_ms
            except Exception:
                logger.debug(
                    "Failed to measure warm source/driver overlap", exc_info=True
                )

        self._append_uvm_weight_prefetch_trace(
            {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "layer_id": self._uvm_layer_id(),
                "step": step,
                "action": "warm_prefetch_target_barrier",
                "generation": ticket.get("generation"),
                "source_layer_name": ticket.get("source_layer_name"),
                "source_layer_id": ticket.get("source_layer_id"),
                "background": background,
                "expected_records": len(expected_ids),
                "successful_records": len(successful_ids),
                "failed_region_ids": sorted(failed_ids),
                "missing_region_ids": missing_ids,
                "unexpected_region_ids": unexpected_ids,
                "logical_coverage_ok": logical_coverage_ok,
                "expected_physical_records": len(expected_physical_ids),
                "successful_physical_records": len(successful_physical_ids),
                "assigned_physical_records": len(assigned_physical_ids),
                "missing_physical_region_ids": missing_physical_ids,
                "unassigned_physical_region_ids": unassigned_physical_ids,
                "physical_coverage_ok": physical_coverage_ok,
                "driver_requested_bytes": driver_requested_bytes,
                "driver_accepted_bytes": driver_accepted_bytes,
                "driver_success_count": int(
                    result.get("reserved_driver_success_count", 0) or 0
                ),
                "driver_failure_count": int(
                    result.get("reserved_driver_failure_count", 0) or 0
                ),
                "driver_ordinary_resident_bytes": int(
                    result.get("reserved_driver_ordinary_resident_bytes", 0) or 0
                ),
                "driver_footprint_contract_broken": int(
                    result.get("reserved_driver_footprint_contract_broken", 0) or 0
                ),
                "coverage_ok": coverage_ok,
                "worker_us": worker_us,
                "target_wait_us": wait_us,
                "pipeline_hidden_us": hidden_us,
                "pipeline_overlap_ratio": overlap_ratio,
                "prefetch_pending_at_barrier": prefetch_pending_at_barrier,
                "source_compute_pending_at_barrier": (
                    source_compute_pending_at_barrier
                ),
                "overlap_observed_at_barrier": (
                    prefetch_pending_at_barrier and source_compute_pending_at_barrier
                ),
                "source_compute_ms": source_compute_ms,
                "driver_ioctl_start_offset_ms": ioctl_start_offset_ms,
                "driver_ioctl_end_offset_ms": ioctl_end_offset_ms,
                "driver_ioctl_duration_ms": ioctl_duration_ms,
                "direct_compute_ioctl_overlap_ms": direct_overlap_ms,
                "direct_compute_ioctl_overlap_ratio": direct_overlap_ratio,
                "direct_compute_ioctl_overlap_observed": direct_overlap_ms > 0.0,
            }
        )
        if not coverage_ok:
            message = (
                "warm expert prefetch coverage incomplete before MoE compute: "
                f"layer={self.layer_name} step={step} missing={missing_ids} "
                f"failed={sorted(failed_ids)} unexpected={unexpected_ids} "
                f"missing_physical={missing_physical_ids} "
                f"unassigned_physical={unassigned_physical_ids} "
                f"driver_bytes={driver_accepted_bytes}/{driver_requested_bytes}"
            )
            if self.uvm_expert_warm_fail_closed:
                raise RuntimeError(message)
            logger.error(message)
            return

        for item in cast(list[dict[str, object]], ticket.get("items") or []):
            region_id = str(item["residency_region_id"])
            metadata = cast(dict[str, object], item.get("residency_metadata") or {})
            mark_uvm_residency_state(
                region_id=region_id,
                state="gpu",
                metadata={
                    **metadata,
                    "warm_prefetch_generation": ticket.get("generation"),
                    "warm_prefetch_coverage_complete": True,
                    "placement": "GPU_RESERVED_SLOT",
                },
            )
            self._uvm_update_weight_expert_role_cache(
                layer_id=cast(int | None, item.get("layer_id")),
                global_expert_id=int(item["global_expert_id"]),
                role=str(item["role"]),
                state="gpu",
                estimated_gpu_resident=True,
            )
        self._uvm_warm_prefetch_ready_step = step
        self._uvm_warm_prefetch_ready_global_expert_ids = {
            pair[0] for pair in expert_pairs
        }
        ticket["consumed"] = True

    def _mark_uvm_warm_source_compute_end(self) -> None:
        tickets = getattr(self, "_uvm_warm_prefetch_source_tickets", [])
        if not tickets:
            return
        self._uvm_warm_prefetch_source_tickets = []
        if not torch.cuda.is_available():
            return
        try:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream())
            for ticket in tickets:
                ticket["source_compute_end_event"] = end_event
                ticket["source_compute_end_enqueued_ns"] = time.perf_counter_ns()
        except Exception:
            logger.debug("Failed to record warm source compute end", exc_info=True)

    def _maybe_prefetch_next_layer_uvm_warm_expert_weights(
        self,
        *,
        num_tokens: int,
    ) -> None:
        if not self.uvm_expert_warm_prefetch_enable:
            return
        if num_tokens < self.uvm_expert_warm_async_min_tokens:
            self._append_uvm_weight_prefetch_trace(
                {
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "layer_name": self.layer_name,
                    "layer_id": self._uvm_layer_id(),
                    "step": self._uvm_moe_trace_step,
                    "action": "next_layer_warm_prefetch_below_token_threshold",
                    "num_tokens": int(num_tokens),
                    "min_tokens": self.uvm_expert_warm_async_min_tokens,
                    "fallback": "synchronous_restore_on_actual_topk_hit",
                }
            )
            return
        slot_policy = (
            os.environ.get("VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY", "dynamic")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if slot_policy not in ("layer_mod_bank", "layer_mod"):
            return
        bank_count = min(
            _env_int(
                "VLLM_UVM_EXPERT_RESERVED_BANK_COUNT",
                default=2,
            ),
            self.uvm_expert_reserved_slot_count,
        )
        if bank_count < 2:
            return
        target_module = self._uvm_next_warm_prefetch_module()
        if target_module is None:
            return
        source_layer_id = self._uvm_layer_id()
        target_layer_id = target_module._uvm_layer_id()
        if source_layer_id is None or target_layer_id is None:
            return
        if source_layer_id % bank_count == target_layer_id % bank_count:
            return
        source_start_event = None
        if torch.cuda.is_available() and self.uvm_expert_warm_background_enable:
            try:
                source_start_event = torch.cuda.Event(enable_timing=True)
                source_start_event.record(torch.cuda.current_stream())
            except Exception:
                source_start_event = None
                logger.debug(
                    "Failed to record warm source compute start",
                    exc_info=True,
                )
        ticket = target_module._prefetch_uvm_offloaded_warm_expert_weights(
            source_layer_name=self.layer_name,
            source_layer_id=source_layer_id,
            source_compute_start_event=source_start_event,
        )
        if ticket is None:
            return
        if source_start_event is not None and ticket.get("background"):
            ticket["source_compute_start_enqueued_ns"] = time.perf_counter_ns()
        source_tickets = getattr(self, "_uvm_warm_prefetch_source_tickets", None)
        if source_tickets is None:
            source_tickets = []
            self._uvm_warm_prefetch_source_tickets = source_tickets
        if not any(existing is ticket for existing in source_tickets):
            source_tickets.append(ticket)

    def _maybe_restore_uvm_reserved_expert_weights(
        self,
        topk_ids: torch.Tensor,
        *,
        num_tokens: int | None = None,
        layer_id: int | None = None,
        active_expert_pairs: list[tuple[int, int]] | None = None,
        reserved_expert_pairs: list[tuple[int, int]] | None = None,
        plans_refreshed: bool = False,
        cleanup_already_done: bool = False,
    ) -> None:
        if not self.uvm_weight_prefetch_enabled:
            return
        if not self.uvm_expert_reserved_enable:
            return
        if (
            self.uvm_expert_reserved_enable_after_warmup
            and os.environ.get("VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY", "0") != "1"
        ):
            if self.uvm_weight_prefetch_trace_enabled:
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_prefetch_mode,
                        "prefetch_source": "reserved_fast_gate",
                        "action": "reserved_prefetch_deferred_until_after_warmup",
                    }
                )
            return
        if (
            self.uvm_expert_reserved_cleanup_delay_steps > 0
            and self._uvm_expert_reserved_active_steps
            and not cleanup_already_done
        ):
            self._maybe_cleanup_uvm_expert_reserved_slots(
                step=(
                    self._uvm_moe_trace_step
                    - self.uvm_expert_reserved_cleanup_delay_steps
                )
            )
        every_n_steps = self.uvm_weight_router_topk_every_n_steps
        if every_n_steps > 1 and self._uvm_moe_trace_step % every_n_steps != 0:
            return

        try:
            if not plans_refreshed:
                self._refresh_uvm_weight_action_plans()
            if layer_id is None:
                layer_id = self._uvm_layer_id()
            all_active_expert_pairs = (
                active_expert_pairs
                if active_expert_pairs is not None
                else self._uvm_active_expert_pairs(topk_ids)
            )
            all_active_global_experts = {
                global_expert_id for global_expert_id, _ in all_active_expert_pairs
            }
            expert_pairs = (
                reserved_expert_pairs
                if reserved_expert_pairs is not None
                else self._uvm_active_reserved_offload_expert_pairs(
                    layer_id,
                    all_active_expert_pairs,
                )
            )
            if (
                getattr(self, "_uvm_warm_prefetch_ready_step", -1)
                == self._uvm_moe_trace_step
            ):
                ready_warm_experts = getattr(
                    self,
                    "_uvm_warm_prefetch_ready_global_expert_ids",
                    set(),
                )
                expert_pairs = [
                    pair for pair in expert_pairs if pair[0] not in ready_warm_experts
                ]
            if not expert_pairs:
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_prefetch_mode,
                        "prefetch_source": "reserved_fast_gate",
                        "action": "reserved_prefetch_no_offloaded_active",
                        "active_global_experts": sorted(all_active_global_experts),
                        "candidate_expert_records": 0,
                        "reserved_offload_mask_hit_records": 0,
                    }
                )
                return

            device = (
                int(torch.cuda.current_device())
                if self.uvm_weight_prefetch_device < 0
                else self.uvm_weight_prefetch_device
            )
            prefetch_hit_stats = self._probe_uvm_weight_prefetch_hit_before_access(
                layer_id=layer_id,
                active_expert_pairs=all_active_expert_pairs,
                device=device,
            )

            issued_records = 0
            issued_bytes = 0
            attempted_bytes = 0
            candidate_expert_records = 0
            candidate_region_records = 0
            trace_candidate_records = 0
            batch_prefetch_items: list[dict[str, object]] = []

            for global_expert_id, local_expert_id in expert_pairs:
                candidate_expert_records += 1
                for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_weight_prefetch_target_roles,
                ):
                    candidate_region_records += 1
                    attempted_bytes += size_bytes
                    region_id = self._uvm_expert_weight_region_id(
                        layer_id,
                        global_expert_id,
                        role,
                    )
                    existing_region = get_uvm_residency_region_snapshot(region_id)
                    existing_state = (
                        str(existing_region.get("state"))
                        if existing_region is not None
                        else self.uvm_residency_weight_initial_state
                    )
                    existing_estimated_gpu = bool(
                        existing_region is not None
                        and existing_region.get("estimated_gpu_resident")
                    )
                    self._uvm_update_weight_expert_role_cache(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        role=role,
                        state=existing_state,
                        estimated_gpu_resident=existing_estimated_gpu,
                    )
                    residency_metadata = self._uvm_expert_weight_residency_metadata(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        local_expert_id=local_expert_id,
                        role=role,
                        ptr=ptr,
                        size_bytes=size_bytes,
                        active_this_step=True,
                        source_action="reserved_fast_gate_restore",
                    )
                    residency_metadata = {
                        **residency_metadata,
                        "placement": "CPU_OFFLOADED_RESERVED",
                        "cpu_offloaded_reserved": True,
                        "driver_tag_skip": bool(self.uvm_expert_envelope_enable),
                        "driver_tag_skip_reason": (
                            self._uvm_expert_envelope_skip_reason()
                            if self.uvm_expert_envelope_enable
                            else None
                        ),
                        "driver_flags": (
                            int(residency_metadata.get("driver_flags") or 0) | (1 << 14)
                        ),
                        "existing_state": existing_state,
                    }
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state=existing_state,
                        evictable=False,
                        active=True,
                        metadata=residency_metadata,
                    )
                    record_uvm_residency_access(
                        region_id=region_id,
                        step=self._uvm_moe_trace_step,
                        metadata=residency_metadata,
                    )

                    item = {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_prefetch_mode,
                        "prefetch_source": "reserved_fast_gate",
                        "role": role,
                        "layer_id": layer_id,
                        "residency_region_id": region_id,
                        "global_expert_id": global_expert_id,
                        "local_expert_id": local_expert_id,
                        "ptr": ptr,
                        "bytes": size_bytes,
                        **self._uvm_driver_capability_trace_fields(residency_metadata),
                        "device": device,
                        "residency_metadata": residency_metadata,
                    }
                    if self.uvm_weight_prefetch_mode == "prefetch":
                        issued_records += 1
                        issued_bytes += size_bytes
                        batch_prefetch_items.append(item)
                    else:
                        trace_candidate_records += 1
                        self._append_uvm_weight_prefetch_trace(
                            {
                                **{
                                    key: value
                                    for key, value in item.items()
                                    if key != "residency_metadata"
                                },
                                "ptr": f"0x{ptr:x}",
                                "action": "reserved_prefetch_trace_candidate",
                                "success": False,
                            }
                        )

            reserved_batch_stats: dict[str, object] = {}
            force_synchronous_restore = bool(
                num_tokens is not None
                and num_tokens < self.uvm_expert_warm_async_min_tokens
            )
            if batch_prefetch_items:
                reserved_batch_stats = self._run_uvm_expert_reserved_prefetch_batch(
                    batch_prefetch_items=batch_prefetch_items,
                    device=device,
                    step=self._uvm_moe_trace_step,
                    async_=False if force_synchronous_restore else None,
                    sync_before_compute=(True if force_synchronous_restore else None),
                )
                self._append_uvm_expert_reserved_coverage_trace(
                    layer_id=layer_id,
                    all_active_expert_pairs=all_active_expert_pairs,
                    requested_expert_pairs=expert_pairs,
                    reserved_batch_stats=reserved_batch_stats,
                    batch_prefetch_items=batch_prefetch_items,
                )

            if (
                issued_records
                or trace_candidate_records
                or candidate_expert_records
                or candidate_region_records
                or prefetch_hit_stats
            ):
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_prefetch_mode,
                        "prefetch_source": "reserved_fast_gate",
                        "action": "reserved_fast_gate_step_summary",
                        "active_global_experts": sorted(all_active_global_experts),
                        "requested_reserved_global_experts": [
                            global_expert_id for global_expert_id, _ in expert_pairs
                        ],
                        "requested_reserved_local_experts": [
                            local_expert_id for _, local_expert_id in expert_pairs
                        ],
                        "issued_records": issued_records,
                        "issued_bytes": issued_bytes,
                        "attempted_bytes": attempted_bytes,
                        "trace_candidate_records": trace_candidate_records,
                        "candidate_expert_records": candidate_expert_records,
                        "candidate_region_records": candidate_region_records,
                        "num_tokens": num_tokens,
                        "warm_async_min_tokens": (
                            self.uvm_expert_warm_async_min_tokens
                        ),
                        "forced_synchronous_restore": (force_synchronous_restore),
                        **reserved_batch_stats,
                        **prefetch_hit_stats,
                    }
                )
        except Exception:
            logger.debug("Failed to run UVM reserved expert restore", exc_info=True)
            if self.uvm_expert_reserved_enable and self.uvm_expert_reserved_fail_fast:
                raise

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.fused_experts.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            # Slice before all_reduce to enable possible fusion
            if self.hidden_size != og_hidden_states:
                states = states[..., :og_hidden_states]

            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                compute_start_ns = time.perf_counter_ns()
                self._append_uvm_reserved_debug_trace(
                    "moe_forward_op_enter",
                    shared_experts=False,
                    hidden_shape=list(hidden_states.shape),
                    router_shape=list(router_logits.shape),
                )
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
                self._append_uvm_reserved_debug_trace(
                    "moe_forward_op_exit",
                    shared_experts=False,
                    elapsed_us=(time.perf_counter_ns() - compute_start_ns) / 1000.0,
                )
                self._maybe_sync_uvm_reserved_debug_compute("moe_forward")
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                result = (
                    reduce_output(fused_output)
                    + zero_expert_result[..., :og_hidden_states]
                )
            else:
                result = reduce_output(fused_output)
            self._maybe_cleanup_uvm_expert_reserved_slots()
            return result
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                compute_start_ns = time.perf_counter_ns()
                self._append_uvm_reserved_debug_trace(
                    "moe_forward_op_enter",
                    shared_experts=True,
                    hidden_shape=list(hidden_states.shape),
                    router_shape=list(router_logits.shape),
                )
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
                self._append_uvm_reserved_debug_trace(
                    "moe_forward_op_exit",
                    shared_experts=True,
                    elapsed_us=(time.perf_counter_ns() - compute_start_ns) / 1000.0,
                )
                self._maybe_sync_uvm_reserved_debug_compute("moe_forward_shared")
            result = (
                reduce_output(shared_output),
                reduce_output(fused_output),
            )
            self._maybe_cleanup_uvm_expert_reserved_slots()
            return result

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with uvm_enabled_allocation_phase("enabled:moe"):
            return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            compute_start_ns = time.perf_counter_ns()
            self._append_uvm_reserved_debug_trace(
                "moe_quant_apply_enter",
                stage="chunked",
                chunk_idx=chunk_idx,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                hidden_shape=list(staged_hidden_states.shape),
                router_shape=list(staged_router_logits.shape),
            )
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not self.rocm_aiter_fmoe_enabled
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )
            self._mark_uvm_warm_source_compute_end()
            self._append_uvm_reserved_debug_trace(
                "moe_quant_apply_exit",
                stage="chunked",
                chunk_idx=chunk_idx,
                elapsed_us=(time.perf_counter_ns() - compute_start_ns) / 1000.0,
            )
            self._maybe_sync_uvm_reserved_debug_compute("quant_apply_chunked")

            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(staged_hidden_states)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                assert self.shared_experts is None
                final_hidden_states, zero_expert_result = final_hidden_states
                if zero_expert_result is not None:
                    final_hidden_states += zero_expert_result

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )
                self._maybe_cleanup_uvm_expert_reserved_slots()

        if self.shared_experts is None:
            result = full_fused_final_hidden_states
        else:
            result = (full_shared_final_hidden_states, full_fused_final_hidden_states)
        self._maybe_cleanup_uvm_expert_reserved_slots()
        return result

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self._wait_for_uvm_warm_prefetch_before_compute()

        self.ensure_moe_quant_config_init()
        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not isinstance(self.quant_method, FusedMoEModularMethod)
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared_experts, use_chunked_impl
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            result = self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )
            self._maybe_cleanup_uvm_expert_reserved_slots()
            return result

        do_naive_dispatch_combine: bool = self.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states_combined, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )
            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_output = self.shared_experts(hidden_states)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            compute_start_ns = time.perf_counter_ns()
            self._append_uvm_reserved_debug_trace(
                "moe_quant_apply_enter",
                stage="eager",
                hidden_shape=list(hidden_states.shape),
                router_shape=list(router_logits.shape),
                do_naive_dispatch_combine=do_naive_dispatch_combine,
                has_separate_shared_experts=has_separate_shared_experts,
            )
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states_combined
                if do_naive_dispatch_combine
                else hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not self.rocm_aiter_fmoe_enabled
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )
            self._mark_uvm_warm_source_compute_end()
            self._append_uvm_reserved_debug_trace(
                "moe_quant_apply_exit",
                stage="eager",
                elapsed_us=(time.perf_counter_ns() - compute_start_ns) / 1000.0,
            )
            self._maybe_sync_uvm_reserved_debug_compute("quant_apply_eager")

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        self._append_uvm_reserved_debug_trace(
                            "shared_experts_stream_compute_enter",
                            stage="eager",
                        )
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                        self._append_uvm_reserved_debug_trace(
                            "shared_experts_stream_compute_exit",
                            stage="eager",
                        )
                    current_stream().wait_stream(self.shared_experts_stream)
                    self._append_uvm_reserved_debug_trace(
                        "reserved_stream_current_wait_shared_experts",
                        stage="eager",
                    )

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                result = (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                result = (combine_output(final_hidden_states), zero_expert_result)
            else:
                result = combine_output(final_hidden_states)
            self._maybe_cleanup_uvm_expert_reserved_slots()
            return result

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    with uvm_enabled_allocation_phase("enabled:moe"):
        return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    with uvm_enabled_allocation_phase("enabled:moe"):
        return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
