# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from enum import Enum
from functools import partial
import json
import os
import time
from typing import Literal, cast, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.device_allocator.uvm import (
    advise_range_preferred_location,
    prefetch_range_to_cpu,
    prefetch_range_to_device,
    uvm_enabled_allocation_phase,
)
from vllm.device_allocator.uvm_pool_coordinator import request_uvm_pool_action
from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_residency_region_snapshot,
    mark_uvm_residency_state,
    record_uvm_residency_action,
    record_uvm_residency_access,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
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
        logger.warning("Invalid integer for %s=%s; using %s", env_name, env_value, default)
        return default


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
        self.uvm_weight_prefetch_max_bytes_per_step = _env_int(
            "VLLM_UVM_WEIGHT_PREFETCH_MAX_BYTES_PER_STEP",
            default=64 * 1024 * 1024,
        )
        self.uvm_weight_prefetch_max_experts_per_layer = _env_int(
            "VLLM_UVM_WEIGHT_PREFETCH_MAX_EXPERTS_PER_LAYER",
            default=2,
        )
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
        self.uvm_weight_prefetch_require_plan = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_PREFETCH_REQUIRE_PLAN",
            default=False,
        )
        self.uvm_weight_prefetch_plan_keys = _parse_plan_expert_keys(
            self.uvm_weight_prefetch_plan_file,
            "prefetch_plan",
        )
        self.uvm_weight_offload_enable = _env_flag_enabled(
            "VLLM_UVM_WEIGHT_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_weight_offload_mode = os.environ.get(
            "VLLM_UVM_WEIGHT_OFFLOAD_MODE",
            "trace_only",
        )
        if self.uvm_weight_offload_mode not in ("trace_only", "advise_cpu", "prefetch_cpu"):
            logger.warning(
                "Unknown VLLM_UVM_WEIGHT_OFFLOAD_MODE=%s; using trace_only",
                self.uvm_weight_offload_mode,
            )
            self.uvm_weight_offload_mode = "trace_only"
        self.uvm_weight_offload_plan_file = os.environ.get(
            "VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE",
            self.uvm_weight_prefetch_plan_file,
        )
        self.uvm_weight_offload_max_bytes_per_step = _env_int(
            "VLLM_UVM_WEIGHT_OFFLOAD_MAX_BYTES_PER_STEP",
            default=64 * 1024 * 1024,
        )
        self.uvm_weight_offload_max_experts_per_layer = _env_int(
            "VLLM_UVM_WEIGHT_OFFLOAD_MAX_EXPERTS_PER_LAYER",
            default=1,
        )
        self.uvm_weight_offload_target_roles = {
            role.strip()
            for role in os.environ.get(
                "VLLM_UVM_WEIGHT_OFFLOAD_TARGET_ROLES",
                ",".join(sorted(self.uvm_weight_prefetch_target_roles)),
            ).split(",")
            if role.strip()
        }
        self.uvm_weight_offload_plan_keys = _parse_plan_expert_keys(
            self.uvm_weight_offload_plan_file,
            "offload_plan",
        )
        self._uvm_weight_offload_completed_keys: set[tuple[int, int]] = set()
        self._uvm_weight_prefetch_plan_loaded = bool(
            self.uvm_weight_prefetch_plan_keys
        )
        self._uvm_weight_offload_plan_loaded = bool(
            self.uvm_weight_offload_plan_keys
        )
        self.uvm_residency_mode = os.environ.get(
            "VLLM_UVM_RESIDENCY_MODE", "trace_only"
        )
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
        self.uvm_residency_weight_enforce_hot_set = _env_flag_enabled(
            "VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET",
            default=False,
        )
        self.uvm_residency_weight_sync_offload = _env_flag_enabled(
            "VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD",
            default=True,
        )
        self.uvm_residency_weight_max_victims_per_step = _env_int(
            "VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP",
            default=0,
        )
        self.uvm_residency_weight_victim_policy = os.environ.get(
            "VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY",
            "cold_plan_lru",
        )
        self.uvm_residency_weight_victim_action = os.environ.get(
            "VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION",
            "advise_cpu",
        )
        if self.uvm_residency_weight_victim_action not in (
            "advise_cpu",
            "prefetch_cpu",
        ):
            logger.warning(
                "Unknown VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION=%s; "
                "using advise_cpu",
                self.uvm_residency_weight_victim_action,
            )
            self.uvm_residency_weight_victim_action = "advise_cpu"

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

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
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
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

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
        self._maybe_prefetch_uvm_expert_weights(topk_ids)
        self._maybe_offload_uvm_cold_expert_weights(topk_ids)
        return topk_weights, topk_ids, zero_expert_result

    def _log_uvm_moe_routing_trace(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        if not self.uvm_moe_trace_enabled:
            return
        try:
            ids = topk_ids.detach()
            valid_ids = ids[ids >= 0].to(dtype=torch.long)
            counts = torch.bincount(
                valid_ids.flatten(),
                minlength=max(int(self.global_num_experts), 0),
            ).cpu()
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
                "num_tokens": int(hidden_states.shape[0])
                if hidden_states.ndim > 0
                else 1,
                "top_k": int(self.top_k),
                "global_num_experts": int(self.global_num_experts),
                "logical_num_experts": int(self.logical_num_experts),
                "local_num_experts": int(self.local_num_experts),
                "topk_shape": list(topk_ids.shape),
                "router_logits_shape": list(router_logits.shape),
                "expert_token_counts": active_counts,
                "active_experts": [int(expert_id) for expert_id in active],
                "topk_weight_sum": float(topk_weights.detach().sum().cpu().item()),
                "enable_eplb": bool(self.enable_eplb),
            }
            self._uvm_moe_trace_step += 1
            with open(self.uvm_moe_trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write UVM MoE routing trace", exc_info=True)

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

    def _uvm_expert_weight_region_id(
        self,
        layer_id: int | None,
        global_expert_id: int,
        role: str,
    ) -> str:
        layer_label = "unknown" if layer_id is None else str(layer_id)
        return f"weights:{layer_label}:{global_expert_id}:{role}"

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
            "plan_hot": (
                layer_id is not None
                and (layer_id, global_expert_id)
                in self.uvm_weight_prefetch_plan_keys
            ),
            "plan_cold": (
                layer_id is not None
                and (layer_id, global_expert_id)
                in self.uvm_weight_offload_plan_keys
            ),
            "active_this_step": active_this_step,
            "source_action": source_action,
        }

    def _register_uvm_residency_cold_weight_candidates(
        self,
        *,
        layer_id: int | None,
        active_global_experts: set[int],
        active_region_ids: set[str],
    ) -> None:
        if layer_id is None or not self.uvm_weight_offload_plan_keys:
            return
        try:
            expert_pairs = [
                (global_expert_id, local_expert_id)
                for global_expert_id, local_expert_id in self._uvm_plan_local_experts(
                    self.uvm_weight_offload_plan_keys
                )
                if global_expert_id not in active_global_experts
            ]
            if self.uvm_weight_offload_max_experts_per_layer > 0:
                expert_pairs = expert_pairs[
                    : self.uvm_weight_offload_max_experts_per_layer
                ]
            for global_expert_id, local_expert_id in expert_pairs:
                for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_weight_offload_target_roles,
                ):
                    region_id = self._uvm_expert_weight_region_id(
                        layer_id,
                        global_expert_id,
                        role,
                    )
                    if region_id in active_region_ids:
                        continue
                    metadata = self._uvm_expert_weight_residency_metadata(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        local_expert_id=local_expert_id,
                        role=role,
                        ptr=ptr,
                        size_bytes=size_bytes,
                        active_this_step=False,
                        source_action="l2_cold_candidate",
                    )
                    existing_region = get_uvm_residency_region_snapshot(region_id)
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state=(
                            str(existing_region.get("state"))
                            if existing_region is not None
                            else "gpu"
                        ),
                        evictable=True,
                        active=False,
                        metadata=metadata,
                    )
        except Exception:
            logger.debug("Failed to register Stage L2 cold candidates", exc_info=True)

    def _offload_uvm_residency_weight_victims(
        self,
        *,
        victims: tuple[str, ...],
        active_region_ids: set[str],
        reason_region_id: str,
        scope_key: str,
    ) -> int:
        if not victims:
            return 0
        max_victims = self.uvm_residency_weight_max_victims_per_step
        selected_victims = list(victims)
        if max_victims > 0:
            selected_victims = selected_victims[:max_victims]

        issued = 0
        for victim_region_id in selected_victims:
            if victim_region_id in active_region_ids:
                record_uvm_residency_action(
                    action="residency_weight_victim_skip_active",
                    pool="weights",
                    region_id=victim_region_id,
                    metadata={
                        "stage": "stage_l2",
                        "reason_region_id": reason_region_id,
                        "scope_key": scope_key,
                    },
                )
                continue
            snapshot = get_uvm_residency_region_snapshot(victim_region_id)
            if not snapshot:
                continue
            snapshot_metadata = snapshot.get("metadata")
            if not isinstance(snapshot_metadata, dict):
                snapshot_metadata = {}
            if (
                snapshot.get("active") is True
                or snapshot_metadata.get("active_this_step") is True
            ):
                record_uvm_residency_action(
                    action="residency_weight_victim_skip_active",
                    pool="weights",
                    region_id=victim_region_id,
                    bytes=int(snapshot.get("bytes") or 0),
                    metadata={
                        "stage": "stage_l2",
                        "reason_region_id": reason_region_id,
                        "scope_key": scope_key,
                        "snapshot_active": snapshot.get("active") is True,
                        "snapshot_active_this_step": (
                            snapshot_metadata.get("active_this_step") is True
                        ),
                    },
                )
                continue
            ptr = snapshot.get("ptr")
            size_bytes = int(snapshot.get("bytes") or 0)
            if not isinstance(ptr, int) or size_bytes <= 0:
                record_uvm_residency_action(
                    action="residency_weight_victim_skip_no_ptr",
                    pool="weights",
                    region_id=victim_region_id,
                    bytes=size_bytes,
                    metadata={
                        "stage": "stage_l2",
                        "reason_region_id": reason_region_id,
                        "scope_key": scope_key,
                    },
                )
                continue

            success = False
            if self.uvm_residency_weight_victim_action == "prefetch_cpu":
                success = prefetch_range_to_cpu(ptr, size_bytes)
            else:
                success = advise_range_preferred_location(ptr, size_bytes, -1)
            if success:
                issued += 1
                mark_uvm_residency_state(
                    region_id=victim_region_id,
                    state="cpu"
                    if self.uvm_residency_weight_sync_offload
                    else "prefetching_cpu",
                    metadata={
                        "stage": "stage_l2",
                        "reason_region_id": reason_region_id,
                        "scope_key": scope_key,
                        "victim_action": self.uvm_residency_weight_victim_action,
                    },
                )
            record_uvm_residency_action(
                action=(
                    "residency_weight_victim_offload_issued"
                    if success
                    else "residency_weight_victim_offload_failed"
                ),
                pool="weights",
                region_id=victim_region_id,
                bytes=size_bytes,
                metadata={
                    "stage": "stage_l2",
                    "reason_region_id": reason_region_id,
                    "scope_key": scope_key,
                    "victim_action": self.uvm_residency_weight_victim_action,
                    "sync_offload": self.uvm_residency_weight_sync_offload,
                },
                success=success,
            )
        return issued

    def _append_uvm_weight_prefetch_trace(self, record: dict[str, object]) -> None:
        try:
            with open(self.uvm_weight_prefetch_trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            logger.debug("Failed to write UVM weight prefetch trace", exc_info=True)

    def _maybe_prefetch_uvm_expert_weights(self, topk_ids: torch.Tensor) -> None:
        if not self.uvm_weight_prefetch_enabled:
            return
        try:
            self._refresh_uvm_weight_action_plans()
            layer_id = self._uvm_layer_id()
            all_active_expert_pairs = self._uvm_active_expert_pairs(topk_ids)
            all_active_global_experts = {
                global_expert_id
                for global_expert_id, _ in all_active_expert_pairs
            }
            expert_pairs = all_active_expert_pairs
            if self.uvm_weight_prefetch_plan_keys:
                expert_pairs = [
                    (global_expert_id, local_expert_id)
                    for global_expert_id, local_expert_id in expert_pairs
                    if layer_id is not None
                    and (layer_id, global_expert_id)
                    in self.uvm_weight_prefetch_plan_keys
                ]
            elif self.uvm_weight_prefetch_require_plan:
                expert_pairs = []
            if self.uvm_weight_prefetch_max_experts_per_layer > 0:
                expert_pairs = expert_pairs[
                    : self.uvm_weight_prefetch_max_experts_per_layer
                ]
            active_region_ids = {
                self._uvm_expert_weight_region_id(layer_id, global_expert_id, role)
                for global_expert_id, local_expert_id in all_active_expert_pairs
                for role, _, _ in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_weight_prefetch_target_roles,
                )
            }
            if self.uvm_residency_weight_enforce_hot_set:
                self._register_uvm_residency_cold_weight_candidates(
                    layer_id=layer_id,
                    active_global_experts=all_active_global_experts,
                    active_region_ids=active_region_ids,
                )

            issued_records = 0
            attempted_bytes = 0
            issued_bytes = 0
            rejected_records = 0
            device = (
                int(torch.cuda.current_device())
                if self.uvm_weight_prefetch_device < 0
                else self.uvm_weight_prefetch_device
            )

            for global_expert_id, local_expert_id in expert_pairs:
                for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_weight_prefetch_target_roles,
                ):
                    region_id = self._uvm_expert_weight_region_id(
                        layer_id,
                        global_expert_id,
                        role,
                    )
                    residency_metadata = (
                        self._uvm_expert_weight_residency_metadata(
                            layer_id=layer_id,
                            global_expert_id=global_expert_id,
                            local_expert_id=local_expert_id,
                            role=role,
                            ptr=ptr,
                            size_bytes=size_bytes,
                            active_this_step=True,
                            source_action="prefetch",
                        )
                    )
                    existing_region = get_uvm_residency_region_snapshot(region_id)
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state=(
                            str(existing_region.get("state"))
                            if existing_region is not None
                            else self.uvm_residency_weight_initial_state
                        ),
                        evictable=False,
                        active=True,
                        metadata=residency_metadata,
                    )
                    record_uvm_residency_access(
                        region_id=region_id,
                        step=self._uvm_moe_trace_step,
                        metadata=residency_metadata,
                    )
                    residency_decision = request_uvm_gpu_residency(
                        pool="weights",
                        region_id=region_id,
                        bytes=size_bytes,
                        scope_key=(
                            f"moe_step:{os.getpid()}:"
                            f"{self._uvm_moe_trace_step}"
                        ),
                        active_region_ids=active_region_ids,
                        metadata=residency_metadata,
                    )
                    if (
                        self.uvm_residency_weight_enforce_hot_set
                        and not residency_decision.allowed
                    ):
                        self._offload_uvm_residency_weight_victims(
                            victims=residency_decision.victims,
                            active_region_ids=active_region_ids,
                            reason_region_id=region_id,
                            scope_key=residency_decision.scope_key,
                        )
                        residency_decision = request_uvm_gpu_residency(
                            pool="weights",
                            region_id=region_id,
                            bytes=size_bytes,
                            scope_key=(
                                f"moe_step:{os.getpid()}:"
                                f"{self._uvm_moe_trace_step}:retry"
                            ),
                            active_region_ids=active_region_ids,
                            metadata={
                                **residency_metadata,
                                "retry_after_victim_offload": True,
                            },
                        )
                    if (
                        self.uvm_residency_weight_enforce_hot_set
                        and not residency_decision.allowed
                    ):
                        rejected_records += 1
                        record_uvm_residency_action(
                            action="residency_weight_hot_prefetch_skipped",
                            pool="weights",
                            region_id=region_id,
                            bytes=size_bytes,
                            metadata={
                                **residency_metadata,
                                "reason": residency_decision.reason,
                            },
                            would_deny=residency_decision.would_deny,
                            victims=list(residency_decision.victims),
                        )
                        self._append_uvm_weight_prefetch_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "step": self._uvm_moe_trace_step,
                                "mode": self.uvm_weight_prefetch_mode,
                                "action": "residency_deny_skip_prefetch",
                                "role": role,
                                "residency_region_id": region_id,
                                "residency_allowed": residency_decision.allowed,
                                "residency_would_deny": (
                                    residency_decision.would_deny
                                ),
                                "residency_reason": residency_decision.reason,
                                "residency_victims": list(
                                    residency_decision.victims
                                ),
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                            }
                        )
                        continue
                    attempted_bytes += size_bytes
                    if (
                        self.uvm_weight_prefetch_max_bytes_per_step > 0
                        and issued_bytes + size_bytes
                        > self.uvm_weight_prefetch_max_bytes_per_step
                    ):
                        rejected_records += 1
                        self._append_uvm_weight_prefetch_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "step": self._uvm_moe_trace_step,
                                "mode": self.uvm_weight_prefetch_mode,
                                "action": "budget_reject",
                                "role": role,
                                "residency_region_id": region_id,
                                "residency_allowed": residency_decision.allowed,
                                "residency_would_deny": (
                                    residency_decision.would_deny
                                ),
                                "residency_reason": residency_decision.reason,
                                "residency_victims": list(
                                    residency_decision.victims
                                ),
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "issued_bytes_before": issued_bytes,
                                "max_bytes_per_step": (
                                    self.uvm_weight_prefetch_max_bytes_per_step
                                ),
                            }
                        )
                        continue

                    action = "trace_prefetch_candidate"
                    success = False
                    coordinator_decision = request_uvm_pool_action(
                        pool="weights",
                        action="expert_prefetch_gpu",
                        requested_bytes=size_bytes,
                        scope_key=f"moe_step:{os.getpid()}:{self._uvm_moe_trace_step}",
                        metadata={
                            "stage": "stage_i",
                            "stage_l1_residency_region_id": region_id,
                            "layer_name": self.layer_name,
                            "role": role,
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "mode": self.uvm_weight_prefetch_mode,
                        },
                    )
                    if not coordinator_decision.allowed:
                        rejected_records += 1
                        self._append_uvm_weight_prefetch_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "step": self._uvm_moe_trace_step,
                                "mode": self.uvm_weight_prefetch_mode,
                                "action": "coordinator_reject",
                                "role": role,
                                "residency_region_id": region_id,
                                "residency_allowed": residency_decision.allowed,
                                "residency_would_deny": (
                                    residency_decision.would_deny
                                ),
                                "residency_reason": residency_decision.reason,
                                "residency_victims": list(
                                    residency_decision.victims
                                ),
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "pool": coordinator_decision.pool,
                                "coordinator_mode": coordinator_decision.mode,
                                "coordinator_reason": coordinator_decision.reason,
                            }
                        )
                        continue
                    if self.uvm_weight_prefetch_mode == "prefetch":
                        success = prefetch_range_to_device(ptr, size_bytes, device)
                        action = "prefetch_issued" if success else "prefetch_skipped"
                        if success:
                            mark_uvm_residency_state(
                                region_id=region_id,
                                state="prefetching_gpu",
                                metadata=residency_metadata,
                            )

                    issued_records += 1
                    issued_bytes += size_bytes
                    self._append_uvm_weight_prefetch_trace(
                        {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "layer_name": self.layer_name,
                            "step": self._uvm_moe_trace_step,
                            "mode": self.uvm_weight_prefetch_mode,
                            "action": action,
                            "role": role,
                            "residency_region_id": region_id,
                            "residency_allowed": residency_decision.allowed,
                            "residency_would_deny": (
                                residency_decision.would_deny
                            ),
                            "residency_reason": residency_decision.reason,
                            "residency_victims": list(
                                residency_decision.victims
                            ),
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "ptr": f"0x{ptr:x}",
                            "bytes": size_bytes,
                            "device": device,
                            "success": success,
                            "coordinator_allowed": coordinator_decision.allowed,
                            "coordinator_would_deny": (
                                coordinator_decision.would_deny
                            ),
                            "coordinator_reason": coordinator_decision.reason,
                        }
                    )

            if issued_records or rejected_records:
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_prefetch_mode,
                        "action": "step_summary",
                        "active_global_experts": [
                            global_expert_id for global_expert_id, _ in expert_pairs
                        ],
                        "active_local_experts": [
                            local_expert_id for _, local_expert_id in expert_pairs
                        ],
                        "issued_records": issued_records,
                        "issued_bytes": issued_bytes,
                        "attempted_bytes": attempted_bytes,
                        "budget_reject_records": rejected_records,
                        "max_bytes_per_step": (
                            self.uvm_weight_prefetch_max_bytes_per_step
                        ),
                    }
                )
        except Exception:
            logger.debug("Failed to run UVM expert prefetch", exc_info=True)

    def _maybe_offload_uvm_cold_expert_weights(self, topk_ids: torch.Tensor) -> None:
        if not self.uvm_weight_offload_enable:
            return
        self._refresh_uvm_weight_action_plans()
        if not self.uvm_weight_offload_plan_keys:
            return
        try:
            layer_id = self._uvm_layer_id()
            if layer_id is None:
                return
            active_global_experts = {
                global_expert_id
                for global_expert_id, _ in self._uvm_active_expert_pairs(topk_ids)
            }
            expert_pairs = [
                (global_expert_id, local_expert_id)
                for global_expert_id, local_expert_id in self._uvm_plan_local_experts(
                    self.uvm_weight_offload_plan_keys
                )
                if (layer_id, global_expert_id)
                not in self._uvm_weight_offload_completed_keys
                and global_expert_id not in active_global_experts
            ]
            if self.uvm_weight_offload_max_experts_per_layer > 0:
                expert_pairs = expert_pairs[
                    : self.uvm_weight_offload_max_experts_per_layer
                ]
            if not expert_pairs:
                return

            issued_records = 0
            attempted_bytes = 0
            issued_bytes = 0
            rejected_records = 0

            for global_expert_id, local_expert_id in expert_pairs:
                expert_issued = False
                for role, ptr, size_bytes in self._uvm_expert_weight_slices(
                    local_expert_id,
                    self.uvm_weight_offload_target_roles,
                ):
                    region_id = self._uvm_expert_weight_region_id(
                        layer_id,
                        global_expert_id,
                        role,
                    )
                    residency_metadata = (
                        self._uvm_expert_weight_residency_metadata(
                            layer_id=layer_id,
                            global_expert_id=global_expert_id,
                            local_expert_id=local_expert_id,
                            role=role,
                            ptr=ptr,
                            size_bytes=size_bytes,
                            active_this_step=False,
                            source_action="offload",
                        )
                    )
                    existing_region = get_uvm_residency_region_snapshot(region_id)
                    register_uvm_residency_region(
                        pool="weights",
                        region_id=region_id,
                        ptr=ptr,
                        bytes=size_bytes,
                        state=(
                            str(existing_region.get("state"))
                            if existing_region is not None
                            else "gpu"
                        ),
                        evictable=True,
                        active=False,
                        metadata=residency_metadata,
                    )
                    attempted_bytes += size_bytes
                    if (
                        self.uvm_weight_offload_max_bytes_per_step > 0
                        and issued_bytes + size_bytes
                        > self.uvm_weight_offload_max_bytes_per_step
                    ):
                        rejected_records += 1
                        self._append_uvm_weight_prefetch_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "step": self._uvm_moe_trace_step,
                                "mode": self.uvm_weight_offload_mode,
                                "action": "offload_budget_reject",
                                "role": role,
                                "residency_region_id": region_id,
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "issued_bytes_before": issued_bytes,
                                "max_bytes_per_step": (
                                    self.uvm_weight_offload_max_bytes_per_step
                                ),
                            }
                        )
                        continue

                    action = "trace_offload_candidate"
                    success = False
                    coordinator_decision = request_uvm_pool_action(
                        pool="weights",
                        action=f"expert_{self.uvm_weight_offload_mode}",
                        requested_bytes=size_bytes,
                        scope_key=f"moe_step:{os.getpid()}:{self._uvm_moe_trace_step}",
                        metadata={
                            "stage": "stage_i",
                            "stage_l1_residency_region_id": region_id,
                            "layer_name": self.layer_name,
                            "role": role,
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "mode": self.uvm_weight_offload_mode,
                        },
                    )
                    if not coordinator_decision.allowed:
                        rejected_records += 1
                        self._append_uvm_weight_prefetch_trace(
                            {
                                "timestamp": time.time(),
                                "pid": os.getpid(),
                                "layer_name": self.layer_name,
                                "step": self._uvm_moe_trace_step,
                                "mode": self.uvm_weight_offload_mode,
                                "action": "offload_coordinator_reject",
                                "role": role,
                                "residency_region_id": region_id,
                                "global_expert_id": global_expert_id,
                                "local_expert_id": local_expert_id,
                                "ptr": f"0x{ptr:x}",
                                "bytes": size_bytes,
                                "pool": coordinator_decision.pool,
                                "coordinator_mode": coordinator_decision.mode,
                                "coordinator_reason": coordinator_decision.reason,
                            }
                        )
                        continue
                    if self.uvm_weight_offload_mode == "advise_cpu":
                        success = advise_range_preferred_location(ptr, size_bytes, -1)
                        action = (
                            "offload_advise_cpu_issued"
                            if success
                            else "offload_skipped"
                        )
                    elif self.uvm_weight_offload_mode == "prefetch_cpu":
                        success = prefetch_range_to_cpu(ptr, size_bytes)
                        action = (
                            "offload_prefetch_cpu_issued"
                            if success
                            else "offload_skipped"
                        )
                    if success:
                        mark_uvm_residency_state(
                            region_id=region_id,
                            state="prefetching_cpu",
                            metadata=residency_metadata,
                        )

                    issued_records += 1
                    issued_bytes += size_bytes
                    expert_issued = True
                    self._append_uvm_weight_prefetch_trace(
                        {
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "layer_name": self.layer_name,
                            "step": self._uvm_moe_trace_step,
                            "mode": self.uvm_weight_offload_mode,
                            "action": action,
                            "role": role,
                            "residency_region_id": region_id,
                            "global_expert_id": global_expert_id,
                            "local_expert_id": local_expert_id,
                            "ptr": f"0x{ptr:x}",
                            "bytes": size_bytes,
                            "device": -1,
                            "success": success,
                            "coordinator_allowed": coordinator_decision.allowed,
                            "coordinator_would_deny": (
                                coordinator_decision.would_deny
                            ),
                            "coordinator_reason": coordinator_decision.reason,
                        }
                    )
                if expert_issued:
                    self._uvm_weight_offload_completed_keys.add(
                        (layer_id, global_expert_id)
                    )

            if issued_records or rejected_records:
                self._append_uvm_weight_prefetch_trace(
                    {
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "layer_name": self.layer_name,
                        "step": self._uvm_moe_trace_step,
                        "mode": self.uvm_weight_offload_mode,
                        "action": "offload_step_summary",
                        "offload_global_experts": [
                            global_expert_id for global_expert_id, _ in expert_pairs
                        ],
                        "offload_local_experts": [
                            local_expert_id for _, local_expert_id in expert_pairs
                        ],
                        "issued_records": issued_records,
                        "issued_bytes": issued_bytes,
                        "attempted_bytes": attempted_bytes,
                        "budget_reject_records": rejected_records,
                        "max_bytes_per_step": (
                            self.uvm_weight_offload_max_bytes_per_step
                        ),
                    }
                )
        except Exception:
            logger.debug("Failed to run UVM expert offload", exc_info=True)

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
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                return (
                    reduce_output(fused_output)
                    + zero_expert_result[..., :og_hidden_states]
                )
            else:
                return reduce_output(fused_output)
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            return (
                reduce_output(shared_output),
                reduce_output(fused_output),
            )

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

        if self.shared_experts is None:
            return full_fused_final_hidden_states
        else:
            return (full_shared_final_hidden_states, full_fused_final_hidden_states)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

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
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

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

            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

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
                return (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                return (combine_output(final_hidden_states), zero_expert_result)
            else:
                return combine_output(final_hidden_states)

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
