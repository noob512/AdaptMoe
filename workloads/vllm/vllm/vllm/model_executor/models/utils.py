# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import itertools
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig
from typing_extensions import deprecated

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.online_quantization import (
    support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import supports_any_eagle
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_cuda_view_from_cpu_tensor,
)

logger = init_logger(__name__)

WeightsMapping = Mapping[str, str | None]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """
    Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting the
    environment variable `VLLM_LOGGING_LEVEL=DEBUG`.
    """

    # Models trained using early version ColossalAI or quantized by
    # GPTQModel may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(
        self, module: nn.Module, child_params: dict[str, torch.Tensor]
    ):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                msg = (
                    f"There is no module or parameter named '{prefix}' "
                    f"in {type(self.module).__name__}"
                )
                raise ValueError(msg)

    @support_quantized_model_reload_from_hp_weights
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: PretrainedConfig | None = None,
    architectures: list[str] | None = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """
    from vllm.model_executor.model_loader.utils import initialize_model

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = vllm_config.model_config.hf_config

    if hf_config is not None:
        vllm_config = vllm_config.with_hf_config(hf_config, architectures=architectures)

    return initialize_model(vllm_config=vllm_config, prefix=prefix)


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor: ...


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: Literal[True],
) -> torch.Tensor: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor: ...


def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor:
    """
    Flatten the `B` and `N` dimensions of batched multimodal inputs.

    The input tensor should have shape `(B, N, ...)`.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    ranges: list[list[int]] = [[] for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


@deprecated(
    "`merge_multimodal_embeddings` has been replaced with "
    "`SupportsMultiModal.embed_input_ids` and will be "
    "removed in v0.12."
)
def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: int | list[int],
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    `placeholder_token_id` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the `input_ids` MUST MATCH the order of
    their embeddings in `multimodal_embeddings` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates `inputs_embeds` in place.
    """
    if isinstance(placeholder_token_id, list):
        is_multimodal = isin_list(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id

    return _merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


def isin_list(
    elements: torch.Tensor,
    test_elements_list: list[int],
) -> torch.Tensor:
    test_elements = torch.tensor(
        test_elements_list,
        pin_memory=is_pin_memory_available(),
    ).to(device=elements.device, non_blocking=True)

    return torch.isin(elements, test_elements)


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0
_CPU_OFFLOAD_ORDER = 0
_CPU_OFFLOAD_EXACT_NAMES_FILE = ""
_CPU_OFFLOAD_EXACT_NAMES: frozenset[str] | None = None
_CPU_OFFLOAD_EXACT_PENDING: set[str] = set()
_CPU_OFFLOAD_EXACT_SEEN: set[str] = set()
_CPU_OFFLOAD_STATIC_BANK = False
_CPU_OFFLOAD_STATIC_BANKS: dict[tuple[Any, ...], torch.Tensor] = {}
_CPU_OFFLOAD_STATIC_BANK_BYTES = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS = False
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN: dict[str, "_GraphStaticAliasNamePlan"] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = ""
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER: dict[int, tuple[int, ...]] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS: dict[
    tuple[str, str, int], "_GraphStaticAliasArena"
] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = False
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = "vmm"
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LOCK = threading.RLock()
_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS: dict[
    tuple[str, str, int], "_UvmOriginalAliasArena"
] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS: dict[str, "_UvmOriginalAliasBinding"] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "disabled"
_CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = "off"
_CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY: str | None = None
_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
_CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
_CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
_CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
_CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT: tuple[int, ...] = ()
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT: tuple[int, ...] = ()
_CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK = threading.RLock()
_CPU_OFFLOAD_OWNER_THREAD_ID: int | None = None
_CPU_OFFLOAD_OWNER_DEVICE_INDEX: int | None = None
_CPU_OFFLOAD_OWNER_STREAM_ID: int | None = None


@dataclass(frozen=True)
class _GraphStaticAliasNamePlan:
    layer_container: str
    layer_index: int
    local_name: str
    bank_id: int
    consumer_index: int
    consumer_count: int


@dataclass
class _GraphStaticAliasArena:
    allocation: Any
    upload: torch.Tensor
    signature: tuple[Any, ...]
    requested_bytes: int
    aligned_bytes: int
    backing_bytes: int
    bindings: list["_GraphStaticAliasBinding"] = field(default_factory=list)


@dataclass(frozen=True)
class _GraphStaticAliasBinding:
    arena: _GraphStaticAliasArena
    consumer: torch.Tensor
    plan: _GraphStaticAliasNamePlan


@dataclass
class _UvmOriginalAliasArena:
    upload_storage: torch.Tensor
    upload: torch.Tensor
    signature: tuple[Any, ...]
    requested_bytes: int
    aligned_bytes: int
    backing_layer_id: int
    pte_alignment: int
    pte_flags: int
    region_bytes: int
    packed_slice_bytes: int = 0
    consumers: dict[int, "_UvmOriginalAliasBinding"] = field(default_factory=dict)


@dataclass
class _UvmOriginalAliasBinding:
    full_parameter_name: str
    arena: _UvmOriginalAliasArena
    consumer: torch.Tensor
    parameter: Any
    plan: _GraphStaticAliasNamePlan
    requested_bytes: int
    parameter_bytes: int = 0
    expert_ids: tuple[int, ...] | None = None
    expert_stride_bytes: int = 0
    consumer_ranges: tuple[tuple[int, int], ...] = ()
    source: torch.Tensor | None = None


def _parse_graph_static_alias_name(
    full_parameter_name: str,
) -> tuple[str, int, str]:
    parts = full_parameter_name.split(".")
    candidates = [
        index
        for index in range(len(parts) - 2)
        if parts[index] == "layers" and parts[index + 1].isdigit()
    ]
    if not candidates:
        raise ValueError(
            "Graph static-alias parameter names must contain "
            f"'.layers.<integer>.': {full_parameter_name!r}"
        )
    layer_marker = candidates[-1]
    local_parts = parts[layer_marker + 2 :]
    if not local_parts:
        raise ValueError(
            "Graph static-alias parameter name has no layer-local suffix: "
            f"{full_parameter_name!r}"
        )
    return (
        ".".join(parts[: layer_marker + 1]),
        int(parts[layer_marker + 1]),
        ".".join(local_parts),
    )


def _build_graph_static_alias_plan(
    exact_names: frozenset[str],
) -> dict[str, _GraphStaticAliasNamePlan]:
    grouped: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    parsed: dict[str, tuple[str, int, str]] = {}
    for full_name in exact_names:
        layer_container, layer_index, local_name = _parse_graph_static_alias_name(
            full_name
        )
        parsed[full_name] = (layer_container, layer_index, local_name)
        bank_id = layer_index & 1
        grouped.setdefault((layer_container, local_name, bank_id), []).append(
            (layer_index, full_name)
        )

    result: dict[str, _GraphStaticAliasNamePlan] = {}
    for (layer_container, local_name, bank_id), members in grouped.items():
        members.sort()
        layer_indices = [layer_index for layer_index, _ in members]
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError(
                "Graph static-alias exact-name plan contains more than one "
                f"{local_name!r} parameter for the same layer and bank"
            )
        consumer_count = len(members)
        for consumer_index, (layer_index, full_name) in enumerate(members):
            result[full_name] = _GraphStaticAliasNamePlan(
                layer_container=layer_container,
                layer_index=layer_index,
                local_name=local_name,
                bank_id=bank_id,
                consumer_index=consumer_index,
                consumer_count=consumer_count,
            )
    assert result.keys() == parsed.keys()
    return result


def _cpu_offload_manifest_file() -> str:
    return os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_FILE", "")


def _append_cpu_offload_manifest(record: dict[str, Any]) -> None:
    manifest_file = _cpu_offload_manifest_file()
    if not manifest_file:
        return
    try:
        os.makedirs(os.path.dirname(manifest_file) or ".", exist_ok=True)
        with open(manifest_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning(
            "Failed to append vLLM CPU offload manifest %s: %s",
            manifest_file,
            exc,
        )


def _load_cpu_offload_exact_names(exact_names_file: str) -> frozenset[str]:
    try:
        with open(exact_names_file, encoding="utf-8") as handle:
            contents = handle.read()
    except OSError as exc:
        raise ValueError(
            "Unable to read VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE "
            f"{exact_names_file!r}: {exc}"
        ) from exc

    stripped = contents.strip()
    if not stripped:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must contain at least one "
            "full parameter name"
        )

    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE starts with '[' but is "
                f"not a valid JSON string list: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(name, str) for name in parsed
        ):
            raise ValueError(
                "JSON VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must be a list of strings"
            )
        names = parsed
    else:
        names = [line.strip() for line in contents.splitlines() if line.strip()]

    if not names or any(not name.strip() for name in names):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE contains an empty parameter name"
        )
    if any(name != name.strip() for name in names):
        raise ValueError(
            "JSON parameter names in VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must not "
            "have leading or trailing whitespace"
        )
    if len(names) != len(set(names)):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE contains duplicate parameter names"
        )
    return frozenset(names)


def _load_graph_static_alias_expert_plan(
    path: str,
    name_plan: Mapping[str, _GraphStaticAliasNamePlan],
) -> dict[int, tuple[int, ...]]:
    """Load one frozen, equal-width expert slice plan.

    The exact-name file still selects complete ``w13_weight``/``w2_weight``
    Parameters so those allocations are born in managed storage.  This
    optional plan narrows the static aliases and H2D traffic to expert slices
    inside those Parameters.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Unable to load "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE "
            f"{path!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Static-alias expert plan must be a JSON object")

    selected: dict[int, list[int]] = {}
    selected_by_layer = payload.get("selected_by_layer")
    if selected_by_layer is not None:
        if not isinstance(selected_by_layer, dict):
            raise ValueError("selected_by_layer must be a JSON object")
        for raw_layer, raw_experts in selected_by_layer.items():
            try:
                layer_id = int(raw_layer)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid selected_by_layer key {raw_layer!r}"
                ) from exc
            if isinstance(raw_experts, (str, bytes)) or not isinstance(
                raw_experts, list
            ):
                raise ValueError(
                    f"selected_by_layer[{raw_layer!r}] must be a list"
                )
            selected[layer_id] = list(raw_experts)
    else:
        records = payload.get("offload_plan")
        if not isinstance(records, list):
            raise ValueError(
                "Static-alias expert plan requires selected_by_layer or "
                "offload_plan"
            )
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"offload_plan[{index}] is not an object")
            layer_id = record.get("layer_id")
            expert_id = record.get("expert_id")
            if (
                not isinstance(layer_id, int)
                or isinstance(layer_id, bool)
                or not isinstance(expert_id, int)
                or isinstance(expert_id, bool)
            ):
                raise ValueError(
                    f"offload_plan[{index}] requires integer layer_id/expert_id"
                )
            selected.setdefault(int(layer_id), []).append(int(expert_id))

    expected_layers = {entry.layer_index for entry in name_plan.values()}
    if set(selected) != expected_layers:
        missing = sorted(expected_layers - set(selected))
        unexpected = sorted(set(selected) - expected_layers)
        raise ValueError(
            "Static-alias expert plan layers differ from the exact-name plan: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    normalized: dict[int, tuple[int, ...]] = {}
    widths: set[int] = set()
    for layer_id, raw_experts in sorted(selected.items()):
        if any(
            not isinstance(expert_id, int)
            or isinstance(expert_id, bool)
            or expert_id < 0
            for expert_id in raw_experts
        ):
            raise ValueError(
                f"Layer {layer_id} contains an invalid expert id"
            )
        experts = tuple(sorted(int(value) for value in raw_experts))
        if not experts or len(experts) != len(set(experts)):
            raise ValueError(
                f"Layer {layer_id} expert ids must be non-empty and unique"
            )
        normalized[layer_id] = experts
        widths.add(len(experts))
    if len(widths) != 1:
        raise ValueError(
            "Every layer in a static-alias expert plan must select the same "
            f"number of experts, got widths={sorted(widths)!r}"
        )
    return normalized


def _configured_graph_static_alias_backend() -> str:
    raw = (
        os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm")
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "": "vmm",
        "vmm": "vmm",
        "cuda_vmm": "vmm",
        "uvm": "uvm_original_va",
        "original_va": "uvm_original_va",
        "uvm_original_va": "uvm_original_va",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND must be 'vmm' "
            f"or 'uvm_original_va', got {raw!r}"
        ) from exc


def _configured_uvm_original_va_phase_mode() -> str:
    raw = (
        os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "off")
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "": "off",
        "off": "off",
        "disabled": "off",
        "prefill_remote_decode": "prefill_remote_decode",
        "alias_prefill_remote_decode": "prefill_remote_decode",
        "hybrid": "prefill_remote_decode",
        "prefill_graph_remote_decode": "prefill_graph_remote_decode",
        "alias_graph_prefill_remote_graph_decode": (
            "prefill_graph_remote_decode"
        ),
        "dual_topology_graph": "prefill_graph_remote_decode",
        "remote_only": "remote_only",
        "remote": "remote_only",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE must be 'off', "
            "'prefill_remote_decode', 'prefill_graph_remote_decode', or "
            f"'remote_only', got {raw!r}"
        ) from exc


def _configure_cpu_offload_exact_names(max_bytes: int) -> None:
    global _CPU_OFFLOAD_EXACT_NAMES_FILE
    global _CPU_OFFLOAD_EXACT_NAMES, _CPU_OFFLOAD_EXACT_PENDING
    global _CPU_OFFLOAD_EXACT_SEEN, _CPU_OFFLOAD_STATIC_BANK
    global _CPU_OFFLOAD_STATIC_BANKS, _CPU_OFFLOAD_STATIC_BANK_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
    global _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY
    global _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_OWNER_THREAD_ID, _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    global _CPU_OFFLOAD_OWNER_STREAM_ID

    if (
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
        or _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
        or (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
            in ("finalizing", "sealed", "poisoned", "cleanup_pending")
        )
    ):
        raise RuntimeError(
            "Cannot reconfigure CPU offload while a UVM original-VA "
            "static-alias session or cleanup is active; destroy its Graphs "
            "and session first"
        )

    exact_names_file = os.environ.get("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", "").strip()
    static_bank = _strict_env_flag("VLLM_CPU_OFFLOAD_STATIC_BANK", False)
    graph_static_alias = _strict_env_flag("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", False)
    graph_static_alias_eager = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", False
    )
    graph_static_alias_backend = _configured_graph_static_alias_backend()
    original_va_phase_mode = _configured_uvm_original_va_phase_mode()
    graph_static_alias_expert_plan_file = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", ""
    ).strip()

    _CPU_OFFLOAD_EXACT_NAMES_FILE = ""
    _CPU_OFFLOAD_EXACT_NAMES = None
    _CPU_OFFLOAD_EXACT_PENDING = set()
    _CPU_OFFLOAD_EXACT_SEEN = set()
    _CPU_OFFLOAD_STATIC_BANK = False
    _CPU_OFFLOAD_STATIC_BANKS = {}
    _CPU_OFFLOAD_STATIC_BANK_BYTES = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS = False
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = ""
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = False
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = "vmm"
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS = {}
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = {}
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "disabled"
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = []
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = "off"
    _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = None
    _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
    _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT = ()
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None
    _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY = []
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY = []
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = ()
    _CPU_OFFLOAD_OWNER_THREAD_ID = None
    _CPU_OFFLOAD_OWNER_DEVICE_INDEX = None
    _CPU_OFFLOAD_OWNER_STREAM_ID = None

    if (static_bank or graph_static_alias) and not exact_names_file:
        raise ValueError(
            "Each of VLLM_CPU_OFFLOAD_STATIC_BANK=1 and "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 requires "
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE"
        )
    if static_bank and graph_static_alias:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_STATIC_BANK=1 and "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 are mutually exclusive"
        )
    if graph_static_alias_eager and not graph_static_alias:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1 requires "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
        )
    if graph_static_alias_expert_plan_file and not (
        graph_static_alias and graph_static_alias_backend == "uvm_original_va"
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE requires "
            "the uvm_original_va graph static-alias backend"
        )
    if graph_static_alias_eager and original_va_phase_mode != "off":
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1 is only valid with "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        )
    if not graph_static_alias and graph_static_alias_backend != "vmm":
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va "
            "requires VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
        )
    if original_va_phase_mode != "off" and not (
        graph_static_alias and graph_static_alias_backend == "uvm_original_va"
    ):
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE requires the "
            "uvm_original_va graph static-alias backend"
        )
    if original_va_phase_mode != "off" and _strict_env_flag(
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", False
    ):
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE and the legacy "
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ path are mutually exclusive"
        )
    if not exact_names_file:
        return
    if _normalized_cpu_offload_backend() != "copy":
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE is supported only with "
            "VLLM_CPU_OFFLOAD_BACKEND=copy"
        )
    if max_bytes <= 0:
        raise ValueError(
            "Exact-name CPU offload requires a positive cpu_offload_gb budget"
        )

    exact_names = _load_cpu_offload_exact_names(exact_names_file)
    _CPU_OFFLOAD_EXACT_NAMES_FILE = exact_names_file
    _CPU_OFFLOAD_EXACT_NAMES = exact_names
    _CPU_OFFLOAD_EXACT_PENDING = set(exact_names)
    _CPU_OFFLOAD_STATIC_BANK = static_bank
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS = graph_static_alias
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = graph_static_alias_backend
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = original_va_phase_mode
    if graph_static_alias:
        if _strict_env_flag("VLLM_CPU_OFFLOAD_COPY_SYNC", False):
            raise ValueError(
                "VLLM_CPU_OFFLOAD_COPY_SYNC=1 is capture-unsafe with "
                "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
            )
        if graph_static_alias_backend == "uvm_original_va" and not _strict_env_flag(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED", False
        ):
            raise ValueError(
                "The uvm_original_va graph static-alias backend requires "
                "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 so expert parameters are "
                "born in managed storage and retain their allocation VAs"
            )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN = _build_graph_static_alias_plan(
            exact_names
        )
        if graph_static_alias_backend == "uvm_original_va":
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "collecting"
            if graph_static_alias_expert_plan_file:
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = (
                    graph_static_alias_expert_plan_file
                )
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = (
                    _load_graph_static_alias_expert_plan(
                        graph_static_alias_expert_plan_file,
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN,
                    )
                )


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES, _CPU_OFFLOAD_ORDER
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes
    _CPU_OFFLOAD_ORDER = 0
    _configure_cpu_offload_exact_names(max_bytes)
    manifest_file = _cpu_offload_manifest_file()
    if manifest_file and os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_RESET", "1") == "1":
        try:
            os.makedirs(os.path.dirname(manifest_file) or ".", exist_ok=True)
            with open(manifest_file, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "cpu_offload_config",
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "max_offload_bytes": int(max_bytes),
                            "copy_exact_names": bool(
                                _CPU_OFFLOAD_EXACT_NAMES is not None
                            ),
                            "copy_exact_name_count": len(
                                _CPU_OFFLOAD_EXACT_NAMES or ()
                            ),
                            "copy_exact_names_file": (
                                _CPU_OFFLOAD_EXACT_NAMES_FILE or None
                            ),
                            "copy_static_bank": bool(_CPU_OFFLOAD_STATIC_BANK),
                            "graph_static_alias": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
                            "graph_static_alias_backend": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
                                if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
                                else None
                            ),
                            "graph_static_alias_expert_plan_file": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
                                or None
                            ),
                            "graph_static_alias_selected_expert_count": sum(
                                len(expert_ids)
                                for expert_ids in (
                                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.values()
                                )
                            ),
                            "uvm_original_va_phase_mode": (
                                _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
                                if cpu_offload_uvm_original_va_static_alias_enabled()
                                else "off"
                            ),
                            "graph_static_alias_bank_count": (
                                2 if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS else 0
                            ),
                            "copy_bank_bytes": 0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.warning(
                "Failed to initialize vLLM CPU offload manifest %s: %s",
                manifest_file,
                exc,
            )


def _cpu_offload_backend() -> str:
    return os.environ.get("VLLM_CPU_OFFLOAD_BACKEND", "uva").strip().lower()


def _normalized_cpu_offload_backend() -> str:
    backend = _cpu_offload_backend()
    if backend in ("", "native", "native_uva", "pinned_uva"):
        return "uva"
    if backend in ("copy_on_forward", "streaming_copy"):
        return "copy"
    return backend


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _strict_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 0 or 1, got {raw!r}")


def _copy_exact_names_enabled() -> bool:
    return _CPU_OFFLOAD_EXACT_NAMES is not None


def cpu_offload_graph_static_alias_enabled() -> bool:
    """Return whether either experimental graph static-alias path is selected."""
    return _CPU_OFFLOAD_GRAPH_STATIC_ALIAS


def cpu_offload_uvm_original_va_static_alias_enabled() -> bool:
    return bool(
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
        and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
    )


def cpu_offload_uvm_original_va_phase_mode() -> str:
    """Return the frozen original-VA serving topology policy."""
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return "off"
    return _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE


def use_uvm_expert_weight_allocation(
    prefix: str,
    *,
    managed_requested: bool,
) -> bool:
    """Select managed allocation only for complete original-VA expert pairs.

    The legacy UVM paths intentionally allocate every FusedMoE main weight
    from the local managed pool.  The original-VA static-alias path is more
    restrictive: only parameters present in its frozen exact-name plan may be
    managed, because unaliased managed expert weights cannot fault safely
    while automatic UVM servicing is disabled during CUDA Graph execution.
    """
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return bool(managed_requested)

    if not managed_requested:
        raise RuntimeError(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED changed after the UVM "
            "original-VA static-alias configuration was frozen"
        )
    if not prefix:
        raise ValueError(
            "UVM original-VA expert allocation requires a non-empty "
            "FusedMoE module prefix"
        )
    if _CPU_OFFLOAD_EXACT_NAMES is None:
        raise RuntimeError(
            "UVM original-VA expert allocation was queried before the "
            "exact-name configuration was initialized"
        )

    expected = {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
    }
    descendant_prefix = f"{prefix}."
    selected = {
        name
        for name in _CPU_OFFLOAD_EXACT_NAMES
        if name.startswith(descendant_prefix)
    }
    if not selected:
        return False
    if selected != expected:
        missing = sorted(expected - selected)
        unexpected = sorted(selected - expected)
        raise ValueError(
            "UVM original-VA FusedMoE selection must contain exactly one "
            "complete w13_weight/w2_weight pair per selected layer; "
            f"prefix={prefix!r}, missing={missing!r}, unexpected={unexpected!r}"
        )
    return True


def get_cpu_offload_graph_static_alias_diagnostics() -> dict[str, Any]:
    """Return read-only process-local counters and immutable alias VAs."""
    arenas: list[dict[str, Any]] = []
    ordered_uvm_bindings = sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values(),
        key=lambda binding: binding.full_parameter_name,
    )
    original_vas = tuple(
        int(binding.consumer.data_ptr()) for binding in ordered_uvm_bindings
    )
    source_vas = tuple(
        int(binding.source.data_ptr()) if binding.source is not None else None
        for binding in ordered_uvm_bindings
    )
    for (layer_container, local_name, bank_id), arena in sorted(
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.items()
    ):
        allocation = arena.allocation
        arenas.append(
            {
                "layer_container": layer_container,
                "local_name": local_name,
                "bank_id": int(bank_id),
                "token": int(allocation.token),
                "requested_bytes": int(arena.requested_bytes),
                "aligned_bytes": int(arena.aligned_bytes),
                "backing_bytes": int(arena.backing_bytes),
                "upload_va": int(allocation.upload_va),
                "consumer_vas": tuple(int(value) for value in allocation.consumer_vas),
            }
        )
    for (layer_container, local_name, bank_id), arena in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()
    ):
        ordered_consumers = [binding for _, binding in sorted(arena.consumers.items())]
        arenas.append(
            {
                "layer_container": layer_container,
                "local_name": local_name,
                "bank_id": int(bank_id),
                "backing_layer_id": int(arena.backing_layer_id),
                "requested_bytes": int(arena.requested_bytes),
                "aligned_bytes": int(arena.aligned_bytes),
                "backing_bytes": int(arena.aligned_bytes),
                "upload_va": int(arena.upload.data_ptr()),
                "consumer_vas": tuple(
                    int(binding.consumer.data_ptr()) for binding in ordered_consumers
                ),
                "consumer_ranges": tuple(
                    _uvm_original_alias_binding_consumer_ranges(binding)
                    for binding in ordered_consumers
                ),
                "expert_ids": tuple(
                    binding.expert_ids for binding in ordered_consumers
                ),
                "expert_stride_bytes": tuple(
                    int(binding.expert_stride_bytes)
                    for binding in ordered_consumers
                ),
                "selected_bytes": tuple(
                    int(binding.requested_bytes) for binding in ordered_consumers
                ),
                "parameter_bytes": tuple(
                    int(binding.parameter_bytes) for binding in ordered_consumers
                ),
                "original_vas": tuple(
                    int(binding.consumer.data_ptr()) for binding in ordered_consumers
                ),
                "source_vas": tuple(
                    int(binding.source.data_ptr())
                    if binding.source is not None
                    else None
                    for binding in ordered_consumers
                ),
                "source_ready_count": sum(
                    binding.source is not None for binding in arena.consumers.values()
                ),
                "pte_alignment": int(arena.pte_alignment),
                "pte_mode": "64K" if arena.pte_flags else "4K",
                "region_bytes": int(arena.region_bytes),
            }
        )
    return {
        "enabled": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
        "backend": (
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
            if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
            else None
        ),
        "lifecycle_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        "sealed": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed",
        "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
        "original_vas": original_vas,
        "source_vas": source_vas,
        "expert_plan_file": (
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE or None
        ),
        "expert_plan": tuple(
            (int(layer_id), tuple(expert_ids))
            for layer_id, expert_ids in sorted(
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.items()
            )
        ),
        "selected_expert_count": sum(
            len(expert_ids)
            for expert_ids in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.values()
        ),
        "selected_parameter_count": len(ordered_uvm_bindings),
        "selected_parameter_bytes_per_forward": sum(
            int(binding.requested_bytes) for binding in ordered_uvm_bindings
        ),
        "full_parameter_bytes": sum(
            int(binding.parameter_bytes) for binding in ordered_uvm_bindings
        ),
        "consumer_range_count": sum(
            len(_uvm_original_alias_binding_consumer_ranges(binding))
            for binding in ordered_uvm_bindings
        ),
        "hot_range_count": sum(
            len(_uvm_original_alias_binding_hot_ranges(binding))
            for binding in ordered_uvm_bindings
        ),
        "capture_phase": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE),
        "forward_call_count": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS),
        "arena_count": len(arenas),
        "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
        "owner_thread_id": _CPU_OFFLOAD_OWNER_THREAD_ID,
        "owner_device_index": _CPU_OFFLOAD_OWNER_DEVICE_INDEX,
        "last_python_stream_id": _CPU_OFFLOAD_OWNER_STREAM_ID,
        "arenas": arenas,
        "preallocate_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT,
        "map_results": tuple(_CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS),
        "seal_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT,
        "cold_residency_result": (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
        ),
        "hot_residency_result": (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
        ),
        "destroy_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
        "slot_close_result": (_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT),
        "cleanup_pending": (_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"),
        "cleanup_unresolved": bool(_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED),
        "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
        "capture_topology": _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY,
        "capture_success_count": int(_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT),
        "remote_graph_captured": bool(_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED),
        "remote_graph_certificate": _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE,
        "alias_graph_captured": bool(_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED),
        "alias_graphs_destroyed": bool(
            _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
        ),
        "alias_graph_certificate": _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE,
        "dual_graph_partition_certificate": (
            _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
        ),
        "remote_graph_remaining_count": int(
            _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
        ),
        "capture_forward_count": int(_CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT),
        "capture_actual_h2d_bytes": int(
            _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
        ),
        "remote_map_active": bool(_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE),
        "remote_map_result": _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT,
        "remote_map_history": tuple(_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY),
        "phase_transition_history": tuple(
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
        ),
        "remote_transition_history": tuple(
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
        ),
        "original_va_snapshot": _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT,
        "active_driver_backing_bytes": int(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed"
            else 0
        ),
    }


def validate_cpu_offload_graph_static_alias_runtime(
    vllm_config: VllmConfig,
) -> None:
    """Fail closed on execution modes that can reuse one bank concurrently."""
    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return

    parallel_config = vllm_config.parallel_config
    unsupported: list[str] = []
    if parallel_config.tensor_parallel_size != 1:
        unsupported.append(
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}"
        )
    if parallel_config.pipeline_parallel_size != 1:
        unsupported.append(
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}"
        )
    if parallel_config.data_parallel_size != 1:
        unsupported.append(f"data_parallel_size={parallel_config.data_parallel_size}")
    if parallel_config.enable_dbo:
        unsupported.append("dual-batch overlap")

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    mode_name = getattr(cudagraph_mode, "name", str(cudagraph_mode))
    allowed_cudagraph_modes = {"FULL", "FULL_DECODE_ONLY", "FULL_SEPARATE"}
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != "off":
        allowed_cudagraph_modes.add("NONE")
    if _strict_env_flag("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", False):
        allowed_cudagraph_modes.add("NONE")
    if mode_name not in allowed_cudagraph_modes:
        unsupported.append(f"cudagraph_mode={mode_name}")
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        == "prefill_graph_remote_decode"
        and mode_name != "FULL_SEPARATE"
    ):
        unsupported.append(
            "prefill_graph_remote_decode requires cudagraph_mode=FULL_SEPARATE"
        )
    if (
        mode_name == "FULL_SEPARATE"
        and _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        != "prefill_graph_remote_decode"
    ):
        unsupported.append(
            "FULL_SEPARATE is reserved for prefill_graph_remote_decode"
        )
    compilation_mode = vllm_config.compilation_config.mode
    compilation_mode_name = getattr(compilation_mode, "name", str(compilation_mode))
    if compilation_mode_name != "NONE":
        unsupported.append(f"compilation_mode={compilation_mode_name}")
    if unsupported:
        raise RuntimeError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 currently requires "
            "single-GPU, non-overlapping FULL/FULL_DECODE_ONLY/FULL_SEPARATE "
            "execution "
            "(or NONE for an explicit original-VA phase/eager-alias experiment) "
            "with CompilationMode.NONE; "
            "unsupported: " + ", ".join(unsupported)
        )


def set_cpu_offload_graph_static_alias_capture_phase(
    active: bool,
    *,
    expected_topology: str | None = None,
) -> None:
    """Mark vLLM's warmup/capture phase for audit records."""
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return
    if active and cpu_offload_uvm_original_va_static_alias_enabled():
        if expected_topology is not None:
            required_state = expected_topology
        elif (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
            == "prefill_graph_remote_decode"
        ):
            required_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
            if required_state not in ("remote", "sealed"):
                raise RuntimeError(
                    "Dual-topology CUDA Graph capture requires either an "
                    "audited remote or sealed alias state; current state is "
                    f"{required_state!r}"
                )
        else:
            required_state = (
                "remote"
                if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
                in ("prefill_remote_decode", "remote_only")
                else "sealed"
            )
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != required_state:
            required_description = (
                "completely mapped and sealed"
                if required_state == "sealed"
                else "audited remote"
            )
            raise RuntimeError(
                "The UVM original-VA CUDA Graph capture topology must be "
                f"{required_description}; current state is "
                f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
            )
        if required_state == "remote" and not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
            raise RuntimeError(
                "Remote CUDA Graph capture requires an audited active remote "
                "READ mapping for every original parameter VA"
            )
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = bool(active)
    stream_id = int(torch.cuda.current_stream().cuda_stream)
    _append_cpu_offload_manifest(
        {
            "event": "cpu_offload_graph_static_alias_capture_phase",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "active": bool(active),
            "stream_id": stream_id,
            "forward_call_count": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS),
            "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
            "arena_count": (
                len(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS)
                + len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
            ),
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
            "lifecycle_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "sealed": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed",
            "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
        }
    )


@contextmanager
def cpu_offload_graph_static_alias_capture_context(
    *,
    expected_topology: str | None = None,
):
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
    global _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        yield
        return
    topology = (
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if cpu_offload_uvm_original_va_static_alias_enabled()
        else "static_alias"
    )
    if expected_topology is not None and topology != expected_topology:
        raise RuntimeError(
            "CUDA Graph capture topology differs from the requested phase: "
            f"expected={expected_topology!r}, actual={topology!r}"
        )
    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
    _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT = (
        _uvm_original_parameter_va_snapshot()
        if topology in ("remote", "sealed")
        else ()
    )
    succeeded = False
    set_cpu_offload_graph_static_alias_capture_phase(
        True,
        expected_topology=expected_topology,
    )
    try:
        yield
        if topology == "remote":
            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
                raise RuntimeError(
                    "Remote mapping became inactive during CUDA Graph capture"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT <= 0:
                raise RuntimeError(
                    "Remote CUDA Graph capture observed no selected expert "
                    "forward while the stream was capturing"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES != 0:
                raise RuntimeError(
                    "Remote CUDA Graph capture submitted an H2D copy"
                )
            if parameter_vas_after != _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT:
                raise RuntimeError(
                    "Original Parameter VAs changed during remote Graph capture"
                )
            _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = {
                "schema": "uvm_original_va_remote_graph_certificate_v1",
                "topology": "remote",
                "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
                "parameter_vas": parameter_vas_after,
                "capture_forward_count": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
                ),
                "actual_h2d_bytes": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
                ),
                "remote_map_requested_bytes": int(
                    (_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT or {}).get(
                        "requested_bytes", 0
                    )
                ),
            }
        elif topology == "sealed":
            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
                raise RuntimeError(
                    "Alias topology changed during CUDA Graph capture"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT <= 0:
                raise RuntimeError(
                    "Alias CUDA Graph capture observed no selected expert "
                    "forward while the stream was capturing"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES <= 0:
                raise RuntimeError(
                    "Alias CUDA Graph capture did not record an H2D upload"
                )
            if parameter_vas_after != _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT:
                raise RuntimeError(
                    "Original Parameter VAs changed during alias Graph capture"
                )
            _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = {
                "schema": "uvm_original_va_alias_graph_certificate_v1",
                "topology": "sealed",
                "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
                "parameter_vas": parameter_vas_after,
                "capture_forward_count": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
                ),
                "actual_h2d_bytes": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
                ),
                "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                "backing_bytes": int(
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                ),
            }
        succeeded = True
    finally:
        set_cpu_offload_graph_static_alias_capture_phase(False)
        if succeeded:
            _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = topology
            _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT += 1
            if topology == "remote":
                _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = True
            elif topology == "sealed" and (
                cpu_offload_uvm_original_va_static_alias_enabled()
            ):
                _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = True
                _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_graph_static_alias_capture_succeeded",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "topology": topology,
                    "capture_success_count": int(
                        _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
                    ),
                    "remote_graph_captured": bool(
                        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
                    ),
                    "alias_graph_captured": bool(
                        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
                    ),
                }
            )


def _validate_copy_exact_context(
    device: torch.device,
    *,
    check_stream: bool,
) -> None:
    global _CPU_OFFLOAD_OWNER_THREAD_ID, _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    global _CPU_OFFLOAD_OWNER_STREAM_ID

    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            "Exact-name CPU offload requires parameters on CUDA device 0"
        )
    device_count = int(torch.cuda.device_count())
    if device_count != 1:
        raise RuntimeError(
            "Exact-name CPU offload is fail-closed for a single GPU; "
            f"torch.cuda.device_count() returned {device_count}"
        )
    device_index = getattr(device, "index", None)
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    if int(device_index) != 0 or int(torch.cuda.current_device()) != 0:
        raise RuntimeError("Exact-name CPU offload supports only current CUDA device 0")

    thread_id = threading.get_ident()
    if _CPU_OFFLOAD_OWNER_THREAD_ID is None:
        _CPU_OFFLOAD_OWNER_THREAD_ID = thread_id
        _CPU_OFFLOAD_OWNER_DEVICE_INDEX = int(device_index)
    elif (
        thread_id != _CPU_OFFLOAD_OWNER_THREAD_ID
        or int(device_index) != _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    ):
        raise RuntimeError(
            "Exact-name CPU offload must be constructed and executed on one "
            "thread and CUDA device"
        )

    if not check_stream:
        return
    stream = torch.cuda.current_stream(device)
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is None:
        raise RuntimeError(
            "Exact-name CPU offload cannot identify the current CUDA stream"
        )
    stream_id = int(cuda_stream)
    if _CPU_OFFLOAD_OWNER_STREAM_ID is None:
        _CPU_OFFLOAD_OWNER_STREAM_ID = stream_id
    elif _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        # vLLM warms up on its graph-capture stream, while torch.cuda.graph
        # temporarily uses an internal capture stream. Both transitions are
        # serialized by vLLM. Concurrent execution is rejected separately by
        # validate_cpu_offload_graph_static_alias_runtime().
        _CPU_OFFLOAD_OWNER_STREAM_ID = stream_id
    elif stream_id != _CPU_OFFLOAD_OWNER_STREAM_ID:
        raise RuntimeError(
            "Exact-name CPU offload static-bank/copy execution must remain on "
            "one CUDA stream"
        )


def _create_graph_static_alias_binding(
    *,
    full_parameter_name: str,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
) -> _GraphStaticAliasBinding:
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "Graph static-alias parameter is missing from the precomputed "
            f"exact-name plan: {full_parameter_name!r}"
        )
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if is_contiguous is not None and not bool(is_contiguous()):
        raise RuntimeError(
            "Graph static-alias currently requires contiguous parameters: "
            f"{full_parameter_name!r}"
        )

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    signature = (
        shape,
        strides,
        tensor.dtype,
        int(device_index),
        int(requested_bytes),
        int(plan.consumer_count),
    )
    arena_key = (plan.layer_container, plan.local_name, plan.bank_id)
    arena = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.get(arena_key)
    if arena is None:
        from vllm.device_allocator.cumem import StaticAliasAllocation

        allocation = StaticAliasAllocation.create(
            device=int(device_index),
            size=int(requested_bytes),
            consumer_count=int(plan.consumer_count),
        )
        upload = allocation.upload_tensor(
            shape,
            tensor.dtype,
            strides=strides,
        )
        arena = _GraphStaticAliasArena(
            allocation=allocation,
            upload=upload,
            signature=signature,
            requested_bytes=int(allocation.requested_size),
            aligned_bytes=int(allocation.aligned_size),
            backing_bytes=int(allocation.backing_bytes),
        )
        if int(upload.data_ptr()) != int(allocation.upload_va):
            raise RuntimeError(
                "Graph static-alias upload tensor VA disagrees with the "
                f"allocator for {full_parameter_name!r}"
            )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS[arena_key] = arena
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES += arena.backing_bytes
    elif arena.signature != signature:
        raise RuntimeError(
            "Graph static-alias layers sharing one bank/local key have "
            f"different tensor signatures for {full_parameter_name!r}: "
            f"expected {arena.signature!r}, got {signature!r}"
        )

    allocation = arena.allocation
    consumer = allocation.consumer_tensor(
        int(plan.consumer_index),
        shape,
        tensor.dtype,
        strides=strides,
    )
    expected_consumer_va = int(allocation.consumer_vas[plan.consumer_index])
    if int(consumer.data_ptr()) != expected_consumer_va:
        raise RuntimeError(
            "Graph static-alias consumer tensor VA disagrees with the "
            f"allocator for {full_parameter_name!r}"
        )
    if int(consumer.data_ptr()) == int(arena.upload.data_ptr()):
        raise RuntimeError(
            "Graph static-alias upload and consumer mappings unexpectedly "
            f"share one VA for {full_parameter_name!r}"
        )
    binding = _GraphStaticAliasBinding(
        arena=arena,
        consumer=consumer,
        plan=plan,
    )
    arena.bindings.append(binding)
    return binding


_UVM_ORIGINAL_ALIAS_PAGE_BYTES = 4 * 1024
_UVM_ORIGINAL_ALIAS_64K_PTE_BYTES = 64 * 1024
_UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES = 2 * 1024 * 1024
_UVM_ORIGINAL_ALIAS_DEFAULT_REGION_BYTES = 512 * 1024
_UVM_ORIGINAL_ALIAS_MAX_REGION_BYTES = 4 * 1024 * 1024
_UVM_ORIGINAL_ALIAS_BACKING_LAYER_BASE = 0x70000000


def _uvm_original_alias_region_bytes() -> int:
    raw = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES",
        str(_UVM_ORIGINAL_ALIAS_DEFAULT_REGION_BYTES),
    )
    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES must be an "
            f"integer, got {raw!r}"
        ) from exc
    if (
        value <= 0
        or value > _UVM_ORIGINAL_ALIAS_MAX_REGION_BYTES
        or value % _UVM_ORIGINAL_ALIAS_PAGE_BYTES
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES must be a "
            "positive 4 KiB multiple no larger than 4 MiB"
        )
    return value


def _uvm_original_alias_requested_pte_mode() -> str:
    raw = (
        os.environ.get(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE",
            os.environ.get("UVM_STATIC_ALIAS_PTE_MODE", "auto"),
        )
        .strip()
        .lower()
    )
    if raw not in ("auto", "4k", "64k"):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE must be auto, 4k, "
            f"or 64k, got {raw!r}"
        )
    return raw


def _uvm_original_alias_backing_layer_id(arena_index: int) -> int:
    raw = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LAYER_ID_BASE",
        str(_UVM_ORIGINAL_ALIAS_BACKING_LAYER_BASE),
    )
    try:
        base = int(raw, 0)
    except ValueError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LAYER_ID_BASE must be an "
            f"integer, got {raw!r}"
        ) from exc
    result = base + int(arena_index)
    if result < 0 or result > (1 << 32) - 1:
        raise ValueError("UVM original-VA backing layer id exceeds uint32")
    return result


def _allocate_uvm_original_alias_upload(
    *,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
    upload_size: tuple[int, ...] | None = None,
    upload_stride: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one 2 MiB-aligned managed upload VA without replacing params."""
    from vllm.device_allocator.uvm import uvm_local_memory_pool

    if requested_bytes <= 0 or requested_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA static alias requires each selected parameter "
            "size to be a positive 4 KiB multiple"
        )
    allocation_bytes = int(requested_bytes) + _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    with torch.cuda.device(device), uvm_local_memory_pool(enable_prefetch=False):
        storage = torch.empty(
            allocation_bytes,
            dtype=torch.uint8,
            device=device,
        )
    storage_va = int(storage.data_ptr())
    alignment = _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    upload_va = (storage_va + alignment - 1) & ~(alignment - 1)
    upload_offset = upload_va - storage_va
    byte_view = storage[upload_offset : upload_offset + requested_bytes]
    if int(byte_view.data_ptr()) != upload_va or byte_view.numel() != requested_bytes:
        raise RuntimeError("Unable to construct a complete aligned managed upload VA")
    typed_storage = byte_view.view(tensor.dtype)
    size = upload_size or tuple(int(value) for value in tensor.size())
    stride = upload_stride or tuple(int(value) for value in tensor.stride())
    upload = torch.as_strided(
        typed_storage,
        size=size,
        stride=stride,
    )
    if int(upload.data_ptr()) != upload_va:
        raise RuntimeError("Managed upload tensor changed the aligned upload VA")
    return storage, upload


def _uvm_original_alias_parameter_geometry(
    *,
    full_parameter_name: str,
    tensor: torch.Tensor,
    parameter_bytes: int,
) -> tuple[
    tuple[int, ...] | None,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    """Return the packed-source geometry for one selected Parameter.

    A missing expert plan retains the original whole-Parameter behavior.  An
    expert plan packs the selected leading-dimension slices while retaining
    their discontiguous consumer VAs in the original managed allocation.
    """
    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "UVM original-VA parameter is missing from the precomputed plan: "
            f"{full_parameter_name!r}"
        )
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    consumer_va = int(tensor.data_ptr())
    expert_ids = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.get(
        plan.layer_index
    )
    if expert_ids is None:
        return (
            None,
            0,
            int(parameter_bytes),
            shape,
            strides,
            ((consumer_va, int(parameter_bytes)),),
        )

    if not shape or shape[0] <= 0:
        raise RuntimeError(
            "Expert-slice static alias requires an expert-leading tensor: "
            f"{full_parameter_name!r} has shape={shape!r}"
        )
    if expert_ids[-1] >= shape[0]:
        raise RuntimeError(
            f"Expert id {expert_ids[-1]} exceeds the leading dimension "
            f"of {full_parameter_name!r}: {shape[0]}"
        )
    if int(tensor.numel()) % shape[0]:
        raise RuntimeError(
            "Expert-slice tensor has a non-integral leading-dimension payload: "
            f"{full_parameter_name!r}"
        )
    expert_stride_bytes = int(strides[0]) * int(tensor.element_size())
    expert_payload_bytes = (
        int(tensor.numel()) // shape[0] * int(tensor.element_size())
    )
    if expert_stride_bytes != expert_payload_bytes:
        raise RuntimeError(
            "Expert-slice static alias requires densely packed expert slices "
            f"for {full_parameter_name!r}: stride_bytes="
            f"{expert_stride_bytes}, payload_bytes={expert_payload_bytes}"
        )
    if expert_stride_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            f"Expert stride for {full_parameter_name!r} is not 4 KiB aligned"
        )
    if int(parameter_bytes) != shape[0] * expert_stride_bytes:
        raise RuntimeError(
            "Expert-slice Parameter byte size does not match its dense expert "
            f"geometry for {full_parameter_name!r}"
        )

    selected_bytes = len(expert_ids) * expert_stride_bytes
    upload_shape = (len(expert_ids), *shape[1:])
    consumer_ranges = tuple(
        (consumer_va + expert_id * expert_stride_bytes, expert_stride_bytes)
        for expert_id in expert_ids
    )
    return (
        expert_ids,
        expert_stride_bytes,
        selected_bytes,
        upload_shape,
        strides,
        consumer_ranges,
    )


def _uvm_original_alias_binding_consumer_ranges(
    binding: _UvmOriginalAliasBinding,
) -> tuple[tuple[int, int], ...]:
    ranges = tuple(getattr(binding, "consumer_ranges", ()) or ())
    if ranges:
        return ranges
    return (
        (
            int(binding.consumer.data_ptr()),
            int(binding.requested_bytes),
        ),
    )


def _uvm_original_alias_binding_hot_ranges(
    binding: _UvmOriginalAliasBinding,
) -> tuple[tuple[int, int], ...]:
    """Return the complement of packed cold expert ranges in one Parameter."""
    expert_ids = getattr(binding, "expert_ids", None)
    parameter_bytes = int(getattr(binding, "parameter_bytes", 0) or 0)
    if not expert_ids or parameter_bytes <= 0:
        return ()

    parameter_base = int(binding.consumer.data_ptr())
    parameter_end = parameter_base + parameter_bytes
    cold_ranges = sorted(_uvm_original_alias_binding_consumer_ranges(binding))
    result: list[tuple[int, int]] = []
    cursor = parameter_base
    for cold_base, cold_bytes in cold_ranges:
        cold_end = int(cold_base) + int(cold_bytes)
        if cold_base < cursor or cold_end > parameter_end:
            raise RuntimeError(
                "Expert-slice consumer ranges overlap or escape their Parameter: "
                f"{binding.full_parameter_name!r}"
            )
        if cold_base > cursor:
            result.append((cursor, int(cold_base) - cursor))
        cursor = cold_end
    if cursor < parameter_end:
        result.append((cursor, parameter_end - cursor))
    return tuple(result)


def _uvm_original_alias_arena_regions(
    arena: _UvmOriginalAliasArena,
) -> Iterable[tuple[int, int, int]]:
    """Yield (region_id, packed_offset, bytes), never crossing an expert."""
    offset = 0
    region_id = 0
    while offset < arena.aligned_bytes:
        length = min(arena.region_bytes, arena.aligned_bytes - offset)
        if arena.packed_slice_bytes:
            slice_remaining = arena.packed_slice_bytes - (
                offset % arena.packed_slice_bytes
            )
            length = min(length, slice_remaining)
        yield region_id, offset, length
        offset += length
        region_id += 1


def _create_uvm_original_alias_binding(
    *,
    full_parameter_name: str,
    parameter: Any,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
) -> _UvmOriginalAliasBinding:
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "UVM original-VA parameter is missing from the precomputed plan: "
            f"{full_parameter_name!r}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            "UVM original-VA static alias currently requires contiguous "
            f"parameters: {full_parameter_name!r}"
        )
    if plan.consumer_count + 1 > 64:
        raise RuntimeError(
            "UVM original-VA bank exceeds the driver's 64-alias-per-slot "
            f"limit for {full_parameter_name!r}"
        )
    if requested_bytes <= 0 or requested_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA static alias cannot safely round a parameter "
            "outside its allocation; its exact byte size must be 4 KiB "
            f"aligned: {full_parameter_name!r} has {requested_bytes} bytes"
        )
    consumer_va = int(tensor.data_ptr())
    if consumer_va <= 0 or consumer_va % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA parameter allocation is not 4 KiB aligned: "
            f"{full_parameter_name!r} VA=0x{consumer_va:x}"
        )

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    (
        expert_ids,
        expert_stride_bytes,
        selected_bytes,
        upload_shape,
        upload_strides,
        consumer_ranges,
    ) = _uvm_original_alias_parameter_geometry(
        full_parameter_name=full_parameter_name,
        tensor=tensor,
        parameter_bytes=int(requested_bytes),
    )
    signature = (
        shape,
        strides,
        tensor.dtype,
        int(device_index),
        int(selected_bytes),
        upload_shape,
        int(plan.consumer_count),
    )
    arena_key = (plan.layer_container, plan.local_name, plan.bank_id)
    arena = _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.get(arena_key)
    if arena is None:
        upload_storage, upload = _allocate_uvm_original_alias_upload(
            tensor=tensor,
            device=device,
            requested_bytes=int(selected_bytes),
            upload_size=upload_shape,
            upload_stride=upload_strides,
        )
        arena = _UvmOriginalAliasArena(
            upload_storage=upload_storage,
            upload=upload,
            signature=signature,
            requested_bytes=int(selected_bytes),
            aligned_bytes=int(selected_bytes),
            backing_layer_id=_uvm_original_alias_backing_layer_id(
                len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
            ),
            pte_alignment=0,
            pte_flags=0,
            region_bytes=_uvm_original_alias_region_bytes(),
            packed_slice_bytes=int(expert_stride_bytes),
        )
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS[arena_key] = arena
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES += int(selected_bytes)
    elif arena.signature != signature:
        raise RuntimeError(
            "UVM original-VA layers sharing a bank/local key have different "
            f"tensor signatures for {full_parameter_name!r}: expected "
            f"{arena.signature!r}, got {signature!r}"
        )
    if plan.consumer_index in arena.consumers:
        raise RuntimeError(
            "UVM original-VA plan assigned the same consumer index twice for "
            f"{full_parameter_name!r}"
        )
    binding = _UvmOriginalAliasBinding(
        full_parameter_name=full_parameter_name,
        arena=arena,
        consumer=tensor,
        parameter=parameter,
        plan=plan,
        requested_bytes=int(selected_bytes),
        parameter_bytes=int(requested_bytes),
        expert_ids=expert_ids,
        expert_stride_bytes=int(expert_stride_bytes),
        consumer_ranges=consumer_ranges,
    )
    arena.consumers[plan.consumer_index] = binding
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[full_parameter_name] = binding
    return binding


def _uvm_original_alias_select_pte_mode() -> tuple[int, int, str]:
    from vllm.device_allocator.uvm_static_alias import (
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
    )

    target_ranges = [
        (int(arena.upload.data_ptr()), int(arena.aligned_bytes))
        for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values()
    ]
    target_ranges.extend(
        consumer_range
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
        for consumer_range in binding.consumer_ranges
    )
    if not target_ranges:
        raise RuntimeError("UVM original-VA static-alias topology is empty")
    page_aligned = all(
        base % _UVM_ORIGINAL_ALIAS_PAGE_BYTES == 0
        and length % _UVM_ORIGINAL_ALIAS_PAGE_BYTES == 0
        for base, length in target_ranges
    )
    if not page_aligned:
        raise RuntimeError(
            "UVM original-VA topology contains a target outside a complete "
            "4 KiB PTE envelope"
        )
    envelope = _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    supports_64k = all(
        base % envelope == 0 and length % envelope == 0
        for base, length in target_ranges
    )
    requested = _uvm_original_alias_requested_pte_mode()
    if requested == "64k" and not supports_64k:
        raise RuntimeError(
            "64K PTE mode was requested, but at least one original parameter "
            "does not cover a complete 2 MiB-aligned VA-block envelope"
        )
    use_64k = supports_64k and requested != "4k"
    return (
        _UVM_ORIGINAL_ALIAS_64K_PTE_BYTES
        if use_64k
        else _UVM_ORIGINAL_ALIAS_PAGE_BYTES,
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K if use_64k else 0,
        "64K" if use_64k else "4K",
    )


def _uvm_original_alias_map_entries(
    pte_flags: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    from vllm.device_allocator.uvm_static_alias import (
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
    )

    entries: list[tuple[int, int, int, int, int, int, int]] = []
    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()):
        if len(arena.consumers) != int(arena.signature[-1]):
            raise RuntimeError(
                "UVM original-VA arena is missing one or more planned consumer "
                f"VAs for backing layer {arena.backing_layer_id}"
            )
        for region_id, offset, length in _uvm_original_alias_arena_regions(arena):
            if length % arena.pte_alignment:
                raise RuntimeError(
                    "UVM original-VA final region does not match the selected "
                    "PTE granularity"
                )
            identity = (arena.backing_layer_id, region_id, region_id)
            entries.append(
                (
                    *identity,
                    UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW | pte_flags,
                    int(arena.upload.data_ptr()) + offset,
                    length,
                    0,
                )
            )
            for _, binding in sorted(arena.consumers.items()):
                consumer_va = int(binding.consumer.data_ptr()) + offset
                if binding.expert_ids is not None:
                    if binding.expert_stride_bytes <= 0:
                        raise RuntimeError(
                            "Expert-slice alias has no positive expert stride"
                        )
                    packed_expert = offset // binding.expert_stride_bytes
                    expert_offset = offset % binding.expert_stride_bytes
                    if (
                        packed_expert >= len(binding.expert_ids)
                        or expert_offset + length > binding.expert_stride_bytes
                    ):
                        raise RuntimeError(
                            "A packed alias region crosses an expert boundary for "
                            f"{binding.full_parameter_name!r}"
                        )
                    consumer_va = (
                        int(binding.consumer.data_ptr())
                        + binding.expert_ids[packed_expert]
                        * binding.expert_stride_bytes
                        + expert_offset
                    )
                entries.append(
                    (
                        *identity,
                        pte_flags,
                        consumer_va,
                        length,
                        0,
                    )
                )
    return entries


def _uvm_original_alias_preallocate_entries() -> list[
    tuple[int, int, int, int, int, int, int]
]:
    from vllm.device_allocator.uvm import UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT

    entries: list[tuple[int, int, int, int, int, int, int]] = []
    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()):
        for region_id, offset, length in _uvm_original_alias_arena_regions(arena):
            entries.append(
                (
                    int(arena.upload.data_ptr()) + offset,
                    length,
                    int(arena.backing_layer_id),
                    region_id,
                    region_id,
                    0,
                    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                )
            )
    return entries


def _append_uvm_original_alias_lifecycle(event: str, result: dict[str, Any]) -> None:
    _append_cpu_offload_manifest(
        {
            "event": event,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "backend": "uvm_original_va",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
            **result,
        }
    )


def _uvm_original_alias_destroy_cleared(result: Mapping[str, Any]) -> bool:
    """Trust cmd99's authoritative empty-session tuple, even on mismatch."""
    return bool(result.get("ok")) or bool(
        result.get("cache_cleared")
        and result.get("session_cleared")
        and int(result.get("remaining_alias_count") or 0) == 0
        and int(result.get("remaining_alias_bytes") or 0) == 0
        and int(result.get("remaining_backing_bytes") or 0) == 0
    )


def _uvm_original_alias_next_session_id(
    result: Mapping[str, Any], requested_session_id: int
) -> int:
    """Validate one MAP response before trusting its returned session id."""
    if not bool(result.get("ok")):
        raise RuntimeError(f"MAP did not succeed: {dict(result)!r}")
    raw_session_id = result.get("session_id")
    if (
        not isinstance(raw_session_id, int)
        or isinstance(raw_session_id, bool)
        or raw_session_id <= 0
    ):
        raise RuntimeError(
            "A successful UVM original-VA MAP returned an invalid session id"
        )
    returned_session_id = int(raw_session_id)
    if requested_session_id not in (0, returned_session_id):
        raise RuntimeError(
            "UVM original-VA MAP changed session id while appending: "
            f"requested={requested_session_id}, returned={returned_session_id}"
        )
    return returned_session_id


def _uvm_original_alias_close_slot_session(
    uvm: Any, *, device: int
) -> tuple[bool, dict[str, Any] | None]:
    """Close mode-8 backing state while retaining a structured retry result."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT

    if device not in uvm._uvm_slot_direct_sessions:
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
        return True, None
    try:
        raw_result = uvm.close_uvm_slot_direct_session(device=device)
        result = dict(raw_result)
    except Exception as exc:
        result = {
            "ok": False,
            "reason": "slot_session_close_exception",
            "closed": False,
            "error": repr(exc),
        }
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = result
    return bool(result.get("ok")), result


def finalize_cpu_offload_graph_static_alias() -> dict[str, Any]:
    """Build and seal original-parameter-VA aliases after weights are loaded.

    This must run before memory profiling or any CUDA Graph capture. It is a
    no-op for disabled and CUDA-VMM configurations so the model-runner hook can
    be unconditional.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return {
            "ok": True,
            "operation": "finalize",
            "reason": "disabled",
            "backend": None,
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND != "uvm_original_va":
        return {
            "ok": True,
            "operation": "finalize",
            "reason": "vmm_backend_noop",
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
        }

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed":
            return {
                "ok": True,
                "operation": "finalize",
                "reason": "already_sealed",
                "backend": "uvm_original_va",
                "state": "sealed",
                "sealed": True,
                "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
            }
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE not in ("collecting", "remote"):
            raise RuntimeError(
                "Cannot finalize UVM original-VA aliases from lifecycle state "
                f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
            )
        _require_all_copy_exact_names_seen()
        if set(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS) != set(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN
        ):
            missing = sorted(
                set(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN)
                - set(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
            )
            raise RuntimeError(
                "UVM original-VA bindings do not cover the exact-name plan; "
                f"missing={missing[:8]!r}"
            )

        from vllm.device_allocator import uvm, uvm_static_alias

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "finalizing"
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
        session_id = 0
        map_results: list[dict[str, Any]] = []
        map_side_effects = False
        map_session_unresolved = False
        try:
            pte_alignment, pte_flags, pte_mode = _uvm_original_alias_select_pte_mode()
            for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values():
                arena.pte_alignment = pte_alignment
                arena.pte_flags = pte_flags
                if arena.region_bytes % pte_alignment:
                    raise RuntimeError(
                        "Configured UVM static-alias region size is not a "
                        f"multiple of the selected {pte_mode} PTE granularity"
                    )
                managed, reason, code = uvm.probe_driver_capable_range(
                    int(arena.upload.data_ptr()), int(arena.aligned_bytes)
                )
                if not managed:
                    raise RuntimeError(
                        "UVM original-VA upload arena is not a driver-capable "
                        f"managed allocation: reason={reason}, code={code}"
                    )

            source_created_count = 0
            source_reused_count = 0
            for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values():
                current = binding.parameter.data
                if int(current.data_ptr()) != int(binding.consumer.data_ptr()):
                    raise RuntimeError(
                        "A selected expert parameter changed allocation VA "
                        "after construction: "
                        f"{binding.full_parameter_name!r}"
                    )
                for range_base, range_bytes in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                ):
                    managed, reason, code = uvm.probe_driver_capable_range(
                        int(range_base), int(range_bytes)
                    )
                    if not managed:
                        raise RuntimeError(
                            "Selected expert range is not a driver-capable "
                            "managed allocation despite "
                            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1: "
                            f"{binding.full_parameter_name!r}, "
                            f"range=0x{range_base:x}+{range_bytes}, "
                            f"reason={reason}, code={code}"
                        )
                source = binding.source
                if source is None:
                    source = torch.empty_strided(
                        size=binding.arena.upload.size(),
                        stride=binding.arena.upload.stride(),
                        dtype=current.dtype,
                        layout=current.layout,
                        device="cpu",
                        pin_memory=True,
                    )
                    if not source.is_pinned():
                        raise RuntimeError(
                            "UVM original-VA source allocation is not pinned for "
                            f"{binding.full_parameter_name!r}"
                        )
                    if binding.expert_ids is None:
                        source.copy_(current, non_blocking=True)
                    else:
                        for packed_index, expert_id in enumerate(
                            binding.expert_ids
                        ):
                            source[packed_index].copy_(
                                current[expert_id], non_blocking=True
                            )
                    binding.source = source
                    source_created_count += 1
                else:
                    if (
                        not source.is_pinned()
                        or tuple(source.size())
                        != tuple(binding.arena.upload.size())
                        or tuple(source.stride())
                        != tuple(binding.arena.upload.stride())
                        or source.dtype != current.dtype
                    ):
                        raise RuntimeError(
                            "A retained UVM original-VA pinned source changed "
                            f"geometry for {binding.full_parameter_name!r}"
                        )
                    source_reused_count += 1
            torch.cuda.synchronize(device)

            current_original_vas = tuple(
                int(binding.consumer.data_ptr())
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
            )
            if (
                _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
                and current_original_vas != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            ):
                raise RuntimeError(
                    "An original expert Parameter VA changed between topology "
                    "transitions"
                )
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = current_original_vas

            cold_residency_ranges = [
                consumer_range
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                for consumer_range in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                )
            ]
            residency_ranges = list(cold_residency_ranges)
            residency_ranges.extend(
                (int(arena.upload.data_ptr()), int(arena.aligned_bytes))
                for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values()
            )
            prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
                residency_ranges,
                sync=True,
                device=device,
            )
            if int(prepared[0]) != len(residency_ranges):
                raise RuntimeError(
                    "Failed to make every original/upload managed range CPU "
                    f"resident before MAP: result={prepared!r}"
                )

            cold_residency_bytes = sum(
                length for _, length in cold_residency_ranges
            )
            hot_residency_ranges = [
                hot_range
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_residency_bytes = sum(
                length for _, length in hot_residency_ranges
            )
            expert_slice_topology = any(
                binding.expert_ids is not None
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
            )
            cold_residency: dict[str, Any] | None = None
            hot_residency: dict[str, Any] | None = None
            if expert_slice_topology:
                cold_residency = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        cold_residency_ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = (
                    cold_residency
                )
                if not (
                    cold_residency.get("ok") is True
                    and int(cold_residency.get("queried_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("cpu_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("gpu_bytes", -1)) == 0
                    and int(cold_residency.get("any_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("unknown_bytes", -1)) == 0
                ):
                    raise RuntimeError(
                        "Cold expert residency was not exclusively CPU-backed "
                        f"before static-alias MAP: {cold_residency!r}"
                    )

                hot_residency = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        hot_residency_ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = hot_residency
                if hot_residency_ranges and not (
                    hot_residency.get("ok") is True
                    and int(hot_residency.get("queried_bytes", -1))
                    == hot_residency_bytes
                    and int(hot_residency.get("gpu_bytes", -1))
                    == hot_residency_bytes
                    and int(hot_residency.get("cpu_bytes", -1)) == 0
                    and int(hot_residency.get("any_bytes", -1))
                    == hot_residency_bytes
                    and int(hot_residency.get("unknown_bytes", -1)) == 0
                ):
                    raise RuntimeError(
                        "Hot expert residency changed while preparing cold slices: "
                        f"{hot_residency!r}"
                    )

            preallocate_entries = _uvm_original_alias_preallocate_entries()
            preallocate = uvm.driver_slot_preallocate_ranges(
                preallocate_entries, device=device
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT = preallocate
            if not bool(preallocate.get("ok")):
                raise RuntimeError(
                    f"UVM original-VA backing preallocation failed: {preallocate!r}"
                )

            map_entries = _uvm_original_alias_map_entries(pte_flags)
            maximum = uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
            for offset in range(0, len(map_entries), maximum):
                mapped = uvm_static_alias.driver_static_alias_map(
                    map_entries[offset : offset + maximum],
                    session_id=session_id,
                    device=device,
                )
                map_results.append(mapped)
                raw_returned_session_id = mapped.get("session_id")
                returned_session_id = (
                    int(raw_returned_session_id)
                    if isinstance(raw_returned_session_id, int)
                    and not isinstance(raw_returned_session_id, bool)
                    and raw_returned_session_id > 0
                    else 0
                )
                map_side_effects = (
                    map_side_effects
                    or any(
                        bool(mapped.get(field))
                        for field in (
                            "ok",
                            "made_progress",
                            "poisoned",
                            "side_effect_indeterminate",
                        )
                    )
                    or returned_session_id > 0
                )
                if (
                    session_id == 0
                    and not bool(mapped.get("ok"))
                    and (
                        bool(mapped.get("made_progress"))
                        or bool(mapped.get("poisoned"))
                        or bool(mapped.get("side_effect_indeterminate"))
                        or returned_session_id > 0
                    )
                ):
                    # A failed first MAP cannot make its returned id trusted.
                    # Keep the backing fd alive and require process teardown.
                    map_session_unresolved = True
                if session_id > 0 and returned_session_id not in (0, session_id):
                    # Do not issue cmd99 against an unexpected id. The known
                    # session remains cleanable, but the contradictory response
                    # means another possible side effect cannot be disproved.
                    map_session_unresolved = True
                try:
                    next_session_id = _uvm_original_alias_next_session_id(
                        mapped, session_id
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "UVM original-VA MAP failed at batch "
                        f"{len(map_results) - 1}: {mapped!r}"
                    ) from exc
                session_id = next_session_id
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = session_id
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = map_results

            expected_alias_count = len(map_entries)
            expected_alias_bytes = sum(entry[5] for entry in map_entries)
            seal = uvm_static_alias.driver_static_alias_seal(
                session_id,
                expected_alias_count=expected_alias_count,
                expected_alias_bytes=expected_alias_bytes,
                device=device,
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT = seal
            if not bool(seal.get("ok")):
                raise RuntimeError(f"UVM original-VA SEAL failed: {seal!r}")
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
            result = {
                "ok": True,
                "operation": "finalize",
                "reason": "ok",
                "backend": "uvm_original_va",
                "state": "sealed",
                "sealed": True,
                "session_id": session_id,
                "pte_mode": pte_mode,
                "region_bytes": int(
                    next(
                        iter(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values())
                    ).region_bytes
                ),
                "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
                "preallocate_count": len(preallocate_entries),
                "alias_count": expected_alias_count,
                "alias_bytes": expected_alias_bytes,
                "map_batch_count": len(map_results),
                "source_created_count": int(source_created_count),
                "source_reused_count": int(source_reused_count),
                "cold_residency": cold_residency,
                "hot_residency": hot_residency,
                "cold_residency_range_count": len(cold_residency_ranges),
                "cold_residency_bytes": int(cold_residency_bytes),
                "hot_residency_range_count": len(hot_residency_ranges),
                "hot_residency_bytes": int(hot_residency_bytes),
                "upload_vas": tuple(
                    int(arena.upload.data_ptr())
                    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
                ),
                "original_parameter_vas": tuple(
                    int(binding.consumer.data_ptr())
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                ),
                "pinned_source_vas": tuple(
                    int(binding.source.data_ptr())
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                    if binding.source is not None
                ),
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_finalized", result
            )
            return result
        except Exception:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = map_results
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            active_session_id = int(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID or session_id
            )
            aliases_cleared = not map_side_effects
            cleanup_unresolved = bool(map_session_unresolved)
            if active_session_id:
                try:
                    cleanup = uvm_static_alias.driver_static_alias_destroy(
                        active_session_id,
                        device=device,
                        release_backing=True,
                        synchronize=True,
                    )
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = cleanup
                    aliases_cleared = _uvm_original_alias_destroy_cleared(cleanup)
                    if aliases_cleared:
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
                except Exception as cleanup_exc:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = {
                        "ok": False,
                        "operation": "destroy",
                        "reason": "destroy_exception",
                        "session_id": active_session_id,
                        "error": repr(cleanup_exc),
                    }
                    logger.exception(
                        "Failed to clean up a partially finalized UVM "
                        "original-VA static-alias session"
                    )
            elif map_side_effects:
                # There was no response from which a session id could safely
                # be adopted. Closing mode-8 backing could leave a live alias
                # pointing at released memory, so retain every resource until
                # CUDA/UVM process teardown.
                cleanup_unresolved = True

            _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = cleanup_unresolved
            if aliases_cleared and not cleanup_unresolved:
                slot_closed, _ = _uvm_original_alias_close_slot_session(
                    uvm, device=device
                )
                if slot_closed:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
                else:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            else:
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            raise


def _normalize_uvm_dual_graph_entries(
    entries: Iterable[Mapping[str, Any]],
) -> tuple[tuple[int, bool, bool], ...]:
    normalized: list[tuple[int, bool, bool]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("CUDA Graph entry certificates must be mappings")
        num_tokens = entry.get("num_tokens")
        uniform_decode = entry.get("uniform_decode")
        has_lora = entry.get("has_lora", False)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise TypeError("CUDA Graph entry num_tokens must be an int")
        if num_tokens <= 0:
            raise ValueError("CUDA Graph entry num_tokens must be positive")
        if not isinstance(uniform_decode, bool) or not isinstance(has_lora, bool):
            raise TypeError(
                "CUDA Graph entry uniform_decode/has_lora fields must be bool"
            )
        normalized.append((num_tokens, uniform_decode, has_lora))
    if len(normalized) != len(set(normalized)):
        raise ValueError("CUDA Graph entry certificate contains duplicates")
    return tuple(sorted(normalized))


def certify_cpu_offload_uvm_original_va_dual_graph_partition(
    *,
    remote_entries: Iterable[Mapping[str, Any]],
    alias_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind independently captured Graph entries to their PTE topologies."""
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE

    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != (
        "prefill_graph_remote_decode"
    ):
        raise RuntimeError(
            "A dual Graph partition certificate requires "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode"
        )
    remote = _normalize_uvm_dual_graph_entries(remote_entries)
    alias = _normalize_uvm_dual_graph_entries(alias_entries)
    if not remote or not alias:
        raise RuntimeError("Both remote and alias Graph entry sets are required")
    if any(not uniform for _, uniform, _ in remote):
        raise RuntimeError("Every remote Graph entry must be uniform-decode")
    if any(uniform for _, uniform, _ in alias):
        raise RuntimeError("Every alias Graph entry must be non-uniform")
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED or not isinstance(
        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE, dict
    ):
        raise RuntimeError("Remote Graph capture certificate is absent")
    if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED or not isinstance(
        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE, dict
    ):
        raise RuntimeError("Alias Graph capture certificate is absent")
    current_vas = _uvm_original_parameter_va_snapshot()
    if tuple(
        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE.get("parameter_vas", ())
    ) != current_vas or tuple(
        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE.get("parameter_vas", ())
    ) != current_vas:
        raise RuntimeError("Dual Graph capture Parameter VA fingerprints differ")
    certificate = {
        "schema": "uvm_original_va_dual_graph_partition_v1",
        "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
        "parameter_vas": current_vas,
        "remote_entries": remote,
        "alias_entries": alias,
        "remote_entry_count": len(remote),
        "alias_entry_count": len(alias),
    }
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = certificate
    _append_cpu_offload_manifest(
        {
            "event": "cpu_offload_uvm_original_va_dual_graph_partition",
            "timestamp": time.time(),
            "pid": os.getpid(),
            **certificate,
        }
    )
    return dict(certificate)


def certify_cpu_offload_uvm_original_va_alias_graphs_destroyed(
    *,
    destroyed_alias_entries: Iterable[Mapping[str, Any]],
    remaining_remote_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Certify selective alias-Graph reset before cmd99 frees its backing."""
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT

    certificate = _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
    if not isinstance(certificate, dict):
        raise RuntimeError("Dual Graph partition certificate is absent")
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        raise RuntimeError(
            "Alias Graphs may only be retired while alias PTEs are sealed"
        )
    destroyed = _normalize_uvm_dual_graph_entries(destroyed_alias_entries)
    remaining = _normalize_uvm_dual_graph_entries(remaining_remote_entries)
    expected_alias = tuple(certificate.get("alias_entries", ()))
    expected_remote = tuple(certificate.get("remote_entries", ()))
    if destroyed != expected_alias:
        raise RuntimeError(
            "Destroyed alias Graph entries differ from the capture certificate"
        )
    if remaining != expected_remote:
        raise RuntimeError(
            "Remaining remote Graph entries differ from the capture certificate"
        )
    if not remaining:
        raise RuntimeError("No remote decode Graph remains after alias teardown")
    current_vas = _uvm_original_parameter_va_snapshot()
    if tuple(certificate.get("parameter_vas", ())) != current_vas:
        raise RuntimeError("Parameter VA changed before alias Graph teardown")
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = True
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = len(remaining)
    result = {
        "schema": "uvm_original_va_alias_graph_teardown_v1",
        "event": "cpu_offload_uvm_original_va_alias_graphs_destroyed",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": True,
        "destroyed_alias_entries": destroyed,
        "destroyed_alias_graph_count": len(destroyed),
        "remaining_remote_entries": remaining,
        "remaining_remote_graph_count": len(remaining),
        "parameter_vas": current_vas,
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
    _append_cpu_offload_manifest(result)
    return dict(result)


def _uvm_original_alias_can_preserve_remote_graphs(
    *, synchronize: bool
) -> tuple[bool, str]:
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE not in (
        "prefill_remote_decode",
        "prefill_graph_remote_decode",
    ):
        return False, "phase_mode_is_not_hybrid"
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        return False, "no_successful_remote_graph_capture"
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        == "prefill_graph_remote_decode"
    ):
        if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED:
            return False, "alias_graph_capture_certificate_missing"
        if not _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
            return False, "alias_prefill_graphs_are_still_live"
        if _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT <= 0:
            return False, "no_certified_remote_graph_remains"
    certificate = _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    if not isinstance(certificate, dict):
        return False, "remote_graph_certificate_missing"
    if certificate.get("topology") != "remote":
        return False, "remote_graph_certificate_topology_mismatch"
    if int(certificate.get("capture_forward_count", 0)) <= 0:
        return False, "remote_graph_certificate_has_no_forwards"
    if int(certificate.get("actual_h2d_bytes", -1)) != 0:
        return False, "remote_graph_certificate_contains_h2d"
    try:
        current_vas = _uvm_original_parameter_va_snapshot()
    except Exception:
        return False, "current_parameter_va_snapshot_failed"
    if tuple(certificate.get("parameter_vas", ())) != current_vas:
        return False, "remote_graph_parameter_va_fingerprint_changed"
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return False, "cuda_graph_capture_is_active"
    if not synchronize:
        return False, "preserved_graph_transition_requires_synchronize"
    return True, "ok"


def _destroy_vmm_graph_static_alias(
    *,
    graphs_destroyed: bool,
    synchronize: bool,
    remote_graphs_preserved: bool,
) -> dict[str, Any]:
    """Release every CUDA VMM alias arena after its Graph owners are gone.

    DLPack tensors keep native live references to each allocation. Break the
    upload/consumer references owned by this integration before asking the
    allocator to synchronize and unmap. A failed arena remains registered so
    teardown can be retried; successfully closed arenas are never presented to
    the native allocator twice.
    """
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    if (
        not isinstance(graphs_destroyed, bool)
        or not isinstance(synchronize, bool)
        or not isinstance(remote_graphs_preserved, bool)
        or remote_graphs_preserved
    ):
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "validation_failed",
            "error": (
                "VMM teardown requires unambiguous bool arguments and cannot "
                "preserve CUDA Graphs"
            ),
            "backend": "vmm",
        }
    if not graphs_destroyed:
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "cuda_graphs_still_live",
            "error": "Destroy every CUDA Graph which captured VMM aliases first",
            "backend": "vmm",
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "cuda_graph_capture_active",
            "error": "VMM aliases cannot be destroyed during Graph capture",
            "backend": "vmm",
        }

    with _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LOCK:
        arenas = list(sorted(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.items()))
        configured_backing_bytes = int(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
        )
        if not arenas:
            if configured_backing_bytes != 0:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "vmm_state_inconsistent",
                    "error": "VMM backing bytes remain without a live arena",
                    "backend": "vmm",
                    "remaining_backing_bytes": configured_backing_bytes,
                }
            return {
                "ok": True,
                "operation": "destroy",
                "reason": "no_active_allocations",
                "backend": "vmm",
                "arena_count_before": 0,
                "destroyed_arena_count": 0,
                "remaining_arena_count": 0,
                "released_backing_bytes": 0,
                "remaining_backing_bytes": 0,
            }

        computed_backing_bytes = sum(
            int(arena.backing_bytes) for _, arena in arenas
        )
        if computed_backing_bytes != configured_backing_bytes:
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "vmm_state_inconsistent",
                "error": (
                    "Registered VMM arena bytes disagree with the global "
                    "backing-byte counter"
                ),
                "backend": "vmm",
                "arena_count_before": len(arenas),
                "registered_backing_bytes": computed_backing_bytes,
                "configured_backing_bytes": configured_backing_bytes,
            }

        # Drop every tensor reference owned by the integration before the
        # native allocator checks its DLPack live-reference count. Bindings
        # remain reachable from wrapped module closures, so clear their tensor
        # field explicitly instead of merely removing the global arena map.
        for _, arena in arenas:
            object.__setattr__(arena, "upload", None)
            for binding in tuple(arena.bindings):
                object.__setattr__(binding, "consumer", None)
        gc.collect()

        close_results: list[dict[str, Any]] = []
        closed_keys: list[tuple[str, str, int]] = []
        released_backing_bytes = 0
        for arena_key, arena in arenas:
            allocation = arena.allocation
            token = getattr(allocation, "token", None)
            try:
                if synchronize:
                    allocation.close(synchronize=True)
                    barrier = "close_synchronize"
                else:
                    # The caller may already have synchronized CUDA, but the
                    # native allocator intentionally requires its own teardown
                    # barrier bit. Set it after releasing our tensor views.
                    allocation.synchronize()
                    allocation.close(synchronize=False)
                    barrier = "explicit_synchronize"
                if getattr(allocation, "closed", None) is not True:
                    raise RuntimeError(
                        "VMM allocation close returned without a closed state"
                    )
            except Exception as exc:
                close_results.append(
                    {
                        "arena_key": arena_key,
                        "token": token,
                        "ok": False,
                        "error": repr(exc),
                    }
                )
                continue

            closed_keys.append(arena_key)
            released_backing_bytes += int(arena.backing_bytes)
            close_results.append(
                {
                    "arena_key": arena_key,
                    "token": token,
                    "ok": True,
                    "barrier": barrier,
                    "backing_bytes": int(arena.backing_bytes),
                }
            )

        for arena_key in closed_keys:
            arena = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.pop(arena_key)
            arena.bindings.clear()
        remaining_backing_bytes = sum(
            int(arena.backing_bytes)
            for arena in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.values()
        )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = remaining_backing_bytes
        ok = not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
        return {
            "ok": ok,
            "operation": "destroy",
            "reason": "ok" if ok else "vmm_close_failed",
            "backend": "vmm",
            "arena_count_before": len(arenas),
            "destroyed_arena_count": len(closed_keys),
            "remaining_arena_count": len(
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
            ),
            "released_backing_bytes": released_backing_bytes,
            "remaining_backing_bytes": remaining_backing_bytes,
            "close_results": tuple(close_results),
        }


def destroy_cpu_offload_graph_static_alias(
    *,
    graphs_destroyed: bool,
    synchronize: bool = True,
    remote_graphs_preserved: bool = False,
) -> dict[str, Any]:
    """Destroy static aliases after Graph ownership is proven safe.

    Original-VA aliases may retain only an audited remote Graph. CUDA VMM
    aliases always require all captured Graphs to be destroyed first.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return {
            "ok": True,
            "operation": "destroy",
            "reason": "disabled",
            "backend": None,
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "vmm":
        return _destroy_vmm_graph_static_alias(
            graphs_destroyed=graphs_destroyed,
            synchronize=synchronize,
            remote_graphs_preserved=remote_graphs_preserved,
        )
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND != "uvm_original_va":
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "unsupported_backend",
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
        }
    if (
        not isinstance(graphs_destroyed, bool)
        or not isinstance(synchronize, bool)
        or not isinstance(remote_graphs_preserved, bool)
        or (graphs_destroyed and remote_graphs_preserved)
    ):
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "validation_failed",
            "error": (
                "graphs_destroyed, synchronize, and remote_graphs_preserved "
                "must be unambiguous bool values"
            ),
            "backend": "uvm_original_va",
        }

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID)
        cleanup_pending = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"
        if session_id == 0 and not cleanup_pending:
            return {
                "ok": True,
                "operation": "destroy",
                "reason": "no_active_session",
                "backend": "uvm_original_va",
                "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "session_id": 0,
            }
        preserved_graph_contract = False
        if not graphs_destroyed:
            preserved_graph_contract, contract_reason = (
                _uvm_original_alias_can_preserve_remote_graphs(
                    synchronize=synchronize
                )
            )
            if not remote_graphs_preserved or not preserved_graph_contract:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "cuda_graphs_still_live",
                    "error": (
                        "Destroy alias-topology Graphs before cmd99; only a "
                        "successfully captured remote-only Graph may remain live"
                    ),
                    "preserved_graph_contract_reason": contract_reason,
                    "backend": "uvm_original_va",
                    "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                    "session_id": session_id,
                }

        from vllm.device_allocator import uvm, uvm_static_alias

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        if session_id == 0:
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "cleanup_unresolved_requires_process_teardown",
                    "error": (
                        "A MAP may have changed kernel state without returning "
                        "a trustworthy session id; retaining slot backing and "
                        "the UVM fd until process teardown"
                    ),
                    "backend": "uvm_original_va",
                    "state": "cleanup_pending",
                    "session_id": 0,
                    "destroy": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
                    "slot_close": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT,
                }
            slot_closed, slot_close = _uvm_original_alias_close_slot_session(
                uvm, device=device
            )
            if not slot_closed:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "slot_session_close_failed",
                    "backend": "uvm_original_va",
                    "state": "cleanup_pending",
                    "session_id": 0,
                    "destroy": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
                    "slot_close": slot_close,
                }
            destroyed = _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
            destroy_contract_ok = destroyed is None or bool(destroyed.get("ok"))
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
            result = {
                "ok": destroy_contract_ok,
                "operation": "destroy",
                "reason": (
                    "slot_session_close_retried"
                    if destroy_contract_ok
                    else "cmd99_contract_mismatch_session_cleared"
                ),
                "backend": "uvm_original_va",
                "state": "remote",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": slot_close,
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_destroyed", result
            )
            return result

        try:
            destroyed = dict(
                uvm_static_alias.driver_static_alias_destroy(
                    session_id,
                    device=device,
                    release_backing=True,
                    synchronize=synchronize,
                )
            )
        except Exception as exc:
            destroyed = {
                "ok": False,
                "operation": "destroy",
                "reason": "destroy_exception",
                "session_id": session_id,
                "error": repr(exc),
            }
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = destroyed
        if not _uvm_original_alias_destroy_cleared(destroyed):
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "cmd99_failed",
                "backend": "uvm_original_va",
                "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "session_id": session_id,
                "destroy": destroyed,
            }
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "cleanup_unresolved_requires_process_teardown",
                "backend": "uvm_original_va",
                "state": "cleanup_pending",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT,
            }
        slot_closed, slot_close = _uvm_original_alias_close_slot_session(
            uvm, device=device
        )
        if not slot_closed:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            result = {
                "ok": False,
                "operation": "destroy",
                "reason": "slot_session_close_failed",
                "backend": "uvm_original_va",
                "state": "cleanup_pending",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": slot_close,
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_destroyed", result
            )
            return result
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        result = {
            "ok": bool(destroyed.get("ok")),
            "operation": "destroy",
            "reason": (
                "ok"
                if bool(destroyed.get("ok"))
                else "cmd99_contract_mismatch_session_cleared"
            ),
            "backend": "uvm_original_va",
            "state": "remote",
            "session_id": 0,
            "destroy": destroyed,
            "slot_close": slot_close,
            "remote_graphs_preserved": bool(preserved_graph_contract),
        }
        _append_uvm_original_alias_lifecycle(
            "cpu_offload_uvm_static_alias_destroyed", result
        )
        return result


def _uvm_original_parameter_va_snapshot() -> tuple[int, ...]:
    vas: list[int] = []
    for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()):
        current = binding.parameter.data
        current_va = int(current.data_ptr())
        expected_va = int(binding.consumer.data_ptr())
        if current_va != expected_va:
            raise RuntimeError(
                "A selected expert Parameter changed allocation VA: "
                f"{binding.full_parameter_name!r}, current=0x{current_va:x}, "
                f"expected=0x{expected_va:x}"
            )
        vas.append(current_va)
    return tuple(vas)


def poison_cpu_offload_uvm_original_va_phase(reason: str) -> None:
    """Prevent any later selected-expert forward after a phase violation."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    record = {
        "event": "cpu_offload_uvm_original_va_phase_poisoned",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": False,
        "reason": str(reason),
        "post_state": "poisoned",
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(record)
    _append_cpu_offload_manifest(record)


def transition_cpu_offload_uvm_original_va_to_remote(
    *,
    reason: str,
    preserve_remote_graphs: bool,
) -> dict[str, Any]:
    """Replace sealed alias PTEs with audited CPU-backed remote READ PTEs."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {
            "ok": True,
            "reason": "disabled",
            "pre_state": "disabled",
            "post_state": "disabled",
        }
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "off":
        raise RuntimeError(
            "An original-VA phase transition was requested while "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Original-VA remote transition requires a reason")
    if not isinstance(preserve_remote_graphs, bool):
        raise TypeError("preserve_remote_graphs must be bool")

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state == "remote" and _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
            return {
                "ok": True,
                "reason": "already_remote",
                "pre_state": pre_state,
                "post_state": pre_state,
                "remote_map": _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT,
            }
        if pre_state not in ("sealed", "remote"):
            raise RuntimeError(
                "Cannot establish original-VA remote topology from state "
                f"{pre_state!r}"
            )

        started_ns = time.perf_counter_ns()
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        if (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            and parameter_vas_before != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
        ):
            raise RuntimeError("Original Parameter VAs changed before remote cutover")
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = parameter_vas_before
        destroy_result: dict[str, Any] | None = None
        remote_map: dict[str, Any] | None = None
        physical: dict[str, Any] | None = None
        synchronized_ns = started_ns
        destroyed_ns = started_ns
        mapped_ns = started_ns
        try:
            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            torch.cuda.synchronize(device)
            synchronized_ns = time.perf_counter_ns()
            if pre_state == "sealed":
                destroy_result = destroy_cpu_offload_graph_static_alias(
                    graphs_destroyed=not preserve_remote_graphs,
                    synchronize=True,
                    remote_graphs_preserved=preserve_remote_graphs,
                )
                if not bool(destroy_result.get("ok")):
                    raise RuntimeError(
                        "Failed to detach original-VA aliases before remote map: "
                        f"{destroy_result!r}"
                    )
            destroyed_ns = time.perf_counter_ns()

            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote_mapping"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            from vllm.device_allocator import uvm

            ranges = [
                consumer_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for consumer_range in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                )
            ]
            requested_bytes = sum(length for _, length in ranges)
            remote_map = dict(
                uvm.driver_prepare_ranges_remote_read_audited(
                    ranges,
                    device=device,
                )
            )
            mapped_ns = time.perf_counter_ns()
            remote_contract_ok = (
                remote_map.get("ok") is True
                and int(remote_map.get("requested_count", -1)) == len(ranges)
                and int(remote_map.get("success_count", -1)) == len(ranges)
                and int(remote_map.get("failure_count", -1)) == 0
                and int(remote_map.get("requested_bytes", -1)) == requested_bytes
                and int(remote_map.get("cpu_resident_bytes", -1))
                >= requested_bytes
                and int(remote_map.get("remote_mapped_bytes", -1))
                >= requested_bytes
                and int(remote_map.get("covered_bytes", -1)) >= requested_bytes
                and int(remote_map.get("unknown_bytes", -1)) == 0
            )
            if not remote_contract_ok:
                raise RuntimeError(
                    "Audited original-VA remote READ mapping was incomplete: "
                    f"{remote_map!r}"
                )

            torch.cuda.synchronize(device)
            physical = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            physical_contract_ok = (
                physical.get("ok") is True
                and int(physical.get("queried_bytes", -1)) == requested_bytes
                and int(physical.get("cpu_bytes", -1)) == requested_bytes
                and int(physical.get("gpu_bytes", -1)) == 0
                and int(physical.get("any_bytes", -1)) == requested_bytes
                and int(physical.get("unknown_bytes", -1)) == 0
            )
            if not physical_contract_ok:
                raise RuntimeError(
                    "Original-VA remote physical residency audit failed: "
                    f"{physical!r}"
                )

            hot_ranges = [
                hot_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_requested_bytes = sum(length for _, length in hot_ranges)
            hot_physical = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            if hot_ranges and not (
                hot_physical.get("ok") is True
                and int(hot_physical.get("queried_bytes", -1))
                == hot_requested_bytes
                and int(hot_physical.get("gpu_bytes", -1))
                == hot_requested_bytes
                and int(hot_physical.get("cpu_bytes", -1)) == 0
                and int(hot_physical.get("any_bytes", -1))
                == hot_requested_bytes
                and int(hot_physical.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "Hot experts lost exclusive GPU residency during remote "
                    f"cutover: {hot_physical!r}"
                )

            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if parameter_vas_after != parameter_vas_before:
                raise RuntimeError("Original Parameter VAs changed during remote cutover")
            finished_ns = time.perf_counter_ns()
            _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = remote_map
            _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY.append(dict(remote_map))
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = True
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
            result = {
                "event": "cpu_offload_uvm_original_va_remote_transition",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "reason": reason,
                "pre_state": pre_state,
                "post_state": "remote",
                "sync_ns": synchronized_ns - started_ns,
                "destroy_ns": destroyed_ns - synchronized_ns,
                "remote_map_ns": mapped_ns - destroyed_ns,
                "physical_audit_ns": finished_ns - mapped_ns,
                "wall_ns": finished_ns - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": parameter_vas_after,
                "preserve_remote_graphs": bool(preserve_remote_graphs),
                "destroy": destroy_result,
                "remote_map": remote_map,
                "physical_residency": physical,
                "hot_physical_residency": hot_physical,
                "remote_range_count": len(ranges),
                "hot_range_count": len(hot_ranges),
                "hot_requested_bytes": int(hot_requested_bytes),
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            return result
        except Exception as exc:
            finished_ns = time.perf_counter_ns()
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            try:
                failed_parameter_vas_after = (
                    _uvm_original_parameter_va_snapshot()
                )
            except Exception:
                failed_parameter_vas_after = ()
            failed = {
                "event": "cpu_offload_uvm_original_va_remote_transition",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "reason": reason,
                "error": repr(exc),
                "pre_state": pre_state,
                "post_state": "poisoned",
                "sync_ns": synchronized_ns - started_ns,
                "destroy_ns": destroyed_ns - synchronized_ns,
                "remote_map_ns": mapped_ns - destroyed_ns,
                "wall_ns": finished_ns - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": failed_parameter_vas_after,
                "preserve_remote_graphs": bool(preserve_remote_graphs),
                "destroy": destroy_result,
                "remote_map": remote_map,
                "physical_residency": physical,
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Original-VA remote topology transition failed; the process is "
                "poisoned and must be torn down"
            ) from exc


def initialize_cpu_offload_uvm_original_va_phase_topology() -> dict[str, Any]:
    """Put a remote-only process into its serving topology after model load."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode != "remote_only":
        return {"ok": True, "reason": "not_remote_only", "phase_mode": mode}
    return transition_cpu_offload_uvm_original_va_to_remote(
        reason="remote_only_model_load",
        preserve_remote_graphs=False,
    )


def prepare_cpu_offload_uvm_original_va_remote_graph_capture() -> dict[str, Any]:
    """Establish the no-H2D remote topology immediately before capture."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode == "off":
        return {"ok": True, "reason": "phase_mode_off", "phase_mode": mode}
    if _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        raise RuntimeError(
            "A certified original-VA remote Graph already exists; recapture and "
            "lazy capture are unsupported for the phase-switch experiment"
        )
    return transition_cpu_offload_uvm_original_va_to_remote(
        reason="remote_graph_capture_prepare",
        preserve_remote_graphs=False,
    )


def activate_cpu_offload_uvm_original_va_prefill_topology() -> dict[str, Any]:
    """Restore sealed alias banks after all initialization Graph replays end."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode not in (
        "prefill_remote_decode",
        "prefill_graph_remote_decode",
    ):
        return {"ok": True, "reason": "no_alias_prefill", "phase_mode": mode}
    state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    if state == "sealed":
        return {"ok": True, "reason": "already_sealed", "state": state}
    if state != "remote" or not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        raise RuntimeError(
            "Cannot activate alias-prefill topology from lifecycle state "
            f"{state!r}"
        )
    started_ns = time.perf_counter_ns()
    before = _uvm_original_parameter_va_snapshot()
    finalized = finalize_cpu_offload_graph_static_alias()
    after = _uvm_original_parameter_va_snapshot()
    if before != after:
        poison_cpu_offload_uvm_original_va_phase(
            "Parameter VA changed while restoring alias-prefill topology"
        )
        raise RuntimeError("Original Parameter VA changed while restoring aliases")
    result = {
        "event": "cpu_offload_uvm_original_va_alias_prefill_activated",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": bool(finalized.get("ok")),
        "reason": "remote_capture_complete",
        "pre_state": state,
        "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        "wall_ns": time.perf_counter_ns() - started_ns,
        "parameter_vas_before": before,
        "parameter_vas_after": after,
        "finalize": finalized,
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
    _append_cpu_offload_manifest(result)
    return result


def validate_cpu_offload_uvm_original_va_remote_graph_replay() -> None:
    """Fail before launch unless the certified remote Graph topology is active."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode == "off":
        return
    certificate = _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    remote_map = _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    failures: list[str] = []
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "remote":
        failures.append(f"state={_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE}")
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID != 0:
        failures.append(f"session={_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID}")
    if not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        failures.append("remote_map_inactive")
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        failures.append("remote_graph_not_captured")
    if not isinstance(certificate, dict):
        failures.append("remote_graph_certificate_missing")
    else:
        if certificate.get("topology") != "remote":
            failures.append("remote_graph_certificate_topology_mismatch")
        if int(certificate.get("actual_h2d_bytes", -1)) != 0:
            failures.append("capture_h2d_nonzero")
        if tuple(certificate.get("parameter_vas", ())) != (
            _uvm_original_parameter_va_snapshot()
        ):
            failures.append("parameter_va_fingerprint_changed")
    if mode == "prefill_graph_remote_decode":
        if not isinstance(
            _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE, dict
        ):
            failures.append("dual_graph_partition_certificate_missing")
        if not _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
            failures.append("alias_prefill_graphs_still_live")
        if _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT <= 0:
            failures.append("remote_decode_graph_not_preserved")
    if not isinstance(remote_map, dict) or remote_map.get("ok") is not True:
        failures.append("remote_map_audit_missing")
    if failures:
        poison_cpu_offload_uvm_original_va_phase(
            "remote Graph replay contract failed: " + ", ".join(failures)
        )
        raise RuntimeError(
            "Refusing original-VA remote CUDA Graph replay: "
            + ", ".join(failures)
        )


def validate_cpu_offload_uvm_original_va_alias_graph_replay() -> None:
    """Fail before launch unless a certified sealed alias Graph is active."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode != "prefill_graph_remote_decode":
        raise RuntimeError(
            "Alias-prefill Graph replay requires "
            "prefill_graph_remote_decode phase mode"
        )
    certificate = _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    failures: list[str] = []
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        failures.append(f"state={_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE}")
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID <= 0:
        failures.append("alias_session_missing")
    if _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        failures.append("remote_map_unexpectedly_active")
    if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED:
        failures.append("alias_graph_not_captured")
    if _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
        failures.append("alias_graphs_already_destroyed")
    if not isinstance(certificate, dict):
        failures.append("alias_graph_certificate_missing")
    else:
        if certificate.get("topology") != "sealed":
            failures.append("alias_graph_certificate_topology_mismatch")
        if int(certificate.get("actual_h2d_bytes", 0)) <= 0:
            failures.append("alias_graph_certificate_has_no_h2d")
        if tuple(certificate.get("parameter_vas", ())) != (
            _uvm_original_parameter_va_snapshot()
        ):
            failures.append("parameter_va_fingerprint_changed")
    if not isinstance(_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE, dict):
        failures.append("dual_graph_partition_certificate_missing")
    if failures:
        poison_cpu_offload_uvm_original_va_phase(
            "alias Graph replay contract failed: " + ", ".join(failures)
        )
        raise RuntimeError(
            "Refusing original-VA alias CUDA Graph replay: "
            + ", ".join(failures)
        )


def _require_all_copy_exact_names_seen() -> None:
    if not _copy_exact_names_enabled() or not _CPU_OFFLOAD_EXACT_PENDING:
        return
    missing = sorted(_CPU_OFFLOAD_EXACT_PENDING)
    preview = ", ".join(repr(name) for name in missing[:8])
    if len(missing) > 8:
        preview += f", ... ({len(missing)} total)"
    raise RuntimeError(
        f"Exact-name CPU offload did not find every requested parameter: {preview}"
    )


def _is_uvm_cpu_offload_backend(backend: str) -> bool:
    return backend in (
        "isolated_uvm_migration",
        "isolated_uvm_remote",
        "isolated_uvm_readmostly_remote",
        "isolated_uvm_prepare_remote",
        "isolated_uvm_prepare_remote_noreadmostly",
        "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
        "isolated_uvm_hybrid_remote",
        "isolated_uvm_hybrid_remote_noreadmostly",
        "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
        "uvm_remote",
        "uvm_pure_remote",
    )


def _hybrid_start_mode() -> str:
    return (
        os.environ.get("VLLM_UVM_HYBRID_START_MODE", "gate_after_first").strip().lower()
    )


def maybe_offload_to_cpu(
    module: torch.nn.Module,
    module_prefix: str | None = None,
) -> torch.nn.Module:
    if (params := next(module.parameters(), None)) is None:
        return module

    device = params.device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES, _CPU_OFFLOAD_ORDER
    global _CPU_OFFLOAD_EXACT_PENDING, _CPU_OFFLOAD_EXACT_SEEN
    global _CPU_OFFLOAD_STATIC_BANK_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
    exact_names_enabled = _copy_exact_names_enabled()
    if not exact_names_enabled and _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()
    uva_available = is_uva_available()

    offload_backend = _normalized_cpu_offload_backend()
    copy_on_forward = offload_backend == "copy"
    if exact_names_enabled and not copy_on_forward:
        raise RuntimeError(
            "Exact-name CPU offload configuration changed after initialization; "
            "VLLM_CPU_OFFLOAD_BACKEND must remain 'copy'"
        )
    uva_offloading = offload_backend == "uva"
    direct_cuda_parameter = not copy_on_forward
    if uva_offloading:
        assert uva_available, "V1 CPU offloading requires uva support"
    if (
        offload_backend != "uva"
        and not copy_on_forward
        and not _is_uvm_cpu_offload_backend(offload_backend)
    ):
        raise ValueError(
            "Unknown VLLM_CPU_OFFLOAD_BACKEND="
            f"{offload_backend!r}; use 'uva', 'copy', "
            "'isolated_uvm_migration', "
            "'isolated_uvm_remote', 'isolated_uvm_readmostly_remote', "
            "'isolated_uvm_prepare_remote', "
            "'isolated_uvm_prepare_remote_noreadmostly', "
            "'isolated_uvm_prepare_remote_noreadmostly_noaccessedby', "
            "'isolated_uvm_hybrid_remote', or "
            "'isolated_uvm_hybrid_remote_noreadmostly', or "
            "'isolated_uvm_hybrid_remote_noreadmostly_noaccessedby'"
        )

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    exact_local_parameter_names: set[str] = set()
    static_bank_keys: dict[str, tuple[Any, ...]] = {}
    graph_static_alias_bindings: dict[
        str, _GraphStaticAliasBinding | _UvmOriginalAliasBinding
    ] = {}
    static_signature_lanes: dict[tuple[Any, ...], int] = {}
    hybrid_remote_ranges: list[tuple[int, int]] = []
    hybrid_remote_device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    for param_index, (param_name, p) in enumerate(module.named_parameters()):
        if not exact_names_enabled and _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break
        bytes_before = p.data.numel() * p.data.element_size()
        full_parameter_name = (
            f"{module_prefix}.{param_name}"
            if module_prefix and param_name
            else (module_prefix or param_name)
        )
        uvm_original_alias = bool(
            exact_names_enabled
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
        )
        accounted_bytes = int(bytes_before)

        if exact_names_enabled:
            assert _CPU_OFFLOAD_EXACT_NAMES is not None
            if full_parameter_name not in _CPU_OFFLOAD_EXACT_NAMES:
                continue
            if full_parameter_name in _CPU_OFFLOAD_EXACT_SEEN:
                raise RuntimeError(
                    "Exact-name CPU offload encountered the same full parameter "
                    f"name more than once: {full_parameter_name!r}"
                )
            if not pin_memory:
                raise RuntimeError(
                    "Exact-name CPU offload requires pinned host memory, but "
                    "is_pin_memory_available() returned False"
                )
            if uvm_original_alias:
                accounted_bytes = int(
                    _uvm_original_alias_parameter_geometry(
                        full_parameter_name=full_parameter_name,
                        tensor=p.data,
                        parameter_bytes=int(bytes_before),
                    )[2]
                )
            if _CPU_OFFLOAD_BYTES + accounted_bytes > _CPU_OFFLOAD_MAX_BYTES:
                raise RuntimeError(
                    "Exact-name CPU offload selection exceeds cpu_offload_gb: "
                    f"adding {full_parameter_name!r} ({accounted_bytes} "
                    "selected bytes) "
                    f"would exceed {_CPU_OFFLOAD_MAX_BYTES} bytes"
                )
            _validate_copy_exact_context(device, check_stream=False)

        manifest_extra: dict[str, Any] = {"offload_backend": offload_backend}

        if uvm_original_alias:
            binding = _create_uvm_original_alias_binding(
                full_parameter_name=full_parameter_name,
                parameter=p,
                tensor=p.data,
                device=device,
                requested_bytes=int(bytes_before),
            )
            graph_static_alias_bindings[param_name] = binding
            exact_local_parameter_names.add(param_name)
            direct_cuda_parameter = False
            p._vllm_uvm_original_alias_binding = binding
            manifest_extra.update(
                {
                    "cpu_data_pinned": None,
                    "cpu_source_deferred_until_weights_loaded": True,
                    "copy_on_forward": True,
                    "copy_exact_names": True,
                    "copy_exact_names_file": _CPU_OFFLOAD_EXACT_NAMES_FILE,
                    "copy_static_bank": False,
                    "graph_static_alias": True,
                    "graph_static_alias_backend": "uvm_original_va",
                    "graph_static_alias_layer_index": binding.plan.layer_index,
                    "graph_static_alias_bank_id": binding.plan.bank_id,
                    "graph_static_alias_consumer_index": (binding.plan.consumer_index),
                    "graph_static_alias_consumer_count": (binding.plan.consumer_count),
                    "graph_static_alias_upload_va": int(
                        binding.arena.upload.data_ptr()
                    ),
                    "graph_static_alias_source_ptr": None,
                    "graph_static_alias_consumer_va": int(binding.consumer.data_ptr()),
                    "graph_static_alias_requested_bytes": int(binding.requested_bytes),
                    "graph_static_alias_parameter_bytes": int(
                        binding.parameter_bytes
                    ),
                    "graph_static_alias_expert_ids": (
                        list(binding.expert_ids)
                        if binding.expert_ids is not None
                        else None
                    ),
                    "graph_static_alias_expert_stride_bytes": int(
                        binding.expert_stride_bytes
                    ),
                    "graph_static_alias_consumer_ranges": tuple(
                        binding.consumer_ranges
                    ),
                    "graph_static_alias_aligned_bytes": int(
                        binding.arena.aligned_bytes
                    ),
                    "graph_static_alias_backing_bytes": int(
                        binding.arena.aligned_bytes
                    ),
                    "graph_static_alias_granularity": None,
                    "graph_static_alias_upload_access": "read_write",
                    "graph_static_alias_consumer_access": "read_only",
                    "graph_static_alias_total_backing_bytes": int(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                    ),
                    "uvm_original_parameter_va": int(binding.consumer.data_ptr()),
                    "uvm_original_parameter_va_retained": True,
                }
            )
        elif _is_uvm_cpu_offload_backend(offload_backend):
            from vllm.device_allocator.uvm import (
                batch_advise_ranges_accessed_by,
                batch_advise_ranges_preferred_location,
                batch_advise_ranges_read_mostly,
                batch_prefetch_ranges_to_device,
                driver_prepare_ranges_remote_read,
                driver_prepare_ranges_remote_read_bg_deferred,
                ensure_uvm_runtime_library,
                probe_driver_capable_range,
                round_ranges_to_pages,
                uvm_local_memory_pool,
            )

            ensure_uvm_runtime_library()
            with torch.cuda.device(device), uvm_local_memory_pool():
                uvm_data = torch.empty_strided(
                    size=p.data.size(),
                    stride=p.data.stride(),
                    dtype=p.data.dtype,
                    layout=p.data.layout,
                    device=device,
                )
            uvm_data.copy_(p.data)

            ptr = int(uvm_data.data_ptr())
            ranges = [(ptr, int(bytes_before))]
            phase_f_ranges = ranges
            if offload_backend in (
                "isolated_uvm_prepare_remote",
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            ) and _env_flag("VLLM_UVM_HYBRID_ROUND_PAGES", True):
                phase_f_ranges = round_ranges_to_pages(ranges)
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            prefetch_cpu_result = batch_prefetch_ranges_to_device(ranges, -1)
            if _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True):
                # Phase-F prepare only acts on CPU-resident, non-GPU-resident
                # pages. Make the CPU prefetch visible before advising and
                # preparing remote-read/bg-promotion state.
                torch.cuda.synchronize()
            remote_like_backend = offload_backend in (
                "isolated_uvm_remote",
                "isolated_uvm_readmostly_remote",
                "isolated_uvm_prepare_remote",
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
                "uvm_remote",
                "uvm_pure_remote",
            )
            preferred_cpu_result = 0
            if remote_like_backend:
                preferred_cpu_result = batch_advise_ranges_preferred_location(
                    ranges, -1
                )
            readmostly_result = 0
            readmostly_rounded_result = 0
            readmostly_backend = offload_backend in (
                "isolated_uvm_readmostly_remote",
                "isolated_uvm_prepare_remote",
                "isolated_uvm_hybrid_remote",
            )
            if readmostly_backend:
                readmostly_result = batch_advise_ranges_read_mostly(
                    ranges,
                    device=int(device_index),
                )
                if phase_f_ranges != ranges and _env_flag(
                    "VLLM_UVM_HYBRID_ROUND_READMOSTLY", True
                ):
                    readmostly_rounded_result = batch_advise_ranges_read_mostly(
                        phase_f_ranges,
                        device=int(device_index),
                    )
            accessed_by_result = 0
            skip_accessed_by_backend = offload_backend in (
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            )
            if remote_like_backend and not skip_accessed_by_backend:
                accessed_by_result = batch_advise_ranges_accessed_by(
                    ranges, int(device_index)
                )
            remote_read_bg_result = (0, 0, 0, 0)
            if offload_backend == "isolated_uvm_prepare_remote":
                remote_read_bg_result = driver_prepare_ranges_remote_read(
                    phase_f_ranges,
                    device=int(device_index),
                )
            elif offload_backend in (
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
            ):
                remote_read_bg_result = driver_prepare_ranges_remote_read(
                    phase_f_ranges,
                    device=int(device_index),
                )
            elif offload_backend in (
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            ):
                remote_read_bg_result = driver_prepare_ranges_remote_read_bg_deferred(
                    phase_f_ranges,
                    device=int(device_index),
                )
                hybrid_remote_ranges.extend(phase_f_ranges)
            if _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True):
                torch.cuda.synchronize()
            managed, managed_reason, managed_code = probe_driver_capable_range(
                ptr, int(bytes_before)
            )

            p._vllm_offloaded_uvm_data = uvm_data
            p.data = uvm_data
            manifest_extra.update(
                {
                    "uvm_managed": bool(managed),
                    "uvm_managed_reason": managed_reason,
                    "uvm_managed_probe_code": int(managed_code),
                    "uvm_data_ptr": ptr,
                    "uvm_prefetch_cpu_result": int(prefetch_cpu_result),
                    "uvm_preferred_cpu_result": int(preferred_cpu_result),
                    "uvm_readmostly_result": int(readmostly_result),
                    "uvm_readmostly_rounded_result": int(readmostly_rounded_result),
                    "uvm_accessed_by_result": int(accessed_by_result),
                    "uvm_accessed_by_device": int(device_index),
                    "uvm_phase_f_range_count": len(phase_f_ranges),
                    "uvm_phase_f_bytes": int(sum(size for _, size in phase_f_ranges)),
                    "uvm_phase_f_rounded": bool(phase_f_ranges != ranges),
                    "uvm_phase_f_first_ptr": int(
                        phase_f_ranges[0][0] if phase_f_ranges else 0
                    ),
                    "uvm_phase_f_first_bytes": int(
                        phase_f_ranges[0][1] if phase_f_ranges else 0
                    ),
                    "uvm_remote_read_bg_success": int(remote_read_bg_result[0]),
                    "uvm_remote_read_bg_failure": int(remote_read_bg_result[1]),
                    "uvm_remote_read_bg_rm_status": int(remote_read_bg_result[2]),
                    "uvm_remote_read_bg_ioctl_ret": int(remote_read_bg_result[3]),
                    "uvm_bg_start_deferred": bool(
                        offload_backend
                        in (
                            "isolated_uvm_hybrid_remote",
                            "isolated_uvm_hybrid_remote_noreadmostly",
                            "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
                        )
                    ),
                    "uvm_sync": _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True),
                    "cpu_data_pinned": False,
                }
            )
        else:
            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            cpu_data_pinned = bool(
                cpu_data.is_pinned() if hasattr(cpu_data, "is_pinned") else False
            )
            if exact_names_enabled and not cpu_data_pinned:
                raise RuntimeError(
                    "Exact-name CPU offload allocated pageable memory for "
                    f"{full_parameter_name!r}; pinned memory is required"
                )

            bank_key: tuple[Any, ...] | None = None
            bank_lane: int | None = None
            bank_tensor_bytes = 0
            graph_static_alias_binding: _GraphStaticAliasBinding | None = None
            if exact_names_enabled and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
                graph_static_alias_binding = _create_graph_static_alias_binding(
                    full_parameter_name=full_parameter_name,
                    tensor=p.data,
                    device=device,
                    requested_bytes=int(bytes_before),
                )
                graph_static_alias_bindings[param_name] = graph_static_alias_binding
            elif exact_names_enabled and _CPU_OFFLOAD_STATIC_BANK:
                size = tuple(int(value) for value in p.data.size())
                stride = tuple(int(value) for value in p.data.stride())
                signature = (
                    size,
                    stride,
                    p.data.dtype,
                    getattr(device, "type", None),
                    int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0),
                )
                bank_lane = static_signature_lanes.get(signature, 0)
                static_signature_lanes[signature] = bank_lane + 1
                bank_key = (*signature, bank_lane)
                bank = _CPU_OFFLOAD_STATIC_BANKS.get(bank_key)
                if bank is None:
                    bank = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device=device,
                    )
                    _CPU_OFFLOAD_STATIC_BANKS[bank_key] = bank
                    bank_tensor_bytes = int(bank.numel() * bank.element_size())
                    _CPU_OFFLOAD_STATIC_BANK_BYTES += bank_tensor_bytes
                else:
                    bank_tensor_bytes = int(bank.numel() * bank.element_size())
                static_bank_keys[param_name] = bank_key

            if not uva_offloading:
                p.data = cpu_data
                direct_cuda_parameter = False
            else:
                # keep the cpu data alive
                p._vllm_offloaded_cpu_data = cpu_data
                p.data = get_cuda_view_from_cpu_tensor(cpu_data)
            manifest_extra.update(
                {
                    "cpu_data_pinned": cpu_data_pinned,
                    "copy_on_forward": bool(copy_on_forward),
                }
            )
            if exact_names_enabled:
                exact_local_parameter_names.add(param_name)
                alias_allocation = (
                    graph_static_alias_binding.arena.allocation
                    if graph_static_alias_binding is not None
                    else None
                )
                alias_plan = (
                    graph_static_alias_binding.plan
                    if graph_static_alias_binding is not None
                    else None
                )
                manifest_extra.update(
                    {
                        "copy_exact_names": True,
                        "copy_exact_names_file": _CPU_OFFLOAD_EXACT_NAMES_FILE,
                        "copy_static_bank": bool(_CPU_OFFLOAD_STATIC_BANK),
                        "graph_static_alias": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
                        "graph_static_alias_layer_index": (
                            alias_plan.layer_index if alias_plan is not None else None
                        ),
                        "graph_static_alias_bank_id": (
                            alias_plan.bank_id if alias_plan is not None else None
                        ),
                        "graph_static_alias_consumer_index": (
                            alias_plan.consumer_index
                            if alias_plan is not None
                            else None
                        ),
                        "graph_static_alias_consumer_count": (
                            alias_plan.consumer_count
                            if alias_plan is not None
                            else None
                        ),
                        "graph_static_alias_token": (
                            int(alias_allocation.token)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_upload_va": (
                            int(alias_allocation.upload_va)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_source_ptr": (
                            int(cpu_data.data_ptr())
                            if graph_static_alias_binding is not None
                            else None
                        ),
                        "graph_static_alias_consumer_va": (
                            int(graph_static_alias_binding.consumer.data_ptr())
                            if graph_static_alias_binding is not None
                            else None
                        ),
                        "graph_static_alias_requested_bytes": (
                            int(alias_allocation.requested_size)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_aligned_bytes": (
                            int(alias_allocation.aligned_size)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_backing_bytes": (
                            int(alias_allocation.backing_bytes)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_granularity": (
                            int(alias_allocation.granularity)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_upload_access": (
                            str(alias_allocation.upload_access)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_consumer_access": (
                            str(alias_allocation.consumer_access)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_total_backing_bytes": int(
                            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                        ),
                        "copy_bank_bytes": int(_CPU_OFFLOAD_STATIC_BANK_BYTES),
                        "copy_bank_tensor_bytes": int(bank_tensor_bytes),
                        "copy_bank_lane": bank_lane,
                        "copy_bank_key": (
                            {
                                "shape": list(bank_key[0]),
                                "stride": list(bank_key[1]),
                                "dtype": str(bank_key[2]),
                                "device_type": bank_key[3],
                                "device_index": bank_key[4],
                                "lane": bank_key[5],
                            }
                            if bank_key is not None
                            else None
                        ),
                    }
                )

        if exact_names_enabled:
            _CPU_OFFLOAD_EXACT_SEEN.add(full_parameter_name)
            _CPU_OFFLOAD_EXACT_PENDING.remove(full_parameter_name)

        _CPU_OFFLOAD_BYTES += accounted_bytes
        offloaded_parameters = True
        _CPU_OFFLOAD_ORDER += 1
        record = {
            "event": "cpu_offload_parameter",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "module_prefix": module_prefix,
            "module_class": module.__class__.__name__,
            "parameter_name": param_name,
            "full_parameter_name": full_parameter_name,
            "param_index_in_module": param_index,
            "offload_order": _CPU_OFFLOAD_ORDER,
            "shape": list(p.data.size()),
            "stride": list(p.data.stride()),
            "dtype": str(p.data.dtype),
            "bytes": int(accounted_bytes),
            "parameter_bytes": int(bytes_before),
            "cumulative_offload_bytes": int(_CPU_OFFLOAD_BYTES),
            "max_offload_bytes": int(_CPU_OFFLOAD_MAX_BYTES),
            "pin_memory": bool(pin_memory),
            "uva_available": bool(uva_available),
            "uva_offloading": bool(offload_backend == "uva" and uva_offloading),
            "uvm_offloading": bool(
                _is_uvm_cpu_offload_backend(offload_backend) or uvm_original_alias
            ),
            "device_before": str(device),
            "device_after": str(p.data.device),
        }
        record.update(manifest_extra)
        _append_cpu_offload_manifest(record)

    if offloaded_parameters and not direct_cuda_parameter:
        original_forward = module.forward
        copy_forward_index = 0
        copy_trace = _env_flag("VLLM_CPU_OFFLOAD_COPY_TRACE", False)
        copy_sync = _env_flag("VLLM_CPU_OFFLOAD_COPY_SYNC", False)
        copy_static_bank = bool(exact_names_enabled and _CPU_OFFLOAD_STATIC_BANK)
        graph_static_alias = bool(
            exact_names_enabled and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
        )
        uvm_original_graph_alias = bool(
            graph_static_alias
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
        )

        def forward(*args, **kwargs):
            global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
            global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
            global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
            nonlocal copy_forward_index
            module.forward = original_forward
            copy_forward_index += 1
            graph_forward_call_index: int | None = None
            graph_capture_active: bool | None = None
            graph_capture_phase = False
            stream_id: int | None = None
            copy_bytes = 0
            copy_tensor_count = 0
            graph_static_alias_actual_h2d_bytes = 0
            graph_static_alias_actual_h2d_tensor_count = 0
            copy_cuda_ms: float | None = None
            copy_submit_us: float | None = None
            copy_ok = False
            try:
                if exact_names_enabled:
                    _require_all_copy_exact_names_seen()
                    _validate_copy_exact_context(
                        device,
                        check_stream=(copy_static_bank or graph_static_alias),
                    )
                if graph_static_alias:
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS += 1
                    graph_forward_call_index = int(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
                    )
                    graph_capture_active = bool(
                        torch.cuda.is_current_stream_capturing()
                    )
                    graph_capture_phase = bool(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
                    )
                    stream_id = int(torch.cuda.current_stream(device).cuda_stream)

                state_items = list(module.state_dict().items())
                if exact_names_enabled:
                    state_keys = {key for key, _ in state_items}
                    missing_local = exact_local_parameter_names - state_keys
                    if missing_local:
                        raise RuntimeError(
                            "Exact-name CPU offload parameters disappeared from "
                            f"module.state_dict(): {sorted(missing_local)!r}"
                        )
                for key, value in state_items:
                    if (
                        exact_names_enabled
                        and key in exact_local_parameter_names
                        and (
                            value.device.type
                            != ("cuda" if uvm_original_graph_alias else "cpu")
                        )
                    ):
                        raise RuntimeError(
                            "Exact-name static-alias parameter is on the wrong "
                            f"device: {key!r} is on {value.device}"
                        )
                    if (
                        copy_static_bank or graph_static_alias
                    ) and key in exact_local_parameter_names:
                        binding = graph_static_alias_bindings.get(key)
                        if isinstance(binding, _UvmOriginalAliasBinding):
                            copy_bytes += int(binding.requested_bytes)
                        else:
                            copy_bytes += value.numel() * value.element_size()
                        copy_tensor_count += 1
                    elif (
                        copy_static_bank or graph_static_alias
                    ) and value.device.type == "cpu":
                        raise RuntimeError(
                            "Static bank/alias forward found an unregistered CPU "
                            f"state tensor: {key!r}"
                        )
                    elif (
                        not copy_static_bank
                        and not graph_static_alias
                        and value.device.type == "cpu"
                    ):
                        copy_bytes += value.numel() * value.element_size()
                        copy_tensor_count += 1

                start_event = None
                end_event = None
                if copy_sync:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(device))

                copy_submit_start_ns = time.perf_counter_ns()
                if graph_static_alias:
                    device_state = {}
                    for key, value in state_items:
                        binding = graph_static_alias_bindings.get(key)
                        if binding is None:
                            device_state[key] = value
                            continue
                        if isinstance(binding, _UvmOriginalAliasBinding):
                            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed":
                                if binding.source is None:
                                    raise RuntimeError(
                                        "UVM original-VA alias was sealed "
                                        "without a pinned host source for "
                                        f"{binding.full_parameter_name!r}"
                                    )
                                binding.arena.upload.copy_(
                                    binding.source, non_blocking=True
                                )
                                graph_static_alias_actual_h2d_bytes += int(
                                    binding.requested_bytes
                                )
                                graph_static_alias_actual_h2d_tensor_count += 1
                            elif (
                                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "remote"
                                or not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
                            ):
                                raise RuntimeError(
                                    "UVM original-VA forward requires a sealed "
                                    "prefill topology or an audited active "
                                    "remote-decode topology; current state is "
                                    f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}, "
                                    "remote_map_active="
                                    f"{_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE!r}"
                                )
                            if int(value.data_ptr()) != int(
                                binding.consumer.data_ptr()
                            ):
                                raise RuntimeError(
                                    "UVM original-VA state_dict changed a "
                                    "selected parameter VA"
                                )
                            device_state[key] = binding.consumer
                        else:
                            binding.arena.upload.copy_(value, non_blocking=True)
                            graph_static_alias_actual_h2d_bytes += int(
                                value.numel() * value.element_size()
                            )
                            graph_static_alias_actual_h2d_tensor_count += 1
                            device_state[key] = binding.consumer
                elif copy_static_bank:
                    device_state = {}
                    for key, value in state_items:
                        bank_key = static_bank_keys.get(key)
                        if bank_key is None:
                            device_state[key] = value
                            continue
                        bank = _CPU_OFFLOAD_STATIC_BANKS.get(bank_key)
                        if bank is None:
                            raise RuntimeError(
                                "Static CPU-offload bank was reset or released "
                                f"before forward for {key!r}"
                            )
                        bank.copy_(value, non_blocking=True)
                        device_state[key] = bank
                else:
                    device_state = {
                        # GPU values are a no-op. CPU values allocate a
                        # temporary CUDA tensor and enqueue H2D on the current
                        # stream. Keep this native path unchanged when no exact
                        # name file is configured.
                        key: value.to(device, non_blocking=True)
                        for key, value in state_items
                    }
                copy_submit_us = (time.perf_counter_ns() - copy_submit_start_ns) / 1e3

                if copy_sync:
                    assert start_event is not None and end_event is not None
                    end_event.record(torch.cuda.current_stream(device))
                    end_event.synchronize()
                    copy_cuda_ms = float(start_event.elapsed_time(end_event))

                output = functional_call(
                    module,
                    device_state,
                    args=args,
                    kwargs=kwargs,
                )
                copy_ok = True
                return output
            finally:
                module.forward = forward
                if (
                    uvm_original_graph_alias
                    and graph_capture_phase
                    and graph_capture_active
                ):
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT += 1
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES += int(
                        graph_static_alias_actual_h2d_bytes
                    )
                if copy_trace:
                    _append_cpu_offload_manifest(
                        {
                            "event": "cpu_offload_copy_forward",
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "module_prefix": module_prefix,
                            "module_class": module.__class__.__name__,
                            "forward_index": copy_forward_index,
                            "copy_bytes": int(copy_bytes),
                            "copy_tensor_count": int(copy_tensor_count),
                            "copy_submit_us": copy_submit_us,
                            "copy_cuda_ms": copy_cuda_ms,
                            "copy_sync": bool(copy_sync),
                            "copy_exact_names": bool(exact_names_enabled),
                            "copy_exact_name_count": len(exact_local_parameter_names),
                            "copy_exact_names_file": (
                                _CPU_OFFLOAD_EXACT_NAMES_FILE
                                if exact_names_enabled
                                else None
                            ),
                            "copy_static_bank": bool(copy_static_bank),
                            "graph_static_alias": bool(graph_static_alias),
                            "graph_static_alias_backend": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
                                if graph_static_alias
                                else None
                            ),
                            "graph_static_alias_lifecycle_state": (
                                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
                                if uvm_original_graph_alias
                                else None
                            ),
                            "graph_static_alias_actual_h2d_bytes": int(
                                graph_static_alias_actual_h2d_bytes
                            ),
                            "graph_static_alias_actual_h2d_tensor_count": int(
                                graph_static_alias_actual_h2d_tensor_count
                            ),
                            "graph_static_alias_copy_submitted": bool(
                                graph_static_alias_actual_h2d_tensor_count
                            ),
                            "graph_static_alias_forward_call_index": (
                                graph_forward_call_index
                            ),
                            "graph_static_alias_capture_active": (graph_capture_active),
                            "graph_static_alias_capture_phase": bool(
                                graph_capture_phase
                            ),
                            "graph_static_alias_stream_id": stream_id,
                            "graph_static_alias_backing_bytes": int(
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                                if graph_static_alias
                                else 0
                            ),
                            "graph_static_alias_arena_count": int(
                                len(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS)
                                + len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
                                if graph_static_alias
                                else 0
                            ),
                            "copy_bank_bytes": int(
                                _CPU_OFFLOAD_STATIC_BANK_BYTES
                                if copy_static_bank
                                else 0
                            ),
                            "copy_bank_tensor_count": int(
                                len(_CPU_OFFLOAD_STATIC_BANKS)
                                if copy_static_bank
                                else 0
                            ),
                            "ok": bool(copy_ok),
                        }
                    )

        module.forward = forward

    if offloaded_parameters and offload_backend in (
        "isolated_uvm_hybrid_remote",
        "isolated_uvm_hybrid_remote_noreadmostly",
        "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
    ):
        from vllm.device_allocator.uvm import (
            driver_pause_ranges_bg_promotion,
            driver_resume_ranges_bg_promotion,
            driver_start_ranges_bg_promotion,
            round_ranges_to_pages,
        )

        original_forward = module.forward
        bg_started = False
        hybrid_remote_gate_ranges = round_ranges_to_pages(hybrid_remote_ranges)
        hybrid_start_mode = _hybrid_start_mode()
        hybrid_gate_enabled = _env_flag("VLLM_UVM_HYBRID_GATE_ENABLE", True)

        if hybrid_start_mode in ("eager", "immediate", "after_prepare"):
            start_result = driver_start_ranges_bg_promotion(
                hybrid_remote_gate_ranges,
                device=int(hybrid_remote_device_index),
            )
            bg_started = int(start_result[0]) > 0
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_hybrid_remote_eager_start",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "module_prefix": module_prefix,
                    "module_class": module.__class__.__name__,
                    "range_count": len(hybrid_remote_gate_ranges),
                    "bytes": int(sum(size for _, size in hybrid_remote_gate_ranges)),
                    "start_success": int(start_result[0]),
                    "start_failure": int(start_result[1]),
                    "start_rm_status": int(start_result[2]),
                    "start_ioctl_ret": int(start_result[3]),
                    "bg_started": bool(bg_started),
                    "start_mode": hybrid_start_mode,
                }
            )

        if not hybrid_gate_enabled:
            return module

        def hybrid_forward(*args, **kwargs):
            nonlocal bg_started
            pause_result = driver_pause_ranges_bg_promotion(
                hybrid_remote_gate_ranges,
                device=int(hybrid_remote_device_index),
            )
            ok = False
            try:
                output = original_forward(*args, **kwargs)
                ok = True
                return output
            finally:
                resume_result = driver_resume_ranges_bg_promotion(
                    hybrid_remote_gate_ranges,
                    device=int(hybrid_remote_device_index),
                )
                start_result = (0, 0, 0, 0)
                if ok and not bg_started:
                    start_result = driver_start_ranges_bg_promotion(
                        hybrid_remote_gate_ranges,
                        device=int(hybrid_remote_device_index),
                    )
                    bg_started = int(start_result[0]) > 0
                _append_cpu_offload_manifest(
                    {
                        "event": "cpu_offload_hybrid_remote_gate",
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "module_prefix": module_prefix,
                        "module_class": module.__class__.__name__,
                        "range_count": len(hybrid_remote_gate_ranges),
                        "bytes": int(
                            sum(size for _, size in hybrid_remote_gate_ranges)
                        ),
                        "pause_success": int(pause_result[0]),
                        "pause_failure": int(pause_result[1]),
                        "pause_rm_status": int(pause_result[2]),
                        "pause_ioctl_ret": int(pause_result[3]),
                        "resume_success": int(resume_result[0]),
                        "resume_failure": int(resume_result[1]),
                        "resume_rm_status": int(resume_result[2]),
                        "resume_ioctl_ret": int(resume_result[3]),
                        "start_success": int(start_result[0]),
                        "start_failure": int(start_result[1]),
                        "start_rm_status": int(start_result[2]),
                        "start_ioctl_ret": int(start_result[3]),
                        "bg_started": bool(bg_started),
                        "start_mode": hybrid_start_mode,
                        "gate_enabled": bool(hybrid_gate_enabled),
                    }
                )

        module.forward = hybrid_forward

    return module


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            maybe_offload_to_cpu(
                layer_fn(prefix=f"{prefix}.{idx}"),
                module_prefix=f"{prefix}.{idx}",
            )
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    _require_all_copy_exact_names_seen()
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + ".")
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def get_draft_quant_config(
    vllm_config: VllmConfig,
) -> QuantizationConfig | None:
    """Get quantization config for Draft models.

    Draft models should use their own quantization config instead of the verifier/target
    model's config. This helper retrieves the draft model's quantization config.

    Args:
        vllm_config: The vLLM configuration object.

    Returns:
        The draft model's config if available, None otherwise.
    """
    draft_model_config = vllm_config.speculative_config.draft_model_config
    draft_load_config = vllm_config.load_config

    return (
        VllmConfig.get_quantization_config(draft_model_config, draft_load_config)
        if draft_model_config
        else None
    )


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    else:
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(
    values: torch.Tensor, topk: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized topk implementation that uses torch.max for k=1 case.

    This function provides better performance for the common case of k=1
    by using torch.max instead of the more general torch.topk.

    Args:
        values: Input tensor to find top-k values from
        topk: Number of top values to return (k). Must be > 0.
        dim: Dimension along which to compute topk

    Returns:
        Tuple of (values, indices) where values are the top-k values
        and indices are their corresponding indices in the input tensor
    """
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.sequence_parallel_chunk_impl(x)


def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)


def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    seq_len = cdiv(x.size(0), tp_size)
    shape = list(x.shape)
    shape[0] = seq_len
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op(
    op_name="sequence_parallel_chunk_impl",
    op_func=sequence_parallel_chunk_impl,
    fake_impl=sequence_parallel_chunk_impl_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def process_eagle_weight(
    model: nn.Module,
    name: str,
) -> None:
    """
    Update EAGLE model flags based on loaded weight name.
    This should be called during weight loading to detect if a model
    has its own lm_head or embed_tokens weight.
    Args:
        model: The model instance (must support EAGLE)
        name: The name of the weight to process
    """
    if not supports_any_eagle(model):
        return

    # To prevent overriding with target model's layers
    if "lm_head" in name:
        model.has_own_lm_head = True
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True
