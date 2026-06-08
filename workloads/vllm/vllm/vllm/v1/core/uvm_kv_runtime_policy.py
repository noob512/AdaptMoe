# SPDX-License-Identifier: Apache-2.0
"""Stage J UVM KV runtime pressure policy.

This module intentionally stays at the vLLM block-manager boundary.  It traces
and optionally denies new KV block admission; it does not migrate raw allocator
pointers or alter active request block tables behind the scheduler's back.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.device_allocator.uvm_pool_coordinator import (
    record_uvm_pool_pressure,
    request_uvm_pool_action,
)
from vllm.device_allocator.uvm_policy_runtime import get_uvm_policy_runtime
from vllm.device_allocator.uvm import (
    prefetch_range_to_cpu,
    prefetch_range_to_device,
)
from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_residency_region_snapshot,
    get_uvm_pool_residency_manager,
    mark_uvm_residency_state,
    record_uvm_residency_kv_metadata_candidate,
    record_uvm_residency_action,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def _read_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _read_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected non-negative integer.", name, raw)
        return default


def _read_mode_env() -> str:
    mode = os.environ.get("VLLM_UVM_KV_RUNTIME_MODE", "trace_only")
    mode = mode.strip().lower()
    if mode not in ("trace_only", "enforce"):
        logger.warning(
            "Ignoring invalid VLLM_UVM_KV_RUNTIME_MODE=%r; using trace_only.",
            mode,
        )
        return "trace_only"
    return mode


def _read_action_mode_env(name: str, default: str, choices: set[str]) -> str:
    mode = os.environ.get(name, default).strip().lower()
    if mode not in choices:
        logger.warning("Ignoring invalid %s=%r; using %s.", name, mode, default)
        return default
    return mode


def _read_choice_env(name: str, default: str, choices: set[str]) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        logger.warning("Ignoring invalid %s=%r; using %s.", name, value, default)
        return default
    return value


def _block_hash_to_text(block_hash: Any) -> str | None:
    if block_hash is None:
        return None
    if isinstance(block_hash, bytes):
        return block_hash.hex()
    return str(block_hash)


class UvmKvRuntimePolicy:
    """Trace-only or soft-admission Stage J policy for KV block pressure."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_KV_RUNTIME_ENABLE", False)
        self.mode = _read_mode_env()
        self.trace_file = os.environ.get("VLLM_UVM_KV_RUNTIME_TRACE_FILE", "")
        self.policy = os.environ.get(
            "VLLM_UVM_KV_RUNTIME_EVICTION_POLICY", "lru_prefix_cache"
        ).strip().lower()
        self.candidate_limit = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT", 16
        )
        self.prefix_evict_enable = _read_bool_env(
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_ENABLE", False
        )
        self.prefix_evict_max_blocks = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_PREFIX_EVICT_MAX_BLOCKS", 0
        )
        self.stage_l5_metadata_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") in ("stage_l5", "stage_l6"),
        )
        self.stage_l6_data_offload_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l6",
        )
        self.stage_l6_prefetch_back_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l6",
        )
        self.stage_l6_pointer_tag_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_POINTER_TAG_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l6"
            and (
                _read_bool_env("VLLM_UVM_RESIDENCY_TAG_ONLY", False)
                or _read_bool_env("VLLM_UVM_DRIVER_TAG_ENABLE", False)
            ),
        )
        self.stage_n_allocator_kv_driver_tag_enable = _read_bool_env(
            "VLLM_UVM_DRIVER_KV_TAG_ENABLE",
            False,
        )
        self.stage_l6_pointer_tag_granularity = _read_choice_env(
            "VLLM_UVM_RESIDENCY_KV_POINTER_TAG_GRANULARITY",
            "tensor",
            {"tensor", "range"},
        )
        self.stage_l6_data_offload_mode = _read_action_mode_env(
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE",
            "trace_only",
            {"trace_only", "prefetch_cpu"},
        )
        self.stage_l6_prefetch_back_mode = _read_action_mode_env(
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE",
            "trace_only",
            {"trace_only", "prefetch"},
        )
        self.active_protect_enable = _read_bool_env(
            "VLLM_UVM_KV_ACTIVE_PROTECT_ENABLE",
            False,
        )
        self.active_protect_mode = _read_action_mode_env(
            "VLLM_UVM_KV_ACTIVE_PROTECT_MODE",
            "trace_only",
            {"trace_only", "advise", "prefetch", "advise_prefetch"},
        )
        self.active_protect_max_blocks = _read_int_env(
            "VLLM_UVM_KV_ACTIVE_PROTECT_MAX_BLOCKS", 0
        )
        try:
            self.stage_l6_device = int(
                os.environ.get("VLLM_UVM_RESIDENCY_KV_PREFETCH_DEVICE", "0") or 0
            )
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_UVM_RESIDENCY_KV_PREFETCH_DEVICE; using 0."
            )
            self.stage_l6_device = 0
        try:
            self.active_protect_device = int(
                os.environ.get("VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE",
                               str(self.stage_l6_device)) or self.stage_l6_device
            )
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE; using %d.",
                self.stage_l6_device,
            )
            self.active_protect_device = self.stage_l6_device
        self.stage_l6_max_offload_blocks = _read_int_env(
            "VLLM_UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP", 0
        )
        self.stage_l6_pointer_map_file = os.environ.get(
            "VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE", ""
        )
        self._stage_l6_pointer_map_loaded = False
        self._stage_l6_pointer_ranges: dict[str, list[tuple[int, int]]] = {}
        self.budget_bytes = _read_int_env("VLLM_UVM_KV_RUNTIME_BUDGET_BYTES", 0)
        budget_blocks_override = _read_int_env(
            "VLLM_UVM_KV_RUNTIME_BUDGET_BLOCKS", 0
        )
        self.total_bytes = sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)
        self.total_blocks = max(kv_cache_config.num_blocks - 1, 0)
        self.num_kv_cache_groups = max(len(kv_cache_config.kv_cache_groups), 1)
        self.bytes_per_block = (
            self.total_bytes // kv_cache_config.num_blocks
            if kv_cache_config.num_blocks > 0 and self.total_bytes > 0
            else 0
        )
        self.bytes_per_group_block = (
            self.bytes_per_block // self.num_kv_cache_groups
            if self.num_kv_cache_groups > 0
            else self.bytes_per_block
        )
        if budget_blocks_override > 0:
            self.budget_blocks = budget_blocks_override
        elif self.budget_bytes > 0 and self.bytes_per_block > 0:
            self.budget_blocks = max(self.budget_bytes // self.bytes_per_block, 1)
        else:
            self.budget_blocks = 0
        if self.policy not in ("lru_prefix_cache", "scheduler_aware"):
            logger.warning(
                "Ignoring invalid VLLM_UVM_KV_RUNTIME_EVICTION_POLICY=%r; "
                "using lru_prefix_cache.",
                self.policy,
            )
            self.policy = "lru_prefix_cache"
        self._lock = threading.Lock()
        self._config_recorded = False
        self._summary = {
            "allocation_pressure_records": 0,
            "over_budget_records": 0,
            "would_deny_records": 0,
            "denied_records": 0,
            "candidate_records": 0,
            "prefix_evict_attempt_records": 0,
            "prefix_evict_success_blocks": 0,
            "prefix_evict_noop_records": 0,
            "prefix_evict_failed_records": 0,
            "prefix_evict_coordinator_reject_records": 0,
            "prefix_cache_eviction_records": 0,
            "unsafe_prefix_cache_eviction_records": 0,
            "reuse_failure_records": 0,
            "stage_l5_kv_region_register_records": 0,
            "stage_l5_residency_request_records": 0,
            "stage_l5_safe_candidate_records": 0,
            "stage_l5_unsafe_candidate_records": 0,
            "stage_l5_data_offload_records": 0,
            "stage_l6_data_offload_attempt_records": 0,
            "stage_l6_data_offload_issued_records": 0,
            "stage_l6_data_offload_fallback_records": 0,
            "stage_l6_prefetch_back_issued_records": 0,
            "stage_l6_prefetch_back_fallback_records": 0,
            "stage_l6_unsafe_offload_records": 0,
            "active_protect_attempt_records": 0,
            "active_protect_issued_records": 0,
            "active_protect_fallback_records": 0,
            "active_protect_blocks": 0,
            "active_protect_bytes": 0,
        }

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.trace_file)

    def record_config(self) -> None:
        if not self.active or self._config_recorded:
            return
        self._config_recorded = True
        self._write(
            {
                "action": "runtime_config",
                "enabled": self.enabled,
                "mode": self.mode,
                "policy": self.policy,
                "budget_bytes": self.budget_bytes,
                "budget_blocks": self.budget_blocks,
                "bytes_per_block": self.bytes_per_block,
                "total_bytes": self.total_bytes,
                "total_blocks": self.total_blocks,
                "candidate_limit": self.candidate_limit,
                "prefix_evict_enable": self.prefix_evict_enable,
                "prefix_evict_max_blocks": self.prefix_evict_max_blocks,
                "stage_l5_metadata_enable": self.stage_l5_metadata_enable,
                "stage_l6_data_offload_enable": self.stage_l6_data_offload_enable,
                "stage_l6_prefetch_back_enable": self.stage_l6_prefetch_back_enable,
                "stage_l6_pointer_tag_enable": self.stage_l6_pointer_tag_enable,
                "stage_n_allocator_kv_driver_tag_enable": (
                    self.stage_n_allocator_kv_driver_tag_enable
                ),
                "stage_l6_pointer_tag_granularity": (
                    self.stage_l6_pointer_tag_granularity
                ),
                "stage_l6_data_offload_mode": self.stage_l6_data_offload_mode,
                "stage_l6_prefetch_back_mode": self.stage_l6_prefetch_back_mode,
                "active_protect_enable": self.active_protect_enable,
                "active_protect_mode": self.active_protect_mode,
                "active_protect_device": self.active_protect_device,
                "active_protect_max_blocks": self.active_protect_max_blocks,
                "stage_l6_max_offload_blocks": self.stage_l6_max_offload_blocks,
                "stage_l6_pointer_map_file": self.stage_l6_pointer_map_file,
                "num_kv_cache_groups": self.num_kv_cache_groups,
            }
        )

    def register_stage_l5_kv_blocks(self, block_pool: Any) -> None:
        """Register KV block metadata in Stage L residency.

        L6 reuses the same block metadata registration and upgrades individual
        regions to offload_capable later if raw KV pointers become available.
        """
        if not self.active or not self.stage_l5_metadata_enable:
            return
        if not get_uvm_pool_residency_manager().active:
            return
        bytes_per_region = max(self.bytes_per_group_block, 1)
        total_registered = 0
        for group_id in range(self.num_kv_cache_groups):
            for block in getattr(block_pool, "blocks", []):
                if getattr(block, "is_null", False):
                    continue
                block_id = int(getattr(block, "block_id", -1))
                if block_id < 0:
                    continue
                register_uvm_residency_region(
                    pool="kv",
                    region_id=f"kv:{group_id}:{block_id}",
                    ptr=None,
                    bytes=bytes_per_region,
                    state="gpu",
                    evictable=True,
                    active=False,
                    metadata={
                        "stage": "stage_l5",
                        "source": "kv_block_metadata",
                        "group_id": group_id,
                        "block_id": block_id,
                        "bytes_per_block": bytes_per_region,
                        "ref_cnt": getattr(block, "ref_cnt", None),
                        "is_prefix_cache": getattr(block, "block_hash", None)
                        is not None,
                        "in_free_queue": True,
                        "block_hash": _block_hash_to_text(
                            getattr(block, "block_hash", None)
                        ),
                        "offload_capable": False,
                        "metadata_only": not self.stage_l6_data_offload_enable,
                    },
                )
                total_registered += 1
        self._summary["stage_l5_kv_region_register_records"] += total_registered
        self._write(
            {
                "action": "stage_l5_kv_metadata_registered",
                "registered_regions": total_registered,
                "bytes_per_region": bytes_per_region,
                "num_kv_cache_groups": self.num_kv_cache_groups,
                "metadata_only": True,
                "data_offload_issued": False,
            }
        )
        self._maybe_load_stage_l6_pointer_map(reason="after_stage_l5_metadata")

    def register_stage_l6_kv_pointer_map(
        self,
        kv_pointer_map: dict[tuple[int, int], tuple[int, int]],
        *,
        source: str = "kv_pointer_map",
        register_driver_regions: bool | None = None,
    ) -> None:
        """Upgrade registered KV regions with raw pointer ranges when available."""
        if not self.active:
            return
        if not get_uvm_pool_residency_manager().active:
            return
        if register_driver_regions is None:
            register_driver_regions = self.stage_l6_data_offload_enable
        allocator_driver_tag_skip = (
            self.stage_n_allocator_kv_driver_tag_enable
            and bool(register_driver_regions)
        )
        if allocator_driver_tag_skip:
            register_driver_regions = False
        registered = 0
        driver_regions_skipped = 0
        for (group_id, block_id), (ptr, size_bytes) in kv_pointer_map.items():
            if ptr <= 0 or size_bytes <= 0:
                continue
            self._stage_l6_pointer_ranges[f"kv:{group_id}:{block_id}"] = [
                (int(ptr), int(size_bytes))
            ]
            if not register_driver_regions:
                if allocator_driver_tag_skip:
                    driver_regions_skipped += 1
                registered += 1
                continue
            register_uvm_residency_region(
                pool="kv",
                region_id=f"kv:{group_id}:{block_id}",
                ptr=int(ptr),
                bytes=int(size_bytes),
                state="gpu",
                evictable=True,
                active=False,
                metadata={
                    "stage": "stage_l6",
                    "source": source,
                    "group_id": group_id,
                    "block_id": block_id,
                    "bytes_per_block": int(size_bytes),
                    "offload_capable": True,
                    "metadata_only": False,
                    "range_count": 1,
                },
            )
            registered += 1
        self._write(
            {
                "action": "stage_l6_kv_pointer_map_registered",
                "registered_regions": registered,
                "driver_region_records": registered if register_driver_regions else 0,
                "driver_pointer_tags_skipped": driver_regions_skipped,
                "driver_pointer_tag_skip_reason": (
                    "allocator_kv_driver_tag_enable"
                    if allocator_driver_tag_skip
                    else None
                ),
                "allocator_kv_driver_tag_enable": (
                    self.stage_n_allocator_kv_driver_tag_enable
                ),
                "data_offload_capable": registered > 0,
                "pointer_tag_enable": self.stage_l6_pointer_tag_enable,
                "data_offload_enable": self.stage_l6_data_offload_enable,
            }
        )

    def _maybe_load_stage_l6_pointer_map(self, *, reason: str = "unspecified") -> None:
        if self._stage_l6_pointer_map_loaded:
            return
        self._stage_l6_pointer_map_loaded = True
        if (
            not self.enabled
            or not (
                self.stage_l6_pointer_tag_enable
                or self.stage_l6_data_offload_enable
                or (
                    self.active_protect_enable
                    and self.active_protect_mode != "trace_only"
                )
            )
            or not self.stage_l6_pointer_map_file
        ):
            return
        path = self.stage_l6_pointer_map_file
        if not os.path.exists(path):
            self._stage_l6_pointer_map_loaded = False
            self._write(
                {
                    "action": "stage_l6_kv_pointer_map_load_deferred",
                    "path": path,
                    "reason": reason,
                    "deferred_reason": "file_not_found",
                }
            )
            return
        pointer_map: dict[tuple[int, int], tuple[int, int]] = {}
        range_map: dict[str, list[tuple[int, int]]] = {}
        bytes_map: dict[str, int] = {}
        range_metadata_map: dict[str, list[dict[str, Any]]] = {}
        tensor_spans: dict[str, dict[str, Any]] = {}
        records_seen = 0
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("action") != "stage_l6_kv_pointer_map":
                        continue
                    records_seen += 1
                    for item in record.get("blocks") or []:
                        try:
                            group_id = int(item["group_id"])
                            block_id = int(item["block_id"])
                            size_bytes = int(item["bytes"])
                        except (KeyError, TypeError, ValueError):
                            continue
                        ranges: list[tuple[int, int]] = []
                        range_metadata: list[dict[str, Any]] = []
                        for raw_range in item.get("ranges") or []:
                            try:
                                ptr = int(raw_range["ptr"])
                                range_bytes = int(raw_range["bytes"])
                            except (KeyError, TypeError, ValueError):
                                continue
                            if ptr > 0 and range_bytes > 0:
                                ranges.append((ptr, range_bytes))
                                tensor_name = raw_range.get("tensor_name")
                                range_metadata.append(
                                    {
                                        "ptr": ptr,
                                        "bytes": range_bytes,
                                        "tensor_name": tensor_name,
                                    }
                                )
                                if tensor_name:
                                    span = tensor_spans.setdefault(
                                        str(tensor_name),
                                        {
                                            "start": ptr,
                                            "end": ptr + range_bytes,
                                            "range_count": 0,
                                            "bytes_sum": 0,
                                        },
                                    )
                                    span["start"] = min(int(span["start"]), ptr)
                                    span["end"] = max(
                                        int(span["end"]),
                                        ptr + range_bytes,
                                    )
                                    span["range_count"] = int(span["range_count"]) + 1
                                    span["bytes_sum"] = int(span["bytes_sum"]) + range_bytes
                        if not ranges:
                            try:
                                ptr = int(item["ptr"])
                            except (KeyError, TypeError, ValueError):
                                ptr = 0
                            if ptr > 0 and size_bytes > 0:
                                ranges.append((ptr, size_bytes))
                                range_metadata.append(
                                    {
                                        "ptr": ptr,
                                        "bytes": size_bytes,
                                        "tensor_name": item.get("tensor_name"),
                                    }
                                )
                        if group_id >= 0 and block_id >= 0 and ranges:
                            region_id = f"kv:{group_id}:{block_id}"
                            first_ptr = ranges[0][0]
                            total_bytes = sum(size for _, size in ranges)
                            pointer_map[(group_id, block_id)] = (first_ptr, total_bytes)
                            range_map[region_id] = ranges
                            bytes_map[region_id] = total_bytes
                            range_metadata_map[region_id] = range_metadata
        except OSError as exc:
            self._stage_l6_pointer_map_loaded = False
            self._write(
                {
                    "action": "stage_l6_kv_pointer_map_load_failed",
                    "path": path,
                    "reason": reason,
                    "error": repr(exc),
                }
            )
            return
        self._write(
            {
                "action": "stage_l6_kv_pointer_map_loaded",
                "path": path,
                "reason": reason,
                "records_seen": records_seen,
                "entries": len(pointer_map),
                "range_entries": sum(len(ranges) for ranges in range_map.values()),
                "tensor_span_entries": len(tensor_spans),
                "pointer_tag_enable": self.stage_l6_pointer_tag_enable,
                "allocator_kv_driver_tag_enable": (
                    self.stage_n_allocator_kv_driver_tag_enable
                ),
                "pointer_tag_granularity": self.stage_l6_pointer_tag_granularity,
                "data_offload_enable": self.stage_l6_data_offload_enable,
            }
        )
        if not pointer_map:
            self._stage_l6_pointer_map_loaded = False
            return
        if pointer_map:
            allocator_driver_tags = self.stage_n_allocator_kv_driver_tag_enable
            self.register_stage_l6_kv_pointer_map(
                pointer_map,
                source="kv_pointer_map_contiguous_block",
                register_driver_regions=(
                    self.stage_l6_data_offload_enable
                    and not allocator_driver_tags
                ),
            )
            self._stage_l6_pointer_ranges.update(range_map)
            driver_range_regions = 0
            skipped_driver_pointer_regions = 0
            for region_id, ranges in range_map.items():
                try:
                    _, group_id_text, block_id_text = region_id.split(":", 2)
                    group_id = int(group_id_text)
                    block_id = int(block_id_text)
                except ValueError:
                    continue
                first_ptr = ranges[0][0]
                total_bytes = bytes_map.get(region_id, sum(size for _, size in ranges))
                register_uvm_residency_region(
                    pool="kv",
                    region_id=region_id,
                    ptr=(
                        first_ptr
                        if (
                            self.stage_l6_data_offload_enable
                            and not allocator_driver_tags
                        )
                        else None
                    ),
                    bytes=total_bytes,
                    state="gpu",
                    evictable=True,
                    active=False,
                    metadata={
                        "stage": "stage_l6",
                        "source": "kv_pointer_map_logical_block",
                        "group_id": group_id,
                        "block_id": block_id,
                        "bytes_per_block": total_bytes,
                        "offload_capable": True,
                        "metadata_only": (
                            not self.stage_l6_data_offload_enable
                            or allocator_driver_tags
                        ),
                        "driver_range_children": self.stage_l6_pointer_tag_enable,
                        "allocator_kv_driver_tag": allocator_driver_tags,
                        "driver_tag_skipped": (
                            "allocator_kv_driver_tag_enable"
                            if allocator_driver_tags
                            else None
                        ),
                        "range_count": len(ranges),
                    },
                )
                if (
                    not self.stage_l6_pointer_tag_enable
                    or self.stage_l6_pointer_tag_granularity != "range"
                    or allocator_driver_tags
                ):
                    if (
                        allocator_driver_tags
                        and self.stage_l6_pointer_tag_enable
                        and self.stage_l6_pointer_tag_granularity == "range"
                    ):
                        skipped_driver_pointer_regions += len(ranges)
                    continue
                range_metadata = range_metadata_map.get(region_id) or []
                for range_index, (ptr, range_bytes) in enumerate(ranges):
                    if ptr <= 0 or range_bytes <= 0:
                        continue
                    range_meta = (
                        range_metadata[range_index]
                        if range_index < len(range_metadata)
                        else {}
                    )
                    register_uvm_residency_region(
                        pool="kv",
                        region_id=f"{region_id}:range:{range_index}",
                        ptr=int(ptr),
                        bytes=int(range_bytes),
                        state="gpu",
                        evictable=True,
                        active=False,
                        metadata={
                            "stage": "stage_l6",
                            "source": "kv_pointer_map_range",
                            "parent_region_id": region_id,
                            "group_id": group_id,
                            "block_id": block_id,
                            "range_index": range_index,
                            "tensor_name": range_meta.get("tensor_name"),
                            "bytes_per_block": int(range_bytes),
                            "logical_block_bytes": total_bytes,
                            "offload_capable": False,
                            "metadata_only": False,
                            "driver_range_child": True,
                        },
                    )
                    driver_range_regions += 1
            driver_tensor_regions = 0
            if (
                self.stage_l6_pointer_tag_enable
                and self.stage_l6_pointer_tag_granularity == "tensor"
                and not allocator_driver_tags
            ):
                for tensor_index, (tensor_name, span) in enumerate(
                    sorted(tensor_spans.items())
                ):
                    start = int(span.get("start") or 0)
                    end = int(span.get("end") or 0)
                    if start <= 0 or end <= start:
                        continue
                    register_uvm_residency_region(
                        pool="kv",
                        region_id=f"kv:tensor:{tensor_index}",
                        ptr=start,
                        bytes=end - start,
                        state="gpu",
                        evictable=True,
                        active=False,
                        metadata={
                            "stage": "stage_l6",
                            "source": "kv_pointer_map_tensor_span",
                            "tensor_name": tensor_name,
                            "tensor_index": tensor_index,
                            "range_count": int(span.get("range_count") or 0),
                            "bytes_sum": int(span.get("bytes_sum") or 0),
                            "offload_capable": False,
                            "metadata_only": False,
                            "driver_tensor_span": True,
                        },
                    )
                    driver_tensor_regions += 1
            elif (
                allocator_driver_tags
                and self.stage_l6_pointer_tag_enable
                and self.stage_l6_pointer_tag_granularity == "tensor"
            ):
                skipped_driver_pointer_regions += len(tensor_spans)
            if self.stage_l6_pointer_tag_enable:
                self._write(
                    {
                        "action": "stage_l6_kv_pointer_driver_tags_registered",
                        "logical_regions": len(range_map),
                        "granularity": self.stage_l6_pointer_tag_granularity,
                        "driver_tensor_regions": driver_tensor_regions,
                        "driver_range_regions": driver_range_regions,
                        "driver_pointer_tags_skipped": (
                            skipped_driver_pointer_regions
                        ),
                        "driver_pointer_tag_skip_reason": (
                            "allocator_kv_driver_tag_enable"
                            if allocator_driver_tags
                            else None
                        ),
                        "allocator_kv_driver_tag_enable": allocator_driver_tags,
                        "data_offload_enable": self.stage_l6_data_offload_enable,
                    }
                )

    def should_allow_allocation(
        self,
        *,
        request_id: str,
        num_blocks_to_allocate: int,
        block_pool: Any,
    ) -> bool:
        """Trace pressure and decide whether new KV block admission is allowed."""
        if not self.active or num_blocks_to_allocate <= 0:
            return True
        self.record_config()
        self._maybe_load_stage_l6_pointer_map(reason="allocation_pressure")
        pressure = self._pressure_snapshot(block_pool, num_blocks_to_allocate)
        over_budget = bool(pressure["over_budget"])
        record_uvm_pool_pressure(
            pool="kv",
            pressure_bytes=int(pressure.get("pressure_bytes") or 0),
            pressure_ratio=(
                (
                    float(pressure.get("projected_used_blocks") or 0)
                    / float(pressure.get("budget_blocks") or 1)
                )
                if int(pressure.get("budget_blocks") or 0) > 0
                else None
            ),
            action_queue_depth=num_blocks_to_allocate,
            metadata={
                "stage": "stage_j",
                "request_id": request_id,
                "over_budget": over_budget,
            },
        )
        if self.stage_l5_metadata_enable:
            request_uvm_gpu_residency(
                pool="kv",
                region_id=f"kv:allocation:{request_id}",
                bytes=max(num_blocks_to_allocate, 0)
                * max(self.bytes_per_block, 0),
                scope_key=f"kv_request:{request_id}",
                metadata={
                    "stage": "stage_l5",
                    "source": "kv_runtime_admission",
                    "request_id": request_id,
                    "num_blocks_to_allocate": num_blocks_to_allocate,
                    "metadata_only": True,
                    "data_offload_issued": False,
                    **pressure,
                },
            )
            self._summary["stage_l5_residency_request_records"] += 1
        self._write(
            {
                "action": "allocation_pressure",
                "request_id": request_id,
                "num_blocks_to_allocate": num_blocks_to_allocate,
                **pressure,
            }
        )
        self._summary["allocation_pressure_records"] += 1
        if over_budget:
            self._summary["over_budget_records"] += 1
            self._trace_candidates(
                request_id=request_id,
                block_pool=block_pool,
                reason="runtime_budget_pressure",
            )
            self._maybe_execute_stage_l6_safe_offload(
                request_id=request_id,
                block_pool=block_pool,
                pressure=pressure,
                reason="runtime_budget_pressure",
            )
            self._maybe_execute_prefix_eviction(
                request_id=request_id,
                block_pool=block_pool,
                pressure=pressure,
                reason="runtime_budget_pressure",
            )
            self._write(
                {
                    "action": "would_deny_allocation",
                    "request_id": request_id,
                    "mode": self.mode,
                    "policy": self.policy,
                    "num_blocks_to_allocate": num_blocks_to_allocate,
                    **pressure,
                }
            )
            self._summary["would_deny_records"] += 1
            allowed = self.mode != "enforce"
            if not allowed:
                self._summary["denied_records"] += 1
                self._write(
                    {
                        "action": "deny_allocation",
                        "request_id": request_id,
                        "mode": self.mode,
                        "policy": self.policy,
                        "num_blocks_to_allocate": num_blocks_to_allocate,
                        **pressure,
                    }
                )
            self._record_summary()
            return allowed
        self._record_summary()
        return True

    def record_reuse_failure(
        self,
        *,
        request_id: str,
        num_blocks_to_allocate: int,
        block_pool: Any,
    ) -> None:
        if not self.active:
            return
        self.record_config()
        self._write(
            {
                "action": "allocation_no_free_blocks",
                "request_id": request_id,
                "num_blocks_to_allocate": num_blocks_to_allocate,
                **self._pressure_snapshot(block_pool, num_blocks_to_allocate),
            }
        )
        self._summary["reuse_failure_records"] += 1
        self._record_summary()

    def record_prefix_cache_eviction(self, *, block: Any, block_hash: Any) -> None:
        """Record a real vLLM-safe prefix-cache eviction/reuse event."""
        if not self.active:
            return
        self.record_config()
        safe_ref_cnt_zero = getattr(block, "ref_cnt", None) == 0
        self._write(
            {
                "action": "evict_prefix_cache_block",
                "block_id": getattr(block, "block_id", None),
                "ref_cnt": getattr(block, "ref_cnt", None),
                "is_null": getattr(block, "is_null", None),
                "block_hash": _block_hash_to_text(block_hash),
                "safe_ref_cnt_zero": safe_ref_cnt_zero,
            }
        )
        self._summary["prefix_cache_eviction_records"] += 1
        if not safe_ref_cnt_zero:
            self._summary["unsafe_prefix_cache_eviction_records"] += 1
        self._record_summary()

    def _pressure_snapshot(
        self, block_pool: Any, num_blocks_to_allocate: int
    ) -> dict[str, Any]:
        total_blocks = max(getattr(block_pool, "num_gpu_blocks", 0) - 1, 0)
        free_blocks = max(block_pool.get_num_free_blocks(), 0)
        used_blocks = max(total_blocks - free_blocks, 0)
        projected_used_blocks = used_blocks + max(num_blocks_to_allocate, 0)
        budget_blocks = self.budget_blocks
        pressure_blocks = (
            max(projected_used_blocks - budget_blocks, 0)
            if budget_blocks > 0
            else 0
        )
        return {
            "budget_blocks": budget_blocks,
            "budget_bytes": self.budget_bytes,
            "bytes_per_block": self.bytes_per_block,
            "total_blocks": total_blocks,
            "free_blocks": free_blocks,
            "used_blocks": used_blocks,
            "projected_used_blocks": projected_used_blocks,
            "pressure_blocks": pressure_blocks,
            "pressure_bytes": pressure_blocks * self.bytes_per_block,
            "over_budget": budget_blocks > 0 and pressure_blocks > 0,
        }

    def _trace_candidates(
        self,
        *,
        request_id: str,
        block_pool: Any,
        reason: str,
    ) -> None:
        if self.candidate_limit <= 0:
            return
        try:
            free_blocks = block_pool.free_block_queue.get_all_free_blocks()
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._write(
                {
                    "action": "candidate_snapshot_failed",
                    "request_id": request_id,
                    "reason": reason,
                    "error": repr(exc),
                }
            )
            return
        for index, block in enumerate(free_blocks[: self.candidate_limit]):
            block_hash = getattr(block, "block_hash", None)
            block_id = int(getattr(block, "block_id", -1))
            safe = (
                block_id >= 0
                and not bool(getattr(block, "is_null", False))
                and getattr(block, "ref_cnt", None) == 0
            )
            if self.stage_l5_metadata_enable and block_id >= 0:
                for group_id in range(self.num_kv_cache_groups):
                    record_uvm_residency_kv_metadata_candidate(
                        region_id=f"kv:{group_id}:{block_id}",
                        block_id=block_id,
                        group_id=group_id,
                        bytes_per_block=max(self.bytes_per_group_block, 1),
                        safe=safe,
                        reason=reason,
                        metadata={
                            "stage": "stage_l5",
                            "source": "free_queue_candidate_snapshot",
                            "candidate_index": index,
                            "ref_cnt": getattr(block, "ref_cnt", None),
                            "is_null": getattr(block, "is_null", None),
                            "is_prefix_cache": block_hash is not None,
                            "in_free_queue": True,
                            "block_hash": _block_hash_to_text(block_hash),
                            "offload_capable": self._kv_region_offload_capable(
                                group_id, block_id
                            ),
                            "metadata_only": not self.stage_l6_data_offload_enable,
                        },
                    )
                    if safe:
                        self._summary["stage_l5_safe_candidate_records"] += 1
                    else:
                        self._summary["stage_l5_unsafe_candidate_records"] += 1
            self._write(
                {
                    "action": "would_evict_candidate"
                    if block_hash is not None
                    else "would_reuse_free_block",
                    "request_id": request_id,
                    "reason": reason,
                    "candidate_index": index,
                    "block_id": getattr(block, "block_id", None),
                    "ref_cnt": getattr(block, "ref_cnt", None),
                    "is_null": getattr(block, "is_null", None),
                    "has_block_hash": block_hash is not None,
                    "block_hash": _block_hash_to_text(block_hash),
                    "safe_ref_cnt_zero": safe,
                }
            )
            self._summary["candidate_records"] += 1

    def on_blocks_allocated(
        self,
        *,
        blocks: list[Any],
        reason: str,
    ) -> None:
        """Prefetch CPU-resident KV blocks before they are reused."""
        if not self.active or not self.stage_l6_prefetch_back_enable:
            return
        for block in blocks:
            block_id = int(getattr(block, "block_id", -1))
            if block_id < 0 or bool(getattr(block, "is_null", False)):
                continue
            for group_id in range(self.num_kv_cache_groups):
                self._maybe_prefetch_back_region(
                    region_id=f"kv:{group_id}:{block_id}",
                    block_id=block_id,
                    group_id=group_id,
                    reason=reason,
                    ref_cnt=getattr(block, "ref_cnt", None),
                )

    def protect_active_blocks(
        self,
        *,
        request_id: str,
        blocks_by_group: tuple[list[Any], ...],
        reason: str,
    ) -> None:
        """Mark active request KV blocks as protected GPU-resident regions."""
        if not self.active or not self.active_protect_enable:
            return
        self.record_config()
        if self.active_protect_mode != "trace_only":
            self._maybe_load_stage_l6_pointer_map()

        runtime = get_uvm_policy_runtime()
        if not runtime.active:
            self._write(
                {
                    "action": "active_kv_protect_skipped",
                    "request_id": request_id,
                    "reason": "policy_runtime_disabled",
                    "protect_mode": self.active_protect_mode,
                }
            )
            return

        active_region_ids: set[str] = set()
        candidates: list[tuple[int, int, Any, str]] = []
        for group_id, blocks in enumerate(blocks_by_group):
            for block in blocks:
                block_id = int(getattr(block, "block_id", -1))
                if block_id < 0 or bool(getattr(block, "is_null", False)):
                    continue
                region_id = f"kv:{group_id}:{block_id}"
                active_region_ids.add(region_id)
                candidates.append((group_id, block_id, block, region_id))

        if self.active_protect_max_blocks > 0:
            candidates = candidates[: self.active_protect_max_blocks]
        if not candidates:
            self._write(
                {
                    "action": "active_kv_protect_noop",
                    "request_id": request_id,
                    "reason": reason,
                    "protect_mode": self.active_protect_mode,
                }
            )
            return

        attempted = 0
        issued = 0
        fallback = 0
        issued_bytes = 0
        for group_id, block_id, block, region_id in candidates:
            snapshot = get_uvm_residency_region_snapshot(region_id)
            ranges = self._stage_l6_pointer_ranges.get(region_id)
            size_bytes = int(
                (snapshot or {}).get("bytes")
                or sum(size for _, size in (ranges or ()))
                or self.bytes_per_group_block
            )
            metadata = dict((snapshot or {}).get("metadata") or {})
            protect_metadata = {
                **metadata,
                "stage": "stage_m",
                "source": "kv_active_protection",
                "request_id": request_id,
                "reason": reason,
                "block_id": block_id,
                "group_id": group_id,
                "ref_cnt": getattr(block, "ref_cnt", None),
                "is_prefix_cache": getattr(block, "block_hash", None) is not None,
                "block_hash": _block_hash_to_text(getattr(block, "block_hash", None)),
                "metadata_only": not bool(ranges),
            }
            result = runtime.protect_kv_region(
                region_id=region_id,
                block_id=block_id,
                group_id=group_id,
                bytes=size_bytes,
                ranges=ranges,
                mode=self.active_protect_mode,
                device=self.active_protect_device,
                scope_key=f"kv_active:{request_id}",
                active_region_ids=active_region_ids,
                metadata=protect_metadata,
            )
            attempted += 1
            if result.issued:
                issued += 1
                issued_bytes += result.bytes
            else:
                fallback += 1
            self._write(
                {
                    "action": (
                        "active_kv_protect_issued"
                        if result.issued
                        else "active_kv_protect_fallback"
                    ),
                    "request_id": request_id,
                    "region_id": region_id,
                    "block_id": block_id,
                    "group_id": group_id,
                    "bytes": result.bytes,
                    "range_count": result.range_count,
                    "protect_mode": result.mode,
                    "result_reason": result.reason,
                    "runtime_enabled": result.enabled,
                    "allowed": result.allowed,
                }
            )

        self._summary["active_protect_attempt_records"] += attempted
        self._summary["active_protect_issued_records"] += issued
        self._summary["active_protect_fallback_records"] += fallback
        self._summary["active_protect_blocks"] += issued
        self._summary["active_protect_bytes"] += issued_bytes
        self._record_summary()

    def on_blocks_freed(self, *, blocks: list[Any], reason: str) -> None:
        if not self.active or not self.stage_l5_metadata_enable:
            return
        for block in blocks:
            block_id = int(getattr(block, "block_id", -1))
            if block_id < 0 or bool(getattr(block, "is_null", False)):
                continue
            safe = getattr(block, "ref_cnt", None) == 0
            for group_id in range(self.num_kv_cache_groups):
                record_uvm_residency_kv_metadata_candidate(
                    region_id=f"kv:{group_id}:{block_id}",
                    block_id=block_id,
                    group_id=group_id,
                    bytes_per_block=max(self.bytes_per_group_block, 1),
                    safe=safe,
                    reason=reason,
                    metadata={
                        "stage": "stage_l6" if self.stage_l6_data_offload_enable else "stage_l5",
                        "source": "block_free_lifecycle",
                        "ref_cnt": getattr(block, "ref_cnt", None),
                        "is_null": getattr(block, "is_null", None),
                        "is_prefix_cache": getattr(block, "block_hash", None) is not None,
                        "in_free_queue": safe,
                        "block_hash": _block_hash_to_text(getattr(block, "block_hash", None)),
                        "offload_capable": self._kv_region_offload_capable(group_id, block_id),
                        "metadata_only": not self.stage_l6_data_offload_enable,
                    },
                )

    def _kv_region_offload_capable(self, group_id: int, block_id: int) -> bool:
        snapshot = get_uvm_residency_region_snapshot(f"kv:{group_id}:{block_id}")
        if not snapshot:
            return False
        metadata = snapshot.get("metadata") or {}
        return bool(snapshot.get("ptr")) and bool(metadata.get("offload_capable"))

    def _maybe_execute_stage_l6_safe_offload(
        self,
        *,
        request_id: str,
        block_pool: Any,
        pressure: dict[str, Any],
        reason: str,
    ) -> None:
        if not self.stage_l6_data_offload_enable:
            return
        self._maybe_load_stage_l6_pointer_map()
        try:
            free_blocks = block_pool.free_block_queue.get_all_free_blocks()
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._write(
                {
                    "action": "stage_l6_kv_safe_offload_snapshot_failed",
                    "request_id": request_id,
                    "reason": reason,
                    "error": repr(exc),
                }
            )
            return
        pressure_blocks = int(pressure.get("pressure_blocks") or 0)
        max_blocks = self.stage_l6_max_offload_blocks or pressure_blocks
        if max_blocks <= 0:
            max_blocks = self.candidate_limit
        attempted_blocks = 0
        issued_blocks = 0
        for block in free_blocks:
            if attempted_blocks >= max_blocks:
                break
            block_id = int(getattr(block, "block_id", -1))
            safe = (
                block_id >= 0
                and not bool(getattr(block, "is_null", False))
                and getattr(block, "ref_cnt", None) == 0
            )
            for group_id in range(self.num_kv_cache_groups):
                if attempted_blocks >= max_blocks:
                    break
                attempted_blocks += 1
                did_issue = self._maybe_offload_region(
                    region_id=f"kv:{group_id}:{block_id}",
                    block_id=block_id,
                    group_id=group_id,
                    safe=safe,
                    request_id=request_id,
                    reason=reason,
                    block=block,
                    pressure=pressure,
                )
                if did_issue:
                    issued_blocks += 1

    def _maybe_offload_region(
        self,
        *,
        region_id: str,
        block_id: int,
        group_id: int,
        safe: bool,
        request_id: str,
        reason: str,
        block: Any,
        pressure: dict[str, Any],
    ) -> bool:
        snapshot = get_uvm_residency_region_snapshot(region_id)
        metadata = dict((snapshot or {}).get("metadata") or {})
        ptr = int((snapshot or {}).get("ptr") or 0)
        size_bytes = int((snapshot or {}).get("bytes") or self.bytes_per_group_block)
        state = str((snapshot or {}).get("state") or "unknown")
        ranges = self._stage_l6_pointer_ranges.get(region_id)
        offload_capable = bool(metadata.get("offload_capable")) and bool(ranges)
        attempt_metadata = {
            "stage": "stage_l6",
            "source": "kv_safe_block_offload",
            "request_id": request_id,
            "reason": reason,
            "block_id": block_id,
            "group_id": group_id,
            "ref_cnt": getattr(block, "ref_cnt", None),
            "is_null": getattr(block, "is_null", None),
            "is_prefix_cache": getattr(block, "block_hash", None) is not None,
            "in_free_queue": True,
            "block_hash": _block_hash_to_text(getattr(block, "block_hash", None)),
            "safe": safe,
            "offload_capable": offload_capable,
            "ptr_present": ptr > 0,
            "range_count": len(ranges or ()),
            **pressure,
        }
        self._summary["stage_l6_data_offload_attempt_records"] += 1
        record_uvm_residency_action(
            action="residency_kv_data_offload_attempt",
            pool="kv",
            region_id=region_id,
            bytes=size_bytes,
            metadata=attempt_metadata,
        )
        if not safe:
            self._summary["stage_l6_unsafe_offload_records"] += 1
            self._summary["stage_l6_data_offload_fallback_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_data_offload_fallback",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                fallback_reason="unsafe_block",
                metadata=attempt_metadata,
            )
            return False
        if state == "cpu":
            return True
        if not offload_capable:
            self._summary["stage_l6_data_offload_fallback_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_data_offload_fallback",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                fallback_reason="missing_ptr_or_not_offload_capable",
                metadata=attempt_metadata,
            )
            return False
        coordinator_decision = request_uvm_pool_action(
            pool="kv",
            action="kv_safe_block_prefetch_cpu",
            requested_bytes=size_bytes,
            scope_key=f"kv_request:{request_id}",
            metadata={
                "stage": "stage_l7",
                "source": "kv_safe_block_offload",
                "request_id": request_id,
                "region_id": region_id,
                "block_id": block_id,
                "group_id": group_id,
                "range_count": len(ranges or ()),
                "mode": self.stage_l6_data_offload_mode,
            },
        )
        if not coordinator_decision.allowed:
            self._summary["stage_l6_data_offload_fallback_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_data_offload_fallback",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                fallback_reason="stage_k_coordinator_reject",
                coordinator_mode=coordinator_decision.mode,
                coordinator_reason=coordinator_decision.reason,
                metadata=attempt_metadata,
            )
            self._write(
                {
                    "action": "stage_l7_kv_offload_coordinator_reject",
                    "request_id": request_id,
                    "region_id": region_id,
                    "requested_bytes": size_bytes,
                    "coordinator_mode": coordinator_decision.mode,
                    "coordinator_reason": coordinator_decision.reason,
                    "scope_key": coordinator_decision.scope_key,
                }
            )
            return False
        mark_uvm_residency_state(
            region_id=region_id,
            state="prefetching_cpu",
            metadata=attempt_metadata,
        )
        issued = self.stage_l6_data_offload_mode == "trace_only"
        if self.stage_l6_data_offload_mode == "prefetch_cpu":
            issued = all(
                prefetch_range_to_cpu(range_ptr, range_bytes)
                for range_ptr, range_bytes in (ranges or ())
            )
        if issued:
            mark_uvm_residency_state(
                region_id=region_id,
                state="cpu",
                metadata=attempt_metadata,
            )
            self._summary["stage_l6_data_offload_issued_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_data_offload_issued",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                mode=self.stage_l6_data_offload_mode,
                ptr=f"0x{ptr:x}",
                range_count=len(ranges or ()),
                coordinator_mode=coordinator_decision.mode,
                coordinator_reason=coordinator_decision.reason,
                metadata=attempt_metadata,
            )
            return True
        mark_uvm_residency_state(
            region_id=region_id,
            state="gpu",
            metadata=attempt_metadata,
        )
        self._summary["stage_l6_data_offload_fallback_records"] += 1
        record_uvm_residency_action(
            action="residency_kv_data_offload_fallback",
            pool="kv",
            region_id=region_id,
            bytes=size_bytes,
            fallback_reason="prefetch_cpu_not_issued",
            metadata=attempt_metadata,
        )
        return False

    def _maybe_prefetch_back_region(
        self,
        *,
        region_id: str,
        block_id: int,
        group_id: int,
        reason: str,
        ref_cnt: Any,
    ) -> None:
        snapshot = get_uvm_residency_region_snapshot(region_id)
        if not snapshot:
            return
        state = str(snapshot.get("state") or "unknown")
        if state not in ("cpu", "prefetching_cpu"):
            return
        ptr = int(snapshot.get("ptr") or 0)
        size_bytes = int(snapshot.get("bytes") or self.bytes_per_group_block)
        metadata = dict(snapshot.get("metadata") or {})
        ranges = self._stage_l6_pointer_ranges.get(region_id)
        prefetch_metadata = {
            "stage": "stage_l6",
            "source": "kv_prefetch_back_before_reuse",
            "reason": reason,
            "block_id": block_id,
            "group_id": group_id,
            "ref_cnt": ref_cnt,
            "previous_state": state,
            "offload_capable": bool(metadata.get("offload_capable")),
            "ptr_present": ptr > 0,
            "range_count": len(ranges or ()),
        }
        if not ranges or not bool(metadata.get("offload_capable")):
            self._summary["stage_l6_prefetch_back_fallback_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_prefetch_back_fallback",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                fallback_reason="missing_ptr_or_not_offload_capable",
                metadata=prefetch_metadata,
            )
            return
        coordinator_decision = request_uvm_pool_action(
            pool="kv",
            action="kv_block_prefetch_gpu",
            requested_bytes=size_bytes,
            scope_key=f"kv_prefetch_back:{region_id}",
            metadata={
                "stage": "stage_l7",
                "source": "kv_prefetch_back_before_reuse",
                "region_id": region_id,
                "block_id": block_id,
                "group_id": group_id,
                "range_count": len(ranges),
                "mode": self.stage_l6_prefetch_back_mode,
            },
        )
        if not coordinator_decision.allowed:
            self._summary["stage_l6_prefetch_back_fallback_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_prefetch_back_fallback",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                fallback_reason="stage_k_coordinator_reject",
                coordinator_mode=coordinator_decision.mode,
                coordinator_reason=coordinator_decision.reason,
                metadata=prefetch_metadata,
            )
            self._write(
                {
                    "action": "stage_l7_kv_prefetch_back_coordinator_reject",
                    "region_id": region_id,
                    "requested_bytes": size_bytes,
                    "coordinator_mode": coordinator_decision.mode,
                    "coordinator_reason": coordinator_decision.reason,
                    "scope_key": coordinator_decision.scope_key,
                }
            )
            return
        mark_uvm_residency_state(
            region_id=region_id,
            state="prefetching_gpu",
            metadata=prefetch_metadata,
        )
        issued = self.stage_l6_prefetch_back_mode == "trace_only"
        if self.stage_l6_prefetch_back_mode == "prefetch":
            issued = all(
                prefetch_range_to_device(range_ptr, range_bytes, self.stage_l6_device)
                for range_ptr, range_bytes in ranges
            )
        if issued:
            mark_uvm_residency_state(
                region_id=region_id,
                state="gpu",
                metadata=prefetch_metadata,
            )
            self._summary["stage_l6_prefetch_back_issued_records"] += 1
            record_uvm_residency_action(
                action="residency_kv_prefetch_back_issued",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                mode=self.stage_l6_prefetch_back_mode,
                ptr=f"0x{ptr:x}",
                range_count=len(ranges),
                coordinator_mode=coordinator_decision.mode,
                coordinator_reason=coordinator_decision.reason,
                metadata=prefetch_metadata,
            )
            return
        mark_uvm_residency_state(
            region_id=region_id,
            state="cpu",
            metadata=prefetch_metadata,
        )
        self._summary["stage_l6_prefetch_back_fallback_records"] += 1
        record_uvm_residency_action(
            action="residency_kv_prefetch_back_fallback",
            pool="kv",
            region_id=region_id,
            bytes=size_bytes,
            fallback_reason="prefetch_gpu_not_issued",
            metadata=prefetch_metadata,
        )

    def _maybe_execute_prefix_eviction(
        self,
        *,
        request_id: str,
        block_pool: Any,
        pressure: dict[str, Any],
        reason: str,
    ) -> None:
        if not self.prefix_evict_enable:
            return
        max_blocks = self.prefix_evict_max_blocks
        if max_blocks <= 0:
            max_blocks = int(pressure.get("pressure_blocks") or 0)
        if max_blocks <= 0:
            return

        self._summary["prefix_evict_attempt_records"] += 1
        requested_bytes = max_blocks * int(pressure.get("bytes_per_block") or 0)
        coordinator_decision = request_uvm_pool_action(
            pool="kv",
            action="prefix_cache_evict",
            requested_bytes=requested_bytes,
            scope_key=f"kv_request:{request_id}",
            metadata={
                "stage": "stage_j",
                "request_id": request_id,
                "reason": reason,
                "max_blocks": max_blocks,
                "policy": self.policy,
            },
        )
        if not coordinator_decision.allowed:
            self._summary["prefix_evict_coordinator_reject_records"] += 1
            self._write(
                {
                    "action": "prefix_evict_coordinator_reject",
                    "request_id": request_id,
                    "reason": reason,
                    "max_blocks": max_blocks,
                    "requested_bytes": requested_bytes,
                    "coordinator_mode": coordinator_decision.mode,
                    "coordinator_reason": coordinator_decision.reason,
                    **pressure,
                }
            )
            return
        self._write(
            {
                "action": "prefix_evict_attempt",
                "request_id": request_id,
                "reason": reason,
                "max_blocks": max_blocks,
                **pressure,
            }
        )
        try:
            evicted_blocks = int(block_pool.evict_cached_free_blocks(max_blocks))
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._summary["prefix_evict_failed_records"] += 1
            self._write(
                {
                    "action": "prefix_evict_failed",
                    "request_id": request_id,
                    "reason": reason,
                    "max_blocks": max_blocks,
                    "error": repr(exc),
                }
            )
            return

        if evicted_blocks > 0:
            self._summary["prefix_evict_success_blocks"] += evicted_blocks
            action = "prefix_evict_success"
        else:
            self._summary["prefix_evict_noop_records"] += 1
            action = "prefix_evict_noop"
        self._write(
            {
                "action": action,
                "request_id": request_id,
                "reason": reason,
                "requested_blocks": max_blocks,
                "evicted_blocks": evicted_blocks,
                **pressure,
            }
        )

    def _record_summary(self) -> None:
        if not self.active:
            return
        self._write(
            {
                "action": "runtime_summary",
                "mode": self.mode,
                "policy": self.policy,
                "budget_blocks": self.budget_blocks,
                "budget_bytes": self.budget_bytes,
                "prefix_evict_enable": self.prefix_evict_enable,
                "prefix_evict_max_blocks": self.prefix_evict_max_blocks,
                **self._summary,
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        if not self.trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("stage", "stage_j")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
            with open(self.trace_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
