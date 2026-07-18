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
from vllm.device_allocator.uvm import (
    batch_offload_ranges_to_cpu_with_executor,
    batch_prefetch_ranges_to_device_with_executor,
)
from vllm.device_allocator.uvm_policy_runtime import get_uvm_policy_runtime
from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_residency_region_snapshot,
    get_uvm_pool_residency_manager,
    mark_uvm_residency_state,
    record_uvm_residency_kv_metadata_candidate,
    record_uvm_residency_action,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
)
from vllm.device_allocator.uvm_unified_residency_controller import (
    get_uvm_unified_residency_controller,
    record_unified_kv_target,
    record_unified_residency_snapshot,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

_KV_PHYSICAL_RANGE_LOCK = threading.Lock()
_KV_PHYSICAL_RANGE_RECORDS: dict[tuple[int, str, int, int], dict[str, Any]] = {}


def _coalesce_block_ids(block_ids: list[int]) -> list[tuple[int, int]]:
    ids = sorted({int(block_id) for block_id in block_ids if int(block_id) > 0})
    if not ids:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = ids[0]
    for block_id in ids[1:]:
        if block_id == prev + 1:
            prev = block_id
            continue
        ranges.append((start, prev))
        start = prev = block_id
    ranges.append((start, prev))
    return ranges


def register_uvm_kv_cache_physical_ranges(
    records: list[dict[str, Any]],
    *,
    clear_existing: bool = False,
) -> dict[str, Any]:
    """Register physical KV tensor spans used to derive block VA ranges."""
    valid_records: list[dict[str, Any]] = []
    skipped = 0
    for record in records:
        try:
            group_id = int(record.get("group_id", -1))
            tensor_index = int(record.get("tensor_index", 0))
            ptr = int(record.get("ptr") or 0)
            size_bytes = int(record.get("size_bytes") or 0)
            page_size = int(record.get("page_size") or 0)
            total_blocks = int(record.get("total_blocks") or 0)
            layer_name = str(record.get("layer_name") or "")
        except (TypeError, ValueError):
            skipped += 1
            continue
        if (
            group_id < 0
            or not layer_name
            or ptr <= 0
            or size_bytes <= 0
            or page_size <= 0
            or total_blocks <= 0
        ):
            skipped += 1
            continue
        allocation_blocks = size_bytes // page_size
        max_block_id = min(total_blocks, allocation_blocks - 1)
        if max_block_id <= 0:
            skipped += 1
            continue
        valid = {
            **record,
            "group_id": group_id,
            "layer_name": layer_name,
            "tensor_index": tensor_index,
            "ptr": ptr,
            "size_bytes": size_bytes,
            "page_size": page_size,
            "total_blocks": total_blocks,
            "allocation_blocks": allocation_blocks,
            "max_block_id": max_block_id,
        }
        valid_records.append(valid)

    with _KV_PHYSICAL_RANGE_LOCK:
        if clear_existing:
            _KV_PHYSICAL_RANGE_RECORDS.clear()
        for record in valid_records:
            key = (
                int(record["group_id"]),
                str(record["layer_name"]),
                int(record["tensor_index"]),
                int(record["ptr"]),
            )
            _KV_PHYSICAL_RANGE_RECORDS[key] = record

    manager = get_uvm_pool_residency_manager()
    if manager.active:
        for record in valid_records:
            region_id = (
                f"kv_phys:{record['group_id']}:{record['layer_name']}:"
                f"{record['tensor_index']}"
            )
            register_uvm_residency_region(
                pool="kv",
                region_id=region_id,
                ptr=int(record["ptr"]),
                bytes=int(record["size_bytes"]),
                state="gpu",
                evictable=True,
                active=False,
                metadata={
                    "stage": "stage_l5_physical",
                    "source": "kv_cache_physical_range_map",
                    "group_id": int(record["group_id"]),
                    "layer_name": str(record["layer_name"]),
                    "tensor_index": int(record["tensor_index"]),
                    "page_size": int(record["page_size"]),
                    "total_blocks": int(record["total_blocks"]),
                    "max_block_id": int(record["max_block_id"]),
                    "block_addressable": True,
                    "metadata_only": False,
                },
            )

    return {
        "registered": len(valid_records),
        "skipped": skipped,
        "clear_existing": bool(clear_existing),
    }


def collect_uvm_kv_cache_physical_ranges_for_blocks(
    block_ids: list[int],
) -> dict[str, Any]:
    """Collect deduplicated physical ranges covering the given KV block ids."""
    intervals = _coalesce_block_ids(block_ids)
    if not intervals:
        return {
            "ok": True,
            "reason": "no_blocks",
            "ranges": [],
            "range_records": [],
            "registered_spans": 0,
        }

    with _KV_PHYSICAL_RANGE_LOCK:
        records = list(_KV_PHYSICAL_RANGE_RECORDS.values())

    ranges: list[tuple[int, int]] = []
    range_records: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for record in records:
        ptr = int(record["ptr"])
        page_size = int(record["page_size"])
        max_block_id = int(record["max_block_id"])
        for first_block, last_block in intervals:
            first = max(int(first_block), 1)
            last = min(int(last_block), max_block_id)
            if first > last:
                continue
            range_ptr = ptr + first * page_size
            range_bytes = (last - first + 1) * page_size
            range_key = (range_ptr, range_bytes)
            if range_key in seen_ranges:
                continue
            seen_ranges.add(range_key)
            ranges.append(range_key)
            range_records.append(
                {
                    "group_id": int(record["group_id"]),
                    "layer_name": str(record["layer_name"]),
                    "tensor_index": int(record["tensor_index"]),
                    "ptr": f"0x{range_ptr:x}",
                    "ptr_int": range_ptr,
                    "bytes": range_bytes,
                    "page_size": page_size,
                    "first_block": first,
                    "last_block": last,
                }
            )

    return {
        "ok": True,
        "reason": "ranges_collected" if ranges else "no_registered_ranges",
        "requested_block_count": len({int(block_id) for block_id in block_ids}),
        "block_intervals": [
            {"first_block": first, "last_block": last}
            for first, last in intervals
        ],
        "registered_spans": len(records),
        "range_count": len(ranges),
        "range_bytes": sum(size for _, size in ranges),
        "ranges": ranges,
        "range_records": range_records,
    }


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


def _block_hash_to_text(block_hash: Any) -> str | None:
    if block_hash is None:
        return None
    if isinstance(block_hash, bytes):
        return block_hash.hex()
    return str(block_hash)


class UvmKvRuntimePolicy:
    """Trace-only or soft-admission Stage J policy for KV block pressure."""

    def __init__(self, kv_cache_config: KVCacheConfig) -> None:
        post_warmup_target = (
            os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
        ).strip()
        placement_stage = (
            os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                "post_warmup",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        automatic_post_warmup_pool = bool(post_warmup_target) and placement_stage in (
            "post_warmup",
            "after_warmup",
            "deferred",
        )
        self.enabled = _read_bool_env(
            "VLLM_UVM_KV_RUNTIME_ENABLE",
            automatic_post_warmup_pool,
        )
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
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l5",
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
            self.active_protect_device = int(
                os.environ.get("VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE",
                               "0") or 0
            )
        except ValueError:
            logger.warning(
                "Ignoring invalid VLLM_UVM_KV_ACTIVE_PROTECT_DEVICE; using 0."
            )
            self.active_protect_device = 0
        self.prefer_gpu_resident_free_blocks = _read_bool_env(
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS",
            False,
        )
        self.prefer_gpu_resident_allow_cached = _read_bool_env(
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_ALLOW_CACHED",
            False,
        )
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
        self.dynamic_pool_enable = (
            self.enabled
            and _read_bool_env(
                "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE",
                automatic_post_warmup_pool,
            )
        )
        self.dynamic_pool_control_file = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE", ""
        )
        self.dynamic_pool_initial_blocks = _read_int_env(
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS", 0
        )
        self.dynamic_pool_target_blocks = self._clamp_dynamic_pool_target(
            self.dynamic_pool_initial_blocks
            if self.dynamic_pool_initial_blocks > 0
            else self.total_blocks
        )
        self.dynamic_pool_physical_migration_enable = (
            self.dynamic_pool_enable
            and _read_bool_env(
                "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE",
                self.dynamic_pool_enable,
            )
        )
        self.dynamic_pool_physical_migration_executor = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR",
            "auto",
        ).strip().lower()
        if self.dynamic_pool_physical_migration_executor not in (
            "auto",
            "cpp_batch",
            "driver_batch",
        ):
            logger.warning(
                "Ignoring invalid "
                "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR=%r; "
                "using auto.",
                self.dynamic_pool_physical_migration_executor,
            )
            self.dynamic_pool_physical_migration_executor = "auto"
        self.dynamic_pool_physical_migration_sync = _read_bool_env(
            "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_SYNC",
            True,
        )
        try:
            self.dynamic_pool_physical_migration_device = int(
                os.environ.get(
                    "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_DEVICE",
                    "0",
                )
                or 0
            )
        except ValueError:
            logger.warning(
                "Ignoring invalid "
                "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_DEVICE; using 0."
            )
            self.dynamic_pool_physical_migration_device = 0
        # The residency ledger starts with all registered KV regions on GPU.
        # Applying the target later lets the initial shrink be traced as a
        # safe, explicit pool-size transition.
        self._dynamic_pool_applied_target_blocks = self.total_blocks
        self._dynamic_pool_control_mtime_ns: int | None = None
        self._dynamic_pool_generation = 0
        self._dynamic_pool_saved_lifecycle: dict[int, str] = {}
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
            "active_protect_attempt_records": 0,
            "active_protect_issued_records": 0,
            "active_protect_fallback_records": 0,
            "active_protect_blocks": 0,
            "active_protect_bytes": 0,
            "dynamic_pool_resize_records": 0,
            "dynamic_pool_offload_blocks": 0,
            "dynamic_pool_prefetch_blocks": 0,
            "dynamic_pool_deferred_active_blocks": 0,
            "dynamic_pool_physical_migration_records": 0,
            "dynamic_pool_physical_migration_ranges": 0,
            "dynamic_pool_physical_migration_bytes": 0,
            "dynamic_pool_physical_migration_successes": 0,
        }
        self._unified_config_recorded = False

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.trace_file)

    def record_config(self) -> None:
        if not self._unified_config_recorded:
            self._unified_config_recorded = True
            record_unified_residency_snapshot(
                reason="kv_runtime_config",
                kv_pressure={
                    "runtime_enabled": self.enabled,
                    "runtime_mode": self.mode,
                    "budget_bytes": self.budget_bytes,
                    "budget_blocks": self.budget_blocks,
                    "bytes_per_block": self.bytes_per_block,
                    "total_bytes": self.total_bytes,
                    "total_blocks": self.total_blocks,
                    "dynamic_pool_enable": self.dynamic_pool_enable,
                    "dynamic_pool_target_blocks": self.dynamic_pool_target_blocks,
                    "dynamic_pool_physical_migration_enable": (
                        self.dynamic_pool_physical_migration_enable
                    ),
                    "dynamic_pool_physical_migration_executor": (
                        self.dynamic_pool_physical_migration_executor
                    ),
                    "dynamic_pool_physical_migration_sync": (
                        self.dynamic_pool_physical_migration_sync
                    ),
                    "dynamic_pool_physical_migration_device": (
                        self.dynamic_pool_physical_migration_device
                    ),
                    "num_kv_cache_groups": self.num_kv_cache_groups,
                },
            )
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
                "active_protect_enable": self.active_protect_enable,
                "active_protect_mode": self.active_protect_mode,
                "active_protect_device": self.active_protect_device,
                "active_protect_max_blocks": self.active_protect_max_blocks,
                "prefer_gpu_resident_free_blocks": (
                    self.prefer_gpu_resident_free_blocks
                ),
                "prefer_gpu_resident_allow_cached": (
                    self.prefer_gpu_resident_allow_cached
                ),
                "dynamic_pool_enable": self.dynamic_pool_enable,
                "dynamic_pool_control_file": self.dynamic_pool_control_file,
                "dynamic_pool_initial_blocks": self.dynamic_pool_initial_blocks,
                "dynamic_pool_target_blocks": self.dynamic_pool_target_blocks,
                "dynamic_pool_physical_migration_enable": (
                    self.dynamic_pool_physical_migration_enable
                ),
                "dynamic_pool_physical_migration_executor": (
                    self.dynamic_pool_physical_migration_executor
                ),
                "dynamic_pool_physical_migration_sync": (
                    self.dynamic_pool_physical_migration_sync
                ),
                "dynamic_pool_physical_migration_device": (
                    self.dynamic_pool_physical_migration_device
                ),
                "num_kv_cache_groups": self.num_kv_cache_groups,
            }
        )

    def dynamic_pool_active(self) -> bool:
        return self.dynamic_pool_enable

    def refresh_dynamic_pool_target(self, block_pool: Any, reason: str) -> None:
        self._refresh_dynamic_pool_target(block_pool=block_pool, reason=reason)

    def get_effective_num_gpu_blocks(self, block_pool: Any) -> int:
        if not self.dynamic_pool_active():
            return max(getattr(block_pool, "num_gpu_blocks", 0) - 1, 0)
        self._refresh_dynamic_pool_target(
            block_pool=block_pool,
            reason="effective_num_gpu_blocks",
        )
        return self._dynamic_pool_applied_target_blocks

    def get_num_allocatable_free_blocks(self, block_pool: Any) -> int:
        if not self.dynamic_pool_active():
            return int(getattr(block_pool.free_block_queue, "num_free_blocks", 0))
        self._refresh_dynamic_pool_target(
            block_pool=block_pool,
            reason="allocatable_free_blocks",
        )
        return int(getattr(block_pool.free_block_queue, "num_free_blocks", 0))

    def apply_unified_kv_target_blocks(
        self,
        target_blocks: int,
        *,
        block_pool: Any | None,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        target = self._clamp_dynamic_pool_target(target_blocks)
        metadata = metadata or {}
        if not self.dynamic_pool_active():
            return {
                "ok": False,
                "reason": "dynamic_pool_disabled",
                "requested_target_blocks": max(int(target_blocks or 0), 0),
                "target_blocks": target,
            }
        if block_pool is None:
            return {
                "ok": False,
                "reason": "block_pool_unavailable",
                "requested_target_blocks": max(int(target_blocks or 0), 0),
                "target_blocks": target,
            }

        old_target = self.dynamic_pool_target_blocks
        old_applied_target = self._dynamic_pool_applied_target_blocks
        if target != old_target:
            self.dynamic_pool_target_blocks = target
            self._dynamic_pool_generation += 1
            record_unified_kv_target(
                target,
                old_target_blocks=old_target,
                reason=reason,
                metadata={
                    "source": "unified_control_file",
                    "generation": self._dynamic_pool_generation,
                    **metadata,
                },
            )
            self._write(
                {
                    "action": "kv_dynamic_pool_target_update",
                    "reason": reason,
                    "source": "unified_control_file",
                    "old_target_blocks": old_target,
                    "new_target_blocks": target,
                    "generation": self._dynamic_pool_generation,
                    "metadata": metadata,
                }
            )
        apply_result = self._ensure_dynamic_pool_applied(
            block_pool=block_pool,
            reason=reason,
            external_physical_ready=bool(
                metadata.get("physical_migration_complete", False)
            ),
        )
        return {
            "ok": bool(apply_result.get("ok")),
            "reason": str(apply_result.get("reason") or "applied"),
            "old_target_blocks": old_target,
            "new_target_blocks": target,
            "old_applied_target_blocks": old_applied_target,
            "generation": self._dynamic_pool_generation,
            "applied_target_blocks": self._dynamic_pool_applied_target_blocks,
            "apply_result": apply_result,
        }

    def prefetch_kv_blocks_from_unified(
        self,
        block_specs: list[dict[str, int | None]],
        *,
        block_pool: Any | None,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata = metadata or {}
        if not self.active:
            return {
                "ok": False,
                "reason": "kv_runtime_inactive",
                "requested_blocks": block_specs,
            }
        self.record_config()
        records: list[dict[str, Any]] = []
        for spec in block_specs:
            block_id = int(spec.get("block_id") if spec.get("block_id") is not None else -1)
            if block_id < 0:
                continue
            group_value = spec.get("group_id")
            group_ids = (
                [int(group_value)]
                if group_value is not None
                else list(range(self.num_kv_cache_groups))
            )
            ref_cnt = None
            if block_pool is not None:
                try:
                    block = getattr(block_pool, "blocks", [])[block_id]
                    ref_cnt = getattr(block, "ref_cnt", None)
                    if getattr(block, "lifecycle_state", None) == "cpu_offloaded":
                        block.lifecycle_state = "used_free"
                except Exception:
                    ref_cnt = None
            for group_id in group_ids:
                region_id = f"kv:{group_id}:{block_id}"
                before = get_uvm_residency_region_snapshot(region_id) or {}
                mark_uvm_residency_state(
                    region_id=region_id,
                    state="gpu",
                    metadata={
                        **dict(before.get("metadata") or {}),
                        "stage": "stage_j",
                        "source": "unified_control_prefetch",
                        "reason": reason,
                        "block_id": block_id,
                        "group_id": group_id,
                        "ref_cnt": ref_cnt,
                        "metadata_only": True,
                    },
                )
                after = get_uvm_residency_region_snapshot(region_id) or {}
                records.append(
                    {
                        "region_id": region_id,
                        "group_id": group_id,
                        "block_id": block_id,
                        "before_state": before.get("state"),
                        "after_state": after.get("state"),
                        "bytes": int(after.get("bytes") or before.get("bytes") or 0),
                        "metadata": metadata,
                    }
                )
        issued = [
            item for item in records
            if item.get("before_state") in ("cpu", "prefetching_cpu")
            and item.get("after_state") in ("gpu", "prefetching_gpu")
        ]
        self._write(
            {
                "action": "unified_prefetch_kv_blocks_result",
                "reason": reason,
                "requested_blocks": block_specs,
                "attempted_regions": len(records),
                "issued_regions": len(issued),
                "records": records,
                "metadata": metadata,
            }
        )
        return {
            "ok": True,
            "requested_blocks": block_specs,
            "attempted_regions": len(records),
            "issued_regions": len(issued),
            "records": records,
        }

    def select_free_blocks_for_allocation(
        self,
        *,
        block_pool: Any,
        num_blocks: int,
        enable_caching: bool,
    ) -> list[Any] | None:
        """Prefer free KV blocks that are already resident on GPU.

        This only changes which free block IDs vLLM reuses. Correctness still
        relies on the normal active KV protect/prefetch path after allocation.
        """
        dynamic_active = self.dynamic_pool_active()
        if (
            num_blocks <= 0
            or not self.prefer_gpu_resident_free_blocks
            and not dynamic_active
        ):
            return None
        self._refresh_dynamic_pool_target(
            block_pool=block_pool,
            reason="select_free_blocks_for_allocation",
        )
        try:
            free_blocks = block_pool.free_block_queue.get_all_free_blocks()
        except Exception as exc:  # pragma: no cover - defensive trace path.
            self._write(
                {
                    "action": "gpu_resident_free_block_select_failed",
                    "reason": "free_queue_snapshot_failed",
                    "error": repr(exc),
                    "requested_blocks": num_blocks,
                }
            )
            return None

        if len(free_blocks) < num_blocks:
            return [] if dynamic_active else None

        gpu_fresh_blocks: list[Any] = []
        gpu_warm_blocks: list[Any] = []
        unknown_blocks: list[Any] = []
        cpu_blocks: list[Any] = []
        skipped_cached = 0
        skipped_dynamic_pool = 0
        skipped_unsafe = 0

        for block in free_blocks:
            block_id = int(getattr(block, "block_id", -1))
            safe = (
                block_id >= 0
                and not bool(getattr(block, "is_null", False))
                and getattr(block, "ref_cnt", None) == 0
            )
            if not safe:
                skipped_unsafe += 1
                continue
            if dynamic_active and not self._dynamic_pool_block_id_allocatable(block_id):
                skipped_dynamic_pool += 1
                continue
            if (
                enable_caching
                and not self.prefer_gpu_resident_allow_cached
                and getattr(block, "block_hash", None) is not None
            ):
                skipped_cached += 1
                continue

            state = self._kv_block_residency_class(block_id)
            if state == "gpu":
                lifecycle_state = self._kv_block_lifecycle_class(block)
                if lifecycle_state == "never_used":
                    gpu_fresh_blocks.append(block)
                else:
                    gpu_warm_blocks.append(block)
            elif state == "cpu":
                cpu_blocks.append(block)
            else:
                unknown_blocks.append(block)

        selected = (
            gpu_fresh_blocks + gpu_warm_blocks + unknown_blocks + cpu_blocks
        )[:num_blocks]
        if len(selected) != num_blocks:
            self._write(
                {
                    "action": "gpu_resident_free_block_select_fallback",
                    "requested_blocks": num_blocks,
                    "free_blocks": len(free_blocks),
                    "gpu_candidates": len(gpu_fresh_blocks) + len(gpu_warm_blocks),
                    "gpu_fresh_candidates": len(gpu_fresh_blocks),
                    "gpu_warm_candidates": len(gpu_warm_blocks),
                    "unknown_candidates": len(unknown_blocks),
                    "cpu_candidates": len(cpu_blocks),
                    "skipped_cached": skipped_cached,
                    "skipped_dynamic_pool": skipped_dynamic_pool,
                    "skipped_unsafe": skipped_unsafe,
                    "enable_caching": enable_caching,
                    "dynamic_pool_target_blocks": (
                        self.dynamic_pool_target_blocks if dynamic_active else None
                    ),
                }
            )
            return [] if dynamic_active else None

        selected_ids = [int(getattr(block, "block_id", -1)) for block in selected]
        self._write(
            {
                "action": "gpu_resident_free_block_select",
                "requested_blocks": num_blocks,
                "selected_block_ids": selected_ids,
                "selected_lifecycle_states": [
                    self._kv_block_lifecycle_class(block) for block in selected
                ],
                "gpu_candidates": len(gpu_fresh_blocks) + len(gpu_warm_blocks),
                "gpu_fresh_candidates": len(gpu_fresh_blocks),
                "gpu_warm_candidates": len(gpu_warm_blocks),
                "unknown_candidates": len(unknown_blocks),
                "cpu_candidates": len(cpu_blocks),
                "skipped_cached": skipped_cached,
                "skipped_dynamic_pool": skipped_dynamic_pool,
                "skipped_unsafe": skipped_unsafe,
                "enable_caching": enable_caching,
                "dynamic_pool_target_blocks": (
                    self.dynamic_pool_target_blocks if dynamic_active else None
                ),
            }
        )
        return selected

    def _clamp_dynamic_pool_target(self, target_blocks: int) -> int:
        return min(max(int(target_blocks or 0), 0), self.total_blocks)

    def _dynamic_pool_block_id_allocatable(self, block_id: int) -> bool:
        if not self.dynamic_pool_active():
            return True
        return 0 < block_id <= self._dynamic_pool_applied_target_blocks

    def is_block_id_allocatable(self, block_id: int) -> bool:
        """Public allocation gate used by BlockPool and prefix-cache lookup."""
        return self._dynamic_pool_block_id_allocatable(int(block_id))

    def _dynamic_pool_block_allocatable(self, block: Any) -> bool:
        block_id = int(getattr(block, "block_id", -1))
        return (
            block_id >= 0
            and not bool(getattr(block, "is_null", False))
            and getattr(block, "ref_cnt", None) == 0
            and self._dynamic_pool_block_id_allocatable(block_id)
        )

    def _refresh_dynamic_pool_target(
        self,
        *,
        block_pool: Any | None,
        reason: str,
    ) -> None:
        get_uvm_unified_residency_controller().apply_control_file(
            kv_runtime_policy=self,
            block_pool=block_pool,
            reason=reason,
        )
        if not self.dynamic_pool_active():
            return
        target = self.dynamic_pool_target_blocks
        raw_value: str | None = None
        if self.dynamic_pool_control_file:
            try:
                stat = os.stat(self.dynamic_pool_control_file)
            except FileNotFoundError:
                stat = None
            except OSError as exc:  # pragma: no cover - diagnostic path.
                self._write(
                    {
                        "action": "kv_dynamic_pool_control_read_failed",
                        "control_file": self.dynamic_pool_control_file,
                        "reason": reason,
                        "error": repr(exc),
                    }
                )
                stat = None
            if stat is not None and stat.st_mtime_ns != self._dynamic_pool_control_mtime_ns:
                self._dynamic_pool_control_mtime_ns = stat.st_mtime_ns
                try:
                    with open(
                        self.dynamic_pool_control_file,
                        "r",
                        encoding="utf-8",
                        errors="replace",
                    ) as handle:
                        raw_value = handle.read()
                    parsed = self._parse_dynamic_pool_control(raw_value)
                    if parsed is not None:
                        target = self._clamp_dynamic_pool_target(parsed)
                except Exception as exc:  # pragma: no cover - diagnostic path.
                    self._write(
                        {
                            "action": "kv_dynamic_pool_control_read_failed",
                            "control_file": self.dynamic_pool_control_file,
                            "reason": reason,
                            "error": repr(exc),
                        }
                    )
        if target != self.dynamic_pool_target_blocks:
            old_target = self.dynamic_pool_target_blocks
            self.dynamic_pool_target_blocks = target
            self._dynamic_pool_generation += 1
            record_unified_kv_target(
                target,
                old_target_blocks=old_target,
                reason=reason,
                metadata={
                    "source": "kv_dynamic_pool_target_update",
                    "generation": self._dynamic_pool_generation,
                    "control_file": self.dynamic_pool_control_file,
                    "raw_value": raw_value,
                },
            )
            self._write(
                {
                    "action": "kv_dynamic_pool_target_update",
                    "reason": reason,
                    "old_target_blocks": old_target,
                    "new_target_blocks": target,
                    "generation": self._dynamic_pool_generation,
                    "control_file": self.dynamic_pool_control_file,
                    "raw_value": raw_value,
                }
            )
        self._ensure_dynamic_pool_applied(block_pool=block_pool, reason=reason)

    def _parse_dynamic_pool_control(self, raw_value: str) -> int | None:
        text = (raw_value or "").strip()
        if not text:
            return None
        if text.startswith("{"):
            data = json.loads(text)
            for key in ("target_blocks", "blocks", "target_resident_blocks"):
                if key in data:
                    return int(data[key])
            return None
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                if key.strip() in (
                    "target_blocks",
                    "blocks",
                    "target_resident_blocks",
                ):
                    return int(value.strip())
                continue
            return int(line)
        return None

    def _migrate_dynamic_pool_physical_blocks(
        self,
        *,
        block_ids: list[int],
        direction: str,
        reason: str,
        old_target: int,
        new_target: int,
    ) -> dict[str, Any]:
        unique_block_ids = sorted(
            {int(block_id) for block_id in block_ids if block_id > 0}
        )
        if not unique_block_ids:
            return {
                "ok": True,
                "reason": "no_blocks",
                "direction": direction,
                "block_count": 0,
            }
        if not self.dynamic_pool_physical_migration_enable:
            result = {
                "ok": False,
                "reason": "physical_migration_disabled",
                "direction": direction,
                "block_count": len(unique_block_ids),
            }
            self._write(
                {
                    "action": "kv_dynamic_pool_physical_migration_skipped",
                    **result,
                    "old_target_blocks": old_target,
                    "new_target_blocks": new_target,
                    "source_reason": reason,
                }
            )
            return result

        collection = collect_uvm_kv_cache_physical_ranges_for_blocks(unique_block_ids)
        ranges = list(collection.get("ranges") or [])
        if not ranges:
            result = {
                "ok": False,
                "reason": str(collection.get("reason") or "no_registered_ranges"),
                "direction": direction,
                "block_count": len(unique_block_ids),
                "registered_spans": int(collection.get("registered_spans") or 0),
            }
            self._write(
                {
                    "action": "kv_dynamic_pool_physical_migration_skipped",
                    **result,
                    "old_target_blocks": old_target,
                    "new_target_blocks": new_target,
                    "source_reason": reason,
                    "collection": {
                        key: value
                        for key, value in collection.items()
                        if key != "ranges"
                    },
                }
            )
            return result

        start = time.monotonic()
        if direction == "shrink":
            successes, executor_used, advise_result, prefetch_result = (
                batch_offload_ranges_to_cpu_with_executor(
                    ranges,
                    sync=self.dynamic_pool_physical_migration_sync,
                    executor=self.dynamic_pool_physical_migration_executor,
                )
            )
            driver_result = {
                "advise_result": advise_result,
                "prefetch_result": prefetch_result,
            }
            state = "cpu"
            action = "kv_dynamic_pool_physical_offload"
        else:
            successes, executor_used, prefetch_result = (
                batch_prefetch_ranges_to_device_with_executor(
                    ranges,
                    self.dynamic_pool_physical_migration_device,
                    executor=self.dynamic_pool_physical_migration_executor,
                )
            )
            driver_result = {"prefetch_result": prefetch_result}
            state = "gpu"
            action = "kv_dynamic_pool_physical_prefetch"

        elapsed = time.monotonic() - start
        complete = int(successes) == len(ranges)
        metadata = {
            "stage": "stage_j",
            "source": "kv_dynamic_pool_resize",
            "direction": direction,
            "reason": reason,
            "old_target_blocks": old_target,
            "new_target_blocks": new_target,
            "physical_migration_requested": True,
            "physical_migration_issued": int(successes) > 0,
            "physical_migration_complete": complete,
            "physical_migration_executor": executor_used,
            "physical_migration_sync": self.dynamic_pool_physical_migration_sync,
            "physical_migration_device": (
                -1
                if direction == "shrink"
                else self.dynamic_pool_physical_migration_device
            ),
            "range_count": len(ranges),
            "range_bytes": int(collection.get("range_bytes") or 0),
            "successes": int(successes),
            **driver_result,
        }
        for block_id in unique_block_ids:
            for group_id in range(self.num_kv_cache_groups):
                mark_uvm_residency_state(
                    region_id=f"kv:{group_id}:{block_id}",
                    state=state,
                    metadata={
                        **metadata,
                        "group_id": group_id,
                        "block_id": block_id,
                    },
                )

        self._summary["dynamic_pool_physical_migration_records"] += 1
        self._summary["dynamic_pool_physical_migration_ranges"] += len(ranges)
        self._summary["dynamic_pool_physical_migration_bytes"] += int(
            collection.get("range_bytes") or 0
        )
        self._summary["dynamic_pool_physical_migration_successes"] += int(successes)
        record_uvm_residency_action(
            action=action,
            pool="kv",
            region_id=f"kv:dynamic_pool:{direction}:{old_target}:{new_target}",
            bytes=int(collection.get("range_bytes") or 0),
            mode=executor_used,
            range_count=len(ranges),
            metadata={
                **metadata,
                "block_count": len(unique_block_ids),
                "block_intervals": collection.get("block_intervals"),
                "range_records": collection.get("range_records"),
                "elapsed_seconds": elapsed,
            },
        )
        record_unified_residency_snapshot(
            reason="kv_dynamic_pool_physical_migration",
            kv_pressure={
                **metadata,
                "block_count": len(unique_block_ids),
                "elapsed_seconds": elapsed,
            },
        )
        result = {
            "ok": complete,
            "reason": "complete" if complete else "partial",
            "direction": direction,
            "block_count": len(unique_block_ids),
            "range_count": len(ranges),
            "range_bytes": int(collection.get("range_bytes") or 0),
            "successes": int(successes),
            "executor": executor_used,
            "elapsed_seconds": elapsed,
            **driver_result,
        }
        self._write(
            {
                "action": "kv_dynamic_pool_physical_migration",
                **result,
                "old_target_blocks": old_target,
                "new_target_blocks": new_target,
                "source_reason": reason,
                "range_records": collection.get("range_records"),
            }
        )
        return result

    def _ensure_dynamic_pool_applied(
        self,
        *,
        block_pool: Any | None,
        reason: str,
        external_physical_ready: bool = False,
    ) -> dict[str, Any]:
        if not self.dynamic_pool_active() or block_pool is None:
            return {"ok": False, "reason": "dynamic_pool_or_block_pool_unavailable"}
        target = self.dynamic_pool_target_blocks
        old_target = self._dynamic_pool_applied_target_blocks
        if old_target == target:
            return {
                "ok": True,
                "reason": "already_applied",
                "applied_target_blocks": old_target,
            }

        direction = "grow" if target > old_target else "shrink"
        offloaded = 0
        prefetched = 0
        skipped_active = 0
        physical_offload_block_ids: list[int] = []
        physical_prefetch_block_ids: list[int] = []
        blocks = getattr(block_pool, "blocks", [])
        first_block = min(old_target, target) + 1
        last_block = max(old_target, target)
        candidates = [
            blocks[block_id]
            for block_id in range(first_block, last_block + 1)
            if 0 < block_id < len(blocks)
        ]

        if direction == "shrink":
            active_candidates = [
                block
                for block in candidates
                if not self._dynamic_pool_block_safe_for_resize(block)
            ]
            if active_candidates:
                skipped_active = len(active_candidates)
                queue_result = {
                    "moved_gpu": 0,
                    "moved_offline": 0,
                    "deferred_active": skipped_active,
                    "gpu_free_blocks": (
                        block_pool.free_block_queue.num_free_blocks
                    ),
                    "offline_free_blocks": (
                        block_pool.offline_block_queue.num_free_blocks
                    ),
                }
                physical_result = {
                    "ok": False,
                    "reason": "active_blocks_must_drain_before_shrink",
                    "direction": direction,
                    "block_count": 0,
                    "active_block_ids": [
                        int(getattr(block, "block_id", -1))
                        for block in active_candidates
                    ],
                }
                committed = False
                physical_complete = False
            else:
                physical_complete = False
            # Quarantine before migration. A partial migration stays
            # quarantined; a zero-progress failure can safely roll back.
            if not active_candidates:
                queue_result = block_pool.apply_uvm_dynamic_pool_target(
                    target,
                    old_target_blocks=old_target,
                )
                for block in candidates:
                    physical_offload_block_ids.append(
                        int(getattr(block, "block_id", -1))
                    )
                if external_physical_ready:
                    physical_result = {
                        "ok": True,
                        "reason": "external_physical_migration_complete",
                        "direction": direction,
                        "block_count": len(physical_offload_block_ids),
                    }
                else:
                    physical_result = self._migrate_dynamic_pool_physical_blocks(
                        block_ids=physical_offload_block_ids,
                        direction="shrink",
                        reason=reason,
                        old_target=old_target,
                        new_target=target,
                    )
                physical_complete = bool(physical_result.get("ok")) or not (
                    self.dynamic_pool_physical_migration_enable
                )
                if physical_complete:
                    for block in candidates:
                        if self._mark_dynamic_pool_block_cpu(
                            block=block,
                            reason=reason,
                            old_target=old_target,
                            new_target=target,
                        ):
                            offloaded += 1
                    self._dynamic_pool_applied_target_blocks = target
                    committed = True
                else:
                    committed = False
                    if int(physical_result.get("successes") or 0) == 0:
                        queue_result = block_pool.apply_uvm_dynamic_pool_target(
                            old_target,
                            old_target_blocks=target,
                        )
        else:
            for block in candidates:
                block_id = int(getattr(block, "block_id", -1))
                if not self._dynamic_pool_block_safe_for_resize(block):
                    skipped_active += 1
                    continue
                physical_prefetch_block_ids.append(block_id)
            if external_physical_ready:
                physical_result = {
                    "ok": True,
                    "reason": "external_physical_migration_complete",
                    "direction": direction,
                    "block_count": len(physical_prefetch_block_ids),
                }
            else:
                physical_result = self._migrate_dynamic_pool_physical_blocks(
                    block_ids=physical_prefetch_block_ids,
                    direction="grow",
                    reason=reason,
                    old_target=old_target,
                    new_target=target,
                )
            physical_complete = bool(physical_result.get("ok")) or not (
                self.dynamic_pool_physical_migration_enable
            )
            if physical_complete:
                for block in candidates:
                    if not self._dynamic_pool_block_safe_for_resize(block):
                        continue
                    if self._mark_dynamic_pool_block_gpu(
                        block=block,
                        reason=reason,
                        old_target=old_target,
                        new_target=target,
                    ):
                        prefetched += 1
                queue_result = block_pool.apply_uvm_dynamic_pool_target(
                    target,
                    old_target_blocks=old_target,
                )
                self._dynamic_pool_applied_target_blocks = target
                committed = True
            else:
                queue_result = {
                    "moved_gpu": 0,
                    "moved_offline": 0,
                    "deferred_active": skipped_active,
                    "gpu_free_blocks": block_pool.free_block_queue.num_free_blocks,
                    "offline_free_blocks": (
                        block_pool.offline_block_queue.num_free_blocks
                    ),
                }
                committed = False

        self._summary["dynamic_pool_resize_records"] += 1
        self._summary["dynamic_pool_offload_blocks"] += offloaded
        self._summary["dynamic_pool_prefetch_blocks"] += prefetched
        self._summary["dynamic_pool_deferred_active_blocks"] += skipped_active
        self._write(
            {
                "action": "kv_dynamic_pool_resize",
                "reason": reason,
                "direction": direction,
                "old_target_blocks": old_target,
                "new_target_blocks": target,
                "offloaded_blocks": offloaded,
                "prefetched_blocks": prefetched,
                "deferred_active_blocks": skipped_active,
                "committed": committed,
                "queue_transition": queue_result,
                "generation": self._dynamic_pool_generation,
                "physical_migration": physical_result,
            }
        )
        record_unified_residency_snapshot(
            reason="kv_dynamic_pool_resize",
            kv_pressure={
                "direction": direction,
                "old_target_blocks": old_target,
                "new_target_blocks": target,
                "offloaded_blocks": offloaded,
                "prefetched_blocks": prefetched,
                "deferred_active_blocks": skipped_active,
                "committed": committed,
                "queue_transition": queue_result,
                "generation": self._dynamic_pool_generation,
                "source_reason": reason,
                "physical_migration": physical_result,
            },
        )
        return {
            "ok": committed,
            "reason": (
                "applied"
                if committed
                else str(
                    physical_result.get("reason")
                    or "physical_migration_incomplete"
                )
            ),
            "direction": direction,
            "desired_target_blocks": target,
            "applied_target_blocks": self._dynamic_pool_applied_target_blocks,
            "queue_transition": queue_result,
            "physical_migration": physical_result,
            "deferred_active_blocks": skipped_active,
        }

    def _dynamic_pool_block_safe_for_resize(self, block: Any) -> bool:
        return (
            int(getattr(block, "block_id", -1)) > 0
            and not bool(getattr(block, "is_null", False))
            and getattr(block, "ref_cnt", None) == 0
        )

    def _mark_dynamic_pool_block_cpu(
        self,
        *,
        block: Any,
        reason: str,
        old_target: int,
        new_target: int,
    ) -> bool:
        block_id = int(getattr(block, "block_id", -1))
        previous_lifecycle = self._kv_block_lifecycle_class(block)
        if previous_lifecycle != "cpu_offloaded":
            self._dynamic_pool_saved_lifecycle[block_id] = previous_lifecycle
        block.lifecycle_state = "cpu_offloaded"
        issued = False
        for group_id in range(self.num_kv_cache_groups):
            region_id = f"kv:{group_id}:{block_id}"
            snapshot = get_uvm_residency_region_snapshot(region_id)
            snapshot_metadata = dict((snapshot or {}).get("metadata") or {})
            state = str((snapshot or {}).get("state") or "unknown")
            size_bytes = int(
                (snapshot or {}).get("bytes")
                or self.bytes_per_group_block
            )
            if state == "cpu":
                issued = True
                continue
            metadata = {
                **snapshot_metadata,
                "stage": "stage_j",
                "source": "kv_dynamic_pool_resize",
                "direction": "shrink",
                "reason": reason,
                "block_id": block_id,
                "group_id": group_id,
                "old_target_blocks": old_target,
                "new_target_blocks": new_target,
                "ref_cnt": getattr(block, "ref_cnt", None),
                "previous_lifecycle_state": previous_lifecycle,
                "previous_state": state,
                "requested_mode": "logical_dynamic_pool",
                "physical_migration_requested": (
                    self.dynamic_pool_physical_migration_enable
                ),
                "physical_migration_issued": False,
                "metadata_only": True,
            }
            record_uvm_residency_action(
                action="residency_kv_data_offload_attempt",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                metadata=metadata,
            )
            mark_uvm_residency_state(
                region_id=region_id,
                state="cpu",
                metadata=metadata,
            )
            record_uvm_residency_action(
                action="residency_kv_data_offload_issued",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                mode="logical_dynamic_pool",
                ptr=None,
                range_count=0,
                metadata=metadata,
            )
            self._summary["stage_l5_data_offload_records"] += 1
            issued = True
        return issued

    def _mark_dynamic_pool_block_gpu(
        self,
        *,
        block: Any,
        reason: str,
        old_target: int,
        new_target: int,
    ) -> bool:
        block_id = int(getattr(block, "block_id", -1))
        restored_lifecycle = self._dynamic_pool_saved_lifecycle.pop(
            block_id,
            "used_free",
        )
        if restored_lifecycle not in ("never_used", "used_free"):
            restored_lifecycle = "used_free"
        block.lifecycle_state = restored_lifecycle
        issued = False
        for group_id in range(self.num_kv_cache_groups):
            region_id = f"kv:{group_id}:{block_id}"
            snapshot = get_uvm_residency_region_snapshot(region_id)
            snapshot_metadata = dict((snapshot or {}).get("metadata") or {})
            state = str((snapshot or {}).get("state") or "unknown")
            size_bytes = int(
                (snapshot or {}).get("bytes")
                or self.bytes_per_group_block
            )
            if state == "gpu":
                issued = True
                continue
            metadata = {
                **snapshot_metadata,
                "stage": "stage_j",
                "source": "kv_dynamic_pool_resize",
                "direction": "grow",
                "reason": reason,
                "block_id": block_id,
                "group_id": group_id,
                "old_target_blocks": old_target,
                "new_target_blocks": new_target,
                "ref_cnt": getattr(block, "ref_cnt", None),
                "restored_lifecycle_state": restored_lifecycle,
                "previous_state": state,
                "requested_mode": "logical_dynamic_pool",
                "physical_migration_requested": (
                    self.dynamic_pool_physical_migration_enable
                ),
                "physical_migration_issued": False,
                "metadata_only": True,
            }
            record_uvm_residency_action(
                action="residency_kv_prefetch_back_attempt",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                metadata=metadata,
            )
            mark_uvm_residency_state(
                region_id=region_id,
                state="prefetching_gpu",
                metadata=metadata,
            )
            mark_uvm_residency_state(
                region_id=region_id,
                state="gpu",
                metadata=metadata,
            )
            record_uvm_residency_action(
                action="residency_kv_prefetch_back_issued",
                pool="kv",
                region_id=region_id,
                bytes=size_bytes,
                mode="logical_dynamic_pool",
                ptr=None,
                range_count=0,
                metadata=metadata,
            )
            issued = True
        return issued

    def _kv_block_lifecycle_class(self, block: Any) -> str:
        state = str(getattr(block, "lifecycle_state", "never_used") or "never_used")
        if state in ("never_used", "used_free", "cpu_offloaded"):
            return state
        return "used_free"

    def _kv_block_residency_class(self, block_id: int) -> str:
        saw_unknown = False
        for group_id in range(self.num_kv_cache_groups):
            snapshot = get_uvm_residency_region_snapshot(f"kv:{group_id}:{block_id}")
            state = str((snapshot or {}).get("state") or "unknown")
            if state in ("cpu", "prefetching_cpu", "released"):
                return "cpu"
            if state not in ("gpu", "prefetching_gpu"):
                saw_unknown = True
        return "unknown" if saw_unknown else "gpu"

    def register_stage_l5_kv_blocks(self, block_pool: Any) -> None:
        """Register KV block metadata in Stage L residency.

        The post-warmup path handles physical KV migration directly from the
        model runner; the runtime policy keeps only scheduler-side metadata.
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
                        "lifecycle_state": self._kv_block_lifecycle_class(block),
                        "ever_used": (
                            self._kv_block_lifecycle_class(block) != "never_used"
                        ),
                        "is_prefix_cache": getattr(block, "block_hash", None)
                        is not None,
                        "in_free_queue": True,
                        "block_hash": _block_hash_to_text(
                            getattr(block, "block_hash", None)
                        ),
                        "offload_capable": False,
                        "metadata_only": True,
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

    def should_allow_allocation(
        self,
        *,
        request_id: str,
        num_blocks_to_allocate: int,
        block_pool: Any,
    ) -> bool:
        """Trace pressure and decide whether new KV block admission is allowed."""
        if num_blocks_to_allocate <= 0:
            return True
        self.record_config()
        pressure = self._pressure_snapshot(block_pool, num_blocks_to_allocate)
        record_unified_residency_snapshot(
            reason="kv_allocation_pressure",
            kv_pressure={
                "request_id": request_id,
                "num_blocks_to_allocate": num_blocks_to_allocate,
                **pressure,
            },
        )
        if not self.active:
            return True
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
        physical_total_blocks = max(getattr(block_pool, "num_gpu_blocks", 0) - 1, 0)
        total_blocks = (
            self.get_effective_num_gpu_blocks(block_pool)
            if self.dynamic_pool_active()
            else physical_total_blocks
        )
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
            "physical_total_blocks": physical_total_blocks,
            "dynamic_pool_enable": self.dynamic_pool_active(),
            "dynamic_pool_target_blocks": (
                self.dynamic_pool_target_blocks
                if self.dynamic_pool_active()
                else None
            ),
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
                            "lifecycle_state": self._kv_block_lifecycle_class(block),
                            "ever_used": (
                                self._kv_block_lifecycle_class(block) != "never_used"
                            ),
                            "is_null": getattr(block, "is_null", None),
                            "is_prefix_cache": block_hash is not None,
                            "in_free_queue": True,
                            "block_hash": _block_hash_to_text(block_hash),
                            "offload_capable": self._kv_region_offload_capable(
                                group_id, block_id
                            ),
                            "metadata_only": True,
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
                    "lifecycle_state": self._kv_block_lifecycle_class(block),
                    "ever_used": self._kv_block_lifecycle_class(block) != "never_used",
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
        """Keep the block-pool hook for compatibility."""
        return

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
            size_bytes = int(
                (snapshot or {}).get("bytes")
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
                "metadata_only": True,
            }
            result = runtime.protect_kv_region(
                region_id=region_id,
                block_id=block_id,
                group_id=group_id,
                bytes=size_bytes,
                ranges=None,
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
                        "stage": "stage_l5",
                        "source": "block_free_lifecycle",
                        "ref_cnt": getattr(block, "ref_cnt", None),
                        "lifecycle_state": self._kv_block_lifecycle_class(block),
                        "ever_used": self._kv_block_lifecycle_class(block) != "never_used",
                        "is_null": getattr(block, "is_null", None),
                        "is_prefix_cache": getattr(block, "block_hash", None) is not None,
                        "in_free_queue": safe,
                        "block_hash": _block_hash_to_text(getattr(block, "block_hash", None)),
                        "offload_capable": self._kv_region_offload_capable(group_id, block_id),
                        "metadata_only": True,
                    },
                )

    def _kv_region_offload_capable(self, group_id: int, block_id: int) -> bool:
        snapshot = get_uvm_residency_region_snapshot(f"kv:{group_id}:{block_id}")
        if not snapshot:
            return False
        metadata = snapshot.get("metadata") or {}
        return bool(snapshot.get("ptr")) and bool(metadata.get("offload_capable"))

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
                "dynamic_pool_enable": self.dynamic_pool_active(),
                "dynamic_pool_target_blocks": (
                    self.dynamic_pool_target_blocks
                    if self.dynamic_pool_active()
                    else None
                ),
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
                handle.flush()
