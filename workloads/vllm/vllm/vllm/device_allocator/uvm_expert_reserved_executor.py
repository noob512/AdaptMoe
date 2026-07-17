# SPDX-License-Identifier: Apache-2.0
"""Reserved-backed expert region executor for the modified UVM driver.

This is the vLLM-side Phase 4 shim for the Phase 2C driver path:
registered expert regions are copied into a fixed reserved PMM slot pool and
their original managed VAs are remapped to readonly slot-backed GPU PTEs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import threading
import time
from typing import Any, Sequence

import torch

from vllm.device_allocator.uvm import (
    UVM_REGION_FLAG_ACTIVE,
    UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED,
    UVM_REGION_FLAG_PREFETCHABLE,
    UVM_REGION_FLAG_PROTECTED,
    UVM_REGION_POOL_WEIGHT,
    UVM_REGION_ROLE_DENSE_WEIGHT,
    UVM_REGION_ROLE_MOE_EXPERT_W13,
    UVM_REGION_ROLE_MOE_EXPERT_W2,
    batch_offload_ranges_to_cpu_with_executor,
    batch_prefetch_ranges_to_cpu,
    driver_expert_reserved_pool_query,
    driver_expert_reserved_prefetch_regions,
    driver_query_region,
    driver_register_region,
    driver_unregister_region,
)
from vllm.device_allocator.uvm_pool_residency import mark_uvm_residency_state
from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %s", name, raw, default)
        return default


def _debug_trace_enabled() -> bool:
    return _env_flag(
        "VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE",
        _env_flag("VLLM_UVM_WEIGHT_PREFETCH_TRACE_ENABLE", False),
    )


def _append_debug_trace(record: dict[str, Any]) -> None:
    if not _debug_trace_enabled():
        return
    path = (
        os.environ.get("VLLM_UVM_EXPERT_RESERVED_DEBUG_TRACE_FILE")
        or os.environ.get("VLLM_UVM_WEIGHT_PREFETCH_TRACE_FILE")
        or "/tmp/vllm_reserved_debug_trace.jsonl"
    )
    try:
        enriched = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            "executor": "expert_reserved_regions_async",
            **record,
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(enriched, sort_keys=True) + "\n")
    except OSError:
        logger.debug("Failed to write reserved expert debug trace", exc_info=True)


def _page_size() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return 4096


def _align_up(value: int, alignment: int) -> int:
    if value <= 0:
        return 0
    alignment = max(int(alignment), 1)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _role_id(role: str) -> int:
    normalized = str(role or "").strip().lower()
    if normalized in ("moe_gate_up", "w13", "w13_weight", "moe_expert_w13"):
        return UVM_REGION_ROLE_MOE_EXPERT_W13
    if normalized in ("moe_down", "w2", "w2_weight", "moe_expert_w2"):
        return UVM_REGION_ROLE_MOE_EXPERT_W2
    return UVM_REGION_ROLE_DENSE_WEIGHT


@dataclass
class ReservedRegionItem:
    region_id: str
    driver_region_id: int
    ptr: int
    bytes: int
    layer_id: int | None
    global_expert_id: int
    local_expert_id: int
    role: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _SlotState:
    slot_id: int
    owner_regions: list[ReservedRegionItem] = field(default_factory=list)
    owner_layer_name: str = ""
    owner_step: int = -1
    owner_layer_id: int | None = None
    owner_expert_id: int | None = None
    last_event: torch.cuda.Event | None = None
    state: str = "FREE"
    epoch: int = 0


@dataclass(frozen=True)
class _RegisteredRegionState:
    ptr: int
    bytes: int
    layer_id: int
    expert_id: int
    role_id: int
    flags: int


class ExpertReservedRegionsExecutor:
    """Small fixed-window executor for CPU_OFFLOADED_RESERVED experts."""

    def __init__(
        self,
        *,
        device: int,
        pool_id: int,
        slot_count: int,
        slot_bytes: int,
        strict_footprint: bool,
        async_: bool,
        wait_on_reuse: bool,
        sync_before_compute: bool,
        cleanup_synchronize: bool,
        prefetch_cpu_source: bool,
    ) -> None:
        self.device = int(device)
        self.pool_id = int(pool_id)
        self.slot_count = max(int(slot_count), 0)
        self.configured_slot_bytes = max(int(slot_bytes), 0)
        self.slot_bytes = self.configured_slot_bytes
        self.strict_footprint = bool(strict_footprint)
        self.async_ = bool(async_)
        self.wait_on_reuse = bool(wait_on_reuse)
        self.sync_before_compute = bool(sync_before_compute)
        self.cleanup_synchronize = bool(cleanup_synchronize)
        self.prefetch_cpu_source = bool(prefetch_cpu_source)
        self.slot_policy = (
            os.environ.get("VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY", "dynamic")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.slot_policy not in ("dynamic", "layer_mod_bank", "layer_mod"):
            logger.warning(
                "Ignoring unknown VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY=%r; "
                "using dynamic",
                self.slot_policy,
            )
            self.slot_policy = "dynamic"
        self.bank_count = max(
            _env_int("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", 2),
            1,
        )
        self.bank_sync_on_reuse = _env_flag(
            "VLLM_UVM_EXPERT_RESERVED_BANK_SYNC_ON_REUSE",
            True,
        )
        self._lock = threading.Lock()
        self._slots = {
            slot_id: _SlotState(slot_id=slot_id)
            for slot_id in range(self.slot_count)
        }
        self._free_cursor = 0
        self._prealloc_done = False
        self._active_layer_slots: dict[tuple[str, int], set[int]] = {}
        self._registered_regions: dict[int, _RegisteredRegionState] = {}

    def _layer_mod_bank_enabled(self) -> bool:
        return self.slot_policy in ("layer_mod_bank", "layer_mod")

    def _bank_slot_ids_for_layer(self, layer_id: int | None) -> list[int] | None:
        if not self._layer_mod_bank_enabled() or layer_id is None:
            return None
        if self.slot_count <= 0 or self.bank_count <= 0:
            return []
        bank_count = min(self.bank_count, self.slot_count)
        bank_size = self.slot_count // bank_count
        if bank_size <= 0:
            return []
        bank_id = int(layer_id) % bank_count
        start = bank_id * bank_size
        end = self.slot_count if bank_id == bank_count - 1 else start + bank_size
        return list(range(start, end))

    def _ensure_driver_region(self, item: ReservedRegionItem) -> int:
        if item.driver_region_id <= 0 or item.ptr <= 0 or item.bytes <= 0:
            return -2
        flags = int(item.metadata.get("driver_flags", 0) or 0)
        flags |= (
            UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED
            | UVM_REGION_FLAG_PREFETCHABLE
            | UVM_REGION_FLAG_PROTECTED
            | UVM_REGION_FLAG_ACTIVE
        )
        desired = _RegisteredRegionState(
            ptr=int(item.ptr),
            bytes=int(item.bytes),
            layer_id=max(int(item.layer_id or 0), 0),
            expert_id=max(int(item.global_expert_id), 0),
            role_id=_role_id(item.role),
            flags=flags,
        )
        cached = self._registered_regions.get(int(item.driver_region_id))
        if cached == desired:
            item.metadata["driver_registered"] = True
            item.metadata["driver_cpu_offloaded_reserved_registered"] = True
            _append_debug_trace({
                "action": "reserved_debug_register_region_cached",
                "region_id": item.region_id,
                "driver_region_id": item.driver_region_id,
                "ptr": f"0x{item.ptr:x}",
                "bytes": item.bytes,
            })
            return 0

        queried = driver_query_region(int(item.driver_region_id))
        if queried is not None:
            queried_state = _RegisteredRegionState(
                ptr=int(queried.get("base", 0)),
                bytes=int(queried.get("length", 0)),
                layer_id=int(queried.get("layer_id", 0)),
                expert_id=int(queried.get("expert_id", 0)),
                role_id=int(queried.get("role", 0)),
                flags=int(queried.get("flags", 0)),
            )
            if queried_state == desired:
                self._registered_regions[int(item.driver_region_id)] = desired
                item.metadata["driver_registered"] = True
                item.metadata["driver_cpu_offloaded_reserved_registered"] = True
                _append_debug_trace({
                    "action": "reserved_debug_register_region_driver_cached",
                    "region_id": item.region_id,
                    "driver_region_id": item.driver_region_id,
                    "ptr": f"0x{item.ptr:x}",
                    "bytes": item.bytes,
                })
                return 0
            if (
                item.metadata.get("stage")
                in ("uvm_2mb_envelope", "uvm_chunk_envelope")
                and queried_state.ptr == desired.ptr
                and queried_state.bytes == desired.bytes
            ):
                self._registered_regions[int(item.driver_region_id)] = desired
                item.metadata["driver_registered"] = True
                item.metadata["driver_cpu_offloaded_reserved_registered"] = True
                _append_debug_trace({
                    "action": "reserved_debug_register_region_envelope_va_cached",
                    "region_id": item.region_id,
                    "driver_region_id": item.driver_region_id,
                    "ptr": f"0x{item.ptr:x}",
                    "bytes": item.bytes,
                    "queried": queried,
                    "desired": desired.__dict__,
                })
                return 0

        should_unregister = (
            bool(item.metadata.get("driver_registered"))
            or (cached is not None and cached != desired)
            or (queried is not None and queried_state != desired)
        )
        if should_unregister:
            unregister_status = driver_unregister_region(int(item.driver_region_id))
            if unregister_status < 0:
                _append_debug_trace({
                    "action": "reserved_debug_unregister_before_register_failed",
                    "region_id": item.region_id,
                    "driver_region_id": item.driver_region_id,
                    "status": unregister_status,
                    "cached": cached.__dict__ if cached is not None else None,
                    "queried": queried,
                })
                return int(unregister_status)
            self._registered_regions.pop(int(item.driver_region_id), None)
        status = driver_register_region(
            region_id=int(item.driver_region_id),
            ptr=int(item.ptr),
            nbytes=int(item.bytes),
            pool=UVM_REGION_POOL_WEIGHT,
            layer_id=desired.layer_id,
            expert_id=desired.expert_id,
            role=desired.role_id,
            flags=flags,
        )
        if status >= 0:
            self._registered_regions[int(item.driver_region_id)] = desired
            item.metadata["driver_registered"] = True
            item.metadata["driver_cpu_offloaded_reserved_registered"] = True
        else:
            self._registered_regions.pop(int(item.driver_region_id), None)
        return int(status)

    def _ensure_preallocated_locked(self, required_slot_bytes: int) -> dict[str, Any]:
        required_slot_bytes = max(int(required_slot_bytes), self.configured_slot_bytes)
        required_slot_bytes = _align_up(required_slot_bytes, _page_size())
        if self.slot_count <= 0 or required_slot_bytes <= 0:
            return {
                "ok": False,
                "reason": "invalid_slot_config",
                "slot_count": self.slot_count,
                "slot_bytes": required_slot_bytes,
            }

        if self._prealloc_done and self.slot_bytes >= required_slot_bytes:
            return {
                "ok": True,
                "reason": "already_preallocated",
                "slot_count": self.slot_count,
                "slot_bytes": self.slot_bytes,
            }

        query = driver_expert_reserved_pool_query(prefer_existing=True)
        existing_slots: set[int] = set()
        for entry in query.get("entries") or []:
            try:
                if int(entry.get("pool_id", -1)) != self.pool_id:
                    continue
                slot_id = int(entry.get("slot_id", -1))
                slot_bytes = int(
                    entry.get("slot_bytes", entry.get("logical_slot_bytes", 0)) or 0
                )
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= slot_id < self.slot_count and slot_bytes >= required_slot_bytes:
                existing_slots.add(slot_id)

        missing_slot_ids = [
            slot_id
            for slot_id in range(self.slot_count)
            if slot_id not in existing_slots
        ]
        if missing_slot_ids:
            return {
                "ok": False,
                "reason": "slots_not_preallocated",
                "slot_count": self.slot_count,
                "slot_bytes": required_slot_bytes,
                "query": query,
                "backed_slots": len(existing_slots),
                "missing_slot_ids": missing_slot_ids,
                "runtime_lazy_prealloc_enabled": False,
            }

        self.slot_bytes = required_slot_bytes
        self._prealloc_done = True
        return {
            "ok": True,
            "reason": "preallocated_verified",
            "slot_count": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "query": query,
        }

    def _wait_for_slot_locked(self, slot: _SlotState) -> bool:
        event = slot.last_event
        if event is None:
            return False
        if not self.wait_on_reuse and event.query():
            slot.last_event = None
            return False
        if self.wait_on_reuse:
            wait_start_ns = time.perf_counter_ns()
            _append_debug_trace({
                "action": "reserved_debug_slot_event_wait_enter",
                "pool_id": self.pool_id,
                "slot_id": slot.slot_id,
                "slot_state": slot.state,
                "owner_layer_name": slot.owner_layer_name,
                "owner_step": slot.owner_step,
                "owner_expert_id": slot.owner_expert_id,
            })
            event.synchronize()
            slot.last_event = None
            _append_debug_trace({
                "action": "reserved_debug_slot_event_wait_exit",
                "pool_id": self.pool_id,
                "slot_id": slot.slot_id,
                "elapsed_us": (time.perf_counter_ns() - wait_start_ns) / 1000.0,
            })
            return True
        return False

    def _choose_slot_locked(
        self,
        used_slots: set[int],
        allowed_slot_ids: Sequence[int] | None = None,
    ) -> _SlotState | None:
        if self.slot_count <= 0:
            return None
        if allowed_slot_ids is not None:
            for slot_id in allowed_slot_ids:
                if slot_id in used_slots:
                    continue
                slot = self._slots.get(int(slot_id))
                if slot is None:
                    continue
                self._wait_for_slot_locked(slot)
                return slot
            return None
        for offset in range(self.slot_count):
            slot_id = (self._free_cursor + offset) % self.slot_count
            if slot_id in used_slots:
                continue
            slot = self._slots[slot_id]
            self._wait_for_slot_locked(slot)
            return slot
        return None

    def _invalidate_slots_for_reuse_locked(
        self,
        *,
        slots: Sequence[_SlotState],
        layer_name: str,
        step: int,
        reason: str,
    ) -> dict[str, Any]:
        stale_slots = [slot for slot in slots if slot.owner_regions]
        if not stale_slots:
            return {
                "ok": True,
                "status": "no_stale_slots",
                "invalidated_slots": 0,
                "regions": [],
                "driver_result": None,
            }

        for slot in stale_slots:
            if slot.last_event is not None:
                self._wait_for_slot_locked(slot)
            elif slot.state == "IN_USE" and self.bank_sync_on_reuse:
                sync_start_ns = time.perf_counter_ns()
                _append_debug_trace({
                    "action": "reserved_debug_bank_reuse_device_sync_enter",
                    "layer_name": layer_name,
                    "step": int(step),
                    "pool_id": self.pool_id,
                    "slot_id": slot.slot_id,
                    "owner_layer_name": slot.owner_layer_name,
                    "owner_step": slot.owner_step,
                    "owner_layer_id": slot.owner_layer_id,
                    "owner_expert_id": slot.owner_expert_id,
                    "reason": reason,
                })
                torch.cuda.synchronize(self.device)
                _append_debug_trace({
                    "action": "reserved_debug_bank_reuse_device_sync_exit",
                    "layer_name": layer_name,
                    "step": int(step),
                    "pool_id": self.pool_id,
                    "slot_id": slot.slot_id,
                    "elapsed_us": (
                        time.perf_counter_ns() - sync_start_ns
                    ) / 1000.0,
                    "reason": reason,
                })

        assignments = [
            {
                "pool_id": self.pool_id,
                "slot_id": slot.slot_id,
                "strict_footprint": self.strict_footprint,
            }
            for slot in stale_slots
        ]
        ioctl_start_ns = time.perf_counter_ns()
        _append_debug_trace({
            "action": "reserved_debug_bank_reuse_invalidate_ioctl_enter",
            "layer_name": layer_name,
            "step": int(step),
            "slot_ids": [slot.slot_id for slot in stale_slots],
            "assignment_count": len(assignments),
            "reason": reason,
        })
        driver_result = driver_expert_reserved_prefetch_regions(
            assignments=assignments,
            device=self.device,
            async_=False,
            strict_footprint=self.strict_footprint,
            mode="invalidate",
            prefer_existing=True,
            return_entries=False,
        )
        ok = driver_result.get("ok") is True
        _append_debug_trace({
            "action": "reserved_debug_bank_reuse_invalidate_ioctl_exit",
            "layer_name": layer_name,
            "step": int(step),
            "ok": ok,
            "rm_status": driver_result.get("rm_status"),
            "success_count": driver_result.get("success_count"),
            "failure_count": driver_result.get("failure_count"),
            "invalidated_slots": driver_result.get("invalidated_slots"),
            "elapsed_us": (time.perf_counter_ns() - ioctl_start_ns) / 1000.0,
            "reason": reason,
        })
        if not ok:
            return {
                "ok": False,
                "status": "invalidate_failed",
                "invalidated_slots": 0,
                "regions": [],
                "driver_result": driver_result,
            }

        invalidated_regions: list[dict[str, Any]] = []
        for slot in stale_slots:
            for item in slot.owner_regions:
                invalidated_regions.append(
                    {
                        "region_id": item.region_id,
                        "driver_region_id": item.driver_region_id,
                        "layer_id": item.layer_id,
                        "global_expert_id": item.global_expert_id,
                        "local_expert_id": item.local_expert_id,
                        "role": item.role,
                        "bytes": item.bytes,
                        "slot_id": slot.slot_id,
                    }
                )
                mark_uvm_residency_state(
                    region_id=item.region_id,
                    state="cpu",
                    metadata={
                        **item.metadata,
                        "executor": "expert_reserved_regions_async",
                        "reserved_cleanup_step": int(step),
                        "reserved_slot_id": slot.slot_id,
                        "reserved_cleanup_reason": reason,
                    },
                )
            slot.owner_regions = []
            slot.owner_layer_name = ""
            slot.owner_step = -1
            slot.owner_layer_id = None
            slot.owner_expert_id = None
            slot.state = "READY"
            slot.last_event = None

        return {
            "ok": True,
            "status": "ok",
            "invalidated_slots": len(stale_slots),
            "regions": invalidated_regions,
            "driver_result": driver_result,
        }

    def prefetch_for_layer(
        self,
        *,
        layer_name: str,
        step: int,
        items: Sequence[ReservedRegionItem],
    ) -> dict[str, Any]:
        start_ns = time.perf_counter_ns()
        _append_debug_trace({
            "action": "reserved_debug_prefetch_enter",
            "layer_name": layer_name,
            "step": int(step),
            "item_count": len(items),
            "pool_id": self.pool_id,
            "slot_count": self.slot_count,
            "configured_slot_bytes": self.configured_slot_bytes,
            "async": self.async_,
            "prefetch_cpu_source": self.prefetch_cpu_source,
            "slot_policy": self.slot_policy,
            "bank_count": self.bank_count,
        })
        page_size = _page_size()
        valid_items: list[ReservedRegionItem] = []
        for item in items:
            if item.driver_region_id <= 0 or item.ptr <= 0 or item.bytes <= 0:
                continue
            valid_items.append(item)

        if not valid_items:
            _append_debug_trace({
                "action": "reserved_debug_prefetch_no_regions",
                "layer_name": layer_name,
                "step": int(step),
                "elapsed_us": (time.perf_counter_ns() - start_ns) / 1000.0,
            })
            return {
                "ok": True,
                "status": "no_regions",
                "executor": "expert_reserved_regions_async",
                "success_count": 0,
                "failure_count": 0,
                "assigned_slots": [],
                "successful_region_ids": [],
                "failed_region_ids": [],
                "driver_result": None,
                "elapsed_us": 0.0,
            }

        required_slot_bytes = 0
        for item in valid_items:
            required_slot_bytes = max(
                required_slot_bytes,
                _align_up(int(item.bytes), page_size),
            )

        current_layer_id = valid_items[0].layer_id
        bank_slot_ids = self._bank_slot_ids_for_layer(current_layer_id)
        bank_reuse_result: dict[str, Any] | None = None

        with self._lock:
            slot_check_start_ns = time.perf_counter_ns()
            _append_debug_trace({
                "action": "reserved_debug_preallocated_slot_check_enter",
                "layer_name": layer_name,
                "step": int(step),
                "required_slot_bytes": required_slot_bytes,
                "slot_count": self.slot_count,
                "pool_id": self.pool_id,
            })
            slot_check = self._ensure_preallocated_locked(required_slot_bytes)
            _append_debug_trace({
                "action": "reserved_debug_preallocated_slot_check_exit",
                "layer_name": layer_name,
                "step": int(step),
                "ok": slot_check.get("ok") is True,
                "reason": slot_check.get("reason"),
                "slot_bytes": self.slot_bytes,
                "elapsed_us": (time.perf_counter_ns() - slot_check_start_ns) / 1000.0,
            })
            if slot_check.get("ok") is not True:
                return {
                    "ok": False,
                    "status": "preallocated_slot_check_failed",
                    "executor": "expert_reserved_regions_async",
                    "success_count": 0,
                    "failure_count": len(valid_items),
                    "assigned_slots": [],
                    "successful_region_ids": [],
                    "failed_region_ids": [item.region_id for item in valid_items],
                    "driver_result": None,
                    "prealloc_result": slot_check,
                    "preallocated_slot_check": slot_check,
                    "elapsed_us": (time.perf_counter_ns() - start_ns) / 1000.0,
                }

            if self._layer_mod_bank_enabled():
                if not bank_slot_ids:
                    return {
                        "ok": False,
                        "status": "invalid_bank_config",
                        "executor": "expert_reserved_regions_async",
                        "success_count": 0,
                        "failure_count": len(valid_items),
                        "assigned_slots": [],
                        "successful_region_ids": [],
                        "failed_region_ids": [item.region_id for item in valid_items],
                        "driver_result": None,
                        "slot_policy": self.slot_policy,
                        "bank_count": self.bank_count,
                        "bank_slot_ids": bank_slot_ids,
                        "elapsed_us": (
                            time.perf_counter_ns() - start_ns
                        ) / 1000.0,
                    }
                bank_slots = [
                    self._slots[slot_id]
                    for slot_id in bank_slot_ids
                    if slot_id in self._slots
                ]
                bank_reuse_result = self._invalidate_slots_for_reuse_locked(
                    slots=bank_slots,
                    layer_name=layer_name,
                    step=int(step),
                    reason="layer_mod_bank_reuse",
                )
                if bank_reuse_result.get("ok") is not True:
                    return {
                        "ok": False,
                        "status": "bank_reuse_invalidate_failed",
                        "executor": "expert_reserved_regions_async",
                        "success_count": 0,
                        "failure_count": len(valid_items),
                        "assigned_slots": [],
                        "successful_region_ids": [],
                        "failed_region_ids": [item.region_id for item in valid_items],
                        "driver_result": bank_reuse_result.get("driver_result"),
                        "bank_reuse_result": bank_reuse_result,
                        "slot_policy": self.slot_policy,
                        "bank_count": self.bank_count,
                        "bank_slot_ids": bank_slot_ids,
                        "elapsed_us": (
                            time.perf_counter_ns() - start_ns
                        ) / 1000.0,
                    }

            used_slots: set[int] = set()
            assignments: list[dict[str, Any]] = []
            slot_records: list[dict[str, Any]] = []
            source_ranges: list[tuple[int, int]] = []
            wait_count = 0
            register_failures: list[dict[str, Any]] = []

            for item in valid_items:
                _append_debug_trace({
                    "action": "reserved_debug_assign_item_enter",
                    "layer_name": layer_name,
                    "step": int(step),
                    "region_id": item.region_id,
                    "driver_region_id": item.driver_region_id,
                    "bytes": item.bytes,
                    "expert_id": item.global_expert_id,
                    "role": item.role,
                    "slot_policy": self.slot_policy,
                    "bank_slot_ids": bank_slot_ids,
                })
                slot = self._choose_slot_locked(used_slots, bank_slot_ids)
                if slot is None:
                    register_failures.append({
                        "region_id": item.region_id,
                        "reason": "no_free_slot",
                        "slot_policy": self.slot_policy,
                        "bank_slot_ids": bank_slot_ids,
                    })
                    continue
                waited = self._wait_for_slot_locked(slot)
                wait_count += 1 if waited else 0
                used_slots.add(slot.slot_id)
                self._free_cursor = (slot.slot_id + 1) % self.slot_count

                register_status = self._ensure_driver_region(item)
                _append_debug_trace({
                    "action": "reserved_debug_register_region_exit",
                    "layer_name": layer_name,
                    "step": int(step),
                    "region_id": item.region_id,
                    "driver_region_id": item.driver_region_id,
                    "slot_id": slot.slot_id,
                    "register_status": register_status,
                })
                if register_status < 0:
                    register_failures.append({
                        "region_id": item.region_id,
                        "driver_region_id": item.driver_region_id,
                        "reason": "driver_register_failed",
                        "status": register_status,
                    })
                    continue
                assignments.append(
                    {
                        "region_id": int(item.driver_region_id),
                        "pool_id": self.pool_id,
                        "slot_id": slot.slot_id,
                        "slot_offset": 0,
                        "bytes_override": int(item.bytes),
                        "strict_footprint": self.strict_footprint,
                    }
                )
                source_ranges.append((int(item.ptr), int(item.bytes)))
                slot.epoch += 1
                slot.owner_regions = [item]
                slot.owner_layer_name = layer_name
                slot.owner_step = int(step)
                slot.owner_layer_id = item.layer_id
                slot.owner_expert_id = item.global_expert_id
                slot.state = "LOADING"
                slot_records.append(
                    {
                        "slot_id": slot.slot_id,
                        "layer_id": item.layer_id,
                        "expert_id": item.global_expert_id,
                        "role": item.role,
                        "region_id": item.region_id,
                        "region_count": 1,
                        "bytes": int(item.bytes),
                        "slot_epoch": slot.epoch,
                    }
                )

        if not assignments:
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            _append_debug_trace({
                "action": "reserved_debug_prefetch_no_assignments",
                "layer_name": layer_name,
                "step": int(step),
                "elapsed_us": elapsed_us,
                "register_failures": register_failures,
            })
            return {
                "ok": False,
                "status": "no_assignments",
                "executor": "expert_reserved_regions_async",
                "success_count": 0,
                "failure_count": len(valid_items),
                "assigned_slots": slot_records,
                "successful_region_ids": [],
                "failed_region_ids": [item.region_id for item in valid_items],
                "register_failures": register_failures,
                "driver_result": None,
                "elapsed_us": elapsed_us,
            }

        source_offload_result: dict[str, Any] | None = None
        if self.prefetch_cpu_source:
            source_start_ns = time.perf_counter_ns()
            _append_debug_trace({
                "action": "reserved_debug_source_offload_enter",
                "layer_name": layer_name,
                "step": int(step),
                "range_count": len(source_ranges),
                "bytes": sum(size for _, size in source_ranges),
            })
            try:
                (
                    source_successes,
                    source_executor,
                    source_advise_result,
                    source_prefetch_result,
                ) = batch_offload_ranges_to_cpu_with_executor(
                    source_ranges,
                    sync=True,
                    executor="cpp_batch",
                )
                if int(source_successes) != len(source_ranges):
                    source_prefetch_result = batch_prefetch_ranges_to_cpu(
                        source_ranges,
                        allow_tag_only=True,
                    )
                    source_successes = int(source_prefetch_result)
                    source_executor = "cpp_batch_tag_only_bypass"
                    source_advise_result = 0
                    if int(source_successes) > 0:
                        try:
                            torch.cuda.synchronize(self.device)
                        except Exception:
                            logger.debug(
                                "Failed to synchronize after reserved source "
                                "CPU prefetch",
                                exc_info=True,
                            )
                source_offload_result = {
                    "enabled": True,
                    "successes": int(source_successes),
                    "executor": source_executor,
                    "advise_result": int(source_advise_result),
                    "prefetch_result": int(source_prefetch_result),
                    "range_count": len(source_ranges),
                }
                _append_debug_trace({
                    "action": "reserved_debug_source_offload_exit",
                    "layer_name": layer_name,
                    "step": int(step),
                    "ok": int(source_successes) == len(source_ranges),
                    "successes": int(source_successes),
                    "advise_result": int(source_advise_result),
                    "prefetch_result": int(source_prefetch_result),
                    "elapsed_us": (
                        time.perf_counter_ns() - source_start_ns
                    ) / 1000.0,
                })
            except Exception as exc:
                logger.debug(
                    "Failed to CPU-offload reserved expert source ranges",
                    exc_info=True,
                )
                source_offload_result = {
                    "enabled": True,
                    "successes": 0,
                    "executor": "driver_batch",
                    "error": repr(exc),
                    "range_count": len(source_ranges),
                }
                _append_debug_trace({
                    "action": "reserved_debug_source_offload_exception",
                    "layer_name": layer_name,
                    "step": int(step),
                    "error": repr(exc),
                    "elapsed_us": (
                        time.perf_counter_ns() - source_start_ns
                    ) / 1000.0,
                })
        else:
            source_offload_result = {
                "enabled": False,
                "successes": 0,
                "range_count": len(source_ranges),
            }

        ioctl_start_ns = time.perf_counter_ns()
        _append_debug_trace({
            "action": "reserved_debug_prefetch_ioctl_enter",
            "layer_name": layer_name,
            "step": int(step),
            "assignment_count": len(assignments),
            "assigned_slots": [record["slot_id"] for record in slot_records],
            "requested_bytes": sum(int(item["bytes_override"]) for item in assignments),
            "mode": "reserved_prefetch",
        })
        driver_result = driver_expert_reserved_prefetch_regions(
            assignments=assignments,
            device=self.device,
            async_=self.async_,
            strict_footprint=self.strict_footprint,
            mode="reserved_prefetch",
            prefer_existing=True,
            return_entries=True,
        )
        _append_debug_trace({
            "action": "reserved_debug_prefetch_ioctl_exit",
            "layer_name": layer_name,
            "step": int(step),
            "ok": driver_result.get("ok") is True,
            "rm_status": driver_result.get("rm_status"),
            "success_count": driver_result.get("success_count"),
            "failure_count": driver_result.get("failure_count"),
            "accepted_bytes": driver_result.get("accepted_bytes"),
            "elapsed_us": (time.perf_counter_ns() - ioctl_start_ns) / 1000.0,
        })

        if self.async_ and self.sync_before_compute and driver_result.get("ok") is True:
            try:
                sync_start_ns = time.perf_counter_ns()
                _append_debug_trace({
                    "action": "reserved_debug_sync_before_compute_enter",
                    "layer_name": layer_name,
                    "step": int(step),
                    "device": self.device,
                })
                torch.cuda.synchronize(self.device)
                _append_debug_trace({
                    "action": "reserved_debug_sync_before_compute_exit",
                    "layer_name": layer_name,
                    "step": int(step),
                    "device": self.device,
                    "elapsed_us": (
                        time.perf_counter_ns() - sync_start_ns
                    ) / 1000.0,
                })
            except Exception:
                logger.debug(
                    "Failed to synchronize after reserved expert prefetch",
                    exc_info=True,
                )

        successful_driver_ids: set[int] = set()
        failed_driver_ids: set[int] = set()
        if driver_result.get("compact_result") and driver_result.get("ok") is True:
            successful_driver_ids = {
                int(assignment["region_id"]) for assignment in assignments
            }
        else:
            for entry in driver_result.get("entries") or []:
                try:
                    driver_id = int(entry.get("region_id", 0))
                    status = int(entry.get("status", -1))
                except (TypeError, ValueError, AttributeError):
                    continue
                if status == 0:
                    successful_driver_ids.add(driver_id)
                else:
                    failed_driver_ids.add(driver_id)

        by_driver_id = {int(item.driver_region_id): item for item in valid_items}
        successful_region_ids = [
            by_driver_id[driver_id].region_id
            for driver_id in successful_driver_ids
            if driver_id in by_driver_id
        ]
        failed_region_ids = [
            by_driver_id[driver_id].region_id
            for driver_id in failed_driver_ids
            if driver_id in by_driver_id
        ]
        if driver_result.get("ok") is not True and not failed_region_ids:
            failed_region_ids = [
                item.region_id
                for item in valid_items
                if item.region_id not in successful_region_ids
            ]
        if register_failures:
            failed_region_ids.extend(str(item["region_id"]) for item in register_failures)

        layer_key = (str(layer_name), int(step))
        with self._lock:
            active_slots = self._active_layer_slots.setdefault(layer_key, set())
            for record in slot_records:
                slot_id = int(record["slot_id"])
                slot = self._slots.get(slot_id)
                if slot is None:
                    continue
                if driver_result.get("ok") is True:
                    slot.state = "IN_USE"
                    active_slots.add(slot_id)
                else:
                    slot.state = "READY"

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        _append_debug_trace({
            "action": "reserved_debug_prefetch_exit",
            "layer_name": layer_name,
            "step": int(step),
            "ok": driver_result.get("ok") is True and not register_failures,
            "success_count": len(successful_region_ids),
            "failure_count": len(failed_region_ids),
            "elapsed_us": elapsed_us,
            "successful_region_ids": successful_region_ids,
            "failed_region_ids": failed_region_ids,
        })
        return {
            "ok": driver_result.get("ok") is True and not register_failures,
            "status": "ok" if driver_result.get("ok") is True else "driver_failed",
            "executor": "expert_reserved_regions_async",
            "success_count": len(successful_region_ids),
            "failure_count": len(failed_region_ids),
            "assigned_slots": slot_records,
            "successful_region_ids": successful_region_ids,
            "failed_region_ids": failed_region_ids,
            "register_failures": register_failures,
            "driver_result": driver_result,
            "source_offload_result": source_offload_result,
            "elapsed_us": elapsed_us,
            "wait_on_reuse_count": wait_count,
            "slot_count": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "pool_id": self.pool_id,
            "async": self.async_,
            "sync_before_compute": self.sync_before_compute,
            "prefetch_cpu_source": self.prefetch_cpu_source,
            "slot_policy": self.slot_policy,
            "bank_count": self.bank_count,
            "bank_slot_ids": bank_slot_ids,
            "bank_reuse_result": bank_reuse_result,
        }

    def cleanup_layer(
        self,
        *,
        layer_name: str,
        step: int,
        record_event: bool = True,
        invalidate: bool = True,
    ) -> dict[str, Any]:
        start_ns = time.perf_counter_ns()
        layer_key = (str(layer_name), int(step))
        _append_debug_trace({
            "action": "reserved_debug_cleanup_enter",
            "layer_name": layer_name,
            "step": int(step),
            "record_event": bool(record_event),
            "invalidate": bool(invalidate),
            "cleanup_synchronize": self.cleanup_synchronize,
        })
        with self._lock:
            slot_ids = sorted(self._active_layer_slots.pop(layer_key, set()))
            slots = [self._slots[slot_id] for slot_id in slot_ids if slot_id in self._slots]

        if not slots:
            _append_debug_trace({
                "action": "reserved_debug_cleanup_no_active_slots",
                "layer_name": layer_name,
                "step": int(step),
                "elapsed_us": (time.perf_counter_ns() - start_ns) / 1000.0,
            })
            return {
                "ok": True,
                "status": "no_active_slots",
                "invalidated_slots": 0,
                "regions": [],
                "driver_result": None,
                "elapsed_us": 0.0,
            }

        event: torch.cuda.Event | None = None
        if record_event:
            event_start_ns = time.perf_counter_ns()
            _append_debug_trace({
                "action": "reserved_debug_cleanup_event_record_enter",
                "layer_name": layer_name,
                "step": int(step),
                "slot_ids": slot_ids,
            })
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            _append_debug_trace({
                "action": "reserved_debug_cleanup_event_record_exit",
                "layer_name": layer_name,
                "step": int(step),
                "elapsed_us": (time.perf_counter_ns() - event_start_ns) / 1000.0,
            })
            if self.cleanup_synchronize or invalidate:
                sync_start_ns = time.perf_counter_ns()
                _append_debug_trace({
                    "action": "reserved_debug_cleanup_event_sync_enter",
                    "layer_name": layer_name,
                    "step": int(step),
                    "slot_ids": slot_ids,
                    "invalidate_after_sync": bool(invalidate),
                })
                event.synchronize()
                _append_debug_trace({
                    "action": "reserved_debug_cleanup_event_sync_exit",
                    "layer_name": layer_name,
                    "step": int(step),
                    "elapsed_us": (time.perf_counter_ns() - sync_start_ns) / 1000.0,
                })
                if self.cleanup_synchronize:
                    device_sync_start_ns = time.perf_counter_ns()
                    _append_debug_trace({
                        "action": "reserved_debug_cleanup_device_sync_enter",
                        "layer_name": layer_name,
                        "step": int(step),
                        "slot_ids": slot_ids,
                        "device": self.device,
                    })
                    torch.cuda.synchronize(self.device)
                    _append_debug_trace({
                        "action": "reserved_debug_cleanup_device_sync_exit",
                        "layer_name": layer_name,
                        "step": int(step),
                        "device": self.device,
                        "elapsed_us": (
                            time.perf_counter_ns() - device_sync_start_ns
                        ) / 1000.0,
                    })

        if self._layer_mod_bank_enabled():
            pending_regions: list[dict[str, Any]] = []
            with self._lock:
                for slot in slots:
                    for item in slot.owner_regions:
                        pending_regions.append(
                            {
                                "region_id": item.region_id,
                                "driver_region_id": item.driver_region_id,
                                "layer_id": item.layer_id,
                                "global_expert_id": item.global_expert_id,
                                "local_expert_id": item.local_expert_id,
                                "role": item.role,
                                "bytes": item.bytes,
                                "slot_id": slot.slot_id,
                            }
                        )
                    slot.state = (
                        "PENDING_REUSE"
                        if event is not None or self.cleanup_synchronize
                        else "IN_USE"
                    )
                    slot.last_event = None if self.cleanup_synchronize else event
            elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
            _append_debug_trace({
                "action": "reserved_debug_cleanup_defer_invalidate_exit",
                "layer_name": layer_name,
                "step": int(step),
                "slot_ids": slot_ids,
                "pending_region_count": len(pending_regions),
                "elapsed_us": elapsed_us,
                "slot_policy": self.slot_policy,
            })
            return {
                "ok": True,
                "status": "deferred_until_bank_reuse",
                "invalidated_slots": 0,
                "regions": [],
                "pending_regions": pending_regions,
                "driver_result": None,
                "elapsed_us": elapsed_us,
                "cleanup_synchronize": self.cleanup_synchronize,
                "invalidate": False,
                "slot_policy": self.slot_policy,
            }

        assignments = [
            {
                "pool_id": self.pool_id,
                "slot_id": slot.slot_id,
                "strict_footprint": self.strict_footprint,
            }
            for slot in slots
        ]
        driver_result: dict[str, Any] | None = None
        if invalidate:
            ioctl_start_ns = time.perf_counter_ns()
            _append_debug_trace({
                "action": "reserved_debug_cleanup_invalidate_ioctl_enter",
                "layer_name": layer_name,
                "step": int(step),
                "slot_ids": slot_ids,
                "assignment_count": len(assignments),
            })
            driver_result = driver_expert_reserved_prefetch_regions(
                assignments=assignments,
                device=self.device,
                async_=False,
                strict_footprint=self.strict_footprint,
                mode="invalidate",
                prefer_existing=True,
                return_entries=False,
            )
            _append_debug_trace({
                "action": "reserved_debug_cleanup_invalidate_ioctl_exit",
                "layer_name": layer_name,
                "step": int(step),
                "ok": driver_result.get("ok") is True,
                "rm_status": driver_result.get("rm_status"),
                "success_count": driver_result.get("success_count"),
                "failure_count": driver_result.get("failure_count"),
                "invalidated_slots": driver_result.get("invalidated_slots"),
                "elapsed_us": (time.perf_counter_ns() - ioctl_start_ns) / 1000.0,
            })

        invalidated_regions: list[dict[str, Any]] = []
        with self._lock:
            for slot in slots:
                for item in slot.owner_regions:
                    invalidated_regions.append(
                        {
                            "region_id": item.region_id,
                            "driver_region_id": item.driver_region_id,
                            "layer_id": item.layer_id,
                            "global_expert_id": item.global_expert_id,
                            "local_expert_id": item.local_expert_id,
                            "role": item.role,
                            "bytes": item.bytes,
                            "slot_id": slot.slot_id,
                        }
                    )
                    mark_uvm_residency_state(
                        region_id=item.region_id,
                        state="cpu",
                        metadata={
                            **item.metadata,
                            "executor": "expert_reserved_regions_async",
                            "reserved_cleanup_step": int(step),
                            "reserved_slot_id": slot.slot_id,
                        },
                    )
                slot.owner_regions = []
                slot.owner_layer_name = ""
                slot.owner_step = -1
                slot.owner_layer_id = None
                slot.owner_expert_id = None
                slot.state = "READY"
                slot.last_event = None if self.cleanup_synchronize else event

        ok = True
        status = "ok"
        if driver_result is not None and driver_result.get("ok") is not True:
            ok = False
            status = "invalidate_failed"
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1000.0
        _append_debug_trace({
            "action": "reserved_debug_cleanup_exit",
            "layer_name": layer_name,
            "step": int(step),
            "ok": ok,
            "status": status,
            "invalidated_slots": len(slots),
            "region_count": len(invalidated_regions),
            "elapsed_us": elapsed_us,
        })
        return {
            "ok": ok,
            "status": status,
            "invalidated_slots": len(slots),
            "regions": invalidated_regions,
            "driver_result": driver_result,
            "elapsed_us": elapsed_us,
            "cleanup_synchronize": self.cleanup_synchronize,
            "invalidate": invalidate,
        }


_EXECUTOR_LOCK = threading.Lock()
_EXECUTORS: dict[tuple[int, int, int, int], ExpertReservedRegionsExecutor] = {}


def get_expert_reserved_regions_executor(
    *,
    device: int,
    pool_id: int | None = None,
    slot_count: int | None = None,
    slot_bytes: int | None = None,
    strict_footprint: bool | None = None,
    async_: bool | None = None,
    wait_on_reuse: bool | None = None,
    sync_before_compute: bool | None = None,
    cleanup_synchronize: bool | None = None,
) -> ExpertReservedRegionsExecutor:
    resolved_pool_id = (
        _env_int("VLLM_UVM_EXPERT_RESERVED_POOL_ID", 0)
        if pool_id is None else int(pool_id)
    )
    resolved_slot_count = (
        _env_int("VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT", 6)
        if slot_count is None else int(slot_count)
    )
    resolved_slot_bytes = (
        _env_int("VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES", 0)
        if slot_bytes is None else int(slot_bytes)
    )
    resolved_strict = (
        _env_flag("VLLM_UVM_EXPERT_RESERVED_STRICT_FOOTPRINT", True)
        if strict_footprint is None else bool(strict_footprint)
    )
    resolved_async = (
        _env_flag("VLLM_UVM_EXPERT_RESERVED_ASYNC", False)
        if async_ is None else bool(async_)
    )
    resolved_wait_on_reuse = (
        _env_flag("VLLM_UVM_EXPERT_RESERVED_WAIT_ON_REUSE", True)
        if wait_on_reuse is None else bool(wait_on_reuse)
    )
    resolved_sync_before_compute = (
        _env_flag("VLLM_UVM_EXPERT_RESERVED_SYNC_BEFORE_COMPUTE", True)
        if sync_before_compute is None else bool(sync_before_compute)
    )
    resolved_cleanup_sync = (
        _env_flag("VLLM_UVM_EXPERT_RESERVED_CLEANUP_SYNCHRONIZE", True)
        if cleanup_synchronize is None else bool(cleanup_synchronize)
    )
    resolved_prefetch_cpu_source = _env_flag(
        "VLLM_UVM_EXPERT_RESERVED_PREFETCH_CPU_SOURCE",
        True,
    )

    key = (
        int(device),
        resolved_pool_id,
        max(resolved_slot_count, 0),
        max(resolved_slot_bytes, 0),
    )
    with _EXECUTOR_LOCK:
        executor = _EXECUTORS.get(key)
        if executor is None:
            executor = ExpertReservedRegionsExecutor(
                device=int(device),
                pool_id=resolved_pool_id,
                slot_count=resolved_slot_count,
                slot_bytes=resolved_slot_bytes,
                strict_footprint=resolved_strict,
                async_=resolved_async,
                wait_on_reuse=resolved_wait_on_reuse,
                sync_before_compute=resolved_sync_before_compute,
                cleanup_synchronize=resolved_cleanup_sync,
                prefetch_cpu_source=resolved_prefetch_cpu_source,
            )
            _EXECUTORS[key] = executor
        return executor
