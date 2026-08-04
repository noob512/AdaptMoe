# SPDX-License-Identifier: Apache-2.0
"""Stage L0 UVM pool residency accounting.

This module maintains an application-level estimate of GPU residency intent for
weights, KV, and scratch pools.  When explicitly enabled, it can also sample
page-granular real UVM residency from inside the vLLM process.  It does not
migrate, evict, or prefetch memory by itself.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.device_allocator.uvm import (
    driver_batch_query_ranges_residency_bytes,
    driver_get_pool_state,
    driver_reclaim_pool,
    driver_register_region,
    driver_set_pool_budget,
    driver_unregister_region,
    driver_update_region_state,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


VALID_POOLS = {"weights", "kv", "scratch"}
VALID_STATES = {
    "unknown",
    "gpu",
    "cpu",
    "prefetching_gpu",
    "prefetching_cpu",
    "released",
}

DRIVER_POOL_IDS = {
    "unknown": 0,
    "kv": 1,
    "weights": 2,
    "scratch": 3,
}

DRIVER_ROLE_IDS = {
    "unknown": 0,
    "kv_block": 1,
    "dense_weight": 2,
    "moe_expert_w13": 3,
    "moe_expert_w2": 4,
    "scratch": 5,
    "workspace": 6,
}

DRIVER_FLAG_EVICTABLE = 1 << 0
DRIVER_FLAG_PROTECTED = 1 << 1
DRIVER_FLAG_ACTIVE = 1 << 2
DRIVER_FLAG_PREFETCHABLE = 1 << 3
DRIVER_FLAG_PINNED = 1 << 4
DRIVER_FLAG_TRANSIENT = 1 << 5
DRIVER_FLAG_HOT_SUBPOOL = 1 << 11
DRIVER_FLAG_STREAMING_SUBPOOL = 1 << 12
DRIVER_FLAG_COLD_SUBPOOL = 1 << 13
DRIVER_FLAG_CPU_OFFLOADED_RESERVED = 1 << 14
DRIVER_FLAG_EXPERT_ENVELOPE = 1 << 15
DRIVER_FLAG_MASK = (
    DRIVER_FLAG_EVICTABLE
    | DRIVER_FLAG_PROTECTED
    | DRIVER_FLAG_ACTIVE
    | DRIVER_FLAG_PREFETCHABLE
    | DRIVER_FLAG_PINNED
    | DRIVER_FLAG_TRANSIENT
    | DRIVER_FLAG_HOT_SUBPOOL
    | DRIVER_FLAG_STREAMING_SUBPOOL
    | DRIVER_FLAG_COLD_SUBPOOL
    | DRIVER_FLAG_CPU_OFFLOADED_RESERVED
    | DRIVER_FLAG_EXPERT_ENVELOPE
)

DRIVER_REGION_STATES = {
    "unknown": 0,
    "cpu": 1,
    "released": 1,
    "prefetching_cpu": 2,
    "gpu": 3,
    "prefetching_gpu": 3,
    "active": 4,
}

DRIVER_BUDGET_MODE_IDS = {
    "trace_only": 0,
    "trace": 0,
    "soft": 1,
    "enforce": 2,
}

DEFAULT_DRIVER_TEMP_BUFFER_BYTES = 100 * 1024 * 1024


def _empty_real_residency_totals() -> dict[str, int]:
    return {
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
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
        logger.warning(
            "Ignoring invalid %s=%r; expected non-negative integer.",
            name,
            raw,
        )
        return default


def _read_driver_budget_mode_env() -> int:
    raw = os.environ.get("VLLM_UVM_DRIVER_POOL_BUDGET_MODE", "trace_only")
    mode = raw.strip().lower()
    if mode not in DRIVER_BUDGET_MODE_IDS:
        logger.warning(
            "Ignoring invalid VLLM_UVM_DRIVER_POOL_BUDGET_MODE=%r; using trace_only.",
            raw,
        )
        return DRIVER_BUDGET_MODE_IDS["trace_only"]
    return DRIVER_BUDGET_MODE_IDS[mode]


def _parse_pool_budget_overrides(raw: str | None) -> dict[str, int]:
    budgets: dict[str, int] = {}
    if not raw:
        return budgets
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        pool_name, value = item.split("=", 1)
        pool = _normalize_driver_pool_name(pool_name.strip())
        if pool == "unknown":
            continue
        try:
            budgets[pool] = max(int(value.strip()), 0)
        except ValueError:
            logger.warning(
                "Ignoring invalid pool budget override %r; expected bytes.",
                item,
            )
    return budgets


def _read_mode_env() -> str:
    mode = os.environ.get("VLLM_UVM_RESIDENCY_MODE", "trace_only")
    mode = mode.strip().lower()
    if mode not in ("trace_only", "enforce"):
        logger.warning(
            "Ignoring invalid VLLM_UVM_RESIDENCY_MODE=%r; using trace_only.",
            mode,
        )
        return "trace_only"
    return mode


def _normalize_pool(pool: str) -> str:
    pool = (pool or "unknown").strip().lower()
    return pool if pool in VALID_POOLS else "unknown"


def _normalize_driver_pool_name(pool: str) -> str:
    pool = (pool or "unknown").strip().lower()
    if pool in ("temp", "temporary", "buffer", "buffers", "workspace"):
        return "scratch"
    if pool == "weight":
        return "weights"
    if pool in ("kvcache", "kv_cache", "kv-cache"):
        return "kv"
    return _normalize_pool(pool)


def _normalize_state(state: str) -> str:
    state = (state or "unknown").strip().lower()
    return state if state in VALID_STATES else "unknown"


def _stable_driver_region_id(region_id: str) -> int:
    text = str(region_id or "")
    if not text:
        return 0
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    # Keep the key positive for radix-tree users and reserve 0 as invalid.
    return (int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF) or 1


def _metadata_int(metadata: dict[str, Any], key: str, default: int = 0) -> int:
    value = metadata.get(key, default)
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError):
        return default


def _driver_role_id(pool: str, metadata: dict[str, Any]) -> int:
    explicit = metadata.get("driver_role_id")
    if explicit is not None:
        return _metadata_int(metadata, "driver_role_id", 0)
    role = str(metadata.get("role") or "").strip().lower()
    source = str(metadata.get("source") or "").strip().lower()
    if pool == "kv":
        return DRIVER_ROLE_IDS["kv_block"]
    if pool == "scratch":
        if "workspace" in role or "workspace" in source:
            return DRIVER_ROLE_IDS["workspace"]
        return DRIVER_ROLE_IDS["scratch"]
    if pool == "weights":
        if role in ("moe_gate_up", "w13", "w13_weight", "moe_expert_w13"):
            return DRIVER_ROLE_IDS["moe_expert_w13"]
        if role in ("moe_down", "w2", "w2_weight", "moe_expert_w2"):
            return DRIVER_ROLE_IDS["moe_expert_w2"]
        return DRIVER_ROLE_IDS["dense_weight"]
    return DRIVER_ROLE_IDS["unknown"]


def _driver_flags(
    *,
    pool: str,
    state: str,
    evictable: bool,
    active: bool,
    metadata: dict[str, Any],
) -> int:
    flags = _metadata_int(metadata, "driver_flags", 0)
    if evictable:
        flags |= DRIVER_FLAG_EVICTABLE
    if active:
        flags |= DRIVER_FLAG_ACTIVE | DRIVER_FLAG_PROTECTED
    if bool(metadata.get("protected")):
        flags |= DRIVER_FLAG_PROTECTED
    if bool(metadata.get("prefetchable")):
        flags |= DRIVER_FLAG_PREFETCHABLE
    if bool(metadata.get("pinned")):
        flags |= DRIVER_FLAG_PINNED
    if pool == "scratch" or bool(metadata.get("transient")):
        flags |= DRIVER_FLAG_TRANSIENT
    pool_class = str(
        metadata.get("pool_class")
        or metadata.get("driver_subpool")
        or metadata.get("subpool")
        or ""
    ).strip().lower()
    if pool_class == "hot" or bool(metadata.get("init_hot")):
        flags |= (
            DRIVER_FLAG_HOT_SUBPOOL
            | DRIVER_FLAG_PROTECTED
            | DRIVER_FLAG_PINNED
            | DRIVER_FLAG_PREFETCHABLE
        )
    if pool_class == "streaming" or bool(metadata.get("streaming")):
        flags |= (
            DRIVER_FLAG_STREAMING_SUBPOOL
            | DRIVER_FLAG_EVICTABLE
            | DRIVER_FLAG_PREFETCHABLE
            | DRIVER_FLAG_TRANSIENT
        )
    if (
        pool_class in ("cold", "init_cold", "cold_init")
        or bool(metadata.get("init_cold"))
    ):
        flags |= DRIVER_FLAG_COLD_SUBPOOL | DRIVER_FLAG_EVICTABLE
    if (
        bool(metadata.get("cpu_offloaded_reserved"))
        or str(metadata.get("placement") or "").strip().upper()
        == "CPU_OFFLOADED_RESERVED"
    ):
        flags |= (
            DRIVER_FLAG_CPU_OFFLOADED_RESERVED
            | DRIVER_FLAG_PREFETCHABLE
            | DRIVER_FLAG_PROTECTED
        )
    if bool(metadata.get("expert_envelope")):
        flags |= DRIVER_FLAG_EXPERT_ENVELOPE
    if state == "gpu" and pool == "weights" and not evictable:
        flags |= DRIVER_FLAG_PROTECTED
    return flags


def _driver_region_state_id(
    state: str,
    *,
    active: bool = False,
) -> int:
    if active:
        return DRIVER_REGION_STATES["active"]
    return DRIVER_REGION_STATES.get(_normalize_state(state), 0)


def _state_counts_as_gpu(state: str, previous_estimated: bool = False) -> bool:
    state = _normalize_state(state)
    if state in ("gpu", "prefetching_gpu"):
        return True
    if state == "prefetching_cpu":
        # Conservative accounting: a GPU -> CPU prefetch still counts until it
        # is confirmed as CPU.  This avoids releasing budget too early.
        return previous_estimated
    return False


@dataclass
class UvmResidencyRegion:
    region_id: str
    pool: str
    ptr: int | None
    bytes: int
    driver_region_id: int = 0
    driver_registered: bool = False
    driver_register_status: int | None = None
    state: str = "unknown"
    estimated_gpu_resident: bool = False
    evictable: bool = False
    active: bool = False
    last_access_step: int | None = None
    last_state_update_ts: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UvmResidencyDecision:
    enabled: bool
    mode: str
    pool: str
    region_id: str
    requested_bytes: int
    incremental_bytes: int
    allowed: bool
    would_deny: bool
    reason: str
    scope_key: str
    pool_budget_bytes: int
    pool_estimated_gpu_resident_bytes: int
    pool_remaining_bytes: int | None
    global_budget_bytes: int
    global_estimated_gpu_resident_bytes: int
    global_remaining_bytes: int | None
    victims: tuple[str, ...] = ()


class UvmPoolResidencyManager:
    """Stage L0 estimated residency ledger for UVM memory pools."""

    def __init__(self) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_RESIDENCY_ENABLE", False)
        self.mode = _read_mode_env()
        self.tag_only_mode = _read_bool_env("VLLM_UVM_RESIDENCY_TAG_ONLY", False)
        self.driver_tag_enable = _read_bool_env(
            "VLLM_UVM_DRIVER_TAG_ENABLE",
            self.tag_only_mode,
        )
        self.global_budget_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES", 0
        )
        self.pool_budgets = {
            "weights": _read_int_env("VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES", 0),
            "kv": _read_int_env("VLLM_UVM_RESIDENCY_KV_GPU_BYTES", 0),
            "scratch": _read_int_env("VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES", 0),
        }
        self.driver_pool_budget_enable = _read_bool_env(
            "VLLM_UVM_DRIVER_POOL_BUDGET_ENABLE",
            self.driver_tag_enable,
        )
        self.driver_pool_budget_mode = _read_driver_budget_mode_env()
        self.driver_pool_budget_overrides = _parse_pool_budget_overrides(
            os.environ.get("VLLM_UVM_DRIVER_POOL_BUDGETS")
        )
        self.driver_temp_buffer_bytes = _read_int_env(
            "VLLM_UVM_DRIVER_TEMP_BUFFER_BYTES",
            DEFAULT_DRIVER_TEMP_BUFFER_BYTES,
        )
        self._driver_pool_budget_status: dict[str, int] = {}
        self._driver_pool_budgets_applied = False
        self.safety_margin_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES", 0
        )
        self.real_residency_enable = _read_bool_env(
            "VLLM_UVM_REAL_RESIDENCY_ENABLE", False
        )
        self.real_residency_device = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_DEVICE", 0
        )
        self.real_residency_max_regions_per_pool = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL", 0
        )
        self._lock = threading.Lock()
        self._regions: dict[str, UvmResidencyRegion] = {}
        self._pool_estimated_bytes = {"weights": 0, "kv": 0, "scratch": 0}

    @property
    def active(self) -> bool:
        return self.enabled

    def sync_driver_pool_budgets(self) -> dict[str, int]:
        with self._lock:
            self._sync_driver_pool_budgets_locked(force=True)
            return dict(self._driver_pool_budget_status)

    def set_driver_pool_budget(
        self,
        pool: str,
        budget_bytes: int,
        *,
        gpu_id: int = 0,
        mode: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pool_name = _normalize_driver_pool_name(pool)
        pool_id = DRIVER_POOL_IDS.get(pool_name)
        budget = max(int(budget_bytes or 0), 0)
        if pool_id is None or pool_name == "unknown":
            return {
                "ok": False,
                "pool": pool_name,
                "budget_bytes": budget,
                "status": -2,
                "reason": "unknown_pool",
            }
        budget_mode = (
            self.driver_pool_budget_mode if mode is None else max(int(mode), 0)
        )

        status = driver_set_pool_budget(
            pool=pool_id,
            budget_bytes=budget,
            gpu_id=max(int(gpu_id or 0), 0),
            mode=budget_mode,
        )
        result = {
            "ok": status >= 0,
            "pool": pool_name,
            "pool_id": pool_id,
            "budget_bytes": budget,
            "gpu_id": max(int(gpu_id or 0), 0),
            "mode": budget_mode,
            "status": status,
            "reason": "updated" if status >= 0 else f"driver_set_failed:{status}",
        }
        with self._lock:
            self.pool_budgets[pool_name] = budget
            self.driver_pool_budget_overrides[pool_name] = budget
            self._driver_pool_budget_status[pool_name] = status
        return result

    def driver_pool_state_snapshot(self, gpu_id: int = 0) -> dict[str, Any]:
        with self._lock:
            self._sync_driver_pool_budgets_locked()

        states: dict[str, Any] = {}
        for pool, pool_id in DRIVER_POOL_IDS.items():
            if pool == "unknown":
                continue
            state = driver_get_pool_state(pool_id, gpu_id=gpu_id)
            if state is None:
                continue
            victim_count = int(state.get("victim_count", 0))
            state["victims"] = [
                {
                    "region_id": int(state.get(f"victim_region_id_{i}", 0)),
                    "bytes": int(state.get(f"victim_bytes_{i}", 0)),
                }
                for i in range(min(max(victim_count, 0), 8))
                if int(state.get(f"victim_region_id_{i}", 0)) != 0
            ]
            states[pool] = state
        return states

    def driver_soft_reclaim_pool(
        self,
        pool: str,
        *,
        gpu_id: int = 0,
        target_bytes: int = 0,
        current_epoch: int = 0,
        cooldown_epochs: int = 0,
        max_victims: int = 8,
        async_reclaim: bool = False,
    ) -> dict[str, Any] | None:
        pool_name = _normalize_driver_pool_name(pool)
        pool_id = DRIVER_POOL_IDS.get(pool_name)
        if pool_id is None or pool_name == "unknown":
            return None

        with self._lock:
            self._sync_driver_pool_budgets_locked()

        result = driver_reclaim_pool(
            pool_id,
            gpu_id=gpu_id,
            target_bytes=target_bytes,
            current_epoch=current_epoch,
            cooldown_epochs=cooldown_epochs,
            max_victims=max_victims,
            async_reclaim=async_reclaim,
        )
        return result

    def register_region(
        self,
        *,
        pool: str,
        region_id: str,
        ptr: int | None,
        bytes: int,
        state: str = "unknown",
        evictable: bool = False,
        active: bool = False,
        driver_register: bool,
        metadata: dict[str, Any] | None = None,
    ) -> UvmResidencyRegion:
        pool = _normalize_pool(pool)
        state = _normalize_state(state)
        bytes = max(int(bytes or 0), 0)
        metadata = metadata or {}
        with self._lock:
            self._sync_driver_pool_budgets_locked()
            region = UvmResidencyRegion(
                region_id=region_id,
                pool=pool,
                ptr=ptr,
                bytes=bytes,
                driver_region_id=_stable_driver_region_id(region_id),
                state=state,
                estimated_gpu_resident=_state_counts_as_gpu(state),
                evictable=evictable,
                active=active,
                metadata=dict(metadata),
            )
            self._regions[region_id] = region

            if (
                region.estimated_gpu_resident
                and pool in self._pool_estimated_bytes
            ):
                self._pool_estimated_bytes[pool] += bytes

            if driver_register and self.driver_tag_enable:
                self._sync_driver_region_tag_locked(region)
            if region.driver_registered:
                self._sync_driver_region_state_locked(region)
            return region

    def request_gpu_residency(
        self,
        *,
        pool: str,
        region_id: str,
        bytes: int,
        scope_key: str | None = None,
        active_region_ids: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UvmResidencyDecision:
        pool = _normalize_pool(pool)
        bytes = max(int(bytes or 0), 0)
        scope_key = scope_key or "global"
        active_region_ids = active_region_ids or set()
        if self.tag_only_mode:
            with self._lock:
                decision = UvmResidencyDecision(
                    enabled=self.enabled,
                    mode="tag_only",
                    pool=pool,
                    region_id=region_id,
                    requested_bytes=bytes,
                    incremental_bytes=0,
                    allowed=True,
                    would_deny=False,
                    reason="tag_only_no_application_residency_decision",
                    scope_key=scope_key,
                    pool_budget_bytes=self.pool_budgets.get(pool, 0),
                    pool_estimated_gpu_resident_bytes=(
                        self._pool_estimated_bytes.get(pool, 0)
                    ),
                    pool_remaining_bytes=None,
                    global_budget_bytes=self.global_budget_bytes,
                    global_estimated_gpu_resident_bytes=(
                        self._global_estimated_bytes_locked()
                    ),
                    global_remaining_bytes=None,
                )
            return decision
        if not self.active or bytes <= 0:
            return UvmResidencyDecision(
                enabled=self.enabled,
                mode=self.mode,
                pool=pool,
                region_id=region_id,
                requested_bytes=bytes,
                incremental_bytes=0,
                allowed=True,
                would_deny=False,
                reason="residency_disabled_or_zero_bytes",
                scope_key=scope_key,
                pool_budget_bytes=0,
                pool_estimated_gpu_resident_bytes=0,
                pool_remaining_bytes=None,
                global_budget_bytes=0,
                global_estimated_gpu_resident_bytes=0,
                global_remaining_bytes=None,
            )

        with self._lock:
            region = self._regions.get(region_id)
            already_gpu = bool(region and region.estimated_gpu_resident)
            incremental_bytes = 0 if already_gpu else bytes
            pool_used = self._pool_estimated_bytes.get(pool, 0)
            global_used = self._global_estimated_bytes_locked()
            pool_budget = self.pool_budgets.get(pool, 0)
            global_budget = self.global_budget_bytes
            if global_budget > 0 and self.safety_margin_bytes > 0:
                global_budget = max(global_budget - self.safety_margin_bytes, 0)
            pool_remaining = (
                max(pool_budget - pool_used, 0) if pool_budget > 0 else None
            )
            global_remaining = (
                max(global_budget - global_used, 0) if global_budget > 0 else None
            )
            pool_over = (
                pool_budget > 0 and incremental_bytes > (pool_remaining or 0)
            )
            global_over = (
                global_budget > 0 and incremental_bytes > (global_remaining or 0)
            )
            would_deny = pool_over or global_over
            # L0 is trace/accounting-only.  Even in enforce mode it does not
            # execute reclaim; callers can inspect would_deny before later stages
            # opt into real action handling.
            allowed = self.mode != "enforce" or not would_deny
            if already_gpu:
                reason = "already_estimated_gpu_resident"
            elif global_over:
                reason = "global_residency_budget_exceeded"
            elif pool_over:
                reason = f"{pool}_residency_budget_exceeded"
            else:
                reason = "granted"

            victims = self._candidate_victims_locked(
                pool=pool,
                required_bytes=incremental_bytes,
                active_region_ids=active_region_ids,
                scope_key=scope_key,
                request_region_id=region_id,
            ) if would_deny else []

            return UvmResidencyDecision(
                enabled=True,
                mode=self.mode,
                pool=pool,
                region_id=region_id,
                requested_bytes=bytes,
                incremental_bytes=incremental_bytes,
                allowed=allowed,
                would_deny=would_deny,
                reason=reason,
                scope_key=scope_key,
                pool_budget_bytes=pool_budget,
                pool_estimated_gpu_resident_bytes=pool_used,
                pool_remaining_bytes=pool_remaining,
                global_budget_bytes=global_budget,
                global_estimated_gpu_resident_bytes=global_used,
                global_remaining_bytes=global_remaining,
                victims=tuple(victims),
            )

    def mark_state(
        self,
        *,
        region_id: str,
        state: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        state = _normalize_state(state)
        metadata = metadata or {}
        if not self.active:
            return
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                return
            old_estimated = region.estimated_gpu_resident
            new_estimated = _state_counts_as_gpu(state, old_estimated)
            before_pool = self._pool_estimated_bytes.get(region.pool, 0)
            if old_estimated and not new_estimated:
                self._pool_estimated_bytes[region.pool] = max(
                    before_pool - region.bytes, 0
                )
            elif not old_estimated and new_estimated:
                self._pool_estimated_bytes[region.pool] = before_pool + region.bytes
            region.state = state
            region.estimated_gpu_resident = new_estimated
            region.metadata.update(metadata)
            region.active = bool(metadata.get("active", region.active))
            if "evictable" in metadata:
                region.evictable = bool(metadata.get("evictable"))
            region.last_state_update_ts = time.time()
            if region.driver_registered:
                self._sync_driver_region_state_locked(region)

    def unregister_region(
        self,
        *,
        region_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        with self._lock:
            region = self._regions.pop(region_id, None)
            if region is None:
                return
            if (
                region.estimated_gpu_resident
                and region.pool in self._pool_estimated_bytes
            ):
                self._pool_estimated_bytes[region.pool] = max(
                    self._pool_estimated_bytes[region.pool] - region.bytes,
                    0,
                )
            if region.driver_registered:
                driver_unregister_region(region.driver_region_id)

    def record_access(
        self,
        *,
        region_id: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                return
            region.last_access_step = int(step)
            if region.driver_registered:
                self._sync_driver_region_state_locked(region)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def region_snapshot(self, region_id: str) -> dict[str, Any] | None:
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                return None
            return self._region_snapshot_locked(region)

    def ensure_driver_region_registered(
        self,
        region_id: str,
    ) -> int:
        """Register a pre-created region once for driver operations."""
        with self._lock:
            region = self._regions[region_id]
            if region.driver_registered:
                return int(region.driver_register_status or 0)
            return self._sync_driver_region_tag_locked(region)

    def regions_snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._region_snapshot_locked(region)
                for region in self._regions.values()
            ]

    def real_residency_snapshot(self, *, trigger: str = "manual") -> dict[str, Any]:
        with self._lock:
            return self._real_residency_snapshot_locked(trigger=trigger)

    def _candidate_victims_locked(
        self,
        *,
        pool: str,
        required_bytes: int,
        active_region_ids: set[str],
        scope_key: str,
        request_region_id: str,
    ) -> list[str]:
        if required_bytes <= 0:
            return []
        candidates = [
            region for region in self._regions.values()
            if region.pool == pool
            and region.estimated_gpu_resident
            and region.evictable
            and not region.active
            and region.region_id not in active_region_ids
            and region.state in ("gpu", "prefetching_gpu")
        ]
        candidates.sort(
            key=lambda item: (
                item.last_access_step if item.last_access_step is not None else -1,
                -item.bytes,
                item.region_id,
            )
        )
        victims: list[str] = []
        reclaimed = 0
        for region in candidates:
            victims.append(region.region_id)
            reclaimed += region.bytes
            if reclaimed >= required_bytes:
                break
        return victims

    def _sync_driver_region_tag_locked(
        self,
        region: UvmResidencyRegion,
    ) -> int:
        metadata = region.metadata
        pool_id = DRIVER_POOL_IDS[region.pool]
        role_id = _driver_role_id(region.pool, metadata)
        flags = _driver_flags(
            pool=region.pool,
            state=region.state,
            evictable=region.evictable,
            active=region.active,
            metadata=metadata,
        )
        layer_id = _metadata_int(metadata, "layer_id", 0)
        expert_id = _metadata_int(metadata, "global_expert_id", 0)
        status = driver_register_region(
            region_id=region.driver_region_id,
            ptr=int(region.ptr),
            nbytes=int(region.bytes),
            pool=pool_id,
            layer_id=layer_id,
            expert_id=expert_id,
            role=role_id,
            flags=flags,
        )
        region.driver_register_status = status
        if status >= 0:
            region.driver_registered = True
        return status

    def _sync_driver_pool_budgets_locked(self, force: bool = False) -> None:
        if self._driver_pool_budgets_applied and not force:
            return
        if not self.driver_pool_budget_enable:
            return

        retry_later = False
        for pool, pool_id in DRIVER_POOL_IDS.items():
            if pool == "unknown":
                continue
            budget = self.driver_pool_budget_overrides.get(pool)
            if budget is None:
                budget = self.pool_budgets.get(pool, 0)
            if pool == "scratch" and budget <= 0:
                budget = self.driver_temp_buffer_bytes
            status = driver_set_pool_budget(
                pool=pool_id,
                budget_bytes=budget,
                mode=self.driver_pool_budget_mode,
            )
            if status == -1:
                retry_later = True
            self._driver_pool_budget_status[pool] = status
        self._driver_pool_budgets_applied = not retry_later

    def _sync_driver_region_state_locked(
        self,
        region: UvmResidencyRegion,
    ) -> int:
        state_id = _driver_region_state_id(region.state, active=region.active)
        flags = _driver_flags(
            pool=region.pool,
            state=region.state,
            evictable=region.evictable,
            active=region.active,
            metadata=region.metadata,
        )
        return driver_update_region_state(
            region_id=region.driver_region_id,
            state=state_id,
            flags_set=flags,
            flags_clear=DRIVER_FLAG_MASK & ~flags,
            priority=_metadata_int(region.metadata, "priority", 0),
            last_access_epoch=max(int(region.last_access_step or 0), 0),
        )

    def _snapshot_locked(self) -> dict[str, Any]:
        pool_region_counts = {pool: 0 for pool in VALID_POOLS}
        for region in self._regions.values():
            if region.pool in pool_region_counts:
                pool_region_counts[region.pool] += 1
        return {
            "weights_estimated_gpu_resident_bytes": (
                self._pool_estimated_bytes["weights"]
            ),
            "kv_estimated_gpu_resident_bytes": self._pool_estimated_bytes["kv"],
            "scratch_estimated_gpu_resident_bytes": (
                self._pool_estimated_bytes["scratch"]
            ),
            "global_estimated_gpu_resident_bytes": (
                self._global_estimated_bytes_locked()
            ),
            "weight_gpu_budget_bytes": self.pool_budgets["weights"],
            "kv_gpu_budget_bytes": self.pool_budgets["kv"],
            "scratch_gpu_budget_bytes": self.pool_budgets["scratch"],
            "global_gpu_budget_bytes": self.global_budget_bytes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "pool_region_counts": pool_region_counts,
            "region_records": len(self._regions),
            "real_residency_enabled": self.real_residency_enable,
            "tag_only_mode": self.tag_only_mode,
            "driver_tag_enable": self.driver_tag_enable,
            "driver_pool_budget_enable": self.driver_pool_budget_enable,
            "driver_pool_budget_status": dict(self._driver_pool_budget_status),
        }

    def _region_snapshot_locked(
        self,
        region: UvmResidencyRegion,
    ) -> dict[str, Any]:
        return {
            "region_id": region.region_id,
            "driver_region_id": region.driver_region_id,
            "driver_registered": region.driver_registered,
            "driver_register_status": region.driver_register_status,
            "pool": region.pool,
            "ptr": region.ptr,
            "bytes": region.bytes,
            "state": region.state,
            "estimated_gpu_resident": region.estimated_gpu_resident,
            "evictable": region.evictable,
            "active": region.active,
            "last_access_step": region.last_access_step,
            "last_state_update_ts": region.last_state_update_ts,
            "metadata": dict(region.metadata),
        }

    def _global_estimated_bytes_locked(self) -> int:
        return sum(self._pool_estimated_bytes.values())

    def _real_residency_snapshot_locked(
        self,
        *,
        trigger: str,
    ) -> dict[str, Any]:
        if not self.real_residency_enable:
            return {"enabled": False, "reason": "disabled"}

        regions_by_pool: dict[str, list[UvmResidencyRegion]] = {
            "weights": [],
            "kv": [],
            "scratch": [],
        }
        skipped_regions = 0
        for region in self._regions.values():
            if region.pool not in regions_by_pool:
                skipped_regions += 1
                continue
            if not isinstance(region.ptr, int) or region.ptr <= 0 or region.bytes <= 0:
                skipped_regions += 1
                continue
            regions_by_pool[region.pool].append(region)

        totals = _empty_real_residency_totals()
        by_pool: dict[str, dict[str, Any]] = {}
        ok = True
        max_regions = self.real_residency_max_regions_per_pool
        query_batch_size = 64
        for pool, pool_regions in regions_by_pool.items():
            queried_regions = (
                pool_regions[:max_regions] if max_regions > 0 else pool_regions
            )
            ranges = [(region.ptr or 0, region.bytes) for region in queried_regions]
            pool_record: dict[str, Any] = {
                "region_count": len(pool_regions),
                "queried_region_count": len(queried_regions),
                "registered_bytes": sum(region.bytes for region in pool_regions),
                **_empty_real_residency_totals(),
                "status": 0,
                "ok": True,
                "query_batch_size": query_batch_size,
                "query_batches": 0,
                "failed_batches": 0,
                "sample_statuses": [],
            }
            if ranges:
                for start in range(0, len(ranges), query_batch_size):
                    batch = ranges[start:start + query_batch_size]
                    pool_record["query_batches"] += 1
                    try:
                        result = driver_batch_query_ranges_residency_bytes(
                            batch,
                            device=self.real_residency_device,
                        )
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        result = {
                            "ok": False,
                            "status": -1,
                            "error": repr(exc),
                            **_empty_real_residency_totals(),
                        }

                    status = int(result.get("status") or 0)
                    if len(pool_record["sample_statuses"]) < 8:
                        pool_record["sample_statuses"].append(status)
                    if not bool(result.get("ok")):
                        pool_record["ok"] = False
                        pool_record["failed_batches"] += 1
                        pool_record["status"] = status
                        if "error" in result and "error" not in pool_record:
                            pool_record["error"] = result["error"]

                    for key in _empty_real_residency_totals():
                        pool_record[key] += int(result.get(key) or 0)
                ok = ok and bool(pool_record.get("ok"))
            for key in totals:
                totals[key] += int(pool_record.get(key) or 0)
            by_pool[pool] = pool_record

        return {
            "ok": ok,
            "trigger": trigger,
            "device": self.real_residency_device,
            "region_records": len(self._regions),
            "skipped_regions": skipped_regions,
            "max_regions_per_pool": max_regions,
            "query_batch_size": query_batch_size,
            **totals,
            "by_pool": by_pool,
        }


_RESIDENCY_MANAGER: UvmPoolResidencyManager | None = None
_RESIDENCY_MANAGER_LOCK = threading.Lock()


def get_uvm_pool_residency_manager() -> UvmPoolResidencyManager:
    global _RESIDENCY_MANAGER
    if _RESIDENCY_MANAGER is None:
        with _RESIDENCY_MANAGER_LOCK:
            if _RESIDENCY_MANAGER is None:
                _RESIDENCY_MANAGER = UvmPoolResidencyManager()
    return _RESIDENCY_MANAGER


def register_uvm_residency_region(
    *,
    pool: str,
    region_id: str,
    ptr: int | None,
    bytes: int,
    state: str = "unknown",
    evictable: bool = False,
    active: bool = False,
    driver_register: bool,
    metadata: dict[str, Any] | None = None,
) -> UvmResidencyRegion:
    return get_uvm_pool_residency_manager().register_region(
        pool=pool,
        region_id=region_id,
        ptr=ptr,
        bytes=bytes,
        state=state,
        evictable=evictable,
        active=active,
        driver_register=driver_register,
        metadata=metadata,
    )


def ensure_uvm_residency_driver_region_registered(
    region_id: str,
) -> int:
    return get_uvm_pool_residency_manager().ensure_driver_region_registered(
        region_id
    )


def request_uvm_gpu_residency(
    *,
    pool: str,
    region_id: str,
    bytes: int,
    scope_key: str | None = None,
    active_region_ids: set[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> UvmResidencyDecision:
    return get_uvm_pool_residency_manager().request_gpu_residency(
        pool=pool,
        region_id=region_id,
        bytes=bytes,
        scope_key=scope_key,
        active_region_ids=active_region_ids,
        metadata=metadata,
    )


def mark_uvm_residency_state(
    *,
    region_id: str,
    state: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().mark_state(
        region_id=region_id,
        state=state,
        metadata=metadata,
    )


def unregister_uvm_residency_region(
    *,
    region_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().unregister_region(
        region_id=region_id,
        metadata=metadata,
    )


def record_uvm_residency_access(
    *,
    region_id: str,
    step: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().record_access(
        region_id=region_id,
        step=step,
        metadata=metadata,
    )


def get_uvm_residency_snapshot() -> dict[str, Any]:
    return get_uvm_pool_residency_manager().snapshot()


def sync_uvm_driver_pool_budgets() -> dict[str, int]:
    return get_uvm_pool_residency_manager().sync_driver_pool_budgets()


def set_uvm_driver_pool_budget(
    pool: str,
    budget_bytes: int,
    *,
    gpu_id: int = 0,
    mode: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return get_uvm_pool_residency_manager().set_driver_pool_budget(
        pool,
        budget_bytes,
        gpu_id=gpu_id,
        mode=mode,
        metadata=metadata,
    )


def get_uvm_driver_pool_state_snapshot(gpu_id: int = 0) -> dict[str, Any]:
    return get_uvm_pool_residency_manager().driver_pool_state_snapshot(
        gpu_id=gpu_id
    )


def soft_reclaim_uvm_driver_pool(
    pool: str,
    *,
    gpu_id: int = 0,
    target_bytes: int = 0,
    current_epoch: int = 0,
    cooldown_epochs: int = 0,
    max_victims: int = 8,
    async_reclaim: bool = False,
) -> dict[str, Any] | None:
    return get_uvm_pool_residency_manager().driver_soft_reclaim_pool(
        pool,
        gpu_id=gpu_id,
        target_bytes=target_bytes,
        current_epoch=current_epoch,
        cooldown_epochs=cooldown_epochs,
        max_victims=max_victims,
        async_reclaim=async_reclaim,
    )


def get_uvm_real_residency_snapshot(trigger: str = "manual") -> dict[str, Any]:
    return get_uvm_pool_residency_manager().real_residency_snapshot(trigger=trigger)


def get_uvm_residency_region_snapshot(region_id: str) -> dict[str, Any] | None:
    return get_uvm_pool_residency_manager().region_snapshot(region_id)


def get_uvm_residency_regions_snapshot() -> list[dict[str, Any]]:
    return get_uvm_pool_residency_manager().regions_snapshot()
