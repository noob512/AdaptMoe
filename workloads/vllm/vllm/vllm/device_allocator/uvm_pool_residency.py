# SPDX-License-Identifier: Apache-2.0
"""Stage L0 UVM pool residency accounting.

This module maintains an application-level estimate of GPU residency intent for
weights, KV, and scratch pools.  Physical movement is handled only by the
explicit region and VA-range APIs below.
"""

from __future__ import annotations

import hashlib
import os
import threading
from typing import Any

from vllm.device_allocator.uvm import (
    batch_prefetch_ranges_to_device_with_executor,
    driver_release_regions,
    driver_register_region,
    driver_update_region_state,
)
from vllm.device_allocator.uvm_region import (
    ExplicitRegionRegistry,
    KernelRegion,
    RegionState,
    normalize_region_state,
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

DRIVER_REGION_STATES = {
    "unknown": 0,
    "cpu": 1,
    "released": 1,
    "prefetching_cpu": 2,
    "gpu": 3,
    "prefetching_gpu": 4,
}

_UNSUPPORTED_DRIVER_POOL_POLICY_STATUS = -95
_UNSUPPORTED_DRIVER_POOL_POLICY_REASON = "unsupported_explicit_region_only"

_EXPLICIT_REGION_REGISTRY = ExplicitRegionRegistry()
_EXPLICIT_REGION_LOCK = threading.Lock()


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


def _driver_region_state_id(state: str) -> int:
    return DRIVER_REGION_STATES.get(_normalize_state(state), 0)


class UvmPoolResidencyManager:
    """Stage L0 estimated residency ledger for UVM memory pools."""

    def __init__(self) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_RESIDENCY_ENABLE", False)
        self.mode = "trace_only"
        self.global_budget_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES", 0
        )
        self.pool_budgets = {
            "weights": _read_int_env("VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES", 0),
            "kv": _read_int_env("VLLM_UVM_RESIDENCY_KV_GPU_BYTES", 0),
            "scratch": _read_int_env("VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES", 0),
        }
        self.driver_pool_budget_enable = False
        self._driver_pool_budget_status: dict[str, int] = {
            pool: _UNSUPPORTED_DRIVER_POOL_POLICY_STATUS for pool in VALID_POOLS
        }
        self.safety_margin_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES", 0
        )
        self._lock = threading.Lock()
        self._pool_estimated_bytes = {"weights": 0, "kv": 0, "scratch": 0}

    @property
    def active(self) -> bool:
        return self.enabled

    def sync_driver_pool_budgets(self) -> dict[str, int]:
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
        budget = max(int(budget_bytes or 0), 0)
        if pool_name == "unknown":
            return {
                "ok": False,
                "pool": pool_name,
                "budget_bytes": budget,
                "status": -2,
                "reason": "unknown_pool",
            }
        with self._lock:
            self.pool_budgets[pool_name] = budget
            self._driver_pool_budget_status[pool_name] = (
                _UNSUPPORTED_DRIVER_POOL_POLICY_STATUS
            )
        return {
            "ok": False,
            "pool": pool_name,
            "budget_bytes": budget,
            "gpu_id": max(int(gpu_id or 0), 0),
            "mode": 0 if mode is None else max(int(mode), 0),
            "status": _UNSUPPORTED_DRIVER_POOL_POLICY_STATUS,
            "reason": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
        }

    def driver_pool_state_snapshot(self, gpu_id: int = 0) -> dict[str, Any]:
        return {
            "ok": False,
            "gpu_id": max(int(gpu_id or 0), 0),
            "reason": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
        }

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
    ) -> dict[str, Any]:
        pool_name = _normalize_driver_pool_name(pool)
        if pool_name == "unknown":
            return {
                "ok": False,
                "pool": pool_name,
                "status": -2,
                "reason": "unknown_pool",
            }
        return {
            "ok": False,
            "pool": pool_name,
            "gpu_id": max(int(gpu_id or 0), 0),
            "target_bytes": max(int(target_bytes or 0), 0),
            "current_epoch": max(int(current_epoch or 0), 0),
            "cooldown_epochs": max(int(cooldown_epochs or 0), 0),
            "max_victims": max(int(max_victims or 0), 0),
            "async_reclaim": bool(async_reclaim),
            "status": _UNSUPPORTED_DRIVER_POOL_POLICY_STATUS,
            "reason": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked()

    def regions_snapshot(self) -> list[dict[str, Any]]:
        return []

    def _snapshot_locked(self) -> dict[str, Any]:
        pool_region_counts = {pool: 0 for pool in VALID_POOLS}
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
            "region_records": 0,
            "driver_pool_budget_enable": self.driver_pool_budget_enable,
            "driver_pool_budget_status": dict(self._driver_pool_budget_status),
        }

    def _global_estimated_bytes_locked(self) -> int:
        return sum(self._pool_estimated_bytes.values())


_RESIDENCY_MANAGER: UvmPoolResidencyManager | None = None
_RESIDENCY_MANAGER_LOCK = threading.Lock()


def register_uvm_region(
    region_id: str,
    base: int,
    length: int,
    *,
    state: str = "gpu",
) -> KernelRegion:
    """Register the minimal region contract in application and driver tables."""
    with _EXPLICIT_REGION_LOCK:
        region = _EXPLICIT_REGION_REGISTRY.register_region(
            region_id,
            base,
            length,
            state,
        )
        driver_region_id = _stable_driver_region_id(region.region_id)
        status = driver_register_region(
            region_id=driver_region_id,
            ptr=region.base,
            nbytes=region.length,
        )
        if status < 0:
            region.state = RegionState.UNKNOWN
        else:
            driver_update_region_state(
                region_id=driver_region_id,
                state=_driver_region_state_id(region.state.value),
            )
        return region


def release_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    remote_read: bool = False,
) -> dict[str, Any]:
    """Release only the explicitly named regions, with idempotent state checks."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    pending: list[KernelRegion] = []
    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None or region.state == RegionState.UNKNOWN:
                statuses[region_id] = False
            elif region.state == RegionState.CPU:
                statuses[region_id] = True
            else:
                region.state = RegionState.PREFETCHING_CPU
                pending.append(region)

    if pending:
        numeric_to_region = {
            _stable_driver_region_id(region.region_id): region for region in pending
        }
        result = driver_release_regions(
            list(numeric_to_region),
            device=device,
            remote_read=remote_read,
            return_entries=True,
        )
        entries = {
            int(entry.get("region_id") or 0): bool(entry.get("ok"))
            for entry in (result.get("entries") or [])
            if isinstance(entry, dict)
        }
        aggregate_ok = bool(result.get("ok"))
        with _EXPLICIT_REGION_LOCK:
            for numeric_id, region in numeric_to_region.items():
                ok = entries.get(numeric_id, aggregate_ok)
                region.state = RegionState.CPU if ok else RegionState.UNKNOWN
                statuses[region.region_id] = ok
                driver_update_region_state(
                    region_id=numeric_id,
                    state=_driver_region_state_id(region.state.value),
                )

    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def restore_uvm_regions(
    region_ids: list[str] | tuple[str, ...],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, Any]:
    """Restore only the explicitly named regions using their registered ranges."""
    requested = [str(region_id) for region_id in region_ids]
    statuses: dict[str, bool] = {}
    pending: list[KernelRegion] = []
    with _EXPLICIT_REGION_LOCK:
        for region_id in requested:
            region = _EXPLICIT_REGION_REGISTRY.regions.get(region_id)
            if region is None or region.state == RegionState.UNKNOWN:
                statuses[region_id] = False
            elif region.state == RegionState.GPU:
                statuses[region_id] = True
            else:
                region.state = RegionState.PREFETCHING_GPU
                pending.append(region)

    if pending:
        ranges = [(region.base, region.length) for region in pending]
        successes, _, _ = batch_prefetch_ranges_to_device_with_executor(
            ranges,
            device,
            executor=executor,
        )
        complete = int(successes) == len(pending)
        with _EXPLICIT_REGION_LOCK:
            for region in pending:
                region.state = RegionState.GPU if complete else RegionState.UNKNOWN
                statuses[region.region_id] = complete
                driver_update_region_state(
                    region_id=_stable_driver_region_id(region.region_id),
                    state=_driver_region_state_id(region.state.value),
                )

    return {
        "ok": all(statuses.get(region_id, False) for region_id in requested),
        "statuses": statuses,
        "failed_region_ids": [
            region_id for region_id in requested if not statuses.get(region_id, False)
        ],
    }


def query_uvm_region_states(
    region_ids: list[str] | tuple[str, ...],
) -> dict[str, str]:
    with _EXPLICIT_REGION_LOCK:
        return {
            region_id: region.state.value
            for region_id in (str(value) for value in region_ids)
            if (region := _EXPLICIT_REGION_REGISTRY.regions.get(region_id)) is not None
        }


def get_uvm_pool_residency_manager() -> UvmPoolResidencyManager:
    global _RESIDENCY_MANAGER
    if _RESIDENCY_MANAGER is None:
        with _RESIDENCY_MANAGER_LOCK:
            if _RESIDENCY_MANAGER is None:
                _RESIDENCY_MANAGER = UvmPoolResidencyManager()
    return _RESIDENCY_MANAGER


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


def get_uvm_residency_regions_snapshot() -> list[dict[str, Any]]:
    return get_uvm_pool_residency_manager().regions_snapshot()
