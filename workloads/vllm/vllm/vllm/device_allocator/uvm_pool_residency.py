# SPDX-License-Identifier: Apache-2.0
"""Stage L0 UVM pool residency accounting.

This module maintains an application-level estimate of GPU residency intent for
weights, KV, and scratch pools.  When explicitly enabled, it can also sample
page-granular real UVM residency from inside the vLLM process.  It does not
migrate, evict, or prefetch memory by itself.
"""

from __future__ import annotations

import json
import os
import threading
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any

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
DRIVER_FLAG_MASK = (
    DRIVER_FLAG_EVICTABLE
    | DRIVER_FLAG_PROTECTED
    | DRIVER_FLAG_ACTIVE
    | DRIVER_FLAG_PREFETCHABLE
    | DRIVER_FLAG_PINNED
    | DRIVER_FLAG_TRANSIENT
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
        self.driver_tag_require_managed = _read_bool_env(
            "VLLM_UVM_DRIVER_TAG_REQUIRE_MANAGED",
            True,
        )
        self.trace_enabled = _read_bool_env("VLLM_UVM_RESIDENCY_TRACE_ENABLE", True)
        self.trace_file = (
            os.environ.get(
                "VLLM_UVM_RESIDENCY_TRACE_FILE",
                "vllm_uvm_pool_residency_stage_l.jsonl",
            )
            if self.trace_enabled
            else ""
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
        self.driver_pool_state_snapshot_on_tag_only_every = _read_int_env(
            "VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_ON_TAG_ONLY_EVERY",
            0,
        )
        self.driver_pool_state_snapshot_device = _read_int_env(
            "VLLM_UVM_DRIVER_POOL_STATE_SNAPSHOT_DEVICE",
            0,
        )
        self._driver_pool_budget_status: dict[str, int] = {}
        self._driver_pool_budgets_applied = False
        self.safety_margin_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES", 0
        )
        self.scratch_reserved_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_SCRATCH_RESERVED_BYTES", 0
        )
        self.scratch_margin_bytes = _read_int_env(
            "VLLM_UVM_RESIDENCY_SCRATCH_MARGIN_BYTES", 0
        )
        self.kv_sizing_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_SIZING_ENABLE", False
        )
        self.kv_sizing_policy = os.environ.get(
            "VLLM_UVM_RESIDENCY_KV_SIZING_POLICY", "report_only"
        ).strip().lower()
        self.kv_sizing_max_blocks = _read_int_env(
            "VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS", 0
        )
        self.kv_metadata_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") in ("stage_l5", "stage_l6"),
        )
        self.kv_data_offload_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l6",
        )
        self.kv_prefetch_back_enable = _read_bool_env(
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE",
            os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "") == "stage_l6",
        )
        self.priority = os.environ.get(
            "VLLM_UVM_RESIDENCY_PRIORITY", "kv,scratch,weights"
        )
        self.stage_label = os.environ.get("VLLM_UVM_RESIDENCY_STAGE", "stage_l0")
        self.weight_initial_state = _normalize_state(
            os.environ.get("VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE", "cpu")
        )
        self.weight_enforce_hot_set = _read_bool_env(
            "VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET", False
        )
        self.weight_sync_offload = _read_bool_env(
            "VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD", True
        )
        self.weight_max_victims_per_step = _read_int_env(
            "VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP", 0
        )
        self.weight_victim_policy = os.environ.get(
            "VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_POLICY", "cold_plan_lru"
        )
        self.real_residency_enable = _read_bool_env(
            "VLLM_UVM_REAL_RESIDENCY_ENABLE", False
        )
        self.real_residency_trace_file = (
            os.environ.get(
                "VLLM_UVM_REAL_RESIDENCY_TRACE_FILE",
                "vllm_uvm_real_residency_stage_l.jsonl",
            )
            if self.real_residency_enable
            else ""
        )
        self.real_residency_device = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_DEVICE", 0
        )
        self.real_residency_on_register_every = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_ON_REGISTER_EVERY", 0
        )
        self.real_residency_on_action_every = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_ON_ACTION_EVERY", 0
        )
        self.real_residency_max_snapshots = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS", 0
        )
        self.real_residency_max_regions_per_pool = _read_int_env(
            "VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL", 0
        )
        self._real_residency_snapshots = 0
        self._real_residency_errors = 0
        self._last_real_residency: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._config_recorded = False
        self._regions: dict[str, UvmResidencyRegion] = {}
        self._pool_estimated_bytes = {"weights": 0, "kv": 0, "scratch": 0}
        self._scratch_summary: dict[str, Any] = {
            "scratch_peak_bytes": 0,
            "scratch_live_bytes": 0,
            "scratch_reserved_bytes": self.scratch_reserved_bytes,
            "scratch_margin_bytes": self.scratch_margin_bytes,
            "scratch_pool_enabled": False,
            "scratch_pool_mode": None,
            "scratch_trace_records": 0,
            "scratch_eligible_records": 0,
            "scratch_device_direct_records": 0,
            "scratch_budget_over_records": 0,
            "scratch_budget_reject_records": 0,
            "scratch_eviction_records": 0,
            "scratch_stage_g_summary_present": False,
        }
        self._kv_sizing_summary: dict[str, Any] = {
            "kv_sizing_records": 0,
            "kv_sizing_config_present": False,
            "kv_sizing_policy": self.kv_sizing_policy,
            "kv_sizing_max_blocks": self.kv_sizing_max_blocks,
            "kv_sizing_mode": "explicit" if self.pool_budgets["kv"] > 0 else "derived",
            "kv_sizing_applied": False,
            "kv_sizing_report_only": True,
            "kv_sizing_fallback_reason": None,
            "kv_sizing_explicit_budget_bytes": self.pool_budgets["kv"],
            "kv_sizing_derived_budget_bytes": 0,
            "kv_sizing_budget_bytes": 0,
            "kv_sizing_budget_blocks": 0,
            "kv_sizing_bytes_per_block": 0,
            "kv_sizing_original_num_gpu_blocks": 0,
            "kv_sizing_final_num_gpu_blocks": 0,
            "kv_sizing_original_kv_bytes": 0,
            "kv_sizing_final_kv_bytes": 0,
        }
        self._summary = {
            "register_region_records": 0,
            "requests": 0,
            "grants": 0,
            "denies": 0,
            "would_denies": 0,
            "state_updates": 0,
            "access_records": 0,
            "would_evict_candidates": 0,
            "real_action_records": 0,
            "scratch_summary_records": 0,
            "kv_sizing_records": 0,
            "kv_metadata_records": 0,
            "kv_safe_candidate_records": 0,
            "kv_unsafe_candidate_records": 0,
            "kv_data_offload_records": 0,
            "kv_data_offload_attempt_records": 0,
            "kv_data_offload_issued_records": 0,
            "kv_data_offload_fallback_records": 0,
            "kv_prefetch_back_records": 0,
            "kv_prefetch_back_issued_records": 0,
            "kv_prefetch_back_fallback_records": 0,
            "driver_tag_register_attempts": 0,
            "driver_tag_register_success": 0,
            "driver_tag_register_skips": 0,
            "driver_tag_register_failures": 0,
            "driver_region_state_update_attempts": 0,
            "driver_region_state_update_success": 0,
            "driver_region_state_update_skips": 0,
            "driver_region_state_update_failures": 0,
            "driver_pool_state_queries": 0,
            "driver_pool_state_failures": 0,
            "driver_pool_reclaim_attempts": 0,
            "driver_pool_reclaim_failures": 0,
        }

    @property
    def active(self) -> bool:
        return self.enabled

    def sync_driver_pool_budgets(self) -> dict[str, int]:
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked(force=True)
            return dict(self._driver_pool_budget_status)

    def driver_pool_state_snapshot(self, gpu_id: int = 0) -> dict[str, Any]:
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()

        try:
            from vllm.device_allocator.uvm import driver_get_pool_state
        except Exception as exc:  # pragma: no cover - diagnostic path
            with self._lock:
                self._summary["driver_pool_state_failures"] += 1
                self._write_locked(
                    {
                        "action": "residency_driver_pool_state",
                        "reason": "import_failed",
                        "error": repr(exc),
                    }
                )
            return {}

        states: dict[str, Any] = {}
        failures = 0
        for pool, pool_id in DRIVER_POOL_IDS.items():
            if pool == "unknown":
                continue
            state = driver_get_pool_state(pool_id, gpu_id=gpu_id)
            if state is None:
                failures += 1
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

        with self._lock:
            self._summary["driver_pool_state_queries"] += 1
            self._summary["driver_pool_state_failures"] += failures
            self._write_locked(
                {
                    "action": "residency_driver_pool_state",
                    "gpu_id": max(int(gpu_id or 0), 0),
                    "states": states,
                    "failures": failures,
                }
            )
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
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()

        try:
            from vllm.device_allocator.uvm import driver_reclaim_pool
        except Exception as exc:  # pragma: no cover - diagnostic path
            with self._lock:
                self._summary["driver_pool_reclaim_failures"] += 1
                self._write_locked(
                    {
                        "action": "residency_driver_pool_reclaim",
                        "pool": pool_name,
                        "reason": "import_failed",
                        "error": repr(exc),
                    }
                )
            return None

        result = driver_reclaim_pool(
            pool_id,
            gpu_id=gpu_id,
            target_bytes=target_bytes,
            current_epoch=current_epoch,
            cooldown_epochs=cooldown_epochs,
            max_victims=max_victims,
            async_reclaim=async_reclaim,
        )
        with self._lock:
            self._summary["driver_pool_reclaim_attempts"] += 1
            if result is None:
                self._summary["driver_pool_reclaim_failures"] += 1
            self._write_locked(
                {
                    "action": "residency_driver_pool_reclaim",
                    "pool": pool_name,
                    "gpu_id": max(int(gpu_id or 0), 0),
                    "target_bytes": max(int(target_bytes or 0), 0),
                    "current_epoch": max(int(current_epoch or 0), 0),
                    "cooldown_epochs": max(int(cooldown_epochs or 0), 0),
                    "max_victims": max(int(max_victims or 0), 0),
                    "async_reclaim": bool(async_reclaim),
                    "result": result,
                }
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
        metadata: dict[str, Any] | None = None,
    ) -> UvmResidencyRegion:
        pool = _normalize_pool(pool)
        state = _normalize_state(state)
        bytes = max(int(bytes or 0), 0)
        metadata = metadata or {}
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()
            existing = self._regions.get(region_id)
            if existing is not None:
                old_bytes = existing.bytes
                old_estimated = existing.estimated_gpu_resident
                if old_estimated and existing.pool in self._pool_estimated_bytes:
                    self._pool_estimated_bytes[existing.pool] = max(
                        self._pool_estimated_bytes[existing.pool] - old_bytes, 0
                    )
                existing.pool = pool
                existing.ptr = ptr
                existing.bytes = bytes
                existing.state = state
                existing.evictable = evictable
                existing.active = active
                existing.metadata.update(metadata)
                existing.estimated_gpu_resident = _state_counts_as_gpu(state)
                existing.last_state_update_ts = time.time()
                region = existing
            else:
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

            driver_tag_record = self._sync_driver_region_tag_locked(region)
            driver_state_record = self._sync_driver_region_state_locked(region)

            self._summary["register_region_records"] += 1
            register_count = self._summary["register_region_records"]
            self._write_locked(
                {
                    "action": "residency_register_region",
                    "pool": pool,
                    "region_id": region_id,
                    "driver_region_id": region.driver_region_id,
                    "driver_registered": region.driver_registered,
                    "driver_register_status": region.driver_register_status,
                    "driver_tag": driver_tag_record,
                    "driver_state": driver_state_record,
                    "ptr": f"0x{ptr:x}" if isinstance(ptr, int) else None,
                    "bytes": bytes,
                    "state": state,
                    "estimated_gpu_resident": region.estimated_gpu_resident,
                    "evictable": evictable,
                    "active": active,
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()
            if (
                self.real_residency_on_register_every > 0
                and register_count % self.real_residency_on_register_every == 0
            ):
                self._record_real_residency_locked(
                    trigger="register_region",
                    trigger_count=register_count,
                    trigger_region_id=region_id,
                )
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
        metadata = metadata or {}
        active_region_ids = active_region_ids or set()
        if self.tag_only_mode:
            snapshot_due = False
            with self._lock:
                self._record_config_locked()
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
                self._record_request_locked(decision, metadata)
                request_count = self._summary["requests"]
                snapshot_every = self.driver_pool_state_snapshot_on_tag_only_every
                snapshot_due = (
                    snapshot_every > 0 and request_count % snapshot_every == 0
                )
            if snapshot_due:
                self.driver_pool_state_snapshot(
                    gpu_id=self.driver_pool_state_snapshot_device
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
            self._record_config_locked()
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

            decision = UvmResidencyDecision(
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
            self._record_request_locked(decision, metadata)
            return decision

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
            self._record_config_locked()
            region = self._regions.get(region_id)
            if region is None:
                self._write_locked(
                    {
                        "action": "residency_state_update_failed",
                        "region_id": region_id,
                        "new_state": state,
                        "reason": "unknown_region",
                        "metadata": metadata,
                    }
                )
                return
            old_state = region.state
            old_estimated = region.estimated_gpu_resident
            new_estimated = _state_counts_as_gpu(state, old_estimated)
            before_pool = self._pool_estimated_bytes.get(region.pool, 0)
            before_global = self._global_estimated_bytes_locked()
            if old_estimated and not new_estimated:
                self._pool_estimated_bytes[region.pool] = max(
                    before_pool - region.bytes, 0
                )
            elif not old_estimated and new_estimated:
                self._pool_estimated_bytes[region.pool] = before_pool + region.bytes
            after_pool = self._pool_estimated_bytes.get(region.pool, 0)
            region.state = state
            region.estimated_gpu_resident = new_estimated
            region.metadata.update(metadata)
            region.active = bool(metadata.get("active", region.active))
            if "evictable" in metadata:
                region.evictable = bool(metadata.get("evictable"))
            region.last_state_update_ts = time.time()
            driver_state_record = self._sync_driver_region_state_locked(region)
            self._summary["state_updates"] += 1
            self._write_locked(
                {
                    "action": "residency_state_update",
                    "pool": region.pool,
                    "region_id": region_id,
                    "old_state": old_state,
                    "new_state": state,
                    "bytes": region.bytes,
                    "old_estimated_gpu_resident": old_estimated,
                    "new_estimated_gpu_resident": new_estimated,
                    "pool_estimated_gpu_resident_bytes_before": before_pool,
                    "pool_estimated_gpu_resident_bytes_after": after_pool,
                    "global_estimated_gpu_resident_bytes_before": before_global,
                    "global_estimated_gpu_resident_bytes_after": (
                        self._global_estimated_bytes_locked()
                    ),
                    "driver_state": driver_state_record,
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()

    def unregister_region(
        self,
        *,
        region_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = metadata or {}
        if not self.active:
            return
        with self._lock:
            self._record_config_locked()
            region = self._regions.pop(region_id, None)
            if region is None:
                self._write_locked(
                    {
                        "action": "residency_unregister_region_failed",
                        "region_id": region_id,
                        "reason": "unknown_region",
                        "metadata": metadata,
                    }
                )
                return
            if (
                region.estimated_gpu_resident
                and region.pool in self._pool_estimated_bytes
            ):
                self._pool_estimated_bytes[region.pool] = max(
                    self._pool_estimated_bytes[region.pool] - region.bytes,
                    0,
                )
            driver_unregister_status: int | None = None
            if self.driver_tag_enable and region.driver_registered:
                try:
                    from vllm.device_allocator.uvm import driver_unregister_region

                    driver_unregister_status = driver_unregister_region(
                        region.driver_region_id
                    )
                except Exception as exc:  # pragma: no cover - diagnostic path
                    driver_unregister_status = -3
                    metadata = {**metadata, "driver_unregister_error": repr(exc)}
            self._write_locked(
                {
                    "action": "residency_unregister_region",
                    "pool": region.pool,
                    "region_id": region_id,
                    "driver_region_id": region.driver_region_id,
                    "driver_registered": region.driver_registered,
                    "driver_unregister_status": driver_unregister_status,
                    "bytes": region.bytes,
                    "state": region.state,
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()

    def record_access(
        self,
        *,
        region_id: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        metadata = metadata or {}
        with self._lock:
            self._record_config_locked()
            region = self._regions.get(region_id)
            if region is None:
                return
            region.last_access_step = int(step)
            driver_state_record = self._sync_driver_region_state_locked(region)
            self._summary["access_records"] += 1
            self._write_locked(
                {
                    "action": "residency_access",
                    "pool": region.pool,
                    "region_id": region_id,
                    "step": int(step),
                    "driver_state": driver_state_record,
                    "metadata": metadata,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._summary_record_locked()

    def region_snapshot(self, region_id: str) -> dict[str, Any] | None:
        with self._lock:
            region = self._regions.get(region_id)
            if region is None:
                return None
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

    def record_action(
        self,
        *,
        action: str,
        pool: str,
        region_id: str | None = None,
        bytes: int = 0,
        metadata: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        if not self.active:
            return
        pool = _normalize_pool(pool)
        metadata = metadata or {}
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()
            self._summary["real_action_records"] += 1
            action_count = self._summary["real_action_records"]
            if pool == "kv":
                if action == "residency_kv_data_offload_attempt":
                    self._summary["kv_data_offload_attempt_records"] += 1
                elif action == "residency_kv_data_offload_issued":
                    self._summary["kv_data_offload_records"] += 1
                    self._summary["kv_data_offload_issued_records"] += 1
                elif action == "residency_kv_data_offload_fallback":
                    self._summary["kv_data_offload_records"] += 1
                    self._summary["kv_data_offload_fallback_records"] += 1
                elif action == "residency_kv_prefetch_back_issued":
                    self._summary["kv_prefetch_back_records"] += 1
                    self._summary["kv_prefetch_back_issued_records"] += 1
                elif action == "residency_kv_prefetch_back_fallback":
                    self._summary["kv_prefetch_back_records"] += 1
                    self._summary["kv_prefetch_back_fallback_records"] += 1
            record: dict[str, Any] = {
                "action": action,
                "pool": pool,
                "region_id": region_id,
                "bytes": max(int(bytes or 0), 0),
                "metadata": metadata,
            }
            record.update(extra)
            self._write_locked(record)
            self._record_summary_locked()
            if (
                self.real_residency_on_action_every > 0
                and action_count % self.real_residency_on_action_every == 0
            ):
                self._record_real_residency_locked(
                    trigger="record_action",
                    trigger_count=action_count,
                    trigger_region_id=region_id,
                    trigger_action=action,
                )

    def real_residency_snapshot(self, *, trigger: str = "manual") -> dict[str, Any]:
        with self._lock:
            self._record_config_locked()
            return self._record_real_residency_locked(trigger=trigger)

    def record_scratch_summary(
        self,
        *,
        scratch_live_bytes: int = 0,
        scratch_peak_bytes: int = 0,
        scratch_reserved_bytes: int | None = None,
        scratch_margin_bytes: int | None = None,
        scratch_pool_enabled: bool | None = None,
        scratch_pool_mode: str | None = None,
        scratch_trace_records: int = 0,
        scratch_eligible_records: int = 0,
        scratch_device_direct_records: int = 0,
        scratch_budget_over_records: int = 0,
        scratch_budget_reject_records: int = 0,
        scratch_eviction_records: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        metadata = metadata or {}
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()
            if scratch_reserved_bytes is None:
                scratch_reserved_bytes = self.scratch_reserved_bytes
            if scratch_margin_bytes is None:
                scratch_margin_bytes = self.scratch_margin_bytes
            scratch_summary = {
                "scratch_peak_bytes": max(int(scratch_peak_bytes or 0), 0),
                "scratch_live_bytes": max(int(scratch_live_bytes or 0), 0),
                "scratch_reserved_bytes": max(int(scratch_reserved_bytes or 0), 0),
                "scratch_margin_bytes": max(int(scratch_margin_bytes or 0), 0),
                "scratch_pool_enabled": bool(scratch_pool_enabled),
                "scratch_pool_mode": scratch_pool_mode,
                "scratch_trace_records": max(int(scratch_trace_records or 0), 0),
                "scratch_eligible_records": max(int(scratch_eligible_records or 0), 0),
                "scratch_device_direct_records": max(
                    int(scratch_device_direct_records or 0), 0
                ),
                "scratch_budget_over_records": max(
                    int(scratch_budget_over_records or 0), 0
                ),
                "scratch_budget_reject_records": max(
                    int(scratch_budget_reject_records or 0), 0
                ),
                "scratch_eviction_records": max(
                    int(scratch_eviction_records or 0), 0
                ),
                "scratch_stage_g_summary_present": True,
            }
            self._scratch_summary.update(scratch_summary)
            self._summary["scratch_summary_records"] += 1
            self._write_locked(
                {
                    "action": "residency_scratch_summary",
                    "pool": "scratch",
                    **scratch_summary,
                    "scratch_gpu_budget_bytes": self.pool_budgets["scratch"],
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()

    def record_kv_sizing(
        self,
        *,
        bytes_per_block: int,
        original_num_gpu_blocks: int,
        final_num_gpu_blocks: int | None = None,
        original_kv_bytes: int | None = None,
        final_kv_bytes: int | None = None,
        explicit_kv_budget_bytes: int | None = None,
        applied: bool | None = None,
        policy: str | None = None,
        max_blocks: int | None = None,
        fallback_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        metadata = metadata or {}
        bytes_per_block = max(int(bytes_per_block or 0), 0)
        original_num_gpu_blocks = max(int(original_num_gpu_blocks or 0), 0)
        if final_num_gpu_blocks is None:
            final_num_gpu_blocks = original_num_gpu_blocks
        final_num_gpu_blocks = max(int(final_num_gpu_blocks or 0), 0)
        if original_kv_bytes is None:
            original_kv_bytes = bytes_per_block * original_num_gpu_blocks
        if final_kv_bytes is None:
            final_kv_bytes = bytes_per_block * final_num_gpu_blocks
        original_kv_bytes = max(int(original_kv_bytes or 0), 0)
        final_kv_bytes = max(int(final_kv_bytes or 0), 0)
        if explicit_kv_budget_bytes is None:
            explicit_kv_budget_bytes = self.pool_budgets["kv"]
        explicit_kv_budget_bytes = max(int(explicit_kv_budget_bytes or 0), 0)
        policy = (policy or self.kv_sizing_policy).strip().lower()
        if max_blocks is None:
            max_blocks = self.kv_sizing_max_blocks
        max_blocks = max(int(max_blocks or 0), 0)
        if applied is None:
            applied = final_num_gpu_blocks != original_num_gpu_blocks

        derived_budget = max(
            self.global_budget_bytes
            - self.pool_budgets["weights"]
            - self.pool_budgets["scratch"]
            - self.safety_margin_bytes,
            0,
        )
        if explicit_kv_budget_bytes > 0:
            budget_mode = "explicit"
            kv_budget_bytes = explicit_kv_budget_bytes
        else:
            budget_mode = "derived"
            kv_budget_bytes = derived_budget
        kv_budget_blocks = (
            kv_budget_bytes // bytes_per_block if bytes_per_block > 0 else 0
        )
        kv_sizing_summary = {
            "kv_sizing_config_present": True,
            "kv_sizing_policy": policy,
            "kv_sizing_max_blocks": max_blocks,
            "kv_sizing_mode": budget_mode,
            "kv_sizing_applied": bool(applied),
            "kv_sizing_report_only": policy == "report_only",
            "kv_sizing_fallback_reason": fallback_reason,
            "kv_sizing_explicit_budget_bytes": explicit_kv_budget_bytes,
            "kv_sizing_derived_budget_bytes": derived_budget,
            "kv_sizing_budget_bytes": kv_budget_bytes,
            "kv_sizing_budget_blocks": kv_budget_blocks,
            "kv_sizing_bytes_per_block": bytes_per_block,
            "kv_sizing_original_num_gpu_blocks": original_num_gpu_blocks,
            "kv_sizing_final_num_gpu_blocks": final_num_gpu_blocks,
            "kv_sizing_original_kv_bytes": original_kv_bytes,
            "kv_sizing_final_kv_bytes": final_kv_bytes,
        }
        with self._lock:
            self._record_config_locked()
            self._sync_driver_pool_budgets_locked()
            self._kv_sizing_summary.update(kv_sizing_summary)
            self._kv_sizing_summary["kv_sizing_records"] = (
                int(self._kv_sizing_summary.get("kv_sizing_records") or 0) + 1
            )
            self._summary["kv_sizing_records"] += 1
            self._write_locked(
                {
                    "action": "residency_kv_sizing",
                    "pool": "kv",
                    **kv_sizing_summary,
                    "weight_gpu_budget_bytes": self.pool_budgets["weights"],
                    "scratch_gpu_budget_bytes": self.pool_budgets["scratch"],
                    "safety_margin_bytes": self.safety_margin_bytes,
                    "global_gpu_budget_bytes": self.global_budget_bytes,
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()

    def record_kv_metadata_candidate(
        self,
        *,
        region_id: str,
        block_id: int,
        group_id: int,
        bytes_per_block: int,
        safe: bool,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        metadata = metadata or {}
        bytes_per_block = max(int(bytes_per_block or 0), 0)
        with self._lock:
            self._record_config_locked()
            if safe:
                self._summary["kv_safe_candidate_records"] += 1
                action = "residency_kv_safe_candidate"
            else:
                self._summary["kv_unsafe_candidate_records"] += 1
                action = "residency_kv_unsafe_candidate"
            self._summary["kv_metadata_records"] += 1
            self._write_locked(
                {
                    "action": action,
                    "pool": "kv",
                    "region_id": region_id,
                    "block_id": int(block_id),
                    "group_id": int(group_id),
                    "bytes_per_block": bytes_per_block,
                    "safe": bool(safe),
                    "reason": reason,
                    "data_offload_issued": False,
                    "metadata": metadata,
                }
            )
            self._record_summary_locked()

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
            self._summary["would_evict_candidates"] += 1
            self._write_locked(
                {
                    "action": "residency_would_evict_candidate",
                    "pool": region.pool,
                    "region_id": region.region_id,
                    "bytes": region.bytes,
                    "state": region.state,
                    "evictable": region.evictable,
                    "active": region.active,
                    "last_access_step": region.last_access_step,
                    "scope_key": scope_key,
                    "request_region_id": request_region_id,
                    "reason": "same_pool_budget_pressure",
                }
            )
            if reclaimed >= required_bytes:
                break
        return victims

    def _sync_driver_region_tag_locked(
        self,
        region: UvmResidencyRegion,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "enabled": self.driver_tag_enable,
            "attempted": False,
            "registered": region.driver_registered,
            "status": region.driver_register_status,
            "reason": None,
            "driver_region_id": region.driver_region_id,
        }
        if not self.driver_tag_enable:
            record["reason"] = "driver_tag_disabled"
            return record
        if region.driver_registered:
            record["reason"] = "already_registered"
            return record
        if region.driver_region_id <= 0:
            self._summary["driver_tag_register_skips"] += 1
            record["reason"] = "invalid_driver_region_id"
            return record
        if region.pool not in DRIVER_POOL_IDS or region.pool == "unknown":
            self._summary["driver_tag_register_skips"] += 1
            record["reason"] = "unknown_pool"
            return record
        if not isinstance(region.ptr, int) or region.ptr <= 0 or region.bytes <= 0:
            self._summary["driver_tag_register_skips"] += 1
            record["reason"] = "missing_ptr_or_size"
            return record
        if (
            self.driver_tag_require_managed
            and region.metadata.get("driver_capable") is False
        ):
            self._summary["driver_tag_register_skips"] += 1
            record["reason"] = (
                region.metadata.get("driver_capability_reason")
                or "not_driver_capable"
            )
            return record

        try:
            from vllm.device_allocator.uvm import driver_register_region
        except Exception as exc:  # pragma: no cover - diagnostic path
            self._summary["driver_tag_register_failures"] += 1
            record["reason"] = "import_failed"
            record["error"] = repr(exc)
            return record

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
        self._summary["driver_tag_register_attempts"] += 1
        record.update(
            {
                "attempted": True,
                "pool_id": pool_id,
                "role_id": role_id,
                "flags": flags,
                "layer_id": layer_id,
                "expert_id": expert_id,
            }
        )
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
        record["status"] = status
        if status >= 0:
            region.driver_registered = True
            self._summary["driver_tag_register_success"] += 1
            record["registered"] = True
            record["reason"] = "registered"
        else:
            self._summary["driver_tag_register_failures"] += 1
            record["registered"] = False
            record["reason"] = f"driver_register_failed:{status}"
        return record

    def _sync_driver_pool_budgets_locked(self, force: bool = False) -> None:
        if self._driver_pool_budgets_applied and not force:
            return
        if not self.driver_pool_budget_enable:
            return

        try:
            from vllm.device_allocator.uvm import driver_set_pool_budget
        except Exception as exc:  # pragma: no cover - diagnostic path
            self._driver_pool_budget_status["import_error"] = -3
            self._write_locked(
                {
                    "action": "residency_driver_pool_budget_config",
                    "enabled": self.driver_pool_budget_enable,
                    "reason": "import_failed",
                    "error": repr(exc),
                }
            )
            return

        records: list[dict[str, Any]] = []
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
            records.append(
                {
                    "pool": pool,
                    "pool_id": pool_id,
                    "budget_bytes": budget,
                    "mode": self.driver_pool_budget_mode,
                    "status": status,
                }
            )

        self._write_locked(
            {
                "action": "residency_driver_pool_budget_config",
                "enabled": self.driver_pool_budget_enable,
                "records": records,
            }
        )
        self._driver_pool_budgets_applied = not retry_later

    def _sync_driver_region_state_locked(
        self,
        region: UvmResidencyRegion,
    ) -> dict[str, Any]:
        state_id = _driver_region_state_id(region.state, active=region.active)
        flags = _driver_flags(
            pool=region.pool,
            state=region.state,
            evictable=region.evictable,
            active=region.active,
            metadata=region.metadata,
        )
        record: dict[str, Any] = {
            "enabled": self.driver_tag_enable,
            "attempted": False,
            "status": None,
            "reason": None,
            "driver_region_id": region.driver_region_id,
            "state_id": state_id,
            "flags": flags,
        }
        if not self.driver_tag_enable:
            record["reason"] = "driver_tag_disabled"
            return record
        if not region.driver_registered:
            self._summary["driver_region_state_update_skips"] += 1
            record["reason"] = "driver_region_not_registered"
            return record

        try:
            from vllm.device_allocator.uvm import driver_update_region_state
        except Exception as exc:  # pragma: no cover - diagnostic path
            self._summary["driver_region_state_update_failures"] += 1
            record["reason"] = "import_failed"
            record["error"] = repr(exc)
            return record

        self._summary["driver_region_state_update_attempts"] += 1
        record["attempted"] = True
        status = driver_update_region_state(
            region_id=region.driver_region_id,
            state=state_id,
            flags_set=flags,
            flags_clear=DRIVER_FLAG_MASK & ~flags,
            priority=_metadata_int(region.metadata, "priority", 0),
            last_access_epoch=max(int(region.last_access_step or 0), 0),
        )
        record["status"] = status
        if status >= 0:
            self._summary["driver_region_state_update_success"] += 1
            record["reason"] = "updated"
        else:
            self._summary["driver_region_state_update_failures"] += 1
            record["reason"] = f"driver_state_update_failed:{status}"
        return record

    def _record_request_locked(
        self,
        decision: UvmResidencyDecision,
        metadata: dict[str, Any],
    ) -> None:
        self._summary["requests"] += 1
        if decision.would_deny:
            self._summary["would_denies"] += 1
        if decision.allowed:
            self._summary["grants"] += 1
            outcome = "residency_grant"
        else:
            self._summary["denies"] += 1
            outcome = "residency_deny"
        self._write_locked(
            {
                "action": "residency_request",
                "outcome": outcome,
                "pool": decision.pool,
                "region_id": decision.region_id,
                "requested_bytes": decision.requested_bytes,
                "incremental_bytes": decision.incremental_bytes,
                "scope_key": decision.scope_key,
                "allowed": decision.allowed,
                "would_deny": decision.would_deny,
                "reason": decision.reason,
                "mode": decision.mode,
                "pool_budget_bytes": decision.pool_budget_bytes,
                "pool_estimated_gpu_resident_bytes": (
                    decision.pool_estimated_gpu_resident_bytes
                ),
                "pool_remaining_bytes": decision.pool_remaining_bytes,
                "global_budget_bytes": decision.global_budget_bytes,
                "global_estimated_gpu_resident_bytes": (
                    decision.global_estimated_gpu_resident_bytes
                ),
                "global_remaining_bytes": decision.global_remaining_bytes,
                "victims": list(decision.victims),
                "metadata": metadata,
            }
        )
        self._record_summary_locked()

    def _record_config_locked(self) -> None:
        if self._config_recorded or not self.trace_file:
            return
        self._config_recorded = True
        self._write_locked(
            {
                "action": "residency_config",
                "enabled": self.enabled,
                "mode": self.mode,
                "tag_only_mode": self.tag_only_mode,
                "driver_tag_enable": self.driver_tag_enable,
                "driver_tag_require_managed": self.driver_tag_require_managed,
                "driver_pool_budget_enable": self.driver_pool_budget_enable,
                "driver_pool_budget_mode": self.driver_pool_budget_mode,
                "driver_pool_budget_overrides": self.driver_pool_budget_overrides,
                "driver_temp_buffer_bytes": self.driver_temp_buffer_bytes,
                "driver_pool_state_snapshot_on_tag_only_every": (
                    self.driver_pool_state_snapshot_on_tag_only_every
                ),
                "driver_pool_state_snapshot_device": (
                    self.driver_pool_state_snapshot_device
                ),
                "trace_file": self.trace_file,
                "global_budget_bytes": self.global_budget_bytes,
                "pool_budgets": self.pool_budgets,
                "safety_margin_bytes": self.safety_margin_bytes,
                "priority": self.priority,
                "estimated_residency_semantics": (
                    "application_managed_intent_not_driver_page_residency"
                ),
                "l0_trace_only_no_real_reclaim": True,
                "stage_label": self.stage_label,
                "weight_initial_state": self.weight_initial_state,
                "weight_enforce_hot_set": self.weight_enforce_hot_set,
                "weight_sync_offload": self.weight_sync_offload,
                "weight_max_victims_per_step": self.weight_max_victims_per_step,
                "weight_victim_policy": self.weight_victim_policy,
                "scratch_reserved_bytes": self.scratch_reserved_bytes,
                "scratch_margin_bytes": self.scratch_margin_bytes,
                "kv_sizing_enable": self.kv_sizing_enable,
                "kv_sizing_policy": self.kv_sizing_policy,
                "kv_sizing_max_blocks": self.kv_sizing_max_blocks,
                "kv_metadata_enable": self.kv_metadata_enable,
                "kv_data_offload_enable": self.kv_data_offload_enable,
                "kv_prefetch_back_enable": self.kv_prefetch_back_enable,
                "real_residency_enable": self.real_residency_enable,
                "real_residency_trace_file": self.real_residency_trace_file,
                "real_residency_device": self.real_residency_device,
                "real_residency_on_register_every": (
                    self.real_residency_on_register_every
                ),
                "real_residency_on_action_every": self.real_residency_on_action_every,
                "real_residency_max_snapshots": self.real_residency_max_snapshots,
                "real_residency_max_regions_per_pool": (
                    self.real_residency_max_regions_per_pool
                ),
            }
        )

    def _record_summary_locked(self) -> None:
        self._write_locked(self._summary_record_locked())

    def _summary_record_locked(self) -> dict[str, Any]:
        pool_region_counts = {pool: 0 for pool in VALID_POOLS}
        for region in self._regions.values():
            if region.pool in pool_region_counts:
                pool_region_counts[region.pool] += 1
        return {
            "action": "residency_summary",
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
            "pool_region_counts": pool_region_counts,
            "region_records": len(self._regions),
            "real_residency_enabled": self.real_residency_enable,
            "tag_only_mode": self.tag_only_mode,
            "driver_tag_enable": self.driver_tag_enable,
            "driver_pool_budget_enable": self.driver_pool_budget_enable,
            "driver_pool_budget_status": dict(self._driver_pool_budget_status),
            "real_residency_trace_file": self.real_residency_trace_file,
            "real_residency_snapshots": self._real_residency_snapshots,
            "real_residency_errors": self._real_residency_errors,
            "real_residency_last": self._last_real_residency,
            **self._scratch_summary,
            **self._kv_sizing_summary,
            **self._summary,
        }

    def _global_estimated_bytes_locked(self) -> int:
        return sum(self._pool_estimated_bytes.values())

    def _record_real_residency_locked(
        self,
        *,
        trigger: str,
        trigger_count: int | None = None,
        trigger_region_id: str | None = None,
        trigger_action: str | None = None,
    ) -> dict[str, Any]:
        if not self.real_residency_enable:
            return {"enabled": False, "reason": "disabled"}
        if (
            self.real_residency_max_snapshots > 0
            and self._real_residency_snapshots >= self.real_residency_max_snapshots
        ):
            return {"enabled": True, "skipped": True, "reason": "max_snapshots"}

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

        try:
            from vllm.device_allocator.uvm import (
                driver_batch_query_ranges_residency_bytes,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            self._real_residency_errors += 1
            record = {
                "action": "real_residency_snapshot",
                "ok": False,
                "trigger": trigger,
                "trigger_count": trigger_count,
                "trigger_region_id": trigger_region_id,
                "trigger_action": trigger_action,
                "reason": "import_failed",
                "error": repr(exc),
                "device": self.real_residency_device,
            }
            self._write_real_locked(record)
            self._last_real_residency = record
            return record

        totals = _empty_real_residency_totals()
        by_pool: dict[str, dict[str, Any]] = {}
        ok = True
        max_regions = self.real_residency_max_regions_per_pool
        query_batch_size = 64
        for pool, pool_regions in regions_by_pool.items():
            queried_regions = pool_regions[:max_regions] if max_regions > 0 else pool_regions
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

        self._real_residency_snapshots += 1
        if not ok:
            self._real_residency_errors += 1

        record = {
            "action": "real_residency_snapshot",
            "ok": ok,
            "snapshot_index": self._real_residency_snapshots,
            "trigger": trigger,
            "trigger_count": trigger_count,
            "trigger_region_id": trigger_region_id,
            "trigger_action": trigger_action,
            "device": self.real_residency_device,
            "region_records": len(self._regions),
            "skipped_regions": skipped_regions,
            "max_regions_per_pool": max_regions,
            "query_batch_size": query_batch_size,
            **totals,
            "by_pool": by_pool,
        }
        self._last_real_residency = record
        self._write_real_locked(record)
        return record

    def _write_locked(self, record: dict[str, Any]) -> None:
        if not self.trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("stage", self.stage_label)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
        with open(self.trace_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _write_real_locked(self, record: dict[str, Any]) -> None:
        if not self.real_residency_trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("stage", self.stage_label)
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        os.makedirs(os.path.dirname(self.real_residency_trace_file) or ".", exist_ok=True)
        with open(self.real_residency_trace_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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
        metadata=metadata,
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


def record_uvm_residency_action(
    *,
    action: str,
    pool: str,
    region_id: str | None = None,
    bytes: int = 0,
    metadata: dict[str, Any] | None = None,
    **extra: Any,
) -> None:
    get_uvm_pool_residency_manager().record_action(
        action=action,
        pool=pool,
        region_id=region_id,
        bytes=bytes,
        metadata=metadata,
        **extra,
    )


def record_uvm_residency_scratch_summary(
    *,
    scratch_live_bytes: int = 0,
    scratch_peak_bytes: int = 0,
    scratch_reserved_bytes: int | None = None,
    scratch_margin_bytes: int | None = None,
    scratch_pool_enabled: bool | None = None,
    scratch_pool_mode: str | None = None,
    scratch_trace_records: int = 0,
    scratch_eligible_records: int = 0,
    scratch_device_direct_records: int = 0,
    scratch_budget_over_records: int = 0,
    scratch_budget_reject_records: int = 0,
    scratch_eviction_records: int = 0,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().record_scratch_summary(
        scratch_live_bytes=scratch_live_bytes,
        scratch_peak_bytes=scratch_peak_bytes,
        scratch_reserved_bytes=scratch_reserved_bytes,
        scratch_margin_bytes=scratch_margin_bytes,
        scratch_pool_enabled=scratch_pool_enabled,
        scratch_pool_mode=scratch_pool_mode,
        scratch_trace_records=scratch_trace_records,
        scratch_eligible_records=scratch_eligible_records,
        scratch_device_direct_records=scratch_device_direct_records,
        scratch_budget_over_records=scratch_budget_over_records,
        scratch_budget_reject_records=scratch_budget_reject_records,
        scratch_eviction_records=scratch_eviction_records,
        metadata=metadata,
    )


def record_uvm_residency_kv_sizing(
    *,
    bytes_per_block: int,
    original_num_gpu_blocks: int,
    final_num_gpu_blocks: int | None = None,
    original_kv_bytes: int | None = None,
    final_kv_bytes: int | None = None,
    explicit_kv_budget_bytes: int | None = None,
    applied: bool | None = None,
    policy: str | None = None,
    max_blocks: int | None = None,
    fallback_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().record_kv_sizing(
        bytes_per_block=bytes_per_block,
        original_num_gpu_blocks=original_num_gpu_blocks,
        final_num_gpu_blocks=final_num_gpu_blocks,
        original_kv_bytes=original_kv_bytes,
        final_kv_bytes=final_kv_bytes,
        explicit_kv_budget_bytes=explicit_kv_budget_bytes,
        applied=applied,
        policy=policy,
        max_blocks=max_blocks,
        fallback_reason=fallback_reason,
        metadata=metadata,
    )


def record_uvm_residency_kv_metadata_candidate(
    *,
    region_id: str,
    block_id: int,
    group_id: int,
    bytes_per_block: int,
    safe: bool,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_residency_manager().record_kv_metadata_candidate(
        region_id=region_id,
        block_id=block_id,
        group_id=group_id,
        bytes_per_block=bytes_per_block,
        safe=safe,
        reason=reason,
        metadata=metadata,
    )
