# SPDX-License-Identifier: Apache-2.0
"""Stage K global UVM pool action coordinator.

The coordinator is deliberately a control-plane shim: it grants or denies
high-level actions at safe vLLM call sites. It does not migrate raw allocator
pointers by itself.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

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
        logger.warning(
            "Ignoring invalid %s=%r; expected non-negative integer.", name, raw
        )
        return default


def _read_mode_env() -> str:
    mode = os.environ.get("VLLM_UVM_POOL_COORDINATOR_MODE", "trace_only")
    mode = mode.strip().lower()
    if mode not in ("trace_only", "enforce"):
        logger.warning(
            "Ignoring invalid VLLM_UVM_POOL_COORDINATOR_MODE=%r; using trace_only.",
            mode,
        )
        return "trace_only"
    return mode


@dataclass(frozen=True)
class UvmPoolCoordinatorDecision:
    enabled: bool
    mode: str
    pool: str
    action: str
    requested_bytes: int
    allowed: bool
    would_deny: bool
    reason: str
    scope_key: str
    pool_budget_bytes: int
    pool_used_bytes: int
    pool_remaining_bytes: int | None
    global_budget_bytes: int
    global_used_bytes: int
    global_remaining_bytes: int | None


class UvmPoolCoordinator:
    """Coordinate cross-pool Stage I/J/K action budgets."""

    def __init__(self) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_POOL_COORDINATOR_ENABLE", False)
        self.mode = _read_mode_env()
        self.global_budget_bytes = _read_int_env(
            "VLLM_UVM_POOL_COORDINATOR_GLOBAL_BYTES_PER_STEP", 0
        )
        self.pool_budgets = {
            "kv": _read_int_env("VLLM_UVM_POOL_COORDINATOR_KV_BYTES_PER_STEP", 0),
            "weights": _read_int_env(
                "VLLM_UVM_POOL_COORDINATOR_WEIGHT_BYTES_PER_STEP", 0
            ),
            "scratch": _read_int_env(
                "VLLM_UVM_POOL_COORDINATOR_SCRATCH_BYTES_PER_STEP", 0
            ),
        }
        self._lock = threading.Lock()
        self._scope_used: dict[str, dict[str, int]] = {}

    @property
    def active(self) -> bool:
        return self.enabled

    def request_action(
        self,
        *,
        pool: str,
        action: str,
        requested_bytes: int,
        scope_key: str | None = None,
        metadata: dict[str, Any] | None = None,
        ) -> UvmPoolCoordinatorDecision:
        """Request budget for a high-level pool action.

        In trace_only mode the action is always allowed, but the decision reports
        whether enforce mode would have denied it.
        """
        pool = (pool or "unknown").strip().lower()
        action = (action or "unknown").strip().lower()
        requested_bytes = max(int(requested_bytes or 0), 0)
        scope_key = scope_key or "global"
        if not self.active or requested_bytes <= 0:
            return UvmPoolCoordinatorDecision(
                enabled=self.enabled,
                mode=self.mode,
                pool=pool,
                action=action,
                requested_bytes=requested_bytes,
                allowed=True,
                would_deny=False,
                reason="coordinator_disabled_or_zero_bytes",
                scope_key=scope_key,
                pool_budget_bytes=0,
                pool_used_bytes=0,
                pool_remaining_bytes=None,
                global_budget_bytes=0,
                global_used_bytes=0,
                global_remaining_bytes=None,
            )

        with self._lock:
            scope = self._scope_used.setdefault(scope_key, {})
            pool_used = scope.get(pool, 0)
            global_used = scope.get("__global__", 0)
            pool_budget = self.pool_budgets.get(pool, 0)
            pool_remaining = (
                max(pool_budget - pool_used, 0) if pool_budget > 0 else None
            )
            global_remaining = (
                max(self.global_budget_bytes - global_used, 0)
                if self.global_budget_bytes > 0
                else None
            )
            pool_over = pool_budget > 0 and requested_bytes > (pool_remaining or 0)
            global_over = (
                self.global_budget_bytes > 0
                and requested_bytes > (global_remaining or 0)
            )
            would_deny = pool_over or global_over
            allowed = self.mode != "enforce" or not would_deny
            if global_over:
                reason = "global_budget_exceeded"
            elif pool_over:
                reason = f"{pool}_budget_exceeded"
            else:
                reason = "granted"

            if allowed:
                scope[pool] = pool_used + requested_bytes
                scope["__global__"] = global_used + requested_bytes

            return UvmPoolCoordinatorDecision(
                enabled=True,
                mode=self.mode,
                pool=pool,
                action=action,
                requested_bytes=requested_bytes,
                allowed=allowed,
                would_deny=would_deny,
                reason=reason,
                scope_key=scope_key,
                pool_budget_bytes=pool_budget,
                pool_used_bytes=pool_used,
                pool_remaining_bytes=pool_remaining,
                global_budget_bytes=self.global_budget_bytes,
                global_used_bytes=global_used,
                global_remaining_bytes=global_remaining,
            )

    def reset_scope(self, scope_key: str) -> None:
        if not self.active:
            return
        with self._lock:
            self._scope_used.pop(scope_key, None)


_COORDINATOR: UvmPoolCoordinator | None = None
_COORDINATOR_LOCK = threading.Lock()


def get_uvm_pool_coordinator() -> UvmPoolCoordinator:
    global _COORDINATOR
    if _COORDINATOR is None:
        with _COORDINATOR_LOCK:
            if _COORDINATOR is None:
                _COORDINATOR = UvmPoolCoordinator()
    return _COORDINATOR


def request_uvm_pool_action(
    *,
    pool: str,
    action: str,
    requested_bytes: int,
    scope_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> UvmPoolCoordinatorDecision:
    return get_uvm_pool_coordinator().request_action(
        pool=pool,
        action=action,
        requested_bytes=requested_bytes,
        scope_key=scope_key,
        metadata=metadata,
    )
