# SPDX-License-Identifier: Apache-2.0
"""Stage K global UVM pool action coordinator.

The coordinator is deliberately a control-plane shim: it grants or denies
high-level actions at safe vLLM call sites, and records a unified JSONL trace.
It does not migrate raw allocator pointers by itself.
"""

from __future__ import annotations

import json
import os
import threading
import time
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
        logger.warning("Ignoring invalid %s=%r; expected non-negative integer.", name, raw)
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
        self.trace_enabled = _read_bool_env(
            "VLLM_UVM_POOL_COORDINATOR_TRACE_ENABLE", True
        )
        self.trace_file = os.environ.get(
            "VLLM_UVM_POOL_COORDINATOR_TRACE_FILE",
            "vllm_uvm_pool_coordinator_stage_k.jsonl",
        )
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
        self.priority = os.environ.get(
            "VLLM_UVM_POOL_COORDINATOR_PRIORITY", "kv,weights,scratch"
        )
        self.priority_order = [
            item.strip().lower()
            for item in self.priority.split(",")
            if item.strip()
        ]
        self._lock = threading.Lock()
        self._config_recorded = False
        self._scope_used: dict[str, dict[str, int]] = {}
        self._summary = {
            "requests": 0,
            "grants": 0,
            "denies": 0,
            "would_denies": 0,
            "requested_bytes": 0,
            "granted_bytes": 0,
            "denied_bytes": 0,
            "kv_requests": 0,
            "weights_requests": 0,
            "scratch_requests": 0,
        }

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

        In trace_only mode the action is always allowed, but the decision records
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
            self._record_config_locked()
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

            decision = UvmPoolCoordinatorDecision(
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
            self._record_decision_locked(decision, metadata or {})
            return decision

    def record_pressure(
        self,
        *,
        pool: str,
        pressure_bytes: int,
        pressure_ratio: float | None = None,
        action_queue_depth: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        with self._lock:
            self._record_config_locked()
            self._write_locked(
                {
                    "action": "coordinator_pressure",
                    "pool": pool,
                    "pressure_bytes": max(int(pressure_bytes or 0), 0),
                    "pressure_ratio": pressure_ratio,
                    "action_queue_depth": action_queue_depth,
                    "metadata": metadata or {},
                }
            )

    def reset_scope(self, scope_key: str) -> None:
        if not self.active:
            return
        with self._lock:
            removed = self._scope_used.pop(scope_key, None)
            self._write_locked(
                {
                    "action": "coordinator_scope_reset",
                    "scope_key": scope_key,
                    "removed": bool(removed),
                }
            )

    def _record_config_locked(self) -> None:
        if self._config_recorded:
            return
        self._config_recorded = True
        self._write_locked(
            {
                "action": "coordinator_config",
                "enabled": self.enabled,
                "mode": self.mode,
                "trace_file": self.trace_file,
                "global_budget_bytes": self.global_budget_bytes,
                "pool_budgets": self.pool_budgets,
                "priority": self.priority,
                "priority_semantics": "global_action_order_only",
                "per_pool_safety_semantics": "budget_and_callsite_rules_are_independent",
            }
        )

    def _record_decision_locked(
        self,
        decision: UvmPoolCoordinatorDecision,
        metadata: dict[str, Any],
    ) -> None:
        self._summary["requests"] += 1
        self._summary["requested_bytes"] += decision.requested_bytes
        pool_key = f"{decision.pool}_requests"
        if pool_key in self._summary:
            self._summary[pool_key] += 1
        if decision.would_deny:
            self._summary["would_denies"] += 1
        if decision.allowed:
            self._summary["grants"] += 1
            self._summary["granted_bytes"] += decision.requested_bytes
            outcome = "coordinator_grant"
        else:
            self._summary["denies"] += 1
            self._summary["denied_bytes"] += decision.requested_bytes
            outcome = "coordinator_deny"

        record = {
            "action": "coordinator_request",
            "outcome": outcome,
            "pool": decision.pool,
            "requested_action": decision.action,
            "requested_bytes": decision.requested_bytes,
            "allowed": decision.allowed,
            "would_deny": decision.would_deny,
            "reason": decision.reason,
            "scope_key": decision.scope_key,
            "mode": decision.mode,
            "pool_budget_bytes": decision.pool_budget_bytes,
            "pool_used_bytes": decision.pool_used_bytes,
            "pool_remaining_bytes": decision.pool_remaining_bytes,
            "global_budget_bytes": decision.global_budget_bytes,
            "global_used_bytes": decision.global_used_bytes,
            "global_remaining_bytes": decision.global_remaining_bytes,
            "priority": self.priority,
            "priority_index": (
                self.priority_order.index(decision.pool)
                if decision.pool in self.priority_order
                else None
            ),
            "priority_rank": (
                self.priority_order.index(decision.pool) + 1
                if decision.pool in self.priority_order
                else None
            ),
            "priority_semantics": "global_action_order_only",
            "per_pool_safety_semantics": "budget_and_callsite_rules_are_independent",
            "global_budget_considered": self.global_budget_bytes > 0,
            "metadata": metadata,
        }
        self._write_locked(record)
        self._write_locked({"action": "coordinator_summary", **self._summary})

    def _write_locked(self, record: dict[str, Any]) -> None:
        if not self.trace_enabled or not self.trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("stage", "stage_k")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
        with open(self.trace_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")


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


def record_uvm_pool_pressure(
    *,
    pool: str,
    pressure_bytes: int,
    pressure_ratio: float | None = None,
    action_queue_depth: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_pool_coordinator().record_pressure(
        pool=pool,
        pressure_bytes=pressure_bytes,
        pressure_ratio=pressure_ratio,
        action_queue_depth=action_queue_depth,
        metadata=metadata,
    )
