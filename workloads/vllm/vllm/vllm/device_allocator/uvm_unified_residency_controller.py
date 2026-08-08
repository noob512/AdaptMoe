# SPDX-License-Identifier: Apache-2.0
"""Unified HBM residency controller for KV and MoE experts.

Phase U1 observes only. Phase U2 can drive application-level KV controls. The
old driver pool-budget/reclaim control plane is intentionally unsupported.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
from typing import Any

from vllm.logger import init_logger
from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_residency_regions_snapshot,
    get_uvm_residency_snapshot,
)

logger = init_logger(__name__)


_KV_REGION_RE = re.compile(r"^kv:(?P<group_id>\d+):(?P<block_id>\d+)(?::|$)")
_UNSUPPORTED_DRIVER_POOL_POLICY_REASON = "unsupported_explicit_region_only"


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
        logger.warning("Ignoring invalid %s=%r; expected integer.", name, raw)
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, str) and value.startswith("0x"):
            return int(value, 16)
        return int(value)
    except (TypeError, ValueError):
        return default


def _object_identity(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "object_type",
        "region_id",
        "pool",
        "bytes",
        "residency_state",
        "group_id",
        "block_id",
        "layer_id",
        "expert_id",
    )
    return {key: record.get(key) for key in keys if record.get(key) is not None}


class UnifiedResidencyController:
    """Unified KV/expert controller.

    U1 methods remain trace-only. U2 manual control-file application is the
    first path that can request real action through existing lower-level APIs.
    """

    def __init__(self) -> None:
        self.enabled = _read_bool_env("VLLM_UVM_UNIFIED_RESIDENCY_ENABLE", False)
        self.trace_file = os.environ.get(
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE",
            "vllm_uvm_unified_residency_trace.jsonl",
        )
        self.trace_objects = _read_bool_env(
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_OBJECTS",
            True,
        )
        self.max_objects_per_snapshot = _read_int_env(
            "VLLM_UVM_UNIFIED_RESIDENCY_MAX_OBJECTS_PER_SNAPSHOT",
            0,
        )
        self.managed_hbm_budget_override = _read_int_env(
            "VLLM_UVM_UNIFIED_MANAGED_HBM_BUDGET_BYTES",
            0,
        )
        self.runtime_safety_margin_override = _read_int_env(
            "VLLM_UVM_UNIFIED_RUNTIME_SAFETY_MARGIN_BYTES",
            0,
        )
        self.kv_min_resident_budget = _read_int_env(
            "VLLM_UVM_UNIFIED_KV_MIN_RESIDENT_BYTES",
            0,
        )
        self.kv_target_resident_budget = _read_int_env(
            "VLLM_UVM_UNIFIED_KV_TARGET_RESIDENT_BYTES",
            0,
        )
        self.kv_max_resident_budget = _read_int_env(
            "VLLM_UVM_UNIFIED_KV_MAX_RESIDENT_BYTES",
            0,
        )
        self.expert_hot_budget = _read_int_env(
            "VLLM_UVM_UNIFIED_EXPERT_HOT_BUDGET_BYTES",
            0,
        )
        self.expert_streaming_budget = _read_int_env(
            "VLLM_UVM_UNIFIED_EXPERT_STREAMING_BUDGET_BYTES",
            0,
        )
        self.control_file = os.environ.get(
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE",
            "",
        )
        self.control_enabled = _read_bool_env(
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_ENABLE",
            bool(self.control_file),
        )
        self._lock = threading.Lock()
        self._config_recorded = False
        self._step_id = 0
        self._last_kv_target_blocks: int | None = None
        self._control_mtime_ns: int | None = None
        self._control_size: int | None = None
        self._control_generation = 0

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.trace_file)

    def on_service_start(self) -> dict[str, Any]:
        return self.snapshot(reason="service_start")

    def apply_control_file(
        self,
        *,
        kv_runtime_policy: Any | None = None,
        block_pool: Any | None = None,
        reason: str = "poll",
    ) -> dict[str, Any]:
        """Apply the U2 manual control file once when it changes."""
        if not self.active or not self.control_enabled or not self.control_file:
            return {}
        try:
            stat = os.stat(self.control_file)
        except FileNotFoundError:
            return {}
        except OSError as exc:  # pragma: no cover - diagnostic path
            result = {
                "action": "unified_control_file_failed",
                "trace_only": False,
                "control_file": self.control_file,
                "reason": reason,
                "error": repr(exc),
            }
            self._write(result)
            return result

        if (
            stat.st_mtime_ns == self._control_mtime_ns
            and stat.st_size == self._control_size
        ):
            return {}
        self._control_mtime_ns = stat.st_mtime_ns
        self._control_size = stat.st_size
        try:
            with open(
                self.control_file,
                "r",
                encoding="utf-8",
                errors="replace",
            ) as handle:
                raw = handle.read()
        except OSError as exc:  # pragma: no cover - diagnostic path
            result = {
                "action": "unified_control_file_failed",
                "trace_only": False,
                "control_file": self.control_file,
                "reason": reason,
                "error": repr(exc),
            }
            self._write(result)
            return result

        commands = self._parse_control_text(raw)
        with self._lock:
            self._control_generation += 1
            generation = self._control_generation
        command_results = [
            self._execute_control_command(
                command,
                kv_runtime_policy=kv_runtime_policy,
                block_pool=block_pool,
                generation=generation,
                reason=reason,
            )
            for command in commands
        ]
        result = {
            "action": "unified_control_file_applied",
            "trace_only": False,
            "control_file": self.control_file,
            "reason": reason,
            "generation": generation,
            "command_count": len(commands),
            "commands": commands,
            "results": command_results,
        }
        self._record_config_once()
        self._write(result)
        self.snapshot(
            reason="unified_control_file_applied",
            kv_pressure={
                "control_generation": generation,
                "control_command_count": len(commands),
            },
        )
        return result

    def on_scheduler_step(
        self,
        kv_pressure: dict[str, Any] | None = None,
        routing_plan: dict[str, Any] | None = None,
        *,
        reason: str = "scheduler_step",
    ) -> dict[str, Any]:
        if not self.active:
            return {}
        with self._lock:
            self._step_id += 1
            step_id = self._step_id
        snapshot = self.snapshot(
            reason=reason,
            step_id=step_id,
            kv_pressure=kv_pressure,
            routing_plan=routing_plan,
        )
        self._write(
            {
                "action": "unified_scheduler_step",
                "trace_only": True,
                "step_id": step_id,
                "reason": reason,
                "kv_pressure": kv_pressure or {},
                "routing_plan": routing_plan or {},
                "kv_resident_bytes": snapshot.get("kv_resident_bytes", 0),
                "expert_resident_bytes": snapshot.get("expert_resident_bytes", 0),
                "runtime_safety_margin": snapshot.get("runtime_safety_margin", 0),
            }
        )
        return snapshot

    def before_layer(
        self,
        layer_id: int | str,
        predicted_experts: list[Any] | tuple[Any, ...] | None = None,
        *,
        reason: str = "before_layer",
    ) -> dict[str, Any]:
        if not self.active:
            return {}
        experts = list(predicted_experts or [])
        self._write(
            {
                "action": "unified_layer_boundary",
                "trace_only": True,
                "boundary": "before",
                "layer_id": layer_id,
                "predicted_experts": experts,
                "reason": reason,
            }
        )
        if experts:
            self.prefetch_experts(layer_id=layer_id, experts=experts)
        return self.snapshot(reason=reason, routing_plan={
            "layer_id": layer_id,
            "predicted_experts": experts,
        })

    def after_layer(
        self,
        layer_id: int | str,
        actual_experts: list[Any] | tuple[Any, ...] | None = None,
        *,
        reason: str = "after_layer",
    ) -> dict[str, Any]:
        if not self.active:
            return {}
        experts = list(actual_experts or [])
        self._write(
            {
                "action": "unified_layer_boundary",
                "trace_only": True,
                "boundary": "after",
                "layer_id": layer_id,
                "actual_experts": experts,
                "reason": reason,
            }
        )
        return self.snapshot(reason=reason, routing_plan={
            "layer_id": layer_id,
            "actual_experts": experts,
        })

    def resize_kv_target(
        self,
        target_blocks: int,
        *,
        old_target_blocks: int | None = None,
        reason: str = "kv_target_request",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        target = max(int(target_blocks or 0), 0)
        old = old_target_blocks
        if old is None:
            old = self._last_kv_target_blocks
        direction = "set"
        if old is not None:
            direction = "grow" if target > old else "shrink" if target < old else "set"
        self._last_kv_target_blocks = target
        action = (
            "unified_kv_grow_request"
            if direction == "grow"
            else "unified_kv_shrink_request"
            if direction == "shrink"
            else "unified_kv_target_request"
        )
        self._write(
            {
                "action": action,
                "trace_only": True,
                "old_target_blocks": old,
                "new_target_blocks": target,
                "direction": direction,
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def reclaim_expert_for_kv(
        self,
        required_bytes: int,
        *,
        reason: str = "kv_pressure",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        self._write(
            {
                "action": "unified_expert_reclaim_request",
                "trace_only": True,
                "required_bytes": max(int(required_bytes or 0), 0),
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def prefetch_experts(
        self,
        layer_id: int | str,
        experts: list[Any] | tuple[Any, ...],
        *,
        reason: str = "routing_prediction",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.active:
            return
        self._write(
            {
                "action": "unified_expert_prefetch_request",
                "trace_only": True,
                "layer_id": layer_id,
                "experts": list(experts),
                "reason": reason,
                "metadata": metadata or {},
            }
        )

    def snapshot(
        self,
        *,
        reason: str = "manual",
        step_id: int | None = None,
        kv_pressure: dict[str, Any] | None = None,
        routing_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.active:
            return {}

        residency = get_uvm_residency_snapshot()
        regions = get_uvm_residency_regions_snapshot()
        objects = [self._object_from_region(region) for region in regions]
        objects = [record for record in objects if record is not None]

        totals = self._totals_from_objects(objects, residency)
        object_limit = self.max_objects_per_snapshot
        visible_objects = objects
        truncated = False
        if object_limit > 0 and len(visible_objects) > object_limit:
            visible_objects = visible_objects[:object_limit]
            truncated = True

        snapshot_record: dict[str, Any] = {
            "trace_only": True,
            "step_id": step_id,
            "reason": reason,
            "object_count": len(objects),
            "objects_truncated": truncated,
            "kv_pressure": kv_pressure or {},
            "routing_plan": routing_plan or {},
            **totals,
        }
        if self.trace_objects:
            snapshot_record["objects"] = visible_objects

        self._record_config_once()
        self._write({"action": "unified_budget_update", **snapshot_record})
        self._write({"action": "unified_object_state", **snapshot_record})
        return snapshot_record

    def _record_config_once(self) -> None:
        if self._config_recorded:
            return
        self._config_recorded = True
        self._write(
            {
                "action": "unified_config",
                "trace_only": True,
                "enabled": self.enabled,
                "trace_file": self.trace_file,
                "trace_objects": self.trace_objects,
                "max_objects_per_snapshot": self.max_objects_per_snapshot,
                "control_enabled": self.control_enabled,
                "control_file": self.control_file,
                "driver_pool_policy": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
                "managed_hbm_budget_override": (
                    self.managed_hbm_budget_override
                ),
                "runtime_safety_margin_override": (
                    self.runtime_safety_margin_override
                ),
                "kv_min_resident_budget": self.kv_min_resident_budget,
                "kv_target_resident_budget": self.kv_target_resident_budget,
                "kv_max_resident_budget": self.kv_max_resident_budget,
                "expert_hot_budget": self.expert_hot_budget,
                "expert_streaming_budget": self.expert_streaming_budget,
                "phase": self._phase_label(),
                "semantics": (
                    "manual_control_file"
                    if self.control_enabled
                    else "trace_only_no_physical_migration"
                ),
            }
        )

    def _parse_control_text(self, raw: str) -> list[dict[str, Any]]:
        text = (raw or "").strip()
        if not text:
            return []
        if text.startswith("{") or text.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                return [{"invalid_control": text, "error": str(exc)}]
            if isinstance(data, list):
                return [
                    dict(item)
                    for item in data
                    if isinstance(item, dict)
                ]
            if isinstance(data, dict):
                commands = data.get("commands")
                if isinstance(commands, list):
                    return [
                        dict(item)
                        for item in commands
                        if isinstance(item, dict)
                    ]
                return [dict(data)]
            return [{"invalid_control": text, "error": "json_root_not_object"}]

        commands: list[dict[str, Any]] = []
        for raw_line in text.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            command: dict[str, Any] = {}
            for token in shlex.split(line):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                command[key.strip()] = value.strip()
            if command:
                commands.append(command)
        return commands

    def _execute_control_command(
        self,
        command: dict[str, Any],
        *,
        kv_runtime_policy: Any | None,
        block_pool: Any | None,
        generation: int,
        reason: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "command": command,
            "generation": generation,
            "reason": reason,
            "results": [],
        }
        if "invalid_control" in command:
            result["ok"] = False
            result["results"].append({
                "action": "invalid_control",
                "ok": False,
                "error": command.get("error"),
            })
            self._write({
                "action": "unified_control_command_result",
                "trace_only": False,
                **result,
            })
            return result

        if "weights_budget_bytes" in command:
            budget = _as_int(command.get("weights_budget_bytes"), 0)
            update = {
                "ok": False,
                "pool": "weights",
                "budget_bytes": budget,
                "status": -95,
                "reason": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
            }
            result["results"].append({
                "action": "weights_budget_bytes",
                **update,
            })
            self._write({
                "action": "unified_weights_budget_update",
                "trace_only": False,
                "generation": generation,
                "requested_budget_bytes": budget,
                "result": update,
            })

        if "kv_target_blocks" in command:
            target_blocks = _as_int(command.get("kv_target_blocks"), 0)
            if kv_runtime_policy is None or not hasattr(
                kv_runtime_policy,
                "apply_unified_kv_target_blocks",
            ):
                target_result = {
                    "ok": False,
                    "reason": "kv_runtime_policy_unavailable",
                    "target_blocks": target_blocks,
                }
            else:
                target_result = kv_runtime_policy.apply_unified_kv_target_blocks(
                    target_blocks,
                    block_pool=block_pool,
                    reason="unified_control_file",
                    metadata={
                        "generation": generation,
                        "command": command,
                    },
                )
            result["results"].append({
                "action": "kv_target_blocks",
                **target_result,
            })
            self._write({
                "action": "unified_kv_target_applied",
                "trace_only": False,
                "generation": generation,
                "requested_target_blocks": target_blocks,
                "result": target_result,
            })

        if "reclaim_pool" in command:
            pool = str(command.get("reclaim_pool") or "")
            required_bytes = _as_int(
                command.get("required_bytes", command.get("target_bytes")),
                0,
            )
            reclaim_result = {
                "ok": False,
                "pool": pool,
                "required_bytes": required_bytes,
                "status": -95,
                "reason": _UNSUPPORTED_DRIVER_POOL_POLICY_REASON,
            }
            result["results"].append({
                "action": "reclaim_pool",
                **reclaim_result,
            })
            self._write({
                "action": "unified_driver_reclaim_pool",
                "trace_only": False,
                "generation": generation,
                **reclaim_result,
            })

        if "prefetch_kv_blocks" in command:
            block_specs = self._parse_kv_block_specs(
                command.get("prefetch_kv_blocks")
            )
            if kv_runtime_policy is None or not hasattr(
                kv_runtime_policy,
                "prefetch_kv_blocks_from_unified",
            ):
                prefetch_result = {
                    "ok": False,
                    "reason": "kv_runtime_policy_unavailable",
                    "block_specs": block_specs,
                }
            else:
                prefetch_result = kv_runtime_policy.prefetch_kv_blocks_from_unified(
                    block_specs,
                    block_pool=block_pool,
                    reason="unified_control_file",
                    metadata={
                        "generation": generation,
                        "command": command,
                    },
                )
            result["results"].append({
                "action": "prefetch_kv_blocks",
                **prefetch_result,
            })
            self._write({
                "action": "unified_prefetch_kv_blocks",
                "trace_only": False,
                "generation": generation,
                "block_specs": block_specs,
                "result": prefetch_result,
            })

        result["ok"] = all(bool(item.get("ok", True)) for item in result["results"])
        self._write({
            "action": "unified_control_command_result",
            "trace_only": False,
            **result,
        })
        return result

    def _parse_kv_block_specs(self, raw: Any) -> list[dict[str, int | None]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            specs: list[dict[str, int | None]] = []
            for item in raw:
                if isinstance(item, dict):
                    specs.append({
                        "group_id": (
                            _as_int(item.get("group_id"))
                            if item.get("group_id") is not None
                            else None
                        ),
                        "block_id": _as_int(item.get("block_id"), -1),
                    })
                else:
                    specs.extend(self._parse_kv_block_specs(item))
            return [
                spec for spec in specs
                if spec.get("block_id") is not None and int(spec["block_id"]) >= 0
            ]
        text = str(raw).strip()
        if not text:
            return []
        specs = []
        for item in text.split(","):
            part = item.strip()
            if not part:
                continue
            group_id: int | None = None
            block_part = part
            if ":" in part:
                group_text, block_part = part.split(":", 1)
                group_id = _as_int(group_text, -1)
                if group_id < 0:
                    continue
            for block_id in self._expand_block_ids(block_part):
                specs.append({"group_id": group_id, "block_id": block_id})
        return specs

    def _expand_block_ids(self, text: str) -> list[int]:
        value = text.strip()
        if ".." in value:
            start_text, end_text = value.split("..", 1)
        elif "-" in value:
            start_text, end_text = value.split("-", 1)
        else:
            block_id = _as_int(value, -1)
            return [block_id] if block_id >= 0 else []
        start = _as_int(start_text, -1)
        end = _as_int(end_text, -1)
        if start < 0 or end < start:
            return []
        return list(range(start, end + 1))

    def _object_from_region(
        self,
        region: dict[str, Any],
    ) -> dict[str, Any] | None:
        pool = str(region.get("pool") or "unknown")
        region_id = str(region.get("region_id") or "")
        object_type = self._object_type(pool, region_id)
        if object_type not in ("kv", "expert", "weight"):
            return None

        record: dict[str, Any] = {
            "object_type": object_type,
            "region_id": region_id,
            "driver_region_id": region.get("driver_region_id"),
            "driver_registered": bool(region.get("driver_registered")),
            "driver_register_status": region.get("driver_register_status"),
            "pool": pool,
            "ptr": region.get("ptr"),
            "bytes": max(_as_int(region.get("bytes"), 0), 0),
            "residency_state": str(region.get("state") or "unknown"),
            "estimated_gpu_resident": bool(
                region.get("estimated_gpu_resident")
            ),
            "evictable": bool(region.get("evictable")),
            "active": bool(region.get("active")),
            "last_access_step": region.get("last_access_step"),
            "last_state_update_ts": region.get("last_state_update_ts"),
        }
        if object_type == "kv":
            group_id, block_id = self._kv_ids(region_id)
            record["group_id"] = group_id
            record["block_id"] = block_id
        elif object_type == "expert":
            layer_id, expert_id = self._expert_ids_from_region_id(region_id)
            record["layer_id"] = layer_id
            record["expert_id"] = expert_id
        return record

    def _object_type(
        self,
        pool: str,
        region_id: str,
    ) -> str:
        if pool == "kv" or region_id.startswith("kv:"):
            return "kv"
        if pool == "weights" and region_id.startswith("weights:") and ":envelope:" in region_id:
            return "expert"
        return "weight" if pool == "weights" else pool

    def _kv_ids(
        self,
        region_id: str,
    ) -> tuple[int | None, int | None]:
        match = _KV_REGION_RE.match(region_id)
        if not match:
            return None, None
        return int(match.group("group_id")), int(match.group("block_id"))

    def _expert_ids_from_region_id(
        self,
        region_id: str,
    ) -> tuple[int | None, int | None]:
        parts = region_id.split(":")
        if len(parts) < 4 or parts[0] != "weights" or parts[3] != "envelope":
            return None, None
        return _as_int(parts[1], None), _as_int(parts[2], None)

    def _totals_from_objects(
        self,
        objects: list[dict[str, Any]],
        residency: dict[str, Any],
    ) -> dict[str, Any]:
        kv_resident = sum(
            int(record.get("bytes") or 0)
            for record in objects
            if record.get("object_type") == "kv"
            and bool(record.get("estimated_gpu_resident"))
        )
        expert_resident = sum(
            int(record.get("bytes") or 0)
            for record in objects
            if record.get("object_type") == "expert"
            and bool(record.get("estimated_gpu_resident"))
        )
        weight_resident = sum(
            int(record.get("bytes") or 0)
            for record in objects
            if record.get("object_type") == "weight"
            and bool(record.get("estimated_gpu_resident"))
        )
        managed_hbm_budget = (
            self.managed_hbm_budget_override
            or _as_int(residency.get("global_gpu_budget_bytes"), 0)
        )
        runtime_safety_margin = (
            self.runtime_safety_margin_override
            or _as_int(residency.get("safety_margin_bytes"), 0)
        )
        managed_used = kv_resident + expert_resident + runtime_safety_margin
        object_counts: dict[str, int] = {}
        for record in objects:
            object_type = str(record.get("object_type") or "unknown")
            object_counts[object_type] = object_counts.get(object_type, 0) + 1
        return {
            "managed_hbm_budget": managed_hbm_budget,
            "runtime_safety_margin": runtime_safety_margin,
            "kv_min_resident_budget": self.kv_min_resident_budget,
            "kv_target_resident_budget": self.kv_target_resident_budget,
            "kv_max_resident_budget": self.kv_max_resident_budget,
            "expert_hot_budget": self.expert_hot_budget,
            "expert_streaming_budget": self.expert_streaming_budget,
            "kv_budget_bytes": _as_int(residency.get("kv_gpu_budget_bytes"), 0),
            "weights_budget_bytes": _as_int(
                residency.get("weight_gpu_budget_bytes"),
                0,
            ),
            "scratch_budget_bytes": _as_int(
                residency.get("scratch_gpu_budget_bytes"),
                0,
            ),
            "kv_resident_bytes": kv_resident,
            "expert_resident_bytes": expert_resident,
            "weight_resident_bytes": weight_resident,
            "controlled_resident_bytes": kv_resident + expert_resident,
            "managed_hbm_accounted_bytes": managed_used,
            "budget_ok": managed_hbm_budget <= 0 or managed_used <= managed_hbm_budget,
            "object_counts": object_counts,
            "residency_summary": {
                "kv_estimated_gpu_resident_bytes": _as_int(
                    residency.get("kv_estimated_gpu_resident_bytes"),
                    0,
                ),
                "weights_estimated_gpu_resident_bytes": _as_int(
                    residency.get("weights_estimated_gpu_resident_bytes"),
                    0,
                ),
                "global_estimated_gpu_resident_bytes": _as_int(
                    residency.get("global_estimated_gpu_resident_bytes"),
                    0,
                ),
                "region_records": _as_int(residency.get("region_records"), 0),
            },
        }

    def _write(self, record: dict[str, Any]) -> None:
        if not self.trace_file:
            return
        record.setdefault("ts", time.time())
        record.setdefault("phase", self._phase_label())
        record.setdefault("trace_schema", "unified_hbm_residency_v1")
        line = json.dumps(record, sort_keys=True, separators=(",", ":"))
        os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
        with open(self.trace_file, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _phase_label(self) -> str:
        return "U2" if self.control_enabled else "U1"


_UNIFIED_CONTROLLER: UnifiedResidencyController | None = None
_UNIFIED_CONTROLLER_LOCK = threading.Lock()


def get_uvm_unified_residency_controller() -> UnifiedResidencyController:
    global _UNIFIED_CONTROLLER
    if _UNIFIED_CONTROLLER is None:
        with _UNIFIED_CONTROLLER_LOCK:
            if _UNIFIED_CONTROLLER is None:
                _UNIFIED_CONTROLLER = UnifiedResidencyController()
    return _UNIFIED_CONTROLLER


def record_unified_residency_snapshot(
    *,
    reason: str = "manual",
    kv_pressure: dict[str, Any] | None = None,
    routing_plan: dict[str, Any] | None = None,
    scheduler_step: bool = False,
) -> dict[str, Any]:
    controller = get_uvm_unified_residency_controller()
    if scheduler_step:
        return controller.on_scheduler_step(
            kv_pressure=kv_pressure,
            routing_plan=routing_plan,
            reason=reason,
        )
    return controller.snapshot(
        reason=reason,
        kv_pressure=kv_pressure,
        routing_plan=routing_plan,
    )


def record_unified_kv_target(
    target_blocks: int,
    *,
    old_target_blocks: int | None = None,
    reason: str = "kv_target_request",
    metadata: dict[str, Any] | None = None,
) -> None:
    get_uvm_unified_residency_controller().resize_kv_target(
        target_blocks,
        old_target_blocks=old_target_blocks,
        reason=reason,
        metadata=metadata,
    )


def record_unified_layer_boundary(
    *,
    layer_id: int | str,
    predicted_experts: list[Any] | tuple[Any, ...] | None = None,
    actual_experts: list[Any] | tuple[Any, ...] | None = None,
    boundary: str = "before",
    reason: str | None = None,
) -> dict[str, Any]:
    controller = get_uvm_unified_residency_controller()
    if boundary == "after":
        return controller.after_layer(
            layer_id,
            actual_experts=actual_experts,
            reason=reason or "after_layer",
        )
    return controller.before_layer(
        layer_id,
        predicted_experts=predicted_experts,
        reason=reason or "before_layer",
    )
