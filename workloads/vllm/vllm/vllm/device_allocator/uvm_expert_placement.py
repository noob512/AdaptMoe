# SPDX-License-Identifier: Apache-2.0
"""Phase-0 expert placement and reserved backing assignment helpers.

This module is a user-space control-plane shim. It does not issue UVM ioctls or
move memory. It decides which selected experts must use the future
reserved-backed path and assigns those experts to a fixed reusable window.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class ExpertPlacement(str, Enum):
    GPU_RESIDENT = "GPU_RESIDENT"
    CPU_OFFLOADED_RESERVED = "CPU_OFFLOADED_RESERVED"


class KvCachePressure(str, Enum):
    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class ExpertKey:
    layer_id: int
    expert_id: int


@dataclass(frozen=True)
class ExpertRegion:
    key: ExpertKey
    region_id: int
    bytes: int
    role: str = "unknown"


@dataclass(frozen=True)
class ReservedPoolConfig:
    pool_id: int = 0
    slot_count: int = 0
    slot_bytes: int = 0

    @property
    def capacity_bytes(self) -> int:
        return max(self.slot_count, 0) * max(self.slot_bytes, 0)


@dataclass(frozen=True)
class ReservedAssignment:
    key: ExpertKey
    region_id: int
    pool_id: int
    slot_id: int
    slot_offset: int
    bytes: int
    role: str


@dataclass(frozen=True)
class ExpertPlacementPlan:
    pressure: KvCachePressure
    placements: dict[ExpertKey, ExpertPlacement]
    assignments: list[ReservedAssignment]
    gpu_resident: list[ExpertKey]
    cpu_offloaded: list[ExpertKey]
    overflow: list[ExpertRegion]
    reserved_capacity_bytes: int
    reserved_requested_bytes: int

    @property
    def ok(self) -> bool:
        return not self.overflow


def normalize_kv_cache_pressure(value: str | KvCachePressure | bool) -> KvCachePressure:
    if isinstance(value, KvCachePressure):
        return value
    if isinstance(value, bool):
        return KvCachePressure.HIGH if value else KvCachePressure.LOW

    normalized = str(value or "low").strip().lower().replace("-", "_")
    if normalized in ("high", "pressure_high", "kv_high", "over_budget", "1", "true"):
        return KvCachePressure.HIGH
    return KvCachePressure.LOW


def build_expert_placement(
    *,
    selected: Iterable[ExpertKey],
    pressure: str | KvCachePressure | bool,
    cpu_offloaded: Iterable[ExpertKey] = (),
) -> dict[ExpertKey, ExpertPlacement]:
    """Return per-selected-expert placement for the current step.

    Low KV pressure means all selected experts remain GPU resident. High KV
    pressure only marks experts present in `cpu_offloaded` for the reserved
    path; hot experts not in that set remain ordinary GPU-resident experts.
    """

    selected_keys = list(dict.fromkeys(selected))
    pressure_state = normalize_kv_cache_pressure(pressure)
    if pressure_state == KvCachePressure.LOW:
        return {key: ExpertPlacement.GPU_RESIDENT for key in selected_keys}

    offloaded_set = set(cpu_offloaded)
    return {
        key: (
            ExpertPlacement.CPU_OFFLOADED_RESERVED
            if key in offloaded_set
            else ExpertPlacement.GPU_RESIDENT
        )
        for key in selected_keys
    }


def assign_reserved_slots(
    regions: Sequence[ExpertRegion],
    *,
    pool: ReservedPoolConfig,
    start_slot: int = 0,
    allow_oversized: bool = False,
) -> tuple[list[ReservedAssignment], list[ExpertRegion]]:
    """Assign offloaded expert regions to a fixed reserved backing window.

    The unit of slot ownership is an expert, not an individual weight region.
    Multiple regions for the same expert share one reserved slot and are laid
    out at increasing offsets inside that slot.
    """

    if pool.slot_count <= 0 or pool.slot_bytes <= 0:
        return [], list(regions)

    assignments: list[ReservedAssignment] = []
    overflow: list[ExpertRegion] = []
    slot_cursor = start_slot % pool.slot_count
    grouped: dict[ExpertKey, list[ExpertRegion]] = {}

    for region in regions:
        grouped.setdefault(region.key, []).append(region)

    for expert_regions in grouped.values():
        total_bytes = 0
        invalid = False
        for region in expert_regions:
            region_bytes = max(int(region.bytes or 0), 0)
            if region_bytes <= 0:
                invalid = True
                break
            total_bytes += region_bytes

        if invalid:
            overflow.extend(expert_regions)
            continue
        if total_bytes > pool.slot_bytes and not allow_oversized:
            overflow.extend(expert_regions)
            continue
        if len(assignments) >= pool.slot_count:
            overflow.extend(expert_regions)
            continue

        slot_id = slot_cursor
        slot_cursor = (slot_cursor + 1) % pool.slot_count
        slot_offset = 0
        for region in expert_regions:
            region_bytes = max(int(region.bytes or 0), 0)
            assignments.append(
                ReservedAssignment(
                    key=region.key,
                    region_id=int(region.region_id),
                    pool_id=int(pool.pool_id),
                    slot_id=int(slot_id),
                    slot_offset=slot_offset,
                    bytes=region_bytes,
                    role=region.role,
                )
            )
            slot_offset += region_bytes

    return assignments, overflow


def _flatten_regions_by_expert(
    regions_by_expert: Mapping[ExpertKey, ExpertRegion | Sequence[ExpertRegion]],
    keys: Iterable[ExpertKey],
) -> tuple[list[ExpertRegion], list[ExpertRegion]]:
    regions: list[ExpertRegion] = []
    missing: list[ExpertRegion] = []

    for key in keys:
        value = regions_by_expert.get(key)
        if value is None:
            missing.append(ExpertRegion(key=key, region_id=0, bytes=0))
            continue
        if isinstance(value, ExpertRegion):
            regions.append(value)
        else:
            expert_regions = list(value)
            if expert_regions:
                regions.extend(expert_regions)
            else:
                missing.append(ExpertRegion(key=key, region_id=0, bytes=0))

    return regions, missing


def plan_reserved_expert_assignments(
    *,
    selected: Iterable[ExpertKey],
    regions_by_expert: Mapping[ExpertKey, ExpertRegion | Sequence[ExpertRegion]],
    pressure: str | KvCachePressure | bool,
    cpu_offloaded: Iterable[ExpertKey],
    pool: ReservedPoolConfig,
    start_slot: int = 0,
) -> ExpertPlacementPlan:
    """Build the Phase-0 placement and reserved assignment plan."""

    pressure_state = normalize_kv_cache_pressure(pressure)
    placements = build_expert_placement(
        selected=selected,
        pressure=pressure_state,
        cpu_offloaded=cpu_offloaded,
    )

    gpu_resident = [
        key for key, placement in placements.items()
        if placement == ExpertPlacement.GPU_RESIDENT
    ]
    cpu_offloaded_selected = [
        key for key, placement in placements.items()
        if placement == ExpertPlacement.CPU_OFFLOADED_RESERVED
    ]

    offloaded_regions, missing_regions = _flatten_regions_by_expert(
        regions_by_expert,
        cpu_offloaded_selected,
    )
    assignments, overflow = assign_reserved_slots(
        offloaded_regions,
        pool=pool,
        start_slot=start_slot,
    )

    return ExpertPlacementPlan(
        pressure=pressure_state,
        placements=placements,
        assignments=assignments,
        gpu_resident=gpu_resident,
        cpu_offloaded=cpu_offloaded_selected,
        overflow=[*missing_regions, *overflow],
        reserved_capacity_bytes=pool.capacity_bytes,
        reserved_requested_bytes=sum(region.bytes for region in offloaded_regions),
    )
