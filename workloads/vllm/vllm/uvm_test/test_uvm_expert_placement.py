# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

_HELPER = (
    Path(__file__).resolve().parents[1]
    / "vllm"
    / "device_allocator"
    / "uvm_expert_placement.py"
)
_SPEC = importlib.util.spec_from_file_location("uvm_expert_placement", _HELPER)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

ExpertKey = _MODULE.ExpertKey
ExpertPlacement = _MODULE.ExpertPlacement
ExpertRegion = _MODULE.ExpertRegion
KvCachePressure = _MODULE.KvCachePressure
ReservedPoolConfig = _MODULE.ReservedPoolConfig
assign_reserved_slots = _MODULE.assign_reserved_slots
build_expert_placement = _MODULE.build_expert_placement
plan_reserved_expert_assignments = _MODULE.plan_reserved_expert_assignments


def test_low_kv_pressure_keeps_selected_experts_gpu_resident():
    selected = [ExpertKey(0, 1), ExpertKey(0, 2)]
    placements = build_expert_placement(
        selected=selected,
        pressure=KvCachePressure.LOW,
        cpu_offloaded={ExpertKey(0, 2)},
    )

    assert placements == {
        ExpertKey(0, 1): ExpertPlacement.GPU_RESIDENT,
        ExpertKey(0, 2): ExpertPlacement.GPU_RESIDENT,
    }


def test_high_kv_pressure_marks_only_offloaded_selected_experts():
    selected = [ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)]
    placements = build_expert_placement(
        selected=selected,
        pressure=KvCachePressure.HIGH,
        cpu_offloaded={ExpertKey(0, 2), ExpertKey(9, 9)},
    )

    assert placements[ExpertKey(0, 1)] == ExpertPlacement.GPU_RESIDENT
    assert placements[ExpertKey(0, 2)] == ExpertPlacement.CPU_OFFLOADED_RESERVED
    assert placements[ExpertKey(0, 3)] == ExpertPlacement.GPU_RESIDENT


def test_assign_reserved_slots_respects_fixed_window_capacity():
    regions = [
        ExpertRegion(ExpertKey(0, 1), region_id=11, bytes=2 << 20),
        ExpertRegion(ExpertKey(0, 2), region_id=12, bytes=2 << 20),
        ExpertRegion(ExpertKey(0, 3), region_id=13, bytes=2 << 20),
    ]
    assignments, overflow = assign_reserved_slots(
        regions,
        pool=ReservedPoolConfig(pool_id=7, slot_count=2, slot_bytes=2 << 20),
    )

    assert [(a.region_id, a.pool_id, a.slot_id) for a in assignments] == [
        (11, 7, 0),
        (12, 7, 1),
    ]
    assert [region.region_id for region in overflow] == [13]


def test_assign_reserved_slots_groups_multiple_regions_per_expert():
    key = ExpertKey(0, 1)
    regions = [
        ExpertRegion(key, region_id=11, bytes=3, role="moe_gate_up"),
        ExpertRegion(key, region_id=12, bytes=5, role="moe_down"),
    ]
    assignments, overflow = assign_reserved_slots(
        regions,
        pool=ReservedPoolConfig(pool_id=7, slot_count=1, slot_bytes=8),
    )

    assert overflow == []
    assert [(a.region_id, a.slot_id, a.slot_offset, a.bytes) for a in assignments] == [
        (11, 0, 0, 3),
        (12, 0, 3, 5),
    ]


def test_plan_reserved_assignments_filters_gpu_resident_experts():
    selected = [ExpertKey(0, 1), ExpertKey(0, 2), ExpertKey(0, 3)]
    regions = {
        ExpertKey(0, 2): [
            ExpertRegion(
                ExpertKey(0, 2),
                region_id=42,
                bytes=1 << 20,
                role="moe_gate_up",
            ),
            ExpertRegion(
                ExpertKey(0, 2),
                region_id=44,
                bytes=1 << 20,
                role="moe_down",
            ),
        ],
        ExpertKey(0, 3): ExpertRegion(
            ExpertKey(0, 3),
            region_id=43,
            bytes=2 << 20,
            role="moe_down",
        ),
    }
    plan = plan_reserved_expert_assignments(
        selected=selected,
        regions_by_expert=regions,
        pressure="high",
        cpu_offloaded={ExpertKey(0, 2), ExpertKey(0, 3)},
        pool=ReservedPoolConfig(pool_id=3, slot_count=1, slot_bytes=2 << 20),
    )

    assert plan.pressure == KvCachePressure.HIGH
    assert plan.gpu_resident == [ExpertKey(0, 1)]
    assert plan.cpu_offloaded == [ExpertKey(0, 2), ExpertKey(0, 3)]
    assert [(a.key, a.region_id, a.slot_id, a.slot_offset) for a in plan.assignments] == [
        (ExpertKey(0, 2), 42, 0, 0),
        (ExpertKey(0, 2), 44, 0, 1 << 20),
    ]
    assert [region.key for region in plan.overflow] == [ExpertKey(0, 3)]
    assert not plan.ok
