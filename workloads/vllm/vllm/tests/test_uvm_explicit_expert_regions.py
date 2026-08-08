# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.device_allocator.uvm_region import (
    ExplicitRegionRegistry,
    ExpertRecord,
    RegionState,
    compile_expert_envelopes,
    update_aggregate_state,
)
from vllm.model_executor.layers.fused_moe import layer as layer_module
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def test_compile_expert_envelopes_records_only_explicit_ids() -> None:
    owners: dict[str, tuple[int, int]] = {}
    record = compile_expert_envelopes(
        layer_id=7,
        global_expert_id=3,
        slices=[(0x100000, 0x80000), (0x200000, 0x80000)],
        envelope_bytes=0x80000,
        existing_owners=owners,
    )

    assert record == ExpertRecord(
        layer_id=7,
        global_expert_id=3,
        aggregate_state=RegionState.UNKNOWN,
        envelope_region_ids=[
            "weights:7:3:envelope:100000:80000",
            "weights:7:3:envelope:200000:80000",
        ],
    )
    assert owners == {
        "100000:80000": (7, 3),
        "200000:80000": (7, 3),
    }


def test_compile_expert_envelopes_rejects_shared_ownership() -> None:
    owners: dict[str, tuple[int, int]] = {}
    compile_expert_envelopes(
        layer_id=7,
        global_expert_id=3,
        slices=[(0x100000, 0x80000)],
        envelope_bytes=0x80000,
        existing_owners=owners,
    )

    with pytest.raises(ValueError, match="shared by experts"):
        compile_expert_envelopes(
            layer_id=7,
            global_expert_id=4,
            slices=[(0x140000, 0x1000)],
            envelope_bytes=0x80000,
            existing_owners=owners,
        )


def test_aggregate_state_requires_every_envelope() -> None:
    record = ExpertRecord(
        layer_id=7,
        global_expert_id=3,
        envelope_region_ids=["first", "second"],
    )

    state = update_aggregate_state(
        record,
        {"first": RegionState.CPU, "second": RegionState.UNKNOWN},
        target=RegionState.CPU,
    )
    assert state == RegionState.UNKNOWN

    state = update_aggregate_state(
        record,
        {"first": RegionState.CPU, "second": RegionState.CPU},
        target=RegionState.CPU,
    )
    assert state == RegionState.CPU


def test_registry_moves_only_named_expert_regions() -> None:
    registry = ExplicitRegionRegistry()
    first = registry.register_region("first", 0x100000, 0x80000, "gpu")
    second = registry.register_region("second", 0x180000, 0x80000, "gpu")
    untouched = registry.register_region("untouched", 0x200000, 0x80000, "gpu")
    record = ExpertRecord(
        layer_id=7,
        global_expert_id=3,
        aggregate_state=RegionState.GPU,
        envelope_region_ids=[first.region_id, second.region_id],
    )
    registry.register_expert(record)
    moved: list[tuple[int, int]] = []

    result = registry.release_regions(
        record.envelope_region_ids,
        lambda base, length: moved.append((base, length)) or True,
    )
    registry.update_expert_state(record.key, RegionState.CPU)

    assert result["ok"] is True
    assert record.aggregate_state == RegionState.CPU
    assert moved == [(first.base, first.length), (second.base, second.length)]
    assert untouched.state == RegionState.GPU


def test_registry_marks_partial_move_unknown() -> None:
    registry = ExplicitRegionRegistry()
    first = registry.register_region("first", 0x100000, 0x80000, "gpu")
    second = registry.register_region("second", 0x180000, 0x80000, "gpu")
    record = ExpertRecord(
        layer_id=7,
        global_expert_id=3,
        aggregate_state=RegionState.GPU,
        envelope_region_ids=[first.region_id, second.region_id],
    )
    registry.register_expert(record)

    result = registry.release_regions(
        record.envelope_region_ids,
        lambda base, _length: base == first.base,
    )
    registry.update_expert_state(record.key, RegionState.CPU)

    assert result["ok"] is False
    assert record.aggregate_state == RegionState.UNKNOWN
    assert first.state == RegionState.CPU
    assert second.state == RegionState.UNKNOWN


def test_dynamic_swap_uses_explicit_expert_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    records = {
        (7, 3): ExpertRecord(7, 3, RegionState.CPU, ["remote"]),
        (7, 4): ExpertRecord(7, 4, RegionState.GPU, ["local"]),
    }

    def release(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("release", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.CPU
        return {"ok": True}

    def restore(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("restore", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.GPU
        return {"ok": True}

    layer = SimpleNamespace(
        uvm_dynamic_expert_swap_enable=True,
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_layer_id=lambda: 7,
        _uvm_remote_read_global_expert_ids={3},
        _uvm_expert_records=records,
        _uvm_release_expert_record=release,
        _uvm_restore_expert_record=restore,
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = FusedMoE.run_uvm_dynamic_remote_read_swaps(
        layer,
        [{"promote_expert": 3, "demote_expert": 4}],
    )

    assert result["ok"] is True
    assert calls == [("release", 4), ("restore", 3)]
    assert layer._uvm_remote_read_global_expert_ids == {4}
    assert records[(7, 3)].aggregate_state == RegionState.GPU
    assert records[(7, 4)].aggregate_state == RegionState.CPU


def test_expert_release_preserves_remote_read_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = ExpertRecord(7, 3, RegionState.GPU, ["first", "second"])
    calls: list[dict[str, object]] = []

    def release(
        region_ids: list[str],
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"region_ids": list(region_ids), **kwargs})
        return {"ok": True}

    layer = SimpleNamespace(
        _uvm_expert_records={(7, 3): record},
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_record_expert_state=lambda _layer_id, _expert_id, state: setattr(
            record,
            "aggregate_state",
            state,
        ),
    )
    monkeypatch.setattr(layer_module, "release_uvm_regions", release)

    result = FusedMoE._uvm_release_expert_record(
        layer,
        layer_id=7,
        global_expert_id=3,
        device=1,
    )

    assert result["ok"] is True
    assert calls == [
        {
            "region_ids": ["first", "second"],
            "device": 1,
            "remote_read": True,
        }
    ]
    assert record.aggregate_state == RegionState.CPU
