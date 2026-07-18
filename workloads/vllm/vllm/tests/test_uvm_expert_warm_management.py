# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from vllm.device_allocator import uvm_expert_reserved_executor as reserved_module
from vllm.device_allocator.uvm_expert_reserved_executor import (
    ExpertReservedRegionsExecutor,
    ReservedRegionItem,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_stage_m_orders_cold_before_warm_and_excludes_hot() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_enable = True
    layer.uvm_init_weight_warm_offload_enable = True
    layer.uvm_init_weight_complete_expert_plan = True
    layer.uvm_init_weight_hot_plan_keys = {(7, 1)}
    layer.uvm_weight_prefetch_plan_keys = set()
    layer.uvm_init_weight_offload_plan_keys = {(7, 3), (7, 4)}
    layer.uvm_init_weight_warm_plan_keys = {(7, 2)}
    layer._uvm_slot_managed_global_expert_ids = set()

    pairs = [(expert_id, expert_id) for expert_id in range(6)]

    assert layer._uvm_stage_m_offload_expert_pairs(7, pairs) == [
        (3, 3),
        (4, 4),
        (2, 2),
    ]
    layer.uvm_init_weight_warm_plan_keys = set()
    assert layer._uvm_stage_m_offload_expert_pairs(7, pairs) == [
        (3, 3),
        (4, 4),
        (0, 0),
        (2, 2),
        (5, 5),
    ]
    layer.uvm_init_weight_warm_offload_enable = False
    assert layer._uvm_stage_m_offload_expert_pairs(7, pairs) == [
        (3, 3),
        (4, 4),
    ]
    assert layer._uvm_init_weight_expert_class(
        layer_id=7,
        global_expert_id=1,
    ) == "hot"
    assert layer._uvm_init_weight_expert_class(
        layer_id=7,
        global_expert_id=3,
    ) == "cold"
    assert layer._uvm_init_weight_expert_class(
        layer_id=7,
        global_expert_id=2,
    ) == "warm"


def test_slot_managed_mask_only_tracks_released_experts() -> None:
    layer = object.__new__(FusedMoE)
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_offloaded_warm_global_expert_ids = set()
    layer._uvm_reserved_offload_mask_tensor_cache = {("stale", "cpu", -1): None}
    layer.uvm_expert_reserved_offloaded_ids = set()
    layer._uvm_plan_expert_bitmask_width = 8

    layer._uvm_mark_slot_managed_expert(
        global_expert_id=5,
        expert_class="warm",
    )

    assert layer._uvm_slot_managed_global_expert_ids == {5}
    assert layer._uvm_offloaded_warm_global_expert_ids == {5}
    assert layer._uvm_reserved_offload_mask_tensor_cache == {}
    assert layer._uvm_reserved_offload_mask_for_layer(7) == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_stage_m_envelope_cache_is_reused_by_runtime_restore(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_envelope_enable = True
    layer.uvm_expert_envelope_bytes = 512 * 1024
    layer._uvm_expert_envelope_region_cache = {}

    build_calls: list[dict[str, Any]] = []

    def fake_build_envelopes(**kwargs: Any) -> list[dict[str, object]]:
        build_calls.append(kwargs)
        block_bytes = layer.uvm_expert_envelope_bytes
        return [
            {
                "region_id": f"chunk:{index}",
                "driver_region_id": 100 + index,
                "ptr": int(kwargs["ptr"]) + index * block_bytes,
                "bytes": block_bytes,
                "metadata": {
                    **kwargs["metadata"],
                    "logical_region_id": kwargs["source_region_id"],
                    "logical_ptr": kwargs["ptr"],
                    "logical_bytes": kwargs["size_bytes"],
                    "envelope_granularity_bytes": block_bytes,
                    "driver_registered": True,
                    "driver_register_status": 0,
                },
            }
            for index in range(2)
        ]

    monkeypatch.setattr(
        layer,
        "_uvm_build_expert_chunk_envelope_records",
        fake_build_envelopes,
    )

    logical_region_id = "weights:12:6:moe_gate_up"
    stage_items = layer._uvm_get_or_build_expert_envelope_items(
        source_region_id=logical_region_id,
        logical_driver_region_id=6,
        layer_id=12,
        global_expert_id=6,
        local_expert_id=6,
        role="moe_gate_up",
        ptr=0x700000600000,
        size_bytes=1024 * 1024,
        metadata={"active_this_step": False},
        state="gpu",
        source_action="stage_m_driver_release_chunk_envelope",
    )
    stage_items[0].metadata["caller_mutation"] = True

    runtime_logical_item = ReservedRegionItem(
        region_id=logical_region_id,
        driver_region_id=6,
        ptr=0x700000600000,
        bytes=1024 * 1024,
        layer_id=12,
        global_expert_id=6,
        local_expert_id=6,
        role="moe_gate_up",
        metadata={"active_this_step": True, "driver_registered": False},
    )
    runtime_items, envelope_ids_by_logical = (
        layer._uvm_expand_reserved_items_to_chunk_envelopes([runtime_logical_item])
    )

    assert len(build_calls) == 1
    assert list(layer._uvm_expert_envelope_region_cache) == [(12, 6, "moe_gate_up")]
    assert [item.region_id for item in runtime_items] == ["chunk:0", "chunk:1"]
    assert envelope_ids_by_logical == {logical_region_id: ["chunk:0", "chunk:1"]}
    assert runtime_items[0] is not stage_items[0]
    assert runtime_items[0].metadata["active_this_step"] is True
    assert runtime_items[0].metadata["driver_registered"] is True
    assert runtime_items[0].metadata["source"] == ("reserved_prefetch_chunk_envelope")
    assert "caller_mutation" not in runtime_items[0].metadata


def test_stage_m_rejects_release_without_runtime_slot_restore() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_enable = True
    layer.uvm_init_weight_hot_prefetch_enable = False
    layer.uvm_init_weight_register_all_local_experts = False
    layer.uvm_init_weight_offload_mode = "driver_release"
    layer.uvm_expert_reserved_enable = False
    layer.uvm_weight_prefetch_enabled = True
    layer.layer_name = "model.layers.7.mlp.experts"
    traces: list[dict[str, object]] = []
    layer._append_uvm_init_weight_trace = traces.append

    with pytest.raises(RuntimeError, match="requires reserved slot restore"):
        layer.run_uvm_init_weight_offload()

    assert traces[0]["action"] == (
        "init_weight_offload_reject_without_slot_restore"
    )


def test_incremental_offload_wrapper_reopens_one_stage_m_wave(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_incremental_expert_offload_enable = True
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_init_weight_offload_done = True
    layer._uvm_incremental_offload_in_progress = False
    layer._uvm_incremental_allow_warm = False
    layer.layer_name = "model.layers.7.mlp.experts"

    def fake_stage_m() -> None:
        assert layer._uvm_incremental_offload_in_progress is True
        assert layer._uvm_init_weight_offload_done is False
        selected = layer._uvm_incremental_selected_global_expert_id
        if selected is None:
            selected = 4 if layer._uvm_incremental_allow_warm else 6
        layer._uvm_slot_managed_global_expert_ids.add(selected)

    monkeypatch.setattr(layer, "run_uvm_init_weight_offload", fake_stage_m)

    result = layer.run_uvm_incremental_weight_offload(
        allow_warm=False,
        selected_global_expert_id=6,
    )

    assert result["released_experts"] == [6]
    assert result["released_count"] == 1
    assert layer._uvm_incremental_offload_in_progress is False
    assert layer._uvm_incremental_selected_global_expert_id is None


def test_incremental_slot_requirement_counts_aligned_envelopes(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", "2")
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_envelope_enable = True
    layer.uvm_expert_envelope_bytes = 512 * 1024
    layer.uvm_incremental_experts_per_layer_per_round = 1
    layer.uvm_init_weight_offload_target_roles = {
        "moe_gate_up",
        "moe_down",
    }
    layer._uvm_all_local_expert_pairs = lambda: [(6, 6)]  # type: ignore[method-assign]
    layer._uvm_expert_weight_slices = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: [
            ("moe_gate_up", 0x10010000, 512 * 1024),
            ("moe_down", 0x20000000, 512 * 1024),
        ]
    )

    # The unaligned gate-up slice spans two 512 KiB envelopes; moe-down
    # spans one. Two layer-mod banks therefore require six slots.
    assert layer._uvm_expert_reserved_envelope_count(6) == 3
    assert layer.uvm_incremental_reserved_slot_requirement() == 6


def test_incremental_worker_runs_cold_waves_before_warm(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_SYNC", "0")

    class FakeLayer:
        def __init__(
            self,
            layer_name: str,
            *,
            has_cold: bool,
            has_warm: bool,
        ) -> None:
            self.layer_name = layer_name
            self.has_cold = has_cold
            self.has_warm = has_warm
            self.allow_warm_calls: list[bool] = []

        def uvm_next_incremental_expert(
            self,
            expert_class: str,
        ) -> int | None:
            if expert_class == "cold" and self.has_cold:
                return 6
            if expert_class == "warm" and self.has_warm:
                return 4
            return None

        def has_remaining_uvm_incremental_experts(
            self,
            expert_class: str,
        ) -> bool:
            return self.has_cold if expert_class == "cold" else self.has_warm

        def run_uvm_incremental_weight_offload(
            self,
            *,
            allow_warm: bool,
            selected_global_expert_id: int | None = None,
        ) -> dict[str, object]:
            self.allow_warm_calls.append(allow_warm)
            released = 1 if self.has_cold or (allow_warm and self.has_warm) else 0
            if self.has_cold:
                self.has_cold = False
            return {"ok": True, "released_count": released}

    first = FakeLayer("model.layers.7.mlp.experts", has_cold=True, has_warm=True)
    second = FakeLayer("model.layers.8.mlp.experts", has_cold=False, has_warm=True)
    model = type("FakeModel", (), {"modules": lambda self: [first, second]})()
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]

    candidates = runner.get_uvm_incremental_expert_offload_candidates("cold")
    cold_result = runner.run_uvm_incremental_expert_offload()
    warm_result = runner.run_uvm_incremental_expert_offload()

    assert candidates["candidate_count"] == 1
    assert candidates["candidates"] == {
        "model.layers.7.mlp.experts": {
            "layer_name": "model.layers.7.mlp.experts",
            "layer_id": None,
            "global_expert_id": 6,
        }
    }
    assert cold_result["reason"] == "cold_wave"
    assert first.allow_warm_calls[0] is False
    assert second.allow_warm_calls[0] is False
    assert warm_result["reason"] == "warm_wave"
    assert first.allow_warm_calls[1] is True
    assert second.allow_warm_calls[1] is True


def test_warm_async_prefetch_waits_for_token_threshold() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_warm_prefetch_enable = True
    layer.uvm_expert_warm_async_min_tokens = 32
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._uvm_moe_trace_step = 3
    traces: list[dict[str, object]] = []
    layer._append_uvm_weight_prefetch_trace = traces.append

    layer._maybe_prefetch_next_layer_uvm_warm_expert_weights(num_tokens=16)

    assert traces[0]["action"] == (
        "next_layer_warm_prefetch_below_token_threshold"
    )
    assert traces[0]["fallback"] == "synchronous_restore_on_actual_topk_hit"


def test_layer_mod_bank_reuses_prefetched_warm_slot(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY", "layer_mod_bank")
    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", "2")

    executor = ExpertReservedRegionsExecutor(
        device=0,
        pool_id=0,
        slot_count=4,
        slot_bytes=4096,
        strict_footprint=True,
        async_=False,
        wait_on_reuse=True,
        sync_before_compute=True,
        cleanup_synchronize=True,
        prefetch_cpu_source=False,
    )
    executor._prealloc_done = True
    executor.slot_bytes = 4096
    monkeypatch.setattr(executor, "_ensure_driver_region", lambda item: 0)

    driver_calls: list[dict[str, Any]] = []

    def fake_reserved_prefetch(**kwargs: Any) -> dict[str, Any]:
        driver_calls.append(kwargs)
        return {
            "ok": True,
            "compact_result": True,
            "success_count": len(kwargs["assignments"]),
            "failure_count": 0,
            "entries": [],
        }

    sync_calls: list[int] = []
    monkeypatch.setattr(
        reserved_module,
        "driver_expert_reserved_prefetch_regions",
        fake_reserved_prefetch,
    )
    monkeypatch.setattr(
        reserved_module.torch.cuda,
        "synchronize",
        lambda device: sync_calls.append(int(device)),
    )

    item = ReservedRegionItem(
        region_id="weight:layer7:expert3:w13",
        driver_region_id=103,
        ptr=0x100000,
        bytes=4096,
        layer_id=7,
        global_expert_id=3,
        local_expert_id=3,
        role="moe_gate_up",
    )

    warm_result = executor.prefetch_for_layer(
        layer_name="model.layers.7.mlp.experts",
        step=11,
        items=[item],
        async_=True,
        sync_before_compute=False,
    )
    hit_result = executor.prefetch_for_layer(
        layer_name="model.layers.7.mlp.experts",
        step=11,
        items=[item],
        async_=False,
        sync_before_compute=True,
    )

    assert warm_result["success_count"] == 1
    assert warm_result["reused_slot_count"] == 0
    assert hit_result["success_count"] == 1
    assert hit_result["reused_slot_count"] == 1
    assert hit_result["reused_region_ids"] == [item.region_id]
    assert len(driver_calls) == 1
    assert driver_calls[0]["async_"] is True
    assert sync_calls == [0]
