# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest

from vllm.device_allocator import uvm as uvm_module
from vllm.device_allocator import uvm_expert_reserved_executor as reserved_module
from vllm.device_allocator.uvm_expert_reserved_executor import (
    ExpertReservedRegionsExecutor,
    ReservedRegionItem,
)
from vllm.model_executor.layers.fused_moe import layer as layer_module
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_release_remote_entry_requires_complete_cpu_and_remote_bytes() -> None:
    params = uvm_module._ExpertOffloadReleaseParams()
    params.region_id = 11
    params.flags = uvm_module.UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
    params.length = 512 * 1024
    params.block_count = 1
    params.released_bytes = 512 * 1024
    params.cpu_resident_bytes = 512 * 1024
    params.remote_mapped_bytes = 512 * 1024
    params.remote_map_blocks = 1

    complete = uvm_module._expert_offload_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )
    params.remote_mapped_bytes -= 4096
    incomplete = uvm_module._expert_offload_release_entry_to_dict(
        params,
        ret=0,
        saved_errno=0,
    )

    assert complete["ok"] is True
    assert complete["remote_read"] is True
    assert incomplete["ok"] is False
    assert incomplete["reason"] == "remote_mapping_incomplete"


def test_release_remote_wrapper_enables_combined_flag(monkeypatch: Any) -> None:
    captured: dict[str, object] = {}

    def release(region_ids: Any, **kwargs: Any) -> dict[str, object]:
        captured["region_ids"] = list(region_ids)
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(uvm_module, "driver_expert_offload_release_regions", release)

    result = uvm_module.driver_expert_offload_release_remote_regions(
        [10, 11],
        subchunk_64k=True,
    )

    assert result["ok"] is True
    assert captured["region_ids"] == [10, 11]
    assert captured["remote_read"] is True
    assert captured["subchunk_64k"] is True


def test_custom_512k_prefetch_requires_migration_release_and_remote_map(
    monkeypatch: Any,
) -> None:
    envelope_bytes = 512 * 1024
    captured: dict[str, object] = {}

    def release(region_ids: Any, **kwargs: Any) -> dict[str, object]:
        captured["region_ids"] = list(region_ids)
        captured.update(kwargs)
        return {
            "ok": True,
            "entries": [
                {
                    "ok": True,
                    "region_id": 10,
                    "length": envelope_bytes,
                    "released_bytes": envelope_bytes,
                    "cpu_resident_bytes": envelope_bytes,
                    "remote_mapped_bytes": envelope_bytes,
                    "remote_map_failure_blocks": 0,
                },
                {
                    "ok": True,
                    "region_id": 11,
                    "length": envelope_bytes,
                    "released_bytes": envelope_bytes - 64 * 1024,
                    "cpu_resident_bytes": envelope_bytes,
                    "remote_mapped_bytes": envelope_bytes,
                    "remote_map_failure_blocks": 0,
                },
            ],
        }

    monkeypatch.setattr(
        uvm_module,
        "driver_expert_offload_release_remote_regions",
        release,
    )

    result = uvm_module.driver_prefetch_cpu_release_remote_512k_regions([10, 11])

    assert captured["region_ids"] == [10, 11]
    assert captured["subchunk_64k"] is True
    assert result["operation"] == "prefetch_cpu_release_remote_512k"
    assert result["ok"] is False
    assert result["validated_success_count"] == 1
    assert result["validated_failure_count"] == 1
    assert result["entries"][0]["prefetch_release_complete"] is True
    assert result["entries"][1]["prefetch_release_reason"] == (
        "gpu_backing_release_incomplete"
    )


@pytest.mark.parametrize(
    ("access_mode", "prepare_mode", "requires_envelopes", "requires_logical"),
    [
        ("restore", "native", True, False),
        ("remote_read", "native", False, False),
        ("remote_read", "release_remote", True, False),
    ],
)
def test_stage_m_release_registration_granularity(
    access_mode: str,
    prepare_mode: str,
    requires_envelopes: bool,
    requires_logical: bool,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer.uvm_expert_offload_access_mode = access_mode
    layer.uvm_remote_prepare_mode = prepare_mode

    assert layer._uvm_stage_m_requires_release_envelopes() is requires_envelopes
    assert (
        layer._uvm_stage_m_requires_logical_release_registration()
        is requires_logical
    )


def test_empty_weight_plan_is_loaded_only_once(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    plan_file = tmp_path / "empty-plan.json"
    plan_file.write_text(json.dumps({"policy": "routing_heatmap_v1"}))

    layer = object.__new__(FusedMoE)
    layer.uvm_weight_prefetch_plan_file = str(plan_file)
    layer.uvm_weight_offload_plan_file = str(plan_file)
    layer.uvm_init_weight_offload_plan_file = str(plan_file)
    layer.uvm_weight_prefetch_plan_keys = set()
    layer.uvm_weight_offload_plan_keys = set()
    layer.uvm_init_weight_offload_plan_keys = set()
    layer.uvm_init_weight_hot_plan_keys = set()
    layer.uvm_init_weight_warm_plan_keys = set()
    layer.uvm_init_weight_plan_policy = ""
    layer._uvm_weight_prefetch_plan_loaded = False
    layer._uvm_weight_offload_plan_loaded = False
    layer._uvm_init_weight_offload_plan_loaded = False

    parse_calls: list[tuple[str, str]] = []
    real_parse = layer_module._parse_plan_expert_keys

    def record_parse(path: str, plan_key: str) -> set[tuple[int, int]]:
        parse_calls.append((path, plan_key))
        return real_parse(path, plan_key)

    monkeypatch.setattr(layer_module, "_parse_plan_expert_keys", record_parse)

    layer._refresh_uvm_weight_action_plans()
    first_refresh_calls = list(parse_calls)
    layer._refresh_uvm_weight_action_plans()

    assert first_refresh_calls
    assert parse_calls == first_refresh_calls
    assert layer._uvm_weight_prefetch_plan_loaded is True
    assert layer._uvm_weight_offload_plan_loaded is True
    assert layer._uvm_init_weight_offload_plan_loaded is True


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
    assert (
        layer._uvm_init_weight_expert_class(
            layer_id=7,
            global_expert_id=1,
        )
        == "hot"
    )
    assert (
        layer._uvm_init_weight_expert_class(
            layer_id=7,
            global_expert_id=3,
        )
        == "cold"
    )
    assert (
        layer._uvm_init_weight_expert_class(
            layer_id=7,
            global_expert_id=2,
        )
        == "warm"
    )

    # Runtime prefetch may intentionally target the same experts Stage M
    # released. The explicit offload plan must win for initial placement.
    layer.uvm_weight_prefetch_plan_keys = {(7, 3)}
    assert (
        layer._uvm_init_weight_expert_class(
            layer_id=7,
            global_expert_id=3,
        )
        == "cold"
    )


def test_hot_repair_excludes_planned_and_already_offloaded_experts() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_plan_keys = {(7, 1)}
    layer._uvm_slot_managed_global_expert_ids = {2}
    layer._uvm_remote_read_global_expert_ids = {3}
    layer._uvm_all_local_expert_pairs = lambda: [
        (expert_id, expert_id) for expert_id in range(5)
    ]

    assert layer._uvm_init_hot_repair_expert_pairs(7) == [(0, 0), (4, 4)]


def test_hot_repair_prefers_prefetches_and_verifies_gpu_residency(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_hot_repair_enable = True
    layer.uvm_init_weight_hot_repair_preferred_gpu = True
    layer.uvm_init_weight_hot_repair_require_full_residency = True
    layer._uvm_init_weight_hot_repair_done = False
    layer.uvm_init_weight_hot_prefetch_device = 0
    layer.uvm_weight_executor = "cpp_batch"
    layer.uvm_init_weight_plan_policy = "routing_heatmap_v1"
    layer.uvm_init_weight_offload_target_roles = {"moe_gate_up", "moe_down"}
    layer.uvm_init_weight_offload_plan_keys = set()
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_remote_read_global_expert_ids = set()
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._refresh_uvm_weight_action_plans = lambda: None
    layer._uvm_layer_id = lambda: 7
    layer._uvm_all_local_expert_pairs = lambda: [(0, 0)]
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 0x100000, 4096, None),
        ("moe_down", 0x200000, 4096, None),
    ]
    layer._uvm_expert_weight_region_id = (
        lambda layer_id, expert_id, role: f"{layer_id}:{expert_id}:{role}"
    )
    trace: list[dict[str, object]] = []
    layer._append_uvm_init_weight_trace = trace.append
    layer._uvm_update_weight_expert_role_cache = lambda **kwargs: None

    monkeypatch.setattr(
        layer_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, device: len(ranges),
    )
    monkeypatch.setattr(
        layer_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, device, executor: (len(ranges), executor, 0),
    )
    sync_calls: list[bool] = []
    monkeypatch.setattr(
        layer_module.torch.cuda,
        "synchronize",
        lambda: sync_calls.append(True),
    )
    monkeypatch.setattr(
        layer_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, device, **kwargs: {
            "ok": True,
            "status": 2,
            "queried_bytes": 8192,
            "gpu_bytes": 8192,
            "cpu_bytes": 0,
            "any_bytes": 8192,
            "unknown_bytes": 0,
        },
    )
    marked: list[dict[str, object]] = []
    monkeypatch.setattr(
        layer_module,
        "mark_uvm_residency_state",
        lambda **kwargs: marked.append(kwargs),
    )

    result = layer.repair_uvm_init_hot_weight_residency()

    assert result["hot_repair_fully_gpu_resident"] is True
    assert result["hot_repair_initially_fully_gpu_resident"] is True
    assert result["hot_repair_executor"] == "query_skip"
    assert result["hot_repair_gpu_ratio"] == 1.0
    assert sync_calls == [True]
    assert layer._uvm_init_weight_hot_repair_done is True
    assert len(marked) == 2
    assert trace[-1]["action"] == "init_weight_hot_repair_summary"


def test_hot_repair_fails_closed_on_cpu_resident_bytes(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_hot_repair_enable = True
    layer.uvm_init_weight_hot_repair_preferred_gpu = True
    layer.uvm_init_weight_hot_repair_require_full_residency = True
    layer._uvm_init_weight_hot_repair_done = False
    layer.uvm_init_weight_hot_prefetch_device = 0
    layer.uvm_weight_executor = "cpp_batch"
    layer.uvm_init_weight_plan_policy = "routing_heatmap_v1"
    layer.uvm_init_weight_offload_target_roles = {"moe_gate_up", "moe_down"}
    layer.uvm_init_weight_offload_plan_keys = set()
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_remote_read_global_expert_ids = set()
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._refresh_uvm_weight_action_plans = lambda: None
    layer._uvm_layer_id = lambda: 7
    layer._uvm_all_local_expert_pairs = lambda: [(0, 0)]
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 0x100000, 4096, None),
    ]
    layer._uvm_expert_weight_region_id = (
        lambda layer_id, expert_id, role: f"{layer_id}:{expert_id}:{role}"
    )
    layer._append_uvm_init_weight_trace = lambda record: None

    monkeypatch.setattr(
        layer_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, device: len(ranges),
    )
    monkeypatch.setattr(
        layer_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, device, executor: (len(ranges), executor, 0),
    )
    monkeypatch.setattr(layer_module.torch.cuda, "synchronize", lambda: None)
    residency_results = iter(
        [
            {
                "ok": True,
                "status": 1,
                "queried_bytes": 4096,
                "gpu_bytes": 0,
                "cpu_bytes": 4096,
                "any_bytes": 4096,
                "unknown_bytes": 0,
            },
            {
                "ok": True,
                "status": 1,
                "queried_bytes": 4096,
                "gpu_bytes": 0,
                "cpu_bytes": 4096,
                "any_bytes": 4096,
                "unknown_bytes": 0,
            },
        ]
    )
    monkeypatch.setattr(
        layer_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, device, **kwargs: next(residency_results),
    )

    with pytest.raises(RuntimeError, match="residency verification failed"):
        layer.repair_uvm_init_hot_weight_residency()
    assert layer._uvm_init_weight_hot_repair_done is False


def test_hot_repair_derives_hot_residency_from_layer_and_excluded_bytes() -> None:
    result = FusedMoE._uvm_init_hot_residency_from_aggregate_queries(
        layer_residency={
            "ok": True,
            "status": 2,
            "queried_bytes": 12_288,
            "gpu_bytes": 8_192,
            "cpu_bytes": 4_096,
            "any_bytes": 12_288,
            "unknown_bytes": 0,
        },
        excluded_residency={
            "ok": True,
            "status": 1,
            "queried_bytes": 4_096,
            "gpu_bytes": 0,
            "cpu_bytes": 4_096,
            "any_bytes": 4_096,
            "unknown_bytes": 0,
        },
        layer_requested_bytes=12_288,
        excluded_requested_bytes=4_096,
        hot_requested_bytes=8_192,
    )

    assert result["ok"] is True
    assert result["queried_bytes"] == 8_192
    assert result["gpu_bytes"] == 8_192
    assert result["cpu_bytes"] == 0
    assert result["unknown_bytes"] == 0


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


def test_remote_read_experts_do_not_enter_reserved_restore_mask() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_offload_access_mode = "remote_read"
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer.uvm_init_weight_offload_sync = False
    layer.uvm_init_weight_plan_policy = "exact_used_oracle_v1"
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_remote_read_global_expert_ids = set()
    layer._uvm_offloaded_warm_global_expert_ids = set()
    layer._uvm_reserved_offload_mask_values_cache = None
    layer._uvm_reserved_offload_mask_tensor_cache = {}
    layer.uvm_expert_reserved_offloaded_ids = set()
    layer._uvm_plan_expert_bitmask_width = 8

    metadata = layer._uvm_init_weight_class_metadata(
        metadata={},
        expert_class="cold",
        cpu_offload_placement=True,
    )
    layer._uvm_mark_remote_read_expert(5)

    assert layer._uvm_stage_m_uses_remote_read() is True
    assert layer._uvm_stage_m_sync_enabled() is True
    assert metadata["placement"] == "CPU_REMOTE_READ"
    assert metadata["target_placement"] == "CPU_REMOTE_READ"
    assert metadata["cpu_remote_read"] is True
    assert metadata["cpu_offloaded_reserved"] is False
    assert metadata["reserved_prefetch_eligible"] is False
    assert layer._uvm_remote_read_global_expert_ids == {5}
    assert layer._uvm_reserved_offload_mask_for_layer(7) is None


def test_restore_mode_keeps_reserved_restore_metadata() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_offload_access_mode = "restore"
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer.uvm_init_weight_offload_sync = False
    layer.uvm_init_weight_plan_policy = "exact_used_oracle_v1"

    metadata = layer._uvm_init_weight_class_metadata(
        metadata={},
        expert_class="cold",
        cpu_offload_placement=True,
    )

    assert layer._uvm_stage_m_uses_remote_read() is False
    assert layer._uvm_stage_m_sync_enabled() is False
    assert metadata["placement"] == "CPU_OFFLOADED_RESERVED"
    assert metadata["cpu_offloaded_reserved"] is True
    assert metadata["reserved_prefetch_eligible"] is True


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


def test_stage_m_weight_slices_reuse_preregistered_logical_regions(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_target_roles = {
        "moe_gate_up",
        "moe_down",
    }
    layer.uvm_init_weight_register_all_local_experts = True
    snapshots = {
        "weights:12:6:moe_gate_up": {
            "ptr": 0x700000600000,
            "bytes": 1024 * 1024,
            "state": "gpu",
            "driver_region_id": 101,
        },
        "weights:12:6:moe_down": {
            "ptr": 0x710000300000,
            "bytes": 512 * 1024,
            "state": "gpu",
            "driver_region_id": 102,
        },
    }
    monkeypatch.setattr(
        layer_module,
        "get_uvm_residency_region_snapshot",
        snapshots.get,
    )
    monkeypatch.setattr(
        layer,
        "_uvm_expert_weight_slices",
        lambda *_args, **_kwargs: pytest.fail("tensor slices must not be rebuilt"),
    )

    resolved = layer._uvm_stage_m_expert_weight_slices(
        layer_id=12,
        global_expert_id=6,
        local_expert_id=6,
    )

    assert [(role, ptr, size) for role, ptr, size, _ in resolved] == [
        ("moe_gate_up", 0x700000600000, 1024 * 1024),
        ("moe_down", 0x710000300000, 512 * 1024),
    ]
    assert [snapshot["driver_region_id"] for *_, snapshot in resolved] == [
        101,
        102,
    ]


def test_stage_m_weight_slices_fail_closed_when_preregister_is_incomplete(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_target_roles = {
        "moe_gate_up",
        "moe_down",
    }
    layer.uvm_init_weight_register_all_local_experts = True
    monkeypatch.setattr(
        layer_module,
        "get_uvm_residency_region_snapshot",
        lambda _region_id: None,
    )

    with pytest.raises(RuntimeError, match="preregistration is incomplete"):
        layer._uvm_stage_m_expert_weight_slices(
            layer_id=12,
            global_expert_id=6,
            local_expert_id=6,
        )


def test_reserved_executor_fallback_register_preserves_envelope_flag(
    monkeypatch: Any,
) -> None:
    executor = object.__new__(ExpertReservedRegionsExecutor)
    executor._registered_regions = {}
    register_calls: list[dict[str, int]] = []
    monkeypatch.setattr(reserved_module, "driver_query_region", lambda _region_id: None)
    monkeypatch.setattr(
        reserved_module,
        "driver_register_region",
        lambda **kwargs: register_calls.append(kwargs) or 0,
    )
    item = ReservedRegionItem(
        region_id="weights:12:uvmchunk:80000:moe_gate_up:700000600000:80000",
        driver_region_id=101,
        ptr=0x700000600000,
        bytes=512 * 1024,
        layer_id=12,
        global_expert_id=6,
        local_expert_id=6,
        role="moe_gate_up",
        metadata={"expert_envelope": True},
    )

    assert executor._ensure_driver_region(item) == 0
    assert len(register_calls) == 1
    assert register_calls[0]["flags"] & reserved_module.UVM_REGION_FLAG_EXPERT_ENVELOPE


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

    assert traces[0]["action"] == ("init_weight_offload_reject_without_slot_restore")


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


def test_incremental_remote_read_tracks_prepared_expert(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_incremental_expert_offload_enable = True
    layer.uvm_expert_offload_access_mode = "remote_read"
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer._uvm_slot_managed_global_expert_ids = set()
    layer._uvm_remote_read_global_expert_ids = set()
    layer._uvm_init_weight_offload_done = True
    layer._uvm_incremental_offload_in_progress = False
    layer._uvm_incremental_allow_warm = False
    layer.layer_name = "model.layers.7.mlp.experts"

    def fake_stage_m() -> None:
        selected = layer._uvm_incremental_selected_global_expert_id
        assert selected is not None
        layer._uvm_remote_read_global_expert_ids.add(selected)

    monkeypatch.setattr(layer, "run_uvm_init_weight_offload", fake_stage_m)

    result = layer.run_uvm_incremental_weight_offload(
        allow_warm=False,
        selected_global_expert_id=6,
    )

    assert result["released_experts"] == [6]
    assert result["released_count"] == 1
    assert result["access_mode"] == "remote_read"
    assert layer._uvm_slot_managed_global_expert_ids == set()


def test_incremental_remote_read_does_not_require_reserved_capacity() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_offload_access_mode = "remote_read"
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer.uvm_expert_reserved_slot_count = 0

    assert layer._uvm_incremental_slot_capacity_allows(6) is True


def test_incremental_release_remote_initial_stage_requires_envelopes() -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_init_weight_offload_mode = "driver_release_cpu"
    layer.uvm_expert_offload_access_mode = "remote_read"
    layer.uvm_remote_prepare_mode = "release_remote"

    assert layer._uvm_stage_m_requires_release_envelopes() is True
    assert layer._uvm_stage_m_requires_logical_release_registration() is False


def test_dynamic_expert_heat_accumulates_and_resets() -> None:
    import torch

    layer = object.__new__(FusedMoE)
    layer.uvm_dynamic_expert_swap_enable = True
    layer.global_num_experts = 4
    layer.local_num_experts = 4
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._uvm_dynamic_expert_heat_counts = None
    layer._uvm_remote_read_global_expert_ids = {3}

    with torch.inference_mode():
        layer._record_uvm_dynamic_expert_heat(torch.tensor([[0, 1], [1, 3]]))
    snapshot = layer.get_uvm_dynamic_expert_heat(reset=True)

    assert snapshot["expert_token_counts"] == [1, 2, 0, 1]
    assert snapshot["remote_experts"] == [3]
    assert layer.get_uvm_dynamic_expert_heat()["expert_token_counts"] == [0, 0, 0, 0]


def test_dynamic_remote_read_swap_demotes_before_promoting(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_dynamic_expert_swap_enable = True
    layer.uvm_weight_executor = "cpp_batch"
    layer.local_num_experts = 8
    layer._uvm_remote_read_global_expert_ids = {5}
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_layer_id = lambda: 7
    layer._uvm_global_to_local_expert = lambda expert_id: expert_id
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 1000 + int(kwargs["global_expert_id"]) * 100, 40, None),
        ("moe_down", 1040 + int(kwargs["global_expert_id"]) * 100, 60, None),
    ]
    placements: list[tuple[int, str]] = []
    layer._uvm_mark_dynamic_expert_placement = lambda **kwargs: placements.append(
        (int(kwargs["global_expert_id"]), str(kwargs["state"]))
    )
    events: list[str] = []

    def remote_prepare(ranges: Any, **_kwargs: Any) -> tuple[int, str, int, int, int]:
        events.append("demote")
        count = len(ranges)
        return count, "cpp_batch_native_cuda", count, count, count

    def advise_gpu(ranges: Any, _device: int) -> int:
        events.append("advise_gpu")
        return len(ranges)

    def prefetch_gpu(ranges: Any, **_kwargs: Any) -> tuple[int, str, int]:
        events.append("prefetch_gpu")
        return len(ranges), "cpp_batch", 0

    monkeypatch.setattr(
        layer_module,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        remote_prepare,
    )
    monkeypatch.setattr(
        layer_module,
        "batch_advise_ranges_preferred_location",
        advise_gpu,
    )
    monkeypatch.setattr(
        layer_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch_gpu,
    )
    monkeypatch.setattr(
        layer_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, _device: {
            "ok": True,
            "status": 0,
            "queried_bytes": sum(size for _, size in ranges),
            "gpu_bytes": sum(size for _, size in ranges),
            "cpu_bytes": 0,
            "any_bytes": sum(size for _, size in ranges),
            "unknown_bytes": 0,
        },
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = layer.run_uvm_dynamic_remote_read_swaps(
        [{"promote_expert": 5, "demote_expert": 2}]
    )

    assert result["ok"] is True
    assert events == ["demote", "advise_gpu", "prefetch_gpu"]
    assert placements == [(2, "cpu"), (5, "gpu")]
    assert layer._uvm_remote_read_global_expert_ids == {2}


def test_dynamic_release_remote_swap_uses_combined_driver_path(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_dynamic_expert_swap_enable = True
    layer.uvm_remote_prepare_mode = "release_remote"
    layer.uvm_weight_executor = "cpp_batch"
    layer.local_num_experts = 8
    layer._uvm_remote_read_global_expert_ids = {5}
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_layer_id = lambda: 7
    layer._uvm_global_to_local_expert = lambda expert_id: expert_id
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 1000 + int(kwargs["global_expert_id"]) * 100, 40, None),
        ("moe_down", 1040 + int(kwargs["global_expert_id"]) * 100, 60, None),
    ]
    placements: list[tuple[int, str]] = []
    layer._uvm_mark_dynamic_expert_placement = lambda **kwargs: placements.append(
        (int(kwargs["global_expert_id"]), str(kwargs["state"]))
    )
    events: list[str] = []

    def release_remote(**kwargs: Any) -> dict[str, object]:
        events.append("release_remote")
        assert int(kwargs["global_expert_id"]) == 2
        return {"ok": True, "released_bytes": 100, "remote_mapped_bytes": 100}

    def advise_gpu(ranges: Any, _device: int) -> int:
        events.append("advise_gpu")
        return len(ranges)

    def prefetch_gpu(ranges: Any, **_kwargs: Any) -> tuple[int, str, int]:
        events.append("prefetch_gpu")
        return len(ranges), "cpp_batch", 0

    layer._uvm_release_remote_expert_slices = release_remote
    monkeypatch.setattr(
        layer_module,
        "batch_advise_ranges_preferred_location",
        advise_gpu,
    )
    monkeypatch.setattr(
        layer_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch_gpu,
    )
    monkeypatch.setattr(
        layer_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, _device: {
            "ok": True,
            "status": 0,
            "queried_bytes": sum(size for _, size in ranges),
            "gpu_bytes": sum(size for _, size in ranges),
            "cpu_bytes": 0,
            "any_bytes": sum(size for _, size in ranges),
            "unknown_bytes": 0,
        },
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = layer.run_uvm_dynamic_remote_read_swaps(
        [{"promote_expert": 5, "demote_expert": 2}]
    )

    assert result["ok"] is True
    assert events == ["release_remote", "advise_gpu", "prefetch_gpu"]
    assert placements == [(2, "cpu"), (5, "gpu")]
    assert layer._uvm_remote_read_global_expert_ids == {2}
    assert result["results"][0]["demote_driver_release_result"]["ok"] is True


def test_dynamic_release_remote_uses_registered_512k_envelopes(
    monkeypatch: Any,
) -> None:
    envelope_bytes = 512 * 1024
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_envelope_enable = True
    layer.uvm_expert_envelope_bytes = envelope_bytes
    layer._uvm_expert_weight_region_id = (
        lambda layer_id, expert_id, role: f"weights:{layer_id}:{expert_id}:{role}"
    )
    layer._uvm_expert_weight_residency_metadata = lambda **_kwargs: {}
    recorded: list[tuple[object, dict[int, dict[str, object]]]] = []
    layer._uvm_record_release_remote_envelope_states = (
        lambda records, entries: recorded.append((records, entries))
    )

    def envelopes(**kwargs: Any) -> list[Any]:
        role = str(kwargs["role"])
        driver_region_id = 101 if role == "moe_gate_up" else 102
        return [
            layer_module.ReservedRegionItem(
                region_id=f"envelope:{role}",
                driver_region_id=driver_region_id,
                ptr=int(kwargs["ptr"]),
                bytes=envelope_bytes,
                layer_id=7,
                global_expert_id=2,
                local_expert_id=2,
                role=role,
                metadata={"driver_registered": True},
            )
        ]

    layer._uvm_get_or_build_expert_envelope_items = envelopes
    captured: dict[str, object] = {}

    def release(region_ids: Any, **kwargs: Any) -> dict[str, object]:
        captured["region_ids"] = list(region_ids)
        captured.update(kwargs)
        return {
            "ok": True,
            "success_count": 2,
            "validated_success_count": 2,
            "entries": [
                {
                    "ok": True,
                    "prefetch_release_complete": True,
                    "region_id": region_id,
                    "length": envelope_bytes,
                    "released_bytes": envelope_bytes,
                    "cpu_resident_bytes": envelope_bytes,
                    "remote_mapped_bytes": envelope_bytes,
                    "remote_map_failure_blocks": 0,
                }
                for region_id in (101, 102)
            ],
        }

    monkeypatch.setattr(
        layer_module,
        "driver_prefetch_cpu_release_remote_512k_regions",
        release,
    )

    result = layer._uvm_release_remote_expert_slices(
        layer_id=7,
        global_expert_id=2,
        local_expert_id=2,
        slices=[
            ("moe_gate_up", 0x10000000, envelope_bytes, None),
            ("moe_down", 0x20000000, envelope_bytes, None),
        ],
        device=0,
        source_action="test_dynamic_swap",
    )

    assert result["ok"] is True
    assert result["envelope_region_count"] == 2
    assert result["expected_bytes"] == 2 * envelope_bytes
    assert result["released_bytes"] == 2 * envelope_bytes
    assert result["cpu_resident_bytes"] == 2 * envelope_bytes
    assert result["remote_mapped_bytes"] == 2 * envelope_bytes
    assert captured["region_ids"] == [101, 102]
    assert captured["prefer_existing"] is True
    assert len(recorded) == 1


def test_dynamic_release_remote_failure_does_not_commit_placement(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_dynamic_expert_swap_enable = True
    layer.uvm_remote_prepare_mode = "release_remote"
    layer.local_num_experts = 8
    layer._uvm_remote_read_global_expert_ids = {5}
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_layer_id = lambda: 7
    layer._uvm_global_to_local_expert = lambda expert_id: expert_id
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 1000 + int(kwargs["global_expert_id"]) * 100, 40, None),
        ("moe_down", 1040 + int(kwargs["global_expert_id"]) * 100, 60, None),
    ]
    layer._uvm_release_remote_expert_slices = lambda **_kwargs: {
        "ok": False,
        "released_bytes": 40,
        "remote_mapped_bytes": 0,
    }
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = layer.run_uvm_dynamic_remote_read_swaps(
        [{"promote_expert": 5, "demote_expert": 2}]
    )

    assert result["ok"] is False
    assert result["reason"] == "demote_remote_prepare_failed"
    assert layer._uvm_remote_read_global_expert_ids == {5}


def test_dynamic_swap_requires_exact_promoted_gpu_residency(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_dynamic_expert_swap_enable = True
    layer.uvm_remote_prepare_mode = "release_remote"
    layer.uvm_weight_executor = "cpp_batch"
    layer.local_num_experts = 8
    layer._uvm_remote_read_global_expert_ids = {5}
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_layer_id = lambda: 7
    layer._uvm_global_to_local_expert = lambda expert_id: expert_id
    layer._uvm_stage_m_expert_weight_slices = lambda **kwargs: [
        ("moe_gate_up", 1000 + int(kwargs["global_expert_id"]) * 100, 40, None),
        ("moe_down", 1040 + int(kwargs["global_expert_id"]) * 100, 60, None),
    ]
    layer._uvm_release_remote_expert_slices = lambda **_kwargs: {
        "ok": True,
        "released_bytes": 100,
        "cpu_resident_bytes": 100,
        "remote_mapped_bytes": 100,
    }
    placements: list[tuple[int, str]] = []
    layer._uvm_mark_dynamic_expert_placement = lambda **kwargs: placements.append(
        (int(kwargs["global_expert_id"]), str(kwargs["state"]))
    )
    monkeypatch.setattr(
        layer_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        layer_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, **_kwargs: (len(ranges), "cpp_batch", 0),
    )
    monkeypatch.setattr(
        layer_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, _device: {
            "ok": True,
            "status": 0,
            "queried_bytes": sum(size for _, size in ranges),
            "gpu_bytes": sum(size for _, size in ranges) - 20,
            "cpu_bytes": 20,
            "any_bytes": sum(size for _, size in ranges),
            "unknown_bytes": 0,
        },
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = layer.run_uvm_dynamic_remote_read_swaps(
        [{"promote_expert": 5, "demote_expert": 2}]
    )

    assert result["ok"] is False
    assert result["reason"] == "promote_gpu_residency_incomplete"
    assert result["partial_demote_committed"] is True
    assert layer._uvm_remote_read_global_expert_ids == {2, 5}
    assert placements == [(2, "cpu")]
    assert result["results"][0]["promote_residency_complete"] is False


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

    assert traces[0]["action"] == ("next_layer_warm_prefetch_below_token_threshold")
    assert traces[0]["fallback"] == "synchronous_restore_on_actual_topk_hit"


def _make_warm_barrier_layer(
    future: Future[dict[str, object]],
) -> FusedMoE:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_warm_prefetch_enable = True
    layer.uvm_expert_warm_fail_closed = True
    layer._uvm_moe_trace_step = 4
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._uvm_warm_prefetch_expert_pairs = lambda: [(3, 3)]
    layer._uvm_layer_id = lambda: 7
    layer._append_uvm_weight_prefetch_trace = lambda _record: None
    layer._uvm_update_weight_expert_role_cache = lambda **_kwargs: None
    region_ids = [
        "weight:layer7:expert3:moe_gate_up",
        "weight:layer7:expert3:moe_down",
    ]
    layer._uvm_warm_prefetch_ticket = {
        "generation": 2,
        "target_step": 4,
        "future": future,
        "expected_region_ids": region_ids,
        "items": [
            {
                "residency_region_id": region_id,
                "layer_id": 7,
                "global_expert_id": 3,
                "role": role,
                "residency_metadata": {},
            }
            for region_id, role in zip(
                region_ids,
                ("moe_gate_up", "moe_down"),
            )
        ],
    }
    return layer


def _complete_warm_prefetch_result(
    logical_region_ids: list[str],
) -> dict[str, object]:
    physical_region_ids = [f"{region_id}:envelope" for region_id in logical_region_ids]
    return {
        "reserved_success_records": len(logical_region_ids),
        "reserved_success_region_ids": logical_region_ids,
        "reserved_failed_region_ids": [],
        "reserved_expected_physical_region_ids": physical_region_ids,
        "reserved_success_physical_region_ids": physical_region_ids,
        "reserved_assigned_physical_region_ids": physical_region_ids,
        "reserved_driver_requested_bytes": 8192,
        "reserved_driver_accepted_bytes": 8192,
        "reserved_driver_success_count": len(physical_region_ids),
        "reserved_driver_failure_count": 0,
        "reserved_driver_ordinary_resident_bytes": 0,
        "reserved_driver_footprint_contract_broken": 0,
    }


def test_warm_target_barrier_waits_for_complete_w13_w2(
    monkeypatch: Any,
) -> None:
    future: Future[dict[str, object]] = Future()
    layer = _make_warm_barrier_layer(future)
    marked: list[str] = []
    monkeypatch.setattr(
        layer_module,
        "mark_uvm_residency_state",
        lambda **kwargs: marked.append(str(kwargs["region_id"])),
    )

    waiter = threading.Thread(
        target=layer._wait_for_uvm_warm_prefetch_before_compute,
    )
    waiter.start()
    time.sleep(0.02)
    assert waiter.is_alive()

    expected = list(layer._uvm_warm_prefetch_ticket["expected_region_ids"])
    future.set_result(_complete_warm_prefetch_result(expected))
    waiter.join(timeout=1)

    assert not waiter.is_alive()
    assert marked == expected
    assert layer._uvm_warm_prefetch_ready_step == 4
    assert layer._uvm_warm_prefetch_ready_global_expert_ids == {3}


def test_warm_target_barrier_fails_closed_on_missing_role() -> None:
    future: Future[dict[str, object]] = Future()
    layer = _make_warm_barrier_layer(future)
    first_region = str(layer._uvm_warm_prefetch_ticket["expected_region_ids"][0])
    future.set_result(
        {
            "reserved_success_records": 1,
            "reserved_success_region_ids": [first_region],
            "reserved_failed_region_ids": [],
        }
    )

    with pytest.raises(RuntimeError, match="coverage incomplete"):
        layer._wait_for_uvm_warm_prefetch_before_compute()


def test_warm_target_barrier_fails_closed_on_missing_physical_envelope() -> None:
    future: Future[dict[str, object]] = Future()
    layer = _make_warm_barrier_layer(future)
    expected = list(layer._uvm_warm_prefetch_ticket["expected_region_ids"])
    result = _complete_warm_prefetch_result(expected)
    result["reserved_success_physical_region_ids"] = [f"{expected[0]}:envelope"]
    future.set_result(result)

    with pytest.raises(RuntimeError, match="missing_physical"):
        layer._wait_for_uvm_warm_prefetch_before_compute()


def test_warm_target_barrier_records_direct_compute_ioctl_overlap(
    monkeypatch: Any,
) -> None:
    class FakeEvent:
        def __init__(self, timestamp_ms: float) -> None:
            self.timestamp_ms = timestamp_ms

        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            return None

        def elapsed_time(self, other: "FakeEvent") -> float:
            return other.timestamp_ms - self.timestamp_ms

    future: Future[dict[str, object]] = Future()
    layer = _make_warm_barrier_layer(future)
    expected = list(layer._uvm_warm_prefetch_ticket["expected_region_ids"])
    future.set_result(_complete_warm_prefetch_result(expected))
    ticket = layer._uvm_warm_prefetch_ticket
    ticket.update(
        {
            "background": True,
            "source_compute_start_event": FakeEvent(0.0),
            "source_compute_end_event": FakeEvent(5.0),
            "ioctl_cuda_start_event": FakeEvent(1.0),
            "ioctl_cuda_end_event": FakeEvent(9.0),
        }
    )
    traces: list[dict[str, object]] = []
    layer._append_uvm_weight_prefetch_trace = traces.append
    monkeypatch.setattr(layer_module, "mark_uvm_residency_state", lambda **_: None)

    layer._wait_for_uvm_warm_prefetch_before_compute()

    barrier = traces[-1]
    assert barrier["direct_compute_ioctl_overlap_ms"] == pytest.approx(4.0)
    assert barrier["direct_compute_ioctl_overlap_ratio"] == pytest.approx(0.5)
    assert barrier["direct_compute_ioctl_overlap_observed"] is True


def test_warm_prefetch_background_submission_returns_before_driver(
    monkeypatch: Any,
) -> None:
    layer = object.__new__(FusedMoE)
    layer.uvm_expert_warm_prefetch_enable = True
    layer.uvm_expert_reserved_enable = True
    layer.uvm_weight_prefetch_enabled = True
    layer.uvm_expert_reserved_enable_after_warmup = False
    layer.uvm_expert_warm_background_enable = True
    layer.uvm_weight_prefetch_device = 0
    layer.uvm_weight_prefetch_target_roles = {"moe_gate_up", "moe_down"}
    layer.uvm_residency_weight_initial_state = "cpu"
    layer.uvm_expert_envelope_enable = False
    layer.layer_name = "model.layers.7.mlp.experts"
    layer._uvm_moe_trace_step = 9
    layer._uvm_warm_prefetch_generation = 0
    layer._uvm_warm_prefetch_ticket = None
    layer._uvm_layer_id = lambda: 7
    layer._uvm_warm_prefetch_expert_pairs = lambda: [(3, 3)]
    layer._uvm_expert_weight_slices = lambda *_args: [
        ("moe_gate_up", 0x100000, 4096),
        ("moe_down", 0x200000, 4096),
    ]
    layer._uvm_expert_weight_region_id = lambda layer_id, expert_id, role: (
        f"{layer_id}:{expert_id}:{role}"
    )
    layer._uvm_expert_weight_residency_metadata = lambda **_kwargs: {}
    layer._uvm_driver_capability_trace_fields = lambda _metadata: {}
    layer._append_uvm_weight_prefetch_trace = lambda _record: None
    release_driver = threading.Event()

    def blocked_driver(**_kwargs: Any) -> dict[str, object]:
        release_driver.wait(timeout=1)
        return {
            "reserved_success_records": 2,
            "reserved_success_region_ids": [
                "7:3:moe_gate_up",
                "7:3:moe_down",
            ],
            "reserved_failed_region_ids": [],
        }

    layer._run_uvm_expert_reserved_prefetch_batch = blocked_driver
    monkeypatch.setattr(
        layer_module,
        "get_uvm_residency_region_snapshot",
        lambda _: None,
    )
    monkeypatch.setattr(
        layer_module,
        "register_uvm_residency_region",
        lambda **_kwargs: None,
    )

    ticket = layer._prefetch_uvm_offloaded_warm_expert_weights(
        source_layer_name="model.layers.6.mlp.experts",
        source_layer_id=6,
    )

    assert ticket is not None
    future = ticket["future"]
    assert isinstance(future, Future)
    assert ticket["background"] is True
    assert not future.done()
    release_driver.set()
    assert future.result(timeout=1)["reserved_success_records"] == 2


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


def test_fused_bank_reuse_defers_old_mapping_release_to_install(
    monkeypatch: Any,
) -> None:
    class CompleteEvent:
        def query(self) -> bool:
            return True

        def synchronize(self) -> None:
            return None

    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_SLOT_POLICY", "layer_mod_bank")
    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_BANK_COUNT", "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_RESERVED_FUSED_BANK_REUSE", "1")
    executor = ExpertReservedRegionsExecutor(
        device=0,
        pool_id=0,
        slot_count=1,
        slot_bytes=4096,
        strict_footprint=True,
        async_=False,
        wait_on_reuse=True,
        sync_before_compute=False,
        cleanup_synchronize=False,
        prefetch_cpu_source=False,
    )
    executor._prealloc_done = True
    executor.slot_bytes = 4096
    monkeypatch.setattr(executor, "_ensure_driver_region", lambda item: 0)
    monkeypatch.setattr(
        reserved_module,
        "mark_uvm_residency_state",
        lambda **_kwargs: None,
    )
    driver_modes: list[str] = []

    def fake_reserved_prefetch(**kwargs: Any) -> dict[str, Any]:
        driver_modes.append(str(kwargs["mode"]))
        return {
            "ok": True,
            "compact_result": True,
            "success_count": len(kwargs["assignments"]),
            "failure_count": 0,
            "entries": [],
        }

    monkeypatch.setattr(
        reserved_module,
        "driver_expert_reserved_prefetch_regions",
        fake_reserved_prefetch,
    )
    first = ReservedRegionItem(
        region_id="weight:layer0:expert0:w13",
        driver_region_id=100,
        ptr=0x100000,
        bytes=4096,
        layer_id=0,
        global_expert_id=0,
        local_expert_id=0,
        role="moe_gate_up",
    )
    second = ReservedRegionItem(
        region_id="weight:layer1:expert0:w13",
        driver_region_id=200,
        ptr=0x200000,
        bytes=4096,
        layer_id=1,
        global_expert_id=0,
        local_expert_id=0,
        role="moe_gate_up",
    )
    executor.prefetch_for_layer(
        layer_name="model.layers.0.mlp.experts",
        step=1,
        items=[first],
    )
    executor._slots[0].last_event = CompleteEvent()  # type: ignore[assignment]
    callback_order: list[str] = []
    result = executor.prefetch_for_layer(
        layer_name="model.layers.1.mlp.experts",
        step=1,
        items=[second],
        before_driver_ioctl=lambda: callback_order.append("before"),
        after_driver_ioctl=lambda: callback_order.append("after"),
    )

    assert result["bank_reuse_result"]["status"] == (
        "deferred_to_install_ioctl"
    )
    assert driver_modes == ["reserved_prefetch", "reserved_prefetch"]
    assert callback_order == ["before", "after"]
