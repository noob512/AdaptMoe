# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.device_allocator import uvm as uvm_module
from vllm.device_allocator import uvm_pool_residency as residency_module
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker import gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _uvm_kv_tail_range_for_allocation,
)
from vllm.v1.worker.gpu_worker import Worker


@pytest.fixture(autouse=True)
def _isolated_slot_session_contract(monkeypatch: Any):
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_SESSION_CONTRACT", None)


def _mock_complete_slot_registry(monkeypatch: Any) -> None:
    va_block_bytes = 2 * 1024 * 1024
    base = 0x40000000
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda region_ids: {
            "ok": True,
            "regions": [
                {
                    "region_id": str(region_id),
                    "base": base + index * va_block_bytes,
                    "length": va_block_bytes,
                }
                for index, region_id in enumerate(region_ids)
            ],
        },
    )


def _enable_registered_slot_plan(monkeypatch: Any) -> None:
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")


def test_uvm_kv_tail_range_respects_block_boundaries() -> None:
    page_size = 4096
    ptr = 0x100000

    assert _uvm_kv_tail_range_for_allocation(
        ptr=ptr,
        size_bytes=11 * page_size,
        page_size=page_size,
        total_blocks=10,
        target_blocks=3,
    ) == (ptr + 4 * page_size, 7 * page_size, 4, 10)

    assert (
        _uvm_kv_tail_range_for_allocation(
            ptr=ptr,
            size_bytes=11 * page_size,
            page_size=page_size,
            total_blocks=10,
            target_blocks=10,
        )
        is None
    )


def test_post_warmup_kv_rebalance_returns_control_fields_only(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._collect_uvm_post_warmup_kv_tail_ranges = (  # type: ignore[method-assign]
        lambda _target: (
            [(0x100000, 0x2000)],
            {"ok": True, "reason": "ranges_collected", "target_blocks": 4},
        )
    )
    monkeypatch.setenv("VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR", "cpp_batch")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, *, sync, executor: (
            len(ranges),
            executor,
            {"diagnostic": "unused"},
            {"diagnostic": "unused"},
        ),
    )

    result = runner.rebalance_uvm_kv_cache_after_warmup(4)

    assert result == {
        "ok": True,
        "reason": "ranges_collected",
        "target_blocks": 4,
        "offload_complete": True,
    }


def test_pre_kv_stage_orders_expert_placement_before_kv(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class FakeModelRunner:
        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("cold_placement")

        def initialize_kv_cache(self, kv_cache_config: object) -> None:
            events.append("kv_allocation")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "pre_kv")
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_initialized",
        lambda config, kv_config: None,
    )

    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False)
    )
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: False
    worker._maybe_get_kv_cache_uvm_memory_pool_context = nullcontext

    worker.initialize_from_config(object())

    assert events == ["cold_placement", "kv_allocation"]


def test_post_warmup_rebalance_orders_kv_before_experts(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    control_file = tmp_path / "kv_dynamic_pool.control"

    class FakeModelRunner:
        def rebalance_uvm_kv_cache_after_warmup(
            self,
            target_blocks: int,
        ) -> dict[str, Any]:
            assert control_file.read_text(encoding="utf-8") == "target_blocks=4\n"
            events.append(f"kv:{target_blocks}")
            return {"offload_complete": True}

        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("experts")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "4")
    monkeypatch.setenv("VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE", str(control_file))
    monkeypatch.setenv("VLLM_USE_UVM", "1")

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: True

    worker._run_uvm_post_warmup_rebalance()

    assert events == ["kv:4", "experts"]

    worker._run_uvm_post_warmup_rebalance()
    assert events == ["kv:4", "experts"]


def test_post_warmup_target_automatically_enables_dynamic_pool(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "4")
    monkeypatch.delenv("VLLM_UVM_KV_RUNTIME_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", raising=False)

    policy = UvmKvRuntimePolicy(
        KVCacheConfig(
            num_blocks=11,
            kv_cache_tensors=[KVCacheTensor(size=11 * 4096, shared_by=["layer.0"])],
            kv_cache_groups=[],
        )
    )

    assert policy.enabled is True
    assert policy.dynamic_pool_active() is True
    assert policy.total_blocks == 10


def test_registered_slot_plan_configuration_fails_before_bank_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_INIT_BANK", raising=False)
    allocated = False

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal allocated
        allocated = True
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match="complete fixed-bank configuration"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert allocated is False


def test_prefill_p3_decode_remote_requires_complete_fence_configuration(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", raising=False)
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", raising=False)
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", raising=False)

    with pytest.raises(RuntimeError, match="complete P3 fence configuration"):
        runner.finalize_uvm_slot_bank_after_warmup()


def _complete_p3_close_result() -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "abort": {
                "ok": True,
                "contract_ok": True,
                "session_cleared": True,
                "remaining_slot_count": 0,
                "remaining_bytes": 0,
            }
        },
    }


def test_prefill_p3_switches_to_remote_read_before_first_decode(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    module = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={7, 9},
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_layer_id=lambda: 3,
        _uvm_expert_records={
            (3, 7): SimpleNamespace(envelope_region_ids=["r7a", "r7b"]),
            (3, 9): SimpleNamespace(envelope_region_ids=["r9a"]),
        },
    )
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    runner._uvm_prefill_p3_seen = True
    runner._uvm_prefill_p3_decode_remote_state = "prefill_p3"
    runner._uvm_slot_init_bank_done = True
    runner._uvm_slot_init_bank_result = {"ok": True}
    runner._uvm_slot_registered_plan_result = {"ok": True}
    runner._uvm_slot_session_contract = {"slot_session_enabled": True}
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: events.append("close") or _complete_p3_close_result(),
    )

    def remote_map(region_ids: list[str], **_kwargs: Any) -> dict[str, Any]:
        events.append("remote:" + ",".join(region_ids))
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "transition_uvm_regions_to_remote_read",
        remote_map,
    )

    runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )

    assert events == ["sync", "close", "remote:r7a,r7b,r9a"]
    assert module.uvm_expert_offload_access_mode == "remote_read"
    assert module._uvm_offloaded_global_expert_ids == set()
    assert module._uvm_remote_read_global_expert_ids == {7, 9}
    assert runner._uvm_prefill_p3_decode_remote_state == "decode_remote_read"
    assert runner._uvm_slot_init_bank_done is False

    runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )
    assert events == ["sync", "close", "remote:r7a,r7b,r9a"]


def test_prefill_p3_remote_read_failure_poison_is_sticky(
    monkeypatch: Any,
) -> None:
    module = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={2},
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_layer_id=lambda: 1,
        _uvm_expert_records={
            (1, 2): SimpleNamespace(envelope_region_ids=["r2"]),
        },
    )
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    runner._uvm_prefill_p3_seen = True
    runner._uvm_prefill_p3_decode_remote_state = "prefill_p3"
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", lambda _device: None
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: _complete_p3_close_result(),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "transition_uvm_regions_to_remote_read",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )

    with pytest.raises(RuntimeError, match="worker is poisoned"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"request": 1}),
            "enabled:decode_step",
        )

    assert runner._uvm_prefill_p3_decode_remote_state == "poisoned"
    assert module.uvm_expert_offload_access_mode == "poisoned"
    with pytest.raises(RuntimeError, match="worker is poisoned"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"request": 1}),
            "enabled:decode_step",
        )


def test_prefill_p3_remote_rejects_mixed_step_before_driver_work(
    monkeypatch: Any,
) -> None:
    module = SimpleNamespace(uvm_expert_offload_access_mode="slot_restore")
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mixed step must fail before close")
        ),
    )

    with pytest.raises(RuntimeError, match="unmixed single request"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"prefill": 4, "decode": 1}),
            "enabled:mixed_step",
        )

    assert runner._uvm_prefill_p3_decode_remote_state == "poisoned"


def test_remote_transition_requires_complete_mapping_audit(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH", "0")
    monkeypatch.setattr(
        residency_module,
        "release_uvm_regions",
        lambda region_ids, **_kwargs: {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
            "audits": {
                "r0": {
                    "ok": True,
                    "remote_read": True,
                    "remote_map_failure_blocks": 0,
                    "cpu_resident_bytes": 512 * 1024,
                    "remote_mapped_bytes": 512 * 1024,
                    "registered_length": 512 * 1024,
                },
                "r1": {
                    "ok": True,
                    "remote_read": True,
                    "remote_map_failure_blocks": 1,
                    "cpu_resident_bytes": 512 * 1024,
                    "remote_mapped_bytes": 0,
                    "registered_length": 512 * 1024,
                },
            },
        },
    )

    result = residency_module.transition_uvm_regions_to_remote_read(["r0", "r1"])

    assert result["ok"] is False
    assert result["reason"] == "remote_read_audit_failed"


def test_remote_transition_uses_one_audited_batch_plan(
    monkeypatch: Any,
) -> None:
    registry = SimpleNamespace(
        regions={
            "r0": residency_module.KernelRegion(
                "r0", 0x40000000, 512 * 1024, residency_module.RegionState.CPU
            ),
            "r1": residency_module.KernelRegion(
                "r1", 0x50000000, 512 * 1024, residency_module.RegionState.CPU
            ),
        }
    )
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    captured: list[list[tuple[int, int, int]]] = []

    def prepare(
        regions: list[tuple[int, int, int]], **_kwargs: Any
    ) -> dict[str, Any]:
        captured.append(regions)
        return {
            "ok": True,
            "ioctl_calls": 1,
            "remote_mapped_bytes": 1024 * 1024,
        }

    monkeypatch.setattr(
        residency_module,
        "driver_prepare_regions_remote_read_audited",
        prepare,
    )

    result = residency_module.transition_uvm_regions_to_remote_read(["r0", "r1"])

    assert result["ok"] is True
    assert result["batch"] is True
    assert captured == [
        [
            (
                residency_module._stable_driver_region_id("r0"),
                0x40000000,
                512 * 1024,
            ),
            (
                residency_module._stable_driver_region_id("r1"),
                0x50000000,
                512 * 1024,
            ),
        ]
    ]


def test_audited_remote_batch_checks_every_range_result(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_open_or_adopt_initialized_uvm_fd",
        lambda **_kwargs: 123,
    )
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls = 0

    def ioctl(_fd: int, _request: int, params: Any) -> int:
        nonlocal calls
        calls += 1
        params.success_count = params.count
        params.failure_count = 0
        params.rmStatus = 0
        for index in range(int(params.count)):
            length = int(params.ranges[index].length)
            item = params.results[index]
            item.status = 0
            item.queried_bytes = length
            item.resident_cpu_bytes = length
            item.resident_gpu_bytes = length
            item.resident_any_bytes = length
            item.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_prepare_ranges_remote_read_audited(
        [(0x100000 + i * 0x10000, 0x10000) for i in range(65)]
    )

    assert result["ok"] is True
    assert result["ioctl_calls"] == 2
    assert result["success_count"] == 65
    assert result["remote_mapped_bytes"] == 65 * 0x10000
    assert calls == 2


def _complete_registered_remote_batch(
    params: Any,
    lengths: dict[int, int],
    *,
    newly_mapped: bool = True,
) -> None:
    params.success_count = params.count
    params.failure_count = 0
    params.rmStatus = 0
    for index in range(int(params.count)):
        length = lengths[int(params.region_ids[index])]
        item = params.results[index]
        item.status = 0
        item.queried_bytes = length
        item.resident_cpu_bytes = length
        item.resident_gpu_bytes = length if newly_mapped else 0
        item.resident_any_bytes = length
        item.resident_unknown_bytes = 0


def test_registered_remote_batch_uses_owner_fd_and_exact_new_mapping(
    monkeypatch: Any,
) -> None:
    regions = [
        (index + 1, 0x40000000 + index * 0x10000, 0x10000)
        for index in range(65)
    ]
    lengths = {region_id: length for region_id, _base, length in regions}
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [10, 20])
    monkeypatch.setattr(uvm_module, "_initialize_duplicated_uvm_fd", lambda fd: fd + 100)
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: Any) -> int:
        calls.append(fd)
        assert request == uvm_module._UVM_PHASE_F_BATCH_REGIONS_IOCTL
        if fd == 110:
            params.success_count = 0
            params.failure_count = params.count
            params.rmStatus = 0
            return 0
        _complete_registered_remote_batch(params, lengths)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert result["ioctl_calls"] == 3
    assert result["mapping_ioctl_calls"] == 2
    assert result["remote_mapped_bytes"] == 65 * 0x10000
    assert calls == [110, 120, 120]


def test_registered_remote_batch_rejects_existing_mapping_without_fd_retry(
    monkeypatch: Any,
) -> None:
    regions = [(1, 0x40000000, 0x80000)]
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [10, 20])
    monkeypatch.setattr(uvm_module, "_initialize_duplicated_uvm_fd", lambda fd: fd + 100)
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls: list[int] = []

    def ioctl(fd: int, _request: int, params: Any) -> int:
        calls.append(fd)
        _complete_registered_remote_batch(params, {1: 0x80000}, newly_mapped=False)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is False
    assert result["reason"] == "range_audit_failed"
    assert result["made_progress"] is True
    assert calls == [110]


@pytest.mark.parametrize(
    "regions",
    [
        [(True, 0x40000000, 0x80000)],
        [(1, 0x40000001, 0x80000)],
        [(1, 0x40000000, 0x80000), (2, 0x40040000, 0x80000)],
        [(1, 0x40000000, 0x80000), (1, 0x50000000, 0x80000)],
    ],
)
def test_registered_remote_batch_rejects_invalid_input_before_ioctl(
    monkeypatch: Any,
    regions: list[tuple[Any, int, int]],
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is False
    assert result["ioctl_calls"] == 0


def test_phase_f_ctypes_batch_abi_matches_kernel_header() -> None:
    assert uvm_module._phase_f_batch_abi_matches_kernel() is True


def test_init_bank_requires_cleanup_owning_slot_access_mode(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_INIT_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "0")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "restore")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid access mode must fail before mode8")
        ),
    )

    with pytest.raises(RuntimeError, match="slot init-bank requires"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert residency_module.get_uvm_slot_direct_session_contract() is None


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            [(0, [(0, ["r0"])]), (0, [(0, ["r1"])])],
            "layer_id values must be unique",
        ),
        (
            [(layer_id, [(0, [f"r{layer_id}"])]) for layer_id in range(64)],
            "exceed the kernel limit",
        ),
    ],
)
def test_registered_slot_plan_preflight_rejects_before_mode8(
    monkeypatch: Any,
    candidates: list[tuple[int, list[tuple[int, list[str]]]]],
    message: str,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = candidates
    _enable_registered_slot_plan(monkeypatch)
    allocated = False

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal allocated
        allocated = True
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match=message):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert allocated is False


def test_registered_slot_plan_preflight_allows_different_expert_ids(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (0, [(0, ["l0-e0"]), (1, ["l0-e1"])]),
        (1, [(2, ["l1-e2"]), (3, ["l1-e3"])]),
    ]
    _enable_registered_slot_plan(monkeypatch)
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == ["mode8", "register"]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40001000,
                    "length": 2 * 1024 * 1024,
                }
            ],
            "not 64 KiB aligned",
        ),
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40000000,
                    "length": 2 * 1024 * 1024 - 4096,
                }
            ],
            "not 64 KiB aligned",
        ),
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40000000,
                    "length": 1024 * 1024,
                },
                {
                    "region_id": "r1",
                    "base": 0x40080000,
                    "length": 1024 * 1024,
                },
            ],
            "64 KiB ranges overlap",
        ),
    ],
)
def test_registered_slot_plan_rejects_invalid_64k_layout_before_mode8(
    monkeypatch: Any,
    records: list[dict[str, Any]],
    message: str,
) -> None:
    runner = object.__new__(GPUModelRunner)
    region_ids = [str(record["region_id"]) for record in records]
    runner._uvm_slot_init_bank_candidates = [(0, [(0, region_ids)])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda requested: {
            "ok": list(requested) == region_ids,
            "regions": records,
        },
    )
    mode8_calls = 0

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal mode8_calls
        mode8_calls += 1
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match=message):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert mode8_calls == 0
    assert residency_module.get_uvm_slot_direct_session_contract() is None


def test_registered_slot_plan_allows_sparse_512k_envelopes(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    region_ids = ["cold-envelope-0", "cold-envelope-1"]
    records = [
        {
            "region_id": region_ids[0],
            "base": 0x40000000,
            "length": 512 * 1024,
        },
        {
            "region_id": region_ids[1],
            "base": 0x40100000,
            "length": 512 * 1024,
        },
    ]
    runner._uvm_slot_init_bank_candidates = [(0, [(7, region_ids)])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda requested: {
            "ok": list(requested) == region_ids,
            "regions": records,
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == ["mode8", "register"]


def test_slot_finalizer_freezes_contract_and_rejects_later_env_drift(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    _enable_registered_slot_plan(monkeypatch)
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()
    assert runner._uvm_slot_session_contract["registered_plan"] is True
    assert runner._uvm_slot_session_contract["slot_session_enabled"] is True
    assert calls == ["mode8", "register"]

    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "0")
    with pytest.raises(RuntimeError, match="contract changed"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert calls == ["mode8", "register"]


def test_slot_finalizer_allocates_two_banks_before_dual_plan_registration(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(4)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda _groups, *, layer_id, device: (
            calls.append(("bank0", int(layer_id))) or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_registered_plan_second_bank",
        lambda _groups, *, layer_id, device: (
            calls.append(("bank1", int(layer_id))) or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda specs, *, device: (
            calls.append(("register", len(specs))) or {"ok": True}
        ),
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == [("bank0", 0), ("bank1", 1), ("register", 4)]
    assert runner._uvm_slot_init_bank_result["dual_bank_event"] is True
    assert len(runner._uvm_slot_init_bank_result["bank_results"]) == 2
    assert dict(runner._uvm_slot_session_contract["environment"])[
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT"
    ] == "1"
    assert dict(runner._uvm_slot_session_contract["environment"])[
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE"
    ] == "1"


def test_slot_finalizer_rejects_cuda_fence_without_dual_bank_before_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "0")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete CUDA fence bundle must not allocate")
        ),
    )

    with pytest.raises(RuntimeError, match="requires registered plans and dual-bank"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert residency_module.get_uvm_slot_direct_session_contract() is None


def test_slot_finalizer_rejects_odd_dual_bank_layer_count_before_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(3)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe dual-bank topology must not allocate")
        ),
    )

    with pytest.raises(RuntimeError, match="even number of layers"):
        runner.finalize_uvm_slot_bank_after_warmup()


def test_slot_finalizer_aborts_partial_dual_bank_initialization(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(4)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_registered_plan_second_bank",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plans must not register after bank-1 failure")
        ),
    )
    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: cleanup_calls.append(int(device)) or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="second-bank preallocation failed"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert cleanup_calls == [0]
    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None
    assert runner._uvm_slot_session_contract is None


@pytest.mark.parametrize("failure_stage", ["preallocate", "register"])
def test_slot_init_failure_closes_partial_bank_and_resets_runner_state(
    monkeypatch: Any,
    failure_stage: str,
) -> None:
    _mock_complete_slot_registry(monkeypatch)
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {
            "ok": failure_stage != "preallocate",
            "reason": "injected",
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: {
            "ok": failure_stage != "register",
            "reason": "injected",
        },
    )
    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: cleanup_calls.append(int(device)) or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="injected"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert cleanup_calls == [0]
    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None


def test_slot_init_cleanup_exception_still_resets_runner_state(
    monkeypatch: Any,
) -> None:
    _mock_complete_slot_registry(monkeypatch)
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    runner._uvm_slot_init_bank_done = True
    runner._uvm_slot_init_bank_result = {"stale": True}
    runner._uvm_slot_registered_plan_result = {"stale": True}
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    # Reach the rollback path instead of the idempotent early return.
    runner._uvm_slot_init_bank_done = False
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )

    def cleanup(*, device: int) -> dict[str, Any]:
        assert device == 0
        raise OSError("cleanup exploded")

    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        cleanup,
    )

    with pytest.raises(RuntimeError, match="slot init rollback raised"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None


def test_shutdown_closes_frozen_slot_session_even_after_environment_drift(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 3,
        "access_mode": "slot_restore",
    }
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("environment changed")),
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: events.append(("close", int(device))) or {"ok": True},
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append(("kv", -1))
    )
    worker.profiler = SimpleNamespace(
        shutdown=lambda: events.append(("profiler", -1))
    )

    with pytest.raises(RuntimeError, match="contract drifted"):
        worker.shutdown()
    assert events == [
        ("sync", 3),
        ("close", 3),
        ("kv", -1),
        ("profiler", -1),
    ]


def test_shutdown_sync_failure_retains_frozen_slot_session(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 2,
        "access_mode": "slot_restore",
    }
    sync_failure = OSError("sync exploded")
    events: list[str] = []
    close_calls = 0
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: contract,
    )

    def synchronize(device: int) -> None:
        assert device == 2
        events.append("sync")
        raise sync_failure

    monkeypatch.setattr(gpu_worker_module.torch.cuda, "synchronize", synchronize)

    def close(**_kwargs: Any) -> dict[str, Any]:
        nonlocal close_calls
        close_calls += 1
        return {"ok": True}

    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        close,
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("kv")
    )
    worker.profiler = SimpleNamespace(shutdown=lambda: events.append("profiler"))

    with pytest.raises(RuntimeError, match="synchronize failed") as raised:
        worker.shutdown()

    assert events == ["sync", "kv", "profiler"]
    assert close_calls == 0
    slot_error = raised.value.__cause__
    assert isinstance(slot_error, RuntimeError)
    assert slot_error.__cause__ is sync_failure


def test_shutdown_combines_contract_and_close_failures_without_losing_causes(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 1,
        "access_mode": "slot_restore",
    }
    contract_failure = OSError("contract drifted")
    close_failure = RuntimeError("close exploded")
    events: list[str] = []
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: (_ for _ in ()).throw(contract_failure),
    )

    def synchronize(_device: int) -> None:
        events.append("sync")

    def close(*, device: int) -> dict[str, Any]:
        events.append(f"close:{device}")
        raise close_failure

    monkeypatch.setattr(gpu_worker_module.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        close,
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("kv")
    )
    worker.profiler = None

    with pytest.raises(RuntimeError) as raised:
        worker.shutdown()

    assert "close failed" in str(raised.value)
    assert "contract drifted" in str(raised.value)
    assert events == ["sync", "close:1", "kv"]
    slot_error = raised.value.__cause__
    assert isinstance(slot_error, RuntimeError)
    assert slot_error.__cause__ is close_failure


def test_shutdown_does_not_open_cleanup_from_mutable_environment(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: None,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutable environment must not trigger slot cleanup")
        ),
    )
    events: list[str] = []
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("runner")
    )
    worker.profiler = None

    worker.shutdown()
    assert events == ["runner"]


def test_post_warmup_remote_read_runs_explicit_expert_placement(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class FakeModelRunner:
        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("experts")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "remote_read")

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()

    worker._run_uvm_post_warmup_rebalance()

    assert events == ["experts"]


class _FakeExpertModule:
    uvm_weight_executor = "cpp_batch"

    def __init__(
        self,
        name: str,
        region_ids: list[str],
        events: list[str],
        *,
        release_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.region_ids = region_ids
        self.events = events
        self.release_error = release_error

    def run_uvm_init_weight_offload(self) -> None:
        self.events.append(f"release:{self.name}")
        if self.release_error is not None:
            raise self.release_error

    def get_uvm_init_hot_expert_region_ids(self) -> list[str]:
        self.events.append(f"collect:{self.name}")
        return list(self.region_ids)

    def set_uvm_init_hot_expert_state(self, state: str) -> None:
        self.events.append(f"state:{self.name}:{state}")


def _make_init_weight_placement_runner(
    modules: list[object],
    *,
    matched_offload: Any | None = None,
) -> Any:
    return SimpleNamespace(
        _uvm_init_weight_placement_done=False,
        _run_uvm_matched_cpu_offload_manifest=(
            matched_offload if matched_offload is not None else lambda: None
        ),
        _env_flag_enabled=lambda _name, default=False: True,
        get_model=lambda: SimpleNamespace(modules=lambda: modules),
    )


def _patch_init_weight_placement_phase(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )


def _unexpected_call(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("unexpected call")


def test_init_weight_placement_releases_all_modules_before_one_hot_batch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    modules = [
        _FakeExpertModule("a", ["hot-a"], events),
        _FakeExpertModule("b", ["hot-b"], events),
    ]
    runner = _make_init_weight_placement_runner(modules)

    def prepare(
        region_ids: list[str],
        *,
        device: int,
        executor: str,
    ) -> dict[str, object]:
        assert events == [
            "release:a",
            "release:b",
            "collect:a",
            "collect:b",
        ]
        events.append(f"prepare:{','.join(region_ids)}:{device}:{executor}")
        return {"ok": True}

    def commit(
        region_ids: list[str],
        *,
        synchronized: bool,
    ) -> dict[str, object]:
        events.append(f"commit:{','.join(region_ids)}:{synchronized}")
        return {"ok": True}

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        prepare,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        commit,
    )
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "release:b",
        "collect:a",
        "collect:b",
        "prepare:hot-a,hot-b:2:cpp_batch",
        "sync:2",
        "commit:hot-a,hot-b:True",
        "state:a:gpu",
        "state:b:gpu",
    ]
    assert runner._uvm_init_weight_placement_done is True


def test_init_weight_placement_is_idempotent(monkeypatch: Any) -> None:
    events: list[str] = []
    module = _FakeExpertModule("a", ["hot"], events)
    runner = _make_init_weight_placement_runner(
        [module],
        matched_offload=lambda: events.append("matched"),
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, **_kwargs: events.append("commit") or {"ok": True},
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)
    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "matched",
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit",
        "state:a:gpu",
        "matched",
    ]


def test_init_weight_placement_without_hot_experts_does_not_sync(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner([_FakeExpertModule("cold", [], events)])

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        _unexpected_call,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        _unexpected_call,
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == ["release:cold", "collect:cold"]
    assert runner._uvm_init_weight_placement_done is True


def test_init_weight_placement_manifest_proves_release_and_residency(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "offload_plan": [{"layer_id": 0, "expert_id": 1}],
                "hot_plan": [{"layer_id": 0, "expert_id": 2}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "placement.json"
    monkeypatch.setenv(
        "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST", str(output_path)
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda region_ids: {
            "ok": True,
            "requested_region_count": len(region_ids),
            "missing_region_ids": [],
            "regions": [
                {
                    "region_id": "remote",
                    "base": 0x100000,
                    "length": 0x80000,
                    "state": "cpu",
                },
                {
                    "region_id": "hot",
                    "base": 0x180000,
                    "length": 0x80000,
                    "state": "gpu",
                },
            ],
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "query_uvm_regions_physical_residency",
        lambda *_args, **_kwargs: {
            "ok": True,
            "queried_bytes": 0x80000,
            "cpu_bytes": 0x80000,
            "gpu_bytes": 0,
            "unknown_bytes": 0,
        },
    )
    audit = {
        "region_id": "remote",
        "ok": True,
        "remote_read": True,
        "registered_length": 0x80000,
        "length": 0x80000,
        "cpu_resident_bytes": 0x80000,
        "remote_mapped_bytes": 0x80000,
        "remote_map_failure_blocks": 0,
        "app_state_after": "cpu",
    }
    snapshot = {
        "layer_id": 0,
        "plan_file": str(plan_path),
        "experts": [
            {"layer_id": 0, "expert_id": 1, "placement": "remote"},
            {"layer_id": 0, "expert_id": 2, "placement": "hot"},
        ],
        "region_owners": [
            {
                "region_id": "remote",
                "layer_id": 0,
                "expert_id": 1,
                "placement": "remote",
            },
            {
                "region_id": "hot",
                "layer_id": 0,
                "expert_id": 2,
                "placement": "hot",
            },
        ],
        "release_audits": [audit],
    }
    module = SimpleNamespace(
        get_uvm_init_weight_placement_snapshot=lambda: snapshot
    )

    GPUModelRunner._write_uvm_init_weight_placement_manifest(
        SimpleNamespace(), [module]
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["complete"] is True
    assert manifest["remote_experts"] == [[0, 1]]
    assert manifest["remote_region_bytes"] == 0x80000
    assert manifest["release_time_remote_mapping_verified"] is True


def test_init_weight_placement_prepare_failure_rolls_back_without_sync(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": False},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    with pytest.raises(RuntimeError, match="Hot Expert prefetch failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_sync_failure_marks_hot_unknown(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    def fail_sync(_device: int) -> None:
        events.append("sync")
        raise RuntimeError("mock sync failure")

    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "synchronize", fail_sync)

    with pytest.raises(RuntimeError, match="mock sync failure"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_commit_failure_marks_hot_unknown(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    with pytest.raises(RuntimeError, match="Hot Expert placement commit failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit:True",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_cold_failure_skips_hot_prefetch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    modules = [
        _FakeExpertModule("a", ["hot-a"], events),
        _FakeExpertModule(
            "b",
            ["hot-b"],
            events,
            release_error=RuntimeError("Cold Expert release failed"),
        ),
    ]
    runner = _make_init_weight_placement_runner(modules)

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        _unexpected_call,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        _unexpected_call,
    )

    with pytest.raises(RuntimeError, match="Cold Expert release failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == ["release:a", "release:b"]
    assert runner._uvm_init_weight_placement_done is False


def test_incremental_expert_worker_protocol_is_minimal(
    monkeypatch: Any,
) -> None:
    class ExpertModule:
        layer_name = "layer.7"
        uvm_incremental_expert_offload_enable = True

        @staticmethod
        def uvm_next_incremental_expert(expert_class: str) -> int | None:
            return 6 if expert_class == "cold" else None

        @staticmethod
        def has_remaining_uvm_incremental_experts(expert_class: str) -> bool:
            return expert_class == "cold"

        @staticmethod
        def run_uvm_incremental_weight_offload(
            *, allow_warm: bool, selected_global_expert_id: int | None
        ) -> dict[str, object]:
            assert allow_warm is False
            return {
                "ok": True,
                "released_experts": [selected_global_expert_id],
            }

    module = ExpertModule()
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(modules=lambda: [module])  # type: ignore[method-assign]
    runner._env_flag_enabled = lambda *_args, **_kwargs: True  # type: ignore[method-assign]
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )

    candidates = GPUModelRunner.get_uvm_incremental_expert_offload_candidates(
        runner, "cold"
    )
    assert candidates == {"ok": True, "candidates": {"layer.7": 6}}
    assert GPUModelRunner.run_uvm_incremental_expert_offload(
        runner,
        False,
        {"layer.7": 6},
    ) == {"ok": True}


def _make_worker_resize_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=7)
    runner._env_flag_enabled = lambda *_args, **_kwargs: False
    return runner


def test_worker_exact_kv_resize_runs_shrink_before_grow(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    events: list[tuple[str, tuple[int, ...]] | tuple[str, int]] = []

    def collect(block_ids: list[int]) -> list[tuple[int, int]]:
        ids = tuple(block_ids)
        events.append(("collect", ids))
        return [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in ids
        ]

    def offload(
        ranges: list[tuple[int, int]],
        **_kwargs: object,
    ) -> tuple[int, str, int, int]:
        events.append(("offload", len(ranges)))
        return len(ranges), "test", 0, 0

    def prefetch(
        ranges: list[tuple[int, int]],
        _device: int,
        **_kwargs: object,
    ) -> tuple[int, str, int]:
        events.append(("prefetch", len(ranges)))
        return len(ranges), "test", 0

    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        collect,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        offload,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [4, 5, 6],
    )

    assert result == {"ok": True, "made_progress": True}
    assert events == [
        ("collect", (6,)),
        ("offload", 1),
        ("collect", (4, 5)),
        ("prefetch", 2),
        ("sync", 0),
    ]


def test_worker_partial_kv_resize_only_reports_complete_direction(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in block_ids
        ],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, **_kwargs: (len(ranges), "test", 0, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda *_args, **_kwargs: (1, "test", 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: None,
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [4, 5, 6],
    )

    assert result == {"ok": False, "made_progress": True}


def test_worker_kv_resize_sync_failure_never_reports_completion(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in block_ids
        ],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (len(ranges), "test", 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )

    def fail_sync(_device: int) -> None:
        raise RuntimeError("mock KV synchronization failure")

    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        fail_sync,
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [],
    )

    assert result == {"ok": False, "made_progress": True}
