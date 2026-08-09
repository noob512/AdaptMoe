# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker import gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _uvm_kv_tail_range_for_allocation,
)
from vllm.v1.worker.gpu_worker import Worker


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


def test_init_weight_placement_releases_all_modules_before_one_hot_batch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class FakeExpertModule:
        uvm_weight_executor = "cpp_batch"

        def __init__(self, name: str, region_ids: list[str]) -> None:
            self.name = name
            self.region_ids = region_ids

        def run_uvm_init_weight_offload(self) -> None:
            events.append(f"release:{self.name}")

        def get_uvm_init_hot_expert_region_ids(self) -> list[str]:
            events.append(f"collect:{self.name}")
            return list(self.region_ids)

        def set_uvm_init_hot_expert_state(self, state: str) -> None:
            events.append(f"state:{self.name}:{state}")

    modules = [
        FakeExpertModule("a", ["hot-a"]),
        FakeExpertModule("b", ["hot-b"]),
    ]
    model = SimpleNamespace(modules=lambda: modules)
    runner = SimpleNamespace(
        _run_uvm_matched_cpu_offload_manifest=lambda: 0,
        _env_flag_enabled=lambda _name, default=False: True,
        get_model=lambda: model,
    )

    def prepare(
        region_ids: list[str],
        *,
        device: int,
        executor: str,
    ) -> dict[str, object]:
        assert events[:4] == [
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

    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )
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
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device=None: events.append(f"sync:{device}"),
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "release:b",
        "collect:a",
        "collect:b",
        "state:a:prefetching_gpu",
        "state:b:prefetching_gpu",
        "prepare:hot-a,hot-b:2:cpp_batch",
        "sync:2",
        "commit:hot-a,hot-b:True",
        "state:a:gpu",
        "state:b:gpu",
    ]
    assert runner._uvm_init_weight_placement_done is True


def test_init_weight_placement_sync_failure_marks_hot_unknown(
    monkeypatch: Any,
) -> None:
    states: list[str] = []
    commits: list[bool] = []

    module = SimpleNamespace(
        uvm_weight_executor="cpp_batch",
        run_uvm_init_weight_offload=lambda: None,
        get_uvm_init_hot_expert_region_ids=lambda: ["hot"],
        set_uvm_init_hot_expert_state=lambda state: states.append(state),
    )
    runner = SimpleNamespace(
        _run_uvm_matched_cpu_offload_manifest=lambda: 0,
        _env_flag_enabled=lambda _name, default=False: True,
        get_model=lambda: SimpleNamespace(modules=lambda: [module]),
    )

    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: {"ok": True},
    )

    def commit(
        _region_ids: list[str],
        *,
        synchronized: bool,
    ) -> dict[str, object]:
        commits.append(synchronized)
        return {"ok": synchronized}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        commit,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "is_available", lambda: True
    )
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)

    def fail_sync(_device: int) -> None:
        raise RuntimeError("mock sync failure")

    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "synchronize", fail_sync)

    with pytest.raises(RuntimeError, match="mock sync failure"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert commits == [False]
    assert states == ["prefetching_gpu", "unknown"]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_registration_failure_skips_hot_prefetch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    def fail_registration() -> None:
        events.append("register")
        raise RuntimeError("Explicit UVM driver region registration failed")

    module = SimpleNamespace(
        uvm_weight_executor="cpp_batch",
        run_uvm_init_weight_offload=fail_registration,
        get_uvm_init_hot_expert_region_ids=lambda: events.append("collect") or ["hot"],
        set_uvm_init_hot_expert_state=lambda state: events.append(f"state:{state}"),
    )
    runner = SimpleNamespace(
        _run_uvm_matched_cpu_offload_manifest=lambda: 0,
        _env_flag_enabled=lambda _name, default=False: True,
        get_model=lambda: SimpleNamespace(modules=lambda: [module]),
    )

    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prefetch") or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="driver region registration failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == ["register"]
    assert runner._uvm_init_weight_placement_done is False


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
        "mixed-reconciliation",
        [4, 5, 6],
        "plan-token",
    )

    assert result["ok"] is True
    assert result["plan_token"] == "plan-token"
    assert result["requested_cpu_block_ids"] == [6]
    assert result["requested_gpu_block_ids"] == [4, 5]
    assert result["completed_cpu_block_ids"] == [6]
    assert result["completed_gpu_block_ids"] == [4, 5]
    assert result["synchronized"] is True
    assert events == [
        ("collect", (6,)),
        ("offload", 1),
        ("sync", 0),
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
        "partial-reconciliation",
        [4, 5, 6],
        "partial-token",
    )

    assert result["ok"] is False
    assert result["reason"] == "partial"
    assert result["made_progress"] is True
    assert result["completed_block_ids"] == [6]
    assert result["completed_cpu_block_ids"] == [6]
    assert result["completed_gpu_block_ids"] == []
    assert result["synchronized"] is True


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
        "sync-failure",
        [],
        "sync-token",
    )

    assert result["ok"] is False
    assert result["reason"] == "migration_or_sync_exception"
    assert result["made_progress"] is True
    assert result["completed_block_ids"] == []
    assert result["synchronized"] is False
