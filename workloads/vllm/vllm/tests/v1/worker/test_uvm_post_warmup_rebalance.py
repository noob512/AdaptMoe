# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

import pytest

from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker.gpu_model_runner import _uvm_kv_tail_range_for_allocation
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


def test_post_warmup_rebalance_order_and_reserved_gate(
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

        def run_uvm_init_time_weight_placement(
            self,
            *,
            prealloc_reserved: bool,
        ) -> None:
            assert prealloc_reserved is False
            events.append("experts")

        def _maybe_prealloc_uvm_expert_reserved_slots_after_stage_m(
            self,
        ) -> dict[str, Any]:
            events.append("slots")
            return {"requested": True, "ok": True}

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "4")
    monkeypatch.setenv("VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE", str(control_file))
    monkeypatch.setenv("VLLM_USE_UVM", "1")
    monkeypatch.setenv("VLLM_UVM_WEIGHT_EXECUTOR", "expert_reserved_regions_async")
    monkeypatch.delenv("VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY", raising=False)

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: True

    worker._run_uvm_post_warmup_rebalance()

    assert events == ["kv:4", "experts", "slots"]
    assert os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] == "1"

    worker._run_uvm_post_warmup_rebalance()
    assert events == ["kv:4", "experts", "slots"]


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


def test_reserved_runtime_stays_closed_when_preallocation_fails(
    monkeypatch: Any,
) -> None:
    class FakeModelRunner:
        def run_uvm_init_time_weight_placement(
            self,
            *,
            prealloc_reserved: bool,
        ) -> None:
            assert prealloc_reserved is False

        def _maybe_prealloc_uvm_expert_reserved_slots_after_stage_m(
            self,
        ) -> dict[str, Any]:
            return {"requested": True, "ok": False, "reason": "test_failure"}

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_WEIGHT_EXECUTOR", "expert_reserved_regions_async")
    monkeypatch.delenv("VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY", raising=False)

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()

    with pytest.raises(RuntimeError, match="preallocation was not ready"):
        worker._run_uvm_post_warmup_rebalance()

    assert os.environ["VLLM_UVM_EXPERT_RESERVED_RUNTIME_READY"] == "0"
