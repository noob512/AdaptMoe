# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.device_allocator.uvm_region import RegionState
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.engine.core import EngineCore


class _FakeScheduler:
    def __init__(self, policy: Any) -> None:
        self.kv_cache_manager = SimpleNamespace(
            uvm_kv_runtime_policy=policy,
            block_pool=object(),
        )

    @staticmethod
    def has_unfinished_requests() -> bool:
        return False


def _make_core(*, applied_target: int = 8, total_blocks: int = 32) -> EngineCore:
    core = object.__new__(EngineCore)
    policy = SimpleNamespace(
        _dynamic_pool_applied_target_blocks=applied_target,
        _dynamic_pool_reconciliation_block_ids=set(),
        _dynamic_pool_external_reconciliation_required=False,
        total_blocks=total_blocks,
        dynamic_pool_physical_migration_enable=True,
        dynamic_pool_active=lambda: True,
    )
    policy.get_dynamic_pool_reconciliation_block_ids = lambda: sorted(
        policy._dynamic_pool_reconciliation_block_ids
    )
    core.scheduler = _FakeScheduler(policy)  # type: ignore[assignment]
    core.batch_queue = None
    core._uvm_pending_kv_grow_target_blocks = None
    core._uvm_pending_kv_grow_reason = ""
    return core


def _make_real_policy_core() -> tuple[EngineCore, UvmKvRuntimePolicy, BlockPool]:
    policy = object.__new__(UvmKvRuntimePolicy)
    policy.dynamic_pool_enable = True
    policy.dynamic_pool_target_blocks = 6
    policy.dynamic_pool_physical_migration_enable = True
    policy.dynamic_pool_physical_migration_sync = True
    policy.dynamic_pool_physical_migration_executor = "auto"
    policy.dynamic_pool_physical_migration_device = 0
    policy.num_kv_cache_groups = 1
    policy.total_blocks = 6
    policy._dynamic_pool_applied_target_blocks = 6
    policy._dynamic_pool_generation = 0
    policy._dynamic_pool_saved_lifecycle = {}
    policy._dynamic_pool_reconciliation_needed = False
    policy._dynamic_pool_reconciliation_block_ids = set()
    policy._dynamic_pool_external_reconciliation_required = False
    policy._logical_kv_blocks = {}
    pool = BlockPool(
        num_gpu_blocks=7,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,
    )
    policy.register_stage_l5_kv_blocks(pool)
    core = object.__new__(EngineCore)
    core.scheduler = _FakeScheduler(policy)  # type: ignore[assignment]
    core.scheduler.kv_cache_manager.block_pool = pool
    core.batch_queue = None
    core._uvm_pending_kv_grow_target_blocks = None
    core._uvm_pending_kv_grow_reason = ""
    return core, policy, pool


def _seed_partial_shrink(
    policy: UvmKvRuntimePolicy,
    pool: BlockPool,
) -> None:
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_unknown(block=pool.blocks[block_id])
    policy.dynamic_pool_target_blocks = 3
    policy._dynamic_pool_reconciliation_needed = True
    policy._dynamic_pool_reconciliation_block_ids = {4, 5, 6}


def _worker_resize_result(
    *,
    ok: bool,
    made_progress: bool,
) -> dict[str, object]:
    return {"ok": ok, "made_progress": made_progress}


def _queue_ids(queue: Any) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


def _make_policy_from_env() -> UvmKvRuntimePolicy:
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=4096)],
        num_blocks=7,
        kv_cache_groups=[object()],
    )
    return UvmKvRuntimePolicy(kv_cache_config)  # type: ignore[arg-type]


def test_single_gpu_validation_catches_automatic_post_warmup_pool(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("VLLM_UVM_KV_RUNTIME_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", raising=False)
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    policy = _make_policy_from_env()
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))

    assert policy.dynamic_pool_active()
    with pytest.raises(RuntimeError, match="supports one GPU only"):
        EngineCore._validate_uvm_dynamic_pool_single_gpu(config, policy)  # type: ignore[arg-type]


def test_single_gpu_validation_allows_runtime_disabled_policy(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_RUNTIME_ENABLE", "0")
    monkeypatch.setenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "1")
    policy = _make_policy_from_env()
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))

    assert not policy.dynamic_pool_active()
    EngineCore._validate_uvm_dynamic_pool_single_gpu(config, policy)  # type: ignore[arg-type]


def test_preemption_grows_per_preempted_request(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION", "3")
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    calls: list[tuple[int, str, bool]] = []

    def fake_resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", fake_resize)
    output = SimpleNamespace(preempted_req_ids={"r1", "r2"})

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert calls == [(14, "scheduler_preemption_pressure", True)]


def test_preemption_waits_for_batch_queue_to_drain(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    core.batch_queue = deque([(object(), object())])  # type: ignore[assignment]
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION", "4")
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    calls: list[tuple[int, str, bool]] = []

    def fake_resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", fake_resize)
    output = SimpleNamespace(preempted_req_ids={"r1"})

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert calls == []
    assert core._uvm_pending_kv_grow_target_blocks == 12
    assert core._apply_pending_uvm_kv_growth_if_ready() is None

    core.batch_queue.clear()
    result = core._apply_pending_uvm_kv_growth_if_ready()
    assert result == {"ok": True}
    assert calls == [
        (
            12,
            "scheduler_preemption_pressure_after_batch_queue_drain",
            True,
        )
    ]
    assert core._uvm_pending_kv_grow_target_blocks is None


def test_worker_resize_commits_target_after_expert_and_physical_steps(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    events: list[str] = []

    def prepare(
        target_blocks: int,
        *,
        block_pool: object,
        reason: str,
    ) -> dict[str, object]:
        assert target_blocks == 12
        assert block_pool is core.scheduler.kv_cache_manager.block_pool
        assert reason == "test_pressure"
        events.append("prepare")
        return {"ok": True}

    def finalize(
        preparation: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        assert preparation["ok"] is True
        assert kwargs["worker_ok"] is True
        assert kwargs["worker_made_progress"] is True
        events.append("apply")
        return {"ok": True}

    policy.prepare_dynamic_pool_resize = prepare
    policy.finalize_dynamic_pool_resize = finalize

    def fake_collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        if method == "get_uvm_incremental_expert_offload_candidates":
            assert args == ("cold",)
            events.append("candidates")
            return [
                {
                    "ok": True,
                    "candidates": {"model.layers.7.mlp.experts": 6},
                }
            ]
        if method == "run_uvm_incremental_expert_offload":
            assert args == (
                False,
                {"model.layers.7.mlp.experts": 6},
            )
            events.append("expert")
            return [{"ok": True}]
        assert method == "resize_uvm_kv_cache_target"
        assert args == (8, 12, [])
        events.append("physical")
        return [{"ok": True, "made_progress": True}]

    monkeypatch.setattr(core, "collective_rpc", fake_collective_rpc)

    result = core._resize_uvm_kv_target(
        12,
        reason="test_pressure",
        release_expert_wave=True,
    )

    assert result == {"ok": True, "reason": "applied"}
    assert events == [
        "candidates",
        "expert",
        "prepare",
        "physical",
        "apply",
    ]


def test_equal_applied_target_with_dirty_blocks_runs_workers_and_reconciles(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    rpc_calls: list[tuple[object, ...]] = []

    def collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert method == "resize_uvm_kv_cache_target"
        assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        rpc_calls.append(args)
        return [
            _worker_resize_result(
                ok=True,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="cross-entry-revert",
        release_expert_wave=False,
    )

    assert result["ok"] is True
    assert rpc_calls == [(6, 6, [4, 5, 6])]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert not policy._dynamic_pool_reconciliation_needed
    assert not policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )


def test_worker_partial_does_not_commit_and_requires_external_retry(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            _worker_resize_result(
                ok=False,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="worker-partial",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="ordinary-refresh",
    ) == {"ok": False, "reason": "external_reconciliation_required"}


def test_unexpected_worker_cardinality_keeps_entire_plan_quarantined(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        complete = _worker_resize_result(
            ok=True,
            made_progress=True,
        )
        incomplete = _worker_resize_result(
            ok=False,
            made_progress=True,
        )
        return [complete, incomplete]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="worker-cardinality-mismatch",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]


def test_worker_rpc_exception_marks_clean_shrink_dirty_before_return(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(*_args: object, **_kwargs: object) -> list[object]:
        assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        raise RuntimeError("mock worker transport failure")

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        3,
        reason="rpc-exception",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )


def test_zero_progress_worker_failure_rolls_back_new_quarantine(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        return [
            _worker_resize_result(
                ok=False,
                made_progress=False,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        3,
        reason="worker-zero-progress",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert not policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []


def test_dynamic_expert_swap_plan_preserves_count_and_uses_heat_delta() -> None:
    plan = EngineCore._plan_uvm_dynamic_expert_swaps(
        [
            {
                "ok": True,
                "layers": {
                    "layer.7": {
                        "global_num_experts": 6,
                        "local_num_experts": 6,
                        "expert_token_counts": [0, 2, 80, 1, 5, 100],
                        "remote_experts": [0, 5],
                    }
                },
            }
        ],
        max_swaps_per_layer=1,
        min_count_delta=10,
    )

    assert plan["ok"] is True
    assert plan["swaps_by_layer"] == {
        "layer.7": [
            {
                "promote_expert": 5,
                "demote_expert": 3,
                "promote_count": 100,
                "demote_count": 1,
                "count_delta": 99,
            }
        ]
    }
    layer = plan["layers"]["layer.7"]
    assert layer["remote_count"] == 2
    assert layer["target_remote_experts"] == [0, 3]


def test_dynamic_expert_swap_plan_rejects_distributed_expert_slices() -> None:
    plan = EngineCore._plan_uvm_dynamic_expert_swaps(
        [
            {
                "ok": True,
                "layers": {
                    "layer.7": {
                        "global_num_experts": 8,
                        "local_num_experts": 4,
                        "expert_token_counts": [0] * 8,
                        "remote_experts": [0],
                    }
                },
            }
        ],
        max_swaps_per_layer=1,
        min_count_delta=1,
    )

    assert plan["ok"] is False
    assert "distributed_expert_placement_unsupported" in plan["errors"][0]
