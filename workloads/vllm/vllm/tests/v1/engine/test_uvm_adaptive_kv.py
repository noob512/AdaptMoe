# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace
from typing import Any

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
    old_target: int,
    new_target: int,
    plan_token: str,
    requested: list[int],
    requested_cpu: list[int],
    requested_gpu: list[int],
    completed: list[int],
    completed_cpu: list[int],
    completed_gpu: list[int],
    ok: bool,
    made_progress: bool,
) -> dict[str, object]:
    return {
        "ok": ok,
        "old_target_blocks": old_target,
        "new_target_blocks": new_target,
        "plan_token": plan_token,
        "requested_block_ids": requested,
        "requested_cpu_block_ids": requested_cpu,
        "requested_gpu_block_ids": requested_gpu,
        "completed_block_ids": completed,
        "completed_cpu_block_ids": completed_cpu,
        "completed_gpu_block_ids": completed_gpu,
        "made_progress": made_progress,
        "synchronized": made_progress,
    }


def _queue_ids(queue: Any) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


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

    monkeypatch.setattr(core, "_resize_uvm_kv_target_with_workers", fake_resize)
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

    monkeypatch.setattr(core, "_resize_uvm_kv_target_with_workers", fake_resize)
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
    plan_token = "test-plan"

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
        return {
            "ok": True,
            "plan_token": plan_token,
            "dirty_block_ids": [],
            "expected_block_ids": [9, 10, 11, 12],
            "expected_cpu_block_ids": [],
            "expected_gpu_block_ids": [9, 10, 11, 12],
        }

    def finalize(
        preparation: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        assert preparation["plan_token"] == plan_token
        assert kwargs["worker_ok"] is True
        assert kwargs["completed_block_ids"] == [9, 10, 11, 12]
        assert kwargs["completed_cpu_block_ids"] == []
        assert kwargs["completed_gpu_block_ids"] == [9, 10, 11, 12]
        events.append("apply")
        return {"ok": True, "applied_target_blocks": 12}

    policy.prepare_external_dynamic_pool_resize = prepare
    policy.finalize_external_dynamic_pool_resize = finalize

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
                    "candidates": {
                        "model.layers.7.mlp.experts": {
                            "global_expert_id": 6,
                        }
                    },
                }
            ]
        if method == "run_uvm_incremental_expert_offload":
            assert args == (
                False,
                {"model.layers.7.mlp.experts": 6},
            )
            events.append("expert")
            return [
                {
                    "ok": True,
                    "released_count": 1,
                    "results": [
                        {
                            "layer_key": "model.layers.7.mlp.experts",
                            "released_experts": [6],
                        }
                    ],
                }
            ]
        assert method == "resize_uvm_kv_cache_target"
        assert args == (8, 12, "test_pressure", [], plan_token)
        events.append("physical")
        return [
            {
                "ok": True,
                "old_target_blocks": 8,
                "new_target_blocks": 12,
                "plan_token": plan_token,
                "requested_block_ids": [9, 10, 11, 12],
                "requested_cpu_block_ids": [],
                "requested_gpu_block_ids": [9, 10, 11, 12],
                "completed_block_ids": [9, 10, 11, 12],
                "completed_cpu_block_ids": [],
                "completed_gpu_block_ids": [9, 10, 11, 12],
                "made_progress": True,
                "synchronized": True,
            }
        ]

    monkeypatch.setattr(core, "collective_rpc", fake_collective_rpc)

    result = core._resize_uvm_kv_target_with_workers(
        12,
        reason="test_pressure",
        release_expert_wave=True,
    )

    assert result["ok"] is True
    assert result["selected_experts_by_layer"] == {
        "model.layers.7.mlp.experts": 6
    }
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
        plan_token = str(args[4])
        return [
            _worker_resize_result(
                old_target=6,
                new_target=6,
                plan_token=plan_token,
                requested=[4, 5, 6],
                requested_cpu=[],
                requested_gpu=[4, 5, 6],
                completed=[4, 5, 6],
                completed_cpu=[],
                completed_gpu=[4, 5, 6],
                ok=True,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target_with_workers(
        6,
        reason="cross-entry-revert",
        release_expert_wave=False,
    )

    assert result["ok"] is True
    assert rpc_calls and rpc_calls[0][3] == [4, 5, 6]
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
                old_target=6,
                new_target=6,
                plan_token=str(args[4]),
                requested=[4, 5, 6],
                requested_cpu=[],
                requested_gpu=[4, 5, 6],
                completed=[],
                completed_cpu=[],
                completed_gpu=[],
                ok=False,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target_with_workers(
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


def test_worker_rank_completion_mismatch_keeps_entire_plan_quarantined(
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
        token = str(args[4])
        complete = _worker_resize_result(
            old_target=6,
            new_target=6,
            plan_token=token,
            requested=[4, 5, 6],
            requested_cpu=[],
            requested_gpu=[4, 5, 6],
            completed=[4, 5, 6],
            completed_cpu=[],
            completed_gpu=[4, 5, 6],
            ok=True,
            made_progress=True,
        )
        incomplete = _worker_resize_result(
            old_target=6,
            new_target=6,
            plan_token=token,
            requested=[4, 5, 6],
            requested_cpu=[],
            requested_gpu=[4, 5, 6],
            completed=[],
            completed_cpu=[],
            completed_gpu=[],
            ok=False,
            made_progress=True,
        )
        return [complete, incomplete]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target_with_workers(
        6,
        reason="rank-mismatch",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert result["finalize_result"]["completed_block_ids"] == []
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

    result = core._resize_uvm_kv_target_with_workers(
        3,
        reason="rpc-exception",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert result["finalize_result"]["made_progress"] is True
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
                old_target=6,
                new_target=3,
                plan_token=str(args[4]),
                requested=[4, 5, 6],
                requested_cpu=[4, 5, 6],
                requested_gpu=[],
                completed=[],
                completed_cpu=[],
                completed_gpu=[],
                ok=False,
                made_progress=False,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target_with_workers(
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


def test_incremental_candidate_merge_selects_one_global_expert_per_layer() -> None:
    selected = EngineCore._select_uvm_incremental_experts_by_layer(
        [
            {
                "ok": True,
                "candidates": {
                    "layer.7": {"global_expert_id": 6},
                    "layer.8": {"global_expert_id": 4},
                },
            },
            {
                "ok": True,
                "candidates": {
                    "layer.7": {"global_expert_id": 2},
                },
            },
        ]
    )

    assert selected == {"layer.7": 2, "layer.8": 4}


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
