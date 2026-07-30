# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace
from typing import Any

from vllm.v1.engine.core import EngineCore


class _FakeScheduler:
    def __init__(self, policy: Any) -> None:
        self.kv_cache_manager = SimpleNamespace(
            uvm_kv_runtime_policy=policy,
            block_pool=object(),
        )


def _make_core(*, applied_target: int = 8, total_blocks: int = 32) -> EngineCore:
    core = object.__new__(EngineCore)
    policy = SimpleNamespace(
        _dynamic_pool_applied_target_blocks=applied_target,
        total_blocks=total_blocks,
        dynamic_pool_active=lambda: True,
    )
    core.scheduler = _FakeScheduler(policy)  # type: ignore[assignment]
    core.batch_queue = None
    core._uvm_pending_kv_grow_target_blocks = None
    core._uvm_pending_kv_grow_reason = ""
    return core


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
        assert args == (8, 12, "test_pressure")
        events.append("physical")
        return [{"ok": True}]

    def fake_apply(
        target_blocks: int,
        *,
        block_pool: object,
        reason: str,
        metadata: dict[str, object],
    ) -> dict[str, object]:
        assert target_blocks == 12
        assert block_pool is core.scheduler.kv_cache_manager.block_pool
        assert reason == "test_pressure"
        assert metadata["physical_migration_complete"] is True
        events.append("apply")
        return {"ok": True, "applied_target_blocks": 12}

    monkeypatch.setattr(core, "collective_rpc", fake_collective_rpc)
    policy.apply_unified_kv_target_blocks = fake_apply

    result = core._resize_uvm_kv_target_with_workers(
        12,
        reason="test_pressure",
        release_expert_wave=True,
    )

    assert result["ok"] is True
    assert result["selected_experts_by_layer"] == {
        "model.layers.7.mlp.experts": 6
    }
    assert events == ["candidates", "expert", "physical", "apply"]


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
