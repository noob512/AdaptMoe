# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.models import utils as offload_utils


@dataclass
class _FakeStream:
    cuda_stream: int


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (2, 2),
        stride: tuple[int, ...] = (2, 1),
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda:0",
        pinned: bool = False,
    ) -> None:
        self._shape = shape
        self._stride = stride
        self.dtype = dtype
        self.layout = torch.strided
        self.device = torch.device(device)
        self._pinned = pinned
        self.copy_calls: list[tuple[_FakeTensor, bool]] = []
        self.to_calls: list[tuple[torch.device, bool, _FakeTensor]] = []

    def size(self) -> tuple[int, ...]:
        return self._shape

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def numel(self) -> int:
        result = 1
        for value in self._shape:
            result *= value
        return result

    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def is_pinned(self) -> bool:
        return self._pinned

    def copy_(self, source: _FakeTensor, *, non_blocking: bool = False) -> _FakeTensor:
        self.copy_calls.append((source, non_blocking))
        return self

    def to(
        self, device: str | torch.device, *, non_blocking: bool = False
    ) -> _FakeTensor:
        target = _FakeTensor(
            shape=self._shape,
            stride=self._stride,
            dtype=self.dtype,
            device=device,
        )
        self.to_calls.append((torch.device(device), non_blocking, target))
        return target


@dataclass
class _FakeParameter:
    data: _FakeTensor

    @property
    def device(self) -> torch.device:
        return self.data.device


class _FakeModule:
    def __init__(
        self,
        parameters: dict[str, _FakeTensor],
        *,
        extra_state: dict[str, _FakeTensor] | None = None,
    ) -> None:
        self._parameters = {
            name: _FakeParameter(tensor) for name, tensor in parameters.items()
        }
        self._extra_state = extra_state or {}

    def parameters(self) -> Iterator[_FakeParameter]:
        return iter(self._parameters.values())

    def named_parameters(self) -> Iterator[tuple[str, _FakeParameter]]:
        return iter(self._parameters.items())

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {
            **{name: parameter.data for name, parameter in self._parameters.items()},
            **self._extra_state,
        }

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return args


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch):
    allocations: list[_FakeTensor] = []
    functional_states: list[dict[str, _FakeTensor]] = []
    stream = _FakeStream(17)
    allocation_pinned = True

    def empty_strided(*, size, stride, dtype, layout, device, **kwargs):
        tensor = _FakeTensor(
            shape=tuple(size),
            stride=tuple(stride),
            dtype=dtype,
            device=device,
            pinned=(allocation_pinned if torch.device(device).type == "cpu" else False),
        )
        allocations.append(tensor)
        return tensor

    def functional_call(module, state, *, args, kwargs):
        functional_states.append(state)
        return state

    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: True)
    monkeypatch.setattr(offload_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(offload_utils.torch, "empty_strided", empty_strided)
    monkeypatch.setattr(offload_utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(offload_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        offload_utils.torch.cuda, "current_stream", lambda device=None: stream
    )
    monkeypatch.setattr(offload_utils, "functional_call", functional_call)
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_COPY_SYNC", raising=False)
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_COPY_TRACE", raising=False)

    class Controls:
        @property
        def allocations(self) -> list[_FakeTensor]:
            return allocations

        @property
        def functional_states(self) -> list[dict[str, _FakeTensor]]:
            return functional_states

        def set_stream(self, value: int) -> None:
            stream.cuda_stream = value

        def make_pageable(self) -> None:
            nonlocal allocation_pinned
            allocation_pinned = False

    return Controls()


def _write_exact_names(tmp_path, names: list[str], *, json_format: bool = True):
    exact_file = tmp_path / "exact_names.txt"
    if json_format:
        exact_file.write_text(json.dumps(names), encoding="utf-8")
    else:
        exact_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    return exact_file


def _configure_exact(
    monkeypatch: pytest.MonkeyPatch,
    exact_file,
    *,
    static_bank: bool,
    budget: int = 1 << 20,
) -> None:
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", str(exact_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_STATIC_BANK", "1" if static_bank else "0")
    offload_utils.set_cpu_offload_max_bytes(budget)


@pytest.mark.parametrize("json_format", [True, False])
def test_exact_name_file_formats(tmp_path, json_format):
    names = ["model.layers.0.a", "model.layers.1.b"]
    exact_file = _write_exact_names(tmp_path, names, json_format=json_format)
    assert offload_utils._load_cpu_offload_exact_names(str(exact_file)) == frozenset(
        names
    )


@pytest.mark.parametrize(
    "contents",
    ["[]", '["a", "a"]', '["a", 1]', "[not-json", "\n\n"],
)
def test_exact_name_file_rejects_invalid_input(tmp_path, contents):
    exact_file = tmp_path / "invalid.txt"
    exact_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        offload_utils._load_cpu_offload_exact_names(str(exact_file))


def test_exact_gate_and_native_copy_forward(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.cold"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    hot = _FakeTensor()
    cold = _FakeTensor()
    module = _FakeModule({"hot": hot, "cold": cold})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    assert module._parameters["hot"].data is hot
    cold_cpu = module._parameters["cold"].data
    assert cold_cpu.device.type == "cpu"
    module.forward()
    assert cold_cpu.to_calls[0][:2] == (torch.device("cuda:0"), True)
    assert fake_cuda.functional_states[-1]["cold"].device.type == "cuda"
    assert offload_utils._CPU_OFFLOAD_BYTES == cold.numel() * cold.element_size()


def test_static_bank_reuses_signature_and_lane_across_layers(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight", "layers.1.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    source0 = _FakeTensor()
    source1 = _FakeTensor()
    module0 = _FakeModule({"weight": source0})
    module1 = _FakeModule({"weight": source1})

    offload_utils.maybe_offload_to_cpu(module0, module_prefix="layers.0")
    offload_utils.maybe_offload_to_cpu(module1, module_prefix="layers.1")

    gpu_allocations = [
        tensor for tensor in fake_cuda.allocations if tensor.device.type == "cuda"
    ]
    assert len(gpu_allocations) == 1
    bank = gpu_allocations[0]
    cpu0 = module0._parameters["weight"].data
    cpu1 = module1._parameters["weight"].data
    module0.forward()
    module1.forward()
    assert bank.copy_calls[-2:] == [(cpu0, True), (cpu1, True)]
    assert fake_cuda.functional_states[-2]["weight"] is bank
    assert fake_cuda.functional_states[-1]["weight"] is bank
    assert offload_utils._CPU_OFFLOAD_STATIC_BANK_BYTES == 16


def test_static_bank_uses_distinct_lanes_within_module(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.left", "layers.0.right"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule({"left": _FakeTensor(), "right": _FakeTensor()})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    module.forward()

    state = fake_cuda.functional_states[-1]
    assert state["left"] is not state["right"]
    assert len(offload_utils._CPU_OFFLOAD_STATIC_BANKS) == 2


def test_static_bank_rejects_unregistered_cpu_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule(
        {"weight": _FakeTensor()},
        extra_state={"unexpected": _FakeTensor(device="cpu", pinned=True)},
    )
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    with pytest.raises(RuntimeError, match="unregistered CPU"):
        module.forward()


def test_exact_mode_is_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: False)
    module = _FakeModule({"weight": _FakeTensor()})
    with pytest.raises(RuntimeError, match="requires pinned"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: True)
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    monkeypatch.setattr(offload_utils.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="single GPU"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")


def test_pageable_allocation_and_missing_name_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    fake_cuda.make_pageable()
    module = _FakeModule({"weight": _FakeTensor()})
    with pytest.raises(RuntimeError, match="pageable"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    exact_file = _write_exact_names(tmp_path, ["layers.0.missing"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    with pytest.raises(RuntimeError, match="did not find"):
        offload_utils._require_all_copy_exact_names_seen()


def test_thread_and_stream_changes_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    thread_id = 100
    monkeypatch.setattr(offload_utils.threading, "get_ident", lambda: thread_id)
    module = _FakeModule({"weight": _FakeTensor()})
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    thread_id = 101
    with pytest.raises(RuntimeError, match="one thread"):
        module.forward()
    thread_id = 100
    module.forward()
    fake_cuda.set_stream(99)
    with pytest.raises(RuntimeError, match="one CUDA stream"):
        module.forward()


def test_copy_backend_without_exact_file_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", raising=False)
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_STATIC_BANK", raising=False)
    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: False)
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    module = _FakeModule({"weight": _FakeTensor()})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    cpu_weight = module._parameters["weight"].data
    assert cpu_weight.device.type == "cpu"
    module.forward()
    assert cpu_weight.to_calls[0][:2] == (torch.device("cuda:0"), True)


def test_manifest_records_exact_static_bank_fields(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    manifest_file = tmp_path / "manifest.jsonl"
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_MANIFEST_FILE", str(manifest_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_COPY_TRACE", "1")
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule({"weight": _FakeTensor()})
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    module.forward()

    records = [json.loads(line) for line in manifest_file.read_text().splitlines()]
    parameter = next(
        record for record in records if record["event"] == "cpu_offload_parameter"
    )
    forward = next(
        record for record in records if record["event"] == "cpu_offload_copy_forward"
    )
    for record in (parameter, forward):
        assert record["copy_exact_names"] is True
        assert record["copy_static_bank"] is True
        assert record["copy_bank_bytes"] == 16


def test_static_bank_requires_exact_file(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", raising=False)
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_STATIC_BANK", "1")
    with pytest.raises(ValueError, match="requires"):
        offload_utils.set_cpu_offload_max_bytes(1 << 20)


def test_graph_static_alias_plan_uses_layer_parity_and_unique_consumers():
    names = frozenset(
        f"model.layers.{layer}.mlp.experts.{weight}"
        for layer in range(4)
        for weight in ("w13_weight", "w2_weight")
    )
    plan = offload_utils._build_graph_static_alias_plan(names)

    for weight in ("w13_weight", "w2_weight"):
        members = [
            plan[f"model.layers.{layer}.mlp.experts.{weight}"] for layer in range(4)
        ]
        assert [member.bank_id for member in members] == [0, 1, 0, 1]
        assert [member.consumer_index for member in members] == [0, 0, 1, 1]
        assert [member.consumer_count for member in members] == [2, 2, 2, 2]


class _FakeVmmAllocation:
    def __init__(self, token: int, *, fail_close: bool = False) -> None:
        self.token = token
        self.fail_close = fail_close
        self.closed = False
        self.calls: list[object] = []

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def close(self, *, synchronize: bool = False) -> None:
        self.calls.append(("close", synchronize))
        if self.fail_close:
            raise RuntimeError("fault-injected close failure")
        self.closed = True


def _fake_vmm_arena(
    allocation: _FakeVmmAllocation,
    *,
    layer_index: int,
    backing_bytes: int,
):
    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=layer_index,
        local_name="weight",
        bank_id=layer_index & 1,
        consumer_index=0,
        consumer_count=1,
    )
    arena = offload_utils._GraphStaticAliasArena(
        allocation=allocation,
        upload=object(),
        signature=((backing_bytes,), (1,), torch.uint8, 0, backing_bytes, 1),
        requested_bytes=backing_bytes,
        aligned_bytes=backing_bytes,
        backing_bytes=backing_bytes,
    )
    binding = offload_utils._GraphStaticAliasBinding(
        arena=arena,
        consumer=object(),
        plan=plan,
    )
    arena.bindings.append(binding)
    return arena, binding


def _set_vmm_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
    arenas: dict[tuple[str, str, int], Any],
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS", arenas
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        sum(arena.backing_bytes for arena in arenas.values()),
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE", False
    )


def test_vmm_destroy_requires_graph_teardown_without_releasing_references(
    monkeypatch: pytest.MonkeyPatch,
):
    allocation = _FakeVmmAllocation(11)
    arena, binding = _fake_vmm_arena(
        allocation, layer_index=0, backing_bytes=4096
    )
    arenas = {("model.layers", "weight", 0): arena}
    _set_vmm_destroy_state(monkeypatch, arenas)

    result = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=False
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_graphs_still_live"
    assert allocation.calls == []
    assert arena.upload is not None
    assert binding.consumer is not None
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == arenas


def test_vmm_destroy_releases_views_synchronizes_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    allocation = _FakeVmmAllocation(12)
    arena, binding = _fake_vmm_arena(
        allocation, layer_index=0, backing_bytes=8192
    )
    _set_vmm_destroy_state(
        monkeypatch, {("model.layers", "weight", 0): arena}
    )

    result = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True,
        synchronize=False,
    )

    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["destroyed_arena_count"] == 1
    assert result["released_backing_bytes"] == 8192
    assert allocation.calls == ["synchronize", ("close", False)]
    assert arena.upload is None
    assert binding.consumer is None
    assert arena.bindings == []
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {}
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 0

    again = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )
    assert again["ok"] is True
    assert again["reason"] == "no_active_allocations"
    assert allocation.calls == ["synchronize", ("close", False)]


def test_vmm_destroy_retains_only_failed_arenas_for_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    first_allocation = _FakeVmmAllocation(21)
    second_allocation = _FakeVmmAllocation(22, fail_close=True)
    first_arena, first_binding = _fake_vmm_arena(
        first_allocation, layer_index=0, backing_bytes=4096
    )
    second_arena, second_binding = _fake_vmm_arena(
        second_allocation, layer_index=1, backing_bytes=8192
    )
    first_key = ("model.layers", "weight", 0)
    second_key = ("model.layers", "weight", 1)
    _set_vmm_destroy_state(
        monkeypatch,
        {first_key: first_arena, second_key: second_arena},
    )

    failed = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )

    assert failed["ok"] is False
    assert failed["reason"] == "vmm_close_failed"
    assert failed["destroyed_arena_count"] == 1
    assert failed["remaining_arena_count"] == 1
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {
        second_key: second_arena
    }
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 8192
    assert first_binding.consumer is None
    assert second_binding.consumer is None

    second_allocation.fail_close = False
    retried = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )
    assert retried["ok"] is True
    assert retried["destroyed_arena_count"] == 1
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {}
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 0


@pytest.mark.parametrize("backend", ["vmm", "uvm_original_va"])
def test_runner_shutdown_resets_graphs_before_static_alias_destroy(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
):
    from vllm.v1.worker import gpu_model_runner as runner_module

    events: list[object] = []

    class FakeGraph:
        def reset(self) -> None:
            events.append("graph_reset")

    graph = FakeGraph()
    entry = SimpleNamespace(
        cudagraph=graph,
        output=object(),
        input_addresses=object(),
    )

    class FakeCudagraphWrapper:
        def __init__(self) -> None:
            self.concrete_cudagraph_entries = {"entry": entry}

    monkeypatch.setattr(runner_module, "CUDAGraphWrapper", FakeCudagraphWrapper)
    monkeypatch.setattr(
        offload_utils,
        "get_cpu_offload_graph_static_alias_diagnostics",
        lambda: {"enabled": True, "backend": backend},
    )

    def destroy(**kwargs: Any) -> dict[str, Any]:
        events.append(("destroy", kwargs))
        return {"ok": True, "backend": backend}

    monkeypatch.setattr(
        offload_utils, "destroy_cpu_offload_graph_static_alias", destroy
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "synchronize",
        lambda device=None: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        runner_module.gc, "collect", lambda: events.append("gc_collect")
    )

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cuda:0")
    runner.model = FakeCudagraphWrapper()
    result = runner.shutdown_cpu_offload_graph_static_alias()

    assert events == [
        ("synchronize", torch.device("cuda:0")),
        "graph_reset",
        "gc_collect",
        ("synchronize", torch.device("cuda:0")),
        (
            "destroy",
            {"graphs_destroyed": True, "synchronize": False},
        ),
    ]
    assert result["ok"] is True
    assert result["destroyed_graph_count"] == 1
    assert entry.cudagraph is None
    assert entry.output is None
    assert entry.input_addresses is None
    assert runner.model.concrete_cudagraph_entries == {}


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY"])
def test_graph_static_alias_runtime_accepts_full_modes(monkeypatch, mode_name):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=mode_name),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_allows_explicit_eager(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", "1")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE", "off"
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_rejects_implicit_eager(monkeypatch):
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", raising=False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE", "off"
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires single-GPU"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


@pytest.mark.parametrize(
    ("mode_name", "parallel_overrides"),
    [
        ("PIECEWISE", {}),
        ("FULL_AND_PIECEWISE", {}),
        ("FULL", {"enable_dbo": True}),
        ("FULL", {"tensor_parallel_size": 2}),
    ],
)
def test_graph_static_alias_runtime_rejects_overlap(
    monkeypatch, mode_name, parallel_overrides
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_dbo": False,
        **parallel_overrides,
    }
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=mode_name),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires single-GPU"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_rejects_dynamo_compile(monkeypatch):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            mode=SimpleNamespace(name="VLLM_COMPILE"),
        ),
    )
    with pytest.raises(RuntimeError, match="compilation_mode=VLLM_COMPILE"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_uvm_original_backend_requires_managed_expert_allocation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    exact_file = _write_exact_names(tmp_path, ["model.layers.0.weight"])
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", str(exact_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "1")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "uvm_original_va")
    monkeypatch.delenv("VLLM_UVM_EXPERT_WEIGHTS_MANAGED", raising=False)

    with pytest.raises(ValueError, match="EXPERT_WEIGHTS_MANAGED"):
        offload_utils.set_cpu_offload_max_bytes(1 << 20)

    monkeypatch.setenv("VLLM_UVM_EXPERT_WEIGHTS_MANAGED", "1")
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    assert offload_utils.cpu_offload_uvm_original_va_static_alias_enabled()
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "collecting"
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "0")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE")
    offload_utils.set_cpu_offload_max_bytes(0)


def _set_uvm_original_exact_names(
    monkeypatch: pytest.MonkeyPatch,
    names: frozenset[str],
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_NAMES", names)


@pytest.mark.parametrize("json_format", [True, False])
def test_uvm_original_managed_allocation_uses_only_selected_expert_layers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    json_format: bool,
):
    prefix = "model.layers.0.mlp.experts"
    names = [f"{prefix}.w13_weight", f"{prefix}.w2_weight"]
    exact_file = _write_exact_names(tmp_path, names, json_format=json_format)
    _set_uvm_original_exact_names(
        monkeypatch,
        offload_utils._load_cpu_offload_exact_names(str(exact_file)),
    )

    assert offload_utils.use_uvm_expert_weight_allocation(
        prefix,
        managed_requested=True,
    )
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.1.mlp.experts",
        managed_requested=True,
    )
    # The delimiter in the descendant match prevents layers.1/layers.10
    # prefix collisions.
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.10.mlp.experts",
        managed_requested=True,
    )


@pytest.mark.parametrize("local_name", ["w13_weight", "w2_weight"])
def test_uvm_original_managed_allocation_rejects_half_selected_expert(
    monkeypatch: pytest.MonkeyPatch,
    local_name: str,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset({f"{prefix}.{local_name}"}),
    )

    with pytest.raises(ValueError, match="complete w13_weight/w2_weight pair"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=True,
        )


def test_uvm_original_managed_allocation_rejects_unsupported_descendant(
    monkeypatch: pytest.MonkeyPatch,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset(
            {
                f"{prefix}.w13_weight",
                f"{prefix}.w2_weight",
                f"{prefix}.w13_weight_scale",
            }
        ),
    )

    with pytest.raises(ValueError, match="unexpected"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=True,
        )


def test_uvm_original_managed_allocation_rejects_configuration_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset({f"{prefix}.w13_weight", f"{prefix}.w2_weight"}),
    )

    with pytest.raises(RuntimeError, match="configuration was frozen"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=False,
        )


def test_legacy_managed_expert_allocation_remains_global(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "vmm",
    )

    assert offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.22.mlp.experts",
        managed_requested=True,
    )
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.22.mlp.experts",
        managed_requested=False,
    )


def test_uvm_original_capture_requires_sealed_topology(monkeypatch):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "collecting"
    )

    with pytest.raises(RuntimeError, match="mapped and sealed"):
        offload_utils.set_cpu_offload_graph_static_alias_capture_phase(True)


def test_uvm_original_multi_region_alias_geometry(monkeypatch):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan0 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=2,
    )
    plan2 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=2,
        local_name="weight",
        bank_id=0,
        consumer_index=1,
        consumer_count=2,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((1 << 20,), (1,), torch.uint8, 0, 1 << 20, 2),
        requested_bytes=1 << 20,
        aligned_bytes=1 << 20,
        backing_layer_id=0x70000001,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=512 * 1024,
    )
    binding0 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan0,
        requested_bytes=1 << 20,
    )
    binding2 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.2.weight",
        arena=arena,
        consumer=PointerTensor(0x600000),
        parameter=None,
        plan=plan2,
        requested_bytes=1 << 20,
    )
    arena.consumers = {0: binding0, 1: binding2}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )

    aliases = offload_utils._uvm_original_alias_map_entries(0)
    slots = offload_utils._uvm_original_alias_preallocate_entries()
    assert len(aliases) == 6
    assert len(slots) == 2
    assert [entry[4] for entry in aliases[:3]] == [
        0x200000,
        0x400000,
        0x600000,
    ]
    assert [entry[:3] for entry in aliases[:3]] == [
        (0x70000001, 0, 0),
    ] * 3
    assert [entry[:3] for entry in aliases[3:]] == [
        (0x70000001, 1, 1),
    ] * 3
    assert slots[0][0:6] == (
        0x200000,
        512 * 1024,
        0x70000001,
        0,
        0,
        0,
    )


def test_uvm_original_expert_slice_aliases_map_packed_offsets_to_original_vas(
    monkeypatch: pytest.MonkeyPatch,
):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan0 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=2,
    )
    plan2 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=2,
        local_name="weight",
        bank_id=0,
        consumer_index=1,
        consumer_count=2,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((2, 8192), (8192, 1), torch.uint8, 0, 16384, (2, 8192), 2),
        requested_bytes=16384,
        aligned_bytes=16384,
        backing_layer_id=0x70000001,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=4096,
        packed_slice_bytes=8192,
    )
    binding0 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan0,
        requested_bytes=16384,
        parameter_bytes=32768,
        expert_ids=(1, 3),
        expert_stride_bytes=8192,
        consumer_ranges=((0x402000, 8192), (0x406000, 8192)),
    )
    binding2 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.2.weight",
        arena=arena,
        consumer=PointerTensor(0x600000),
        parameter=None,
        plan=plan2,
        requested_bytes=16384,
        parameter_bytes=32768,
        expert_ids=(0, 2),
        expert_stride_bytes=8192,
        consumer_ranges=((0x600000, 8192), (0x604000, 8192)),
    )
    arena.consumers = {0: binding0, 1: binding2}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )

    aliases = offload_utils._uvm_original_alias_map_entries(0)
    slots = offload_utils._uvm_original_alias_preallocate_entries()

    assert len(aliases) == 12
    assert len(slots) == 4
    assert [entry[4] for entry in aliases[0:3]] == [
        0x200000,
        0x402000,
        0x600000,
    ]
    assert [entry[4] for entry in aliases[3:6]] == [
        0x201000,
        0x403000,
        0x601000,
    ]
    assert [entry[4] for entry in aliases[6:9]] == [
        0x202000,
        0x406000,
        0x604000,
    ]
    assert [entry[4] for entry in aliases[9:12]] == [
        0x203000,
        0x407000,
        0x605000,
    ]
    assert offload_utils._uvm_original_alias_binding_hot_ranges(binding0) == (
        (0x400000, 8192),
        (0x404000, 8192),
    )


def test_uvm_original_expert_plan_requires_equal_complete_layer_selection(
    tmp_path,
):
    names = frozenset(
        {
            "model.layers.0.experts.w13_weight",
            "model.layers.0.experts.w2_weight",
            "model.layers.1.experts.w13_weight",
            "model.layers.1.experts.w2_weight",
        }
    )
    plan = offload_utils._build_graph_static_alias_plan(names)
    plan_file = tmp_path / "cold.json"
    plan_file.write_text(
        json.dumps({"selected_by_layer": {"0": [7, 1], "1": [4, 2]}}),
        encoding="utf-8",
    )
    assert offload_utils._load_graph_static_alias_expert_plan(
        str(plan_file), plan
    ) == {0: (1, 7), 1: (2, 4)}

    plan_file.write_text(
        json.dumps({"selected_by_layer": {"0": [7], "1": [4, 2]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same number of experts"):
        offload_utils._load_graph_static_alias_expert_plan(str(plan_file), plan)


def test_uvm_original_destroy_clear_contract():
    assert offload_utils._uvm_original_alias_destroy_cleared({"ok": True})
    assert offload_utils._uvm_original_alias_destroy_cleared(
        {
            "ok": False,
            "cache_cleared": True,
            "session_cleared": True,
            "remaining_alias_count": 0,
            "remaining_alias_bytes": 0,
            "remaining_backing_bytes": 0,
        }
    )
    assert not offload_utils._uvm_original_alias_destroy_cleared(
        {
            "ok": False,
            "session_cleared": True,
            "remaining_alias_count": 1,
            "remaining_alias_bytes": 4096,
            "remaining_backing_bytes": 0,
        }
    )


def test_uvm_original_64k_mode_keeps_512k_region_alignment(
    monkeypatch: pytest.MonkeyPatch,
):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((2 << 20,), (1,), torch.uint8, 0, 2 << 20, 1),
        requested_bytes=2 << 20,
        aligned_bytes=2 << 20,
        backing_layer_id=0x70000000,
        pte_alignment=0,
        pte_flags=0,
        region_bytes=512 * 1024,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan,
        requested_bytes=2 << 20,
    )
    arena.consumers = {0: binding}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE", "auto")

    alignment, flags, mode = offload_utils._uvm_original_alias_select_pte_mode()
    assert alignment == 64 * 1024
    assert flags != 0
    assert mode == "64K"
    assert arena.region_bytes % alignment == 0


def _set_uvm_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: int,
    state: str,
    cleanup_unresolved: bool = False,
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", session_id
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", state)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED",
        cleanup_unresolved,
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT", None
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT", None
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_DEVICE_INDEX", 0)


def test_uvm_original_map_session_id_is_trusted_only_after_success():
    assert (
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 17}, 0
        )
        == 17
    )
    assert (
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 17}, 17
        )
        == 17
    )
    with pytest.raises(RuntimeError, match="did not succeed"):
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": False, "session_id": 99}, 0
        )
    with pytest.raises(RuntimeError, match="changed session id"):
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 99}, 17
        )


def test_uvm_original_failed_first_map_keeps_untrusted_session_unresolved(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm, uvm_static_alias

    class ManagedTensor:
        def __init__(self, pointer: int, *, device: str = "cuda:0"):
            self._pointer = pointer
            self.device = torch.device(device)
            self.dtype = torch.uint8
            self.layout = torch.strided

        def data_ptr(self):
            return self._pointer

        def size(self):
            return (4096,)

        def stride(self):
            return (1,)

        def is_pinned(self):
            return self.device.type == "cpu"

        def copy_(self, source, *, non_blocking=False):
            return self

    name = "model.layers.0.weight"
    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    consumer = ManagedTensor(0x400000)
    upload = ManagedTensor(0x200000)
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=upload,
        upload=upload,
        signature=((4096,), (1,), torch.uint8, 0, 4096, 1),
        requested_bytes=4096,
        aligned_bytes=4096,
        backing_layer_id=0x70000000,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=4096,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name=name,
        arena=arena,
        consumer=consumer,
        parameter=SimpleNamespace(data=consumer),
        plan=plan,
        requested_bytes=4096,
    )
    arena.consumers = {0: binding}

    _set_uvm_destroy_state(monkeypatch, session_id=0, state="collecting")
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN", {name: plan}
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS", {name: binding}
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils, "_require_all_copy_exact_names_seen", lambda: None
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_select_pte_mode",
        lambda: (4096, 0, "4K"),
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_preallocate_entries",
        lambda: [(0x200000, 4096, 0x70000000, 0, 0, 0, 4096)],
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_map_entries",
        lambda flags: [(0x70000000, 0, 0, flags, 0x400000, 4096, 0)],
    )
    monkeypatch.setattr(
        offload_utils.torch,
        "empty_strided",
        lambda **kwargs: ManagedTensor(0x800000, device="cpu"),
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)
    monkeypatch.setattr(
        uvm, "probe_driver_capable_range", lambda *args: (True, "ok", 0)
    )
    monkeypatch.setattr(
        uvm,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        lambda ranges, **kwargs: (len(ranges), 0),
    )
    monkeypatch.setattr(
        uvm,
        "driver_slot_preallocate_ranges",
        lambda entries, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    close_calls = []
    monkeypatch.setattr(
        uvm,
        "close_uvm_slot_direct_session",
        lambda **kwargs: close_calls.append(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_map",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "fault_injected",
            "session_id": 99,
            "made_progress": False,
            "poisoned": False,
            "side_effect_indeterminate": False,
        },
    )

    with pytest.raises(RuntimeError, match="MAP failed"):
        offload_utils.finalize_cpu_offload_graph_static_alias()

    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID == 0
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED is True
    assert close_calls == []
    diagnostics = offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
    assert diagnostics["cleanup_pending"] is True
    assert diagnostics["cleanup_unresolved"] is True


def test_uvm_original_destroy_retries_failed_slot_close(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm, uvm_static_alias

    _set_uvm_destroy_state(monkeypatch, session_id=7, state="sealed")
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_destroy",
        lambda *args, **kwargs: {"ok": True, "session_cleared": True},
    )
    close_results = iter(
        [
            {"ok": False, "reason": "fault_injected", "closed": False},
            {"ok": True, "reason": "closed", "closed": True},
        ]
    )
    monkeypatch.setattr(
        uvm, "close_uvm_slot_direct_session", lambda **kwargs: next(close_results)
    )

    first = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert first["ok"] is False
    assert first["reason"] == "slot_session_close_failed"
    assert first["state"] == "cleanup_pending"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID == 0
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"

    second = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert second["ok"] is True
    assert second["reason"] == "slot_session_close_retried"
    assert second["state"] == "remote"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote"


def test_uvm_original_unresolved_cleanup_never_closes_backing(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm

    _set_uvm_destroy_state(
        monkeypatch,
        session_id=0,
        state="cleanup_pending",
        cleanup_unresolved=True,
    )
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    close_calls = []
    monkeypatch.setattr(
        uvm,
        "close_uvm_slot_direct_session",
        lambda **kwargs: close_calls.append(kwargs) or {"ok": True},
    )

    result = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert result["ok"] is False
    assert result["reason"] == "cleanup_unresolved_requires_process_teardown"
    assert close_calls == []
    with pytest.raises(RuntimeError, match="cleanup is active"):
        offload_utils._configure_cpu_offload_exact_names(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", "off"),
        ("hybrid", "prefill_remote_decode"),
        ("alias-prefill-remote-decode", "prefill_remote_decode"),
        ("dual-topology-graph", "prefill_graph_remote_decode"),
        ("remote", "remote_only"),
    ],
)
def test_uvm_original_phase_mode_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: str,
):
    monkeypatch.setenv("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", raw)
    assert offload_utils._configured_uvm_original_va_phase_mode() == expected


def test_uvm_original_phase_runtime_allows_explicit_eager(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_uvm_dual_topology_runtime_requires_full_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_graph_remote_decode",
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_dbo=False,
    )
    valid = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_SEPARATE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(valid)

    invalid = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires cudagraph_mode=FULL_SEPARATE"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(invalid)


def test_full_separate_dispatcher_partitions_prefill_and_decode_keys():
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    config = SimpleNamespace(
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_SEPARATE,
            cudagraph_capture_sizes=[1, 4, 314, 1217],
            cudagraph_specialize_lora=False,
            is_attention_compiled_piecewise=lambda: False,
        ),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_SEPARATE, 1)
    assert dispatcher.cudagraph_keys[CUDAGraphMode.FULL] == {
        BatchDescriptor(1, uniform_decode=True),
        BatchDescriptor(4, uniform_decode=True),
        BatchDescriptor(314, uniform_decode=False),
        BatchDescriptor(1217, uniform_decode=False),
    }


def test_uvm_original_remote_capture_issues_zero_h2d_certificate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT",
        {"ok": True, "requested_bytes": 4096},
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000,),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=17),
    )

    with offload_utils.cpu_offload_graph_static_alias_capture_context():
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 3
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0

    certificate = offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    assert certificate is not None
    assert certificate["topology"] == "remote"
    assert certificate["parameter_vas"] == (0x400000,)
    assert certificate["capture_forward_count"] == 3
    assert certificate["actual_h2d_bytes"] == 0
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY == "remote"


def test_uvm_original_dual_graph_certificates_survive_alias_capture(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000, 0x800000),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=17),
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE", None)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE", None)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE",
        None,
    )

    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT",
        {"ok": True, "requested_bytes": 8192},
    )
    with offload_utils.cpu_offload_graph_static_alias_capture_context(
        expected_topology="remote"
    ):
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 2
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0

    remote_certificate = dict(
        offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE or {}
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        4096,
    )
    with offload_utils.cpu_offload_graph_static_alias_capture_context(
        expected_topology="sealed"
    ):
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 4
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 16384

    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE == (
        remote_certificate
    )
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE == {
        "schema": "uvm_original_va_alias_graph_certificate_v1",
        "topology": "sealed",
        "phase_mode": "prefill_graph_remote_decode",
        "parameter_vas": (0x400000, 0x800000),
        "capture_forward_count": 4,
        "actual_h2d_bytes": 16384,
        "session_id": 7,
        "backing_bytes": 4096,
    }

    partition = offload_utils.certify_cpu_offload_uvm_original_va_dual_graph_partition(
        remote_entries=[
            {"num_tokens": 1, "uniform_decode": True, "has_lora": False},
            {"num_tokens": 4, "uniform_decode": True, "has_lora": False},
        ],
        alias_entries=[
            {"num_tokens": 314, "uniform_decode": False, "has_lora": False},
            {"num_tokens": 1217, "uniform_decode": False, "has_lora": False},
        ],
    )
    assert partition["remote_entry_count"] == 2
    assert partition["alias_entry_count"] == 2

    teardown = (
        offload_utils.certify_cpu_offload_uvm_original_va_alias_graphs_destroyed(
            destroyed_alias_entries=[
                {
                    "num_tokens": 314,
                    "uniform_decode": False,
                    "has_lora": False,
                },
                {
                    "num_tokens": 1217,
                    "uniform_decode": False,
                    "has_lora": False,
                },
            ],
            remaining_remote_entries=[
                {"num_tokens": 1, "uniform_decode": True, "has_lora": False},
                {"num_tokens": 4, "uniform_decode": True, "has_lora": False},
            ],
        )
    )
    assert teardown["destroyed_alias_graph_count"] == 2
    assert teardown["remaining_remote_graph_count"] == 2
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert ok, reason


def test_uvm_original_preserved_graph_requires_zero_h2d_certificate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", True)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY", "remote")
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000,),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE",
        {
            "topology": "remote",
            "capture_forward_count": 1,
            "actual_h2d_bytes": 4096,
            "parameter_vas": (0x400000,),
        },
    )
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert not ok
    assert reason == "remote_graph_certificate_contains_h2d"

    offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE[
        "actual_h2d_bytes"
    ] = 0
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert ok
    assert reason == "ok"


def test_uvm_original_transition_to_remote_audits_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm

    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    pointer = PointerTensor(0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)

    destroy_calls: list[dict[str, Any]] = []

    def destroy(**kwargs):
        destroy_calls.append(kwargs)
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        return {"ok": True, "state": "remote", "session_id": 0}

    monkeypatch.setattr(
        offload_utils, "destroy_cpu_offload_graph_static_alias", destroy
    )
    remote = {
        "ok": True,
        "reason": "ok",
        "requested_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "requested_bytes": 4096,
        "cpu_resident_bytes": 4096,
        "remote_mapped_bytes": 4096,
        "covered_bytes": 4096,
        "unknown_bytes": 0,
    }
    monkeypatch.setattr(
        uvm,
        "driver_prepare_ranges_remote_read_audited",
        lambda ranges, **kwargs: remote,
    )
    monkeypatch.setattr(
        uvm,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, **kwargs: {
            "ok": True,
            "queried_bytes": 4096,
            "cpu_bytes": 4096,
            "gpu_bytes": 0,
            "any_bytes": 4096,
            "unknown_bytes": 0,
        },
    )

    result = offload_utils.transition_cpu_offload_uvm_original_va_to_remote(
        reason="test_cutover",
        preserve_remote_graphs=False,
    )
    assert result["ok"] is True
    assert result["pre_state"] == "sealed"
    assert result["post_state"] == "remote"
    assert result["parameter_vas_before"] == (0x400000,)
    assert result["parameter_vas_after"] == (0x400000,)
    assert result["remote_map"] == remote
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE is True
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote"
    assert destroy_calls == [
        {
            "graphs_destroyed": True,
            "synchronize": True,
            "remote_graphs_preserved": False,
        }
    ]
