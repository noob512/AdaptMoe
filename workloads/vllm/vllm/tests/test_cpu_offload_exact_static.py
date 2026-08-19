# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
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

    def copy_(
        self, source: _FakeTensor, *, non_blocking: bool = False
    ) -> _FakeTensor:
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
            pinned=(
                allocation_pinned
                if torch.device(device).type == "cpu"
                else False
            ),
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
    monkeypatch.setenv(
        "VLLM_CPU_OFFLOAD_STATIC_BANK", "1" if static_bank else "0"
    )
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
    exact_file = _write_exact_names(
        tmp_path, ["layers.0.weight", "layers.1.weight"]
    )
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
    exact_file = _write_exact_names(
        tmp_path, ["layers.0.left", "layers.0.right"]
    )
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

    records = [
        json.loads(line) for line in manifest_file.read_text().splitlines()
    ]
    parameter = next(
        record for record in records if record["event"] == "cpu_offload_parameter"
    )
    forward = next(
        record
        for record in records
        if record["event"] == "cpu_offload_copy_forward"
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
