# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest

from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.models import utils as offload_utils
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _FakeGraph:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1


def _wrapper_with_class(class_name: str, tokens: int) -> CUDAGraphWrapper:
    wrapper = object.__new__(CUDAGraphWrapper)
    entries: dict[BatchDescriptor, CUDAGraphEntry] = {}
    descriptor = BatchDescriptor(tokens, uniform_decode=False)
    entries[descriptor] = CUDAGraphEntry(descriptor, cudagraph=_FakeGraph())
    wrapper.concrete_cudagraph_entries = entries
    wrapper._cudagraph_class_entries = {class_name: entries}
    wrapper._active_cudagraph_class = class_name
    return wrapper


def _registry(active: str = "c8") -> dict[str, object]:
    return {
        "enabled": True,
        "active_class": active,
        "initial_class": "c8",
        "classes": {
            f"c{width}": {"experts_per_layer": width}
            for width in (2, 4, 6, 8)
        },
    }


def test_startup_adopts_initial_graph_table_without_recapture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_with_class("default", 8)
    runner = object.__new__(GPUModelRunner)
    runner.model = wrapper
    runner.device = 0
    capture_calls: list[str] = []
    runner.capture_model = lambda: capture_calls.append("capture") or 1  # type: ignore[method-assign]
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: _registry(),
    )
    monkeypatch.setattr(
        offload_utils,
        "register_cpu_offload_uvm_plan_class_graph_certificates",
        lambda class_name: {"ok": True, "class_name": class_name},
    )
    monkeypatch.setattr(
        offload_utils,
        "switch_cpu_offload_uvm_original_va_plan_class",
        lambda *args, **kwargs: pytest.fail("initial class must not switch"),
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.torch.cuda.synchronize",
        lambda device=0: None,
    )

    result = runner.capture_uvm_expert_graph_class_at_idle_boundary(
        "c8",
        scheduler_idle=True,
    )

    assert result["ok"] is True
    assert result["reason"] == "active_class_registered"
    assert result["graph_memory_bytes"] == 0
    assert capture_calls == []
    assert wrapper.cudagraph_class_diagnostics()["active_class"] == "c8"


def test_graph_class_activation_switches_plan_and_table_without_recapture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_with_class("c2", 2)
    c2_graph = next(iter(wrapper.concrete_cudagraph_entries.values())).cudagraph
    wrapper.begin_cudagraph_class_capture("c4")
    c4_descriptor = BatchDescriptor(4, uniform_decode=False)
    c4_graph = _FakeGraph()
    wrapper.concrete_cudagraph_entries[c4_descriptor] = CUDAGraphEntry(
        c4_descriptor,
        cudagraph=c4_graph,
    )
    wrapper.activate_cudagraph_class("c2")
    runner = object.__new__(GPUModelRunner)
    runner.model = wrapper
    runner.device = 0
    switches: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: _registry(active="c2"),
    )
    monkeypatch.setattr(
        offload_utils,
        "switch_cpu_offload_uvm_original_va_plan_class",
        lambda class_name, **kwargs: switches.append(class_name)
        or {"ok": True, "new_class": class_name},
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_plan_class_graph_certificates",
        lambda class_name: {"ok": True, "class_name": class_name},
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.torch.cuda.synchronize",
        lambda device=0: None,
    )

    result = runner.activate_uvm_expert_graph_class_at_idle_boundary(
        "c4",
        scheduler_idle=True,
    )

    assert result["ok"] is True
    assert switches == ["c4"]
    assert wrapper.cudagraph_class_diagnostics()["active_class"] == "c4"
    assert c2_graph is not None and c2_graph.reset_count == 0
    assert c4_graph.reset_count == 0


def test_certificate_failure_restores_previous_plan_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = _wrapper_with_class("c2", 2)
    wrapper.begin_cudagraph_class_capture("c4")
    descriptor = BatchDescriptor(4, uniform_decode=False)
    wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
        descriptor,
        cudagraph=_FakeGraph(),
    )
    wrapper.activate_cudagraph_class("c2")
    runner = object.__new__(GPUModelRunner)
    runner.model = wrapper
    runner.device = 0
    switches: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: _registry(active="c2"),
    )
    monkeypatch.setattr(
        offload_utils,
        "switch_cpu_offload_uvm_original_va_plan_class",
        lambda class_name, **kwargs: switches.append(class_name)
        or {"ok": True, "new_class": class_name},
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_plan_class_graph_certificates",
        lambda class_name: {
            "ok": class_name == "c2",
            "class_name": class_name,
        },
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.torch.cuda.synchronize",
        lambda device=0: None,
    )

    result = runner.activate_uvm_expert_graph_class_at_idle_boundary(
        "c4",
        scheduler_idle=True,
    )

    assert result["ok"] is False
    assert result["reason"] == "class_certificate_activation_failed"
    assert switches == ["c4", "c2"]
    assert result["rollback_plan"]["ok"] is True
    assert result["rollback_certificate"]["ok"] is True
    assert wrapper.cudagraph_class_diagnostics()["active_class"] == "c2"


def test_configured_startup_capture_visits_all_four_classes_and_restores_c8(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(GPUModelRunner)
    calls: list[tuple[str, str]] = []

    def capture(self, class_name: str, *, scheduler_idle: bool):
        assert scheduler_idle is True
        calls.append(("capture", class_name))
        return {"ok": True, "class_name": class_name, "graph_memory_bytes": 1}

    def activate(self, class_name: str, *, scheduler_idle: bool):
        assert scheduler_idle is True
        calls.append(("activate", class_name))
        return {"ok": True, "class_name": class_name}

    runner.capture_uvm_expert_graph_class_at_idle_boundary = MethodType(  # type: ignore[method-assign]
        capture,
        runner,
    )
    runner.activate_uvm_expert_graph_class_at_idle_boundary = MethodType(  # type: ignore[method-assign]
        activate,
        runner,
    )
    runner.model = SimpleNamespace(
        cudagraph_class_diagnostics=lambda: {"active_class": "c8"}
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: _registry(),
    )

    result = runner.capture_configured_uvm_expert_graph_classes_at_idle_boundary(
        scheduler_idle=True
    )

    assert result["ok"] is True
    assert result["graph_memory_bytes"] == 4
    assert calls == [
        ("capture", "c8"),
        ("capture", "c2"),
        ("capture", "c4"),
        ("capture", "c6"),
        ("activate", "c8"),
    ]
