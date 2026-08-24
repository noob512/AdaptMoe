#!/usr/bin/env python3
"""Run the phase-switch benchmark with hot-weight reads around cmd99.

This is a fail-fast diagnostic wrapper, not a performance runner.  It proves
whether an ordinary hot expert remains GPU-readable immediately before and
after the static-alias destroy boundary.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _hot_read(stage: str) -> None:
    import torch
    from vllm.model_executor.models import utils

    parameter_name = os.environ.get(
        "VLLM_UVM_DIAG_HOT_PARAMETER",
        "model.layers.0.mlp.experts.w13_weight",
    )
    expert_id = int(os.environ.get("VLLM_UVM_DIAG_HOT_EXPERT", "40"))
    binding = utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[parameter_name]
    tensor = binding.consumer[expert_id]
    print(
        "UVM_HOT_READ_BEGIN "
        f"stage={stage} parameter={parameter_name} expert={expert_id} "
        f"va=0x{int(tensor.data_ptr()):x} bytes={tensor.numel() * tensor.element_size()}",
        file=sys.stderr,
        flush=True,
    )
    # A reduction forces a graphics-engine read of every byte in the expert.
    checksum = tensor.view(torch.uint8).sum(dtype=torch.int64)
    value = int(checksum.item())
    torch.cuda.synchronize()
    print(
        f"UVM_HOT_READ_OK stage={stage} checksum={value}",
        file=sys.stderr,
        flush=True,
    )


def _hot_residency(stage: str) -> None:
    import torch
    from vllm.device_allocator import uvm
    from vllm.model_executor.models import utils

    parameter_name = os.environ.get(
        "VLLM_UVM_DIAG_HOT_PARAMETER",
        "model.layers.0.mlp.experts.w13_weight",
    )
    expert_id = int(os.environ.get("VLLM_UVM_DIAG_HOT_EXPERT", "40"))
    binding = utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[parameter_name]
    tensor = binding.consumer[expert_id]
    result = uvm.driver_batch_query_ranges_residency_bytes(
        [(int(tensor.data_ptr()), tensor.numel() * tensor.element_size())],
        device=int(torch.cuda.current_device()),
        wait_for_trackers=False,
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        "UVM_HOT_RESIDENCY "
        f"stage={stage} va=0x{int(tensor.data_ptr()):x} "
        f"gpu={result.get('gpu_bytes')} cpu={result.get('cpu_bytes')} "
        f"unknown={result.get('unknown_bytes')} free={free_bytes} "
        f"total={total_bytes} allocated={torch.cuda.memory_allocated()} "
        f"reserved={torch.cuda.memory_reserved()}",
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    from vllm.device_allocator import uvm
    from vllm.model_executor.models import utils

    original_destroy = utils.destroy_cpu_offload_graph_static_alias
    original_remote_map = uvm.driver_prepare_ranges_remote_read_audited

    def diagnosed_destroy(*args, **kwargs):
        _hot_read("before_destroy")
        result = original_destroy(*args, **kwargs)
        print(
            "UVM_CMD99_RETURN "
            f"ok={result.get('ok')} state={result.get('state')} "
            f"session_id={result.get('session_id')}",
            file=sys.stderr,
            flush=True,
        )
        _hot_read("after_destroy_before_remote_map")
        return result

    utils.destroy_cpu_offload_graph_static_alias = diagnosed_destroy

    def diagnosed_remote_map(*args, **kwargs):
        result = original_remote_map(*args, **kwargs)
        print(
            "UVM_REMOTE_MAP_RETURN "
            f"ok={result.get('ok')} requested={result.get('requested_bytes')} "
            f"mapped={result.get('remote_mapped_bytes')}",
            file=sys.stderr,
            flush=True,
        )
        _hot_read("after_remote_map")
        return result

    uvm.driver_prepare_ranges_remote_read_audited = diagnosed_remote_map

    from experiments.vllm_uvm_prefill_alias_decode_remote_graph import (
        benchmark_phase_switch_steps,
    )

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_profile_run = GPUModelRunner.profile_run
    original_capture_model = GPUModelRunner.capture_model

    def diagnosed_profile_run(self, *args, **kwargs):
        _hot_residency("before_profile")
        result = original_profile_run(self, *args, **kwargs)
        _hot_residency("after_profile_before_kv")
        return result

    def diagnosed_capture_model(self, *args, **kwargs):
        _hot_residency("after_kv_before_capture")
        return original_capture_model(self, *args, **kwargs)

    GPUModelRunner.profile_run = diagnosed_profile_run
    GPUModelRunner.capture_model = diagnosed_capture_model

    return int(benchmark_phase_switch_steps.main())


if __name__ == "__main__":
    raise SystemExit(main())
