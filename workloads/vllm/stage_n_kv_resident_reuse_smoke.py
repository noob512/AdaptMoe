#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for GPU-resident KV free-block reuse.

This is intentionally a vLLM block-pool level test. It verifies that, when the
UVM runtime shadow state says some free KV blocks are already GPU resident,
BlockPool.get_new_blocks() reuses those blocks before CPU-resident free blocks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace


SOURCE_ROOT = Path(__file__).resolve().parent / "vllm" / "vllm"


class _SmokeLogger:
    def warning(self, *args, **kwargs) -> None:
        pass

    def info(self, *args, **kwargs) -> None:
        pass

    def debug(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass


def _module(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _load_source_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_lightweight_vllm_imports() -> None:
    """Install enough vLLM stubs to load the real allocator modules.

    The default system Python on the test host does not include torch. Loading
    these source files directly lets the smoke still exercise the real
    BlockPool and UvmKvRuntimePolicy code paths changed by this experiment.
    """
    _module("vllm", package=True)
    _module("vllm.device_allocator", package=True)
    _module("vllm.distributed", package=True)
    _module("vllm.utils", package=True)
    _module("vllm.v1", package=True)
    _module("vllm.v1.core", package=True)

    logger_mod = _module("vllm.logger")
    logger_mod.init_logger = lambda _name: _SmokeLogger()

    envs_mod = _module("vllm.envs")
    envs_mod.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES = False

    config_mod = _module("vllm.config")
    config_mod.VllmConfig = object

    hashing_mod = _module("vllm.utils.hashing")
    hashing_mod.sha256_cbor = lambda payload: b"0" * 32

    math_mod = _module("vllm.utils.math_utils")
    math_mod.cdiv = lambda a, b: -(-a // b)

    mem_mod = _module("vllm.utils.mem_constants")
    mem_mod.GiB_bytes = 1024**3

    request_mod = _module("vllm.v1.request")
    request_mod.Request = object

    v1_utils_mod = _module("vllm.v1.utils")
    v1_utils_mod.tensor_data = lambda *args, **kwargs: None

    kv_iface_mod = _module("vllm.v1.kv_cache_interface")
    for name in (
        "ChunkedLocalAttentionSpec",
        "FullAttentionSpec",
        "KVCacheConfig",
        "KVCacheGroupSpec",
        "KVCacheSpec",
        "KVCacheTensor",
        "SlidingWindowSpec",
        "UniformTypeKVCacheSpecs",
    ):
        setattr(kv_iface_mod, name, type(name, (), {}))

    kv_events_mod = _module("vllm.distributed.kv_events")
    kv_events_mod.MEDIUM_GPU = "gpu"
    for name in (
        "AllBlocksCleared",
        "BlockRemoved",
        "BlockStored",
        "KVCacheEvent",
    ):
        setattr(kv_events_mod, name, type(name, (), {
            "__init__": lambda self, *args, **kwargs: None
        }))

    coordinator_mod = _module("vllm.device_allocator.uvm_pool_coordinator")
    coordinator_mod.record_uvm_pool_pressure = lambda **kwargs: None
    coordinator_mod.request_uvm_pool_action = lambda **kwargs: SimpleNamespace(
        allowed=True,
        mode="smoke",
        reason="allowed",
        scope_key=kwargs.get("scope_key"),
    )

    policy_runtime_mod = _module("vllm.device_allocator.uvm_policy_runtime")
    policy_runtime_mod.get_uvm_policy_runtime = lambda: SimpleNamespace(active=False)

    uvm_mod = _module("vllm.device_allocator.uvm")
    uvm_mod.prefetch_range_to_cpu = lambda *args, **kwargs: True
    uvm_mod.prefetch_range_to_device = lambda *args, **kwargs: True

    _load_source_module(
        "vllm.device_allocator.uvm_pool_residency",
        SOURCE_ROOT / "device_allocator" / "uvm_pool_residency.py",
    )
    _load_source_module(
        "vllm.v1.core.uvm_kv_runtime_policy",
        SOURCE_ROOT / "v1" / "core" / "uvm_kv_runtime_policy.py",
    )
    _load_source_module(
        "vllm.v1.core.kv_cache_utils",
        SOURCE_ROOT / "v1" / "core" / "kv_cache_utils.py",
    )
    _load_source_module(
        "vllm.v1.core.block_pool",
        SOURCE_ROOT / "v1" / "core" / "block_pool.py",
    )


def configure_env(run_root: Path) -> None:
    os.environ.setdefault("VLLM_UVM_KV_RUNTIME_ENABLE", "1")
    os.environ.setdefault("VLLM_UVM_KV_RUNTIME_TRACE_FILE",
                          str(run_root / "kv_runtime_trace.jsonl"))
    os.environ.setdefault("VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS", "1")
    os.environ.setdefault("VLLM_UVM_RESIDENCY_ENABLE", "1")
    os.environ.setdefault("VLLM_UVM_RESIDENCY_TRACE_ENABLE", "1")
    os.environ.setdefault("VLLM_UVM_RESIDENCY_TRACE_FILE",
                          str(run_root / "pool_residency_trace.jsonl"))
    os.environ.setdefault("VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE", "1")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--num-blocks", type=int, default=7)
    parser.add_argument("--alloc-blocks", type=int, default=2)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    configure_env(run_root)
    install_lightweight_vllm_imports()

    from vllm.device_allocator.uvm_pool_residency import (  # noqa: PLC0415
        get_uvm_residency_region_snapshot,
        mark_uvm_residency_state,
    )
    from vllm.v1.core.block_pool import BlockPool  # noqa: PLC0415
    from vllm.v1.core.uvm_kv_runtime_policy import (  # noqa: PLC0415
        UvmKvRuntimePolicy,
    )

    kv_cache_config = SimpleNamespace(
        num_blocks=args.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[object()],
    )
    policy = UvmKvRuntimePolicy(kv_cache_config)
    block_pool = BlockPool(
        num_gpu_blocks=args.num_blocks,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,
    )

    # Initial free queue is [1, 2, 3, 4, 5, 6]. Mark the head blocks as CPU and
    # later blocks as GPU. Without the residency-aware path, allocation would
    # take [1, 2]; with it, allocation should take [3, 4].
    target_states = {
        1: "cpu",
        2: "cpu",
        3: "gpu",
        4: "gpu",
        5: "cpu",
        6: "gpu",
    }
    for block_id, state in target_states.items():
        if block_id >= args.num_blocks:
            continue
        mark_uvm_residency_state(region_id=f"kv:0:{block_id}",
                                 state=state,
                                 metadata={"source": "resident_reuse_smoke"})

    free_before = [
        block.block_id for block in block_pool.free_block_queue.get_all_free_blocks()
    ]
    selected = block_pool.get_new_blocks(args.alloc_blocks)
    selected_ids = [block.block_id for block in selected]
    free_after = [
        block.block_id for block in block_pool.free_block_queue.get_all_free_blocks()
    ]
    selected_states = {
        str(block_id): (get_uvm_residency_region_snapshot(f"kv:0:{block_id}") or {}
                       ).get("state")
        for block_id in selected_ids
    }

    expected = [3, 4][:args.alloc_blocks]
    ok = selected_ids == expected
    summary = {
        "ok": ok,
        "expected_selected": expected,
        "selected_ids": selected_ids,
        "selected_states": selected_states,
        "free_before": free_before,
        "free_after": free_after,
        "prefer_gpu_resident_free_blocks": (
            os.environ.get("VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS")
        ),
        "trace_file": os.environ.get("VLLM_UVM_KV_RUNTIME_TRACE_FILE"),
        "residency_trace_file": os.environ.get("VLLM_UVM_RESIDENCY_TRACE_FILE"),
    }
    write_json(run_root / "stage_n_kv_resident_reuse_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
