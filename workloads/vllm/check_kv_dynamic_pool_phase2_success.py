#!/usr/bin/env python3
"""Validate runtime-adjustable KV dynamic pool sizing.

This is an in-process block-manager check. It does not start a vLLM server or
require a GPU. The test enables the Stage J dynamic KV pool hook, writes a
control file at runtime, and verifies that allocation only uses blocks inside
the current logical pool target.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check runtime-adjustable KV dynamic pool sizing."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--residency-jsonl")
    parser.add_argument("--control-file")
    parser.add_argument("--output-json")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_kv_dynamic_pool_phase2_{stamp}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_self_test(
    run_dir: Path,
    trace_jsonl: Path,
    residency_jsonl: Path,
    control_file: Path,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_log = run_dir / "kv_dynamic_pool_phase2_runner.log"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text("target_blocks=2\n", encoding="utf-8")
    for path in (trace_jsonl, residency_jsonl):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    code = r'''
import os
import time
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

control_file = os.environ["VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE"]

config = KVCacheConfig(
    num_blocks=7,
    kv_cache_tensors=[KVCacheTensor(size=7 * 1024, shared_by=["layer.0"])],
    kv_cache_groups=[
        KVCacheGroupSpec(
            layer_names=["layer.0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        )
    ],
)

manager = KVCacheManager(
    config,
    max_model_len=128,
    hash_block_size=16,
    enable_caching=False,
)
pool = manager.block_pool

assert pool.null_block.block_id == 0
assert pool.get_num_free_blocks() == 2, pool.get_num_free_blocks()
assert round(pool.get_usage(), 6) == 0.0, pool.get_usage()

first = pool.get_new_blocks(1)[0]
assert first.block_id == 1, first.block_id
pool.free_blocks([first])

second = pool.get_new_blocks(1)[0]
assert second.block_id == 2, second.block_id
pool.free_blocks([second])

third = pool.get_new_blocks(1)[0]
assert third.block_id == 1, third.block_id
pool.free_blocks([third])

with open(control_file, "w", encoding="utf-8") as handle:
    handle.write("target_blocks=4\n")
time.sleep(0.01)
assert pool.get_num_free_blocks() == 4, pool.get_num_free_blocks()

grown = pool.get_new_blocks(2)
assert [block.block_id for block in grown] == [3, 4], [b.block_id for b in grown]
pool.free_blocks(grown)

with open(control_file, "w", encoding="utf-8") as handle:
    handle.write("target_blocks=1\n")
time.sleep(0.01)
assert pool.get_num_free_blocks() == 1, pool.get_num_free_blocks()

shrunk = pool.get_new_blocks(1)[0]
assert shrunk.block_id == 1, shrunk.block_id
pool.free_blocks([shrunk])

try:
    pool.get_new_blocks(2)
except ValueError:
    pass
else:
    raise AssertionError("dynamic pool allowed allocation beyond target")

print("phase2_dynamic_pool_resize_ok")
'''

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_KV_RUNTIME_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_MODE": "trace_only",
            "VLLM_UVM_KV_RUNTIME_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT": "64",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS": "1",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_ALLOW_CACHED": "1",
            "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE": "1",
            "VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE": str(control_file),
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS": "2",
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l5",
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(residency_jsonl),
        }
    )

    cmd = [
        "uv",
        "run",
        "--directory",
        str(SCRIPT_DIR),
        "python",
        "-c",
        code,
    ]
    print("===========================================================")
    print(" KV Dynamic Pool Phase 2 Check: manual runtime resize")
    print(f" Output dir: {run_dir}")
    print(f" Runtime trace JSONL: {trace_jsonl}")
    print(f" Residency trace JSONL: {residency_jsonl}")
    print(f" Control file: {control_file}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        rc = process.wait()
    if rc != 0:
        raise SystemExit(
            f"KV dynamic pool Phase 2 self-test failed with exit code {rc}; "
            f"log={runner_log}"
        )
    return runner_log


def summarize(
    trace_jsonl: Path,
    residency_jsonl: Path,
    runner_log: Path,
) -> dict[str, Any]:
    runtime_records = read_jsonl(trace_jsonl)
    residency_records = read_jsonl(residency_jsonl)
    select_records = [
        record
        for record in runtime_records
        if record.get("action") == "gpu_resident_free_block_select"
    ]
    resize_records = [
        record
        for record in runtime_records
        if record.get("action") == "kv_dynamic_pool_resize"
    ]
    target_updates = [
        record
        for record in runtime_records
        if record.get("action") == "kv_dynamic_pool_target_update"
    ]
    offload_records = [
        record
        for record in residency_records
        if record.get("action") == "residency_kv_data_offload_issued"
    ]
    prefetch_records = [
        record
        for record in residency_records
        if record.get("action") == "residency_kv_prefetch_back_issued"
    ]
    selected_ids = [
        block_id
        for record in select_records
        for block_id in (record.get("selected_block_ids") or [])
    ]
    target_by_select = [
        (
            record.get("dynamic_pool_target_blocks"),
            record.get("selected_block_ids"),
        )
        for record in select_records
    ]
    runner_text = runner_log.read_text(encoding="utf-8", errors="replace")
    checks = [
        check(
            "runner_completed",
            "phase2_dynamic_pool_resize_ok" in runner_text,
            str(runner_log),
        ),
        check(
            "selection_trace_present",
            len(select_records) >= 5,
            f"records={len(select_records)}",
        ),
        check(
            "initial_target_respected",
            [1, 2, 1] == selected_ids[:3],
            f"first_three_selected={selected_ids[:3]}",
        ),
        check(
            "grow_target_exposes_new_blocks",
            [3, 4] in [record.get("selected_block_ids") for record in select_records],
            f"target_by_select={target_by_select}",
        ),
        check(
            "shrink_target_respected",
            any(
                record.get("dynamic_pool_target_blocks") == 1
                and record.get("selected_block_ids") == [1]
                for record in select_records
            ),
            f"target_by_select={target_by_select}",
        ),
        check(
            "resize_trace_present",
            any(record.get("direction") == "shrink" for record in resize_records)
            and any(record.get("direction") == "grow" for record in resize_records),
            f"resize_records={resize_records}",
        ),
        check(
            "target_update_trace_present",
            {record.get("new_target_blocks") for record in target_updates} >= {1, 4},
            f"target_updates={target_updates}",
        ),
        check(
            "logical_offload_trace_present",
            bool(offload_records),
            f"offload_records={len(offload_records)}",
        ),
        check(
            "logical_prefetch_trace_present",
            bool(prefetch_records),
            f"prefetch_records={len(prefetch_records)}",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "residency_jsonl": str(residency_jsonl),
        "runner_log": str(runner_log),
        "checks": checks,
        "selected_ids": selected_ids,
        "target_by_select": target_by_select,
        "resize_records": resize_records,
        "target_updates": target_updates,
        "offload_records": len(offload_records),
        "prefetch_records": len(prefetch_records),
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    trace_jsonl = (
        Path(args.trace_jsonl)
        if args.trace_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2.jsonl"
    )
    residency_jsonl = (
        Path(args.residency_jsonl)
        if args.residency_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2_residency.jsonl"
    )
    control_file = (
        Path(args.control_file)
        if args.control_file
        else run_dir / "kv_dynamic_pool.control"
    )
    runner_log = run_self_test(run_dir, trace_jsonl, residency_jsonl, control_file)
    summary = summarize(trace_jsonl, residency_jsonl, runner_log)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else run_dir / "kv_dynamic_pool_phase2_summary.json"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
