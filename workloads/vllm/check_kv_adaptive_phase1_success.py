#!/usr/bin/env python3
"""Validate Phase 1 KV adaptive fresh-before-warm block selection.

This is an in-process block-manager check. It does not start a vLLM server or
require a GPU. The test enables the existing
VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS policy hook, constructs a small
KVCacheManager, creates one warm free block, and verifies that allocation still
chooses never-used blocks before reusing the warm block.
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
        description="Check Phase 1 KV adaptive fresh-before-warm block selection."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--output-json")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_kv_adaptive_phase1_{stamp}")


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


def run_self_test(run_dir: Path, trace_jsonl: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_log = run_dir / "kv_adaptive_phase1_runner.log"
    try:
        trace_jsonl.unlink()
    except FileNotFoundError:
        pass

    code = r'''
import os
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

config = KVCacheConfig(
    num_blocks=6,
    kv_cache_tensors=[KVCacheTensor(size=6 * 1024, shared_by=["layer.0"])],
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

# Create one warm free block. Block 1 has been used and then freed.
warm = pool.get_new_blocks(1)[0]
assert warm.block_id == 1
assert warm.lifecycle_state == "used_free"
pool.free_blocks([warm])
assert warm.ref_cnt == 0
assert warm.lifecycle_state == "used_free"

# Phase 1 expectation: choose a never-used fresh block before the warm block.
fresh = pool.get_new_blocks(1)[0]
assert fresh.block_id == 2, f"expected fresh block 2 before warm block 1, got {fresh.block_id}"
assert fresh.lifecycle_state == "used_free"

# Use the remaining fresh blocks. Only after fresh blocks are gone should the
# policy fall back to the warm block.
more = pool.get_new_blocks(3)
assert [block.block_id for block in more] == [3, 4, 5]

# With all fresh blocks consumed, the original warm block is now the only free
# candidate. This verifies warm reuse still works.
again = pool.get_new_blocks(1)[0]
assert again.block_id == 1, f"expected warm block 1 after fresh exhaustion, got {again.block_id}"

print("phase1_fresh_before_warm_ok")
'''

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_KV_RUNTIME_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_MODE": "trace_only",
            "VLLM_UVM_KV_RUNTIME_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS": "1",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_ALLOW_CACHED": "1",
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l5",
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
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
    print(" KV Adaptive Phase 1 Check: fresh-before-warm self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
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
            f"KV adaptive Phase 1 self-test failed with exit code {rc}; "
            f"log={runner_log}"
        )
    return runner_log


def summarize(trace_jsonl: Path, runner_log: Path) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    select_records = [
        record
        for record in records
        if record.get("action") == "gpu_resident_free_block_select"
    ]
    target_select = next(
        (
            record
            for record in select_records
            if record.get("selected_block_ids") == [2]
        ),
        {},
    )
    checks = [
        check(
            "runner_completed",
            "phase1_fresh_before_warm_ok" in runner_log.read_text(
                encoding="utf-8", errors="replace"
            ),
            str(runner_log),
        ),
        check(
            "selection_trace_present",
            bool(select_records),
            f"records={len(select_records)}",
        ),
        check(
            "fresh_selected_before_warm",
            bool(target_select),
            f"target_selected={target_select.get('selected_block_ids')}",
        ),
        check(
            "fresh_candidate_count_present",
            int(target_select.get("gpu_fresh_candidates") or 0) > 0,
            f"gpu_fresh_candidates={target_select.get('gpu_fresh_candidates')}",
        ),
        check(
            "warm_candidate_count_present",
            int(target_select.get("gpu_warm_candidates") or 0) > 0,
            f"gpu_warm_candidates={target_select.get('gpu_warm_candidates')}",
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "runner_log": str(runner_log),
        "checks": checks,
        "target_select": target_select,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    trace_jsonl = (
        Path(args.trace_jsonl)
        if args.trace_jsonl
        else run_dir / "vllm_uvm_kv_adaptive_phase1.jsonl"
    )
    runner_log = run_self_test(run_dir, trace_jsonl)
    summary = summarize(trace_jsonl, runner_log)
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "kv_adaptive_phase1_summary.json"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
