#!/usr/bin/env python3
"""Run a real vLLM dynamic KV pool resize/prefetch check.

The test uses a tiny GPT-2 offline run and drives the dynamic target sequence
2 -> 4 -> 1 through the control file. It expects the Stage L6 pointer map to be
available and verifies that dynamic resize emits real prefetch/offload calls
when UVM-managed ranges are present.
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
VLLM_DIR = SCRIPT_DIR / "vllm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check real vLLM KV dynamic pool prefetch/offload."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--residency-jsonl")
    parser.add_argument("--real-residency-jsonl")
    parser.add_argument("--pointer-map-jsonl")
    parser.add_argument("--phase-jsonl")
    parser.add_argument("--control-file")
    parser.add_argument("--output-json")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_kv_dynamic_pool_phase2_real_{stamp}")


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


def run_real_test(
    *,
    run_dir: Path,
    trace_jsonl: Path,
    residency_jsonl: Path,
    real_residency_jsonl: Path,
    pointer_map_jsonl: Path,
    phase_jsonl: Path,
    control_file: Path,
    timeout_seconds: int,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_log = run_dir / "kv_dynamic_pool_phase2_prefetch_real_runner.log"
    control_file.parent.mkdir(parents=True, exist_ok=True)
    control_file.write_text("target_blocks=2\n", encoding="utf-8")
    for path in (
        trace_jsonl,
        residency_jsonl,
        real_residency_jsonl,
        pointer_map_jsonl,
        phase_jsonl,
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    code = r'''
import json
import os
import subprocess
import sys
import time
from pathlib import Path

vllm_dir = Path(os.environ["VLLM_SOURCE_DIR"])
if str(vllm_dir) not in sys.path:
    sys.path.insert(0, str(vllm_dir))

import torch
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.device_allocator.uvm import is_uvm_enabled
from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_real_residency_snapshot,
    get_uvm_residency_snapshot,
)

control_file = Path(os.environ["VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE"])
pointer_map_file = Path(os.environ["VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE"])
phase_jsonl = Path(os.environ["CHECK_KV_DYNAMIC_POOL_PHASE_JSONL"])


def nvidia_smi_snapshot():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {"ok": False, "raw": line}
    try:
        return {
            "ok": True,
            "index": int(parts[0]),
            "name": parts[1],
            "memory_used_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
        }
    except ValueError:
        return {"ok": False, "raw": line}


def cuda_snapshot():
    if not torch.cuda.is_available():
        return {"ok": False, "reason": "cuda_unavailable"}
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    record = {
        "ok": True,
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }
    try:
        record["allocated_bytes"] = int(torch.cuda.memory_allocated())
    except Exception as exc:
        record["allocated_bytes"] = None
        record["allocated_error"] = repr(exc)
    try:
        record["reserved_bytes"] = int(torch.cuda.memory_reserved())
    except Exception as exc:
        record["reserved_bytes"] = None
        record["reserved_error"] = repr(exc)
    return record


def wait_for_pointer_map(timeout_s=120.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if pointer_map_file.is_file():
            try:
                with pointer_map_file.open(
                    "r", encoding="utf-8", errors="replace"
                ) as handle:
                    for line in handle:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if record.get("action") == "stage_l6_kv_pointer_map":
                            return record
            except OSError:
                pass
        time.sleep(0.2)
    raise TimeoutError(f"Stage L6 KV pointer map was not written: {pointer_map_file}")


def phase_snapshot(label, target_blocks, prompt_token_count, generate_seconds, output_tokens):
    torch.cuda.synchronize()
    time.sleep(0.05)
    ledger = get_uvm_residency_snapshot()
    real = get_uvm_real_residency_snapshot(trigger=label)
    real_kv = (real.get("by_pool") or {}).get("kv") or {}
    record = {
        "phase": label,
        "target_blocks": target_blocks,
        "prompt_token_count": prompt_token_count,
        "output_tokens": output_tokens,
        "generate_seconds": generate_seconds,
        "uvm_enabled": bool(is_uvm_enabled()),
        "ledger_kv_estimated_gpu_resident_bytes": int(
            ledger.get("kv_estimated_gpu_resident_bytes") or 0
        ),
        "ledger_global_estimated_gpu_resident_bytes": int(
            ledger.get("global_estimated_gpu_resident_bytes") or 0
        ),
        "real_ok": bool(real.get("ok")),
        "real_kv_registered_bytes": int(real_kv.get("registered_bytes") or 0),
        "real_kv_queried_bytes": int(real_kv.get("queried_bytes") or 0),
        "real_kv_gpu_bytes": int(real_kv.get("gpu_bytes") or 0),
        "real_kv_cpu_bytes": int(real_kv.get("cpu_bytes") or 0),
        "real_kv_any_bytes": int(real_kv.get("any_bytes") or 0),
        "real_kv_unknown_bytes": int(real_kv.get("unknown_bytes") or 0),
        "real_kv_status": int(real_kv.get("status") or 0),
        "cuda": cuda_snapshot(),
        "nvidia_smi": nvidia_smi_snapshot(),
    }
    phase_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with phase_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the real vLLM prefetch check")

control_file.write_text("target_blocks=2\n", encoding="utf-8")

llm = LLM(
    model="gpt2",
    dtype="float16",
    enforce_eager=True,
    max_model_len=64,
    max_num_seqs=1,
    max_num_batched_tokens=64,
    gpu_memory_utilization=0.20,
    kv_cache_memory_bytes=4 * 1024 * 1024,
    block_size=16,
    enable_prefix_caching=False,
    disable_log_stats=True,
    seed=0,
)
pointer_record = wait_for_pointer_map()
phase_snapshot("after_load", None, 0, 0.0, 0)

sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)


def run_phase(label, target_blocks, prompt_token_count):
    control_file.write_text(f"target_blocks={target_blocks}\n", encoding="utf-8")
    time.sleep(0.05)
    prompt = TokensPrompt(
        prompt_token_ids=list(range(1, prompt_token_count + 1))
    )
    started = time.time()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    output_tokens = 0
    if outputs and outputs[0].outputs:
        output_tokens = len(outputs[0].outputs[0].token_ids)
    phase_snapshot(label, target_blocks, prompt_token_count, elapsed, output_tokens)


run_phase("target_2", 2, 24)
run_phase("target_4", 4, 48)
run_phase("target_1", 1, 8)

print(
    json.dumps(
        {
            "marker": "phase2_dynamic_pool_real_prefetch_ok",
            "uvm_enabled": bool(is_uvm_enabled()),
            "pointer_map_mapped_blocks": pointer_record.get("mapped_blocks"),
        },
        sort_keys=True,
    ),
    flush=True,
)
'''

    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["VLLM_SOURCE_DIR"] = str(VLLM_DIR)
    env["PYTHONPATH"] = (
        str(VLLM_DIR)
        + os.pathsep
        + str(SCRIPT_DIR)
        + (
            os.pathsep + env["PYTHONPATH"]
            if env.get("PYTHONPATH")
            else ""
        )
    )
    env.update(
        {
            "CHECK_KV_DYNAMIC_POOL_PHASE_JSONL": str(phase_jsonl),
            "VLLM_USE_UVM": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
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
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l6",
            "VLLM_UVM_RESIDENCY_TRACE_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(residency_jsonl),
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE": "prefetch_cpu",
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE": "prefetch",
            "VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE": str(pointer_map_jsonl),
            "VLLM_UVM_REAL_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_REAL_RESIDENCY_TRACE_FILE": str(real_residency_jsonl),
            "VLLM_UVM_REAL_RESIDENCY_DEVICE": "0",
            "VLLM_UVM_REAL_RESIDENCY_MAX_REGIONS_PER_POOL": "0",
            "VLLM_UVM_REAL_RESIDENCY_MAX_SNAPSHOTS": "64",
        }
    )

    cmd = [
        "timeout",
        str(max(int(timeout_seconds or 0), 60)),
        "uv",
        "run",
        "--directory",
        str(SCRIPT_DIR),
        "python",
        "-c",
        code,
    ]
    print("===========================================================", flush=True)
    print(" KV Dynamic Pool Phase 2 Check: real vLLM prefetch", flush=True)
    print(f" Output dir: {run_dir}", flush=True)
    print(f" Runtime trace JSONL: {trace_jsonl}", flush=True)
    print(f" Residency trace JSONL: {residency_jsonl}", flush=True)
    print(f" Real residency JSONL: {real_residency_jsonl}", flush=True)
    print(f" Pointer map JSONL: {pointer_map_jsonl}", flush=True)
    print(f" Phase metrics JSONL: {phase_jsonl}", flush=True)
    print(f" Control file: {control_file}", flush=True)
    print("===========================================================", flush=True)
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
            f"real vLLM KV dynamic pool prefetch check failed with exit code {rc}; "
            f"log={runner_log}"
        )
    return runner_log


def phase_by_name(phase_records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next((record for record in phase_records if record.get("phase") == name), {})


def summarize(
    *,
    trace_jsonl: Path,
    residency_jsonl: Path,
    real_residency_jsonl: Path,
    pointer_map_jsonl: Path,
    phase_jsonl: Path,
    runner_log: Path,
) -> dict[str, Any]:
    runtime_records = read_jsonl(trace_jsonl)
    residency_records = read_jsonl(residency_jsonl)
    real_residency_records = read_jsonl(real_residency_jsonl)
    pointer_map_records = read_jsonl(pointer_map_jsonl)
    phase_records = read_jsonl(phase_jsonl)
    runner_text = runner_log.read_text(encoding="utf-8", errors="replace")

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
        and (record.get("metadata") or {}).get("source") == "kv_dynamic_pool_resize"
    ]
    prefetch_records = [
        record
        for record in residency_records
        if record.get("action") == "residency_kv_prefetch_back_issued"
        and (record.get("metadata") or {}).get("source") == "kv_dynamic_pool_resize"
    ]
    physical_offloads = [
        record
        for record in offload_records
        if bool((record.get("metadata") or {}).get("physical_migration_issued"))
    ]
    physical_prefetches = [
        record
        for record in prefetch_records
        if bool((record.get("metadata") or {}).get("physical_migration_issued"))
    ]
    target2 = phase_by_name(phase_records, "target_2")
    target4 = phase_by_name(phase_records, "target_4")
    target1 = phase_by_name(phase_records, "target_1")
    kv_est_target2 = int(
        target2.get("ledger_kv_estimated_gpu_resident_bytes") or 0
    )
    kv_est_target4 = int(
        target4.get("ledger_kv_estimated_gpu_resident_bytes") or 0
    )
    kv_est_target1 = int(
        target1.get("ledger_kv_estimated_gpu_resident_bytes") or 0
    )
    target4_selection = next(
        (
            record
            for record in select_records
            if record.get("dynamic_pool_target_blocks") == 4
            and (record.get("selected_block_ids") or [])[:2] == [3, 4]
        ),
        {},
    )
    pointer_map = next(
        (
            record
            for record in pointer_map_records
            if record.get("action") == "stage_l6_kv_pointer_map"
        ),
        {},
    )

    real_phase_observation = {
        "target_2_gpu_bytes": int(target2.get("real_kv_gpu_bytes") or 0),
        "target_4_gpu_bytes": int(target4.get("real_kv_gpu_bytes") or 0),
        "target_1_gpu_bytes": int(target1.get("real_kv_gpu_bytes") or 0),
        "target_2_cpu_bytes": int(target2.get("real_kv_cpu_bytes") or 0),
        "target_4_cpu_bytes": int(target4.get("real_kv_cpu_bytes") or 0),
        "target_1_cpu_bytes": int(target1.get("real_kv_cpu_bytes") or 0),
        "all_real_ok": all(
            bool(record.get("real_ok"))
            for record in (target2, target4, target1)
            if record
        ),
    }
    real_gpu_decreased_on_shrink = (
        real_phase_observation["target_1_gpu_bytes"]
        < real_phase_observation["target_4_gpu_bytes"]
        if real_phase_observation["all_real_ok"]
        else None
    )
    real_gpu_increased_on_grow = (
        real_phase_observation["target_4_gpu_bytes"]
        > real_phase_observation["target_2_gpu_bytes"]
        if real_phase_observation["all_real_ok"]
        else None
    )

    checks = [
        check(
            "runner_completed",
            "phase2_dynamic_pool_real_prefetch_ok" in runner_text,
            str(runner_log),
        ),
        check(
            "pointer_map_present",
            int(pointer_map.get("mapped_blocks") or 0) >= 4,
            f"mapped_blocks={pointer_map.get('mapped_blocks')}",
        ),
        check(
            "resize_sequence_present",
            [
                (record.get("direction"), record.get("new_target_blocks"))
                for record in resize_records
            ][:3]
            == [("shrink", 2), ("grow", 4), ("shrink", 1)],
            f"resize_records={resize_records}",
        ),
        check(
            "target_updates_present",
            {record.get("new_target_blocks") for record in target_updates} >= {1, 4},
            f"target_updates={target_updates}",
        ),
        check(
            "physical_offload_trace_present",
            bool(physical_offloads),
            f"physical_offloads={len(physical_offloads)}",
        ),
        check(
            "physical_prefetch_trace_present",
            bool(physical_prefetches),
            f"physical_prefetches={len(physical_prefetches)}",
        ),
        check(
            "physical_ranges_present",
            any(int(record.get("range_count") or 0) > 0 for record in physical_offloads)
            and any(int(record.get("range_count") or 0) > 0 for record in physical_prefetches),
            (
                f"offload_ranges={[record.get('range_count') for record in physical_offloads[:4]]} "
                f"prefetch_ranges={[record.get('range_count') for record in physical_prefetches[:4]]}"
            ),
        ),
        check(
            "target4_uses_prefetched_blocks",
            bool(target4_selection),
            f"target4_selection={target4_selection.get('selected_block_ids')}",
        ),
        check(
            "ledger_resident_window_changes",
            kv_est_target2 > 0 and kv_est_target4 > kv_est_target2 and kv_est_target1 < kv_est_target4,
            (
                f"kv_est_target2={kv_est_target2} "
                f"kv_est_target4={kv_est_target4} "
                f"kv_est_target1={kv_est_target1}"
            ),
        ),
    ]
    passed = all(item["passed"] for item in checks)
    return {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "residency_jsonl": str(residency_jsonl),
        "real_residency_jsonl": str(real_residency_jsonl),
        "pointer_map_jsonl": str(pointer_map_jsonl),
        "phase_jsonl": str(phase_jsonl),
        "runner_log": str(runner_log),
        "checks": checks,
        "resize_records": resize_records,
        "target_updates": target_updates,
        "selected_records": [
            {
                "target": record.get("dynamic_pool_target_blocks"),
                "selected_block_ids": record.get("selected_block_ids"),
                "lifecycle": record.get("selected_lifecycle_states"),
            }
            for record in select_records
        ],
        "physical_offload_records": len(physical_offloads),
        "physical_prefetch_records": len(physical_prefetches),
        "ledger_estimated_kv_gpu_bytes": {
            "target_2": kv_est_target2,
            "target_4": kv_est_target4,
            "target_1": kv_est_target1,
        },
        "real_residency_observation": {
            **real_phase_observation,
            "gpu_increased_on_grow": real_gpu_increased_on_grow,
            "gpu_decreased_on_shrink": real_gpu_decreased_on_shrink,
            "real_snapshot_records": len(real_residency_records),
        },
        "phase_records": phase_records,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    trace_jsonl = (
        Path(args.trace_jsonl)
        if args.trace_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2_real.jsonl"
    )
    residency_jsonl = (
        Path(args.residency_jsonl)
        if args.residency_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2_real_residency.jsonl"
    )
    real_residency_jsonl = (
        Path(args.real_residency_jsonl)
        if args.real_residency_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2_real_pages.jsonl"
    )
    pointer_map_jsonl = (
        Path(args.pointer_map_jsonl)
        if args.pointer_map_jsonl
        else run_dir / "vllm_uvm_kv_pointer_map.jsonl"
    )
    phase_jsonl = (
        Path(args.phase_jsonl)
        if args.phase_jsonl
        else run_dir / "vllm_uvm_kv_dynamic_pool_phase2_real_phases.jsonl"
    )
    control_file = (
        Path(args.control_file)
        if args.control_file
        else run_dir / "kv_dynamic_pool.control"
    )
    runner_log = run_real_test(
        run_dir=run_dir,
        trace_jsonl=trace_jsonl,
        residency_jsonl=residency_jsonl,
        real_residency_jsonl=real_residency_jsonl,
        pointer_map_jsonl=pointer_map_jsonl,
        phase_jsonl=phase_jsonl,
        control_file=control_file,
        timeout_seconds=args.timeout_seconds,
    )
    summary = summarize(
        trace_jsonl=trace_jsonl,
        residency_jsonl=residency_jsonl,
        real_residency_jsonl=real_residency_jsonl,
        pointer_map_jsonl=pointer_map_jsonl,
        phase_jsonl=phase_jsonl,
        runner_log=runner_log,
    )
    output_json = (
        Path(args.output_json)
        if args.output_json
        else run_dir / "kv_dynamic_pool_phase2_real_summary.json"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
