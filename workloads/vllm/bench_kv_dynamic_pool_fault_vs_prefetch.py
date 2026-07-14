#!/usr/bin/env python3
"""Compare GPU-resident KV reuse, UVM fault migration, and explicit prefetch.

The benchmark runs three real vLLM/GPT-2 cases:

1. gpu_resident: keep the target window large, so the request uses GPU-resident
   KV blocks.
2. fault_on_touch: shrink KV blocks to CPU with UVM prefetch-to-CPU, grow the
   logical window back without prefetching to GPU, then let the model touch
   those CPU-resident KV pages.
3. explicit_prefetch: same shrink/grow sequence, but grow issues
   prefetch_range_to_device before the request touches the blocks.

The goal is to answer whether UVM-managed KV pages fault back automatically
instead of erroring, and to quantify the latency of fault-on-touch versus
explicit prefetch and already-GPU-resident reuse.
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
        description="Benchmark KV UVM fault-on-touch versus explicit prefetch."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output-json")
    parser.add_argument("--small-target", type=int, default=2)
    parser.add_argument("--large-target", type=int, default=6)
    parser.add_argument("--small-prompt-tokens", type=int, default=24)
    parser.add_argument("--large-prompt-tokens", type=int, default=80)
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_kv_fault_vs_prefetch_{stamp}")


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


def case_paths(run_dir: Path, case_name: str) -> dict[str, Path]:
    case_dir = run_dir / case_name
    return {
        "case_dir": case_dir,
        "control": case_dir / "kv_dynamic_pool.control",
        "runtime": case_dir / "runtime.jsonl",
        "residency": case_dir / "residency.jsonl",
        "real": case_dir / "real_residency.jsonl",
        "pointer": case_dir / "pointer_map.jsonl",
        "phase": case_dir / "phases.jsonl",
        "runner_log": case_dir / "runner.log",
    }


def run_case(
    *,
    case_name: str,
    run_dir: Path,
    timeout_seconds: int,
    small_target: int,
    large_target: int,
    small_prompt_tokens: int,
    large_prompt_tokens: int,
) -> dict[str, Path]:
    paths = case_paths(run_dir, case_name)
    case_dir = paths["case_dir"]
    case_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        if key == "case_dir":
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    initial_target = large_target if case_name == "gpu_resident" else small_target
    paths["control"].write_text(
        f"target_blocks={initial_target}\n",
        encoding="utf-8",
    )

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

case_name = os.environ["KV_FAULT_CASE"]
control_file = Path(os.environ["VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE"])
pointer_map_file = Path(os.environ["VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE"])
phase_jsonl = Path(os.environ["KV_FAULT_PHASE_JSONL"])
small_target = int(os.environ["KV_FAULT_SMALL_TARGET"])
large_target = int(os.environ["KV_FAULT_LARGE_TARGET"])
small_prompt_tokens = int(os.environ["KV_FAULT_SMALL_PROMPT_TOKENS"])
large_prompt_tokens = int(os.environ["KV_FAULT_LARGE_PROMPT_TOKENS"])


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
            with pointer_map_file.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("action") == "stage_l6_kv_pointer_map":
                        return record
        time.sleep(0.2)
    raise TimeoutError(f"Stage L6 KV pointer map was not written: {pointer_map_file}")


def phase_snapshot(label, target_blocks, prompt_tokens, generate_seconds, output_tokens):
    torch.cuda.synchronize()
    time.sleep(0.05)
    ledger = get_uvm_residency_snapshot()
    real = get_uvm_real_residency_snapshot(trigger=f"{case_name}:{label}")
    real_kv = (real.get("by_pool") or {}).get("kv") or {}
    record = {
        "case": case_name,
        "phase": label,
        "target_blocks": target_blocks,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "generate_seconds": generate_seconds,
        "uvm_enabled": bool(is_uvm_enabled()),
        "ledger_kv_gpu_bytes": int(ledger.get("kv_estimated_gpu_resident_bytes") or 0),
        "real_ok": bool(real.get("ok")),
        "real_kv_registered_bytes": int(real_kv.get("registered_bytes") or 0),
        "real_kv_queried_bytes": int(real_kv.get("queried_bytes") or 0),
        "real_kv_gpu_bytes": int(real_kv.get("gpu_bytes") or 0),
        "real_kv_cpu_bytes": int(real_kv.get("cpu_bytes") or 0),
        "real_kv_unknown_bytes": int(real_kv.get("unknown_bytes") or 0),
        "cuda": cuda_snapshot(),
        "nvidia_smi": nvidia_smi_snapshot(),
    }
    phase_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with phase_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def run_phase(label, target_blocks, prompt_tokens, sampling):
    control_file.write_text(f"target_blocks={target_blocks}\n", encoding="utf-8")
    time.sleep(0.05)
    prompt = TokensPrompt(prompt_token_ids=list(range(1, prompt_tokens + 1)))
    started = time.time()
    outputs = llm.generate([prompt], sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    output_tokens = 0
    if outputs and outputs[0].outputs:
        output_tokens = len(outputs[0].outputs[0].token_ids)
    phase_snapshot(label, target_blocks, prompt_tokens, elapsed, output_tokens)


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")

llm = LLM(
    model="gpt2",
    dtype="float16",
    enforce_eager=True,
    max_model_len=96,
    max_num_seqs=1,
    max_num_batched_tokens=96,
    gpu_memory_utilization=0.20,
    kv_cache_memory_bytes=4 * 1024 * 1024,
    block_size=16,
    enable_prefix_caching=False,
    disable_log_stats=True,
    seed=0,
)
pointer_record = wait_for_pointer_map()
sampling = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
phase_snapshot("after_load", None, 0, 0.0, 0)

if case_name == "gpu_resident":
    run_phase("large", large_target, large_prompt_tokens, sampling)
    run_phase("large_repeat", large_target, large_prompt_tokens, sampling)
else:
    run_phase("small_after_shrink", small_target, small_prompt_tokens, sampling)
    run_phase("large_after_grow", large_target, large_prompt_tokens, sampling)
    run_phase("large_repeat", large_target, large_prompt_tokens, sampling)

print(
    json.dumps(
        {
            "marker": "kv_fault_vs_prefetch_case_ok",
            "case": case_name,
            "uvm_enabled": bool(is_uvm_enabled()),
            "pointer_map_mapped_blocks": pointer_record.get("mapped_blocks"),
        },
        sort_keys=True,
    ),
    flush=True,
)
'''

    prefetch_back_enable = "1" if case_name == "explicit_prefetch" else "0"
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env["VLLM_SOURCE_DIR"] = str(VLLM_DIR)
    env["PYTHONPATH"] = (
        str(VLLM_DIR)
        + os.pathsep
        + str(SCRIPT_DIR)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    env.update(
        {
            "KV_FAULT_CASE": case_name,
            "KV_FAULT_PHASE_JSONL": str(paths["phase"]),
            "KV_FAULT_SMALL_TARGET": str(small_target),
            "KV_FAULT_LARGE_TARGET": str(large_target),
            "KV_FAULT_SMALL_PROMPT_TOKENS": str(small_prompt_tokens),
            "KV_FAULT_LARGE_PROMPT_TOKENS": str(large_prompt_tokens),
            "VLLM_USE_UVM": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "VLLM_UVM_KV_RUNTIME_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_MODE": "trace_only",
            "VLLM_UVM_KV_RUNTIME_TRACE_FILE": str(paths["runtime"]),
            "VLLM_UVM_KV_RUNTIME_CANDIDATE_LIMIT": "64",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS": "1",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_ALLOW_CACHED": "1",
            "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE": "1",
            "VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE": str(paths["control"]),
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS": str(initial_target),
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l6",
            "VLLM_UVM_RESIDENCY_TRACE_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(paths["residency"]),
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE": prefetch_back_enable,
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE": "prefetch_cpu",
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE": "prefetch",
            "VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE": str(paths["pointer"]),
            "VLLM_UVM_REAL_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_REAL_RESIDENCY_TRACE_FILE": str(paths["real"]),
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
    print(f"=== running case={case_name} dir={case_dir} ===", flush=True)
    with paths["runner_log"].open("w", encoding="utf-8") as handle:
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
            f"case {case_name} failed with exit code {rc}; "
            f"log={paths['runner_log']}"
        )
    return paths


def phase_by_name(records: list[dict[str, Any]], phase: str) -> dict[str, Any]:
    return next((record for record in records if record.get("phase") == phase), {})


def physical_record_count(
    records: list[dict[str, Any]],
    *,
    action: str,
    physical: bool,
) -> int:
    return sum(
        1
        for record in records
        if record.get("action") == action
        and bool((record.get("metadata") or {}).get("physical_migration_issued"))
        is physical
    )


def summarize_case(case_name: str, paths: dict[str, Path]) -> dict[str, Any]:
    phase_records = read_jsonl(paths["phase"])
    residency_records = read_jsonl(paths["residency"])
    runtime_records = read_jsonl(paths["runtime"])
    pointer_records = read_jsonl(paths["pointer"])
    marker_ok = "kv_fault_vs_prefetch_case_ok" in paths["runner_log"].read_text(
        encoding="utf-8",
        errors="replace",
    )
    large_phase = phase_by_name(phase_records, "large_after_grow") or phase_by_name(
        phase_records, "large"
    )
    repeat_phase = phase_by_name(phase_records, "large_repeat")
    small_phase = phase_by_name(phase_records, "small_after_shrink")
    pointer = next(
        (
            record
            for record in pointer_records
            if record.get("action") == "stage_l6_kv_pointer_map"
        ),
        {},
    )
    selects = [
        {
            "target": record.get("dynamic_pool_target_blocks"),
            "selected_block_ids": record.get("selected_block_ids"),
        }
        for record in runtime_records
        if record.get("action") == "gpu_resident_free_block_select"
    ]
    return {
        "case": case_name,
        "ok": marker_ok,
        "paths": {key: str(value) for key, value in paths.items() if key != "case_dir"},
        "pointer_map_mapped_blocks": int(pointer.get("mapped_blocks") or 0),
        "physical_offloads": physical_record_count(
            residency_records,
            action="residency_kv_data_offload_issued",
            physical=True,
        ),
        "physical_prefetches": physical_record_count(
            residency_records,
            action="residency_kv_prefetch_back_issued",
            physical=True,
        ),
        "metadata_prefetches": physical_record_count(
            residency_records,
            action="residency_kv_prefetch_back_issued",
            physical=False,
        ),
        "selected_records": selects,
        "small_phase": small_phase,
        "large_phase": large_phase,
        "repeat_phase": repeat_phase,
        "large_seconds": float(large_phase.get("generate_seconds") or 0.0),
        "repeat_seconds": float(repeat_phase.get("generate_seconds") or 0.0),
        "small_real_kv_gpu_bytes": int(small_phase.get("real_kv_gpu_bytes") or 0),
        "small_real_kv_cpu_bytes": int(small_phase.get("real_kv_cpu_bytes") or 0),
        "large_real_kv_gpu_bytes": int(large_phase.get("real_kv_gpu_bytes") or 0),
        "large_real_kv_cpu_bytes": int(large_phase.get("real_kv_cpu_bytes") or 0),
    }


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    cases = ["gpu_resident", "fault_on_touch", "explicit_prefetch"]
    summaries = {}
    for case_name in cases:
        paths = run_case(
            case_name=case_name,
            run_dir=run_dir,
            timeout_seconds=args.timeout_seconds,
            small_target=args.small_target,
            large_target=args.large_target,
            small_prompt_tokens=args.small_prompt_tokens,
            large_prompt_tokens=args.large_prompt_tokens,
        )
        summaries[case_name] = summarize_case(case_name, paths)

    gpu = summaries["gpu_resident"]
    fault = summaries["fault_on_touch"]
    prefetch = summaries["explicit_prefetch"]
    checks = [
        {
            "name": "all_cases_completed",
            "passed": all(summary["ok"] for summary in summaries.values()),
        },
        {
            "name": "fault_case_touched_cpu_resident_kv_without_error",
            "passed": (
                fault["physical_offloads"] > 0
                and fault["physical_prefetches"] == 0
                and int(fault["small_phase"].get("real_kv_cpu_bytes") or 0) > 0
                and int(fault["large_phase"].get("output_tokens") or 0) > 0
            ),
        },
        {
            "name": "fault_case_auto_migrated_pages_to_gpu",
            "passed": (
                fault["large_real_kv_gpu_bytes"] > fault["small_real_kv_gpu_bytes"]
            ),
        },
        {
            "name": "explicit_prefetch_was_issued",
            "passed": prefetch["physical_prefetches"] > 0,
        },
        {
            "name": "gpu_resident_case_avoids_offload",
            "passed": gpu["physical_offloads"] == 0,
        },
    ]
    result = {
        "passed": all(item["passed"] for item in checks),
        "run_dir": str(run_dir),
        "checks": checks,
        "cases": summaries,
        "latency_ratios": {
            "fault_large_vs_fault_repeat": ratio(
                fault["large_seconds"],
                fault["repeat_seconds"],
            ),
            "fault_large_vs_gpu_resident_large": ratio(
                fault["large_seconds"],
                gpu["large_seconds"],
            ),
            "fault_large_vs_explicit_prefetch_large": ratio(
                fault["large_seconds"],
                prefetch["large_seconds"],
            ),
            "explicit_prefetch_large_vs_gpu_resident_large": ratio(
                prefetch["large_seconds"],
                gpu["large_seconds"],
            ),
        },
    }
    output_json = (
        Path(args.output_json)
        if args.output_json
        else run_dir / "kv_fault_vs_prefetch_summary.json"
    )
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
