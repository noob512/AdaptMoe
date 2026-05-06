#!/usr/bin/env python3
"""Validate Stage L5 KV block metadata-only residency."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Stage L5 KV block metadata-only residency. The default "
            "self-test validates the trace contract without a GPU; --gpu-run "
            "records metadata from the real vLLM BlockPool/runtime path."
        )
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--runtime-trace-jsonl")
    parser.add_argument("--bench-log")
    parser.add_argument("--runner-log")
    parser.add_argument("--allocator-log")
    parser.add_argument("--mode", choices=("trace_only", "enforce"), default="trace_only")
    parser.add_argument("--global-gpu-bytes", type=int, default=24 * 1024**3)
    parser.add_argument("--weight-gpu-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--kv-gpu-bytes", type=int, default=1 * 1024**3)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--safety-margin-bytes", type=int, default=1 * 1024**3)
    parser.add_argument("--runtime-budget-blocks", type=int, default=1)
    parser.add_argument("--runtime-candidate-limit", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l5_success_check_{stamp}")


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    return run_dir / "vllm_uvm_pool_residency_stage_l5.jsonl"


def resolve_runtime_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.runtime_trace_jsonl:
        return Path(args.runtime_trace_jsonl)
    return run_dir / "vllm_uvm_kv_runtime_stage_l5.jsonl"


def resolve_allocator_log(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.allocator_log:
        return Path(args.allocator_log)
    return run_dir / "vllm_uvm_allocator_trace_stage_l5.log"


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError:
                continue
    return records


def parse_failed_requests(bench_log: Path | None) -> int | None:
    if bench_log is None or not bench_log.is_file():
        return None
    text = bench_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Failed requests:\s+(\d+)", text)
    if not matches:
        return None
    return int(matches[-1])


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l5",
            "VLLM_UVM_RESIDENCY_MODE": args.mode,
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES": str(args.global_gpu_bytes),
            "VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES": str(args.weight_gpu_bytes),
            "VLLM_UVM_RESIDENCY_KV_GPU_BYTES": str(args.kv_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES": str(args.scratch_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES": str(
                args.safety_margin_bytes
            ),
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
        }
    )
    return env


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l5_self_test_runner.log"
    )
    try:
        trace_jsonl.unlink()
    except FileNotFoundError:
        pass
    print("===========================================================")
    print(" Stage L5 Success Check: running KV metadata self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    code = r'''
from vllm.device_allocator.uvm_pool_residency import (
    record_uvm_residency_kv_metadata_candidate,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
)

bytes_per_block = 1572864
for block_id in range(1, 9):
    register_uvm_residency_region(
        pool="kv",
        region_id=f"kv:0:{block_id}",
        ptr=None,
        bytes=bytes_per_block,
        state="gpu",
        evictable=True,
        active=False,
        metadata={
            "stage": "stage_l5",
            "source": "self_test",
            "group_id": 0,
            "block_id": block_id,
            "bytes_per_block": bytes_per_block,
            "ref_cnt": 0,
            "is_prefix_cache": block_id % 2 == 0,
            "in_free_queue": True,
            "offload_capable": False,
            "metadata_only": True,
        },
    )
request_uvm_gpu_residency(
    pool="kv",
    region_id="kv:allocation:self_test",
    bytes=bytes_per_block * 4,
    scope_key="kv_request:self_test",
    metadata={
        "stage": "stage_l5",
        "source": "self_test_admission",
        "metadata_only": True,
        "data_offload_issued": False,
    },
)
for block_id in range(1, 5):
    record_uvm_residency_kv_metadata_candidate(
        region_id=f"kv:0:{block_id}",
        block_id=block_id,
        group_id=0,
        bytes_per_block=bytes_per_block,
        safe=True,
        reason="self_test_budget_pressure",
        metadata={
            "stage": "stage_l5",
            "source": "self_test_candidate",
            "ref_cnt": 0,
            "is_null": False,
            "in_free_queue": True,
            "offload_capable": False,
            "metadata_only": True,
        },
    )
'''
    with runner_log.open("w", encoding="utf-8") as handle:
        handle.write("stage_l5_self_test: writing KV metadata-only trace\n")
    subprocess.run(
        ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code],
        cwd=SCRIPT_DIR,
        env=residency_env(args, trace_jsonl),
        check=True,
    )
    return trace_jsonl, runner_log, None


def run_gpu_probe(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runtime_trace = resolve_runtime_trace(args, run_dir)
    allocator_log = resolve_allocator_log(args, run_dir)
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l5.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l5_gpu_runner.log"
    )
    for path in (trace_jsonl, runtime_trace):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model",
        args.model,
        "--mode",
        "trace",
        "--allocator-log",
        str(allocator_log),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_l5.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_l5.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_l5.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-residency-enable",
        "1",
        "--uvm-residency-stage",
        "stage_l5",
        "--uvm-residency-mode",
        args.mode,
        "--uvm-residency-trace-file",
        str(trace_jsonl),
        "--uvm-residency-global-gpu-bytes",
        str(args.global_gpu_bytes),
        "--uvm-residency-weight-gpu-bytes",
        str(args.weight_gpu_bytes),
        "--uvm-residency-kv-gpu-bytes",
        str(args.kv_gpu_bytes),
        "--uvm-residency-scratch-gpu-bytes",
        str(args.scratch_gpu_bytes),
        "--uvm-residency-safety-margin-bytes",
        str(args.safety_margin_bytes),
        "--uvm-residency-kv-metadata-enable",
        "1",
        "--uvm-kv-runtime-enable",
        "1",
        "--uvm-kv-runtime-mode",
        "trace_only",
        "--uvm-kv-runtime-budget-blocks",
        str(args.runtime_budget_blocks),
        "--uvm-kv-runtime-trace-file",
        str(runtime_trace),
        "--uvm-kv-runtime-candidate-limit",
        str(args.runtime_candidate_limit),
        "--uvm-kv-runtime-prefix-evict-enable",
        "0",
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L5 Success Check: running GPU KV metadata probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Runtime trace: {runtime_trace}")
    print(f" Allocator trace: {allocator_log}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=os.environ.copy(),
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
        raise SystemExit(f"Stage L5 GPU probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, runner_log, bench_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    kv_region_records: list[dict[str, Any]] = []
    kv_request_records: list[dict[str, Any]] = []
    safe_candidates: list[dict[str, Any]] = []
    unsafe_candidates: list[dict[str, Any]] = []
    kv_data_offloads: list[dict[str, Any]] = []
    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        if action == "residency_register_region" and record.get("pool") == "kv":
            kv_region_records.append(record)
        if action == "residency_request" and record.get("pool") == "kv":
            kv_request_records.append(record)
        if action == "residency_kv_safe_candidate":
            safe_candidates.append(record)
        if action == "residency_kv_unsafe_candidate":
            unsafe_candidates.append(record)
        if record.get("pool") == "kv" and "offload" in action:
            kv_data_offloads.append(record)
    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "config": config,
        "summary": summary,
        "kv_region_records": kv_region_records,
        "kv_request_records": kv_request_records,
        "safe_candidates": safe_candidates,
        "unsafe_candidates": unsafe_candidates,
        "kv_data_offloads": kv_data_offloads,
    }


def validate(
    *,
    trace_jsonl: Path,
    runner_log: Path | None,
    bench_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace = summarize_trace(records)
    config = trace["config"]
    summary = trace["summary"]
    kv_regions = trace["kv_region_records"]
    kv_requests = trace["kv_request_records"]
    safe_candidates = trace["safe_candidates"]
    unsafe_candidates = trace["unsafe_candidates"]
    kv_data_offloads = trace["kv_data_offloads"]
    failed_requests = parse_failed_requests(bench_log)
    bytes_positive = all(int(record.get("bytes") or 0) > 0 for record in kv_regions)
    region_ids_valid = all(
        re.match(r"^kv:\d+:\d+$", str(record.get("region_id") or ""))
        for record in kv_regions
    )
    metadata_only = all(
        (record.get("metadata") or {}).get("metadata_only") is True
        for record in kv_regions[: min(len(kv_regions), 32)]
    )
    candidates_safe = all(record.get("safe") is True for record in safe_candidates)
    data_offload_count = len(kv_data_offloads)
    checks = [
        check("trace_jsonl_present", trace_jsonl.is_file(), f"trace_jsonl={trace_jsonl}"),
        check("trace_records_present", trace["trace_records"] > 0, f"records={trace['trace_records']}"),
        check("residency_config_present", bool(config), f"config={config}"),
        check(
            "stage_l5_config_present",
            config.get("stage") == "stage_l5" or config.get("stage_label") == "stage_l5",
            f"config={config}",
        ),
        check(
            "kv_metadata_enabled_visible",
            config.get("kv_metadata_enable") is True,
            f"kv_metadata_enable={config.get('kv_metadata_enable')}",
        ),
        check(
            "kv_region_register_records_present",
            len(kv_regions) > 0,
            f"kv_region_register_records={len(kv_regions)}",
        ),
        check("kv_region_ids_valid", region_ids_valid, f"sample={kv_regions[:3]}"),
        check(
            "kv_region_bytes_per_block_positive",
            bytes_positive,
            f"kv_regions={len(kv_regions)}",
        ),
        check(
            "kv_metadata_only_visible",
            metadata_only,
            f"sample={kv_regions[:3]}",
        ),
        check(
            "kv_residency_request_records_present",
            len(kv_requests) > 0,
            f"kv_request_records={len(kv_requests)}",
        ),
        check(
            "kv_safe_candidate_records_present",
            len(safe_candidates) > 0,
            f"safe_candidate_records={len(safe_candidates)}",
        ),
        check(
            "kv_safe_candidates_are_safe",
            candidates_safe,
            f"safe_candidate_records={len(safe_candidates)}",
        ),
        check(
            "unsafe_kv_candidate_records_zero",
            len(unsafe_candidates) == 0,
            f"unsafe_candidate_records={len(unsafe_candidates)}",
        ),
        check(
            "kv_data_offload_records_zero",
            data_offload_count == 0,
            f"kv_data_offload_records={data_offload_count}",
        ),
        check("benchmark_no_failed_requests", failed_requests in (None, 0), f"failed_requests={failed_requests}"),
    ]
    runner_log_flags = {"contains_traceback": False, "contains_server_exit": False}
    if runner_log and runner_log.is_file():
        text = runner_log.read_text(encoding="utf-8", errors="replace")
        runner_log_flags["contains_traceback"] = "traceback" in text.lower()
        runner_log_flags["contains_server_exit"] = "Server exited early" in text
        checks.append(
            check(
                "runner_log_clean",
                not runner_log_flags["contains_traceback"]
                and not runner_log_flags["contains_server_exit"],
                str(runner_log_flags),
            )
        )
    passed = all(item["passed"] for item in checks)
    output_json = Path(args.output_json) if args.output_json else (
        trace_jsonl.parent / "stage_l5_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "bench_log": str(bench_log) if bench_log else None,
        "trace_summary": trace,
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    trace = result["trace_summary"]
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L5 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- kv_metadata="
        f"regions={len(trace['kv_region_records'])} "
        f"requests={len(trace['kv_request_records'])} "
        f"safe_candidates={len(trace['safe_candidates'])} "
        f"unsafe_candidates={len(trace['unsafe_candidates'])} "
        f"kv_data_offloads={len(trace['kv_data_offloads'])}"
    )
    print(f"- final_summary={trace['summary']}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={result['output_json']}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    runner_log: Path | None = None
    bench_log: Path | None = Path(args.bench_log) if args.bench_log else None
    if args.skip_run:
        trace_jsonl = resolve_trace(args, run_dir)
        runner_log = Path(args.runner_log) if args.runner_log else None
    elif args.gpu_run:
        trace_jsonl, runner_log, bench_log = run_gpu_probe(args, run_dir)
    else:
        trace_jsonl, runner_log, bench_log = run_self_test(args, run_dir)
    result = validate(
        trace_jsonl=trace_jsonl,
        runner_log=runner_log,
        bench_log=bench_log,
        args=args,
    )
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
