#!/usr/bin/env python3
"""Validate Stage L6 KV safe-block offload / prefetch-back prototype."""

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
            "Check Stage L6 safe KV block offload / prefetch-back. The default "
            "self-test validates the trace contract without a GPU; --gpu-run "
            "validates the real vLLM BlockPool/runtime path."
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
    parser.add_argument("--max-offload-blocks", type=int, default=4)
    parser.add_argument("--data-offload-mode", choices=("trace_only", "prefetch_cpu"), default="trace_only")
    parser.add_argument("--prefetch-back-mode", choices=("trace_only", "prefetch"), default="trace_only")
    parser.add_argument(
        "--strict-issued",
        action="store_true",
        help="Require real/issued KV offload records. Use for L6.1 validation.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l6_success_check_{stamp}")


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    return Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_pool_residency_stage_l6.jsonl"
    )


def resolve_runtime_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    return Path(args.runtime_trace_jsonl) if args.runtime_trace_jsonl else (
        run_dir / "vllm_uvm_kv_runtime_stage_l6.jsonl"
    )


def resolve_allocator_log(args: argparse.Namespace, run_dir: Path) -> Path:
    return Path(args.allocator_log) if args.allocator_log else (
        run_dir / "vllm_uvm_allocator_trace_stage_l6.log"
    )


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def parse_failed_requests(bench_log: Path | None) -> int | None:
    if bench_log is None or not bench_log.is_file():
        return None
    text = bench_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Failed requests:\s+(\d+)", text)
    return int(matches[-1]) if matches else None


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l6",
            "VLLM_UVM_RESIDENCY_MODE": args.mode,
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES": str(args.global_gpu_bytes),
            "VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES": str(args.weight_gpu_bytes),
            "VLLM_UVM_RESIDENCY_KV_GPU_BYTES": str(args.kv_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES": str(args.scratch_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES": str(args.safety_margin_bytes),
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_DATA_OFFLOAD_MODE": args.data_offload_mode,
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_PREFETCH_BACK_MODE": args.prefetch_back_mode,
            "VLLM_UVM_RESIDENCY_KV_MAX_OFFLOAD_BLOCKS_PER_STEP": str(
                args.max_offload_blocks
            ),
            "VLLM_UVM_RESIDENCY_KV_POINTER_MAP_FILE": str(
                trace_jsonl.parent / "vllm_uvm_kv_pointer_map_stage_l6.jsonl"
            ),
        }
    )
    return env


def resolve_pointer_map(args: argparse.Namespace, run_dir: Path) -> Path:
    return run_dir / "vllm_uvm_kv_pointer_map_stage_l6.jsonl"


def run_self_test(
    args: argparse.Namespace, run_dir: Path
) -> tuple[Path, Path, None, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runtime_trace = resolve_runtime_trace(args, run_dir)
    pointer_map_file = resolve_pointer_map(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l6_self_test_runner.log"
    )
    try:
        trace_jsonl.unlink()
    except FileNotFoundError:
        pass
    print("===========================================================")
    print(" Stage L6 Success Check: running KV offload self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    code = r'''
from vllm.device_allocator.uvm_pool_residency import (
    mark_uvm_residency_state,
    record_uvm_residency_action,
    register_uvm_residency_region,
)

bytes_per_block = 1572864
for block_id in range(1, 5):
    register_uvm_residency_region(
        pool="kv",
        region_id=f"kv:0:{block_id}",
        ptr=0x10000000 + block_id * bytes_per_block,
        bytes=bytes_per_block,
        state="gpu",
        evictable=True,
        active=False,
        metadata={
            "stage": "stage_l6",
            "source": "self_test",
            "group_id": 0,
            "block_id": block_id,
            "bytes_per_block": bytes_per_block,
            "ref_cnt": 0,
            "is_prefix_cache": False,
            "in_free_queue": True,
            "offload_capable": True,
            "metadata_only": False,
        },
    )

record_uvm_residency_action(
    action="residency_kv_data_offload_attempt",
    pool="kv",
    region_id="kv:0:1",
    bytes=bytes_per_block,
    metadata={"stage": "stage_l6", "safe": True, "ref_cnt": 0, "in_free_queue": True},
)
mark_uvm_residency_state(
    region_id="kv:0:1",
    state="prefetching_cpu",
    metadata={"stage": "stage_l6", "source": "self_test_offload"},
)
mark_uvm_residency_state(
    region_id="kv:0:1",
    state="cpu",
    metadata={"stage": "stage_l6", "source": "self_test_offload"},
)
record_uvm_residency_action(
    action="residency_kv_data_offload_issued",
    pool="kv",
    region_id="kv:0:1",
    bytes=bytes_per_block,
    mode="trace_only",
    metadata={"stage": "stage_l6", "safe": True, "ref_cnt": 0, "in_free_queue": True},
)
mark_uvm_residency_state(
    region_id="kv:0:1",
    state="prefetching_gpu",
    metadata={"stage": "stage_l6", "source": "self_test_prefetch_back"},
)
mark_uvm_residency_state(
    region_id="kv:0:1",
    state="gpu",
    metadata={"stage": "stage_l6", "source": "self_test_prefetch_back"},
)
record_uvm_residency_action(
    action="residency_kv_prefetch_back_issued",
    pool="kv",
    region_id="kv:0:1",
    bytes=bytes_per_block,
    mode="trace_only",
    metadata={"stage": "stage_l6", "reason": "self_test_reuse"},
)
'''
    runner_log.write_text("stage_l6_self_test: writing KV offload trace\n", encoding="utf-8")
    subprocess.run(
        ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code],
        cwd=SCRIPT_DIR,
        env=residency_env(args, trace_jsonl),
        check=True,
    )
    return trace_jsonl, runner_log, None, runtime_trace, pointer_map_file


def run_gpu_probe(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runtime_trace = resolve_runtime_trace(args, run_dir)
    allocator_log = resolve_allocator_log(args, run_dir)
    pointer_map_file = resolve_pointer_map(args, run_dir)
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l6.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l6_gpu_runner.log"
    )
    for path in (trace_jsonl, runtime_trace, pointer_map_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model", args.model,
        "--mode", "trace",
        "--allocator-log", str(allocator_log),
        "--trace-log", str(run_dir / "uvm_kv_fault_stats_stage_l6.log"),
        "--address-log", str(run_dir / "vllm_uvm_address_regions_stage_l6.log"),
        "--server-log", str(run_dir / "vllm_server_stage_l6.log"),
        "--bench-log", str(bench_log),
        "--uvm-residency-enable", "1",
        "--uvm-residency-stage", "stage_l6",
        "--uvm-residency-mode", args.mode,
        "--uvm-residency-trace-file", str(trace_jsonl),
        "--uvm-residency-global-gpu-bytes", str(args.global_gpu_bytes),
        "--uvm-residency-weight-gpu-bytes", str(args.weight_gpu_bytes),
        "--uvm-residency-kv-gpu-bytes", str(args.kv_gpu_bytes),
        "--uvm-residency-scratch-gpu-bytes", str(args.scratch_gpu_bytes),
        "--uvm-residency-safety-margin-bytes", str(args.safety_margin_bytes),
        "--uvm-residency-kv-metadata-enable", "1",
        "--uvm-residency-kv-data-offload-enable", "1",
        "--uvm-residency-kv-data-offload-mode", args.data_offload_mode,
        "--uvm-residency-kv-prefetch-back-enable", "1",
        "--uvm-residency-kv-prefetch-back-mode", args.prefetch_back_mode,
        "--uvm-residency-kv-max-offload-blocks-per-step", str(args.max_offload_blocks),
        "--uvm-residency-kv-pointer-map-file", str(pointer_map_file),
        "--uvm-kv-runtime-enable", "1",
        "--uvm-kv-runtime-mode", "trace_only",
        "--uvm-kv-runtime-budget-blocks", str(args.runtime_budget_blocks),
        "--uvm-kv-runtime-trace-file", str(runtime_trace),
        "--uvm-kv-runtime-candidate-limit", str(args.runtime_candidate_limit),
        "--uvm-kv-runtime-prefix-evict-enable", "0",
        "--prompts", str(args.prompts),
        "--request-rate", str(args.request_rate),
        "--output-len", str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L6 Success Check: running GPU KV safe-offload probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Runtime trace: {runtime_trace}")
    print(f" Pointer map: {pointer_map_file}")
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
        raise SystemExit(f"Stage L6 GPU probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, runner_log, bench_log, runtime_trace, pointer_map_file


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    kv_regions: list[dict[str, Any]] = []
    offload_attempts: list[dict[str, Any]] = []
    offload_issued: list[dict[str, Any]] = []
    offload_fallbacks: list[dict[str, Any]] = []
    prefetch_issued: list[dict[str, Any]] = []
    prefetch_fallbacks: list[dict[str, Any]] = []
    unsafe_offloads: list[dict[str, Any]] = []
    state_updates: list[dict[str, Any]] = []
    pointer_map_loaded: list[dict[str, Any]] = []
    pointer_map_registered: list[dict[str, Any]] = []
    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        if action == "residency_config":
            config = record
        elif action == "residency_summary":
            summary = record
        elif action == "residency_register_region" and record.get("pool") == "kv":
            kv_regions.append(record)
        elif action == "residency_kv_data_offload_attempt":
            offload_attempts.append(record)
        elif action == "residency_kv_data_offload_issued":
            offload_issued.append(record)
        elif action == "residency_kv_data_offload_fallback":
            offload_fallbacks.append(record)
            if (record.get("metadata") or {}).get("safe") is False:
                unsafe_offloads.append(record)
        elif action == "residency_kv_prefetch_back_issued":
            prefetch_issued.append(record)
        elif action == "residency_kv_prefetch_back_fallback":
            prefetch_fallbacks.append(record)
        elif action == "residency_state_update" and record.get("pool") == "kv":
            state_updates.append(record)
        elif action == "stage_l6_kv_pointer_map_loaded":
            pointer_map_loaded.append(record)
        elif action == "stage_l6_kv_pointer_map_registered":
            pointer_map_registered.append(record)
    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "config": config,
        "summary": summary,
        "kv_region_records": kv_regions,
        "offload_attempts": offload_attempts,
        "offload_issued": offload_issued,
        "offload_fallbacks": offload_fallbacks,
        "prefetch_issued": prefetch_issued,
        "prefetch_fallbacks": prefetch_fallbacks,
        "unsafe_offloads": unsafe_offloads,
        "state_updates": state_updates,
        "pointer_map_loaded": pointer_map_loaded,
        "pointer_map_registered": pointer_map_registered,
    }


def validate(
    *,
    trace_jsonl: Path,
    runtime_trace_jsonl: Path | None,
    pointer_map_file: Path | None,
    runner_log: Path | None,
    bench_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace = summarize_trace(records)
    runtime_records = read_jsonl(runtime_trace_jsonl)
    runtime_trace = summarize_trace(runtime_records)
    pointer_map_records = [
        record for record in read_jsonl(pointer_map_file)
        if record.get("action") == "stage_l6_kv_pointer_map"
    ]
    pointer_map_entries = sum(
        int(record.get("mapped_blocks") or 0) for record in pointer_map_records
    )
    pointer_map_loaded = (
        bool(trace["pointer_map_loaded"])
        or bool(runtime_trace["pointer_map_loaded"])
        or len(trace["offload_issued"]) > 0
    )
    pointer_map_registered = (
        bool(trace["pointer_map_registered"])
        or bool(runtime_trace["pointer_map_registered"])
        or len(trace["offload_issued"]) > 0
    )
    config = trace["config"]
    summary = trace["summary"]
    failed_requests = parse_failed_requests(bench_log)
    has_cpu_state = any(record.get("new_state") == "cpu" for record in trace["state_updates"])
    has_gpu_state = any(record.get("new_state") == "gpu" for record in trace["state_updates"])
    offload_activity = len(trace["offload_issued"]) + len(trace["offload_fallbacks"])
    checks = [
        check("trace_jsonl_present", trace_jsonl.is_file(), f"trace_jsonl={trace_jsonl}"),
        check("trace_records_present", trace["trace_records"] > 0, f"records={trace['trace_records']}"),
        check("residency_config_present", bool(config), f"config={config}"),
        check(
            "stage_l6_config_present",
            config.get("stage") == "stage_l6" or config.get("stage_label") == "stage_l6",
            f"config={config}",
        ),
        check(
            "kv_metadata_enabled_visible",
            config.get("kv_metadata_enable") is True,
            f"kv_metadata_enable={config.get('kv_metadata_enable')}",
        ),
        check(
            "kv_data_offload_enabled_visible",
            config.get("kv_data_offload_enable") is True,
            f"kv_data_offload_enable={config.get('kv_data_offload_enable')}",
        ),
        check(
            "kv_prefetch_back_enabled_visible",
            config.get("kv_prefetch_back_enable") is True,
            f"kv_prefetch_back_enable={config.get('kv_prefetch_back_enable')}",
        ),
        check(
            "kv_region_register_records_present",
            len(trace["kv_region_records"]) > 0,
            f"kv_region_register_records={len(trace['kv_region_records'])}",
        ),
        check(
            "kv_safe_offload_attempt_records_present",
            len(trace["offload_attempts"]) > 0,
            f"offload_attempts={len(trace['offload_attempts'])}",
        ),
        check(
            "kv_safe_offload_activity_present",
            offload_activity > 0,
            f"issued={len(trace['offload_issued'])} fallbacks={len(trace['offload_fallbacks'])}",
        ),
        check(
            "kv_pointer_map_loaded",
            pointer_map_loaded or not args.gpu_run,
            "residency_records="
            f"{trace['pointer_map_loaded'][-1:] if trace['pointer_map_loaded'] else []} "
            "runtime_records="
            f"{runtime_trace['pointer_map_loaded'][-1:] if runtime_trace['pointer_map_loaded'] else []} "
            f"pointer_map_entries={pointer_map_entries}",
        ),
        check(
            "kv_pointer_map_registered",
            pointer_map_registered or not args.gpu_run,
            "residency_records="
            f"{trace['pointer_map_registered'][-1:] if trace['pointer_map_registered'] else []} "
            "runtime_records="
            f"{runtime_trace['pointer_map_registered'][-1:] if runtime_trace['pointer_map_registered'] else []} "
            f"offload_issued={len(trace['offload_issued'])}",
        ),
        check(
            "unsafe_kv_offload_records_zero",
            len(trace["unsafe_offloads"]) == 0,
            f"unsafe_offloads={len(trace['unsafe_offloads'])}",
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]
    if args.strict_issued:
        checks.extend(
            [
                check(
                    "strict_kv_data_offload_issued_present",
                    len(trace["offload_issued"]) > 0,
                    f"offload_issued={len(trace['offload_issued'])}",
                ),
                check(
                    "strict_kv_state_reached_cpu",
                    has_cpu_state,
                    f"has_cpu_state={has_cpu_state}",
                ),
            ]
        )
    if not args.gpu_run:
        checks.extend(
            [
                check(
                    "kv_data_offload_issued_present",
                    len(trace["offload_issued"]) > 0,
                    f"offload_issued={len(trace['offload_issued'])}",
                ),
                check(
                    "kv_prefetch_back_issued_present",
                    len(trace["prefetch_issued"]) > 0,
                    f"prefetch_issued={len(trace['prefetch_issued'])}",
                ),
                check("kv_state_reached_cpu", has_cpu_state, f"has_cpu_state={has_cpu_state}"),
                check("kv_state_returned_gpu", has_gpu_state, f"has_gpu_state={has_gpu_state}"),
            ]
        )
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
        trace_jsonl.parent / "stage_l6_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "runtime_trace_jsonl": str(runtime_trace_jsonl) if runtime_trace_jsonl else None,
        "pointer_map_file": str(pointer_map_file) if pointer_map_file else None,
        "bench_log": str(bench_log) if bench_log else None,
        "trace_summary": trace,
        "runtime_trace_summary": runtime_trace,
        "pointer_map_summary": {
            "records": len(pointer_map_records),
            "mapped_blocks": pointer_map_entries,
        },
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    trace = result["trace_summary"]
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L6 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    if result.get("runtime_trace_jsonl"):
        print(f" Runtime trace: {result['runtime_trace_jsonl']}")
    if result.get("pointer_map_file"):
        print(f" Pointer map: {result['pointer_map_file']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- kv_l6="
        f"regions={len(trace['kv_region_records'])} "
        f"offload_attempts={len(trace['offload_attempts'])} "
        f"offload_issued={len(trace['offload_issued'])} "
        f"offload_fallbacks={len(trace['offload_fallbacks'])} "
        f"prefetch_issued={len(trace['prefetch_issued'])} "
        f"prefetch_fallbacks={len(trace['prefetch_fallbacks'])} "
        f"unsafe_offloads={len(trace['unsafe_offloads'])} "
        f"pointer_map_loaded={len(trace['pointer_map_loaded'])} "
        f"pointer_map_records={result.get('pointer_map_summary', {}).get('records', 0)}"
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
    runtime_trace: Path | None = None
    pointer_map_file: Path | None = None
    if args.skip_run:
        trace_jsonl = resolve_trace(args, run_dir)
        runtime_trace = resolve_runtime_trace(args, run_dir)
        pointer_map_file = resolve_pointer_map(args, run_dir)
        runner_log = Path(args.runner_log) if args.runner_log else None
    elif args.gpu_run:
        trace_jsonl, runner_log, bench_log, runtime_trace, pointer_map_file = (
            run_gpu_probe(args, run_dir)
        )
    else:
        trace_jsonl, runner_log, bench_log, runtime_trace, pointer_map_file = (
            run_self_test(args, run_dir)
        )
    result = validate(
        trace_jsonl=trace_jsonl,
        runtime_trace_jsonl=runtime_trace,
        pointer_map_file=pointer_map_file,
        runner_log=runner_log,
        bench_log=bench_log,
        args=args,
    )
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
