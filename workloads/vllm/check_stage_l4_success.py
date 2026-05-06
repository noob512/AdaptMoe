#!/usr/bin/env python3
"""Validate Stage L4 KV cache sizing reports."""

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
            "Check Stage L4 KV cache sizing reports. The default self-test "
            "validates the L4 budget-to-block trace contract without a GPU. "
            "Use --gpu-run to record a report from the real vLLM KV cache "
            "configuration path."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage L4 run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage L residency JSONL.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--runner-log", help="Existing or new runner log.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="enforce",
        help="Stage L residency mode. L4 first version is report-only either way.",
    )
    parser.add_argument(
        "--kv-sizing-policy",
        choices=("report_only", "clamp", "override"),
        default="override",
        help="Stage L4 policy. Use report_only for non-mutating probes.",
    )
    parser.add_argument(
        "--kv-sizing-max-blocks",
        type=int,
        default=0,
        help="Optional max block guard for override. 0 means no extra guard.",
    )
    parser.add_argument("--global-gpu-bytes", type=int, default=24 * 1024**3)
    parser.add_argument("--weight-gpu-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--kv-gpu-bytes", type=int, default=0)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--safety-margin-bytes", type=int, default=1 * 1024**3)
    parser.add_argument("--bytes-per-block", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--original-num-gpu-blocks", type=int, default=4096)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate an existing trace; use --trace-jsonl or --run-dir.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l4_success_check_{stamp}")


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    return run_dir / "vllm_uvm_pool_residency_stage_l4.jsonl"


def resolve_allocator_log(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.allocator_log:
        return Path(args.allocator_log)
    return run_dir / "vllm_uvm_allocator_trace_stage_l4.log"


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


def optional_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def expected_derived_kv_budget(args: argparse.Namespace) -> int:
    return max(
        args.global_gpu_bytes
        - args.weight_gpu_bytes
        - args.scratch_gpu_bytes
        - args.safety_margin_bytes,
        0,
    )


def expected_effective_kv_budget(args: argparse.Namespace) -> int:
    return args.kv_gpu_bytes if args.kv_gpu_bytes > 0 else expected_derived_kv_budget(args)


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l4",
            "VLLM_UVM_RESIDENCY_MODE": args.mode,
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES": str(args.global_gpu_bytes),
            "VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES": str(args.weight_gpu_bytes),
            "VLLM_UVM_RESIDENCY_KV_GPU_BYTES": str(args.kv_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES": str(args.scratch_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES": str(
                args.safety_margin_bytes
            ),
            "VLLM_UVM_RESIDENCY_KV_SIZING_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_KV_SIZING_POLICY": args.kv_sizing_policy,
            "VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS": str(
                args.kv_sizing_max_blocks
            ),
        }
    )
    return env


def write_kv_sizing_record(
    *,
    args: argparse.Namespace,
    trace_jsonl: Path,
    bytes_per_block: int | None = None,
    original_num_gpu_blocks: int | None = None,
    weight_gpu_bytes: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if bytes_per_block is None:
        bytes_per_block = args.bytes_per_block
    if original_num_gpu_blocks is None:
        original_num_gpu_blocks = args.original_num_gpu_blocks
    code = r'''
import os

from vllm.device_allocator.uvm_pool_residency import (
    record_uvm_residency_kv_sizing,
)

bytes_per_block = int(os.environ["STAGE_L4_BYTES_PER_BLOCK"])
num_blocks = int(os.environ["STAGE_L4_ORIGINAL_NUM_GPU_BLOCKS"])
policy = os.environ.get("VLLM_UVM_RESIDENCY_KV_SIZING_POLICY", "report_only")
kv_budget_bytes = int(os.environ["STAGE_L4_EXPECTED_KV_BUDGET_BYTES"])
max_blocks = int(os.environ.get("VLLM_UVM_RESIDENCY_KV_SIZING_MAX_BLOCKS", "0"))
budget_blocks = kv_budget_bytes // bytes_per_block if bytes_per_block > 0 else 0
final_blocks = num_blocks
if policy == "clamp" and budget_blocks > 0:
    final_blocks = min(num_blocks, budget_blocks)
elif policy == "override" and budget_blocks > 0:
    final_blocks = budget_blocks
    if max_blocks > 0:
        final_blocks = min(final_blocks, max_blocks)
record_uvm_residency_kv_sizing(
    bytes_per_block=bytes_per_block,
    original_num_gpu_blocks=num_blocks,
    final_num_gpu_blocks=final_blocks,
    original_kv_bytes=bytes_per_block * num_blocks,
    final_kv_bytes=bytes_per_block * final_blocks,
    applied=final_blocks != num_blocks,
    policy=policy,
    max_blocks=max_blocks,
    fallback_reason="report_only" if policy == "report_only" else None,
    metadata={
        "source": os.environ.get("STAGE_L4_SOURCE", "self_test"),
        "stage": "stage_l4",
        "report_only_no_num_gpu_blocks_change": policy == "report_only",
    },
)
'''
    env = residency_env(args, trace_jsonl)
    if weight_gpu_bytes is not None:
        env["VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES"] = str(weight_gpu_bytes)
    expected_budget = expected_effective_kv_budget(args)
    if weight_gpu_bytes is not None and args.kv_gpu_bytes <= 0:
        expected_budget = max(
            args.global_gpu_bytes
            - weight_gpu_bytes
            - args.scratch_gpu_bytes
            - args.safety_margin_bytes,
            0,
        )
    env.update(
        {
            "STAGE_L4_BYTES_PER_BLOCK": str(bytes_per_block),
            "STAGE_L4_ORIGINAL_NUM_GPU_BLOCKS": str(original_num_gpu_blocks),
            "STAGE_L4_SOURCE": (metadata or {}).get("source", "self_test"),
            "STAGE_L4_EXPECTED_KV_BUDGET_BYTES": str(expected_budget),
        }
    )
    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l4_self_test_runner.log"
    )
    try:
        trace_jsonl.unlink()
    except FileNotFoundError:
        pass
    print("===========================================================")
    print(" Stage L4 Success Check: running KV sizing self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        handle.write("stage_l4_self_test: writing KV sizing reports\n")
    write_kv_sizing_record(
        args=args,
        trace_jsonl=trace_jsonl,
        metadata={"source": "self_test_primary"},
    )
    lower_weight = max(args.weight_gpu_bytes // 2, 0)
    write_kv_sizing_record(
        args=args,
        trace_jsonl=trace_jsonl,
        weight_gpu_bytes=lower_weight,
        metadata={"source": "self_test_lower_weight_budget"},
    )
    return trace_jsonl, runner_log, None


def run_gpu_probe(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    allocator_log = resolve_allocator_log(args, run_dir)
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l4.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l4_gpu_runner.log"
    )
    try:
        trace_jsonl.unlink()
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
        str(run_dir / "uvm_kv_fault_stats_stage_l4.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_l4.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_l4.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-residency-enable",
        "1",
        "--uvm-residency-stage",
        "stage_l4",
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
        "--uvm-residency-kv-sizing-enable",
        "1",
        "--uvm-residency-kv-sizing-policy",
        args.kv_sizing_policy,
        "--uvm-residency-kv-sizing-max-blocks",
        str(args.kv_sizing_max_blocks),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L4 Success Check: running GPU KV sizing probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
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
        raise SystemExit(f"Stage L4 GPU probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, runner_log, bench_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    kv_sizing_records: list[dict[str, Any]] = []
    kv_action_records = 0
    scratch_eviction_records = 0
    weight_offload_records = 0

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        pool = str(record.get("pool") or "")
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        if action == "residency_kv_sizing":
            kv_sizing_records.append(record)
        if pool == "kv" and action != "residency_kv_sizing":
            kv_action_records += 1
        if pool == "scratch" and "evict" in action:
            scratch_eviction_records += 1
        if pool == "weights" and "offload" in action:
            weight_offload_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "config": config,
        "summary": summary,
        "kv_sizing_records": kv_sizing_records,
        "kv_sizing_record_count": len(kv_sizing_records),
        "kv_action_records": kv_action_records,
        "scratch_eviction_records": scratch_eviction_records,
        "weight_offload_records": weight_offload_records,
    }


def validate(
    *,
    trace_jsonl: Path,
    runner_log: Path | None,
    bench_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace_summary = summarize_trace(records)
    config = trace_summary["config"]
    summary = trace_summary["summary"]
    kv_records = trace_summary["kv_sizing_records"]
    primary = kv_records[0] if kv_records else {}
    failed_requests = parse_failed_requests(bench_log)
    expected_derived = expected_derived_kv_budget(args)
    expected_budget = expected_effective_kv_budget(args)
    bytes_per_block = optional_int(primary.get("kv_sizing_bytes_per_block"))
    expected_blocks = expected_budget // bytes_per_block if bytes_per_block > 0 else 0
    expected_final_blocks = optional_int(
        primary.get("kv_sizing_original_num_gpu_blocks")
    )
    if args.kv_sizing_policy == "clamp" and expected_blocks > 0:
        expected_final_blocks = min(expected_final_blocks, expected_blocks)
    elif args.kv_sizing_policy == "override" and expected_blocks > 0:
        expected_final_blocks = expected_blocks
        if args.kv_sizing_max_blocks > 0:
            expected_final_blocks = min(
                expected_final_blocks,
                args.kv_sizing_max_blocks,
            )
    lower_weight_record = kv_records[1] if len(kv_records) > 1 else {}
    lower_weight_budget = max(args.weight_gpu_bytes // 2, 0)
    lower_weight_expected_derived = max(
        args.global_gpu_bytes
        - lower_weight_budget
        - args.scratch_gpu_bytes
        - args.safety_margin_bytes,
        0,
    )

    checks = [
        check("trace_jsonl_present", trace_jsonl.is_file(), f"trace_jsonl={trace_jsonl}"),
        check(
            "trace_records_present",
            trace_summary["trace_records"] > 0,
            f"trace_records={trace_summary['trace_records']}",
        ),
        check("residency_config_present", bool(config), f"config={config}"),
        check(
            "residency_mode_matches",
            config.get("mode") == args.mode,
            f"mode={config.get('mode')} expected={args.mode}",
        ),
        check(
            "stage_l4_config_present",
            config.get("stage") == "stage_l4"
            or config.get("stage_label") == "stage_l4",
            f"config={config}",
        ),
        check(
            "kv_sizing_config_present",
            trace_summary["kv_sizing_record_count"] > 0
            and summary.get("kv_sizing_config_present") is True,
            (
                "kv_sizing_record_count="
                f"{trace_summary['kv_sizing_record_count']} "
                f"summary_present={summary.get('kv_sizing_config_present')}"
            ),
        ),
        check(
            "weight_budget_bytes_present",
            primary.get("weight_gpu_budget_bytes") == args.weight_gpu_bytes,
            (
                "weight_budget="
                f"{primary.get('weight_gpu_budget_bytes')} "
                f"expected={args.weight_gpu_bytes}"
            ),
        ),
        check(
            "scratch_budget_bytes_present",
            primary.get("scratch_gpu_budget_bytes") == args.scratch_gpu_bytes,
            (
                "scratch_budget="
                f"{primary.get('scratch_gpu_budget_bytes')} "
                f"expected={args.scratch_gpu_bytes}"
            ),
        ),
        check(
            "derived_kv_budget_bytes_present",
            primary.get("kv_sizing_derived_budget_bytes") == expected_derived,
            (
                "derived_kv_budget="
                f"{primary.get('kv_sizing_derived_budget_bytes')} "
                f"expected={expected_derived}"
            ),
        ),
        check(
            "effective_kv_budget_bytes_present",
            primary.get("kv_sizing_budget_bytes") == expected_budget,
            (
                "kv_budget="
                f"{primary.get('kv_sizing_budget_bytes')} expected={expected_budget}"
            ),
        ),
        check(
            "derived_kv_budget_blocks_present",
            primary.get("kv_sizing_budget_blocks") == expected_blocks,
            (
                "kv_budget_blocks="
                f"{primary.get('kv_sizing_budget_blocks')} expected={expected_blocks} "
                f"bytes_per_block={bytes_per_block}"
            ),
        ),
        check(
            "final_num_gpu_blocks_present",
            isinstance(primary.get("kv_sizing_final_num_gpu_blocks"), int)
            and primary.get("kv_sizing_final_num_gpu_blocks") >= 0,
            f"final_num_gpu_blocks={primary.get('kv_sizing_final_num_gpu_blocks')}",
        ),
        check(
            "kv_sizing_policy_matches",
            primary.get("kv_sizing_policy") == args.kv_sizing_policy,
            (
                f"policy={primary.get('kv_sizing_policy')} "
                f"expected={args.kv_sizing_policy}"
            ),
        ),
        check(
            "final_num_gpu_blocks_matches_policy",
            primary.get("kv_sizing_final_num_gpu_blocks") == expected_final_blocks,
            (
                f"final_blocks={primary.get('kv_sizing_final_num_gpu_blocks')} "
                f"expected={expected_final_blocks} "
                f"policy={args.kv_sizing_policy}"
            ),
        ),
        check(
            "kv_sizing_application_state_matches_policy",
            (
                primary.get("kv_sizing_applied") is False
                and primary.get("kv_sizing_report_only") is True
                and primary.get("kv_sizing_final_num_gpu_blocks")
                == primary.get("kv_sizing_original_num_gpu_blocks")
            )
            if args.kv_sizing_policy == "report_only"
            else (
                primary.get("kv_sizing_applied")
                == (
                    primary.get("kv_sizing_final_num_gpu_blocks")
                    != primary.get("kv_sizing_original_num_gpu_blocks")
                )
                and primary.get("kv_sizing_report_only") is False
            ),
            (
                f"applied={primary.get('kv_sizing_applied')} "
                f"report_only={primary.get('kv_sizing_report_only')} "
                "original_blocks="
                f"{primary.get('kv_sizing_original_num_gpu_blocks')} "
                f"final_blocks={primary.get('kv_sizing_final_num_gpu_blocks')} "
                f"policy={args.kv_sizing_policy}"
            ),
        ),
        check(
            "kv_action_records_zero",
            trace_summary["kv_action_records"] == 0,
            f"kv_action_records={trace_summary['kv_action_records']}",
        ),
        check(
            "scratch_eviction_records_zero",
            trace_summary["scratch_eviction_records"] == 0,
            f"scratch_eviction_records={trace_summary['scratch_eviction_records']}",
        ),
        check(
            "weight_offload_records_zero",
            trace_summary["weight_offload_records"] == 0,
            f"weight_offload_records={trace_summary['weight_offload_records']}",
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]
    if not args.gpu_run and not args.skip_run:
        lower_weight_budget_blocks = (
            lower_weight_expected_derived // bytes_per_block
            if bytes_per_block > 0
            else 0
        )
        checks.extend(
            [
                check(
                    "weight_budget_smaller_makes_kv_budget_larger",
                    lower_weight_record.get("kv_sizing_derived_budget_bytes")
                    == lower_weight_expected_derived
                    and lower_weight_expected_derived > expected_derived,
                    (
                        "lower_weight_derived="
                        f"{lower_weight_record.get('kv_sizing_derived_budget_bytes')} "
                        f"expected={lower_weight_expected_derived} "
                        f"primary_derived={expected_derived}"
                    ),
                ),
                check(
                    "final_num_gpu_blocks_changes_with_budget",
                    (
                        args.kv_sizing_policy == "override"
                        and args.kv_gpu_bytes <= 0
                        and args.kv_sizing_max_blocks == 0
                        and lower_weight_record.get(
                            "kv_sizing_final_num_gpu_blocks"
                        )
                        == lower_weight_budget_blocks
                        and lower_weight_record.get(
                            "kv_sizing_final_num_gpu_blocks"
                        )
                        > primary.get("kv_sizing_final_num_gpu_blocks")
                    )
                    or args.kv_sizing_policy != "override"
                    or args.kv_gpu_bytes > 0
                    or args.kv_sizing_max_blocks > 0,
                    (
                        "lower_final_blocks="
                        f"{lower_weight_record.get('kv_sizing_final_num_gpu_blocks')} "
                        f"expected={lower_weight_budget_blocks} "
                        "primary_final_blocks="
                        f"{primary.get('kv_sizing_final_num_gpu_blocks')} "
                        f"policy={args.kv_sizing_policy}"
                    ),
                ),
            ]
        )

    runner_log_flags = {
        "contains_traceback": False,
        "contains_server_exit": False,
    }
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
        trace_jsonl.parent / "stage_l4_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "bench_log": str(bench_log) if bench_log else None,
        "trace_summary": trace_summary,
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    trace = result["trace_summary"]
    summary = trace["summary"]
    kv_records = trace["kv_sizing_records"]
    primary = kv_records[0] if kv_records else {}
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L4 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- kv_sizing_primary="
        f"records={trace['kv_sizing_record_count']} "
        f"mode={primary.get('kv_sizing_mode')} "
        f"budget_bytes={primary.get('kv_sizing_budget_bytes')} "
        f"derived_budget_bytes={primary.get('kv_sizing_derived_budget_bytes')} "
        f"budget_blocks={primary.get('kv_sizing_budget_blocks')} "
        f"bytes_per_block={primary.get('kv_sizing_bytes_per_block')} "
        f"original_blocks={primary.get('kv_sizing_original_num_gpu_blocks')} "
        f"final_blocks={primary.get('kv_sizing_final_num_gpu_blocks')} "
        f"applied={primary.get('kv_sizing_applied')} "
        f"report_only={primary.get('kv_sizing_report_only')} "
        f"max_blocks={primary.get('kv_sizing_max_blocks')} "
        f"fallback={primary.get('kv_sizing_fallback_reason')}"
    )
    print(
        "- kv_sizing_final_summary="
        f"records={trace['kv_sizing_record_count']} "
        f"mode={summary.get('kv_sizing_mode')} "
        f"budget_bytes={summary.get('kv_sizing_budget_bytes')} "
        f"derived_budget_bytes={summary.get('kv_sizing_derived_budget_bytes')} "
        f"budget_blocks={summary.get('kv_sizing_budget_blocks')} "
        f"bytes_per_block={summary.get('kv_sizing_bytes_per_block')} "
        f"original_blocks={summary.get('kv_sizing_original_num_gpu_blocks')} "
        f"final_blocks={summary.get('kv_sizing_final_num_gpu_blocks')} "
        f"applied={summary.get('kv_sizing_applied')} "
        f"report_only={summary.get('kv_sizing_report_only')} "
        f"max_blocks={summary.get('kv_sizing_max_blocks')} "
        f"fallback={summary.get('kv_sizing_fallback_reason')}"
    )
    print(f"- final_summary={summary}")
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
