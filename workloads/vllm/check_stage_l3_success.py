#!/usr/bin/env python3
"""Validate Stage L3 scratch residency reserve reporting."""

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
            "Check Stage L3 scratch residency reserve reporting. The default "
            "self-test validates the L3 trace contract without a GPU. Use "
            "--gpu-run to execute vLLM and fold Stage G scratch telemetry into "
            "a Stage L3 residency summary."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage L3 run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage L residency JSONL.")
    parser.add_argument("--allocator-log", help="Existing or new allocator trace log.")
    parser.add_argument(
        "--scratch-summary-json",
        help="Existing or new summarize_gap_watch_metrics.py JSON output.",
    )
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--runner-log", help="Existing or new runner log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="enforce",
        help="Stage L residency mode.",
    )
    parser.add_argument(
        "--scratch-pool-mode",
        choices=("trace_only", "enforce"),
        default="enforce",
        help="Stage G scratch pool mode for GPU probes.",
    )
    parser.add_argument("--global-gpu-bytes", type=int, default=3072)
    parser.add_argument("--weight-gpu-bytes", type=int, default=0)
    parser.add_argument("--kv-gpu-bytes", type=int, default=0)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=1048576)
    parser.add_argument("--safety-margin-bytes", type=int, default=0)
    parser.add_argument("--scratch-reserved-bytes", type=int, default=1048576)
    parser.add_argument("--scratch-margin-bytes", type=int, default=0)
    parser.add_argument(
        "--scratch-pool-budget-bytes",
        type=int,
        default=1,
        help="Small default intentionally exercises Stage G over-budget reporting.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate existing trace/summary files; use --trace-jsonl or --run-dir.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_l3_success_check_{stamp}")


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    return run_dir / "vllm_uvm_pool_residency_stage_l3.jsonl"


def resolve_scratch_summary(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.scratch_summary_json:
        return Path(args.scratch_summary_json)
    return run_dir / "vllm_stage_l3_scratch_metrics_summary.json"


def resolve_allocator_log(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.allocator_log:
        return Path(args.allocator_log)
    return run_dir / "vllm_uvm_allocator_trace_stage_l3.log"


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


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return {}


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


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    parsed = optional_int(value, -1)
    if parsed in (0, 1):
        return bool(parsed)
    return None


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l3",
            "VLLM_UVM_RESIDENCY_MODE": args.mode,
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES": str(args.global_gpu_bytes),
            "VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES": str(args.weight_gpu_bytes),
            "VLLM_UVM_RESIDENCY_KV_GPU_BYTES": str(args.kv_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES": str(args.scratch_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES": str(
                args.safety_margin_bytes
            ),
            "VLLM_UVM_RESIDENCY_SCRATCH_RESERVED_BYTES": str(
                args.scratch_reserved_bytes
            ),
            "VLLM_UVM_RESIDENCY_SCRATCH_MARGIN_BYTES": str(
                args.scratch_margin_bytes
            ),
        }
    )
    return env


def write_stage_l3_scratch_summary(
    *,
    args: argparse.Namespace,
    trace_jsonl: Path,
    scratch_summary: dict[str, Any],
) -> None:
    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    code = r'''
import os

from vllm.device_allocator.uvm_pool_residency import (
    record_uvm_residency_scratch_summary,
)

record_uvm_residency_scratch_summary(
    scratch_live_bytes=int(os.environ.get("STAGE_L3_SCRATCH_LIVE_BYTES", "0")),
    scratch_peak_bytes=int(os.environ.get("STAGE_L3_SCRATCH_PEAK_BYTES", "0")),
    scratch_reserved_bytes=int(os.environ.get("STAGE_L3_SCRATCH_RESERVED_BYTES", "0")),
    scratch_margin_bytes=int(os.environ.get("STAGE_L3_SCRATCH_MARGIN_BYTES", "0")),
    scratch_pool_enabled=os.environ.get("STAGE_L3_SCRATCH_POOL_ENABLED", "0") == "1",
    scratch_pool_mode=os.environ.get("STAGE_L3_SCRATCH_POOL_MODE") or None,
    scratch_trace_records=int(os.environ.get("STAGE_L3_SCRATCH_TRACE_RECORDS", "0")),
    scratch_eligible_records=int(os.environ.get("STAGE_L3_SCRATCH_ELIGIBLE_RECORDS", "0")),
    scratch_device_direct_records=int(os.environ.get("STAGE_L3_SCRATCH_DEVICE_DIRECT_RECORDS", "0")),
    scratch_budget_over_records=int(os.environ.get("STAGE_L3_SCRATCH_BUDGET_OVER_RECORDS", "0")),
    scratch_budget_reject_records=int(os.environ.get("STAGE_L3_SCRATCH_BUDGET_REJECT_RECORDS", "0")),
    scratch_eviction_records=int(os.environ.get("STAGE_L3_SCRATCH_EVICTION_RECORDS", "0")),
    metadata={
        "source": "stage_g_scratch_pool",
        "stage": "stage_l3",
        "summary_json": os.environ.get("STAGE_L3_SCRATCH_SUMMARY_JSON"),
        "semantics": "scratch_reserve_report_no_eviction",
    },
)
'''
    pool_enabled = optional_bool(scratch_summary.get("scratch_pool_enabled"))
    env = residency_env(args, trace_jsonl)
    env.update(
        {
            "STAGE_L3_SCRATCH_LIVE_BYTES": str(
                optional_int(scratch_summary.get("scratch_pool_live_bytes"))
            ),
            "STAGE_L3_SCRATCH_PEAK_BYTES": str(
                optional_int(scratch_summary.get("scratch_pool_peak_live_bytes"))
            ),
            "STAGE_L3_SCRATCH_RESERVED_BYTES": str(args.scratch_reserved_bytes),
            "STAGE_L3_SCRATCH_MARGIN_BYTES": str(args.scratch_margin_bytes),
            "STAGE_L3_SCRATCH_POOL_ENABLED": "1" if pool_enabled else "0",
            "STAGE_L3_SCRATCH_POOL_MODE": str(
                scratch_summary.get("scratch_pool_mode") or args.scratch_pool_mode
            ),
            "STAGE_L3_SCRATCH_TRACE_RECORDS": str(
                optional_int(scratch_summary.get("scratch_pool_trace_records"))
            ),
            "STAGE_L3_SCRATCH_ELIGIBLE_RECORDS": str(
                optional_int(scratch_summary.get("scratch_pool_eligible_records"))
            ),
            "STAGE_L3_SCRATCH_DEVICE_DIRECT_RECORDS": str(
                optional_int(scratch_summary.get("scratch_pool_device_direct_records"))
            ),
            "STAGE_L3_SCRATCH_BUDGET_OVER_RECORDS": str(
                optional_int(scratch_summary.get("scratch_pool_budget_over_records"))
            ),
            "STAGE_L3_SCRATCH_BUDGET_REJECT_RECORDS": str(
                optional_int(scratch_summary.get("scratch_pool_budget_reject_records"))
            ),
            "STAGE_L3_SCRATCH_EVICTION_RECORDS": "0",
            "STAGE_L3_SCRATCH_SUMMARY_JSON": str(
                scratch_summary.get("_summary_json_path") or ""
            ),
        }
    )
    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, None, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l3_self_test_runner.log"
    )
    scratch_summary_json = resolve_scratch_summary(args, run_dir)
    try:
        trace_jsonl.unlink()
    except FileNotFoundError:
        pass
    scratch_summary = {
        "_summary_json_path": str(scratch_summary_json),
        "scratch_pool_enabled": True,
        "scratch_pool_mode": args.scratch_pool_mode,
        "scratch_pool_trace_records": 3,
        "scratch_pool_eligible_records": 2,
        "scratch_pool_device_direct_records": 1,
        "scratch_pool_live_bytes": 0,
        "scratch_pool_peak_live_bytes": args.scratch_reserved_bytes,
        "scratch_pool_budget_over_records": 1,
        "scratch_pool_budget_reject_records": (
            1 if args.scratch_pool_mode == "enforce" else 0
        ),
        "scratch_pool_reason_counts": {
            "scratch_pool_budget_exceeded_fallback_managed": (
                1 if args.scratch_pool_mode == "enforce" else 0
            )
        },
    }
    scratch_summary_json.write_text(
        json.dumps(scratch_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("===========================================================")
    print(" Stage L3 Success Check: running scratch reserve self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Scratch summary: {scratch_summary_json}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        handle.write("stage_l3_self_test: writing scratch summary\n")
    write_stage_l3_scratch_summary(
        args=args,
        trace_jsonl=trace_jsonl,
        scratch_summary=scratch_summary,
    )
    return trace_jsonl, runner_log, None, scratch_summary_json


def run_gpu_probe(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    allocator_log = resolve_allocator_log(args, run_dir)
    scratch_summary_json = resolve_scratch_summary(args, run_dir)
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l3.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l3_gpu_runner.log"
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
        str(run_dir / "uvm_kv_fault_stats_stage_l3.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_l3.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_l3.log"),
        "--bench-log",
        str(bench_log),
        "--gap-watch-metrics-summary-json",
        str(scratch_summary_json),
        "--uvm-pool-registry-enable",
        "1",
        "--uvm-device-direct-enable",
        "1",
        "--uvm-device-direct-min-bytes",
        "4096",
        "--uvm-device-direct-max-bytes",
        "1048576",
        "--uvm-device-direct-max-total-bytes",
        str(max(args.scratch_pool_budget_bytes, 1)),
        "--uvm-scratch-pool-enable",
        "1",
        "--uvm-scratch-pool-budget-bytes",
        str(args.scratch_pool_budget_bytes),
        "--uvm-scratch-pool-mode",
        args.scratch_pool_mode,
        "--uvm-scratch-pool-target-phases",
        "enabled:attention,enabled:moe,enabled:model_forward",
        "--uvm-residency-enable",
        "1",
        "--uvm-residency-stage",
        "stage_l3",
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
        "--uvm-residency-scratch-reserved-bytes",
        str(args.scratch_reserved_bytes),
        "--uvm-residency-scratch-margin-bytes",
        str(args.scratch_margin_bytes),
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L3 Success Check: running GPU scratch reserve probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Allocator trace: {allocator_log}")
    print(f" Scratch summary: {scratch_summary_json}")
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
        raise SystemExit(f"Stage L3 GPU probe failed with exit code {rc}; log={runner_log}")
    scratch_summary = read_json(scratch_summary_json)
    scratch_summary["_summary_json_path"] = str(scratch_summary_json)
    write_stage_l3_scratch_summary(
        args=args,
        trace_jsonl=trace_jsonl,
        scratch_summary=scratch_summary,
    )
    return trace_jsonl, runner_log, bench_log, scratch_summary_json


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    scratch_summary_records = 0
    scratch_action_records = 0
    scratch_eviction_records = 0
    kv_action_records = 0
    weight_offload_records = 0

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        pool = str(record.get("pool") or "")
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        if action == "residency_scratch_summary":
            scratch_summary_records += 1
        if pool == "scratch":
            scratch_action_records += 1
            if "evict" in action:
                scratch_eviction_records += 1
        if pool == "kv":
            kv_action_records += 1
        if pool == "weights" and "offload" in action:
            weight_offload_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "config": config,
        "summary": summary,
        "scratch_summary_records": scratch_summary_records,
        "scratch_action_records": scratch_action_records,
        "scratch_eviction_records": scratch_eviction_records,
        "kv_action_records": kv_action_records,
        "weight_offload_records": weight_offload_records,
    }


def validate(
    *,
    trace_jsonl: Path,
    runner_log: Path | None,
    bench_log: Path | None,
    scratch_summary_json: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace_summary = summarize_trace(records)
    config = trace_summary["config"]
    summary = trace_summary["summary"]
    scratch_summary = read_json(scratch_summary_json)
    failed_requests = parse_failed_requests(bench_log)
    scratch_budget = summary.get("scratch_gpu_budget_bytes")
    scratch_peak = summary.get("scratch_peak_bytes")
    scratch_reserved = summary.get("scratch_reserved_bytes")
    scratch_margin = summary.get("scratch_margin_bytes")
    scratch_over_budget = optional_int(summary.get("scratch_budget_over_records"))
    scratch_rejects = optional_int(summary.get("scratch_budget_reject_records"))
    scratch_eligible_records = optional_int(summary.get("scratch_eligible_records"))
    scratch_reason_counts = scratch_summary.get("scratch_pool_reason_counts")
    if not isinstance(scratch_reason_counts, dict):
        scratch_reason_counts = {}
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
            "stage_l3_config_present",
            config.get("stage") == "stage_l3"
            or config.get("stage_label") == "stage_l3",
            f"config={config}",
        ),
        check(
            "scratch_summary_present",
            trace_summary["scratch_summary_records"] > 0
            and summary.get("scratch_stage_g_summary_present") is True,
            (
                "scratch_summary_records="
                f"{trace_summary['scratch_summary_records']} "
                "scratch_stage_g_summary_present="
                f"{summary.get('scratch_stage_g_summary_present')}"
            ),
        ),
        check(
            "scratch_peak_bytes_present",
            isinstance(scratch_peak, int),
            f"scratch_peak_bytes={scratch_peak}",
        ),
        check(
            "scratch_budget_bytes_present",
            scratch_budget == args.scratch_gpu_bytes,
            f"scratch_budget={scratch_budget} expected={args.scratch_gpu_bytes}",
        ),
        check(
            "scratch_reserved_bytes_present",
            scratch_reserved == args.scratch_reserved_bytes,
            (
                "scratch_reserved_bytes="
                f"{scratch_reserved} expected={args.scratch_reserved_bytes}"
            ),
        ),
        check(
            "scratch_margin_bytes_present",
            scratch_margin == args.scratch_margin_bytes,
            (
                "scratch_margin_bytes="
                f"{scratch_margin} expected={args.scratch_margin_bytes}"
            ),
        ),
        check(
            "scratch_pool_enabled_visible",
            summary.get("scratch_pool_enabled") is True,
            f"scratch_pool_enabled={summary.get('scratch_pool_enabled')}",
        ),
        check(
            "scratch_pool_mode_visible",
            summary.get("scratch_pool_mode") == args.scratch_pool_mode,
            (
                "scratch_pool_mode="
                f"{summary.get('scratch_pool_mode')} expected={args.scratch_pool_mode}"
            ),
        ),
        check(
            "scratch_eviction_records_zero",
            trace_summary["scratch_eviction_records"] == 0
            and summary.get("scratch_eviction_records") == 0,
            (
                "trace_scratch_evictions="
                f"{trace_summary['scratch_eviction_records']} "
                "summary_scratch_evictions="
                f"{summary.get('scratch_eviction_records')}"
            ),
        ),
        check(
            "kv_action_records_zero",
            trace_summary["kv_action_records"] == 0,
            f"kv_action_records={trace_summary['kv_action_records']}",
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
    if args.gpu_run or args.skip_run:
        checks.extend(
            [
                check(
                    "scratch_stage_g_json_present",
                    bool(scratch_summary),
                    f"scratch_summary_json={scratch_summary_json}",
                ),
                check(
                    "scratch_trace_records_present",
                    optional_int(summary.get("scratch_trace_records")) > 0,
                    (
                        "scratch_trace_records="
                        f"{summary.get('scratch_trace_records')}"
                    ),
                ),
            ]
        )
    if args.scratch_pool_budget_bytes > 0 and args.scratch_pool_mode == "enforce":
        checks.append(
            check(
                "scratch_budget_gate_exercised_or_explained",
                (
                    scratch_over_budget > 0
                    or scratch_rejects > 0
                    or (
                        scratch_eligible_records == 0
                        and bool(scratch_reason_counts)
                    )
                ),
                (
                    f"scratch_eligible_records={scratch_eligible_records} "
                    f"scratch_budget_over_records={scratch_over_budget} "
                    f"scratch_budget_reject_records={scratch_rejects} "
                    f"scratch_reason_counts={scratch_reason_counts}"
                ),
            )
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
        trace_jsonl.parent / "stage_l3_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
        "bench_log": str(bench_log) if bench_log else None,
        "scratch_summary_json": str(scratch_summary_json) if scratch_summary_json else None,
        "trace_summary": trace_summary,
        "scratch_summary": scratch_summary,
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    trace = result["trace_summary"]
    summary = trace["summary"]
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L3 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print(f" Scratch summary: {result['scratch_summary_json']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- scratch="
        f"summary_records={trace['scratch_summary_records']} "
        f"peak_bytes={summary.get('scratch_peak_bytes')} "
        f"live_bytes={summary.get('scratch_live_bytes')} "
        f"budget_bytes={summary.get('scratch_gpu_budget_bytes')} "
        f"reserved_bytes={summary.get('scratch_reserved_bytes')} "
        f"margin_bytes={summary.get('scratch_margin_bytes')} "
        f"trace_records={summary.get('scratch_trace_records')} "
        f"device_direct_records={summary.get('scratch_device_direct_records')} "
        f"over_budget_records={summary.get('scratch_budget_over_records')} "
        f"reject_records={summary.get('scratch_budget_reject_records')} "
        f"evictions={summary.get('scratch_eviction_records')}"
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
    scratch_summary_json: Path | None = (
        Path(args.scratch_summary_json) if args.scratch_summary_json else None
    )
    if args.skip_run:
        trace_jsonl = resolve_trace(args, run_dir)
        runner_log = Path(args.runner_log) if args.runner_log else None
        scratch_summary_json = resolve_scratch_summary(args, run_dir)
    elif args.gpu_run:
        trace_jsonl, runner_log, bench_log, scratch_summary_json = run_gpu_probe(
            args, run_dir
        )
    else:
        trace_jsonl, runner_log, bench_log, scratch_summary_json = run_self_test(
            args, run_dir
        )
    result = validate(
        trace_jsonl=trace_jsonl,
        runner_log=runner_log,
        bench_log=bench_log,
        scratch_summary_json=scratch_summary_json,
        args=args,
    )
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
