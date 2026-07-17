#!/usr/bin/env python3
"""Run Stage L end-to-end evaluation suites and summarize results.

This script is intentionally an orchestration layer.  It delegates correctness
checks to the stage-specific check_stage_l*.py scripts, keeps each run in a
stable output directory, then extracts a common set of metrics into JSON, CSV,
and Markdown reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

CHECK_SCRIPTS = {
    "l0": "check_stage_l_success.py",
    "l1": "check_stage_l1_success.py",
    "l2": "check_stage_l2_success.py",
    "l3": "check_stage_l3_success.py",
    "l5": "check_stage_l5_success.py",
    "l6": "check_stage_l6_success.py",
    "l7": "check_stage_l7_success.py",
    "l7_cross": "check_stage_l7_cross_pool_success.py",
}

PY_COMPILE_FILES = [
    "check_stage_l_success.py",
    "check_stage_l1_success.py",
    "check_stage_l2_success.py",
    "check_stage_l3_success.py",
    "check_stage_l5_success.py",
    "check_stage_l6_success.py",
    "check_stage_l7_success.py",
    "check_stage_l7_cross_pool_success.py",
    "vllm/vllm/device_allocator/uvm_pool_residency.py",
    "vllm/vllm/device_allocator/uvm_pool_coordinator.py",
    "vllm/vllm/v1/core/uvm_kv_runtime_policy.py",
    "vllm/vllm/v1/core/kv_cache_utils.py",
    "vllm/vllm/v1/worker/gpu_model_runner.py",
    "vllm/vllm/model_executor/layers/fused_moe/layer.py",
]


@dataclass
class Variant:
    suite: str
    name: str
    cmd: list[str]
    run_dir: Path
    output_json: Path | None = None
    bench_log: Path | None = None
    kind: str = "checker"
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full Stage L implementation.  By default only the "
            "fast smoke suite is run; GPU suites require --gpu-run."
        )
    )
    parser.add_argument(
        "--suite",
        action="append",
        choices=(
            "smoke",
            "gpu-functional",
            "performance-baseline",
            "kv-sweep",
            "coordinator-sweep",
            "cross-pool-sweep",
            "all-gpu",
        ),
        help="Suite to run. Can be repeated. Default: smoke.",
    )
    parser.add_argument("--run-root", type=Path, help="Evaluation output root.")
    parser.add_argument("--gpu-run", action="store_true", help="Allow GPU suites.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--prompts-list", default="2,4,8")
    parser.add_argument("--output-len-list", default="128,256,512")
    parser.add_argument("--runtime-budget-blocks-list", default="1,2,4")
    parser.add_argument("--max-offload-blocks-list", default="2,4,8")
    parser.add_argument("--global-bytes-per-step-list", default="512MiB,1GiB,2GiB")
    parser.add_argument("--kv-bytes-per-step-list", default="256MiB,512MiB,1GiB")
    parser.add_argument("--weight-bytes-per-step-list", default="256MiB,512MiB,1GiB")
    parser.add_argument(
        "--gpu-stages",
        default="l1,l2,l3,l5,l6,l7,l7_cross",
        help="Comma-separated stages for gpu-functional.",
    )
    parser.add_argument(
        "--max-sweep-runs",
        type=int,
        default=0,
        help="Cap generated variants per sweep suite. 0 means no cap.",
    )
    return parser.parse_args()


def default_run_root() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l_full_eval_{stamp}")


def parse_csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in parse_csv_items(raw)]


def parse_size(value: str) -> int:
    text = value.strip()
    match = re.fullmatch(r"(\d+)([KMG]i?B?|[kmg]i?B?)?", text)
    if not match:
        raise ValueError(f"invalid size value: {value!r}")
    number = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix in ("k", "kb", "kib"):
        return number * 1024
    if suffix in ("m", "mb", "mib"):
        return number * 1024**2
    if suffix in ("g", "gb", "gib"):
        return number * 1024**3
    return number


def parse_size_list(raw: str) -> list[int]:
    return [parse_size(item) for item in parse_csv_items(raw)]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", text).strip("_")


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def checker_variant(
    *,
    suite: str,
    stage: str,
    run_root: Path,
    args: argparse.Namespace,
    extra: list[str] | None = None,
    name_suffix: str = "",
    gpu: bool = False,
) -> Variant:
    script = CHECK_SCRIPTS[stage]
    name = safe_name(f"{stage}{name_suffix}")
    run_dir = run_root / suite / name
    output_json = run_dir / f"{name}_check.json"
    cmd = [f"./{script}", "--run-dir", str(run_dir), "--output-json", str(output_json)]
    if gpu:
        cmd.extend(
            [
                "--gpu-run",
                "--model",
                args.model,
                "--prompts",
                str(args.prompts),
                "--output-len",
                str(args.output_len),
                "--request-rate",
                str(args.request_rate),
            ]
        )
    if extra:
        cmd.extend(extra)
    return Variant(
        suite=suite,
        name=name,
        cmd=cmd,
        run_dir=run_dir,
        output_json=output_json,
        metadata={"stage": stage, "gpu": gpu},
    )


def baseline_variant(
    *,
    suite: str,
    name: str,
    run_root: Path,
    args: argparse.Namespace,
    prompts: int,
    output_len: int,
) -> Variant:
    run_dir = run_root / suite / safe_name(name)
    bench_log = run_dir / "vllm_bench_baseline.log"
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model",
        args.model,
        "--mode",
        "trace",
        "--allocator-log",
        str(run_dir / "vllm_uvm_allocator_trace_baseline.log"),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_baseline.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_baseline.log"),
        "--server-log",
        str(run_dir / "vllm_server_baseline.log"),
        "--bench-log",
        str(bench_log),
        "--prompts",
        str(prompts),
        "--output-len",
        str(output_len),
        "--request-rate",
        str(args.request_rate),
    ]
    return Variant(
        suite=suite,
        name=safe_name(name),
        cmd=cmd,
        run_dir=run_dir,
        bench_log=bench_log,
        kind="baseline",
        metadata={"prompts": prompts, "output_len": output_len},
    )


def build_smoke_variants(run_root: Path, args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    py_run_dir = run_root / "smoke" / "py_compile"
    variants.append(
        Variant(
            suite="smoke",
            name="py_compile",
            cmd=["python3", "-m", "py_compile", *PY_COMPILE_FILES],
            run_dir=py_run_dir,
            kind="static",
        )
    )
    for stage in ("l0", "l1", "l2", "l3", "l5", "l6"):
        variants.append(checker_variant(suite="smoke", stage=stage, run_root=run_root, args=args))
    for stage in ("l7", "l7_cross"):
        script = CHECK_SCRIPTS[stage]
        variants.append(
            Variant(
                suite="smoke",
                name=f"{stage}_help",
                cmd=[f"./{script}", "--help"],
                run_dir=run_root / "smoke" / f"{stage}_help",
                kind="static",
                metadata={"stage": stage},
            )
        )
    return variants


def build_gpu_functional_variants(
    run_root: Path, args: argparse.Namespace
) -> list[Variant]:
    variants: list[Variant] = []
    for stage in parse_csv_items(args.gpu_stages):
        if stage not in CHECK_SCRIPTS:
            raise SystemExit(f"Unknown stage in --gpu-stages: {stage}")
        extra: list[str] = []
        if stage == "l2":
            extra = ["--mode", "enforce"]
        elif stage == "l3":
            extra = ["--mode", "enforce", "--scratch-pool-mode", "enforce"]
        elif stage == "l6":
            extra = [
                "--mode",
                "trace_only",
                "--data-offload-mode",
                "prefetch_cpu",
                "--prefetch-back-mode",
                "prefetch",
                "--strict-issued",
            ]
        elif stage == "l7":
            extra = ["--mode", "trace_only", "--coordinator-mode", "trace_only"]
        elif stage == "l7_cross":
            extra = [
                "--mode",
                "trace_only",
                "--coordinator-mode",
                "trace_only",
                "--priority",
                "kv,scratch,weights",
            ]
        variants.append(
            checker_variant(
                suite="gpu-functional",
                stage=stage,
                run_root=run_root,
                args=args,
                extra=extra,
                gpu=True,
            )
        )
    return variants


def build_performance_baseline_variants(
    run_root: Path, args: argparse.Namespace
) -> list[Variant]:
    variants = [
        baseline_variant(
            suite="performance-baseline",
            name=f"baseline_p{args.prompts}_o{args.output_len}",
            run_root=run_root,
            args=args,
            prompts=args.prompts,
            output_len=args.output_len,
        ),
        checker_variant(
            suite="performance-baseline",
            stage="l5",
            run_root=run_root,
            args=args,
            gpu=True,
            name_suffix=f"_p{args.prompts}_o{args.output_len}",
            extra=["--mode", "trace_only"],
        ),
        checker_variant(
            suite="performance-baseline",
            stage="l6",
            run_root=run_root,
            args=args,
            gpu=True,
            name_suffix=f"_p{args.prompts}_o{args.output_len}",
            extra=[
                "--mode",
                "trace_only",
                "--data-offload-mode",
                "prefetch_cpu",
                "--prefetch-back-mode",
                "prefetch",
                "--strict-issued",
            ],
        ),
        checker_variant(
            suite="performance-baseline",
            stage="l7",
            run_root=run_root,
            args=args,
            gpu=True,
            name_suffix=f"_p{args.prompts}_o{args.output_len}",
            extra=["--mode", "trace_only", "--coordinator-mode", "trace_only"],
        ),
        checker_variant(
            suite="performance-baseline",
            stage="l7_cross",
            run_root=run_root,
            args=args,
            gpu=True,
            name_suffix=f"_p{args.prompts}_o{args.output_len}",
            extra=[
                "--mode",
                "trace_only",
                "--coordinator-mode",
                "trace_only",
                "--priority",
                "kv,scratch,weights",
            ],
        ),
    ]
    return variants


def maybe_cap(variants: list[Variant], limit: int) -> list[Variant]:
    if limit > 0:
        return variants[:limit]
    return variants


def build_kv_sweep_variants(run_root: Path, args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    for budget_blocks in parse_int_list(args.runtime_budget_blocks_list):
        for max_blocks in parse_int_list(args.max_offload_blocks_list):
            suffix = f"_bb{budget_blocks}_off{max_blocks}"
            variants.append(
                checker_variant(
                    suite="kv-sweep",
                    stage="l6",
                    run_root=run_root,
                    args=args,
                    gpu=True,
                    name_suffix=suffix,
                    extra=[
                        "--mode",
                        "trace_only",
                        "--data-offload-mode",
                        "prefetch_cpu",
                        "--prefetch-back-mode",
                        "prefetch",
                        "--strict-issued",
                        "--runtime-budget-blocks",
                        str(budget_blocks),
                        "--max-offload-blocks",
                        str(max_blocks),
                    ],
                )
            )
    return maybe_cap(variants, args.max_sweep_runs)


def build_coordinator_sweep_variants(
    run_root: Path, args: argparse.Namespace
) -> list[Variant]:
    variants: list[Variant] = []
    for global_bytes in parse_size_list(args.global_bytes_per_step_list):
        for kv_bytes in parse_size_list(args.kv_bytes_per_step_list):
            suffix = f"_g{global_bytes}_kv{kv_bytes}"
            variants.append(
                checker_variant(
                    suite="coordinator-sweep",
                    stage="l7",
                    run_root=run_root,
                    args=args,
                    gpu=True,
                    name_suffix=suffix,
                    extra=[
                        "--mode",
                        "trace_only",
                        "--coordinator-mode",
                        "trace_only",
                        "--global-bytes-per-step",
                        str(global_bytes),
                        "--kv-bytes-per-step",
                        str(kv_bytes),
                    ],
                )
            )
    return maybe_cap(variants, args.max_sweep_runs)


def build_cross_pool_sweep_variants(
    run_root: Path, args: argparse.Namespace
) -> list[Variant]:
    variants: list[Variant] = []
    for global_bytes in parse_size_list(args.global_bytes_per_step_list):
        for kv_bytes in parse_size_list(args.kv_bytes_per_step_list):
            for weight_bytes in parse_size_list(args.weight_bytes_per_step_list):
                suffix = f"_g{global_bytes}_kv{kv_bytes}_w{weight_bytes}"
                variants.append(
                    checker_variant(
                        suite="cross-pool-sweep",
                        stage="l7_cross",
                        run_root=run_root,
                        args=args,
                        gpu=True,
                        name_suffix=suffix,
                        extra=[
                            "--mode",
                            "trace_only",
                            "--coordinator-mode",
                            "trace_only",
                            "--priority",
                            "kv,scratch,weights",
                            "--global-bytes-per-step",
                            str(global_bytes),
                            "--kv-bytes-per-step",
                            str(kv_bytes),
                            "--weight-bytes-per-step",
                            str(weight_bytes),
                        ],
                    )
                )
    return maybe_cap(variants, args.max_sweep_runs)


def expand_repeats(variants: list[Variant], repeat: int) -> list[Variant]:
    if repeat <= 1:
        return variants
    repeated: list[Variant] = []
    for variant in variants:
        for idx in range(1, repeat + 1):
            run_dir = variant.run_dir.parent / f"{variant.run_dir.name}_rep{idx}"
            output_json = (
                run_dir / variant.output_json.name if variant.output_json else None
            )
            bench_log = run_dir / variant.bench_log.name if variant.bench_log else None
            cmd = list(variant.cmd)
            if variant.output_json:
                replace_arg(cmd, "--output-json", str(output_json))
            if "--run-dir" in cmd:
                replace_arg(cmd, "--run-dir", str(run_dir))
            if variant.bench_log:
                replace_arg(cmd, "--bench-log", str(bench_log))
                replace_path_arg(cmd, str(variant.run_dir), str(run_dir))
            repeated.append(
                Variant(
                    suite=variant.suite,
                    name=f"{variant.name}_rep{idx}",
                    cmd=cmd,
                    run_dir=run_dir,
                    output_json=output_json,
                    bench_log=bench_log,
                    kind=variant.kind,
                    metadata={**variant.metadata, "repeat": idx},
                )
            )
    return repeated


def replace_arg(cmd: list[str], flag: str, value: str) -> None:
    try:
        idx = cmd.index(flag)
    except ValueError:
        return
    if idx + 1 < len(cmd):
        cmd[idx + 1] = value


def replace_path_arg(cmd: list[str], old: str, new: str) -> None:
    for idx, item in enumerate(cmd):
        if old in item:
            cmd[idx] = item.replace(old, new)


def build_variants(run_root: Path, args: argparse.Namespace) -> list[Variant]:
    suites = args.suite or ["smoke"]
    if "all-gpu" in suites:
        suites = [
            suite for suite in suites if suite != "all-gpu"
        ] + [
            "gpu-functional",
            "performance-baseline",
            "kv-sweep",
            "coordinator-sweep",
            "cross-pool-sweep",
        ]
    variants: list[Variant] = []
    for suite in suites:
        if suite == "smoke":
            variants.extend(build_smoke_variants(run_root, args))
        elif suite == "gpu-functional":
            require_gpu(args, suite)
            variants.extend(build_gpu_functional_variants(run_root, args))
        elif suite == "performance-baseline":
            require_gpu(args, suite)
            variants.extend(build_performance_baseline_variants(run_root, args))
        elif suite == "kv-sweep":
            require_gpu(args, suite)
            variants.extend(build_kv_sweep_variants(run_root, args))
        elif suite == "coordinator-sweep":
            require_gpu(args, suite)
            variants.extend(build_coordinator_sweep_variants(run_root, args))
        elif suite == "cross-pool-sweep":
            require_gpu(args, suite)
            variants.extend(build_cross_pool_sweep_variants(run_root, args))
        else:
            raise SystemExit(f"Unhandled suite: {suite}")
    return expand_repeats(variants, max(args.repeat, 1))


def require_gpu(args: argparse.Namespace, suite: str) -> None:
    if not args.gpu_run and not args.dry_run:
        raise SystemExit(f"{suite} requires --gpu-run or --dry-run.")


def run_variant(variant: Variant, *, dry_run: bool = False) -> dict[str, Any]:
    variant.run_dir.mkdir(parents=True, exist_ok=True)
    eval_log = variant.run_dir / "stage_l_full_eval_command.log"
    started = time.time()
    row: dict[str, Any] = {
        "suite": variant.suite,
        "variant": variant.name,
        "kind": variant.kind,
        "run_dir": str(variant.run_dir),
        "command": quote_cmd(variant.cmd),
        "dry_run": dry_run,
        **variant.metadata,
    }
    if variant.output_json:
        row["check_json"] = str(variant.output_json)
    if dry_run:
        row.update({"returncode": None, "passed": None, "duration_sec": 0.0})
        return row

    with eval_log.open("w", encoding="utf-8") as handle:
        handle.write(f"$ {quote_cmd(variant.cmd)}\n")
        handle.flush()
        process = subprocess.Popen(
            variant.cmd,
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
    duration = time.time() - started
    row.update({"returncode": rc, "duration_sec": round(duration, 3)})
    row.update(extract_metrics(variant, eval_log, rc))
    return row


def extract_metrics(variant: Variant, eval_log: Path, rc: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {"passed": rc == 0}
    check_data: dict[str, Any] = {}
    if variant.output_json and variant.output_json.is_file():
        try:
            check_data = json.loads(variant.output_json.read_text(encoding="utf-8"))
            metrics["passed"] = bool(check_data.get("passed")) and rc == 0
        except json.JSONDecodeError:
            metrics["check_json_parse_error"] = True
            metrics["passed"] = False
    elif variant.kind == "baseline":
        metrics["passed"] = rc == 0

    paths = check_data.get("paths") or {}
    bench_log = resolve_bench_log(variant, paths)
    if bench_log and bench_log.is_file():
        metrics.update(parse_bench_metrics(bench_log.read_text(encoding="utf-8", errors="replace")))
    if eval_log.is_file():
        metrics.update(parse_fault_metrics(eval_log.read_text(encoding="utf-8", errors="replace")))

    metrics.update(extract_check_json_metrics(check_data))
    if variant.kind == "baseline":
        metrics["passed"] = metrics.get("failed_requests") in (None, 0) and rc == 0
    return metrics


def resolve_bench_log(variant: Variant, paths: dict[str, Any]) -> Path | None:
    if variant.bench_log:
        return variant.bench_log
    raw = paths.get("bench")
    if raw:
        return Path(raw)
    return None


def parse_bench_metrics(text: str) -> dict[str, Any]:
    patterns = {
        "successful_requests": r"Successful requests:\s+(\d+)",
        "failed_requests": r"Failed requests:\s+(\d+)",
        "benchmark_duration_s": r"Benchmark duration \(s\):\s+([0-9.]+)",
        "request_throughput_req_s": r"Request throughput \(req/s\):\s+([0-9.]+)",
        "output_token_throughput_tok_s": r"Output token throughput \(tok/s\):\s+([0-9.]+)",
        "total_token_throughput_tok_s": r"Total Token throughput \(tok/s\):\s+([0-9.]+)",
        "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
        "p99_ttft_ms": r"P99 TTFT \(ms\):\s+([0-9.]+)",
        "mean_tpot_ms": r"Mean TPOT \(ms\):\s+([0-9.]+)",
        "p99_tpot_ms": r"P99 TPOT \(ms\):\s+([0-9.]+)",
        "mean_itl_ms": r"Mean ITL \(ms\):\s+([0-9.]+)",
        "p99_itl_ms": r"P99 ITL \(ms\):\s+([0-9.]+)",
    }
    metrics: dict[str, Any] = {}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if not matches:
            continue
        value = matches[-1]
        metrics[key] = int(value) if value.isdigit() else float(value)
    return metrics


def parse_fault_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "delta_faults",
        "delta_duplicates",
        "delta_after_dedup",
        "delta_kv_faults",
        "delta_kv_duplicates",
        "delta_kv_after_dedup",
    ):
        matches = re.findall(rf"\b{key}=([0-9]+)", text)
        if matches:
            metrics[key] = int(matches[-1])
    return metrics


def extract_check_json_metrics(data: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {}
    metrics: dict[str, Any] = {}
    checks = data.get("checks") or []
    metrics["checks_total"] = len(checks)
    metrics["checks_failed"] = sum(1 for item in checks if not item.get("passed"))

    residency = data.get("residency_summary") or data.get("residency") or {}
    coordinator = data.get("coordinator_summary") or data.get("coordinator") or {}
    pointer_map = data.get("pointer_map_summary") or data.get("pointer_map") or {}
    weight_trace = data.get("weight_trace") or {}

    for key in (
        "offload_issued",
        "prefetch_issued",
        "offload_fallback",
        "unsafe_offloads",
        "records",
    ):
        if key in residency:
            metrics[f"residency_{key}"] = residency[key]
    if "summary" in residency and isinstance(residency["summary"], dict):
        summary = residency["summary"]
        for key in (
            "kv_data_offload_issued_records",
            "kv_prefetch_back_issued_records",
            "kv_data_offload_fallback_records",
            "kv_prefetch_back_fallback_records",
            "kv_unsafe_candidate_records",
            "state_updates",
            "real_action_records",
        ):
            if key in summary:
                metrics[key] = summary[key]

    for key in (
        "grants",
        "denies",
        "requests",
        "priority_records",
        "priority_safety_records",
        "kv_stage_l7_grants",
        "kv_stage_l7_denies",
    ):
        if key in coordinator:
            metrics[f"coordinator_{key}"] = coordinator[key]
    if "actions" in coordinator:
        metrics["coordinator_actions"] = json.dumps(
            coordinator["actions"], sort_keys=True
        )
    if "pools" in coordinator:
        metrics["coordinator_pools"] = json.dumps(
            coordinator["pools"], sort_keys=True
        )

    for key in ("mapped_blocks", "records"):
        if key in pointer_map:
            metrics[f"pointer_map_{key}"] = pointer_map[key]

    for key in ("records", "prefetch_candidates", "offload_candidates"):
        if key in weight_trace:
            metrics[f"weight_trace_{key}"] = weight_trace[key]
    return metrics


def write_reports(rows: list[dict[str, Any]], run_root: Path) -> None:
    summary_json = run_root / "stage_l_full_eval_summary.json"
    summary_csv = run_root / "stage_l_full_eval_summary.csv"
    report_md = run_root / "stage_l_full_eval_report.md"

    summary_json.write_text(
        json.dumps({"runs": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    fieldnames = sorted({key for row in rows for key in row})
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    passed = sum(1 for row in rows if row.get("passed") is True)
    failed = sum(1 for row in rows if row.get("passed") is False)
    dry = sum(1 for row in rows if row.get("dry_run") is True)
    lines = [
        "# Stage L Full Evaluation Report",
        "",
        f"- run_root: `{run_root}`",
        f"- total_runs: {len(rows)}",
        f"- passed: {passed}",
        f"- failed: {failed}",
        f"- dry_run: {dry}",
        "",
        "## Runs",
        "",
        "| suite | variant | passed | failed_requests | mean_tpot_ms | output_tok_s | kv_faults | offload | prefetch | run_dir |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {suite} | {variant} | {passed} | {failed_requests} | "
            "{mean_tpot_ms} | {output_token_throughput_tok_s} | "
            "{delta_kv_faults} | {offload} | {prefetch} | `{run_dir}` |".format(
                suite=row.get("suite", ""),
                variant=row.get("variant", ""),
                passed=row.get("passed", ""),
                failed_requests=row.get("failed_requests", ""),
                mean_tpot_ms=row.get("mean_tpot_ms", ""),
                output_token_throughput_tok_s=row.get(
                    "output_token_throughput_tok_s", ""
                ),
                delta_kv_faults=row.get("delta_kv_faults", ""),
                offload=(
                    row.get("residency_offload_issued")
                    or row.get("kv_data_offload_issued_records")
                    or ""
                ),
                prefetch=(
                    row.get("residency_prefetch_issued")
                    or row.get("kv_prefetch_back_issued_records")
                    or ""
                ),
                run_dir=row.get("run_dir", ""),
            )
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- JSON: `{summary_json}`",
            f"- CSV: `{summary_csv}`",
            f"- Markdown: `{report_md}`",
            "",
        ]
    )
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root or default_run_root()
    run_root.mkdir(parents=True, exist_ok=True)
    variants = build_variants(run_root, args)
    if not variants:
        raise SystemExit("No variants selected.")

    print("===========================================================")
    print(" Stage L Full Evaluation")
    print(f" Run root: {run_root}")
    print(f" Variants: {len(variants)}")
    print(f" Dry run: {args.dry_run}")
    print("===========================================================")

    rows: list[dict[str, Any]] = []
    for idx, variant in enumerate(variants, start=1):
        print("===========================================================")
        print(f" [{idx}/{len(variants)}] {variant.suite}/{variant.name}")
        print(f" command: {quote_cmd(variant.cmd)}")
        print("===========================================================")
        row = run_variant(variant, dry_run=args.dry_run)
        rows.append(row)
        write_reports(rows, run_root)
        if args.fail_fast and row.get("passed") is False:
            print(f"Fail-fast stopping after {variant.suite}/{variant.name}.")
            return 1

    write_reports(rows, run_root)
    failed = [row for row in rows if row.get("passed") is False]
    print("===========================================================")
    print(" Stage L Full Evaluation Summary")
    print(f" Run root: {run_root}")
    print(f" Passed: {len(rows) - len(failed)} / {len(rows)}")
    print(f" Summary JSON: {run_root / 'stage_l_full_eval_summary.json'}")
    print(f" Summary CSV: {run_root / 'stage_l_full_eval_summary.csv'}")
    print(f" Report: {run_root / 'stage_l_full_eval_report.md'}")
    print("===========================================================")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
