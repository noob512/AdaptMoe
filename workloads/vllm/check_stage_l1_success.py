#!/usr/bin/env python3
"""Validate Stage L1 expert weight residency integration."""

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
            "Check Stage L1. The default self-test validates the expert-slice "
            "residency trace contract without starting a GPU server. Use "
            "--gpu-run to run a tiny vLLM benchmark and validate real Stage I "
            "expert weight residency records."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage L1 run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage L residency JSONL.")
    parser.add_argument("--prefetch-trace-jsonl", help="Existing or new Stage I JSONL.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--runner-log", help="Existing or new runner log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
        help="Stage L residency mode. Stage L1 does not alter Stage I behavior.",
    )
    parser.add_argument("--global-gpu-bytes", type=int, default=3072)
    parser.add_argument("--weight-gpu-bytes", type=int, default=1024)
    parser.add_argument("--kv-gpu-bytes", type=int, default=0)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=0)
    parser.add_argument("--safety-margin-bytes", type=int, default=0)
    parser.add_argument("--priority", default="kv,scratch,weights")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--max-experts-per-layer", type=int, default=2)
    parser.add_argument("--target-roles", default="moe_gate_up,moe_down")
    parser.add_argument(
        "--gpu-run",
        action="store_true",
        help="Run a real vLLM benchmark and validate Stage I/L1 integration.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate an existing trace; use --trace-jsonl or --run-dir.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_l1_success_check_{stamp}")


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


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


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    return run_dir / "vllm_uvm_pool_residency_stage_l1.jsonl"


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l1",
            "VLLM_UVM_RESIDENCY_MODE": args.mode,
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_RESIDENCY_GLOBAL_GPU_BYTES": str(args.global_gpu_bytes),
            "VLLM_UVM_RESIDENCY_WEIGHT_GPU_BYTES": str(args.weight_gpu_bytes),
            "VLLM_UVM_RESIDENCY_KV_GPU_BYTES": str(args.kv_gpu_bytes),
            "VLLM_UVM_RESIDENCY_SCRATCH_GPU_BYTES": str(
                args.scratch_gpu_bytes
            ),
            "VLLM_UVM_RESIDENCY_SAFETY_MARGIN_BYTES": str(
                args.safety_margin_bytes
            ),
            "VLLM_UVM_RESIDENCY_PRIORITY": args.priority,
        }
    )
    return env


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l1_self_test_runner.log"
    )

    code = r'''
import os

from vllm.device_allocator.uvm_pool_residency import (
    record_uvm_residency_access,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
)

trace_file = os.environ["VLLM_UVM_RESIDENCY_TRACE_FILE"]
try:
    os.unlink(trace_file)
except FileNotFoundError:
    pass

cold_region = "weights:0:3:moe_gate_up"
active_region = "weights:0:7:moe_gate_up"
register_uvm_residency_region(
    pool="weights",
    region_id=cold_region,
    ptr=0x1000,
    bytes=512,
    state="gpu",
    evictable=True,
    active=False,
    metadata={
        "source": "stage_i",
        "stage": "stage_l1",
        "layer_name": "model.layers.0.mlp.experts",
        "layer_id": 0,
        "global_expert_id": 3,
        "local_expert_id": 3,
        "role": "moe_gate_up",
        "plan_cold": True,
        "active_this_step": False,
    },
)
register_uvm_residency_region(
    pool="weights",
    region_id=active_region,
    ptr=0x2000,
    bytes=640,
    state="cpu",
    evictable=False,
    active=True,
    metadata={
        "source": "stage_i",
        "stage": "stage_l1",
        "layer_name": "model.layers.0.mlp.experts",
        "layer_id": 0,
        "global_expert_id": 7,
        "local_expert_id": 7,
        "role": "moe_gate_up",
        "plan_hot": True,
        "active_this_step": True,
    },
)
record_uvm_residency_access(
    region_id=active_region,
    step=42,
    metadata={
        "source": "stage_i",
        "stage": "stage_l1",
        "active_this_step": True,
    },
)
decision = request_uvm_gpu_residency(
    pool="weights",
    region_id=active_region,
    bytes=640,
    scope_key="stage_l1_self_test:42",
    active_region_ids={active_region},
    metadata={
        "source": "stage_i",
        "stage": "stage_l1",
        "active_this_step": True,
    },
)
assert decision.would_deny
assert cold_region in decision.victims
assert active_region not in decision.victims
'''

    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    print("===========================================================")
    print(" Stage L1 Success Check: running expert residency self-test")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Mode: {args.mode}")
    print("===========================================================")
    with runner_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=residency_env(args, trace_jsonl),
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
            f"Stage L1 self-test failed with exit code {rc}; log={runner_log}"
        )
    return trace_jsonl, runner_log, None


def run_gpu_probe(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    prefetch_trace = Path(args.prefetch_trace_jsonl) if args.prefetch_trace_jsonl else (
        run_dir / "vllm_uvm_weight_prefetch_stage_i_l1.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l1.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l1_gpu_runner.log"
    )
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model",
        args.model,
        "--mode",
        "trace",
        "--allocator-log",
        str(run_dir / "vllm_uvm_allocator_trace_stage_l1.log"),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_l1.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_l1.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_l1.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(run_dir / "vllm_uvm_moe_routing_stage_l1.jsonl"),
        "--uvm-weight-prefetch-enable",
        "1",
        "--uvm-weight-prefetch-mode",
        "trace_only",
        "--uvm-weight-prefetch-trace-file",
        str(prefetch_trace),
        "--uvm-weight-prefetch-max-bytes-per-step",
        "67108864",
        "--uvm-weight-prefetch-max-experts-per-layer",
        str(args.max_experts_per_layer),
        "--uvm-weight-prefetch-target-roles",
        args.target_roles,
        "--uvm-weight-prefetch-require-plan",
        "0",
        "--uvm-weight-offload-enable",
        "0",
        "--uvm-residency-enable",
        "1",
        "--uvm-residency-stage",
        "stage_l1",
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
        "--uvm-residency-priority",
        args.priority,
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L1 Success Check: running GPU integration probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Prefetch trace: {prefetch_trace}")
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
        raise SystemExit(f"Stage L1 GPU probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, runner_log, bench_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    request_records = 0
    would_deny_records = 0
    denied_records = 0
    weights_register_records = 0
    weights_request_records = 0
    active_access_records = 0
    stage_i_weight_region_records = 0
    stage_i_weight_request_records = 0
    candidate_records = 0
    unsafe_active_candidate_records = 0
    real_action_records = 0
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    active_region_ids: set[str] = set()
    request_victim_ids: set[str] = set()
    candidate_region_ids: set[str] = set()

    real_actions = {
        "residency_offload_issued",
        "residency_prefetch_issued",
        "residency_sync",
    }
    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        pool = record.get("pool")
        region_id = str(record.get("region_id") or "")
        is_stage_i = metadata.get("source") == "stage_i"
        is_stage_l1 = metadata.get("stage") == "stage_l1"
        if metadata.get("active_this_step") is True or record.get("active") is True:
            if region_id:
                active_region_ids.add(region_id)
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        if action == "residency_register_region" and pool == "weights":
            weights_register_records += 1
            if is_stage_i or is_stage_l1:
                stage_i_weight_region_records += 1
        if action == "residency_request":
            request_records += 1
            if record.get("would_deny") is True:
                would_deny_records += 1
            if record.get("allowed") is False:
                denied_records += 1
            for victim in record.get("victims") or []:
                request_victim_ids.add(str(victim))
            if pool == "weights":
                weights_request_records += 1
                if is_stage_i or is_stage_l1:
                    stage_i_weight_request_records += 1
        if action == "residency_access" and pool == "weights":
            if metadata.get("active_this_step") is True:
                active_access_records += 1
        if action == "residency_would_evict_candidate":
            candidate_records += 1
            if region_id:
                candidate_region_ids.add(region_id)
        if action in real_actions:
            real_action_records += 1

    unsafe_active_candidate_records = len(
        (candidate_region_ids | request_victim_ids) & active_region_ids
    )
    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "request_records": request_records,
        "would_deny_records": would_deny_records,
        "denied_records": denied_records,
        "weights_register_records": weights_register_records,
        "weights_request_records": weights_request_records,
        "stage_i_weight_region_records": stage_i_weight_region_records,
        "stage_i_weight_request_records": stage_i_weight_request_records,
        "active_expert_access_records": active_access_records,
        "candidate_records": candidate_records,
        "unsafe_active_candidate_records": unsafe_active_candidate_records,
        "real_action_records": real_action_records,
        "active_region_ids": sorted(active_region_ids),
        "candidate_region_ids": sorted(candidate_region_ids),
        "request_victim_ids": sorted(request_victim_ids),
        "config": config,
        "summary": summary,
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
    failed_requests = parse_failed_requests(bench_log)
    checks = [
        check("trace_jsonl_present", trace_jsonl.is_file(), f"trace_jsonl={trace_jsonl}"),
        check(
            "trace_records_present",
            trace_summary["trace_records"] > 0,
            f"trace_records={trace_summary['trace_records']}",
        ),
        check("residency_config_present", bool(config), f"config={config}"),
        check("residency_enabled", config.get("enabled") is True, f"config={config}"),
        check(
            "residency_mode_matches",
            config.get("mode") == args.mode,
            f"mode={config.get('mode')} expected={args.mode}",
        ),
        check(
            "stage_i_weight_regions_present",
            trace_summary["stage_i_weight_region_records"] > 0,
            (
                "stage_i_weight_region_records="
                f"{trace_summary['stage_i_weight_region_records']}"
            ),
        ),
        check(
            "weights_residency_request_records_present",
            trace_summary["weights_request_records"] > 0,
            f"weights_request_records={trace_summary['weights_request_records']}",
        ),
        check(
            "stage_i_weight_request_records_present",
            trace_summary["stage_i_weight_request_records"] > 0,
            (
                "stage_i_weight_request_records="
                f"{trace_summary['stage_i_weight_request_records']}"
            ),
        ),
        check(
            "active_expert_access_records_present",
            trace_summary["active_expert_access_records"] > 0,
            (
                "active_expert_access_records="
                f"{trace_summary['active_expert_access_records']}"
            ),
        ),
        check(
            "would_deny_records_present_when_budget_small",
            args.weight_gpu_bytes == 0 or trace_summary["would_deny_records"] > 0,
            f"would_deny_records={trace_summary['would_deny_records']}",
        ),
        check(
            "active_experts_not_victim_candidates",
            trace_summary["unsafe_active_candidate_records"] == 0,
            (
                "unsafe_active_candidate_records="
                f"{trace_summary['unsafe_active_candidate_records']} "
                f"active={trace_summary['active_region_ids']} "
                f"candidates={trace_summary['candidate_region_ids']} "
                f"victims={trace_summary['request_victim_ids']}"
            ),
        ),
        check(
            "no_real_residency_actions",
            trace_summary["real_action_records"] == 0,
            f"real_action_records={trace_summary['real_action_records']}",
        ),
        check(
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]

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
        trace_jsonl.parent / "stage_l1_success_check.json"
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
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L1 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- records="
        f"stage_i_weight_regions={trace['stage_i_weight_region_records']} "
        f"weights_requests={trace['weights_request_records']} "
        f"active_access={trace['active_expert_access_records']} "
        f"would_deny={trace['would_deny_records']} "
        f"candidates={trace['candidate_records']} "
        f"unsafe_active_candidates={trace['unsafe_active_candidate_records']} "
        f"real_actions={trace['real_action_records']}"
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
