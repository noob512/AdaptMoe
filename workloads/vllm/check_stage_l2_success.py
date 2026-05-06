#!/usr/bin/env python3
"""Validate Stage L2 weights residency enforcement."""

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
            "Check Stage L2 weights hot/cold residency enforcement. The "
            "default self-test validates the Stage L2 trace contract without "
            "a GPU. Use --gpu-run for a real vLLM benchmark probe."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage L2 run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage L residency JSONL.")
    parser.add_argument("--prefetch-trace-jsonl", help="Existing or new Stage I JSONL.")
    parser.add_argument("--bench-log", help="Existing or new benchmark log.")
    parser.add_argument("--runner-log", help="Existing or new runner log.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="enforce",
        help="Stage L residency mode. L2 success normally uses enforce.",
    )
    parser.add_argument("--global-gpu-bytes", type=int, default=3072)
    parser.add_argument("--weight-gpu-bytes", type=int, default=1)
    parser.add_argument("--kv-gpu-bytes", type=int, default=0)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=0)
    parser.add_argument("--safety-margin-bytes", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--max-experts-per-layer", type=int, default=2)
    parser.add_argument("--target-roles", default="moe_gate_up,moe_down")
    parser.add_argument("--offload-target-roles", default="moe_gate_up,moe_down")
    parser.add_argument("--plan-layers", type=int, default=64)
    parser.add_argument("--plan-experts", type=int, default=128)
    parser.add_argument(
        "--victim-action",
        choices=("advise_cpu", "prefetch_cpu"),
        default="advise_cpu",
    )
    parser.add_argument(
        "--sync-offload",
        choices=("0", "1"),
        default="1",
        help="Release estimated residency after successful victim offload.",
    )
    parser.add_argument("--max-victims-per-step", type=int, default=0)
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate an existing trace; use --trace-jsonl or --run-dir.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_l2_success_check_{stamp}")


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


def step_from_scope_key(scope_key: object) -> int | None:
    if not isinstance(scope_key, str):
        return None
    for part in reversed(scope_key.split(":")):
        if part.isdigit():
            return int(part)
    return None


def step_from_record(record: dict[str, Any]) -> int | None:
    step = record.get("step")
    if isinstance(step, int):
        return step
    if isinstance(step, str) and step.isdigit():
        return int(step)
    scope_step = step_from_scope_key(record.get("scope_key"))
    if scope_step is not None:
        return scope_step
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return step_from_scope_key(metadata.get("scope_key"))
    return None


def resolve_trace(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    return run_dir / "vllm_uvm_pool_residency_stage_l2.jsonl"


def write_broad_cold_plan(args: argparse.Namespace, run_dir: Path) -> Path:
    plan_json = run_dir / "vllm_stage_l2_broad_cold_plan.json"
    offload_plan = [
        {
            "layer_id": layer_id,
            "expert_id": expert_id,
            "reason": "stage_l2_broad_cold_probe",
        }
        for layer_id in range(max(args.plan_layers, 0))
        for expert_id in range(max(args.plan_experts, 0))
    ]
    plan = {
        "prefetch_plan": [],
        "offload_plan": offload_plan,
        "prefetch_plan_records": 0,
        "offload_plan_records": len(offload_plan),
    }
    plan_json.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan_json


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_l2",
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
            "VLLM_UVM_RESIDENCY_WEIGHT_ENFORCE_HOT_SET": "1",
            "VLLM_UVM_RESIDENCY_WEIGHT_INITIAL_STATE": "cpu",
            "VLLM_UVM_RESIDENCY_WEIGHT_SYNC_OFFLOAD": args.sync_offload,
            "VLLM_UVM_RESIDENCY_WEIGHT_MAX_VICTIMS_PER_STEP": str(
                args.max_victims_per_step
            ),
            "VLLM_UVM_RESIDENCY_WEIGHT_VICTIM_ACTION": args.victim_action,
        }
    )
    return env


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path, None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = resolve_trace(args, run_dir)
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l2_self_test_runner.log"
    )
    code = r'''
import os

from vllm.device_allocator.uvm_pool_residency import (
    mark_uvm_residency_state,
    record_uvm_residency_access,
    record_uvm_residency_action,
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
        "stage": "stage_l2",
        "plan_cold": True,
        "source_action": "l2_cold_candidate",
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
    metadata={"source": "stage_i", "stage": "stage_l2", "active_this_step": True},
)
record_uvm_residency_access(
    region_id=active_region,
    step=1,
    metadata={"source": "stage_i", "stage": "stage_l2", "active_this_step": True},
)
decision = request_uvm_gpu_residency(
    pool="weights",
    region_id=active_region,
    bytes=640,
    scope_key="stage_l2_self_test:1",
    active_region_ids={active_region},
    metadata={"source": "stage_i", "stage": "stage_l2", "active_this_step": True},
)
assert decision.would_deny
assert cold_region in decision.victims
assert active_region not in decision.victims
mark_uvm_residency_state(
    region_id=cold_region,
    state="cpu",
    metadata={"source": "stage_i", "stage": "stage_l2", "victim": True},
)
record_uvm_residency_action(
    action="residency_weight_victim_offload_issued",
    pool="weights",
    region_id=cold_region,
    bytes=512,
    metadata={"source": "stage_i", "stage": "stage_l2", "victim": True},
    success=True,
)
retry = request_uvm_gpu_residency(
    pool="weights",
    region_id=active_region,
    bytes=640,
    scope_key="stage_l2_self_test:1:retry",
    active_region_ids={active_region},
    metadata={
        "source": "stage_i",
        "stage": "stage_l2",
        "active_this_step": True,
        "retry_after_victim_offload": True,
    },
)
assert retry.would_deny
record_uvm_residency_action(
    action="residency_weight_hot_prefetch_skipped",
    pool="weights",
    region_id=active_region,
    bytes=640,
    metadata={"source": "stage_i", "stage": "stage_l2", "active_this_step": True},
    would_deny=True,
)
'''
    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    print("===========================================================")
    print(" Stage L2 Success Check: running weights enforce self-test")
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
            f"Stage L2 self-test failed with exit code {rc}; log={runner_log}"
        )
    return trace_jsonl, runner_log, None


def run_gpu_probe(
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_json = write_broad_cold_plan(args, run_dir)
    trace_jsonl = resolve_trace(args, run_dir)
    prefetch_trace = Path(args.prefetch_trace_jsonl) if args.prefetch_trace_jsonl else (
        run_dir / "vllm_uvm_weight_prefetch_stage_i_l2.jsonl"
    )
    bench_log = Path(args.bench_log) if args.bench_log else (
        run_dir / "vllm_bench_stage_l2.log"
    )
    runner_log = Path(args.runner_log) if args.runner_log else (
        run_dir / "stage_l2_gpu_runner.log"
    )
    cmd = [
        "./run_kv_fault_ratio.sh",
        "--model",
        args.model,
        "--mode",
        "trace",
        "--allocator-log",
        str(run_dir / "vllm_uvm_allocator_trace_stage_l2.log"),
        "--trace-log",
        str(run_dir / "uvm_kv_fault_stats_stage_l2.log"),
        "--address-log",
        str(run_dir / "vllm_uvm_address_regions_stage_l2.log"),
        "--server-log",
        str(run_dir / "vllm_server_stage_l2.log"),
        "--bench-log",
        str(bench_log),
        "--uvm-kv-budget-bytes",
        "0",
        "--uvm-weight-budget-bytes",
        "0",
        "--uvm-moe-routing-trace-enable",
        "1",
        "--uvm-moe-routing-trace-file",
        str(run_dir / "vllm_uvm_moe_routing_stage_l2.jsonl"),
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
        "--uvm-weight-offload-plan-file",
        str(plan_json),
        "--uvm-weight-offload-target-roles",
        args.offload_target_roles,
        "--uvm-weight-offload-max-experts-per-layer",
        "1",
        "--uvm-residency-enable",
        "1",
        "--uvm-residency-stage",
        "stage_l2",
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
        "--uvm-residency-weight-initial-state",
        "cpu",
        "--uvm-residency-weight-enforce-hot-set",
        "1",
        "--uvm-residency-weight-sync-offload",
        args.sync_offload,
        "--uvm-residency-weight-max-victims-per-step",
        str(args.max_victims_per_step),
        "--uvm-residency-weight-victim-policy",
        "cold_plan_lru",
        "--uvm-residency-weight-victim-action",
        args.victim_action,
        "--prompts",
        str(args.prompts),
        "--request-rate",
        str(args.request_rate),
        "--output-len",
        str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L2 Success Check: running GPU weights enforce probe")
    print(f" Output dir: {run_dir}")
    print(f" Trace JSONL: {trace_jsonl}")
    print(f" Prefetch trace: {prefetch_trace}")
    print(f" Cold plan: {plan_json}")
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
        raise SystemExit(f"Stage L2 GPU probe failed with exit code {rc}; log={runner_log}")
    return trace_jsonl, runner_log, bench_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    pool_action_counts: Counter[str] = Counter()
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}
    active_region_ids: set[str] = set()
    active_region_ids_by_step: dict[int, set[str]] = {}
    candidate_region_ids: set[str] = set()
    request_victim_ids: set[str] = set()
    stage_i_weight_region_records = 0
    hot_prefetch_request_records = 0
    cold_weight_victim_records = 0
    weight_offload_issued_records = 0
    hot_prefetch_skipped_records = 0
    kv_action_records = 0
    scratch_action_records = 0
    active_weight_victim_records = 0
    active_weight_victim_details: list[dict[str, Any]] = []
    skip_active_records = 0

    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        pool = str(record.get("pool") or "")
        if pool:
            pool_action_counts[f"{pool}:{action}"] += 1
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        region_id = str(record.get("region_id") or "")
        step = step_from_record(record)
        if (
            action == "residency_access"
            and pool == "weights"
            and region_id
            and step is not None
        ):
            active_region_ids_by_step.setdefault(step, set()).add(region_id)
        if metadata.get("active_this_step") is True or record.get("active") is True:
            if region_id:
                active_region_ids.add(region_id)
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        if action == "residency_register_region" and pool == "weights":
            if metadata.get("source") == "stage_i":
                stage_i_weight_region_records += 1
            if (
                metadata.get("source_action") in ("l2_cold_candidate", "offload")
                or metadata.get("plan_cold") is True
            ):
                cold_weight_victim_records += 1
        if action == "residency_request" and pool == "weights":
            if metadata.get("active_this_step") is True:
                hot_prefetch_request_records += 1
            for victim in record.get("victims") or []:
                victim_id = str(victim)
                request_victim_ids.add(victim_id)
                if (
                    step is not None
                    and victim_id in active_region_ids_by_step.get(step, set())
                ):
                    active_weight_victim_records += 1
                    active_weight_victim_details.append(
                        {
                            "source": "request_victim",
                            "step": step,
                            "scope_key": record.get("scope_key"),
                            "request_region_id": region_id,
                            "victim_region_id": victim_id,
                        }
                    )
        if action == "residency_would_evict_candidate":
            if region_id:
                candidate_region_ids.add(region_id)
                if (
                    step is not None
                    and region_id in active_region_ids_by_step.get(step, set())
                ):
                    active_weight_victim_records += 1
                    active_weight_victim_details.append(
                        {
                            "source": "would_evict_candidate",
                            "step": step,
                            "scope_key": record.get("scope_key"),
                            "request_region_id": record.get("request_region_id"),
                            "victim_region_id": region_id,
                        }
                    )
        if action == "residency_weight_victim_offload_issued":
            weight_offload_issued_records += 1
            if (
                region_id
                and step is not None
                and region_id in active_region_ids_by_step.get(step, set())
            ):
                active_weight_victim_records += 1
                active_weight_victim_details.append(
                    {
                        "source": "offload_issued",
                        "step": step,
                        "scope_key": metadata.get("scope_key"),
                        "request_region_id": metadata.get("reason_region_id"),
                        "victim_region_id": region_id,
                    }
                )
        if action == "residency_weight_victim_skip_active":
            skip_active_records += 1
        if action == "residency_weight_hot_prefetch_skipped":
            hot_prefetch_skipped_records += 1
        if pool == "kv":
            kv_action_records += 1
        if pool == "scratch" and "evict" in action:
            scratch_action_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "pool_action_counts": dict(pool_action_counts),
        "stage_i_weight_region_records": stage_i_weight_region_records,
        "hot_prefetch_request_records": hot_prefetch_request_records,
        "cold_weight_victim_records": cold_weight_victim_records,
        "weight_offload_issued_records": weight_offload_issued_records,
        "hot_prefetch_skipped_records": hot_prefetch_skipped_records,
        "active_weight_victim_records": active_weight_victim_records,
        "active_weight_victim_details": active_weight_victim_details[:20],
        "skip_active_records": skip_active_records,
        "kv_action_records": kv_action_records,
        "scratch_eviction_records": scratch_action_records,
        "active_region_count": len(active_region_ids),
        "active_region_steps": len(active_region_ids_by_step),
        "candidate_region_count": len(candidate_region_ids),
        "request_victim_count": len(request_victim_ids),
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
    summary = trace_summary["summary"]
    failed_requests = parse_failed_requests(bench_log)
    weight_budget = summary.get("weight_gpu_budget_bytes")
    weight_estimated = summary.get("weights_estimated_gpu_resident_bytes")
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
            "weight_enforce_enabled",
            config.get("weight_enforce_hot_set") is True,
            f"config={config}",
        ),
        check(
            "weight_budget_matches",
            weight_budget == args.weight_gpu_bytes,
            f"weight_budget={weight_budget} expected={args.weight_gpu_bytes}",
        ),
        check(
            "weights_estimated_within_budget",
            isinstance(weight_estimated, int)
            and isinstance(weight_budget, int)
            and (weight_budget == 0 or weight_estimated <= weight_budget),
            f"weights_estimated={weight_estimated} weight_budget={weight_budget}",
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
            "hot_prefetch_request_records_present",
            trace_summary["hot_prefetch_request_records"] > 0,
            (
                "hot_prefetch_request_records="
                f"{trace_summary['hot_prefetch_request_records']}"
            ),
        ),
        check(
            "cold_weight_victim_records_present",
            trace_summary["cold_weight_victim_records"] > 0,
            f"cold_weight_victim_records={trace_summary['cold_weight_victim_records']}",
        ),
        check(
            "weight_offload_issued_records_present",
            trace_summary["weight_offload_issued_records"] > 0,
            (
                "weight_offload_issued_records="
                f"{trace_summary['weight_offload_issued_records']}"
            ),
        ),
        check(
            "active_weight_victim_records_zero",
            trace_summary["active_weight_victim_records"] == 0,
            (
                "active_weight_victim_records="
                f"{trace_summary['active_weight_victim_records']}"
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
            "benchmark_no_failed_requests",
            failed_requests in (None, 0),
            f"failed_requests={failed_requests}",
        ),
    ]
    if args.mode == "enforce":
        checks.append(
            check(
                "hot_prefetch_skipped_or_offload_observed",
                trace_summary["hot_prefetch_skipped_records"] > 0
                or trace_summary["weight_offload_issued_records"] > 0,
                (
                    "hot_prefetch_skipped_records="
                    f"{trace_summary['hot_prefetch_skipped_records']} "
                    "weight_offload_issued_records="
                    f"{trace_summary['weight_offload_issued_records']}"
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
        trace_jsonl.parent / "stage_l2_success_check.json"
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
    print(f" Stage L2 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- records="
        f"stage_i_weight_regions={trace['stage_i_weight_region_records']} "
        f"hot_prefetch_requests={trace['hot_prefetch_request_records']} "
        f"cold_victims={trace['cold_weight_victim_records']} "
        f"weight_offload_issued={trace['weight_offload_issued_records']} "
        f"hot_prefetch_skipped={trace['hot_prefetch_skipped_records']} "
        f"active_weight_victims={trace['active_weight_victim_records']} "
        f"kv_actions={trace['kv_action_records']} "
        f"scratch_evictions={trace['scratch_eviction_records']}"
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
