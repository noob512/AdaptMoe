#!/usr/bin/env python3
"""Validate Stage L7.2/L7.3 cross-pool coordinator semantics."""

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
            "Run a cross-pool Stage L7 probe: weights expert actions and KV "
            "safe-block actions share one Stage K coordinator trace, with "
            "priority metadata validated separately from per-pool safety."
        )
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--mode", choices=("trace_only", "enforce"), default="trace_only")
    parser.add_argument("--coordinator-mode", choices=("trace_only", "enforce"), default="trace_only")
    parser.add_argument("--global-bytes-per-step", type=int, default=1024 * 1024**2)
    parser.add_argument("--kv-bytes-per-step", type=int, default=512 * 1024**2)
    parser.add_argument("--weight-bytes-per-step", type=int, default=512 * 1024**2)
    parser.add_argument("--scratch-bytes-per-step", type=int, default=0)
    parser.add_argument("--priority", default="kv,scratch,weights")
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--plan-prompts", type=int, default=1)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--request-rate", default="5")
    parser.add_argument("--output-len", default="128")
    parser.add_argument("--runtime-budget-blocks", type=int, default=1)
    parser.add_argument("--runtime-candidate-limit", type=int, default=16)
    parser.add_argument("--max-offload-blocks", type=int, default=4)
    parser.add_argument("--gpu-run", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_stage_l7_cross_pool_check_{stamp}")


def paths(run_dir: Path) -> dict[str, Path]:
    return {
        "coordinator": run_dir / "vllm_uvm_pool_coordinator_stage_l7_cross_pool.jsonl",
        "residency": run_dir / "vllm_uvm_pool_residency_stage_l7_cross_pool.jsonl",
        "runtime": run_dir / "vllm_uvm_kv_runtime_stage_l7_cross_pool.jsonl",
        "pointer_map": run_dir / "vllm_uvm_kv_pointer_map_stage_l7_cross_pool.jsonl",
        "weight_trace": run_dir / "vllm_uvm_weight_prefetch_stage_l7_cross_pool.jsonl",
        "bench": run_dir / "vllm_bench_stage_l7_cross_pool.log",
        "runner": run_dir / "stage_l7_cross_pool_runner.log",
        "plan_json": run_dir / "vllm_stage_l7_cross_pool_weight_plan.json",
        "plan_summary": run_dir / "vllm_stage_l7_cross_pool_weight_plan_summary.json",
        "planning_weight_map": run_dir / "vllm_uvm_weight_regions_stage_l7_planning.jsonl",
        "planning_moe": run_dir / "vllm_uvm_moe_routing_stage_l7_planning.jsonl",
    }


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


def parse_failed_requests(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    matches = re.findall(
        r"Failed requests:\s+(\d+)",
        path.read_text(encoding="utf-8", errors="replace"),
    )
    return int(matches[-1]) if matches else None


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_cmd(cmd: list[str], *, cwd: Path, log: Path, append: bool = False) -> None:
    mode = "a" if append else "w"
    with log.open(mode, encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
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
        raise SystemExit(f"Command failed with exit code {rc}: {' '.join(cmd)}")


def run_gpu_probe(args: argparse.Namespace, run_dir: Path) -> dict[str, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    p = paths(run_dir)
    for path in p.values():
        if path.suffix in (".jsonl", ".log", ".json"):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    planning_cmd = [
        "./run_kv_fault_ratio.sh",
        "--model", args.model,
        "--mode", "trace",
        "--allocator-log", str(run_dir / "vllm_uvm_allocator_trace_stage_l7_planning.log"),
        "--trace-log", str(run_dir / "uvm_kv_fault_stats_stage_l7_planning.log"),
        "--address-log", str(run_dir / "vllm_uvm_address_regions_stage_l7_planning.log"),
        "--server-log", str(run_dir / "vllm_server_stage_l7_planning.log"),
        "--bench-log", str(run_dir / "vllm_bench_stage_l7_planning.log"),
        "--uvm-weight-map-enable", "1",
        "--uvm-weight-map-file", str(p["planning_weight_map"]),
        "--uvm-moe-routing-trace-enable", "1",
        "--uvm-moe-routing-trace-file", str(p["planning_moe"]),
        "--prompts", str(args.plan_prompts),
        "--request-rate", str(args.request_rate),
        "--output-len", str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L7.2/L7.3: planning weights expert actions")
    print(f" Output dir: {run_dir}")
    print("===========================================================")
    run_cmd(planning_cmd, cwd=SCRIPT_DIR, log=p["runner"])

    subprocess.run(
        [
            "python3",
            str(SCRIPT_DIR / "plan_stage_h_weight_expert_actions.py"),
            "--weight-map", str(p["planning_weight_map"]),
            "--moe-routing-trace", str(p["planning_moe"]),
            "--plan-json", str(p["plan_json"]),
            "--summary-json", str(p["plan_summary"]),
            "--target-roles", "moe_gate_up,moe_down",
            "--hot-top-k", "1024",
            "--cold-bottom-k", "1024",
            "--require-routing",
        ],
        cwd=SCRIPT_DIR,
        check=True,
    )

    main_cmd = [
        "./run_kv_fault_ratio.sh",
        "--model", args.model,
        "--mode", "trace",
        "--allocator-log", str(run_dir / "vllm_uvm_allocator_trace_stage_l7_cross_pool.log"),
        "--trace-log", str(run_dir / "uvm_kv_fault_stats_stage_l7_cross_pool.log"),
        "--address-log", str(run_dir / "vllm_uvm_address_regions_stage_l7_cross_pool.log"),
        "--server-log", str(run_dir / "vllm_server_stage_l7_cross_pool.log"),
        "--bench-log", str(p["bench"]),
        "--uvm-residency-enable", "1",
        "--uvm-residency-stage", "stage_l6",
        "--uvm-residency-mode", args.mode,
        "--uvm-residency-trace-file", str(p["residency"]),
        "--uvm-residency-global-gpu-bytes", str(24 * 1024**3),
        "--uvm-residency-weight-gpu-bytes", str(8 * 1024**3),
        "--uvm-residency-kv-gpu-bytes", str(1 * 1024**3),
        "--uvm-residency-scratch-gpu-bytes", str(2 * 1024**3),
        "--uvm-residency-safety-margin-bytes", str(1 * 1024**3),
        "--uvm-residency-kv-metadata-enable", "1",
        "--uvm-residency-kv-data-offload-enable", "1",
        "--uvm-residency-kv-data-offload-mode", "prefetch_cpu",
        "--uvm-residency-kv-prefetch-back-enable", "1",
        "--uvm-residency-kv-prefetch-back-mode", "prefetch",
        "--uvm-residency-kv-max-offload-blocks-per-step", str(args.max_offload_blocks),
        "--uvm-residency-kv-pointer-map-file", str(p["pointer_map"]),
        "--uvm-kv-runtime-enable", "1",
        "--uvm-kv-runtime-mode", "trace_only",
        "--uvm-kv-runtime-budget-blocks", str(args.runtime_budget_blocks),
        "--uvm-kv-runtime-trace-file", str(p["runtime"]),
        "--uvm-kv-runtime-candidate-limit", str(args.runtime_candidate_limit),
        "--uvm-kv-runtime-prefix-evict-enable", "0",
        "--uvm-weight-prefetch-enable", "1",
        "--uvm-weight-prefetch-mode", "trace_only",
        "--uvm-weight-prefetch-trace-file", str(p["weight_trace"]),
        "--uvm-weight-prefetch-plan-file", str(p["plan_json"]),
        "--uvm-weight-prefetch-require-plan", "1",
        "--uvm-weight-offload-enable", "1",
        "--uvm-weight-offload-mode", "trace_only",
        "--uvm-weight-offload-plan-file", str(p["plan_json"]),
        "--uvm-pool-coordinator-enable", "1",
        "--uvm-pool-coordinator-mode", args.coordinator_mode,
        "--uvm-pool-coordinator-trace-file", str(p["coordinator"]),
        "--uvm-pool-coordinator-global-bytes-per-step", str(args.global_bytes_per_step),
        "--uvm-pool-coordinator-kv-bytes-per-step", str(args.kv_bytes_per_step),
        "--uvm-pool-coordinator-weight-bytes-per-step", str(args.weight_bytes_per_step),
        "--uvm-pool-coordinator-scratch-bytes-per-step", str(args.scratch_bytes_per_step),
        "--uvm-pool-coordinator-priority", args.priority,
        "--prompts", str(args.prompts),
        "--request-rate", str(args.request_rate),
        "--output-len", str(args.output_len),
    ]
    print("===========================================================")
    print(" Stage L7.2/L7.3: running cross-pool coordinator probe")
    print(f" Coordinator trace: {p['coordinator']}")
    print("===========================================================")
    run_cmd(main_cmd, cwd=SCRIPT_DIR, log=p["runner"], append=True)
    return p


def summarize_coordinator(records: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [r for r in records if r.get("action") == "coordinator_request"]
    pools = Counter(str(r.get("pool", "unknown")) for r in requests)
    actions = Counter(str(r.get("requested_action", "unknown")) for r in requests)
    grants = [r for r in requests if r.get("allowed") is True]
    denies = [r for r in requests if r.get("allowed") is False]
    config = next((r for r in records if r.get("action") == "coordinator_config"), {})
    priority = str(config.get("priority") or "")
    priority_items = [item.strip() for item in priority.split(",") if item.strip()]
    priority_records = [
        r for r in requests
        if r.get("priority") == priority and r.get("priority_rank") is not None
    ]
    priority_ranks_by_pool: dict[str, set[int]] = {}
    for request in priority_records:
        pool = str(request.get("pool") or "unknown")
        try:
            rank = int(request.get("priority_rank"))
        except (TypeError, ValueError):
            continue
        priority_ranks_by_pool.setdefault(pool, set()).add(rank)
    expected_ranks = {
        pool: idx + 1
        for idx, pool in enumerate(priority_items)
    }
    weight_prefetch_actions = {
        action: count
        for action, count in actions.items()
        if action == "expert_prefetch_gpu"
    }
    weight_offload_actions = {
        action: count
        for action, count in actions.items()
        if action.startswith("expert_") and action != "expert_prefetch_gpu"
    }
    priority_safety_records = [
        r for r in requests
        if r.get("priority_semantics") == "global_action_order_only"
        and r.get("per_pool_safety_semantics")
        == "budget_and_callsite_rules_are_independent"
    ]
    return {
        "records": len(records),
        "requests": len(requests),
        "pools": dict(pools),
        "actions": dict(actions),
        "weight_prefetch_actions": dict(weight_prefetch_actions),
        "weight_offload_actions": dict(weight_offload_actions),
        "grants": len(grants),
        "denies": len(denies),
        "config": config,
        "priority": priority,
        "priority_items": priority_items,
        "priority_records": len(priority_records),
        "priority_ranks_by_pool": {
            pool: sorted(ranks)
            for pool, ranks in priority_ranks_by_pool.items()
        },
        "expected_ranks": expected_ranks,
        "priority_safety_records": len(priority_safety_records),
    }


def summarize_residency(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(r.get("action", "unknown")) for r in records)
    unsafe = 0
    for record in records:
        if record.get("action") == "residency_kv_data_offload_fallback":
            if (record.get("metadata") or {}).get("safe") is False:
                unsafe += 1
    return {
        "records": len(records),
        "offload_issued": counts.get("residency_kv_data_offload_issued", 0),
        "prefetch_issued": counts.get("residency_kv_prefetch_back_issued", 0),
        "unsafe_offloads": unsafe,
    }


def summarize_weight_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = Counter(str(r.get("action", "unknown")) for r in records)
    prefetch_candidates = (
        actions.get("trace_prefetch_candidate", 0)
        + actions.get("prefetch_issued", 0)
        + actions.get("prefetch_skipped", 0)
    )
    offload_candidates = (
        actions.get("trace_offload_candidate", 0)
        + actions.get("offload_advise_cpu_issued", 0)
        + actions.get("offload_prefetch_cpu_issued", 0)
        + actions.get("offload_skipped", 0)
    )
    return {
        "records": len(records),
        "actions": dict(actions),
        "prefetch_candidates": prefetch_candidates,
        "offload_candidates": offload_candidates,
    }


def validate(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    p = paths(run_dir)
    coordinator = summarize_coordinator(read_jsonl(p["coordinator"]))
    residency = summarize_residency(read_jsonl(p["residency"]))
    weight_trace = summarize_weight_trace(read_jsonl(p["weight_trace"]))
    failed_requests = parse_failed_requests(p["bench"])
    pointer_records = [
        r for r in read_jsonl(p["pointer_map"])
        if r.get("action") == "stage_l6_kv_pointer_map"
    ]
    pointer_entries = sum(int(r.get("mapped_blocks") or 0) for r in pointer_records)
    pools = set(coordinator["pools"])
    kv_expected_rank = coordinator["expected_ranks"].get("kv")
    weights_expected_rank = coordinator["expected_ranks"].get("weights")
    kv_ranks = coordinator["priority_ranks_by_pool"].get("kv", [])
    weights_ranks = coordinator["priority_ranks_by_pool"].get("weights", [])
    checks = [
        check("coordinator_trace_present", p["coordinator"].is_file(), str(p["coordinator"])),
        check("weights_pool_visible", "weights" in pools, f"pools={coordinator['pools']}"),
        check("kv_pool_visible", "kv" in pools, f"pools={coordinator['pools']}"),
        check(
            "weight_prefetch_actions_visible",
            bool(coordinator["weight_prefetch_actions"]),
            f"actions={coordinator['actions']}",
        ),
        check(
            "weight_offload_actions_visible",
            bool(coordinator["weight_offload_actions"]),
            f"actions={coordinator['actions']}",
        ),
        check(
            "kv_actions_visible",
            "kv_safe_block_prefetch_cpu" in coordinator["actions"]
            and "kv_block_prefetch_gpu" in coordinator["actions"],
            f"actions={coordinator['actions']}",
        ),
        check(
            "priority_config_matches",
            coordinator["priority"] == args.priority,
            f"priority={coordinator['priority']} expected={args.priority}",
        ),
        check(
            "priority_rank_visible",
            coordinator["priority_records"] > 0,
            f"priority_records={coordinator['priority_records']}",
        ),
        check(
            "priority_rank_matches_kv",
            kv_expected_rank in kv_ranks,
            f"kv_ranks={kv_ranks} expected={kv_expected_rank}",
        ),
        check(
            "priority_rank_matches_weights",
            weights_expected_rank in weights_ranks,
            f"weights_ranks={weights_ranks} expected={weights_expected_rank}",
        ),
        check(
            "priority_safety_semantics_visible",
            coordinator["priority_safety_records"] > 0,
            f"priority_safety_records={coordinator['priority_safety_records']}",
        ),
        check("kv_offload_issued_present", residency["offload_issued"] > 0, str(residency)),
        check("kv_prefetch_issued_present", residency["prefetch_issued"] > 0, str(residency)),
        check("unsafe_kv_offloads_zero", residency["unsafe_offloads"] == 0, str(residency)),
        check("weight_trace_records_present", weight_trace["records"] > 0, str(weight_trace)),
        check(
            "weight_prefetch_trace_present",
            weight_trace["prefetch_candidates"] > 0,
            str(weight_trace),
        ),
        check(
            "weight_offload_trace_present",
            weight_trace["offload_candidates"] > 0,
            str(weight_trace),
        ),
        check("pointer_map_present", pointer_entries > 0, f"pointer_entries={pointer_entries}"),
        check("benchmark_no_failed_requests", failed_requests in (None, 0), f"failed_requests={failed_requests}"),
    ]
    passed = all(item["passed"] for item in checks)
    output_json = Path(args.output_json) if args.output_json else (
        run_dir / "stage_l7_cross_pool_success_check.json"
    )
    result = {
        "passed": passed,
        "run_dir": str(run_dir),
        "paths": {k: str(v) for k, v in p.items()},
        "coordinator": coordinator,
        "residency": residency,
        "weight_trace": weight_trace,
        "pointer_map": {"records": len(pointer_records), "mapped_blocks": pointer_entries},
        "failed_requests": failed_requests,
        "checks": checks,
        "output_json": str(output_json),
    }
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def print_summary(result: dict[str, Any]) -> None:
    status = "PASS" if result["passed"] else "FAIL"
    print("===========================================================")
    print(f" Stage L7.2/L7.3 Cross-Pool Check: {status}")
    print(f" Run dir: {result['run_dir']}")
    print("===========================================================")
    print(f"- coordinator={result['coordinator']}")
    print(f"- residency={result['residency']}")
    print(f"- weight_trace={result['weight_trace']}")
    print(f"- pointer_map={result['pointer_map']}")
    print("- checks:")
    for item in result["checks"]:
        print(f"  {item['name']}={item['passed']} ({item['detail']})")
    print(f"- check_json={result['output_json']}")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    if not args.skip_run:
        if not args.gpu_run:
            raise SystemExit("Use --gpu-run or --skip-run.")
        run_gpu_probe(args, run_dir)
    result = validate(args, run_dir)
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
