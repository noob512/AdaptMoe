#!/usr/bin/env python3
"""Validate Stage L0 estimated pool residency accounting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Stage L0 PoolResidencyManager. The default self-test does "
            "not start a GPU server; it validates region registration, state "
            "accounting, budget would-deny decisions, same-pool victim "
            "candidates, and summaries."
        )
    )
    parser.add_argument("--run-dir", help="Existing or new Stage L run directory.")
    parser.add_argument("--output-json", help="Where to write the check summary JSON.")
    parser.add_argument("--trace-jsonl", help="Existing or new Stage L JSONL trace.")
    parser.add_argument(
        "--mode",
        choices=("trace_only", "enforce"),
        default="trace_only",
        help="Stage L0 mode. L0 never performs real reclaim actions.",
    )
    parser.add_argument("--global-gpu-bytes", type=int, default=3072)
    parser.add_argument("--weight-gpu-bytes", type=int, default=1024)
    parser.add_argument("--kv-gpu-bytes", type=int, default=1024)
    parser.add_argument("--scratch-gpu-bytes", type=int, default=256)
    parser.add_argument("--safety-margin-bytes", type=int, default=0)
    parser.add_argument("--priority", default="kv,scratch,weights")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Validate an existing trace; use --trace-jsonl or --run-dir.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the fast in-process self-test. This is the default.",
    )
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return Path(f"/tmp/vllm_stage_l_success_check_{stamp}")


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


def resolve_trace(args: argparse.Namespace) -> Path | None:
    if args.trace_jsonl:
        return Path(args.trace_jsonl)
    if args.run_dir:
        candidate = Path(args.run_dir) / "vllm_uvm_pool_residency_stage_l.jsonl"
        if candidate.is_file():
            return candidate
    return None


def residency_env(args: argparse.Namespace, trace_jsonl: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
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


def run_self_test(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_jsonl = Path(args.trace_jsonl) if args.trace_jsonl else (
        run_dir / "vllm_uvm_pool_residency_stage_l.jsonl"
    )
    runner_log = run_dir / "stage_l_self_test_runner.log"

    code = r'''
import os

from vllm.device_allocator.uvm_pool_residency import (
    get_uvm_residency_snapshot,
    mark_uvm_residency_state,
    record_uvm_residency_access,
    register_uvm_residency_region,
    request_uvm_gpu_residency,
)

trace_file = os.environ["VLLM_UVM_RESIDENCY_TRACE_FILE"]
try:
    os.unlink(trace_file)
except FileNotFoundError:
    pass

register_uvm_residency_region(
    pool="weights",
    region_id="weights:0:0:moe_gate_up",
    ptr=0x1000,
    bytes=512,
    state="gpu",
    evictable=True,
    metadata={"stage": "stage_l_self_test", "cold": True},
)
register_uvm_residency_region(
    pool="weights",
    region_id="weights:0:1:moe_down",
    ptr=0x2000,
    bytes=640,
    state="cpu",
    evictable=False,
    active=True,
    metadata={"stage": "stage_l_self_test", "active": True},
)
register_uvm_residency_region(
    pool="kv",
    region_id="kv:0:7",
    ptr=None,
    bytes=768,
    state="gpu",
    evictable=True,
    metadata={"stage": "stage_l_self_test", "safe": True},
)
register_uvm_residency_region(
    pool="scratch",
    region_id="scratch:0x3000",
    ptr=0x3000,
    bytes=128,
    state="gpu",
    evictable=False,
    metadata={"stage": "stage_l_self_test"},
)
record_uvm_residency_access(
    region_id="weights:0:0:moe_gate_up",
    step=1,
    metadata={"stage": "stage_l_self_test"},
)
d1 = request_uvm_gpu_residency(
    pool="weights",
    region_id="weights:0:1:moe_down",
    bytes=640,
    scope_key="stage_l_self_test:0",
    active_region_ids={"weights:0:1:moe_down"},
    metadata={"stage": "stage_l_self_test"},
)
assert d1.would_deny
assert d1.allowed == (os.environ["VLLM_UVM_RESIDENCY_MODE"] != "enforce")
assert "weights:0:0:moe_gate_up" in d1.victims

mark_uvm_residency_state(
    region_id="weights:0:0:moe_gate_up",
    state="cpu",
    metadata={"stage": "stage_l_self_test", "reason": "accounting_release"},
)
d2 = request_uvm_gpu_residency(
    pool="weights",
    region_id="weights:0:1:moe_down",
    bytes=640,
    scope_key="stage_l_self_test:1",
    active_region_ids={"weights:0:1:moe_down"},
    metadata={"stage": "stage_l_self_test"},
)
assert not d2.would_deny
mark_uvm_residency_state(
    region_id="weights:0:1:moe_down",
    state="gpu",
    metadata={"stage": "stage_l_self_test", "reason": "accounting_admit"},
)
d3 = request_uvm_gpu_residency(
    pool="scratch",
    region_id="scratch:0x4000",
    bytes=256,
    scope_key="stage_l_self_test:2",
    metadata={"stage": "stage_l_self_test"},
)
assert d3.would_deny
snapshot = get_uvm_residency_snapshot()
assert snapshot["weights_estimated_gpu_resident_bytes"] == 640
assert snapshot["kv_estimated_gpu_resident_bytes"] == 768
assert snapshot["scratch_estimated_gpu_resident_bytes"] == 128
'''

    cmd = ["uv", "run", "--directory", str(SCRIPT_DIR), "python", "-c", code]
    print("===========================================================")
    print(" Stage L0 Success Check: running residency self-test")
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
            f"Stage L0 self-test failed with exit code {rc}; log={runner_log}"
        )
    return trace_jsonl, runner_log


def summarize_trace(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    pools: Counter[str] = Counter()
    request_records = 0
    would_deny_records = 0
    denied_records = 0
    state_update_records = 0
    register_records = 0
    candidate_records = 0
    real_action_records = 0
    config: dict[str, Any] = {}
    summary: dict[str, Any] = {}

    real_actions = {
        "residency_offload_issued",
        "residency_prefetch_issued",
        "residency_sync",
    }
    for record in records:
        action = str(record.get("action", "unknown"))
        action_counts[action] += 1
        if action == "residency_config":
            config = record
        if action == "residency_summary":
            summary = record
        pool = record.get("pool")
        if isinstance(pool, str):
            pools[pool] += 1
        if action == "residency_register_region":
            register_records += 1
        if action == "residency_request":
            request_records += 1
            if record.get("would_deny") is True:
                would_deny_records += 1
            if record.get("allowed") is False:
                denied_records += 1
        if action == "residency_state_update":
            state_update_records += 1
        if action == "residency_would_evict_candidate":
            candidate_records += 1
        if action in real_actions:
            real_action_records += 1

    return {
        "trace_records": len(records),
        "action_counts": dict(action_counts),
        "pools": dict(pools),
        "request_records": request_records,
        "would_deny_records": would_deny_records,
        "denied_records": denied_records,
        "state_update_records": state_update_records,
        "register_records": register_records,
        "candidate_records": candidate_records,
        "real_action_records": real_action_records,
        "config": config,
        "summary": summary,
    }


def validate(
    *,
    trace_jsonl: Path,
    runner_log: Path | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    trace_summary = summarize_trace(records)
    config = trace_summary["config"]
    summary = trace_summary["summary"]
    pools = set(trace_summary["pools"])
    pool_region_counts = summary.get("pool_region_counts") or {}

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
            "estimated_residency_semantics_present",
            "estimated_residency_semantics" in config,
            f"config={config}",
        ),
        check(
            "register_records_present",
            trace_summary["register_records"] >= 4,
            f"register_records={trace_summary['register_records']}",
        ),
        check(
            "all_pools_visible",
            {"weights", "kv", "scratch"}.issubset(pools),
            f"pools={sorted(pools)}",
        ),
        check(
            "all_pool_region_counts_present",
            all((pool_region_counts.get(pool) or 0) > 0
                for pool in ("weights", "kv", "scratch")),
            f"pool_region_counts={pool_region_counts}",
        ),
        check(
            "request_records_present",
            trace_summary["request_records"] >= 3,
            f"request_records={trace_summary['request_records']}",
        ),
        check(
            "would_deny_records_present",
            trace_summary["would_deny_records"] >= 2,
            f"would_deny_records={trace_summary['would_deny_records']}",
        ),
        check(
            "deny_records_present_in_enforce_mode",
            args.mode != "enforce" or trace_summary["denied_records"] >= 2,
            f"denied_records={trace_summary['denied_records']}",
        ),
        check(
            "state_updates_present",
            trace_summary["state_update_records"] >= 2,
            f"state_update_records={trace_summary['state_update_records']}",
        ),
        check(
            "same_pool_candidate_present",
            trace_summary["candidate_records"] > 0,
            f"candidate_records={trace_summary['candidate_records']}",
        ),
        check("summary_present", bool(summary), f"summary={summary}"),
        check(
            "estimated_bytes_accounting_correct",
            summary.get("weights_estimated_gpu_resident_bytes") == 640
            and summary.get("kv_estimated_gpu_resident_bytes") == 768
            and summary.get("scratch_estimated_gpu_resident_bytes") == 128,
            f"summary={summary}",
        ),
        check(
            "no_real_actions",
            trace_summary["real_action_records"] == 0,
            f"real_action_records={trace_summary['real_action_records']}",
        ),
    ]

    if runner_log and runner_log.is_file():
        text = runner_log.read_text(encoding="utf-8", errors="replace").lower()
        checks.append(
            check(
                "runner_log_clean",
                "traceback" not in text and "failed" not in text,
                f"runner_log={runner_log}",
            )
        )

    passed = all(item["passed"] for item in checks)
    output_json = Path(args.output_json) if args.output_json else (
        trace_jsonl.parent / "stage_l_success_check.json"
    )
    result = {
        "passed": passed,
        "trace_jsonl": str(trace_jsonl),
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
    print(f" Stage L0 Success Check: {status}")
    print(f" Trace JSONL: {result['trace_jsonl']}")
    print("===========================================================")
    print(f"- pools={trace['pools']}")
    print(f"- action_counts={trace['action_counts']}")
    print(
        "- records="
        f"register={trace['register_records']} "
        f"requests={trace['request_records']} "
        f"would_deny={trace['would_deny_records']} "
        f"denied={trace['denied_records']} "
        f"state_updates={trace['state_update_records']} "
        f"candidates={trace['candidate_records']} "
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
    if args.skip_run:
        trace_jsonl = resolve_trace(args)
        if trace_jsonl is None:
            raise SystemExit("--skip-run requires --trace-jsonl or --run-dir")
    else:
        trace_jsonl, runner_log = run_self_test(args, run_dir)
    result = validate(trace_jsonl=trace_jsonl, runner_log=runner_log, args=args)
    print_summary(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
