#!/usr/bin/env python3
"""Validate Phase U2 manual unified HBM budget rebalance.

The check has two parts:

1. A GPU-free vLLM control-file self-test that exercises the unified controller
   commands against the KV dynamic-pool policy and residency ledger.
2. A real Stage-N driver smoke with LOW_REBALANCE=1 that verifies pool-state
   resident bytes move in the expected directions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]


POOL_RE = re.compile(
    r"phase_n_3gb_pool label=(?P<label>\S+) "
    r"kv_budget=(?P<kv_budget>\d+) kv_resident=(?P<kv_resident>\d+) "
    r"kv_pending=(?P<kv_pending>\d+) kv_over=(?P<kv_over>\d+) "
    r"expert_budget=(?P<expert_budget>\d+) "
    r"expert_resident=(?P<expert_resident>\d+) "
    r"expert_pending=(?P<expert_pending>\d+) "
    r"expert_over=(?P<expert_over>\d+) "
    r"total_resident=(?P<total_resident>\d+)"
)
U2_FINAL_RE = re.compile(
    r"phase_n_3gb_u2_final .*total_ok=(?P<total_ok>\d+) "
    r"kv_resident_initial=(?P<kv_initial>\d+) "
    r"kv_resident_high=(?P<kv_high>\d+) "
    r"kv_resident_low=(?P<kv_low>\d+) "
    r"expert_resident_initial=(?P<expert_initial>\d+) "
    r"expert_resident_high=(?P<expert_high>\d+) "
    r"expert_resident_low=(?P<expert_low>\d+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase U2 manual budget rebalance."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--output-json")
    parser.add_argument("--skip-driver", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_unified_hbm_phase2_manual_rebalance_{stamp}")


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


def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_streamed(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_seconds: int | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                handle.write(line)
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            handle.write(f"\nTIMEOUT after {timeout_seconds}s\n")
            return 124


def run_control_file_self_test(run_dir: Path) -> dict[str, Any]:
    trace_jsonl = run_dir / "phase2_unified_control_trace.jsonl"
    residency_jsonl = run_dir / "phase2_residency_trace.jsonl"
    runtime_jsonl = run_dir / "phase2_kv_runtime_trace.jsonl"
    control_file = run_dir / "phase2_unified_control.txt"
    runner_log = run_dir / "phase2_control_file_runner.log"
    for path in (trace_jsonl, residency_jsonl, runtime_jsonl):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    control_file.parent.mkdir(parents=True, exist_ok=True)

    code = r'''
import json
import time
import torch
from pathlib import Path

from vllm.device_allocator.uvm_pool_residency import (
    register_uvm_residency_region,
)
from vllm.device_allocator.uvm_unified_residency_controller import (
    get_uvm_unified_residency_controller,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

control_file = Path(__import__("os").environ["VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE"])
control_file.write_text("", encoding="utf-8")

config = KVCacheConfig(
    num_blocks=7,
    kv_cache_tensors=[KVCacheTensor(size=7 * 1024, shared_by=["layer.0"])],
    kv_cache_groups=[
        KVCacheGroupSpec(
            layer_names=["layer.0"],
            kv_cache_spec=FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        )
    ],
)
manager = KVCacheManager(
    config,
    max_model_len=128,
    hash_block_size=16,
    enable_caching=False,
)
pool = manager.block_pool

for idx, state in enumerate(("gpu", "gpu", "gpu")):
    register_uvm_residency_region(
        pool="weights",
        region_id=f"expert:0:{idx}:w13",
        ptr=None,
        bytes=4 * 1024 * 1024,
        state=state,
        evictable=(idx == 2),
        active=(idx == 0),
        metadata={
            "stage": "stage_u2",
            "source": "synthetic_phase_u2",
            "is_moe_expert": True,
            "layer_id": 0,
            "expert_id": idx,
            "role": "w13",
            "protected": idx == 0,
            "init_cold": idx == 2,
        },
    )

controller = get_uvm_unified_residency_controller()

# Apply the initial small KV target from VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS=2.
initial_free = pool.get_num_free_blocks()
initial = controller.snapshot(reason="phase2_initial_small_kv")

control_file.write_text(
    "\n".join([
        "kv_target_blocks=4",
        "weights_budget_bytes=33554432",
        "reclaim_pool=weights required_bytes=4194304",
        "prefetch_kv_blocks=0:3,0:4",
    ]) + "\n",
    encoding="utf-8",
)
time.sleep(0.02)
high_free = pool.get_num_free_blocks()
high = controller.snapshot(reason="phase2_high_kv_pressure")

control_file.write_text(
    "\n".join([
        "kv_target_blocks=1",
        "weights_budget_bytes=67108864",
    ]) + "\n",
    encoding="utf-8",
)
time.sleep(0.02)
low_free = pool.get_num_free_blocks()
low = controller.snapshot(reason="phase2_low_kv_pressure")

print(json.dumps({
    "marker": "phase_u2_control_file_ok",
    "initial_free_blocks": initial_free,
    "high_free_blocks": high_free,
    "low_free_blocks": low_free,
    "initial_kv_resident_bytes": initial.get("kv_resident_bytes"),
    "high_kv_resident_bytes": high.get("kv_resident_bytes"),
    "low_kv_resident_bytes": low.get("kv_resident_bytes"),
    "initial_weights_budget_bytes": initial.get("weights_budget_bytes"),
    "high_weights_budget_bytes": high.get("weights_budget_bytes"),
    "low_weights_budget_bytes": low.get("weights_budget_bytes"),
}, sort_keys=True))
'''
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_KV_RUNTIME_ENABLE": "1",
            "VLLM_UVM_KV_RUNTIME_MODE": "trace_only",
            "VLLM_UVM_KV_RUNTIME_TRACE_FILE": str(runtime_jsonl),
            "VLLM_UVM_KV_DYNAMIC_POOL_ENABLE": "1",
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS": "2",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_FREE_BLOCKS": "1",
            "VLLM_UVM_KV_PREFER_GPU_RESIDENT_ALLOW_CACHED": "1",
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_u2",
            "VLLM_UVM_RESIDENCY_MODE": "trace_only",
            "VLLM_UVM_RESIDENCY_KV_METADATA_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(residency_jsonl),
            "VLLM_UVM_UNIFIED_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_FILE": str(control_file),
            "VLLM_UVM_UNIFIED_RESIDENCY_CONTROL_ENABLE": "1",
            "VLLM_UVM_UNIFIED_RESIDENCY_DRIVER_POOL_STATE": "1",
            "VLLM_UVM_UNIFIED_DRIVER_BUDGET_MODE": "trace_only",
        }
    )
    rc = run_streamed(
        [
            "uv",
            "run",
            "--directory",
            str(SCRIPT_DIR),
            "python",
            "-c",
            code,
        ],
        cwd=SCRIPT_DIR,
        env=env,
        log_path=runner_log,
    )
    records = read_jsonl(trace_jsonl)
    action_counts: dict[str, int] = {}
    snapshots = {}
    for record in records:
        action = str(record.get("action") or "")
        action_counts[action] = action_counts.get(action, 0) + 1
        reason = record.get("reason")
        if action == "unified_object_state" and reason:
            snapshots[str(reason)] = record
    initial = snapshots.get("phase2_initial_small_kv") or {}
    high = snapshots.get("phase2_high_kv_pressure") or {}
    low = snapshots.get("phase2_low_kv_pressure") or {}
    checks = [
        check("control_runner_completed", rc == 0, f"rc={rc} log={runner_log}"),
        check("control_marker_present",
              "phase_u2_control_file_ok" in runner_log.read_text(
                  encoding="utf-8", errors="replace"),
              str(runner_log)),
        check("control_file_applied",
              action_counts.get("unified_control_file_applied", 0) >= 3,
              f"count={action_counts.get('unified_control_file_applied', 0)}"),
        check("kv_target_command_applied",
              action_counts.get("unified_kv_target_applied", 0) >= 2,
              f"count={action_counts.get('unified_kv_target_applied', 0)}"),
        check("weights_budget_command_applied",
              action_counts.get("unified_weights_budget_update", 0) >= 2,
              f"count={action_counts.get('unified_weights_budget_update', 0)}"),
        check("reclaim_command_seen",
              action_counts.get("unified_driver_reclaim_pool", 0) >= 1,
              f"count={action_counts.get('unified_driver_reclaim_pool', 0)}"),
        check("prefetch_command_seen",
              action_counts.get("unified_prefetch_kv_blocks", 0) >= 1,
              f"count={action_counts.get('unified_prefetch_kv_blocks', 0)}"),
        check("ledger_kv_rises_on_high_pressure",
              int(high.get("kv_resident_bytes") or 0)
              > int(initial.get("kv_resident_bytes") or 0),
              f"initial={initial.get('kv_resident_bytes')} high={high.get('kv_resident_bytes')}"),
        check("ledger_kv_drops_on_low_pressure",
              int(low.get("kv_resident_bytes") or 0)
              < int(high.get("kv_resident_bytes") or 0),
              f"high={high.get('kv_resident_bytes')} low={low.get('kv_resident_bytes')}"),
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "trace_jsonl": str(trace_jsonl),
        "residency_jsonl": str(residency_jsonl),
        "runtime_jsonl": str(runtime_jsonl),
        "control_file": str(control_file),
        "runner_log": str(runner_log),
        "action_counts": action_counts,
        "snapshots": {
            "initial": {
                "kv_resident_bytes": initial.get("kv_resident_bytes"),
                "weights_budget_bytes": initial.get("weights_budget_bytes"),
            },
            "high": {
                "kv_resident_bytes": high.get("kv_resident_bytes"),
                "weights_budget_bytes": high.get("weights_budget_bytes"),
            },
            "low": {
                "kv_resident_bytes": low.get("kv_resident_bytes"),
                "weights_budget_bytes": low.get("weights_budget_bytes"),
            },
        },
    }


def parse_driver_pool_state(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    pools: dict[str, dict[str, int]] = {}
    for match in POOL_RE.finditer(text):
        label = match.group("label")
        pools[label] = {
            key: int(value)
            for key, value in match.groupdict().items()
            if key != "label"
        }
    final_match = U2_FINAL_RE.search(text)
    final = (
        {
            key: int(value)
            for key, value in final_match.groupdict().items()
        }
        if final_match
        else {}
    )
    return {"pools": pools, "u2_final": final}


def run_driver_smoke(run_dir: Path, timeout_seconds: int) -> dict[str, Any]:
    stage_dir = run_dir / "stage_n_low_rebalance"
    log_path = stage_dir / "driver_smoke_wrapper.log"
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(stage_dir),
            "LOW_REBALANCE": "1",
        }
    )
    rc = run_streamed(
        ["bash", "workloads/vllm/run_stage_n_3gb_rebalance_smoke.sh"],
        cwd=ROOT,
        env=env,
        log_path=log_path,
        timeout_seconds=timeout_seconds,
    )
    smoke_log = stage_dir / "smoke.log"
    parsed = parse_driver_pool_state(smoke_log if smoke_log.exists() else log_path)
    pools = parsed.get("pools") or {}
    initial = pools.get("initial") or {}
    high = pools.get("after_rebalance") or {}
    low = pools.get("low_rebalance") or {}
    final = parsed.get("u2_final") or {}
    checks = [
        check("driver_smoke_completed", rc == 0, f"rc={rc} log={log_path}"),
        check("driver_u2_final_present", bool(final), f"final={final}"),
        check("driver_u2_total_ok", int(final.get("total_ok") or 0) == 1,
              f"final={final}"),
        check("driver_kv_resident_rises",
              int(high.get("kv_resident") or 0)
              > int(initial.get("kv_resident") or 0),
              f"initial={initial.get('kv_resident')} high={high.get('kv_resident')}"),
        check("driver_weights_resident_drops_after_reclaim",
              int(high.get("expert_resident") or 0)
              < int(initial.get("expert_resident") or 0),
              f"initial={initial.get('expert_resident')} high={high.get('expert_resident')}"),
        check("driver_kv_resident_drops_on_low_target",
              int(low.get("kv_resident") or 0)
              < int(high.get("kv_resident") or 0),
              f"high={high.get('kv_resident')} low={low.get('kv_resident')}"),
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "run_root": str(stage_dir),
        "wrapper_log": str(log_path),
        "smoke_log": str(smoke_log),
        "parsed": parsed,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    print("===========================================================")
    print(" Unified HBM Residency Phase U2: manual rebalance")
    print(f" Output dir: {run_dir}")
    print("===========================================================")

    control = run_control_file_self_test(run_dir)
    driver = {
        "passed": True,
        "skipped": True,
        "checks": [check("driver_smoke_skipped", True, "--skip-driver")],
    } if args.skip_driver else run_driver_smoke(run_dir, args.timeout_seconds)
    summary = {
        "passed": bool(control.get("passed")) and bool(driver.get("passed")),
        "run_dir": str(run_dir),
        "control": control,
        "driver": driver,
    }
    output_json = (
        Path(args.output_json)
        if args.output_json
        else run_dir / "phase2_manual_rebalance_summary.json"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
