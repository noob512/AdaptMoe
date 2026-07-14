#!/usr/bin/env python3
"""Validate Phase U1 unified HBM residency trace-only controller.

The check is GPU-free.  It registers synthetic KV and expert residency regions,
drives the U1 controller APIs, verifies that every required reconstruction
field is present, and compares a small block-manager loop with U1 disabled vs
enabled to confirm that allocation behavior is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Phase U1 unified residency trace-only controller."
    )
    parser.add_argument("--run-dir")
    parser.add_argument("--trace-jsonl")
    parser.add_argument("--residency-jsonl")
    parser.add_argument("--output-json")
    return parser.parse_args()


def default_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    return Path(f"/tmp/vllm_unified_hbm_phase1_trace_only_{stamp}")


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


def run_subprocess(
    *,
    code: str,
    env: dict[str, str],
    log_path: Path,
) -> list[dict[str, Any]]:
    cmd = [
        "uv",
        "run",
        "--directory",
        str(SCRIPT_DIR),
        "python",
        "-c",
        code,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    json_lines: list[dict[str, Any]] = []
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            try:
                json_lines.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        rc = process.wait()
    if rc != 0:
        raise SystemExit(f"subprocess failed with exit code {rc}; log={log_path}")
    return json_lines


def run_trace_self_test(
    *,
    run_dir: Path,
    trace_jsonl: Path,
    residency_jsonl: Path,
) -> Path:
    for path in (trace_jsonl, residency_jsonl):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    code = r'''
import json

from vllm.device_allocator.uvm_pool_residency import (
    register_uvm_residency_region,
)
from vllm.device_allocator.uvm_unified_residency_controller import (
    get_uvm_unified_residency_controller,
)


def register_objects():
    register_uvm_residency_region(
        pool="kv",
        region_id="kv:0:1",
        ptr=None,
        bytes=65536,
        state="gpu",
        evictable=False,
        active=True,
        metadata={
            "stage": "stage_u1",
            "source": "synthetic_phase_u1",
            "group_id": 0,
            "block_id": 1,
            "ref_cnt": 1,
            "lifecycle_state": "used_free",
            "metadata_only": True,
        },
    )
    register_uvm_residency_region(
        pool="kv",
        region_id="kv:0:2",
        ptr=None,
        bytes=65536,
        state="gpu",
        evictable=True,
        active=False,
        metadata={
            "stage": "stage_u1",
            "source": "synthetic_phase_u1",
            "group_id": 0,
            "block_id": 2,
            "ref_cnt": 0,
            "lifecycle_state": "never_used",
            "in_free_queue": True,
            "metadata_only": True,
        },
    )
    register_uvm_residency_region(
        pool="weights",
        region_id="expert:0:3:w13",
        ptr=None,
        bytes=4 * 1024 * 1024,
        state="gpu",
        evictable=False,
        active=True,
        metadata={
            "stage": "stage_u1",
            "source": "synthetic_phase_u1",
            "is_moe_expert": True,
            "layer_id": 0,
            "expert_id": 3,
            "role": "w13",
            "protected": True,
            "priority": "hot",
        },
    )
    register_uvm_residency_region(
        pool="weights",
        region_id="expert:0:4:w2",
        ptr=None,
        bytes=4 * 1024 * 1024,
        state="gpu",
        evictable=False,
        active=False,
        metadata={
            "stage": "stage_u1",
            "source": "synthetic_phase_u1",
            "is_moe_expert": True,
            "layer_id": 0,
            "expert_id": 4,
            "role": "w2",
            "protected_next": True,
            "priority": "hot",
        },
    )
    register_uvm_residency_region(
        pool="weights",
        region_id="expert:1:9:w13",
        ptr=None,
        bytes=4 * 1024 * 1024,
        state="gpu",
        evictable=True,
        active=False,
        metadata={
            "stage": "stage_u1",
            "source": "synthetic_phase_u1",
            "is_moe_expert": True,
            "layer_id": 1,
            "expert_id": 9,
            "role": "w13",
            "init_cold": True,
        },
    )


register_objects()
controller = get_uvm_unified_residency_controller()
service_snapshot = controller.on_service_start()
step_snapshot = controller.on_scheduler_step(
    reason="synthetic_scheduler_step",
    kv_pressure={
        "running_request_count": 1,
        "waiting_request_count": 2,
        "projected_new_kv_blocks": 3,
        "free_gpu_resident_kv_blocks": 1,
        "cpu_backed_kv_blocks": 0,
        "active_kv_blocks": 1,
    },
    routing_plan={
        "current_layer": 0,
        "predicted_experts": [{"layer_id": 0, "expert_id": 4}],
    },
)
controller.resize_kv_target(
    4,
    old_target_blocks=2,
    reason="synthetic_kv_grow",
)
controller.resize_kv_target(
    1,
    old_target_blocks=4,
    reason="synthetic_kv_shrink",
)
controller.reclaim_expert_for_kv(
    2 * 1024 * 1024,
    reason="synthetic_kv_pressure",
)
controller.before_layer(0, predicted_experts=[{"layer_id": 0, "expert_id": 4}])
controller.after_layer(0, actual_experts=[{"layer_id": 0, "expert_id": 3}])

print(json.dumps({
    "marker": "phase_u1_trace_only_registry_ok",
    "service_kv_resident_bytes": service_snapshot.get("kv_resident_bytes"),
    "step_kv_resident_bytes": step_snapshot.get("kv_resident_bytes"),
    "step_expert_resident_bytes": step_snapshot.get("expert_resident_bytes"),
    "protected_object_count": step_snapshot.get("protected_object_count"),
    "evictable_object_count": step_snapshot.get("evictable_object_count"),
}, sort_keys=True))
'''
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_MODE": "trace_only",
            "VLLM_UVM_RESIDENCY_STAGE": "stage_u1",
            "VLLM_UVM_RESIDENCY_TRACE_ENABLE": "1",
            "VLLM_UVM_RESIDENCY_TRACE_FILE": str(residency_jsonl),
            "VLLM_UVM_UNIFIED_RESIDENCY_ENABLE": "1",
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE": str(trace_jsonl),
            "VLLM_UVM_UNIFIED_MANAGED_HBM_BUDGET_BYTES": str(64 * 1024 * 1024),
            "VLLM_UVM_UNIFIED_RUNTIME_SAFETY_MARGIN_BYTES": str(1024 * 1024),
            "VLLM_UVM_EXPERT_RESERVED_SLOT_COUNT": "2",
            "VLLM_UVM_EXPERT_RESERVED_SLOT_BYTES": str(4 * 1024 * 1024),
        }
    )
    runner_log = run_dir / "phase1_trace_only_registry_runner.log"
    run_subprocess(code=code, env=env, log_path=runner_log)
    return runner_log


def run_perf_case(
    *,
    run_dir: Path,
    enabled: bool,
) -> dict[str, Any]:
    trace_file = run_dir / (
        "phase1_perf_unified_enabled.jsonl"
        if enabled
        else "phase1_perf_unified_disabled.jsonl"
    )
    try:
        trace_file.unlink()
    except FileNotFoundError:
        pass
    code = r'''
import json
import time
import torch

from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

iterations = 20000
config = KVCacheConfig(
    num_blocks=64,
    kv_cache_tensors=[KVCacheTensor(size=64 * 1024, shared_by=["layer.0"])],
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
sequence_sum = 0
sequence_head = []
started = time.perf_counter()
for index in range(iterations):
    block = pool.get_new_blocks(1)[0]
    if index < 16:
        sequence_head.append(block.block_id)
    sequence_sum += block.block_id
    pool.free_blocks([block])
elapsed = time.perf_counter() - started
print(json.dumps({
    "elapsed_seconds": elapsed,
    "iterations": iterations,
    "sequence_head": sequence_head,
    "sequence_sum": sequence_sum,
}, sort_keys=True))
'''
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    env.update(
        {
            "VLLM_UVM_KV_RUNTIME_ENABLE": "0",
            "VLLM_UVM_UNIFIED_RESIDENCY_ENABLE": "1" if enabled else "0",
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_FILE": str(trace_file),
            "VLLM_UVM_UNIFIED_RESIDENCY_TRACE_OBJECTS": "0",
        }
    )
    log_path = run_dir / (
        "phase1_perf_unified_enabled.log"
        if enabled
        else "phase1_perf_unified_disabled.log"
    )
    records = run_subprocess(code=code, env=env, log_path=log_path)
    if not records:
        raise SystemExit(f"missing perf JSON output; log={log_path}")
    record = records[-1]
    record["trace_file"] = str(trace_file)
    record["log_path"] = str(log_path)
    record["enabled"] = enabled
    return record


def summarize_trace(
    *,
    trace_jsonl: Path,
    residency_jsonl: Path,
    runner_log: Path,
    perf_disabled: dict[str, Any],
    perf_enabled: dict[str, Any],
) -> dict[str, Any]:
    records = read_jsonl(trace_jsonl)
    residency_records = read_jsonl(residency_jsonl)
    by_action: dict[str, int] = {}
    for record in records:
        action = str(record.get("action") or "")
        by_action[action] = by_action.get(action, 0) + 1

    object_states = [
        record for record in records
        if record.get("action") == "unified_object_state"
    ]
    budget_updates = [
        record for record in records
        if record.get("action") == "unified_budget_update"
    ]
    all_objects = [
        obj
        for record in object_states
        for obj in (record.get("objects") or [])
    ]
    has_kv = any(obj.get("object_type") == "kv" for obj in all_objects)
    has_expert = any(obj.get("object_type") == "expert" for obj in all_objects)
    required_fields = {
        "kv_resident_bytes",
        "expert_resident_bytes",
        "reserved_slot_budget",
        "runtime_safety_margin",
        "protected_objects",
        "evictable_objects",
    }
    reconstructable = all(
        required_fields.issubset(record.keys()) for record in object_states
    )
    unified_records = [
        record for record in records
        if str(record.get("action") or "").startswith("unified_")
    ]
    non_trace_only = [
        record.get("action") for record in unified_records
        if record.get("trace_only") is not True
    ]
    forbidden_actions = [
        record.get("action") for record in records
        if str(record.get("action") or "") in {
            "unified_expert_reclaim_executed",
            "unified_expert_prefetch_executed",
            "unified_kv_resize_executed",
            "residency_driver_pool_reclaim",
        }
    ]
    perf_base = float(perf_disabled.get("elapsed_seconds") or 0.0)
    perf_u1 = float(perf_enabled.get("elapsed_seconds") or 0.0)
    perf_ratio = perf_u1 / perf_base if perf_base > 0 else 0.0
    perf_abs_delta = abs(perf_u1 - perf_base)
    same_sequence = (
        perf_disabled.get("sequence_head") == perf_enabled.get("sequence_head")
        and perf_disabled.get("sequence_sum") == perf_enabled.get("sequence_sum")
    )
    perf_close = perf_ratio <= 1.25 or perf_abs_delta <= 0.05

    checks = [
        check(
            "runner_completed",
            "phase_u1_trace_only_registry_ok" in runner_log.read_text(
                encoding="utf-8", errors="replace"
            ),
            str(runner_log),
        ),
        check("unified_config_present", by_action.get("unified_config", 0) >= 1,
              f"count={by_action.get('unified_config', 0)}"),
        check("budget_update_present", len(budget_updates) >= 1,
              f"count={len(budget_updates)}"),
        check("object_state_present", len(object_states) >= 1,
              f"count={len(object_states)}"),
        check("scheduler_step_present",
              by_action.get("unified_scheduler_step", 0) >= 1,
              f"count={by_action.get('unified_scheduler_step', 0)}"),
        check("kv_object_present", has_kv, "object_type=kv"),
        check("expert_object_present", has_expert, "object_type=expert"),
        check("reconstructable_step_fields", reconstructable,
              f"required={sorted(required_fields)}"),
        check("protected_objects_present",
              any(record.get("protected_objects") for record in object_states),
              "protected_objects list non-empty"),
        check("evictable_objects_present",
              any(record.get("evictable_objects") for record in object_states),
              "evictable_objects list non-empty"),
        check("kv_grow_request_present",
              by_action.get("unified_kv_grow_request", 0) >= 1,
              f"count={by_action.get('unified_kv_grow_request', 0)}"),
        check("kv_shrink_request_present",
              by_action.get("unified_kv_shrink_request", 0) >= 1,
              f"count={by_action.get('unified_kv_shrink_request', 0)}"),
        check("expert_reclaim_request_trace_only",
              by_action.get("unified_expert_reclaim_request", 0) >= 1,
              f"count={by_action.get('unified_expert_reclaim_request', 0)}"),
        check("expert_prefetch_request_trace_only",
              by_action.get("unified_expert_prefetch_request", 0) >= 1,
              f"count={by_action.get('unified_expert_prefetch_request', 0)}"),
        check("all_unified_actions_trace_only", not non_trace_only,
              f"non_trace_only={non_trace_only[:8]}"),
        check("no_forbidden_real_migration_actions", not forbidden_actions,
              f"forbidden={forbidden_actions[:8]}"),
        check("perf_allocation_sequence_unchanged", same_sequence,
              f"disabled={perf_disabled.get('sequence_head')}, "
              f"enabled={perf_enabled.get('sequence_head')}"),
        check("perf_baseline_close", perf_close,
              f"disabled={perf_base:.6f}s enabled={perf_u1:.6f}s "
              f"ratio={perf_ratio:.3f} abs_delta={perf_abs_delta:.6f}s"),
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "trace_jsonl": str(trace_jsonl),
        "residency_jsonl": str(residency_jsonl),
        "runner_log": str(runner_log),
        "trace_record_count": len(records),
        "residency_record_count": len(residency_records),
        "action_counts": by_action,
        "checks": checks,
        "perf_disabled": perf_disabled,
        "perf_enabled": perf_enabled,
        "perf_ratio_enabled_vs_disabled": perf_ratio,
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir()
    trace_jsonl = (
        Path(args.trace_jsonl)
        if args.trace_jsonl
        else run_dir / "vllm_uvm_unified_residency_phase1.jsonl"
    )
    residency_jsonl = (
        Path(args.residency_jsonl)
        if args.residency_jsonl
        else run_dir / "vllm_uvm_residency_phase1.jsonl"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print("===========================================================")
    print(" Unified HBM Residency Phase U1: trace-only controller")
    print(f" Output dir: {run_dir}")
    print(f" Unified trace JSONL: {trace_jsonl}")
    print(f" Residency trace JSONL: {residency_jsonl}")
    print("===========================================================")

    runner_log = run_trace_self_test(
        run_dir=run_dir,
        trace_jsonl=trace_jsonl,
        residency_jsonl=residency_jsonl,
    )
    perf_disabled = run_perf_case(run_dir=run_dir, enabled=False)
    perf_enabled = run_perf_case(run_dir=run_dir, enabled=True)
    summary = summarize_trace(
        trace_jsonl=trace_jsonl,
        residency_jsonl=residency_jsonl,
        runner_log=runner_log,
        perf_disabled=perf_disabled,
        perf_enabled=perf_enabled,
    )
    output_json = (
        Path(args.output_json)
        if args.output_json
        else run_dir / "phase1_trace_only_summary.json"
    )
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
