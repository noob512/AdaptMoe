#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def gpu_memory_snapshot() -> dict[str, int | str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    try:
        return {
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_total_mib": int(parts[2]),
        }
    except ValueError:
        return None


def parse_token_sequence(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("max token values must be positive")
        values.append(value)
    if not values:
        raise ValueError("empty max token sequence")
    return values


def load_request_fixture(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_requests = payload.get("requests")
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError(f"request fixture has no requests: {path}")

    requests: list[dict[str, Any]] = []
    for index, item in enumerate(raw_requests):
        if not isinstance(item, dict):
            raise ValueError(f"request fixture item {index} is not an object")
        prompt = item.get("prompt")
        max_tokens = item.get("max_tokens")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"request fixture item {index} has no prompt")
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"request fixture item {index} has invalid max_tokens")
        requests.append(
            {
                "request": int(item.get("request", index)),
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
        )
    return requests


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def write_result(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _write_sysfs_param(
    path: Path,
    value: int,
    *,
    use_sudo: bool,
) -> None:
    if use_sudo:
        subprocess.run(
            ["sudo", "-n", "tee", str(path)],
            input=f"{int(value)}\n",
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        return
    path.write_text(f"{int(value)}\n", encoding="utf-8")


def apply_pre_request_manual_fault_guard() -> dict[str, Any] | None:
    mode_text = os.environ.get("VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_MODE", "").strip()
    if not mode_text:
        return None

    param_dir = Path(
        os.environ.get(
            "VLLM_UVM_MANUAL_FAULT_PARAM_DIR",
            "/sys/module/nvidia_uvm/parameters",
        )
    )
    use_sudo = os.environ.get("VLLM_UVM_MANUAL_FAULT_USE_SUDO", "1") != "0"
    pool_mask_text = os.environ.get(
        "VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_POOL_MASK",
        "0",
    ).strip()
    log_enable_text = os.environ.get(
        "VLLM_UVM_MANUAL_FAULT_BEFORE_REQUEST_LOG_ENABLE",
        "1",
    ).strip()

    mode = int(mode_text)
    pool_mask = int(pool_mask_text)
    log_enable = int(log_enable_text)

    updates = [
        ("uvm_manual_migration_fault_pool_mask", pool_mask),
        ("uvm_manual_migration_fault_log_enable", log_enable),
        ("uvm_manual_migration_fault_mode", mode),
    ]
    for name, value in updates:
        _write_sysfs_param(param_dir / name, value, use_sudo=use_sudo)

    observed: dict[str, str] = {}
    for name, _value in updates:
        path = param_dir / name
        try:
            observed[name] = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            observed[name] = f"read_error:{exc!r}"

    return {
        "param_dir": str(param_dir),
        "use_sudo": use_sudo,
        "requested_mode": mode,
        "requested_pool_mask": pool_mask,
        "requested_log_enable": log_enable,
        "observed": observed,
    }


def apply_engine_core_kv_dynamic_pool_target(
    llm: Any,
    target_blocks: int,
    *,
    reason: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is None:
        return {
            "ok": False,
            "reason": "engine_core_unavailable",
            "requested_target_blocks": target_blocks,
        }

    if hasattr(engine_core, "call_utility"):
        result = engine_core.call_utility(
            "apply_uvm_kv_dynamic_pool_target",
            int(target_blocks),
            reason,
            metadata,
        )
    elif hasattr(engine_core, "apply_uvm_kv_dynamic_pool_target"):
        result = engine_core.apply_uvm_kv_dynamic_pool_target(
            int(target_blocks),
            reason,
            metadata,
        )
    elif (
        hasattr(engine_core, "engine_core")
        and hasattr(engine_core.engine_core, "apply_uvm_kv_dynamic_pool_target")
    ):
        result = engine_core.engine_core.apply_uvm_kv_dynamic_pool_target(
            int(target_blocks),
            reason,
            metadata,
        )
    else:
        return {
            "ok": False,
            "reason": "apply_utility_unavailable",
            "requested_target_blocks": target_blocks,
            "engine_core_type": type(engine_core).__name__,
        }

    if isinstance(result, dict):
        return result
    return {
        "ok": True,
        "reason": "non_dict_result",
        "requested_target_blocks": target_blocks,
        "result": result,
    }


def latest_active_residency_record(
    path: Path,
    marker: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(record.get("marker")) == marker:
            latest = record
    return latest


def latest_jsonl_record(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            latest = record
    return latest


def slot_backed_expert_roles(
    record: dict[str, Any] | None,
    *,
    layer_id: int | None,
) -> dict[int, set[str]]:
    if record is None:
        return {}
    prefetch_result = record.get("prefetch_result")
    if not isinstance(prefetch_result, dict) or not prefetch_result.get("ok"):
        return {}

    successful_region_ids = prefetch_result.get("successful_region_ids")
    roles_by_expert: dict[int, set[str]] = {}
    if isinstance(successful_region_ids, list):
        prefix = f"weights:{layer_id}:"
        for item in successful_region_ids:
            region_id = str(item)
            if layer_id is not None and not region_id.startswith(prefix):
                continue
            parts = region_id.split(":")
            if len(parts) < 4:
                continue
            try:
                expert_id = int(parts[2])
            except ValueError:
                continue
            roles_by_expert.setdefault(expert_id, set()).add(parts[3])

    assigned_slots = prefetch_result.get("assigned_slots")
    if isinstance(assigned_slots, list):
        for item in assigned_slots:
            if not isinstance(item, dict):
                continue
            try:
                item_layer_id = int(item.get("layer_id"))
                expert_id = int(item.get("expert_id"))
            except (TypeError, ValueError):
                continue
            if layer_id is not None and item_layer_id != layer_id:
                continue
            role = item.get("role")
            if isinstance(role, str) and role:
                roles_by_expert.setdefault(expert_id, set()).add(role)

    return roles_by_expert


def active_residency_all_gpu(
    record: dict[str, Any] | None,
    *,
    slot_record: dict[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    if record is None:
        return False, {"reason": "missing_marker_record"}
    records = record.get("active_residency_records")
    if not isinstance(records, list) or not records:
        return False, {"reason": "missing_active_residency_records"}

    layer_id = None
    try:
        layer_id = int(record.get("layer_id"))
    except (TypeError, ValueError):
        pass
    slot_roles_by_expert = slot_backed_expert_roles(
        slot_record,
        layer_id=layer_id,
    )
    state_counts: dict[str, int] = {}
    non_gpu: list[dict[str, Any]] = []
    slot_backed_non_gpu: list[dict[str, Any]] = []
    not_slot_backed_non_gpu: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        state = str(item.get("state"))
        state_counts[state] = state_counts.get(state, 0) + 1
        if state != "gpu_full":
            non_gpu.append(item)
            try:
                expert_id = int(item.get("global_expert_id"))
            except (TypeError, ValueError):
                not_slot_backed_non_gpu.append(item)
                continue
            roles = {
                str(role)
                for role in (item.get("roles") or [])
                if isinstance(role, str) and role
            }
            slot_roles = slot_roles_by_expert.get(expert_id, set())
            if (
                state == "cpu_present"
                and bool(item.get("query_ok"))
                and int(item.get("unknown_bytes") or 0) == 0
                and roles
                and roles.issubset(slot_roles)
            ):
                slot_backed_non_gpu.append(
                    {
                        "global_expert_id": expert_id,
                        "state": state,
                        "roles": sorted(roles),
                        "slot_roles": sorted(slot_roles),
                    }
                )
            else:
                not_slot_backed_non_gpu.append(item)

    ok = (
        len(records) > 0
        and not not_slot_backed_non_gpu
        and int(record.get("active_residency_unknown_bytes") or 0) == 0
        and not (record.get("active_residency_query_failed_experts") or [])
    )
    return ok, {
        "marker": record.get("marker"),
        "layer_id": record.get("layer_id"),
        "expert_records": len(records),
        "state_counts": state_counts,
        "queried_bytes": record.get("active_residency_queried_bytes"),
        "gpu_bytes": record.get("active_residency_gpu_bytes"),
        "cpu_bytes": record.get("active_residency_cpu_bytes"),
        "unknown_bytes": record.get("active_residency_unknown_bytes"),
        "slot_trace_present": slot_record is not None,
        "slot_backed_experts": sorted(slot_roles_by_expert),
        "slot_backed_non_gpu_records": slot_backed_non_gpu,
        "not_slot_backed_non_gpu_records": not_slot_backed_non_gpu,
        "non_gpu_records": non_gpu,
        "cpu_present_experts": record.get("active_residency_cpu_present_experts"),
        "non_full_gpu_experts": record.get("active_residency_non_full_gpu_experts"),
        "query_failed_experts": record.get("active_residency_query_failed_experts"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a sequence of vLLM generate() requests in one LLM process."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--prompt", default="Explain unified memory in one sentence.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--progress-jsonl", type=Path, default=None)
    parser.add_argument(
        "--request-fixture-json",
        type=Path,
        default=None,
        help="JSON file with requests: [{prompt, max_tokens}]. Uses exact prompts.",
    )
    parser.add_argument("--max-tokens-sequence", default="16,8")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--sync-between-requests", action="store_true")
    parser.add_argument("--kv-dynamic-pool-grow-after-request", type=int, default=-1)
    parser.add_argument("--kv-dynamic-pool-grow-target-blocks", type=int, default=0)
    parser.add_argument(
        "--kv-dynamic-pool-grow-reason",
        default="between_requests_manual_grow",
    )
    parser.add_argument("--require-active-gpu-trace-jsonl", type=Path, default=None)
    parser.add_argument("--require-active-slot-trace-jsonl", type=Path, default=None)
    parser.add_argument(
        "--require-active-marker",
        default="after_post_offload_slot_use_probe",
    )
    parser.add_argument("--require-active-all-gpu", action="store_true")
    parser.add_argument("--no-enforce-eager", action="store_true")
    args = parser.parse_args()

    request_fixture_path = args.request_fixture_json
    if request_fixture_path is not None:
        request_specs = load_request_fixture(request_fixture_path)
        max_tokens_sequence = [int(item["max_tokens"]) for item in request_specs]
    else:
        max_tokens_sequence = parse_token_sequence(args.max_tokens_sequence)
        request_specs = [
            {
                "request": request_index,
                "prompt": f"{args.prompt} [request {request_index}]",
                "max_tokens": max_tokens,
            }
            for request_index, max_tokens in enumerate(max_tokens_sequence)
        ]

    from vllm import LLM, SamplingParams

    kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "enforce_eager": not args.no_enforce_eager,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    if args.kv_cache_memory_bytes > 0:
        kwargs["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes

    started = time.time()
    append_jsonl(
        args.progress_jsonl,
        {
            "action": "load_begin",
            "timestamp": started,
            "max_tokens_sequence": max_tokens_sequence,
        },
    )
    llm = LLM(**kwargs)
    loaded = time.time()
    gpu_memory_after_load = gpu_memory_snapshot()

    result: dict[str, Any] = {
        "model": args.model,
        "prompt": args.prompt,
        "status": "running",
        "max_tokens_sequence": max_tokens_sequence,
        "request_fixture_json": (
            str(request_fixture_path) if request_fixture_path is not None else None
        ),
        "load_seconds": loaded - started,
        "request_results": [],
        "completed_requests": 0,
        "gpu_memory_after_load": gpu_memory_after_load,
    }
    write_result(args.output_json, result)
    append_jsonl(
        args.progress_jsonl,
        {
            "action": "load_end",
            "timestamp": loaded,
            "load_seconds": loaded - started,
            "gpu_memory_after_load": gpu_memory_after_load,
        },
    )

    if args.require_active_all_gpu:
        gate_record = None
        if args.require_active_gpu_trace_jsonl is not None:
            gate_record = latest_active_residency_record(
                args.require_active_gpu_trace_jsonl,
                args.require_active_marker,
            )
        slot_record = latest_jsonl_record(args.require_active_slot_trace_jsonl)
        gate_ok, gate_summary = active_residency_all_gpu(
            gate_record,
            slot_record=slot_record,
        )
        result["pre_generate_active_gate"] = {
            "ok": gate_ok,
            "required": True,
            "trace_jsonl": (
                str(args.require_active_gpu_trace_jsonl)
                if args.require_active_gpu_trace_jsonl is not None
                else None
            ),
            "slot_trace_jsonl": (
                str(args.require_active_slot_trace_jsonl)
                if args.require_active_slot_trace_jsonl is not None
                else None
            ),
            **gate_summary,
        }
        write_result(args.output_json, result)
        append_jsonl(
            args.progress_jsonl,
            {
                "action": "pre_generate_active_gate",
                "timestamp": time.time(),
                **result["pre_generate_active_gate"],
            },
        )
        if not gate_ok:
            result["status"] = "pre_generate_active_gate_failed"
            result["completed_requests"] = 0
            result["total_seconds"] = time.time() - started
            result["gpu_memory_after_gate_failed"] = gpu_memory_snapshot()
            write_result(args.output_json, result)
            append_jsonl(
                args.progress_jsonl,
                {
                    "action": "pre_generate_active_gate_failed",
                    "timestamp": time.time(),
                    "gpu_memory_after_gate_failed": result[
                        "gpu_memory_after_gate_failed"
                    ],
                },
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 4

    manual_fault_guard = apply_pre_request_manual_fault_guard()
    if manual_fault_guard is not None:
        result["pre_request_manual_fault_guard"] = manual_fault_guard
        write_result(args.output_json, result)
        append_jsonl(
            args.progress_jsonl,
            {
                "action": "pre_request_manual_fault_guard_applied",
                "timestamp": time.time(),
                **manual_fault_guard,
            },
        )

    request_results: list[dict[str, Any]] = []
    try:
        for request_index, request_spec in enumerate(request_specs):
            max_tokens = int(request_spec["max_tokens"])
            prompt = str(request_spec["prompt"])
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=args.temperature,
                ignore_eos=args.ignore_eos,
            )
            request_started = time.time()
            append_jsonl(
                args.progress_jsonl,
                {
                    "action": "request_begin",
                    "timestamp": request_started,
                    "request": request_index,
                    "max_tokens": max_tokens,
                    "prompt": prompt,
                },
            )
            result["status"] = "running_request"
            result["current_request"] = request_index
            result["current_request_max_tokens"] = max_tokens
            write_result(args.output_json, result)

            outputs = llm.generate([prompt], sampling_params)
            request_finished = time.time()
            generated = (
                outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            )
            token_ids = (
                outputs[0].outputs[0].token_ids
                if outputs and outputs[0].outputs
                else []
            )
            request_record = {
                "request": request_index,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "generate_seconds": request_finished - request_started,
                "output_tokens": len(token_ids),
                "generated": generated,
            }
            request_results.append(request_record)

            result["status"] = "running"
            result["request_results"] = request_results
            result["completed_requests"] = len(request_results)
            result["gpu_memory_after_last_request"] = gpu_memory_snapshot()
            write_result(args.output_json, result)
            append_jsonl(
                args.progress_jsonl,
                {
                    "action": "request_end",
                    "timestamp": request_finished,
                    **request_record,
                    "gpu_memory_after_request": result[
                        "gpu_memory_after_last_request"
                    ],
                },
            )
            if request_index + 1 < len(max_tokens_sequence):
                if (
                    args.kv_dynamic_pool_grow_after_request == request_index
                    and args.kv_dynamic_pool_grow_target_blocks > 0
                ):
                    maintenance_started = time.time()
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_kv_dynamic_pool_grow_begin",
                            "timestamp": maintenance_started,
                            "request": request_index,
                            "target_blocks": args.kv_dynamic_pool_grow_target_blocks,
                            "reason": args.kv_dynamic_pool_grow_reason,
                        },
                    )
                    import torch

                    torch.cuda.synchronize()
                    apply_result = apply_engine_core_kv_dynamic_pool_target(
                        llm,
                        args.kv_dynamic_pool_grow_target_blocks,
                        reason=args.kv_dynamic_pool_grow_reason,
                        metadata={
                            "request_index": request_index,
                            "completed_requests": len(request_results),
                            "script": "vllm_generate_sequence",
                        },
                    )
                    torch.cuda.synchronize()
                    maintenance_finished = time.time()
                    result["kv_dynamic_pool_between_requests_grow"] = {
                        "request": request_index,
                        "target_blocks": args.kv_dynamic_pool_grow_target_blocks,
                        "elapsed_seconds": (
                            maintenance_finished - maintenance_started
                        ),
                        "result": apply_result,
                        "gpu_memory_after_grow": gpu_memory_snapshot(),
                    }
                    write_result(args.output_json, result)
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_kv_dynamic_pool_grow_end",
                            "timestamp": maintenance_finished,
                            "request": request_index,
                            "target_blocks": args.kv_dynamic_pool_grow_target_blocks,
                            "elapsed_seconds": (
                                maintenance_finished - maintenance_started
                            ),
                            "result": apply_result,
                            "gpu_memory_after_grow": result[
                                "kv_dynamic_pool_between_requests_grow"
                            ]["gpu_memory_after_grow"],
                        },
                    )
                    if not apply_result.get("ok"):
                        result["status"] = "kv_dynamic_pool_grow_failed"
                        result["completed_requests"] = len(request_results)
                        result["total_seconds"] = time.time() - started
                        write_result(args.output_json, result)
                        print(json.dumps(result, indent=2, sort_keys=True))
                        return 5
                if args.sync_between_requests:
                    sync_started = time.time()
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_sync_begin",
                            "timestamp": sync_started,
                            "request": request_index,
                        },
                    )
                    import torch

                    torch.cuda.synchronize()
                    sync_finished = time.time()
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_sync_end",
                            "timestamp": sync_finished,
                            "request": request_index,
                            "elapsed_seconds": sync_finished - sync_started,
                        },
                    )
                if args.sleep_between_requests > 0:
                    sleep_started = time.time()
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_sleep_begin",
                            "timestamp": sleep_started,
                            "request": request_index,
                            "sleep_seconds": args.sleep_between_requests,
                        },
                    )
                    time.sleep(args.sleep_between_requests)
                    sleep_finished = time.time()
                    append_jsonl(
                        args.progress_jsonl,
                        {
                            "action": "between_requests_sleep_end",
                            "timestamp": sleep_finished,
                            "request": request_index,
                            "elapsed_seconds": sleep_finished - sleep_started,
                        },
                    )
    except Exception as exc:
        failed = time.time()
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["completed_requests"] = len(request_results)
        result["request_results"] = request_results
        result["total_seconds"] = failed - started
        result["gpu_memory_after_error"] = gpu_memory_snapshot()
        write_result(args.output_json, result)
        append_jsonl(
            args.progress_jsonl,
            {
                "action": "exception",
                "timestamp": failed,
                "error": repr(exc),
                "completed_requests": len(request_results),
                "gpu_memory_after_error": result["gpu_memory_after_error"],
            },
        )
        raise

    finished = time.time()
    request_seconds = [
        float(item["generate_seconds"]) for item in request_results
    ]
    result["status"] = "ok"
    result["completed_requests"] = len(request_results)
    result["request_results"] = request_results
    result["total_generate_seconds"] = sum(request_seconds)
    result["total_seconds"] = finished - started
    result["gpu_memory_after_generate"] = gpu_memory_snapshot()
    write_result(args.output_json, result)
    append_jsonl(
        args.progress_jsonl,
        {
            "action": "sequence_end",
            "timestamp": finished,
            "completed_requests": len(request_results),
            "total_generate_seconds": result["total_generate_seconds"],
            "total_seconds": result["total_seconds"],
            "gpu_memory_after_generate": result["gpu_memory_after_generate"],
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
