#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def gpu_memory_snapshot() -> dict[str, int | str] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,utilization.memory",
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
    if len(parts) < 6:
        return None
    try:
        return {
            "name": parts[0],
            "memory_used_mib": int(parts[1]),
            "memory_free_mib": int(parts[2]),
            "memory_total_mib": int(parts[3]),
            "utilization_gpu": int(parts[4]),
            "utilization_memory": int(parts[5]),
        }
    except ValueError:
        return None


def uvm_proactive_migration_snapshot() -> dict[str, str | None]:
    names = (
        "uvm_perf_prefetch_enable",
        "uvm_perf_access_counter_migration_enable",
        "uvm_perf_thrashing_enable",
    )
    values: dict[str, str | None] = {}
    try:
        values["nvidia_uvm_srcversion"] = Path(
            "/sys/module/nvidia_uvm/srcversion"
        ).read_text(encoding="utf-8").strip()
    except OSError:
        values["nvidia_uvm_srcversion"] = None
    for name in names:
        path = Path("/sys/module/nvidia_uvm/parameters") / name
        try:
            values[name] = path.read_text(encoding="utf-8").strip()
        except OSError:
            values[name] = None
    return values


def engine_cuda_memory_snapshot(llm: Any) -> list[dict[str, int]] | None:
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is None or not hasattr(engine_core, "call_utility"):
        return None
    try:
        result = engine_core.call_utility(
            "collective_rpc",
            "get_cuda_memory_info",
            30.0,
            (),
            None,
        )
    except Exception:
        return None
    if not isinstance(result, list):
        return None
    return [item for item in result if isinstance(item, dict)]


def parse_expert_ids(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def load_active_expert_probe(path: Path | None) -> dict[str, list[int]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "layers" in payload:
        layers = payload["layers"]
    else:
        layers = payload
    if not isinstance(layers, dict):
        raise ValueError("active expert probe must be a dict of layer_id -> experts")
    result: dict[str, list[int]] = {}
    for layer_id, experts in layers.items():
        if not isinstance(experts, list):
            raise ValueError(f"active expert probe layer {layer_id} is not a list")
        result[str(int(layer_id))] = sorted({int(expert) for expert in experts})
    return result


def load_weight_regions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def query_active_expert_residency(
    *,
    weight_regions_file: Path,
    active_probe: dict[str, list[int]],
    roles: set[str],
    device: int,
) -> dict[str, Any]:
    if not active_probe:
        return {
            "enabled": False,
            "reason": "no_active_probe",
        }

    from vllm.device_allocator.uvm import (
        driver_batch_query_ranges_residency_bytes,
    )

    records = load_weight_regions(weight_regions_file)
    by_layer_expert: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("is_moe_expert"):
            continue
        layer_id = record.get("layer_id")
        expert_id = record.get("expert_id")
        role = record.get("role")
        if layer_id is None or expert_id is None or role not in roles:
            continue
        by_layer_expert.setdefault((int(layer_id), int(expert_id)), []).append(record)

    expert_records: list[dict[str, Any]] = []
    total_queried = 0
    total_gpu = 0
    total_cpu = 0
    total_unknown = 0
    cpu_present: list[dict[str, int]] = []
    gpu_full: list[dict[str, int]] = []
    missing: list[dict[str, int]] = []

    for layer_id_raw, experts in sorted(active_probe.items(), key=lambda item: int(item[0])):
        layer_id = int(layer_id_raw)
        for expert_id in experts:
            ranges: list[tuple[int, int]] = []
            range_records = by_layer_expert.get((layer_id, int(expert_id)), [])
            for record in range_records:
                start = record.get("start_int", record.get("data_start_int"))
                end = record.get("end_int", record.get("data_end_int"))
                if start is None or end is None:
                    continue
                ranges.append((int(start), int(end) - int(start) + 1))
            if not ranges:
                missing.append({"layer_id": layer_id, "expert_id": int(expert_id)})
                expert_records.append(
                    {
                        "layer_id": layer_id,
                        "expert_id": int(expert_id),
                        "state": "missing_ranges",
                        "range_records": 0,
                    }
                )
                continue
            try:
                result = driver_batch_query_ranges_residency_bytes(ranges, device)
            except Exception as exc:
                result = {
                    "ok": False,
                    "status": -1,
                    "queried_bytes": 0,
                    "gpu_bytes": 0,
                    "cpu_bytes": 0,
                    "unknown_bytes": 0,
                    "error": repr(exc),
                }
            queried = int(result.get("queried_bytes") or 0)
            gpu = int(result.get("gpu_bytes") or 0)
            cpu = int(result.get("cpu_bytes") or 0)
            unknown = int(result.get("unknown_bytes") or 0)
            total_queried += queried
            total_gpu += gpu
            total_cpu += cpu
            total_unknown += unknown
            if bool(result.get("ok")) and queried > 0 and gpu == queried and unknown == 0:
                state = "gpu_full"
                gpu_full.append({"layer_id": layer_id, "expert_id": int(expert_id)})
            elif cpu > 0:
                state = "cpu_present"
                cpu_present.append({"layer_id": layer_id, "expert_id": int(expert_id)})
            elif unknown > 0:
                state = "unknown_present"
            elif not bool(result.get("ok")):
                state = "query_failed"
            else:
                state = "not_full_gpu"
            expert_records.append(
                {
                    "layer_id": layer_id,
                    "expert_id": int(expert_id),
                    "state": state,
                    "range_records": len(ranges),
                    "query_ok": bool(result.get("ok")),
                    "query_status": result.get("status"),
                    "queried_bytes": queried,
                    "gpu_bytes": gpu,
                    "cpu_bytes": cpu,
                    "unknown_bytes": unknown,
                }
            )

    return {
        "enabled": True,
        "weight_regions_file": str(weight_regions_file),
        "device": device,
        "roles": sorted(roles),
        "expert_records": expert_records,
        "expert_count": len(expert_records),
        "gpu_full_count": len(gpu_full),
        "cpu_present_count": len(cpu_present),
        "missing_count": len(missing),
        "queried_bytes": total_queried,
        "gpu_bytes": total_gpu,
        "cpu_bytes": total_cpu,
        "unknown_bytes": total_unknown,
        "gpu_full_experts": gpu_full,
        "cpu_present_experts": cpu_present,
        "missing_experts": missing,
    }


def prealloc_reserved_slots(
    slot_count: int,
    slot_bytes: int,
    *,
    allow_evict: bool = False,
) -> dict[str, Any] | None:
    if slot_count <= 0:
        return None
    from vllm.device_allocator.uvm import driver_expert_reserved_pool_prealloc

    slots = [
        {"pool_id": 0, "slot_id": slot_id, "slot_bytes": slot_bytes}
        for slot_id in range(slot_count)
    ]
    return driver_expert_reserved_pool_prealloc(
        slots,
        device=0,
        allow_evict=allow_evict,
        prefer_existing=True,
    )


def _role_id(role: str) -> int:
    from vllm.device_allocator.uvm import (
        UVM_REGION_ROLE_DENSE_WEIGHT,
        UVM_REGION_ROLE_MOE_EXPERT_W13,
        UVM_REGION_ROLE_MOE_EXPERT_W2,
    )

    normalized = str(role or "").strip().lower()
    if normalized in ("moe_gate_up", "gate_up", "w13", "w13_weight"):
        return UVM_REGION_ROLE_MOE_EXPERT_W13
    if normalized in ("moe_down", "down", "w2", "w2_weight"):
        return UVM_REGION_ROLE_MOE_EXPERT_W2
    return UVM_REGION_ROLE_DENSE_WEIGHT


def _driver_region_id(layer_id: int, expert_id: int, role: str) -> int:
    role_slot = 13 if _role_id(role) == 3 else 2
    return 90_000_000_000 + int(layer_id) * 1_000_000 + int(expert_id) * 100 + role_slot


def post_offload_slot_use_probe(
    *,
    weight_regions_file: Path,
    layer_id: int,
    expert_ids: list[int],
    roles: set[str],
    slot_count: int,
    slot_offset: int,
    slot_bytes: int,
    device: int,
    strict_footprint: bool,
    invalidate_after: bool,
) -> dict[str, Any]:
    if slot_count <= 0 or not expert_ids:
        return {
            "enabled": False,
            "reason": "slot_count_or_experts_empty",
        }

    from vllm.device_allocator.uvm import (
        UVM_REGION_FLAG_ACTIVE,
        UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED,
        UVM_REGION_FLAG_PREFETCHABLE,
        UVM_REGION_FLAG_PROTECTED,
        UVM_REGION_POOL_WEIGHT,
        batch_offload_ranges_to_cpu_with_executor,
        batch_prefetch_ranges_to_cpu,
        driver_batch_query_ranges_residency_bytes,
        driver_expert_reserved_prefetch_regions,
        driver_expert_reserved_pool_query,
        driver_register_region,
        driver_unregister_region,
    )

    records = load_weight_regions(weight_regions_file)
    selected: list[dict[str, Any]] = []
    expert_id_set = set(map(int, expert_ids))
    for record in records:
        if not record.get("is_moe_expert"):
            continue
        record_layer_id = record.get("layer_id")
        record_expert_id = record.get("expert_id")
        if record_layer_id is None or record_expert_id is None:
            continue
        if int(record_layer_id) != int(layer_id):
            continue
        if int(record_expert_id) not in expert_id_set:
            continue
        if str(record.get("role")) not in roles:
            continue
        start = record.get("start_int", record.get("data_start_int"))
        end = record.get("end_int", record.get("data_end_int"))
        if start is None or end is None:
            continue
        nbytes = int(end) - int(start) + 1
        if nbytes <= 0:
            continue
        selected.append({
            "layer_id": int(record["layer_id"]),
            "expert_id": int(record["expert_id"]),
            "role": str(record.get("role")),
            "ptr": int(start),
            "bytes": int(nbytes),
            "name": record.get("name"),
        })

    selected.sort(key=lambda item: (item["expert_id"], item["role"]))
    if not selected:
        return {
            "enabled": True,
            "ok": False,
            "reason": "no_matching_weight_regions",
            "layer_id": int(layer_id),
            "expert_ids": expert_ids,
            "roles": sorted(roles),
        }

    selected = selected[:slot_count]
    ranges = [(int(item["ptr"]), int(item["bytes"])) for item in selected]
    before = driver_batch_query_ranges_residency_bytes(ranges, device)

    register_results: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    registered_region_ids: list[int] = []
    flags = (
        UVM_REGION_FLAG_CPU_OFFLOADED_RESERVED
        | UVM_REGION_FLAG_PREFETCHABLE
        | UVM_REGION_FLAG_PROTECTED
        | UVM_REGION_FLAG_ACTIVE
    )
    for index, item in enumerate(selected):
        region_id = _driver_region_id(item["layer_id"], item["expert_id"], item["role"])
        status = driver_register_region(
            region_id=region_id,
            ptr=int(item["ptr"]),
            nbytes=int(item["bytes"]),
            pool=UVM_REGION_POOL_WEIGHT,
            layer_id=int(item["layer_id"]),
            expert_id=int(item["expert_id"]),
            role=_role_id(item["role"]),
            flags=flags,
        )
        register_results.append({
            "region_id": region_id,
            "status": int(status),
            **item,
        })
        if int(status) >= 0:
            registered_region_ids.append(region_id)
            assignments.append({
                "region_id": region_id,
                "pool_id": 0,
                "slot_id": int(slot_offset) + index,
                "slot_offset": 0,
                "bytes": int(item["bytes"]),
                "strict_footprint": bool(strict_footprint),
            })

    pool_before_prefetch = driver_expert_reserved_pool_query(
        max_entries=max(slot_count + slot_offset + 8, 64),
        prefer_existing=True,
    )
    source_offload_result: dict[str, Any]
    try:
        (
            source_successes,
            source_executor,
            source_advise_result,
            source_prefetch_result,
        ) = batch_offload_ranges_to_cpu_with_executor(
            ranges,
            sync=True,
            executor="cpp_batch",
        )
        if int(source_successes) != len(ranges):
            fallback_result = batch_prefetch_ranges_to_cpu(
                ranges,
                allow_tag_only=True,
            )
            source_successes = int(fallback_result)
            source_executor = "cpp_batch_tag_only_bypass"
            source_advise_result = 0
            source_prefetch_result = int(fallback_result)
        source_offload_result = {
            "enabled": True,
            "successes": int(source_successes),
            "executor": source_executor,
            "advise_result": int(source_advise_result),
            "prefetch_result": int(source_prefetch_result),
            "range_count": len(ranges),
        }
    except Exception as exc:
        source_offload_result = {
            "enabled": True,
            "successes": 0,
            "executor": "cpp_batch",
            "error": repr(exc),
            "range_count": len(ranges),
        }
    prefetch_result = driver_expert_reserved_prefetch_regions(
        assignments=assignments,
        device=device,
        async_=False,
        strict_footprint=strict_footprint,
        mode="reserved_prefetch",
        prefer_existing=True,
        return_entries=True,
    )
    after_prefetch = driver_batch_query_ranges_residency_bytes(ranges, device)
    pool_after_prefetch = driver_expert_reserved_pool_query(
        max_entries=max(slot_count + slot_offset + 8, 64),
        prefer_existing=True,
    )

    invalidate_result = None
    if invalidate_after and assignments:
        invalidate_result = driver_expert_reserved_prefetch_regions(
            assignments=assignments,
            device=device,
            async_=False,
            strict_footprint=strict_footprint,
            mode="invalidate",
            prefer_existing=True,
            return_entries=True,
        )

    unregister_results: list[dict[str, int]] = []
    if invalidate_after:
        for region_id in registered_region_ids:
            unregister_results.append({
                "region_id": int(region_id),
                "status": int(driver_unregister_region(int(region_id))),
            })

    ok = (
        bool(prefetch_result.get("ok"))
        and int(prefetch_result.get("success_count", 0) or 0) == len(assignments)
        and len(assignments) > 0
    )
    return {
        "enabled": True,
        "ok": ok,
        "layer_id": int(layer_id),
        "expert_ids": sorted(map(int, expert_ids)),
        "roles": sorted(roles),
        "slot_count_requested": int(slot_count),
        "slot_offset": int(slot_offset),
        "slot_bytes": int(slot_bytes),
        "selected_region_count": len(selected),
        "assignments": assignments,
        "register_results": register_results,
        "residency_before_prefetch": before,
        "pool_before_prefetch": pool_before_prefetch,
        "source_offload_result": source_offload_result,
        "prefetch_result": prefetch_result,
        "residency_after_prefetch": after_prefetch,
        "pool_after_prefetch": pool_after_prefetch,
        "invalidate_after": bool(invalidate_after),
        "invalidate_result": invalidate_result,
        "unregister_results": unregister_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct a vLLM LLM and exit after load/warmup."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep the loaded engine alive for this many seconds after writing the result.",
    )
    parser.add_argument("--post-warmup-prealloc-slots", type=int, default=0)
    parser.add_argument("--post-warmup-prealloc-slot-bytes", type=int, default=4194304)
    parser.add_argument("--post-warmup-prealloc-allow-evict", action="store_true")
    parser.add_argument("--post-offload-slot-use-probe", action="store_true")
    parser.add_argument("--slot-use-layer-id", type=int, default=22)
    parser.add_argument("--slot-use-experts", default="118,119")
    parser.add_argument("--slot-use-slot-offset", type=int, default=0)
    parser.add_argument("--slot-use-strict-footprint", action="store_true")
    parser.add_argument("--slot-use-invalidate-after", action="store_true")
    parser.add_argument("--active-expert-probe-json", type=Path, default=None)
    parser.add_argument(
        "--active-expert-roles",
        default="moe_gate_up,moe_down",
        help="Comma-separated roles to query for active expert residency.",
    )
    parser.add_argument("--weight-regions-file", type=Path, default=None)
    parser.add_argument("--residency-query-device", type=int, default=0)
    parser.add_argument("--settle-seconds", type=float, default=0.0)
    args = parser.parse_args()

    if args.settle_seconds < 0:
        raise ValueError("settle seconds must be non-negative")

    from vllm import LLM

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
    gpu_memory_before = gpu_memory_snapshot()
    llm = LLM(**kwargs)
    loaded = time.time()
    gpu_memory_after_warmup = gpu_memory_snapshot()
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    gpu_memory_after_settle = gpu_memory_snapshot()
    cuda_memory_after_settle = engine_cuda_memory_snapshot(llm)
    active_probe = load_active_expert_probe(args.active_expert_probe_json)
    weight_regions_file = (
        args.weight_regions_file
        if args.weight_regions_file is not None
        else Path.cwd() / "vllm_uvm_weight_regions.jsonl"
    )
    roles = {role.strip() for role in args.active_expert_roles.split(",") if role.strip()}
    active_residency_before_prealloc = query_active_expert_residency(
        weight_regions_file=weight_regions_file,
        active_probe=active_probe,
        roles=roles,
        device=args.residency_query_device,
    )
    gpu_memory_before_prealloc = gpu_memory_snapshot()
    prealloc_started = time.time()
    reserved_prealloc = prealloc_reserved_slots(
        args.post_warmup_prealloc_slots,
        args.post_warmup_prealloc_slot_bytes,
        allow_evict=args.post_warmup_prealloc_allow_evict,
    )
    prealloc_finished = time.time()
    gpu_memory_after_prealloc = gpu_memory_snapshot()
    slot_use_probe = post_offload_slot_use_probe(
        weight_regions_file=weight_regions_file,
        layer_id=args.slot_use_layer_id,
        expert_ids=parse_expert_ids(args.slot_use_experts),
        roles=roles,
        slot_count=args.post_warmup_prealloc_slots,
        slot_offset=args.slot_use_slot_offset,
        slot_bytes=args.post_warmup_prealloc_slot_bytes,
        device=args.residency_query_device,
        strict_footprint=args.slot_use_strict_footprint,
        invalidate_after=args.slot_use_invalidate_after,
    ) if args.post_offload_slot_use_probe else {
        "enabled": False,
        "reason": "disabled",
    }
    gpu_memory_after_slot_use_probe = gpu_memory_snapshot()
    active_residency_after_prealloc = query_active_expert_residency(
        weight_regions_file=weight_regions_file,
        active_probe=active_probe,
        roles=roles,
        device=args.residency_query_device,
    )

    # Keep the engine alive until after the measurement. No user forward or
    # generate request is issued by this script.
    result = {
        "mode": "warmup_only_no_generate",
        "model": args.model,
        "load_and_warmup_seconds": loaded - started,
        "gpu_memory_before": gpu_memory_before,
        "gpu_memory_after_warmup": gpu_memory_after_warmup,
        "gpu_memory_after_settle": gpu_memory_after_settle,
        "cuda_memory_after_settle": cuda_memory_after_settle,
        "gpu_memory_before_prealloc": gpu_memory_before_prealloc,
        "gpu_memory_after_prealloc": gpu_memory_after_prealloc,
        "gpu_memory_after_slot_use_probe": gpu_memory_after_slot_use_probe,
        "post_warmup_prealloc_slots": args.post_warmup_prealloc_slots,
        "post_warmup_prealloc_slot_bytes": args.post_warmup_prealloc_slot_bytes,
        "post_warmup_prealloc_allow_evict": args.post_warmup_prealloc_allow_evict,
        "post_warmup_prealloc_seconds": prealloc_finished - prealloc_started,
        "reserved_prealloc": reserved_prealloc,
        "slot_use_probe": slot_use_probe,
        "active_residency_before_prealloc": active_residency_before_prealloc,
        "active_residency_after_prealloc": active_residency_after_prealloc,
        "hold_seconds": args.hold_seconds,
        "settle_seconds": args.settle_seconds,
        "uvm_proactive_migration_parameters": uvm_proactive_migration_snapshot(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.hold_seconds > 0:
        print(
            json.dumps(
                {
                    "event": "holding_loaded_engine",
                    "hold_seconds": args.hold_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(args.hold_seconds)
    del llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
