#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_ints(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gpu_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "used_mib": int(parts[1]),
            "free_mib": int(parts[2]),
            "total_mib": int(parts[3]),
        }
    except Exception:
        return None


def load_requests(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    requests = data.get("requests") if isinstance(data, dict) else data
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"fixture has no requests: {path}")
    normalized = []
    for index, item in enumerate(requests):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"invalid request {index} in {path}")
        normalized.append(
            {
                "request": int(item.get("request", index)),
                "prompt": item["prompt"],
                "max_tokens": int(item.get("max_tokens", 1)),
            }
        )
    return normalized


def apply_kv_target(
    llm: Any,
    target_blocks: int,
    *,
    reason: str,
    release_expert_wave: bool,
) -> dict[str, Any]:
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    metadata = {
        "release_expert_wave": release_expert_wave,
        "source": "vllm_generate_phased",
    }
    if engine_core is None:
        return {"ok": False, "reason": "engine_core_unavailable"}
    if hasattr(engine_core, "call_utility"):
        result = engine_core.call_utility(
            "apply_uvm_kv_dynamic_pool_target",
            target_blocks,
            reason,
            metadata,
        )
    elif hasattr(engine_core, "apply_uvm_kv_dynamic_pool_target"):
        result = engine_core.apply_uvm_kv_dynamic_pool_target(
            target_blocks,
            reason,
            metadata,
        )
    else:
        return {"ok": False, "reason": "apply_utility_unavailable"}
    return result if isinstance(result, dict) else {"ok": True, "result": result}


def call_engine_utility(llm: Any, method: str, *args: Any) -> dict[str, Any]:
    llm_engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is None:
        return {"ok": False, "reason": "engine_core_unavailable"}
    if hasattr(engine_core, "call_utility"):
        value = engine_core.call_utility(method, *args)
    elif hasattr(engine_core, method):
        value = getattr(engine_core, method)(*args)
    else:
        return {"ok": False, "reason": f"utility_unavailable:{method}"}
    return value if isinstance(value, dict) else {"ok": True, "result": value}


def request_metrics(output: Any) -> dict[str, Any]:
    completion = output.outputs[0] if output.outputs else None
    token_ids = list(completion.token_ids) if completion is not None else []
    metrics = getattr(output, "metrics", None)
    ttft = getattr(metrics, "first_token_latency", None)
    first_token_ts = getattr(metrics, "first_token_ts", None)
    last_token_ts = getattr(metrics, "last_token_ts", None)
    tpot = None
    if (
        len(token_ids) > 1
        and isinstance(first_token_ts, (int, float))
        and isinstance(last_token_ts, (int, float))
        and last_token_ts >= first_token_ts
    ):
        tpot = (last_token_ts - first_token_ts) / (len(token_ids) - 1)
    return {
        "request_id": str(output.request_id),
        "prompt_tokens": len(output.prompt_token_ids or []),
        "output_tokens": len(token_ids),
        "ttft_ms": float(ttft) * 1000.0 if isinstance(ttft, (int, float)) else None,
        "tpot_ms": tpot * 1000.0 if tpot is not None else None,
        "output_token_ids": token_ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run phased batches in one vLLM process."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--phase-names", default="low,high")
    parser.add_argument("--phase-request-counts", default="8,64")
    parser.add_argument("--phase-request-offsets", default="")
    parser.add_argument("--phase-output-tokens", default="64,256")
    parser.add_argument("--grow-after-phase", type=int, default=0)
    parser.add_argument("--grow-target-blocks", default="")
    parser.add_argument("--maintenance-waves-per-phase", type=int, default=0)
    parser.add_argument("--maintenance-waves-per-phase-list", default="")
    parser.add_argument("--pre-phase-target-blocks", type=int, default=-1)
    parser.add_argument("--release-expert-wave", action="store_true")
    parser.add_argument("--dynamic-expert-swap-after-phases", default="")
    parser.add_argument("--dynamic-expert-swap-max-per-layer", type=int, default=1)
    parser.add_argument("--dynamic-expert-swap-min-count-delta", type=int, default=1)
    parser.add_argument("--reset-dynamic-expert-heat-before-phases", action="store_true")
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    phase_names = [
        value.strip()
        for value in args.phase_names.split(",")
        if value.strip()
    ]
    phase_counts = parse_ints(args.phase_request_counts)
    phase_offsets = parse_ints(args.phase_request_offsets)
    if not phase_offsets:
        phase_offsets = [0] * len(phase_counts)
    phase_outputs = parse_ints(args.phase_output_tokens)
    dynamic_swap_after_phases = set(
        parse_ints(args.dynamic_expert_swap_after_phases)
    )
    grow_targets = parse_ints(args.grow_target_blocks)
    maintenance_wave_list = parse_ints(
        args.maintenance_waves_per_phase_list
    )
    if not (
        len(phase_names)
        == len(phase_counts)
        == len(phase_offsets)
        == len(phase_outputs)
    ):
        raise ValueError(
            "phase names, request counts, offsets, and output tokens must align"
        )
    if args.maintenance_waves_per_phase < 0:
        raise ValueError("maintenance waves per phase must be non-negative")
    if maintenance_wave_list:
        if len(maintenance_wave_list) != len(phase_names):
            raise ValueError(
                "maintenance wave list must align with phase names"
            )
        if any(value < 0 for value in maintenance_wave_list):
            raise ValueError(
                "maintenance wave list values must be non-negative"
            )
    requests = load_requests(args.fixture)
    if any(
        offset < 0 or offset + count > len(requests)
        for offset, count in zip(phase_offsets, phase_counts)
    ):
        raise ValueError("phase request range exceeds fixture size")
    if any(
        index < 0 or index >= len(phase_names)
        for index in dynamic_swap_after_phases
    ):
        raise ValueError("dynamic expert swap phase index is out of range")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("", encoding="utf-8")

    def progress(record: dict[str, Any]) -> None:
        with args.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    from vllm import LLM, SamplingParams

    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=not args.disable_prefix_caching,
        seed=args.seed,
        disable_log_stats=False,
    )
    result: dict[str, Any] = {
        "status": "running",
        "config": vars(args)
        | {
            "fixture": str(args.fixture),
            "output": str(args.output),
            "progress": str(args.progress),
        },
        "load_seconds": time.perf_counter() - load_started,
        "gpu_after_load": gpu_snapshot(),
        "phases": [],
        "maintenance": [],
        "expert_swaps": [],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    try:
        import torch

        next_wave = 1
        grow_cursor = 0

        if (
            dynamic_swap_after_phases
            or args.reset_dynamic_expert_heat_before_phases
        ):
            reset_result = call_engine_utility(
                llm,
                "reset_uvm_dynamic_expert_heat",
            )
            result["dynamic_expert_heat_initial_reset"] = reset_result
            if not bool(reset_result.get("ok")):
                raise RuntimeError(
                    f"initial dynamic expert heat reset failed: {reset_result!r}"
                )

        def run_maintenance(
            target_blocks: int,
            *,
            after_phase: int | None,
            release_expert_wave: bool,
            reason: str,
        ) -> None:
            nonlocal next_wave
            torch.cuda.synchronize()
            maintenance_started = time.perf_counter()
            grow_result = apply_kv_target(
                llm,
                target_blocks,
                reason=reason,
                release_expert_wave=release_expert_wave,
            )
            torch.cuda.synchronize()
            maintenance = {
                "wave": next_wave,
                "after_phase": after_phase,
                "target_blocks": target_blocks,
                "release_expert_wave": release_expert_wave,
                "elapsed_seconds": time.perf_counter() - maintenance_started,
                "result": grow_result,
                "gpu_after_wave": gpu_snapshot(),
            }
            next_wave += 1
            result["maintenance"].append(maintenance)
            progress(
                {
                    "action": "maintenance_wave",
                    "timestamp": time.time(),
                    **maintenance,
                }
            )
            if not bool(grow_result.get("ok")):
                raise RuntimeError(
                    f"KV grow target {target_blocks} failed: {grow_result!r}"
                )

        if args.pre_phase_target_blocks >= 0:
            run_maintenance(
                args.pre_phase_target_blocks,
                after_phase=None,
                release_expert_wave=False,
                reason="phased_pre_phase_target",
            )

        for phase_index, (name, count, offset, output_tokens) in enumerate(
            zip(phase_names, phase_counts, phase_offsets, phase_outputs)
        ):
            prompts = [
                item["prompt"] for item in requests[offset : offset + count]
            ]
            sampling = SamplingParams(
                max_tokens=output_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
            progress(
                {
                    "action": "phase_begin",
                    "phase": name,
                    "request_count": count,
                    "request_offset": offset,
                    "timestamp": time.time(),
                }
            )
            print(
                f"UVM_PHASE_BEGIN name={name} request_count={count} "
                f"output_tokens={output_tokens}",
                flush=True,
            )
            started = time.perf_counter()
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            records = [request_metrics(output) for output in outputs]
            output_count = sum(int(record["output_tokens"]) for record in records)
            ttfts = [
                float(record["ttft_ms"])
                for record in records
                if record["ttft_ms"] is not None
            ]
            tpots = [
                float(record["tpot_ms"])
                for record in records
                if record["tpot_ms"] is not None
            ]
            phase_result = {
                "phase": name,
                "request_count": count,
                "request_offset": offset,
                "output_tokens_per_request": output_tokens,
                "elapsed_seconds": elapsed,
                "prompt_tokens": sum(
                    int(record["prompt_tokens"]) for record in records
                ),
                "output_tokens": output_count,
                "output_tokens_per_second": output_count / elapsed if elapsed else None,
                "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else None,
                "p99_ttft_ms": percentile(ttfts, 99.0),
                "mean_tpot_ms": sum(tpots) / len(tpots) if tpots else None,
                "p99_tpot_ms": percentile(tpots, 99.0),
                "completed_requests": len(records),
                "requests": records,
                "gpu_after_phase": gpu_snapshot(),
            }
            result["phases"].append(phase_result)
            print(
                f"UVM_PHASE_END name={name} elapsed_seconds={elapsed:.6f}",
                flush=True,
            )
            progress(
                {
                    "action": "phase_end",
                    "timestamp": time.time(),
                    **{
                        key: value
                        for key, value in phase_result.items()
                        if key != "requests"
                    },
                }
            )

            if phase_index in dynamic_swap_after_phases:
                torch.cuda.synchronize()
                swap_started = time.perf_counter()
                swap_result = call_engine_utility(
                    llm,
                    "rebalance_uvm_remote_experts",
                    args.dynamic_expert_swap_max_per_layer,
                    args.dynamic_expert_swap_min_count_delta,
                    True,
                )
                torch.cuda.synchronize()
                swap_record = {
                    "after_phase": phase_index,
                    "after_phase_name": name,
                    "elapsed_seconds": time.perf_counter() - swap_started,
                    "max_swaps_per_layer": (
                        args.dynamic_expert_swap_max_per_layer
                    ),
                    "min_count_delta": (
                        args.dynamic_expert_swap_min_count_delta
                    ),
                    "result": swap_result,
                    "gpu_after_swap": gpu_snapshot(),
                }
                result["expert_swaps"].append(swap_record)
                progress(
                    {
                        "action": "dynamic_expert_swap",
                        "timestamp": time.time(),
                        **swap_record,
                    }
                )
                if not bool(swap_result.get("ok")):
                    raise RuntimeError(
                        f"dynamic expert swap failed: {swap_result!r}"
                    )

            if maintenance_wave_list:
                stop = min(
                    grow_cursor + maintenance_wave_list[phase_index],
                    len(grow_targets),
                )
                targets = grow_targets[grow_cursor:stop]
                grow_cursor = stop
            elif args.maintenance_waves_per_phase:
                stop = min(
                    grow_cursor + args.maintenance_waves_per_phase,
                    len(grow_targets),
                )
                targets = grow_targets[grow_cursor:stop]
                grow_cursor = stop
            elif phase_index == args.grow_after_phase:
                targets = grow_targets
                grow_cursor = len(grow_targets)
            else:
                targets = []

            for target_blocks in targets:
                run_maintenance(
                    target_blocks,
                    after_phase=phase_index,
                    release_expert_wave=args.release_expert_wave,
                    reason=f"phased_hot_cold_wave_{next_wave}",
                )

            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if grow_cursor != len(grow_targets):
            raise RuntimeError(
                f"unused KV grow targets: consumed={grow_cursor} "
                f"total={len(grow_targets)}"
            )
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["gpu_after_error"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise

    result["status"] = "ok"
    result["gpu_after_finish"] = gpu_snapshot()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
