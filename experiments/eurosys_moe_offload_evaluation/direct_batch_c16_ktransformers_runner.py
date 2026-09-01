#!/usr/bin/env python3
"""Run one E2 C16 workload through the SGLang Engine API.

Passing one ``list[list[int]]`` to ``Engine.generate`` makes the tokenizer
manager send one ``BatchTokenizedGenerateReqInput``.  This bypasses HTTP and
prevents sixteen independent frontend arrivals from becoming a 1+15 cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from direct_batch_phase_runner import atomic_json, distribution, load_fixture, utc_now


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sampling_params(output_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": output_tokens,
        "min_new_tokens": output_tokens,
        "ignore_eos": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--frequency-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--gpu-experts", type=int, default=63)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--context-length", type=int, default=194)
    parser.add_argument("--max-total-tokens", type=int, default=3104)
    parser.add_argument("--max-prefill-tokens", type=int, default=2048)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    for path in (args.model / "config.json", args.fixture, args.warmup_fixture,
                 args.frequency_plan):
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")
    eval_metadata, evaluation = load_fixture(args.fixture)
    warm_metadata, warmups = load_fixture(args.warmup_fixture)
    if eval_metadata.get("workload") != args.workload or \
            warm_metadata.get("workload") != args.workload:
        raise SystemExit("fixture workload differs from --workload")
    if len(evaluation) < 16 or len(warmups) < args.warmup_requests:
        raise SystemExit("fixture does not contain the required requests")
    if any(
        len(request.prompt_token_ids) != args.prompt_tokens
        for request in evaluation[:16]
    ):
        raise SystemExit("the first 16 evaluation prompts have unexpected lengths")

    import sglang as sgl

    result: dict[str, Any] = {
        "schema": "eurosys_e2_c16_direct_batch_ktransformers_v2",
        "status": "running",
        "started_at": utc_now(),
        "configuration": {
            "arm": "ktransformers",
            "workload": args.workload,
            "model": str(args.model.resolve()),
            "fixture": str(args.fixture.resolve()),
            "warmup_fixture": str(args.warmup_fixture.resolve()),
            "frequency_plan": str(args.frequency_plan.resolve()),
            "concurrency": 16,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "warmup_output_tokens": args.warmup_output_tokens,
            "context_length": args.context_length,
            "max_total_tokens": args.max_total_tokens,
            "max_prefill_tokens": args.max_prefill_tokens,
            "max_running_requests": 16,
            "gpu_experts_per_layer": args.gpu_experts,
            "cuda_graph_batch_sizes": [1, 4, 8, 16],
            "transport": "none",
            "http_server_started": False,
            "client_used": False,
            "measurement_admission": "one list[list[int]] Engine.generate call",
            "scheduler_message": "BatchTokenizedGenerateReqInput(batch=16)",
            "measurement_request_batch_size": 16,
            "prefill_decode_policy": "prefill-priority; mixed chunk disabled",
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
        },
        "warmups": [],
        "measurement": None,
    }
    atomic_json(args.output, result)
    engine: Any | None = None
    try:
        engine = sgl.Engine(
            model_path=str(args.model),
            tp_size=1,
            context_length=args.context_length,
            max_running_requests=16,
            mem_fraction_static=0.97,
            max_total_tokens=args.max_total_tokens,
            chunked_prefill_size=args.max_prefill_tokens,
            max_prefill_tokens=args.max_prefill_tokens,
            schedule_policy="fcfs",
            stream_interval=1,
            random_seed=args.seed,
            trust_remote_code=True,
            kt_weight_path=str(args.model),
            kt_method="BF16",
            kt_cpuinfer=14,
            kt_threadpool_count=1,
            # The CLI normalizes ``--kt-numa-nodes 0`` to a list.  This runner
            # constructs ServerArgs directly, so preserve that normalized
            # type; kt_kernel compares len(numa_nodes) with the thread-pool
            # count.
            kt_numa_nodes=[0],
            kt_num_gpu_experts=args.gpu_experts,
            kt_expert_placement_strategy="frequency",
            init_expert_location=str(args.frequency_plan),
            kt_max_deferred_experts_per_token=0,
            kt_gpu_prefill_token_threshold=0,
            attention_backend="flashinfer",
            cuda_graph_max_bs=16,
            cuda_graph_bs=[1, 4, 8, 16],
            enable_mixed_chunk=False,
            disable_shared_experts_fusion=True,
            disable_radix_cache=True,
            enable_metrics=True,
            decode_log_interval=1,
            log_level="info",
        )
        for index in range(args.warmup_requests):
            print(f"DIRECT_C16_WARMUP_START index={index}", flush=True)
            warm_started = time.perf_counter()
            output = engine.generate(
                input_ids=warmups[index].prompt_token_ids,
                sampling_params=sampling_params(args.warmup_output_tokens),
                stream=False,
                rid=f"c16-warmup-{index}",
            )
            elapsed = time.perf_counter() - warm_started
            output_ids = [int(value) for value in output.get("output_ids", [])]
            warm = {
                "request_count": 1,
                "successful_requests": int(len(output_ids) == args.warmup_output_tokens),
                "completion_tokens": len(output_ids),
                "elapsed_seconds": elapsed,
            }
            result["warmups"].append(warm)
            atomic_json(args.output, result)
            if warm["successful_requests"] != 1:
                raise RuntimeError(f"warmup {index} generated {len(output_ids)} tokens")

        request_ids = [f"c16-measurement-{index:02d}" for index in range(16)]
        prompts = [request.prompt_token_ids for request in evaluation[:16]]
        print(
            "DIRECT_C16_MEASUREMENT_DISPATCH "
            "api_batch_size=16 scheduler_message=BatchTokenizedGenerateReqInput",
            flush=True,
        )
        started = time.perf_counter()
        stream = engine.generate(
            input_ids=prompts,
            sampling_params=[sampling_params(args.output_tokens) for _ in range(16)],
            stream=True,
            rid=request_ids,
        )
        token_times: list[list[float]] = [[] for _ in range(16)]
        output_ids_by_request: list[list[int]] = [[] for _ in range(16)]
        completion_counts = [0] * 16
        grouped_events = 0
        final_meta: list[dict[str, Any]] = [{} for _ in range(16)]
        for chunk in stream:
            observed = time.perf_counter()
            index = int(chunk["index"])
            meta = chunk.get("meta_info") or {}
            completion = int(meta.get("completion_tokens", completion_counts[index]))
            new_tokens = max(0, completion - completion_counts[index])
            ids = [int(value) for value in chunk.get("output_ids", [])]
            if len(ids) == completion:
                output_ids_by_request[index] = ids
            elif new_tokens and len(ids) >= new_tokens:
                output_ids_by_request[index].extend(ids[-new_tokens:])
            if new_tokens > 1:
                grouped_events += 1
            token_times[index].extend([observed] * new_tokens)
            completion_counts[index] = completion
            final_meta[index] = dict(meta)

        finished = time.perf_counter()
        records: list[dict[str, Any]] = []
        for index, fixture_request in enumerate(evaluation[:16]):
            times = token_times[index]
            ids = output_ids_by_request[index]
            itl = [(right - left) * 1000.0 for left, right in zip(times, times[1:])]
            success = len(ids) == args.output_tokens and len(times) == args.output_tokens
            records.append({
                "request_id": request_ids[index],
                "fixture_index": fixture_request.fixture_index,
                "prompt_token_ids_sha256": fixture_request.prompt_token_ids_sha256,
                "completion_tokens": len(ids),
                "output_token_ids": ids,
                "output_token_ids_sha256": token_hash(ids),
                "success": success,
                "ttft_ms": (times[0] - started) * 1000.0 if times else None,
                "tpot_ms": statistics.fmean(itl) if itl else None,
                "itl_ms": itl,
                "e2e_ms": (times[-1] - started) * 1000.0 if times else None,
                "server_meta": final_meta[index],
            })
        successful = [record for record in records if record["success"]]
        measurement = {
            "request_count": 16,
            "successful_requests": len(successful),
            "failed_requests": 16 - len(successful),
            "api_batch_size": 16,
            "admitted_request_count": 16,
            "single_admission_batch": True,
            "all_requests_in_one_generate_call": True,
            "scheduler_batch_proof_source": "run.log Prefill batch #new-seq",
            "grouped_stream_events": grouped_events,
            "elapsed_seconds": finished - started,
            "ttft_ms": distribution([float(r["ttft_ms"]) for r in successful]),
            "tpot_ms": distribution([float(r["tpot_ms"]) for r in successful]),
            "e2e_ms": distribution([float(r["e2e_ms"]) for r in successful]),
            "slo_satisfied_requests": sum(
                r["ttft_ms"] <= args.ttft_slo_ms
                and r["tpot_ms"] <= args.tpot_slo_ms
                for r in successful
            ),
            "slo": {
                "ttft_ms": args.ttft_slo_ms,
                "tpot_ms": args.tpot_slo_ms,
            },
            "requests": records,
        }
        result["measurement"] = measurement
        result["status"] = "ok" if len(successful) == 16 else "error"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = utc_now()
        atomic_json(args.output, result)
        if engine is not None:
            engine.shutdown()
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
