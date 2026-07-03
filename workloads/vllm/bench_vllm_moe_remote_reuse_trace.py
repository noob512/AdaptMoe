#!/usr/bin/env python3
"""Trace expert-weight reuse in a real vLLM generate() run.

This probe monkey-patches vLLM's Python fused MoE launcher and records, for
each W13/W2 grouped GEMM, how many M-blocks use each expert. For a remote/UVA
expert matrix, the same matrix address range is logically reread once per
M-block in that layer. If an expert has m_blocks=1, its W13 or W2 matrix is
consumed by a single grouped-GEMM block in that layer.

The script prints a stdout summary only; no JSON/CSV result files are produced.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_DIR = SCRIPT_DIR / "vllm"
if str(VLLM_DIR) not in sys.path:
    sys.path.insert(0, str(VLLM_DIR))


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def summarize_ints(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(values)),
        "mean": float(statistics.mean(values)),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": float(max(values)),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return {"records": records, "moe_records": [], "bytes": 0}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(rec)
    moe_records = [
        rec
        for rec in records
        if rec.get("event") == "cpu_offload_parameter"
        and ".mlp.experts." in str(rec.get("full_parameter_name", ""))
    ]
    return {
        "records": records,
        "moe_records": moe_records,
        "bytes": int(
            sum(
                int(rec.get("bytes") or 0)
                for rec in records
                if rec.get("event") == "cpu_offload_parameter"
            )
        ),
        "moe_bytes": int(sum(int(rec.get("bytes") or 0) for rec in moe_records)),
    }


def load_routing_trace(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize_routing_records(
    records: list[dict[str, Any]],
    target_assignments: int,
    block_m: int,
) -> dict[str, Any]:
    decode_records: list[dict[str, Any]] = []
    for rec in records:
        counts = rec.get("expert_token_counts")
        if not isinstance(counts, dict):
            continue
        total_assignments = 0
        for value in counts.values():
            try:
                total_assignments += int(value)
            except (TypeError, ValueError):
                pass
        if total_assignments == target_assignments:
            decode_records.append(rec)

    active_experts: list[int] = []
    assignment_values: list[int] = []
    block_values: list[int] = []
    repeated_experts_total = 0
    repeated_extra_m_blocks_total = 0
    layers_with_max_m_blocks_gt1 = 0

    for rec in decode_records:
        counts = rec.get("expert_token_counts")
        if not isinstance(counts, dict):
            continue
        per_layer_blocks: list[int] = []
        active = 0
        for value in counts.values():
            try:
                tokens = int(value)
            except (TypeError, ValueError):
                continue
            if tokens <= 0:
                continue
            active += 1
            blocks = ceil_div(tokens, block_m)
            assignment_values.append(tokens)
            block_values.append(blocks)
            per_layer_blocks.append(blocks)
            if blocks > 1:
                repeated_experts_total += 1
                repeated_extra_m_blocks_total += blocks - 1
        active_experts.append(active)
        if per_layer_blocks and max(per_layer_blocks) > 1:
            layers_with_max_m_blocks_gt1 += 1

    one_block_experts = sum(1 for value in block_values if value == 1)
    return {
        "total_records": len(records),
        "decode_records": len(decode_records),
        "active_experts": summarize_ints(active_experts),
        "assignments_per_active_expert": summarize_ints(assignment_values),
        "m_blocks_per_active_expert": summarize_ints(block_values),
        "layers_with_max_m_blocks_gt1": layers_with_max_m_blocks_gt1,
        "repeated_experts_total": repeated_experts_total,
        "repeated_extra_m_blocks_total": repeated_extra_m_blocks_total,
        "one_block_experts": one_block_experts,
        "total_expert_phase_uses": len(block_values),
    }


def parse_args() -> argparse.Namespace:
    default_model = (
        "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-FP8/"
        "snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument(
        "--unique-prompts",
        action="store_true",
        help="Append a short per-request suffix so decode tokens are not identical.",
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--block-m",
        type=int,
        default=64,
        help="MoE BLOCK_SIZE_M used to convert expert token counts to M-blocks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(tempfile.gettempdir()) / (
        f"vllm_moe_remote_reuse_manifest_{os.getpid()}.jsonl"
    )
    routing_trace_path = Path(tempfile.gettempdir()) / (
        f"vllm_moe_remote_reuse_routing_{os.getpid()}.jsonl"
    )
    os.environ["VLLM_CPU_OFFLOAD_BACKEND"] = "uva"
    os.environ["VLLM_CPU_OFFLOAD_MANIFEST_FILE"] = str(manifest_path)
    os.environ["VLLM_CPU_OFFLOAD_MANIFEST_RESET"] = "1"
    os.environ["VLLM_UVM_MOE_ROUTING_TRACE_ENABLE"] = "1"
    os.environ["VLLM_UVM_MOE_ROUTING_TRACE_FILE"] = str(routing_trace_path)
    os.environ["VLLM_MOE_REMOTE_REUSE_TRACE_STDOUT"] = "1"
    os.environ["VLLM_MOE_REMOTE_REUSE_TRACE_TOPK"] = str(args.topk)
    os.environ["VLLM_MOE_REMOTE_REUSE_TRACE_TARGET_ASSIGNMENTS"] = str(
        args.num_prompts * args.topk
    )

    import torch
    from vllm import LLM, SamplingParams
    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe_mod

    records: list[dict[str, Any]] = []
    original_invoke = fused_moe_mod.invoke_fused_moe_kernel
    ptr_to_layer: dict[int, int] = {}
    current_layer = -1

    def traced_invoke(*call_args: Any, **call_kwargs: Any) -> Any:
        nonlocal current_layer

        A = call_args[0]
        B = call_args[1]
        sorted_token_ids = call_args[7]
        expert_ids = call_args[8]
        num_tokens_post_padded = int(call_args[9].item())
        top_k_arg = int(call_args[11])
        config = dict(call_args[12])
        block_m = int(config["BLOCK_SIZE_M"])
        valid_m_blocks = ceil_div(num_tokens_post_padded, block_m)
        valid_expert_ids = expert_ids[:valid_m_blocks].detach().to("cpu")
        sorted_ids = sorted_token_ids[:num_tokens_post_padded].detach().to("cpu")
        phase = "w13" if top_k_arg == args.topk else "w2"

        ptr = int(B.data_ptr())
        if phase == "w13":
            if ptr not in ptr_to_layer:
                ptr_to_layer[ptr] = len(ptr_to_layer)
            current_layer = ptr_to_layer[ptr]
        layer = current_layer

        counts_by_expert: collections.Counter[int] = collections.Counter()
        assignments_by_expert: collections.Counter[int] = collections.Counter()
        total_assignments = 0
        num_valid_assignments = int(A.size(0) * top_k_arg)

        for block_idx, expert_tensor in enumerate(valid_expert_ids):
            expert = int(expert_tensor.item())
            if expert < 0:
                continue
            block_start = block_idx * block_m
            block_end = min(block_start + block_m, int(sorted_ids.numel()))
            token_block = sorted_ids[block_start:block_end]
            valid_tokens = int((token_block < num_valid_assignments).sum().item())
            if valid_tokens <= 0:
                continue
            counts_by_expert[expert] += 1
            assignments_by_expert[expert] += valid_tokens
            total_assignments += valid_tokens

        m_blocks = list(counts_by_expert.values())
        assignments = list(assignments_by_expert.values())
        repeated_experts = sum(1 for value in m_blocks if value > 1)
        repeated_m_blocks = sum(value - 1 for value in m_blocks if value > 1)

        records.append(
            {
                "phase": phase,
                "layer": layer,
                "ptr": ptr,
                "B_shape": tuple(int(x) for x in B.shape),
                "B_mib_per_expert": float(B[0].numel() * B.element_size() / (1024**2)),
                "block_m": block_m,
                "active_experts": len(m_blocks),
                "total_assignments": total_assignments,
                "m_blocks": m_blocks,
                "assignments": assignments,
                "max_m_blocks": max(m_blocks) if m_blocks else 0,
                "repeated_experts": repeated_experts,
                "repeated_m_blocks": repeated_m_blocks,
            }
        )
        return original_invoke(*call_args, **call_kwargs)

    fused_moe_mod.invoke_fused_moe_kernel = traced_invoke
    try:
        print(
            "loading vLLM "
            f"model={args.model} prompts={args.num_prompts} "
            f"cpu_offload_gb={args.cpu_offload_gb}",
            flush=True,
        )
        llm = LLM(
            model=args.model,
            enforce_eager=True,
            cpu_offload_gb=args.cpu_offload_gb,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            distributed_executor_backend="uni",
            seed=args.seed,
        )
        if args.unique_prompts:
            topics = [
                "volcano",
                "compiler",
                "orchid",
                "neutrino",
                "harbor",
                "circuit",
                "telescope",
                "monsoon",
                "enzyme",
                "satellite",
                "cipher",
                "glacier",
                "market",
                "sonata",
                "quasar",
                "polymer",
                "archive",
                "lantern",
                "railway",
                "isotope",
                "vineyard",
                "protocol",
                "meteor",
                "habitat",
                "algorithm",
                "ceramic",
                "bayonet",
                "galaxy",
                "theorem",
                "citadel",
                "mosaic",
                "plasma",
            ]
            verbs = [
                "Define",
                "Compare",
                "Summarize",
                "Translate",
                "Classify",
                "Estimate",
                "Design",
                "Explain",
            ]
            prompts = [
                f"{verbs[idx % len(verbs)]} a concise fact about {topics[idx % len(topics)]}"
                for idx in range(args.num_prompts)
            ]
        else:
            prompts = [args.prompt for _ in range(args.num_prompts)]
        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.0,
            ignore_eos=True,
        )
        print("running generate", flush=True)
        _ = llm.generate(prompts, sampling, use_tqdm=False)
        torch.cuda.synchronize()
    finally:
        fused_moe_mod.invoke_fused_moe_kernel = original_invoke

    manifest = load_manifest(manifest_path)
    routing_records = load_routing_trace(routing_trace_path)
    target_assignments = args.num_prompts * args.topk
    routing_summary = summarize_routing_records(
        routing_records,
        target_assignments,
        args.block_m,
    )
    decode_records = [
        rec for rec in records if int(rec["total_assignments"]) == target_assignments
    ]
    by_phase = {
        phase: [rec for rec in decode_records if rec["phase"] == phase]
        for phase in ("w13", "w2")
    }

    print("summary")
    print(f"  total_moe_kernel_records={len(records)}")
    print(f"  decode_records_assignments_{target_assignments}={len(decode_records)}")
    print(
        "  offload_manifest "
        f"params={len([r for r in manifest['records'] if r.get('event') == 'cpu_offload_parameter'])} "
        f"bytes_gib={manifest['bytes'] / (1024**3):.3f} "
        f"moe_params={len(manifest['moe_records'])} "
        f"moe_bytes_gib={manifest.get('moe_bytes', 0) / (1024**3):.3f}"
    )
    print(
        "  routing_trace "
        f"records={routing_summary['total_records']} "
        f"decode_records_assignments_{target_assignments}={routing_summary['decode_records']} "
        f"assumed_block_m={args.block_m}"
    )

    for phase, phase_records in by_phase.items():
        active_experts = [int(rec["active_experts"]) for rec in phase_records]
        max_m_blocks = [int(rec["max_m_blocks"]) for rec in phase_records]
        repeated_experts = [int(rec["repeated_experts"]) for rec in phase_records]
        repeated_m_blocks = [int(rec["repeated_m_blocks"]) for rec in phase_records]
        assignment_values: list[int] = []
        block_values: list[int] = []
        for rec in phase_records:
            assignment_values.extend(int(x) for x in rec["assignments"])
            block_values.extend(int(x) for x in rec["m_blocks"])
        one_block_experts = sum(1 for value in block_values if value == 1)
        total_expert_phase_uses = len(block_values)
        print(f"  phase={phase}")
        print(
            "    layers="
            f"{len(phase_records)} active_experts={summarize_ints(active_experts)}"
        )
        print(
            "    assignments_per_active_expert="
            f"{summarize_ints(assignment_values)}"
        )
        print(f"    m_blocks_per_active_expert={summarize_ints(block_values)}")
        print(
            "    repeated_address_ranges "
            f"layers_with_max_m_blocks_gt1={sum(1 for x in max_m_blocks if x > 1)} "
            f"repeated_experts_total={sum(repeated_experts)} "
            f"repeated_extra_m_blocks_total={sum(repeated_m_blocks)}"
        )
        print(
            "    one_mblock_fraction="
            f"{one_block_experts}/{total_expert_phase_uses} "
            f"({(one_block_experts / total_expert_phase_uses * 100.0) if total_expert_phase_uses else 0.0:.2f}%)"
        )

    print("routing_trace_fallback")
    print(
        "  This is recorded immediately after select_experts() in the real vLLM "
        "path, so it is independent of the concrete MoE GEMM backend."
    )
    for phase in ("w13", "w2"):
        total_uses = int(routing_summary["total_expert_phase_uses"])
        one_block = int(routing_summary["one_block_experts"])
        one_pct = (one_block / total_uses * 100.0) if total_uses else 0.0
        print(f"  phase={phase}")
        print(f"    layers={routing_summary['decode_records']}")
        print(f"    active_experts={routing_summary['active_experts']}")
        print(
            "    assignments_per_active_expert="
            f"{routing_summary['assignments_per_active_expert']}"
        )
        print(
            "    m_blocks_per_active_expert="
            f"{routing_summary['m_blocks_per_active_expert']}"
        )
        print(
            "    repeated_address_ranges "
            f"layers_with_max_m_blocks_gt1={routing_summary['layers_with_max_m_blocks_gt1']} "
            f"repeated_experts_total={routing_summary['repeated_experts_total']} "
            f"repeated_extra_m_blocks_total={routing_summary['repeated_extra_m_blocks_total']}"
        )
        print(
            "    one_mblock_fraction="
            f"{one_block}/{total_uses} ({one_pct:.2f}%)"
        )

    print("interpretation")
    print(
        "  In this probe, one active expert with m_blocks=1 means its W13 or W2 "
        "matrix address range is consumed by one grouped-GEMM M-block in that "
        "layer. m_blocks>1 means the same expert matrix range is logically "
        "reread by multiple M-blocks. Hardware cache-line transactions are not "
        "counted per address by this probe."
    )


if __name__ == "__main__":
    main()
