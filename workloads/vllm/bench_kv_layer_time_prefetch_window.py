#!/usr/bin/env python3
"""Measure KV footprint, per-layer time, and expert prefetch feasibility.

This experiment answers a specific scheduling question:

  If KV cache growth pushes an expert out of GPU memory, can the expert be
  prefetched back during the previous layer's compute window?

It measures two independent quantities on the same machine:

  1. UVM prefetch time for one or more Qwen3-MoE experts.
  2. Real vLLM Qwen3-MoE decoder-layer CUDA event time as KV length and
     decode num_tokens change.

The final report compares these tables directly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_hot_cold_expert_checkpoint import (  # noqa: E402
    DEFAULT_CHECKPOINT_DIR,
    CheckpointReader,
    _dtype_bytes,
)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ms": _percentile(values, 0.50),
        "p90_ms": _percentile(values, 0.90),
        "p95_ms": _percentile(values, 0.95),
        "max_ms": float(max(values)),
    }


def _mib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


def _parse_cases(raw: str) -> list[tuple[str, int, int]]:
    cases: list[tuple[str, int, int]] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"case {token!r} must be PROMPT_WORDS:NUM_PROMPTS")
        prompt_s, batch_s = token.split(":", 1)
        prompt_words = int(prompt_s)
        num_prompts = int(batch_s)
        if prompt_words <= 0 or num_prompts <= 0:
            raise ValueError(f"case {token!r} values must be positive")
        cases.append((f"p{prompt_words}_n{num_prompts}", prompt_words, num_prompts))
    if not cases:
        raise ValueError("at least one case is required")
    return cases


def _parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list {raw!r}")
    return values


def _make_prompt(prompt_words: int, idx: int) -> str:
    words = ["system", "memory", "attention", "window"] * max(
        1, math.ceil(prompt_words / 4)
    )
    words = words[:prompt_words]
    words[-1] = f"case{idx % 997}"
    return " ".join(words)


def _load_jsonl_since(path: Path, start_line: int) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records, start_line

    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle):
            total = line_no + 1
            if line_no < start_line:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records, total


def _hf_text_config(model: Path) -> Any:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(model), trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


def _kv_model_info(args: argparse.Namespace) -> dict[str, Any]:
    cfg = _hf_text_config(args.model)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    num_attention_heads = int(getattr(cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(cfg, "head_dim", getattr(cfg, "hidden_size") //
                           num_attention_heads))
    num_experts = int(getattr(cfg, "num_experts", 0))
    sparse_step = int(getattr(cfg, "decoder_sparse_step", 1))
    mlp_only_layers = set(int(x) for x in getattr(cfg, "mlp_only_layers", []) or [])
    sparse_layers = [
        layer for layer in range(num_layers)
        if num_experts > 0 and layer not in mlp_only_layers
        and (layer + 1) % sparse_step == 0
    ]
    kv_per_token_per_layer_bytes = (
        2 * num_kv_heads * head_dim * int(args.kv_dtype_bytes)
    )
    return {
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype_bytes": int(args.kv_dtype_bytes),
        "kv_per_token_per_layer_bytes": kv_per_token_per_layer_bytes,
        "kv_per_token_all_layers_bytes": (
            kv_per_token_per_layer_bytes * num_layers
        ),
        "num_experts": num_experts,
        "decoder_sparse_step": sparse_step,
        "sparse_layer_count": len(sparse_layers),
        "sparse_layers": sparse_layers,
    }


def _expert_info(model: Path, layer: int) -> dict[str, Any]:
    reader = CheckpointReader(model)
    try:
        shapes = reader.expert_shapes(layer, 0)
        dtype = reader.expert_tensor(layer, 0, "gate_proj").dtype
        w13_shape = tuple(int(x) for x in shapes["w13_shape"])
        w2_shape = tuple(int(x) for x in shapes["w2_shape"])
        per_expert_bytes = (
            math.prod(w13_shape) + math.prod(w2_shape)
        ) * _dtype_bytes(dtype)
        return {
            "layer": layer,
            "dtype": str(dtype),
            "w13_shape": list(w13_shape),
            "w2_shape": list(w2_shape),
            "per_expert_bytes": int(per_expert_bytes),
            "per_expert_mib": _mib(per_expert_bytes),
        }
    finally:
        reader.close()


def _run_prefetch_sweep(
    args: argparse.Namespace,
    *,
    expert: dict[str, Any],
) -> dict[str, Any]:
    if args.skip_prefetch:
        return {"skipped": True, "rows": []}

    import torch
    from bench_uvm_prefetch_vs_h2d_copy import run as run_prefetch

    torch.cuda.set_device(args.device)
    multipliers = _parse_int_list(args.expert_multipliers)
    sizes = ",".join(
        str(int(expert["per_expert_bytes"]) * multiplier)
        for multiplier in multipliers
    )
    prefetch_args = argparse.Namespace(
        sizes=sizes,
        repeats=args.prefetch_repeats,
        device=args.device,
        out_json=None,
    )
    result = run_prefetch(prefetch_args)
    for row, multiplier in zip(result.get("rows", []), multipliers):
        row["expert_count"] = multiplier
        row["expert_bytes"] = int(expert["per_expert_bytes"]) * multiplier
        row["expert_mib"] = _mib(row["expert_bytes"])
    try:
        from vllm.device_allocator.uvm import cleanup_uvm_local_memory_pool

        cleanup_uvm_local_memory_pool()
    except Exception:
        pass
    torch.cuda.empty_cache()
    return result


def _classify_phase(record: dict[str, Any], num_prompts: int) -> str:
    try:
        num_tokens = int(record["num_tokens"])
        kv_len = int(record.get("kv_len_approx") or 0)
    except (KeyError, TypeError, ValueError):
        return "unknown"
    if num_tokens == num_prompts and kv_len > num_prompts:
        return "decode_full_batch"
    if num_tokens < num_prompts and kv_len > num_tokens:
        return "decode_partial_or_mixed"
    if num_tokens >= num_prompts:
        return "prefill_or_chunk"
    return "other"


def _annotate_layer_records(
    records: list[dict[str, Any]],
    *,
    case_name: str,
    prompt_words: int,
    num_prompts: int,
    kv_info: dict[str, Any],
    block_size: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    per_token_layer = int(kv_info["kv_per_token_per_layer_bytes"])
    num_layers = int(kv_info["num_layers"])
    for rec in records:
        if rec.get("event") != "qwen3_moe_layer_timing":
            continue
        try:
            num_tokens = int(rec["num_tokens"])
            kv_len = int(rec["kv_len_approx"])
        except (KeyError, TypeError, ValueError):
            continue
        phase = _classify_phase(rec, num_prompts)
        seq_rounded_tokens = int(math.ceil(kv_len / block_size) * block_size)
        block_rounded_tokens = seq_rounded_tokens * num_prompts
        if phase == "decode":
            active_kv_tokens = kv_len * num_prompts
        elif phase == "prefill_or_chunk":
            active_kv_tokens = max(num_tokens, kv_len * num_prompts)
        else:
            active_kv_tokens = max(num_tokens, kv_len)

        item = dict(rec)
        item.update({
            "case": case_name,
            "prompt_words": prompt_words,
            "num_prompts": num_prompts,
            "phase": phase,
            "active_kv_tokens_approx": int(active_kv_tokens),
            "block_rounded_kv_tokens_approx": int(block_rounded_tokens),
            "kv_bytes_per_layer_approx": int(active_kv_tokens * per_token_layer),
            "kv_bytes_all_layers_approx": int(
                active_kv_tokens * per_token_layer * num_layers
            ),
            "kv_block_bytes_per_layer_approx": int(
                block_rounded_tokens * per_token_layer
            ),
            "kv_block_bytes_all_layers_approx": int(
                block_rounded_tokens * per_token_layer * num_layers
            ),
        })
        out.append(item)
    return out


def _run_vllm_layer_sweep(
    args: argparse.Namespace,
    *,
    kv_info: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    if args.skip_vllm:
        return {"skipped": True, "records": [], "case_summaries": []}

    import torch
    from vllm import LLM, SamplingParams

    layer_trace = run_dir / "qwen3_moe_layer_timing.jsonl"
    pre_expert_trace = run_dir / "qwen3_moe_pre_expert_timing.jsonl"
    for path in (layer_trace, pre_expert_trace):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    os.environ["VLLM_QWEN3_MOE_LAYER_TIMING_ENABLE"] = "1"
    os.environ["VLLM_QWEN3_MOE_LAYER_TIMING_FILE"] = str(layer_trace)
    os.environ["VLLM_UVM_PRE_EXPERT_TIMING_ENABLE"] = "1"
    os.environ["VLLM_UVM_PRE_EXPERT_TIMING_FILE"] = str(pre_expert_trace)
    os.environ.setdefault("VLLM_CPU_OFFLOAD_BACKEND", "uva")

    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        distributed_executor_backend="uni",
        seed=args.seed,
    )
    sampling = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    all_records: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    # Drop profile/warmup layer traces emitted during engine initialization.
    warmup_records, cursor = _load_jsonl_since(layer_trace, 0)
    print(
        f"discarded_engine_warmup_records={len(warmup_records)}",
        flush=True,
    )
    try:
        for case_name, prompt_words, num_prompts in _parse_cases(args.cases):
            prompts = [_make_prompt(prompt_words, idx) for idx in range(num_prompts)]
            t0 = time.perf_counter()
            _ = llm.generate(prompts, sampling, use_tqdm=False)
            torch.cuda.synchronize(args.device)
            wall_ms = (time.perf_counter() - t0) * 1000.0
            new_records, cursor = _load_jsonl_since(layer_trace, cursor)
            annotated = _annotate_layer_records(
                new_records,
                case_name=case_name,
                prompt_words=prompt_words,
                num_prompts=num_prompts,
                kv_info=kv_info,
                block_size=args.block_size,
            )
            all_records.extend(annotated)
            case_summaries.append({
                "case": case_name,
                "prompt_words": prompt_words,
                "num_prompts": num_prompts,
                "wall_ms": wall_ms,
                "layer_record_count": len(annotated),
            })
            print(
                "case_done "
                f"case={case_name} records={len(annotated)} "
                f"wall_ms={wall_ms:.1f}",
                flush=True,
            )
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "skipped": False,
        "layer_trace": str(layer_trace),
        "pre_expert_trace": str(pre_expert_trace),
        "records": all_records,
        "case_summaries": case_summaries,
    }


def _summarize_layer_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = (
            rec["case"],
            rec["phase"],
            int(rec["num_tokens"]),
            int(rec["kv_len_approx"]),
            bool(rec["is_sparse_moe"]),
        )
        grouped[key].append(rec)

    rows: list[dict[str, Any]] = []
    for (case, phase, num_tokens, kv_len, is_sparse), items in sorted(grouped.items()):
        elapsed = [float(item["elapsed_ms"]) for item in items]
        first = items[0]
        rows.append({
            "case": case,
            "phase": phase,
            "num_tokens": num_tokens,
            "kv_len_approx": kv_len,
            "is_sparse_moe": is_sparse,
            "num_layers_sampled": len({int(item["layer_id"]) for item in items}),
            "record_count": len(items),
            "active_kv_tokens_approx": int(first["active_kv_tokens_approx"]),
            "block_rounded_kv_tokens_approx": int(
                first["block_rounded_kv_tokens_approx"]
            ),
            "kv_mib_per_layer_approx": _mib(first["kv_bytes_per_layer_approx"]),
            "kv_mib_all_layers_approx": _mib(first["kv_bytes_all_layers_approx"]),
            "kv_block_mib_per_layer_approx": _mib(
                first["kv_block_bytes_per_layer_approx"]
            ),
            "kv_block_mib_all_layers_approx": _mib(
                first["kv_block_bytes_all_layers_approx"]
            ),
            "elapsed": _stats(elapsed),
        })
    return rows


def _prefetch_rows_by_expert_count(prefetch: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for row in prefetch.get("rows", []):
        if "expert_count" not in row:
            continue
        rows[int(row["expert_count"])] = row
    return rows


def _hidden_expert_count(
    *,
    layer_ms: float,
    prefetch_rows: dict[int, dict[str, Any]],
    percentile_key: str,
) -> int:
    best = 0
    for count, row in sorted(prefetch_rows.items()):
        stats = row.get("uvm_prefetch_gpu_ms", {})
        value = float(stats.get(percentile_key, 0.0))
        if value > 0.0 and value <= layer_ms:
            best = count
    return best


def _build_feasibility_rows(
    layer_rows: list[dict[str, Any]],
    prefetch: dict[str, Any],
) -> list[dict[str, Any]]:
    prefetch_by_count = _prefetch_rows_by_expert_count(prefetch)
    out: list[dict[str, Any]] = []
    for row in layer_rows:
        if not str(row["phase"]).startswith("decode"):
            continue
        elapsed = row["elapsed"]
        p50 = float(elapsed["p50_ms"])
        p95 = float(elapsed["p95_ms"])
        out.append({
            "case": row["case"],
            "num_tokens": row["num_tokens"],
            "kv_len_approx": row["kv_len_approx"],
            "is_sparse_moe": row["is_sparse_moe"],
            "kv_block_mib_all_layers_approx": row[
                "kv_block_mib_all_layers_approx"
            ],
            "layer_p50_ms": p50,
            "layer_p95_ms": p95,
            "experts_hidden_by_layer_p50_vs_prefetch_p50": _hidden_expert_count(
                layer_ms=p50,
                prefetch_rows=prefetch_by_count,
                percentile_key="p50_ms",
            ),
            "experts_hidden_by_layer_p95_vs_prefetch_p95": _hidden_expert_count(
                layer_ms=p95,
                prefetch_rows=prefetch_by_count,
                percentile_key="p95_ms",
            ),
        })
    return out


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _write_report(result: dict[str, Any], report_path: Path) -> None:
    kv = result["kv_model_info"]
    expert = result["expert_info"]
    layer_rows = result["layer_summary_rows"]
    prefetch_rows = result["prefetch"].get("rows", [])
    feasibility = result["feasibility_rows"]

    md: list[str] = []
    md.append("# KV Layer Time vs Expert Prefetch Window\n")
    md.append("## Config\n")
    md.append(
        f"- GPU: {result.get('cuda_device_name', 'unknown')}\n"
        f"- Model: `{result['config']['model']}`\n"
        f"- KV per token per layer: {kv['kv_per_token_per_layer_bytes']} bytes\n"
        f"- KV per token all layers: {kv['kv_per_token_all_layers_bytes']} bytes\n"
        f"- Layers: {kv['num_layers']}, sparse MoE layers: {kv['sparse_layer_count']}\n"
        f"- One expert: {_fmt(expert['per_expert_mib'])} MiB "
        f"({expert['dtype']}, w13={expert['w13_shape']}, w2={expert['w2_shape']})\n"
    )

    md.append("## Expert Prefetch\n")
    md.append(
        _markdown_table(
            ["experts", "MiB", "UVM p50 ms", "UVM p95 ms", "H2D p50 ms"],
            [[
                str(row.get("expert_count", "")),
                _fmt(float(row.get("expert_mib", row.get("mib", 0.0)))),
                _fmt(float(row.get("uvm_prefetch_gpu_ms", {}).get("p50_ms", 0.0))),
                _fmt(float(row.get("uvm_prefetch_gpu_ms", {}).get("p95_ms", 0.0))),
                _fmt(float(row.get("h2d_copy_gpu_ms", {}).get("p50_ms", 0.0))),
            ] for row in prefetch_rows],
        )
    )

    md.append("## Layer Timing And KV Size\n")
    timing_rows = []
    for row in layer_rows:
        if row["phase"] not in (
            "decode_full_batch",
            "decode_partial_or_mixed",
            "prefill_or_chunk",
        ):
            continue
        timing_rows.append([
            row["case"],
            row["phase"],
            "sparse" if row["is_sparse_moe"] else "dense",
            str(row["num_tokens"]),
            str(row["kv_len_approx"]),
            _fmt(float(row["kv_block_mib_all_layers_approx"])),
            _fmt(float(row["elapsed"]["p50_ms"])),
            _fmt(float(row["elapsed"]["p95_ms"])),
        ])
    md.append(
        _markdown_table(
            [
                "case",
                "phase",
                "layer",
                "num_tokens",
                "kv_len",
                "KV block MiB all layers",
                "layer p50 ms",
                "layer p95 ms",
            ],
            timing_rows,
        )
    )

    md.append("## Feasibility\n")
    md.append(
        _markdown_table(
            [
                "case",
                "layer",
                "num_tokens",
                "kv_len",
                "KV block MiB",
                "layer p50 ms",
                "hide experts p50",
                "layer p95 ms",
                "hide experts p95",
            ],
            [[
                row["case"],
                "sparse" if row["is_sparse_moe"] else "dense",
                str(row["num_tokens"]),
                str(row["kv_len_approx"]),
                _fmt(float(row["kv_block_mib_all_layers_approx"])),
                _fmt(float(row["layer_p50_ms"])),
                str(row["experts_hidden_by_layer_p50_vs_prefetch_p50"]),
                _fmt(float(row["layer_p95_ms"])),
                str(row["experts_hidden_by_layer_p95_vs_prefetch_p95"]),
            ] for row in feasibility],
        )
    )

    md.append(
        "\nInterpretation: a row can hide at least one evicted expert when the "
        "layer timing column is greater than the corresponding UVM prefetch "
        "time for one expert. This is a scheduling feasibility check; it does "
        "not include contention from simultaneous KV migration or other GPU work.\n"
    )
    report_path.write_text("\n".join(md), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir
    if run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = SCRIPT_DIR / "results" / f"kv_layer_time_prefetch_window_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kv_info = _kv_model_info(args)
    sparse_layers = kv_info.get("sparse_layers") or [0]
    expert = _expert_info(args.model, int(sparse_layers[0]))
    print(
        "model_info "
        f"layers={kv_info['num_layers']} sparse_layers={kv_info['sparse_layer_count']} "
        f"kv_per_token_all_layers={kv_info['kv_per_token_all_layers_bytes']} "
        f"per_expert_mib={expert['per_expert_mib']:.3f}",
        flush=True,
    )

    prefetch = _run_prefetch_sweep(args, expert=expert)
    print("prefetch_sweep_done", flush=True)
    vllm_result = _run_vllm_layer_sweep(args, kv_info=kv_info, run_dir=run_dir)
    layer_rows = _summarize_layer_groups(vllm_result.get("records", []))
    feasibility = _build_feasibility_rows(layer_rows, prefetch)

    cuda_device_name = "unknown"
    try:
        import torch

        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(args.device)
    except Exception:
        pass

    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "run_dir": str(run_dir),
        "cuda_device_name": cuda_device_name,
        "kv_model_info": kv_info,
        "expert_info": expert,
        "prefetch": prefetch,
        "vllm": vllm_result,
        "layer_summary_rows": layer_rows,
        "feasibility_rows": feasibility,
        "passed": True,
    }
    json_path = run_dir / "kv_layer_time_prefetch_window.json"
    report_path = run_dir / "kv_layer_time_prefetch_window.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    _write_report(result, report_path)
    print(f"RESULT_JSON={json_path}")
    print(f"REPORT_MD={report_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cases", default="32:1,128:1,512:1,128:4,128:8")
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    parser.add_argument("--expert-multipliers", default="1,2,4,8")
    parser.add_argument("--prefetch-repeats", type=int, default=5)
    parser.add_argument("--skip-prefetch", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    result = run(args)
    print("KV layer time / expert prefetch experiment: PASS" if result["passed"]
          else "KV layer time / expert prefetch experiment: FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
