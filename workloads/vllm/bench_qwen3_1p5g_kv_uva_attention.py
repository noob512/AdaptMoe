#!/usr/bin/env python3
"""Measure real vLLM Qwen3-MoE attention time at ~1.5 GiB active KV.

The experiment uses vLLM's UVA CPU offload path for roughly 2 GiB of model
weights, then runs one long prompt whose single-token decode step scans about
1.5 GiB of KV across all layers. It records both:

* attention-only timing around ``self.attn(q, k, v)``;
* whole decoder-layer timing for comparison.
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
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VLLM_ROOT = SCRIPT_DIR / "vllm"
if str(VLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(VLLM_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bench_hot_cold_expert_checkpoint import DEFAULT_CHECKPOINT_DIR  # noqa: E402


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


def _gib(num_bytes: int | float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0 * 1024.0)


def _make_prompt(prompt_words: int, idx: int = 0) -> str:
    words = ["system", "memory", "attention", "window"] * max(
        1, math.ceil(prompt_words / 4)
    )
    words = words[:prompt_words]
    words[-1] = f"case{idx % 997}"
    return " ".join(words)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _kv_model_info(model: Path, kv_dtype_bytes: int) -> dict[str, Any]:
    cfg = _hf_text_config(model)
    num_layers = int(getattr(cfg, "num_hidden_layers"))
    num_attention_heads = int(getattr(cfg, "num_attention_heads"))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_attention_heads))
    head_dim = int(getattr(cfg, "head_dim", getattr(cfg, "hidden_size") //
                           num_attention_heads))
    per_token_per_layer = 2 * num_kv_heads * head_dim * int(kv_dtype_bytes)
    return {
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_dtype_bytes": int(kv_dtype_bytes),
        "kv_per_token_per_layer_bytes": per_token_per_layer,
        "kv_per_token_all_layers_bytes": per_token_per_layer * num_layers,
    }


def _token_count(model: Path, prompt: str) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)
    return len(tokenizer.encode(prompt, add_special_tokens=True))


def _find_prompt_words_for_prefill_tokens(
    model: Path,
    target_prefill_tokens: int,
) -> tuple[int, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), trust_remote_code=True)

    def count(words: int) -> int:
        return len(tokenizer.encode(_make_prompt(words), add_special_tokens=True))

    lo = 1
    hi = max(2, target_prefill_tokens + 32)
    best_words = lo
    best_tokens = count(lo)
    while count(hi) < target_prefill_tokens:
        best_words = hi
        best_tokens = count(hi)
        hi *= 2

    while lo <= hi:
        mid = (lo + hi) // 2
        tokens = count(mid)
        if abs(tokens - target_prefill_tokens) < abs(
            best_tokens - target_prefill_tokens
        ):
            best_words = mid
            best_tokens = tokens
        if tokens < target_prefill_tokens:
            lo = mid + 1
        elif tokens > target_prefill_tokens:
            hi = mid - 1
        else:
            return mid, tokens
    return best_words, best_tokens


def _summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    params = [r for r in records if r.get("event") == "cpu_offload_parameter"]
    total_bytes = sum(int(r.get("bytes", 0)) for r in params)
    uva_bytes = sum(
        int(r.get("bytes", 0)) for r in params if bool(r.get("uva_offloading"))
    )
    uvm_bytes = sum(
        int(r.get("bytes", 0)) for r in params if bool(r.get("uvm_offloading"))
    )
    expert_like = [
        r for r in params
        if "experts" in str(r.get("full_parameter_name", ""))
        or "expert" in str(r.get("full_parameter_name", ""))
    ]
    by_module = Counter(str(r.get("module_class", "unknown")) for r in params)
    return {
        "parameter_count": len(params),
        "total_offload_bytes": int(total_bytes),
        "total_offload_gib": _gib(total_bytes),
        "uva_offload_bytes": int(uva_bytes),
        "uva_offload_gib": _gib(uva_bytes),
        "uvm_offload_bytes": int(uvm_bytes),
        "uvm_offload_gib": _gib(uvm_bytes),
        "expert_like_parameter_count": len(expert_like),
        "expert_like_bytes": int(sum(int(r.get("bytes", 0)) for r in expert_like)),
        "expert_like_gib": _gib(sum(int(r.get("bytes", 0)) for r in expert_like)),
        "module_class_counts": dict(by_module),
        "first_parameters": [
            {
                "name": r.get("full_parameter_name"),
                "bytes": int(r.get("bytes", 0)),
                "module_class": r.get("module_class"),
                "uva_offloading": bool(r.get("uva_offloading")),
                "uvm_offloading": bool(r.get("uvm_offloading")),
            }
            for r in params[:12]
        ],
    }


def _decode_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        try:
            if int(rec.get("num_tokens", -1)) == 1:
                out.append(rec)
        except (TypeError, ValueError):
            continue
    return out


def _records_by_layer(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for rec in records:
        try:
            layer_id = int(rec["layer_id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[layer_id] = rec
    return out


def _kv_footprint(
    *,
    kv_len: int,
    kv_info: dict[str, Any],
    block_size: int,
) -> dict[str, Any]:
    rounded_tokens = int(math.ceil(kv_len / block_size) * block_size)
    per_layer = int(kv_info["kv_per_token_per_layer_bytes"])
    layers = int(kv_info["num_layers"])
    active_per_layer = kv_len * per_layer
    active_all = active_per_layer * layers
    rounded_per_layer = rounded_tokens * per_layer
    rounded_all = rounded_per_layer * layers
    return {
        "kv_len": int(kv_len),
        "block_rounded_tokens": int(rounded_tokens),
        "active_kv_bytes_per_layer": int(active_per_layer),
        "active_kv_mib_per_layer": _mib(active_per_layer),
        "active_kv_bytes_all_layers": int(active_all),
        "active_kv_mib_all_layers": _mib(active_all),
        "active_kv_gib_all_layers": _gib(active_all),
        "block_kv_bytes_per_layer": int(rounded_per_layer),
        "block_kv_mib_per_layer": _mib(rounded_per_layer),
        "block_kv_bytes_all_layers": int(rounded_all),
        "block_kv_mib_all_layers": _mib(rounded_all),
        "block_kv_gib_all_layers": _gib(rounded_all),
    }


def _write_report(result: dict[str, Any], report_path: Path) -> None:
    attn = result["decode_attention"]
    layer = result["decode_layer"]
    kv = result["kv_footprint"]
    offload = result["cpu_offload"]
    rows = result["per_layer_rows"]
    steady_rows = [row for row in rows if int(row["layer_id"]) != 0]
    steady_attn = _stats([float(row["attention_ms"]) for row in steady_rows])
    steady_layer = _stats([float(row["layer_ms"]) for row in steady_rows])

    md: list[str] = []
    md.append("# Qwen3-MoE 1.5GiB KV UVA Attention Timing\n")
    md.append("## Config\n")
    md.append(
        f"- Model: `{result['config']['model']}`\n"
        f"- CPU offload backend: `{result['config']['cpu_offload_backend']}`\n"
        f"- CPU offload target: {result['config']['cpu_offload_gb']} GiB\n"
        f"- max_model_len: {result['config']['max_model_len']}\n"
        f"- max_num_batched_tokens: {result['config']['max_num_batched_tokens']}\n"
        f"- prompt_words: {result['prompt']['prompt_words']}\n"
        f"- prompt_tokens_estimate: {result['prompt']['prompt_tokens_estimate']}\n"
        f"- wall_ms: {result['wall_ms']:.3f}\n"
    )

    md.append("## CPU Offload\n")
    md.append(
        f"- Actual offloaded: {offload['total_offload_gib']:.3f} GiB "
        f"({offload['parameter_count']} parameters)\n"
        f"- UVA offloaded: {offload['uva_offload_gib']:.3f} GiB\n"
        f"- Expert-like offloaded: {offload['expert_like_gib']:.3f} GiB\n"
    )

    md.append("## KV Footprint\n")
    md.append(
        f"- decode kv_len: {kv['kv_len']}\n"
        f"- per-layer KV: {kv['block_kv_mib_per_layer']:.3f} MiB\n"
        f"- all-layer KV: {kv['block_kv_mib_all_layers']:.3f} MiB "
        f"({kv['block_kv_gib_all_layers']:.3f} GiB)\n"
    )

    md.append("## Decode Timing\n")
    md.append(
        "| metric | count | p50 ms | p90 ms | p95 ms | mean ms | max ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| attention-only | {attn['count']} | {attn['p50_ms']:.6f} | "
        f"{attn['p90_ms']:.6f} | {attn['p95_ms']:.6f} | "
        f"{attn['mean_ms']:.6f} | {attn['max_ms']:.6f} |\n"
        f"| whole layer | {layer['count']} | {layer['p50_ms']:.6f} | "
        f"{layer['p90_ms']:.6f} | {layer['p95_ms']:.6f} | "
        f"{layer['mean_ms']:.6f} | {layer['max_ms']:.6f} |\n"
    )
    md.append(
        "\n## Steady State (layer 1-47)\n"
        "| metric | count | p50 ms | p90 ms | p95 ms | mean ms | max ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        f"| attention-only | {steady_attn['count']} | {steady_attn['p50_ms']:.6f} | "
        f"{steady_attn['p90_ms']:.6f} | {steady_attn['p95_ms']:.6f} | "
        f"{steady_attn['mean_ms']:.6f} | {steady_attn['max_ms']:.6f} |\n"
        f"| whole layer | {steady_layer['count']} | {steady_layer['p50_ms']:.6f} | "
        f"{steady_layer['p90_ms']:.6f} | {steady_layer['p95_ms']:.6f} | "
        f"{steady_layer['mean_ms']:.6f} | {steady_layer['max_ms']:.6f} |\n"
    )

    md.append("## Per Layer Decode Rows\n")
    md.append("| layer | kv_len | attention ms | whole layer ms |\n")
    md.append("| ---: | ---: | ---: | ---: |\n")
    for row in rows:
        md.append(
            f"| {row['layer_id']} | {row['kv_len_approx']} | "
            f"{row['attention_ms']:.6f} | {row['layer_ms']:.6f} |\n"
        )

    md.append(
        "\nNote: decode does not physically concatenate KV into one tensor; "
        "vLLM scans the paged KV cache through block tables. The attention-only "
        "timing wraps the real `self.attn(q, k, v)` call in Qwen3MoeAttention. "
        "Layer 0 is a one-time warmup outlier from backend/kernel startup, so "
        "the steady-state table is the more representative decode number.\n"
    )
    report_path.write_text("".join(md), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    run_dir = args.run_dir
    if run_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = SCRIPT_DIR / "results" / f"qwen3_1p5g_kv_uva_attention_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    layer_trace = run_dir / "qwen3_moe_layer_timing.jsonl"
    attention_trace = run_dir / "qwen3_moe_attention_timing.jsonl"
    manifest_file = run_dir / "cpu_offload_manifest.jsonl"
    for path in (layer_trace, attention_trace, manifest_file):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    target_prefill_tokens = max(1, int(args.target_decode_kv_len) - 1)
    if args.prompt_words is None:
        prompt_words, prompt_tokens = _find_prompt_words_for_prefill_tokens(
            args.model, target_prefill_tokens
        )
    else:
        prompt_words = int(args.prompt_words)
        prompt_tokens = _token_count(args.model, _make_prompt(prompt_words))
    prompt = _make_prompt(prompt_words)

    kv_info = _kv_model_info(args.model, args.kv_dtype_bytes)
    os.environ["VLLM_CPU_OFFLOAD_BACKEND"] = args.cpu_offload_backend
    os.environ["VLLM_CPU_OFFLOAD_MANIFEST_FILE"] = str(manifest_file)
    os.environ["VLLM_CPU_OFFLOAD_MANIFEST_RESET"] = "1"
    os.environ["VLLM_QWEN3_MOE_LAYER_TIMING_ENABLE"] = "1"
    os.environ["VLLM_QWEN3_MOE_LAYER_TIMING_FILE"] = str(layer_trace)
    os.environ["VLLM_QWEN3_MOE_ATTENTION_TIMING_ENABLE"] = "1"
    os.environ["VLLM_QWEN3_MOE_ATTENTION_TIMING_FILE"] = str(attention_trace)

    llm = LLM(
        model=str(args.model),
        enforce_eager=True,
        cpu_offload_gb=args.cpu_offload_gb,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
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

    _, layer_cursor = _load_jsonl_since(layer_trace, 0)
    _, attention_cursor = _load_jsonl_since(attention_trace, 0)
    t0 = time.perf_counter()
    try:
        _ = llm.generate([prompt], sampling, use_tqdm=False)
        torch.cuda.synchronize(args.device)
    finally:
        wall_ms = (time.perf_counter() - t0) * 1000.0
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    layer_records, _ = _load_jsonl_since(layer_trace, layer_cursor)
    attention_records, _ = _load_jsonl_since(attention_trace, attention_cursor)
    decode_layer = _decode_records(layer_records)
    decode_attention = _decode_records(attention_records)
    if not decode_attention:
        raise RuntimeError("no decode attention timing records captured")

    target_kv_len = max(int(r.get("kv_len_approx", 0)) for r in decode_attention)
    decode_attention = [
        r for r in decode_attention if int(r.get("kv_len_approx", 0)) == target_kv_len
    ]
    decode_layer = [
        r for r in decode_layer if int(r.get("kv_len_approx", 0)) == target_kv_len
    ]

    attention_by_layer = _records_by_layer(decode_attention)
    layer_by_layer = _records_by_layer(decode_layer)
    per_layer_rows: list[dict[str, Any]] = []
    for layer_id in sorted(attention_by_layer):
        attn_rec = attention_by_layer[layer_id]
        layer_rec = layer_by_layer.get(layer_id)
        per_layer_rows.append({
            "layer_id": layer_id,
            "kv_len_approx": int(attn_rec.get("kv_len_approx", 0)),
            "attention_ms": float(attn_rec.get("elapsed_ms", 0.0)),
            "layer_ms": float(layer_rec.get("elapsed_ms", 0.0))
            if layer_rec else 0.0,
        })

    manifest_records = _load_jsonl(manifest_file)
    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "run_dir": str(run_dir),
        "prompt": {
            "prompt_words": int(prompt_words),
            "prompt_tokens_estimate": int(prompt_tokens),
            "target_decode_kv_len": int(args.target_decode_kv_len),
        },
        "kv_model_info": kv_info,
        "kv_footprint": _kv_footprint(
            kv_len=target_kv_len,
            kv_info=kv_info,
            block_size=args.block_size,
        ),
        "cpu_offload": _summarize_manifest(manifest_records),
        "decode_attention": _stats(
            [float(r["elapsed_ms"]) for r in decode_attention]
        ),
        "decode_layer": _stats([float(r["elapsed_ms"]) for r in decode_layer]),
        "per_layer_rows": per_layer_rows,
        "wall_ms": wall_ms,
        "artifacts": {
            "layer_trace": str(layer_trace),
            "attention_trace": str(attention_trace),
            "cpu_offload_manifest": str(manifest_file),
        },
        "passed": True,
    }
    json_path = run_dir / "qwen3_1p5g_kv_uva_attention.json"
    report_path = run_dir / "qwen3_1p5g_kv_uva_attention.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    _write_report(result, report_path)
    print(f"RESULT_JSON={json_path}")
    print(f"REPORT_MD={report_path}")
    print(
        "summary "
        f"kv_len={target_kv_len} "
        f"kv_gib={result['kv_footprint']['block_kv_gib_all_layers']:.3f} "
        f"offload_gib={result['cpu_offload']['total_offload_gib']:.3f} "
        f"attn_p50_ms={result['decode_attention']['p50_ms']:.6f} "
        f"layer_p50_ms={result['decode_layer']['p50_ms']:.6f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-decode-kv-len", type=int, default=16384)
    parser.add_argument("--prompt-words", type=int)
    parser.add_argument("--max-tokens", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=16400)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16400)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--cpu-offload-gb", type=float, default=2.0)
    parser.add_argument("--cpu-offload-backend", default="uva")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-dtype-bytes", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    result = run(args)
    print("Qwen3 1.5GiB KV UVA attention experiment: PASS"
          if result["passed"]
          else "Qwen3 1.5GiB KV UVA attention experiment: FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
