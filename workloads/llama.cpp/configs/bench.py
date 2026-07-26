#!/usr/bin/env python3
"""llama-bench atomic runner: single config, single run, JSON output.

Usage:
    uv run python configs/bench.py --ncmoe 64 --output results/bench_ncmoe64.json
    uv run python configs/bench.py --uvm --output results/bench_uvm.json

The script supports native CUDA and UVM modes.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKLOAD_DIR = SCRIPT_DIR.parent
WORKLOADS_DIR = WORKLOAD_DIR.parent
sys.path.insert(0, str(WORKLOADS_DIR / "scripts"))
from common import cleanup_gpu

LLAMA_BENCH = WORKLOAD_DIR / "build" / "bin" / "llama-bench"
DEFAULT_MODEL = Path.home() / ".cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf"


def run_bench(model: str, ncmoe: int, uvm: bool, pp: int, tg: int) -> dict:
    """Run llama-bench once and return parsed JSON result."""
    cmd = [
        str(LLAMA_BENCH),
        "-m", model,
        "-r", "1",
        "-o", "json",
        "-p", str(pp),
        "-n", str(tg),
    ]
    if ncmoe > 0:
        cmd += ["-ncmoe", str(ncmoe)]

    env = os.environ.copy()
    if uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"llama-bench failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    raw = json.loads(result.stdout)

    # Extract metrics from llama-bench JSON array
    metrics = {}
    for entry in raw:
        if entry.get("n_prompt", 0) > 0:
            metrics["pp_tok_s"] = entry["avg_ts"]
            metrics["pp_stddev"] = entry["stddev_ts"]
            metrics["pp_tokens"] = entry["n_prompt"]
        elif entry.get("n_gen", 0) > 0:
            metrics["tg_tok_s"] = entry["avg_ts"]
            metrics["tg_stddev"] = entry["stddev_ts"]
            metrics["tg_tokens"] = entry["n_gen"]

    # Build config description
    config_name = "bench"
    if uvm:
        config_name += "_uvm"
    if ncmoe > 0:
        config_name += f"_ncmoe{ncmoe}"

    return {
        "workload": "llama.cpp",
        "config": config_name,
        "params": {
            "model": Path(model).name,
            "ncmoe": ncmoe,
            "uvm": uvm,
            "pp": pp,
            "tg": tg,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(elapsed, 2),
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="llama-bench single run")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path")
    parser.add_argument("--ncmoe", type=int, default=0, help="CPU-offloaded MoE experts (0=all GPU)")
    parser.add_argument("--uvm", action="store_true", help="Enable UVM")
    parser.add_argument("--pp", type=int, default=512, help="Prompt tokens")
    parser.add_argument("--tg", type=int, default=128, help="Generation tokens")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup")
    args = parser.parse_args()

    # Validate
    if not LLAMA_BENCH.exists():
        print(f"ERROR: llama-bench not found at {LLAMA_BENCH}", file=sys.stderr)
        print(f"Run: cd {WORKLOAD_DIR} && make build-cuda-no-vmm", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    if not args.no_cleanup:
        cleanup_gpu()

    result = run_bench(args.model, args.ncmoe, args.uvm, args.pp, args.tg)

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
