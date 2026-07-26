#!/usr/bin/env python3
"""llama-server + ShareGPT benchmark atomic runner: single config, single run, JSON output.

Usage:
    uv run python configs/server_bench.py --output results/server_default.json
    uv run python configs/server_bench.py --uvm --output results/server_uvm.json

Starts llama-server, benchmarks with `vllm bench serve` (from vllm workload),
parses output, stops server.

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
from common import cleanup_gpu, wait_for_server, stop_server, parse_vllm_bench_output

VLLM_WORKLOAD_DIR = WORKLOADS_DIR / "vllm"
LLAMA_SERVER = WORKLOAD_DIR / "build" / "bin" / "llama-server"
DEFAULT_MODEL = Path.home() / ".cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf"
DATASET_DIR = WORKLOAD_DIR / "datasets"
DATASET_PATH = DATASET_DIR / "sharegpt_vicuna.json"
# gpt-oss GGUF has no HF tokenizer; use Qwen (locally cached) for token counting
DEFAULT_TOKENIZER = "Qwen/Qwen3-30B-A3B-FP8"

SERVER_STARTUP_TIMEOUT = 300
SERVER_CHECK_INTERVAL = 5


def run_server_bench(model: str, uvm: bool, ctx: int, port: int,
                     prompts: int, max_concurrency: int,
                     request_rate: float) -> dict:
    """Start llama-server, run vllm bench serve, stop server, return result."""
    # Build server command
    cmd = [
        str(LLAMA_SERVER),
        "-m", model,
        "-c", str(ctx),
        "-ngl", "99",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    env = os.environ.copy()
    if uvm:
        env["GGML_CUDA_ENABLE_UNIFIED_MEMORY"] = "1"
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"

    print(f"Starting llama-server (uvm={uvm}, ctx={ctx}, port={port})...", file=sys.stderr)
    server_proc = subprocess.Popen(
        cmd, env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    try:
        if not wait_for_server("127.0.0.1", port, SERVER_STARTUP_TIMEOUT,
                              check_interval=SERVER_CHECK_INTERVAL, process=server_proc):
            print("ERROR: Server failed to start", file=sys.stderr)
            stop_server(server_proc)
            sys.exit(1)

        print(f"Server ready. Running vllm bench serve ({prompts} prompts)...", file=sys.stderr)

        # Download dataset if needed
        if not DATASET_PATH.exists():
            print("Downloading ShareGPT dataset...", file=sys.stderr)
            subprocess.run(
                [sys.executable, str(WORKLOAD_DIR / "download_sharegpt.py"), "--dataset", "vicuna"],
                cwd=str(WORKLOAD_DIR),
            )

        # Use vllm bench serve from the vllm workload as the benchmark client
        model_name = Path(model).stem
        bench_cmd = (
            f"uv run --directory {VLLM_WORKLOAD_DIR} vllm bench serve "
            f"--model {model_name} "
            f"--tokenizer {DEFAULT_TOKENIZER} "
            f"--dataset-name sharegpt "
            f"--dataset-path {DATASET_PATH} "
            f"--base-url http://127.0.0.1:{port} "
            f"--num-prompts {prompts} "
            f"--max-concurrency {max_concurrency} "
            f"--request-rate {request_rate}"
        )

        start = time.time()
        bench_result = subprocess.run(
            bench_cmd, shell=True,
            capture_output=True, text=True,
        )
        elapsed = time.time() - start

        bench_output = bench_result.stdout + bench_result.stderr

        if bench_result.returncode != 0:
            print(f"vllm bench serve failed (exit {bench_result.returncode}):", file=sys.stderr)
            print(bench_output[-2000:], file=sys.stderr)
            stop_server(server_proc)
            sys.exit(1)

        metrics = parse_vllm_bench_output(bench_output)

    finally:
        print("Stopping server...", file=sys.stderr)
        stop_server(server_proc)
        time.sleep(3)

    config_name = "server_bench"
    if uvm:
        config_name += "_uvm"

    return {
        "workload": "llama.cpp",
        "config": config_name,
        "params": {
            "model": Path(model).name,
            "uvm": uvm,
            "ctx": ctx,
            "prompts": prompts,
            "max_concurrency": max_concurrency,
            "request_rate": request_rate,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(elapsed, 2),
        "raw": {"bench_output": bench_output},
    }


def main():
    parser = argparse.ArgumentParser(description="llama-server + ShareGPT benchmark single run")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Model path")
    parser.add_argument("--uvm", action="store_true", help="Enable UVM")
    parser.add_argument("--ctx", type=int, default=65536, help="Context size")
    parser.add_argument("--port", type=int, default=8013, help="Server port")
    parser.add_argument("--prompts", type=int, default=100, help="Number of ShareGPT prompts")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Max concurrent requests")
    parser.add_argument("--request-rate", type=float, default=0.2, help="Requests per second")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup")
    args = parser.parse_args()

    if not LLAMA_SERVER.exists():
        print(f"ERROR: llama-server not found at {LLAMA_SERVER}", file=sys.stderr)
        print(f"Run: cd {WORKLOAD_DIR} && make build-cuda-no-vmm", file=sys.stderr)
        sys.exit(1)
    if not Path(args.model).exists():
        print(f"ERROR: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)
    if not VLLM_WORKLOAD_DIR.exists():
        print(f"ERROR: vllm workload not found at {VLLM_WORKLOAD_DIR}", file=sys.stderr)
        sys.exit(1)

    if not args.no_cleanup:
        cleanup_gpu()

    result = run_server_bench(
        args.model, args.uvm, args.ctx, args.port,
        args.prompts, args.max_concurrency,
        args.request_rate,
    )

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
