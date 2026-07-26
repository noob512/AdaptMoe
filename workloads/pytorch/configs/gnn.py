#!/usr/bin/env python3
"""PyTorch GNN training atomic runner: single config, single run, JSON output.

Usage:
    uv run python configs/gnn.py --nodes 5000000 --output results/gnn_normal_5M.json
    uv run python configs/gnn.py --nodes 10000000 --uvm --output results/gnn_uvm_10M.json

The script supports native CUDA and UVM allocator modes.
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

BENCHMARK = WORKLOAD_DIR / "benchmark_gnn_uvm.py"


def run_gnn(nodes: int, uvm: bool, epochs: int, warmup: int) -> dict:
    """Run GNN benchmark once and return result."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        report_path = f.name

    cmd = [
        sys.executable, str(BENCHMARK),
        "--dataset", "random",
        "--nodes", str(nodes),
        "--edges_per_node", "10",
        "--features", "128",
        "--hidden", "256",
        "--epochs", str(epochs),
        "--warmup", str(warmup),
        "--prop", "chunked",
        "--report_json", report_path,
    ]
    if uvm:
        cmd.append("--use_uvm")

    env = os.environ.copy()
    if uvm:
        env["CUDA_MANAGED_FORCE_DEVICE_ALLOC"] = "1"

    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"GNN benchmark failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[-2000:] if result.stderr else "(no stderr)", file=sys.stderr)
        sys.exit(1)

    raw = json.loads(Path(report_path).read_text())
    Path(report_path).unlink(missing_ok=True)

    # Extract metrics
    metrics = {
        "avg_epoch_time_s": raw.get("avg_epoch_time_s"),
        "median_epoch_time_s": raw.get("median_epoch_time_s"),
    }

    memory = {}
    if "memory" in raw:
        memory = raw["memory"]
    if "uvm_stats" in raw:
        memory["peak_uvm_gb"] = raw["uvm_stats"].get("peak_allocated_GB")

    config_name = "gnn_uvm" if uvm else "gnn_normal"

    return {
        "workload": "pytorch",
        "config": config_name,
        "params": {
            "nodes": nodes,
            "uvm": uvm,
            "epochs": epochs,
            "warmup": warmup,
        },
        "metrics": metrics,
        "memory": memory,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(elapsed, 2),
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="GNN training single run")
    parser.add_argument("--nodes", type=int, default=5_000_000, help="Number of graph nodes")
    parser.add_argument("--uvm", action="store_true", help="Enable UVM allocator")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup")
    args = parser.parse_args()

    if not BENCHMARK.exists():
        print(f"ERROR: benchmark_gnn_uvm.py not found at {BENCHMARK}", file=sys.stderr)
        sys.exit(1)

    if not args.no_cleanup:
        cleanup_gpu()

    result = run_gnn(args.nodes, args.uvm, args.epochs, args.warmup)

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
