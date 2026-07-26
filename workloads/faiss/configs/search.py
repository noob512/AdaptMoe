#!/usr/bin/env python3
"""FAISS vector search atomic runner: single dataset, single run, JSON output.

Usage:
    uv run python configs/search.py --dataset SIFT100M --uvm --output results/sift100m_uvm.json
    uv run python configs/search.py --dataset SIFT20M --output results/sift20m_gpu.json

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

BENCH_SCRIPT = WORKLOAD_DIR / "bench_gpu_1bn.py"
DATA_DIR = WORKLOAD_DIR / "faiss" / "benchs" / "bigann"


def run_search(dataset: str, uvm: bool, nprobe: str, index_key: str) -> dict:
    """Run FAISS benchmark once and return result."""
    # bench_gpu_1bn.py auto-saves JSON to results/ dir, but we also capture stdout
    cmd = [
        sys.executable, str(BENCH_SCRIPT),
        dataset,
        index_key,
        "-nprobe", nprobe,
    ]
    if uvm:
        cmd.append("-uvm")

    # Must run from WORKLOAD_DIR so bench_gpu_1bn.py finds faiss/benchs/bigann/
    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(WORKLOAD_DIR),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"FAISS benchmark failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr[-2000:] if result.stderr else "(no stderr)", file=sys.stderr)
        print(result.stdout[-2000:] if result.stdout else "(no stdout)", file=sys.stderr)
        sys.exit(1)

    # Find the auto-saved JSON result file (most recent)
    results_dir = WORKLOAD_DIR / "results"
    json_files = sorted(results_dir.glob(f"{dataset}_{index_key.replace(',', '_')}*.json"), key=lambda f: f.stat().st_mtime)
    raw = {}
    if json_files:
        raw = json.loads(json_files[-1].read_text())

    # Extract metrics
    metrics = {}
    if "index_add" in raw:
        metrics["add_time_s"] = raw["index_add"].get("total_time")
    if "search" in raw:
        for s in raw["search"]:
            np = s.get("nprobe", "?")
            metrics[f"search_nprobe{np}_s"] = s.get("search_time")
            metrics[f"recall_nprobe{np}"] = s.get("recall", {}).get("1-R@1")
    if "summary" in raw:
        metrics["total_build_time_s"] = raw["summary"].get("total_build_time")

    config_name = f"search_{dataset.lower()}"
    if uvm:
        config_name += "_uvm"

    return {
        "workload": "faiss",
        "config": config_name,
        "params": {
            "dataset": dataset,
            "index_key": index_key,
            "uvm": uvm,
            "nprobe": nprobe,
        },
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "duration_s": round(elapsed, 2),
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="FAISS search single run")
    parser.add_argument("--dataset", default="SIFT100M",
                        choices=["SIFT1M", "SIFT10M", "SIFT20M", "SIFT50M", "SIFT100M"],
                        help="Dataset size")
    parser.add_argument("--uvm", action="store_true", help="Enable UVM")
    parser.add_argument("--nprobe", default="1,4,16", help="Probe values (comma-separated)")
    parser.add_argument("--index-key", default="IVF4096,Flat", help="FAISS index type")
    parser.add_argument("--output", "-o", help="Output JSON path (default: stdout)")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip GPU cleanup")
    args = parser.parse_args()

    if not BENCH_SCRIPT.exists():
        print(f"ERROR: bench_gpu_1bn.py not found at {BENCH_SCRIPT}", file=sys.stderr)
        sys.exit(1)
    if not DATA_DIR.exists():
        print(f"ERROR: SIFT dataset not found at {DATA_DIR}", file=sys.stderr)
        print(f"Run: bash {WORKLOAD_DIR}/download_sift.sh", file=sys.stderr)
        sys.exit(1)

    if not args.no_cleanup:
        cleanup_gpu()

    result = run_search(args.dataset, args.uvm, args.nprobe, args.index_key)

    output_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(output_json)
        print(f"Result written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
