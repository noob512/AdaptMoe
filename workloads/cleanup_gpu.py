#!/usr/bin/env python3
"""Kill all CUDA compute processes on the GPU before running benchmarks.

Usage:
    python cleanup_gpu.py          # kill compute processes, keep display
    python cleanup_gpu.py --all    # kill everything including display
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def nvidia_smi_timeout_seconds() -> int:
    value = os.environ.get("CLEANUP_GPU_NVIDIA_SMI_TIMEOUT", "5")
    try:
        timeout = int(value)
    except ValueError:
        timeout = 5
    return max(timeout, 1)


def get_gpu_processes():
    """Return list of (pid, type, name, mem) from nvidia-smi."""
    if os.environ.get("CLEANUP_GPU_DISABLE_NVIDIA_SMI", "0").lower() in (
        "1", "true", "yes", "on",
    ):
        print("Skipping nvidia-smi GPU process query (CLEANUP_GPU_DISABLE_NVIDIA_SMI=1).")
        return []

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
            timeout=nvidia_smi_timeout_seconds(),
        )
    except subprocess.TimeoutExpired:
        print("Warning: nvidia-smi timed out while querying GPU processes.", file=sys.stderr)
        return []
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    procs = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append({
                "pid": int(parts[0]),
                "name": parts[1],
                "mem_mib": int(parts[2]) if parts[2].isdigit() else 0,
            })
    return procs


def kill_processes(procs, force=False):
    """Kill listed processes. Returns number killed."""
    killed = 0
    my_pid = os.getpid()
    for p in procs:
        if p["pid"] == my_pid:
            continue
        try:
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.kill(p["pid"], sig)
            print(f"  killed PID {p['pid']} ({p['name']}, {p['mem_mib']}MiB)")
            killed += 1
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"  [skip] PID {p['pid']} ({p['name']}) - permission denied, try sudo")
    return killed


def main():
    parser = argparse.ArgumentParser(description="Clean up GPU processes before benchmarks")
    parser.add_argument("--all", action="store_true", help="Also kill display/GUI processes")
    parser.add_argument("--force", action="store_true", help="Use SIGKILL instead of SIGTERM")
    args = parser.parse_args()

    procs = get_gpu_processes()
    if not procs:
        print("No GPU compute processes found.")
        return
        
    print("开始清理GPU")
    print(f"Found {len(procs)} GPU compute process(es):")
    for p in procs:
        print(f"  PID {p['pid']:>8}  {p['mem_mib']:>6} MiB  {p['name']}")

    killed = kill_processes(procs, force=args.force)

    if killed:
        time.sleep(2)
        remaining = get_gpu_processes()
        if remaining:
            print(f"\n{len(remaining)} process(es) still running, sending SIGKILL...")
            kill_processes(remaining, force=True)
            time.sleep(1)

    # Final check
    final = get_gpu_processes()
    if final:
        print(f"\nWarning: {len(final)} process(es) still on GPU")
        for p in final:
            print(f"  PID {p['pid']:>8}  {p['mem_mib']:>6} MiB  {p['name']}")
        sys.exit(1)
    else:
        print("\nGPU cleared.")


if __name__ == "__main__":
    main()
