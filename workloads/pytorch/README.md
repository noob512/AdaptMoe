# PyTorch GNN Training Benchmark (Experiment 3)

Benchmarks Graph Neural Network (GCN) training with native CUDA and CUDA Unified Virtual Memory.

No external dataset needed -- graphs are randomly generated.

## Build

```bash
cd workloads/pytorch

# Build custom CUDA allocators
make all    # produces uvm_allocator.so and gpu_allocator.so

# Create venv and install deps
uv sync
```

## Running Benchmarks

### Config 1: Normal GPU baseline (no custom allocator, OOMs beyond ~7M nodes)

```bash
for NODES in 1000000 3000000 5000000 7000000; do
  uv run python benchmark_gnn_uvm.py --dataset random --nodes $NODES \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked \
    --report_json result/no_uvm_${NODES}.json
done
```

### Config 2: UVM baseline

The UVM allocator replaces PyTorch's default CUDA allocator with `cudaMallocManaged`. Allocations exceeding the GPU memory budget (default: `total_mem - 2 GB`) get `cudaMemAdviseSetPreferredLocation=CPU` to prevent OOM from device memory exhaustion.

```bash
for NODES in 5000000 7000000 8000000 10000000 12000000 15000000; do
  CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
    --dataset random --nodes $NODES \
    --edges_per_node 10 --features 128 --hidden 256 \
    --epochs 2 --warmup 1 --prop chunked --use_uvm \
    --report_json result/uvm_baseline_${NODES}.json
done
```

## Verification

Each run produces a JSON file with `avg_epoch_time` as the key metric.

### Quick verification

```bash
# Normal mode (1M nodes, should complete in <1s per epoch)
uv run python benchmark_gnn_uvm.py --dataset random --nodes 1000000 \
  --edges_per_node 10 --features 128 --hidden 256 \
  --epochs 2 --warmup 1 --prop chunked

# UVM mode (5M nodes, ~6s per epoch)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 5000000 \
  --edges_per_node 10 --features 128 --hidden 256 \
  --epochs 2 --warmup 1 --prop chunked --use_uvm

# UVM oversubscription (15M nodes, ~250s per epoch, 67 GB peak on 33 GB GPU)
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 uv run python benchmark_gnn_uvm.py \
  --dataset random --nodes 15000000 \
  --edges_per_node 10 --features 128 --hidden 256 \
  --epochs 2 --warmup 1 --prop chunked --use_uvm
```

### Reference results (RTX 5090, NVIDIA driver 575, CUDA 12.9)

| Nodes | Normal GPU | UVM Baseline |
|-------|-----------|-------------|
| 1M | 0.22s | - |
| 5M | ~1.1s | 5.6s |
| 10M | OOM | 145.6s |
| 15M | OOM | 247.2s |

### UVM allocator environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UVM_GPU_BUDGET_GB` | auto | GPU memory budget in GB (default: `total_mem - 2GB`) |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | `0` | Set to `1` for UVM mode |

## Directory Structure

```
pytorch/
├── Makefile                 # Build allocator .so files
├── README.md                # This file
├── pyproject.toml           # Python dependencies (uv)
├── uv.lock                  # Version lock
├── benchmark_gnn_uvm.py     # Main benchmark script
├── uvm_allocator.c          # Custom UVM CUDA allocator
├── gpu_allocator.c          # Pure GPU allocator (for baseline)
├── result/                  # Benchmark output JSON
└── docs/                    # Analysis notes
```
