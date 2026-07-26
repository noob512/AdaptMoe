# vLLM KV-cache Offloading Benchmark (Experiment 2)

Benchmarks vLLM with different KV-cache memory strategies to evaluate gpu_ext's UVM optimization for LLM serving.

## Prerequisites

### 1. Install vLLM from source (submodule)

vLLM is included as a git submodule at `workloads/vllm/vllm/` (UVM fork, pinned to commit `3ec7b051`):

```bash
# Initialize submodule (if not already done)
git submodule update --init workloads/vllm/vllm

# Install into workload venv
cd workloads/vllm
uv venv
uv pip install -e vllm/
```

### 2. Build and install UVM allocator

The UVM allocator (`uvm_allocator.abi3.so`) intercepts PyTorch's CUDA memory allocations and replaces them with `cudaMallocManaged`:

```bash
cd workloads/vllm/vllm/uvm_test
make uvm
cp uvm_allocator.so ../vllm/uvm_allocator.abi3.so
```

### 3. Download ShareGPT dataset

```bash
cd workloads/vllm
make download-datasets
```

Model (Qwen/Qwen3-30B-A3B-FP8, ~29 GiB) is auto-downloaded by vLLM on first run.

## How UVM Mode Works

When `VLLM_USE_UVM=1` is set, vLLM's custom allocator replaces PyTorch's default CUDA allocator with one backed by `cudaMallocManaged`. This enables memory oversubscription (allocating more GPU memory than physically available).

**Key mechanism**: The allocator maintains a GPU memory budget (`total_mem - 2 GB` headroom). Allocations within budget stay on GPU (model weights). Allocations exceeding budget get `cudaMemAdviseSetPreferredLocation=CPU` and `cudaMemAdviseSetAccessedBy=GPU`, placing them on CPU with GPU access via demand paging.

**Why `--max-num-seqs 16`**: With NVIDIA driver 575+, CUDA does not automatically evict managed memory pages to satisfy `cudaMalloc` (device memory) requests. Since cuBLAS/Triton kernels need device memory for workspace, we must limit the number of concurrent KV-cache entries to reduce GPU memory pressure. This is controlled via `--max-num-seqs`.

### UVM allocator environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_USE_UVM` | `0` | Enable UVM allocator (`1` to enable) |
| `VLLM_UVM_PREFER_CPU` | `2` (auto) | `0`=disabled, `1`=all CPU, `2`=auto (above budget→CPU) |
| `VLLM_UVM_GPU_BUDGET_GB` | auto | GPU memory budget in GB (default: `total_mem - 2GB`) |
| `VLLM_UVM_LOG_FILE` | `vllm_uvm_allocations.log` | Path for allocation log |

## Running Benchmarks

### One-click experiment (all configs)

```bash
cd workloads/vllm
uv run bash run_exp2_kv_offload.sh
```

### Individual baselines

```bash
# CPU Offload (8GB, no UVM)
uv run python uvm/test_uvm_baselines.py \
  --baselines cpu_offload \
  --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt --num-prompts 100 \
    --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 512 --seed 42 --request-rate 5"

# UVM Baseline
uv run python uvm/test_uvm_baselines.py \
  --baselines uvm_baseline \
  --bench-args "--model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset-name sharegpt --num-prompts 100 \
    --dataset-path datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --sharegpt-output-len 512 --seed 42 --request-rate 5"
```

## Verification

Results are saved to `results/uvm_baseline_results_YYYYMMDD_HHMMSS.json`.

### Reference results (RTX 5090, 100 prompts, NVIDIA driver 575, CUDA 12.9)

| Config | Mean TTFT (ms) | Mean TPOT (ms) | Output Throughput (tok/s) | Duration (s) |
|--------|---------------|----------------|--------------------------|-------------|
| CPU Offload (8GB) | 1,120 | 230 | 364 | 141 |
| UVM Baseline | 261,163 | 152 | 91 | 561 |

**Note on TTFT**: UVM baseline shows high TTFT due to `--max-num-seqs 16` causing request queuing (only 16 concurrent sequences vs 100 for cpu_offload). Per-token generation latency (TPOT=152ms) is actually better than cpu_offload. Peak output throughput reaches 272 tok/s.

### Quick manual verification

```bash
# Start server with UVM
VLLM_USE_UVM=1 uv run vllm serve Qwen/Qwen3-30B-A3B-FP8 \
  --enforce-eager --max-num-seqs 16

# In another terminal, test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-30B-A3B-FP8","prompt":"Hello","max_tokens":32}'
```

## Directory Structure

```
vllm/
├── vllm/                 # vLLM source (git submodule, UVM fork)
├── Makefile              # Quick bench target + dataset download
├── README.md             # This file
├── pyproject.toml        # Python dependencies (uv)
├── uv.lock               # Version lock
├── run_exp2_kv_offload.sh  # One-click experiment script
├── uvm/
│   ├── test_uvm_baselines.py   # Main benchmark automation
│   └── first-iter/
│       └── generate_figures.py  # Plot generation
├── docs/                 # Analysis & setup notes
├── datasets/             # ShareGPT (gitignored)
└── results/              # Benchmark output JSON + logs
```
