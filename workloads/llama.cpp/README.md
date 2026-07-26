# llama.cpp Expert Offloading Benchmark (Experiment 1)

Benchmarks llama.cpp with the GPT-OSS-120B MoE model (~59 GiB) on a 32GB GPU,
comparing framework CPU offloading with UVM oversubscription.

- **Paper**: Figure 6 (RQ1), MoE expert offloading
- **Model**: GPT-OSS-120B MXFP4 MoE (116.83B params, 59.02 GiB)
- **GPU**: RTX 5090 (32 GB VRAM) — model is ~1.8x GPU memory

## Quick Start (Tested on RTX 5090 / CUDA 12.9 / Ubuntu 24.04)

```bash
cd workloads/llama.cpp

# 1. Build llama.cpp from source (submodule)
git submodule update --init llama.cpp
make build-cuda-no-vmm                       # CUDA + NO_VMM for UVM compatibility

# 2. Create venv and install dependencies
uv venv
uv sync                                      # installs requests, numpy, matplotlib, etc.

# 3. Download models
bash download_models.sh 20b                  # gpt-oss-20b (~12 GiB, quick test)
bash download_models.sh 120b                 # gpt-oss-120b (~59 GiB, paper experiment)
# Models cached to ~/.cache/llama.cpp/

# 4. Verify normal mode (20B, fits in VRAM)
MODEL_20B=~/.cache/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf
./build/bin/llama-bench -m $MODEL_20B -r 1
# Expected: pp512 ~9600 tok/s, tg128 ~354 tok/s

# 5. Verify UVM oversubscription (120B, 59 GiB on 32 GB GPU)
MODEL_120B=~/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench -m $MODEL_120B -r 1
# Expected: pp512 ~138 tok/s, tg128 ~48 tok/s (UVM baseline)
```

## Build Details

**`build-cuda-no-vmm`** builds with `-DGGML_CUDA_NO_VMM=ON`. This is required because:
- VMM (Virtual Memory Management) uses `cuMemCreate`/`cuMemMap` which conflicts with
  `cudaMallocManaged` (UVM)
- Without NO_VMM, the memory pool tries VMM first, which fails on RTX 5090 (VMM: no)
- Uses GCC-12 for CUDA 12.9 compatibility

**UVM changes** in the bundled llama.cpp fork:
- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` switches allocations to `cudaMallocManaged`
- First allocation sets `cudaMemAdviseSetPreferredLocation = CPU` (data stays in RAM,
  avoids OOM during model loading since 59 GiB > 32 GB VRAM)
- On first compute forward, switches preferred location to GPU (enables GPU-initiated
  page migration for hot data)
- Forces legacy memory pool (bypasses VMM) when UVM is enabled

## Running Benchmarks

```bash
MODEL=~/.cache/llama.cpp/ggml-org_gpt-oss-120b-GGUF_gpt-oss-120b-mxfp4-00001-of-00003.gguf

# Config 1: Framework CPU offload (ncmoe=64, all MoE experts on CPU)
./build/bin/llama-bench -ncmoe 64 -m $MODEL 2>&1 | tee results/ncmoe64.log

# Config 2: Framework CPU offload (ncmoe=32)
./build/bin/llama-bench -ncmoe 32 -m $MODEL 2>&1 | tee results/ncmoe32.log

# Config 3: UVM baseline
GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 ./build/bin/llama-bench -m $MODEL \
  2>&1 | tee results/uvm_baseline.log
```

## Verified Results (2026-02-16, RTX 5090)

### GPT-OSS-20B (12 GiB, fits in VRAM)

| Mode | pp512 (tok/s) | tg128 (tok/s) |
|------|--------------|--------------|
| Normal (cudaMalloc) | 9609.57 | 354.57 |
| UVM (cudaMallocManaged) | 79.49 | 2.32 |

### GPT-OSS-120B (59 GiB, UVM oversubscription on 32 GB GPU)

| Mode | pp512 (tok/s) | tg128 (tok/s) |
|------|--------------|--------------|
| UVM baseline | 137.81 | 48.28 |

### Paper reference (GPT-OSS-120B, RTX 5090)

| Config | pp512 (tok/s) | tg128 (tok/s) |
|--------|--------------|--------------|
| ncmoe=64 (framework offload) | 245.63 | 16.34 |
| ncmoe=32 (framework offload) | 260.14 | 18.18 |
| UVM baseline | 238.48 | 7.72 |
| UVM + user hint | 144.00 | 49.31 |

## Directory Structure

```
llama.cpp/
├── README.md                 # This file
├── Makefile                  # Build targets + benchmark shortcuts
├── pyproject.toml            # uv dependencies
├── uv.lock                   # Locked dependency versions
├── llama.cpp/                # llama.cpp source with local UVM patches
├── download_models.sh        # One-click model downloader (20b/120b/all)
├── download_sharegpt.py      # ShareGPT dataset for server benchmarks
├── run_exp5_two_tenant.sh    # Co-location experiment (Figure 12)
├── uvm/                      # UVM test scripts
├── docs/                     # Analysis & investigation notes
└── results/                  # Benchmark output logs
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `VMM: no` in CUDA init | Normal for RTX 5090; use `build-cuda-no-vmm` |
| OOM with 120B model | Ensure `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` is set |
| `llama-server: command not found` | Build first: `make build-cuda-no-vmm` |
| Slow UVM performance (~1-2 tok/s tg128) | The preferred-location CPU→GPU switch happens on first inference; subsequent runs should be faster |
