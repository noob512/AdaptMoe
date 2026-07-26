# FAISS Vector Search Benchmark (Experiment 4)

Benchmarks FAISS IVF index build and search with UVM oversubscription on the SIFT dataset.

- **Paper**: Figure 9 (RQ1), vector search with GPU memory oversubscription
- **Index**: IVF4096,Flat
- **Sizes**: SIFT20M (9.5 GB), SIFT50M (24 GB), SIFT100M (48 GB, oversubscribed on 32 GB GPU)

## Quick Start (Tested on RTX 5090 / CUDA 12.9 / Ubuntu 24.04)

```bash
cd workloads/faiss

# 1. Build FAISS from source (submodule)
git submodule update --init faiss
cd faiss
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DBLA_VENDOR=OpenBLAS \
  -DCUDAToolkit_ROOT=/usr/local/cuda
make -C build -j$(nproc) swigfaiss          # ~15 min
cd ..

# 2. Create venv and install dependencies
uv venv
uv sync                                      # installs numpy<2, matplotlib
uv pip install -e faiss/build/faiss/python/  # installs faiss 1.13.0

# 3. Verify
uv run python -c "
import sys; sys.path.insert(0, 'faiss/build/faiss/python')
import faiss; print('GPU count:', faiss.get_num_gpus())
res = faiss.StandardGpuResources(); print('OK')
"
# Expected: GPU count: 1 \n OK

# 4. Download SIFT dataset (or use existing data)
bash download_sift.sh                        # ~60 GB, or --small for subset

# 5. Run benchmark
uv run python bench_gpu_1bn.py SIFT100M IVF4096,Flat -nprobe 1,4,16 -uvm
```

## Build Details

**Do NOT use `-DFAISS_OPT_LEVEL=avx2`** in cmake. The loader.py detects CPU AVX2 support
and tries to import `swigfaiss_avx2` first. Without building the `swigfaiss_avx2` target
(which requires the separate `make swigfaiss_avx2` step), `import faiss` will silently
return an empty module. The default `swigfaiss` target works correctly.

**numpy version**: FAISS 1.13.0 is compiled against numpy 1.x. The `pyproject.toml`
pins `numpy<2` to avoid `RuntimeError: module compiled using NumPy 1.x cannot run in NumPy 2.x`.

**sys.path**: `bench_gpu_1bn.py` adds `faiss/build/faiss/python` to `sys.path` (line 16).
This is necessary because the `faiss/` git submodule directory shadows the installed
`faiss` Python package when running from this directory.

## Dataset

The SIFT1B dataset must be at `faiss/benchs/bigann/`. Required files (~60 GB):

| File | Size | Description |
|------|------|-------------|
| `bigann_base.bvecs` | 47 GB | 375M base vectors (128-d uint8) |
| `bigann_learn.bvecs` | 13 GB | 100M training vectors |
| `bigann_query.bvecs` | 1.3 MB | 10K query vectors |
| `gnd/` | ~512 MB | Ground truth for recall evaluation |

```bash
bash download_sift.sh          # downloads all from ftp://ftp.irisa.fr/local/texmex/corpus/
bash download_sift.sh --small  # skip base vectors (47 GB)
```

## Running Benchmarks

### Normal GPU mode (fits in VRAM)

```bash
uv run python bench_gpu_1bn.py SIFT10M IVF4096,Flat -nprobe 1,4,16
```

### UVM oversubscription (exceeds VRAM)

```bash
# SIFT100M: 100M × 128 × 4B = ~48 GB on 32 GB GPU → true oversubscription
uv run python bench_gpu_1bn.py SIFT100M IVF4096,Flat -nprobe 1,4,16 -uvm
```

Options: `-abs N` (add batch size), `-qbs N` (query batch size), `-nocache` (skip index cache).

## Verified Results (2026-02-16, RTX 5090)

### SIFT10M, normal GPU (no UVM)

| Metric | Value |
|--------|-------|
| Add time | 2.25 s |
| Search nprobe=1 | 9.27 s (cold cache) |
| Search nprobe=4 | 0.044 s |
| 1-R@1 (nprobe=4) | 0.7182 |

### SIFT100M, UVM oversubscription (48 GB on 32 GB GPU)

| Metric | Value |
|--------|-------|
| Add time | 231.2 s |
| Search nprobe=1 | 26.5 s |
| 1-R@1 (nprobe=1) | 0.4486 |

The UVM baseline can be substantially slower than native GPU mode because FAISS uses
`cudaMallocManaged` for the full IVF index, which can cause heavy page migration under
oversubscription.

## Directory Structure

```
faiss/
├── README.md               # This file
├── pyproject.toml           # uv dependencies (numpy<2, matplotlib)
├── uv.lock                  # Locked dependencies
├── bench_gpu_1bn.py         # GPU benchmark (with -uvm flag)
├── bench_cpu_1bn.py         # CPU benchmark (for comparison)
├── download_sift.sh         # SIFT dataset downloader
├── faiss/                   # FAISS source submodule
│   ├── build/               # cmake build output
│   │   └── faiss/python/    # _swigfaiss.so + Python bindings
│   └── benchs/bigann/       # SIFT dataset location
├── docs/                    # Analysis documents
└── results/                 # Benchmark logs, JSON, & plots
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `import faiss` has no attributes | Don't use `-DFAISS_OPT_LEVEL=avx2` in cmake, or build `swigfaiss_avx2` target explicitly |
| `NumPy 1.x cannot run in NumPy 2.x` | Ensure `numpy<2` in pyproject.toml; reinstall with `uv sync` |
| `Could NOT find MKL` during cmake | Safe to ignore; falls back to OpenBLAS (`-DBLA_VENDOR=OpenBLAS`) |
| SIFT100M add time much slower than paper | Expected on stock driver; modified kernel module improves UVM page migration |
