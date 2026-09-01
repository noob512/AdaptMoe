# E3 native copy-on-forward vs pinned-UVA phase crossover

Date: 2026-08-28
Artifact: `artifacts/e3-native-copy-vs-uva-phase-20260828-attempt-01`

## Scope and status

This single-round qualification compares only the native vLLM copy-on-forward and pinned-UVA remote-read paths. It uses Qwen3-30B-A3B-FP8, frozen prompt token IDs, B1/B8, and five request shapes. It does not run H/A/R, KTransformers, HTTP, CUDA Graph, or continuous arrivals.

All 20 runs completed successfully. All ten copy/UVA pairs produced identical output-token hashes. B8 requests were inserted as one batch. The scheduler evidence shows one complete Prefill step followed only by strict cached B8 Decode steps, with no mixed Prefill/Decode step.

## Results

Times are milliseconds. Prefill is the synchronized complete-Prefill step. Decode is the median synchronized one-token batch step. E2E includes one complete request batch and its full output length.

| Workload | B | Copy Prefill | UVA Prefill | Prefill winner | Copy Decode | UVA Decode | Decode winner | Copy E2E | UVA E2E | E2E winner |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| Short 128→32 | 1 | 64.63 | 61.31 | UVA 1.05× | 52.86 | 48.68 | UVA 1.09× | 1,707.24 | 1,580.31 | UVA 1.08× |
| Short 128→32 | 8 | 76.03 | 203.90 | Copy 2.68× | 58.41 | 51.47 | UVA 1.13× | 1,889.58 | 1,814.36 | UVA 1.04× |
| Ordinary 512→128 | 1 | 70.08 | 122.76 | Copy 1.75× | 52.83 | 49.41 | UVA 1.07× | 6,785.47 | 6,402.64 | UVA 1.06× |
| Ordinary 512→128 | 8 | 136.85 | 824.96 | Copy 6.03× | 59.09 | 51.04 | UVA 1.16× | 7,643.56 | 7,343.73 | UVA 1.04× |
| Medium 2048→128 | 1 | 102.59 | 421.06 | Copy 4.10× | 53.00 | 48.63 | UVA 1.09× | 6,854.05 | 6,641.37 | UVA 1.03× |
| Medium 2048→128 | 8 | 563.89 | 7,474.99 | Copy 13.26× | 167.41 | 79.67 | UVA 2.10× | 21,826.37 | 17,573.79 | UVA 1.24× |
| RAG 8192→128 | 1 | 458.68 | 3,784.03 | Copy 8.25× | 161.16 | 49.45 | UVA 3.26× | 20,933.35 | 10,079.25 | UVA 2.08× |
| RAG 8192→128 | 8 | 2,729.38 | 48,136.85 | Copy 17.64× | 281.05 | 125.10 | UVA 2.25× | 38,435.00 | 64,062.12 | Copy 1.67× |
| Long 512→512 | 1 | 70.93 | 120.26 | Copy 1.70× | 52.84 | 49.62 | UVA 1.06× | 27,107.73 | 25,550.52 | UVA 1.06× |
| Long 512→512 | 8 | 137.43 | 816.70 | Copy 5.94× | 58.78 | 50.28 | UVA 1.17× | 30,189.61 | 26,574.25 | UVA 1.14× |

Prefill copy wins 9/10 cells; the only exception is Short/B1, where UVA is 3.33 ms faster. Decode UVA wins 10/10 cells. For the measured full requests, fixed UVA wins 9/10 E2E cells. RAG/B8 is the exception: the 45.41-second UVA Prefill penalty is too large for 127 Decode steps to recover, so fixed copy is 1.67× faster.

## Estimated break-even

For cells in which copy wins Prefill and UVA wins Decode:

`steps = (UVA Prefill - Copy Prefill) / (Copy Decode step - UVA Decode step)`

| Workload | B1 steps | B8 steps |
|---|---:|---:|
| Short | UVA dominates | 18.44 |
| Ordinary | 15.39 | 85.47 |
| Medium | 72.90 | 78.77 |
| RAG | 29.77 | 291.16 |
| Long | 15.33 | 79.95 |

Below a positive threshold, fixed copy is predicted to be faster; above it, fixed UVA is predicted to be faster. These are estimates from separately measured phase costs, not measurements of a hybrid runtime switch.

## Fairness and capacity controls

Native vLLM offloads whole Parameters rather than individual expert slices. Each copy/UVA pair used an identical realized Parameter signature (name, byte size, shape, dtype, and order):

- 63 Parameters / 2,493,476,896 bytes (2.322 GiB) for the small-capacity cells;
- 232 Parameters / 8,527,542,384 bytes (7.942 GiB) for Medium/B8 and RAG/B1;
- 402 Parameters / 14,762,934,464 bytes (13.749 GiB) for RAG/B8.

The larger offload sets were required to fit larger KV caches and Prefill workspaces. Consequently, copy versus UVA is controlled within each logical cell, while raw results across different cells also include the capacity-plan change.

## Qualification limits

- One independent process per arm-cell; no confidence intervals or variance estimate.
- Eager execution with `FLASH_ATTN`; CUDA Graph was not enabled.
- Results must not be pooled directly with earlier `TRITON_ATTN` or expert-slice experiments.
- Every run emitted PyTorch's `destroy_process_group()` exit warning. It was non-blocking here: all processes exited, final GPU use was 0 MiB, and no CUDA, Xid, PTE, PDE, or OOM fault was found.
