# Workloads

This directory contains the CUDA and Unified Virtual Memory workloads used to
evaluate the project.

## Requirements

- CUDA 12.8 or newer
- Python 3.12 or newer with `uv`
- PyTorch 2.9 or newer
- The NVIDIA kernel module from `kernel-module/`

Individual workloads may require model weights or datasets that are not stored
in Git. See each workload README for setup details.

## Layout

| Directory | Purpose |
|---|---|
| `vllm/` | vLLM serving, dynamic KV cache, and MoE expert residency experiments |
| `llama.cpp/` | llama.cpp UVM oversubscription and expert offload baselines |
| `pytorch/` | PyTorch GNN training with native CUDA and UVM allocators |
| `faiss/` | FAISS GPU index build and search under UVM oversubscription |
| `scripts/` | Shared benchmark process and result helpers |
| `_deprecated/` | Historical scripts kept only when they remain useful as baselines |

## Current Focus

The active implementation lives in the vLLM fork and the NVIDIA UVM driver.
The main path combines:

- application-level expert and KV-cache semantics;
- dynamic GPU/CPU residency control;
- reserved GPU slots for expert restore;
- PMM subchunk release and remapping;
- benchmark and correctness probes under `vllm/uvm_test/`.

Start with `vllm/README.md` and the technical reports under `docs/` for the
current architecture. Each benchmark should be run from its own directory so
that local `uv` environments and relative dataset paths resolve correctly.
