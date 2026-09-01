# E3 Prefill-only custom Graph-copy B1/B8

Date: 2026-08-28
Valid artifact: `artifacts/e3-prefill-only-graph-copy-triton-20260828-attempt-02`

## Result

The single-round matrix completed successfully: 10/10 cells and 45/45 requests passed. Every B8 cell admitted all eight requests before the first EngineCore step. Each cell executed exactly one complete Prefill step, produced one sampled token per request, and executed no Decode step. The measured step replayed the pre-captured full CUDA Graph exactly once and caused no recapture.

| Workload | B | Prefill tokens | cold/layer | KV MiB | Prefill ms | tokens/s | Peak HBM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Short 128 | 1 | 128 | 10 | 160.0 | 66.91 | 1,913 | 30,541 |
| Short 128 | 8 | 1,024 | 10 | 160.0 | 86.23 | 11,875 | 30,549 |
| Ordinary 512 | 1 | 512 | 10 | 160.0 | 73.28 | 6,987 | 30,549 |
| Ordinary 512 | 8 | 4,096 | 10 | 492.0 | 148.91 | 27,507 | 30,835 |
| Medium 2048 | 1 | 2,048 | 10 | 205.5 | 111.61 | 18,349 | 30,549 |
| Medium 2048 | 8 | 16,384 | 38 | 1,644.0 | 648.94 | 25,247 | 30,617 |
| RAG 8192 | 1 | 8,192 | 38 | 781.5 | 584.44 | 14,017 | 30,587 |
| RAG 8192 | 8 | 65,536 | 63 | 6,252.0 | 3,708.29 | 17,673 | 30,829 |
| Long 512 | 1 | 512 | 10 | 160.0 | 73.77 | 6,941 | 30,549 |
| Long 512 | 8 | 4,096 | 10 | 780.0 | 149.41 | 27,414 | 30,953 |

The timing boundary is synchronized `llm_engine.step()` around a complete Prompt Prefill plus first-token sampling. It excludes model load, Graph capture, warmup, rebalance, and cleanup.

All ten values reproduce the earlier TRITON Graph-copy qualification within 0.87%; this is a new cold-process pass, not a reuse of the previous result.

## Fixed controls

- Qwen3-30B-A3B-FP8 snapshot `d206ba732169f29bb77fbf80fc2c4b81d4d30782`;
- frozen evaluation token IDs and the original request context/KV envelope;
- direct in-process EngineCore, no HTTP/client, chunking, Decode, or mixed phase;
- custom exact expert-slice coordinates;
- `TRITON_ATTN`, `FULL_SEPARATE`, capture sizes `{B, prompt_tokens*B}`;
- cold10 for normal cells, cold38 for Medium/B8 and RAG/B1, cold63 for RAG/B8.

## Failed preliminary attempt

`e3-prefill-only-graph-copy-flashattn-20260828-attempt-01` is excluded. FlashAttention declared only `UNIFORM_BATCH` Graph support and changed the requested `FULL_SEPARATE` mode to `FULL_DECODE_ONLY`. The custom Prefill Graph mechanism requires `FULL_SEPARATE`, so its runtime contract rejected initialization before a request was measured. This was not an OOM and generated no performance result.

The valid Graph-copy data therefore use TRITON attention. They are not a strict apples-to-apples comparison with the immediately preceding native copy/UVA table, which used FLASH attention and whole-Parameter offload. A direct mechanism speedup claim would require rerunning the native arms with TRITON under the same capacity geometry or adding certified FULL_SEPARATE FlashAttention support.

## Validation

No OOM, CUDA illegal access, Xid, PTE/PDE fault, traceback, cleanup error, or surviving driver session was found. The final GPU allocation is 0 MiB and stock UVM is restored.
