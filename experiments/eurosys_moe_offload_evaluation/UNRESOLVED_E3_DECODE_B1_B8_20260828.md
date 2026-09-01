# Unresolved E3 Decode B1/B8 items

Date: 2026-08-28

Only unresolved items are listed here.

## Custom Medium/B8

Configuration:

- eight requests admitted as one batch;
- 2048 prompt tokens per request, 16,384-token single-step conditioning
  Prefill;
- 128 output tokens per request;
- 1,723,858,944-byte KV cache;
- `FULL_DECODE_ONLY` B8 Graph;
- FLASH_ATTN and original-VA remote-read.

Both cold38 and cold63 failed. cold63 left roughly 8.2 GiB free after initial
reconciliation, so simple HBM exhaustion is not supported. Reducing the
FlashInfer warmup cap from 4096 to 2048 and the MoE internal chunk from 4096
to 1024 allowed startup to complete, but the first formal 16,384-token Prefill
still raised Xid 31 `FAULT_PDE`. Decode was never reached.

Evidence:

- `artifacts/e3-decode-graph-remote-b1-b8-post-warmup-tlb-20260828-attempt-01/groups/cold38_medium_b8`
- `.../groups/cold63_medium_b8_retry1`
- `.../groups/cold63_medium_b8_chunk1024_retry2`

Status: N/A for custom Decode performance. Native copy and UVA both succeed,
so this is isolated to the custom large-conditioning original-VA path.

## Custom RAG/B8

Configuration:

- eight requests admitted as one batch;
- 8192 prompt tokens per request, 65,536-token single-step conditioning
  Prefill;
- 128 output tokens per request;
- 6,555,697,152-byte KV cache;
- cold63/layer, MoE chunk 1024, FlashInfer warmup cap 2048;
- `FULL_DECODE_ONLY` B8 Graph.

The process failed before obtaining a Decode sample. More expert offload does
not reduce the 65,536-token single-step kernel/addressing requirement, and
splitting the eight-request cohort would violate the experiment contract.

Evidence:
`artifacts/e3-decode-graph-remote-b1-b8-post-warmup-tlb-20260828-attempt-01/groups/cold63_rag_b8`.

Status: N/A for custom Decode performance. Native copy and UVA both succeed.

## Recommended follow-up

Implement and validate a synchronous, equal-progress chunked Prefill mode that
keeps all eight requests in the same cohort and proves every Prefill step is
free of Decode work. Only then should Medium/B8 and RAG/B8 be retried. A
stock scheduler mode that admits only a subset of the eight requests at a time
must not be substituted, because that would change the requested B8 contract.
