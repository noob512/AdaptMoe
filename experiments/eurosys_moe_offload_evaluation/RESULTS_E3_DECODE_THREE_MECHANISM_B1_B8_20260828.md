# E3 Decode-only B1/B8 three-mechanism qualification

Date: 2026-08-28

## Outcome

- Native vLLM copy-on-forward and pinned-UVA: 20/20 arm-cells passed.
- Custom CUDA Graph remote-read: 8/10 cells passed.
- Custom Medium/B8 and RAG/B8 are N/A because the required single-cohort
  conditioning Prefill faults before Decode begins. They are not Decode
  failures and no latency is reported for them.
- The eight common valid cells have identical output-token hashes across all
  three mechanisms.
- No HTTP transport was used. Every valid B8 case admitted all eight requests
  together, and every measured Decode step contained eight cached requests
  producing one token each. There was no Prefill/Decode mixed step.
- The GPU was 0 MiB after completion and the stock `nvidia_uvm` module was
  restored.

## Frozen setup

| Item | Value |
| --- | --- |
| Model | Qwen3-30B-A3B-FP8 |
| Snapshot | `d206ba732169f29bb77fbf80fc2c4b81d4d30782` |
| GPU | RTX 5090 32 GiB, TP=1 |
| Batches | B1 and B8 only |
| Requests | 128->32, 512->128, 2048->128, 8192->128, 512->512 |
| Transport | Direct in-process EngineCore; no HTTP/client |
| Attention | FLASH_ATTN |
| Custom Decode | original-VA remote-read, `FULL_DECODE_ONLY`, one Graph for B1 or B8 |
| Native Decode | vLLM copy-on-forward or pinned-UVA, eager |
| Decode warmup | First five physical Decode steps excluded |
| Reported latency | Median wall time for one whole-batch, one-token Decode step |

The custom runner stores its first Decode separately and then stores five more
warmup steps. For a like-for-like five-step exclusion, the table recomputes
the custom median from `warmup_decode[-1] + measured_decode`. Native medians
come from `phase_summary.decode_all_step_ms`. Raw runner-native summaries
remain unchanged in their result files.

## Decode performance

All latency values are milliseconds per whole-batch one-token step. “Speedup”
is native latency divided by custom latency.

| Request | B | Custom Graph remote | vLLM copy | vLLM UVA | Custom vs copy | Custom vs UVA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Short 128->32 | 1 | 9.46 | 52.75 | 48.33 | 5.58x | 5.11x |
| Short 128->32 | 8 | 14.53 | 58.34 | 50.04 | 4.02x | 3.44x |
| Ordinary 512->128 | 1 | 9.33 | 52.80 | 49.74 | 5.66x | 5.33x |
| Ordinary 512->128 | 8 | 17.28 | 59.01 | 52.07 | 3.42x | 3.01x |
| Medium 2048->128 | 1 | 13.25 | 52.91 | 49.84 | 3.99x | 3.76x |
| Medium 2048->128 | 8 | N/A | 167.16 | 79.62 | N/A | N/A |
| RAG 8192->128 | 1 | 22.83 | 160.80 | 48.65 | 7.04x | 2.13x |
| RAG 8192->128 | 8 | N/A | 280.77 | 124.91 | N/A | N/A |
| Long 512->512 | 1 | 10.30 | 52.82 | 48.17 | 5.13x | 4.68x |
| Long 512->512 | 8 | 30.42 | 58.66 | 50.79 | 1.93x | 1.67x |

The custom path is fastest in all eight common valid cells. The advantage
narrows at Long/B8 because attention/KV work grows while Graph launch savings
become a smaller fraction of the step.

These values are a system-arm qualification, not a pure remote-access kernel
ranking. Custom uses CUDA Graph plus exact expert slices; native vLLM is eager
and offloads whole Parameter tensors. The capacity configurations also differ:

- custom generally uses cold10/layer, Long/B8 uses cold38/layer, and RAG/B1
  uses cold63/layer;
- native uses 2.25 GiB for ordinary cells, 7.875 GiB for Medium/B8 and RAG/B1,
  and 13.5 GiB for RAG/B8.

Consequently the speedups include Graph/eager, offload-geometry and HBM-layout
effects. A paper-grade mechanism-only comparison still needs equal geometry,
equal HBM and repeated cold processes.

## Correctness and execution checks

- Native execution table: 20 `ok`, return code 0.
- Custom valid results: 8 `ok`, one Graph capture per process, and 31/127/511
  total Graph replays for 32/128/512-token outputs respectively.
- B8 valid custom Prefill schedules contain eight `new_request_ids`; native B8
  schedules do the same. Measured B8 Decode schedules contain eight cached
  IDs and exactly eight scheduled tokens.
- Successful custom and native logs contain no illegal-memory, Xid, PTE/PDE or
  traceback signatures.
- Every valid custom result reports UVM driver `session_count=0` after shutdown.

## Artifacts

- Custom matrix: `artifacts/e3-decode-graph-remote-b1-b8-post-warmup-tlb-20260828-attempt-01`
- Native matrix: `artifacts/e3-decode-native-copy-uva-b1-b8-20260828-attempt-01`
- Original high-risk successful fix validation:
  `artifacts/e3-decode-eager-remote-long-b8-post-warmup-reconcile-tlb-20260828-attempt-01`

This is one qualification round. It provides point estimates only; it does
not provide confidence intervals or final paper rankings.
