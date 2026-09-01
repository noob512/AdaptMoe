# E3 Prefill Graph-copy B1/B8 error and fix log

## FlashAttention cannot run the required Graph mode

- Attempt: `artifacts/e3-prefill-only-graph-copy-flashattn-20260828-attempt-01`.
- Symptom: during model warmup vLLM reported that `FULL_SEPARATE` is unsupported by FlashAttention and selected `FULL_DECODE_ONLY`; static-alias validation then rejected `prefill_graph_remote_decode`.
- Root cause: the attention backend advertises `AttentionCGSupport.UNIFORM_BATCH`, while this mechanism requires separate Prefill and Decode full graphs so that topology can change at the phase boundary.
- Safety behavior: the experiment stopped before measurement; it did not silently run eager or record a false Graph result.
- Minimal correction: preserve requests, B1/B8, KV, exact expert coordinates, and capacity plans, but use the previously certified `TRITON_ATTN + FULL_SEPARATE` configuration.
- Verification: all 10 corrected cells replayed the full Graph once with zero request-time recaptures; all 45 requests completed.
- Impact: the corrected numbers cannot be presented as a strict direct speedup over the FLASH-attention native copy/UVA run.

To make the attempted configuration reproducible, `run_e3_prefill_only_four_mechanism.sh` now accepts `ATTENTION_BACKEND`; its default remains `TRITON_ATTN`, so existing reproduction commands are unchanged.
