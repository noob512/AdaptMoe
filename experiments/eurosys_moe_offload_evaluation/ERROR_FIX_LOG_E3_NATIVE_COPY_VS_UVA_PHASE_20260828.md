# E3 native copy vs UVA error and warning log

## Resolved: large-Prefill attention failure risk

- Symptom from prior qualification: `TRITON_ATTN` at or above an 8192-token Prefill step could trigger TMA descriptor 700 or CUDA illegal-address faults.
- Root cause boundary: attention backend/shape path, independent of the native copy versus UVA mechanism under test.
- Minimal change: use `FLASH_ATTN` for both arms and leave all paired request/offload controls symmetric.
- Verification: Medium/B8, RAG/B1, and RAG/B8 completed for both arms, output hashes matched, and no CUDA fault appeared.
- Interpretation impact: this run is a valid paired `FLASH_ATTN` result but cannot be numerically pooled with earlier `TRITON_ATTN` runs.

## Non-blocking: process-group exit warning

- Symptom: every run logs `destroy_process_group() was not called before program exit`.
- Impact observed: none on result validity; return codes are zero, each subsequent process starts on a clean GPU, and final GPU use is 0 MiB.
- Follow-up: explicitly destroy the process group in the benchmark teardown before using this runner for repeated formal trials.
