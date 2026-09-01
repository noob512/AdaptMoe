# E3 Decode original-VA fault: error and fix record

Date: 2026-08-28

## Symptom

Long-generation/B8 failed before Decode in the original-VA remote path with:

```text
Xid 31
FAULT_PDE ACCESS_TYPE_VIRT_READ
```

The faulting address belonged to a nominally hot expert that should have been
GPU resident. Increasing the reserved hole and increasing cold experts did not
remove the fault.

## TLB hypothesis and implementation

An opt-in full-GPU TLB completion boundary was added:

```text
uvm_phase_f_remote_read_full_tlb_invalidate=1
VLLM_UVM_POST_RECONCILE_FULL_TLB_INVALIDATE=1
```

After an audited remote-read remap succeeds, the kernel now submits
`tlb_invalidate_all` with `UVM_MEMBAR_SYS` and waits synchronously using
`uvm_push_end_and_wait()`. The user-space reconciliation invokes the audited
remote-read action once more after hot prefetch so that the invalidate occurs
after the final PTE/PDE placement.

Two flush-only reproductions still failed with return code 134:

- `artifacts/e3-decode-eager-remote-long-b8-full-tlb-flush-20260828-attempt-01`
- `artifacts/e3-decode-eager-remote-long-b8-full-tlb-flush-20260828-attempt-02`

This falsified “stale GPU TLB entry is the sole root cause.” `empty_cache()`
would also have been insufficient because it does not establish a GPU MMU
translation-completion boundary.

## Root cause

The original sequence reconciled cold-remote/hot-GPU placement after KV
allocation but before all startup work. Kernel autotuning, CUDA Graph capture,
sampler warmup and allocator-cache population ran afterwards. Those later
allocations could evict a hot managed expert page. In `remote_only` mode there
was no final reconciliation between the last startup forward and the first
formal request, so the request could reach a missing hot-expert PDE/PTE.

## Minimal fix

For `remote_only`, after every startup warmup has completed:

1. repeat the audited cold remote mapping;
2. set the hot complement to GPU preferred location and prefetch all hot
   ranges;
3. preserve the now-populated allocator cache so the fix does not alter the
   established allocation topology;
4. synchronize and issue the opt-in full-GPU TLB invalidate;
5. fail closed if residency or invalidate evidence is incomplete.

The temporary per-layer device-read diagnostics used during localization were
removed after the faulting range was identified.

## Validation

The original Long 512->512/B8 high-risk case passed after the final
post-warmup reconciliation:

| Check | Result |
| --- | --- |
| Return code/status | 0 / `ok` |
| Request cohort | eight requests admitted together |
| Output | 8 x 512 tokens complete |
| Eager Decode median | 56.20 ms/step |
| Fault scan | no Xid, illegal access, PTE or PDE fault |
| Final reconcile free bytes | 5,807,800,320 |
| Cleanup | GPU 0 MiB, stock UVM restored |

Evidence:
`artifacts/e3-decode-eager-remote-long-b8-post-warmup-reconcile-tlb-20260828-attempt-01`.

The subsequent CUDA Graph Long/B8 case also passed with one Decode Graph
capture, 511 replays, complete output and median 30.49 ms/step in the runner's
native steady window.

## Code traceability

- `kernel-module/nvidia-module/kernel-open/nvidia-uvm/uvm.c`: module option,
  audited-action hook and synchronous all-TLB invalidate.
- `workloads/vllm/vllm/vllm/model_executor/models/utils.py`: post-reconcile
  invalidate trigger and fail-closed validation.
- `workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py`: final `remote_only`
  reconciliation after all startup warmup.
- `experiments/eurosys_moe_offload_evaluation/run_formal_e3.sh`: passes the
  opt-in user-space flag.
- `experiments/vllm_uvm_prefill_alias_decode_remote_graph/run_experiment.sh`:
  loads and records the module option.

Static shell checks and Python bytecode compilation passed. Every formal
custom launch also ran the eight-test static preflight successfully.
