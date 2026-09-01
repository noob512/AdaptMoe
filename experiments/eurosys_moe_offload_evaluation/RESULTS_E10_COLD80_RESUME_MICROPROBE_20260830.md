# E10 cold80 34,560-entry RESUME microprobe (2026-08-30)

## Outcome

The one permitted GPU microprobe did not reach the measured RESUME operation.
It therefore produces no new latency sample and must not be used as a
successful performance result.

The cold80 static-alias directory was created and sealed successfully with
36,000 total records (1,440 upload records plus 34,560 consumer records), 4 KiB
PTE mode, and the dual-parent fast path disabled. During the subsequent vLLM
MoE kernel warm-up, before the first serving request, the GPU raised Xid 31 for
a missing read PTE. The faulting address belongs to
`model.layers.2.mlp.experts.w2_weight`, expert 93. Expert 93 is outside the
cold80 set and had been reported GPU-resident by the finalization audit. This
is an initialization/mapping correctness failure, not a RESUME timing result.

No second GPU run was started. The system was restored to the stock UVM module
and GPU memory returned to 0 MiB.

## Existing successful evidence

The preceding complete E10 attempt-05 contains two successful cold80 RESUME
operations in its single sequence:

| occurrence | entries | RESUME ioctl | driver PTE install |
|---|---:|---:|---:|
| request-boundary cold80 resume | 34,560 | 466.543 ms | 453.232 ms |
| cold80 resume before 4→6 GiB switch | 34,560 | 471.785 ms | 459.224 ms |

The two actual PTE-install observations differ by 5.992 ms (1.31% of the
459.224 ms value). They support that the earlier 459.224 ms was a real
measurement, but they are not a substitute for the failed new microprobe.

## Artifacts

- Failed microprobe: `artifacts/e10-cold80-resume-microprobe-20260830-attempt-01-qualification/`
- Runner result: `sequence/result.json` (`status=exception`)
- Kernel log: `kernel.since.log` (Xid 31, PTE read fault)
- Module state: `module.custom.json` and `module.after.json`
- Existing successful result: `artifacts/e10-kv-2-4-6-pressure-20260830-attempt-05-indexed-map-cached-resume-qualification/sequence/result.json`
