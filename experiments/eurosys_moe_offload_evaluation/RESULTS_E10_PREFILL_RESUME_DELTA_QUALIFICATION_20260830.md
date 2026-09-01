# E10 Prefill-resume + large-delta qualification (2026-08-30)

## Scope

This is one complete qualification sequence with no statistical repetition.
The fixed sequence is KV 2 -> 4 -> 6 GiB and expert plan
`cold70 -> cold80 -> cold89`. Growth is triggered only by four blocks of real
backup-queue pressure; prediction is disabled.

Run root:

`experiments/vllm_uvm_e10_multilevel_pressure/artifacts/20260830T040827Z_qualification`

Verifier result: `status=ok`, `completed_sequences=1`,
`statistical_repetitions=0`.

## Timing results

| KV tier | Expert plan | Prefill compute (ms) | Early Prefill resume (ms) | Rebalance excluding resume (ms) | Prefill E2E (ms) | First-Decode phase PARK (ms) | Steady Decode median (ms) | Decode token/s | Request E2E (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 GiB | cold70 | 335.960 | 0.000 | 0.000 | 335.960 | n/a | 17.984 | 55.606 | 3067.570 |
| 4 GiB | cold80 | 390.910 | 332.294 | 715.381 | 1438.585 | 2173.079 | 21.922 | 45.616 | 4146.671 |
| 6 GiB | cold89 | 434.727 | 377.282 | 656.670 | 1468.679 | 2567.963 | 26.248 | 38.098 | 4715.001 |

The maintenance on the Prefill critical path, including the moved-ahead
Prefill resume, is 1047.675 ms for 2 -> 4 GiB and 1033.952 ms for
4 -> 6 GiB.

## New mechanism evidence

### cold70 -> cold80

- Prefill topology was restored from `remote_parked` to `sealed` first:
  332.294 ms.
- The plan then changed from `sealed` to `sealed` with no internal PARK or
  RESUME.
- The 4,500-entry difference used one MAP ioctl; operation wall time was
  545.263 ms and ioctl time was 524.237 ms.
- Instruction cache hit: `cold70->cold80`.
- Expert-plan switch wall time: 613.964 ms.
- Physical KV resize RPC: 89.179 ms.

### cold80 -> cold89

- Prefill topology was restored from `remote_parked` to `sealed` first:
  377.282 ms.
- The plan then changed from `sealed` to `sealed` with no internal PARK or
  RESUME.
- The 4,050-entry difference used one MAP ioctl; operation wall time was
  454.135 ms and ioctl time was 434.775 ms.
- Instruction cache hit: `cold80->cold89`.
- Expert-plan switch wall time: 518.197 ms.
- Physical KV resize RPC: 124.872 ms.

Therefore the intended ordering and the enlarged one-call delta interface
both worked in the real GPU run. The former 36,000/40,050-entry full REBIND
fallback was not used.

## Comparison with the previous attempt-03 qualification

This is a directional single-run comparison, not a statistical result.

| Transition | Previous full-REBIND operation | New delta operation | Previous Prefill E2E | New Prefill E2E |
|---|---:|---:|---:|---:|
| 2 -> 4 GiB | 1029.170 ms, 32 ioctls | 545.263 ms, 1 ioctl | 1934.164 ms | 1438.585 ms (-25.62%) |
| 4 -> 6 GiB | 1680.570 ms, 35 ioctls | 454.135 ms, 1 ioctl | 2815.226 ms | 1468.679 ms (-47.83%) |

Rebalance time excluding the separately-accounted Prefill resume decreased
from 1172.106 to 715.381 ms (-38.97%) and from 1859.043 to 656.670 ms
(-64.68%).

## Correctness and cleanup

- All three tiers produced the same output-token hash:
  `d9510d414d87cc5aaca79755e0e35473b443ec1d4fdbad30fdab7462093de739`.
- KV used one stable 8 GiB VA allocation and had exact 2/4/6 GiB GPU-resident
  prefixes with zero unknown bytes.
- Every measured Decode step replayed its CUDA Graph; capture count remained
  six, so tier switches caused no recapture.
- Kernel Xid, MMU fault, BUG, and general-protection-fault counts were zero.
- Cleanup reported no errors. Stock UVM was restored and final GPU usage was
  0 MiB.

## Qualification conclusion

The implementation objective is met: restoring the Prefill topology before
the tier change makes the 4,500/4,050-entry sealed-state delta path usable,
and each delta is submitted once instead of rebuilding the full directory.
The remaining dominant latency is the delta MAP itself (about 435-524 ms),
plus the independent first-Decode phase PARK and physical audit (about
2.17-2.57 s). Steady Decode becomes slower as more experts are remote, but
this single qualification does not establish a statistical performance
claim.
