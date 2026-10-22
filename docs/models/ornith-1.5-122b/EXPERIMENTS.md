# Ornith 1.5 122B experiments

Kept changes are described in [the model card](README.md). These were tried on
October 21, 2026 and dropped, or found to have a cause other than the one first
suspected.

| Experiment | Result | Decision |
| --- | --- | --- |
| Q6_K down experts on the MMQ vector kernel | Under 1% decode gain | Dropped; the 25 Q6_K down layers keep the per-slot GEMV |
| Wider MMVQ route for more projection shapes | 3 of 6 uncapped MTP conversation runs failed | Dropped |
| MTP with three draft tokens | 29.7 against 30.6 t/s at two; acceptance 49.2% against 56.3% | Kept two |
| Wider serve warm-up (longer prompt, cached twice) | Did not remove the first-request stalls | Dropped; the cause was page migration of the weights |
| `HIP_ENABLE_DEFERRED_LOADING=0` | 6 slow requests in 24 against 4 with the default; loads about 5 s slower | Dropped |
| Automatic RAM continuation cache | Requests slowed as the cache filled (p90 first content 1694 ms against 782 ms at 4 GiB) | Use an explicit `--cache-ram-bytes` |

## Notes

- Kernel traces ranked the per-slot expert GEMV as faster than llama.cpp's,
  yet routing the experts to the MMQ vector kernels gained 6.8% in an
  unprofiled A/B. Per-kernel tracing overhead differs with kernel count, so
  small gaps were settled with end-to-end A/B runs.
- Sending the MTP drafter's stream through the vendored MMQ kernels exposed a
  scratch race. Their scratch pool was per device, and a second stream reused
  it while the first was still reading. Each stream now has its own arena.
