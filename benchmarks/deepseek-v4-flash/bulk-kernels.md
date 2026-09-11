# DS4 bulk-kernel follow-up

2026-09-11, gfx1151, baseline `282c3b4`, unchanged 0731 GGUF and cache precision.
All four targets were investigated. **Prefill improves modestly; this round
has no qualified generation gain.** No runtime selectors were added.

## Retained changes

- IQ2 sign expansion uses integer parity and packed negation, removing shared
  sign tables. Dot products and floating-point reduction order stay unchanged.
  Grid initialization also covers inputs wider than the shared activation cache.
- Prefill MoE down pairs 64-row groups into 128-row groups and keeps a 64-row
  odd tail. This reuses decoded weights without adding padding. The route starts
  at 1,024 prompt tokens, uses the existing scratch buffer and adds at most one
  launch per layer. Narrow prompts keep their existing tile sizes.

## Release measurements

C1, two repetitions, pp4096 and tg128, baseline followed by candidate. Throughput
is tok/s. Generation is slightly lower in this sweep (0.06–0.34%); do not infer a
speedup from the isolated IQ2 experiment.

| Depth | Baseline pp4096 | Current pp4096 | Baseline tg128 | Current tg128 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 488.50 | 490.03 | 17.79 | 17.73 |
| 4,096 | 481.09 | 482.87 | 15.91 | 15.90 |
| 8,192 | 470.38 | 471.78 | 15.76 | 15.75 |
| 12,288 | 465.30 | 465.62 | 15.62 | 15.57 |
| 16,384 | 456.69 | 457.08 | 15.48 | 15.44 |

A reversed-order control at 4K, four repetitions per release, confirms the
small prefill benefit: **480.78 ± 0.22 → 483.78 ± 0.39 tok/s (+0.62%)**.
Generation is **15.91 → 15.90 tok/s**. All **1,792 paired token positions**
match, including the control. Both full sweeps retained over 28.6 GiB available
under the 12 GiB memory guard. Depth-zero prefill includes first-use work.

[Raw evidence](bulk-kernels.json) records source/binary/model identities, repeated
hashes, deviations, controls, guards, profiles and experiment decisions.

## Production profiles

Profiling is diagnostic; the release table above runs without profiling.

| Workload / GPU work | Baseline ms | Current ms |
| --- | ---: | ---: |
| C1, 16 generated tokens at 4K: IQ2 gate/up | 123.35 | 122.63 |
| Same decode interval: all kernels | 965.51 | 964.51 |
| pp4096 at depth zero: all MoE down kernels | 1,411.95 | 1,381.48 |
| Same prefill process: all kernels | 8,407.61 | 8,392.08 |

Q8 projection families still take about 51% of decode GPU time. The bulk read
control measured 240.93 GB/s; the large isolated Q8 head already approaches
that ceiling. Prefill quantized matmuls remain about 36% of GPU time. Shared
memory and unpacking changes need production measurements: isolated gains
substantially overestimated the IQ2 benefit here.

## Quality and maintained checks

Projection and attention checks pass. Target qualification preserves the pinned
trajectory (116/128 top-1, rank sum 142, worst rank 3), the wide-prefill
fingerprint and all 29 concurrent-session choices. The full maintained DSpark
suite passes, including **736 exact scalar replay choices and frontier logits
through C8/16K**, changing membership, short budgets and snapshot continuations.
It retained at least **15.37 GiB available** under the 12 GiB memory guard.

The new coverage joins `ds4.projections`, under
`tests/models/deepseek_v4_flash/`; no new CTest suite or permanent tool is added.
It exhaustively checks all 32,768 IQ2 grid/sign combinations against independent
byte arithmetic, verifies the nonzero-grid assumption, and exercises uncached
32-block inputs. MoE checks cover empty buckets and boundaries around 64/128
rows, prove each output belongs to exactly one tile, compare every F16 output
bit against the previous kernel, and check independent CPU dot-product samples.

This is regression qualification against the retained quantized implementation,
not proof of parity with the official unquantized model. Prior serving/CLI and
external-reference qualification remains separately pinned. Concurrent/DSpark
speed numbers were not rerun. The 75-question capability comparison still
requires separate user approval.

## Experiment decisions

| Target | Result |
| --- | --- |
| Q8 generation | Full-block specialization, packed loads, dot-chain scheduling and row grouping were exact. Rotating at least 128 MiB of weights removed most apparent gains; no new route retained. |
| Prefill MMQ | Y32/Y128 were exact across five Q8/IQ2 shapes. Both slowed four shapes; Y32 helped only the small IQ2 N64 case. Keep Y64. |
| Prefill MoE down | Blanket 128-row tiles lost on tails. Mixed 128/64-row tiles were exact and improved production down time about 2.2%; retain for large prompts. |
| IQ2 generation | Arithmetic signs reduced isolated kernel time about 4–6% at model width, but only 0.6% in the production profile. Keep the simpler implementation; no generation-throughput claim. Extra shared masks, split dot chains and cooperative activation staging were not retained. |

```sh
nix develop -c tools/ds4/check.py kernels
# Run sequentially for each immutable baseline/candidate Nix release:
./result/bin/gufo bench --model "$MODEL" -c 1 -p 4096 -n 128 \
  -d 0,4096,8192,12288,16384 -r 2 -v
```
