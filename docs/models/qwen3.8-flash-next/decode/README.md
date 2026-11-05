# Flash-Next decode optimization tracks

Working notes for raising single-user decode throughput on Strix Halo
`gfx1151`. Each track is a self-contained proposal with its own measurement,
risk and qualification section. Nothing here is implemented; this directory is
planning material, not a retained result. Retained outcomes still belong in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) and speed in
[`../BENCHMARKS.md`](../BENCHMARKS.md).

## Baseline

Pinned `main` `1c75887b`, Nix production build, ROCm 10.1, UD-Q4_K_XL target
with the shared Q8_0 MTP sidecar, `gufo bench -p 2048 -n 128 -d 0,4096 -r 2`
(the PR #877 control rows in [`../BENCHMARKS.md`](../BENCHMARKS.md)):

| Mode | Depth | pp (tok/s) | tg (tok/s) | ms/token |
| --- | ---: | ---: | ---: | ---: |
| AR | 0 | 1,932.39 | 30.05 | 33.3 |
| AR | 4,096 | 1,872.58 | 29.38 | 34.0 |
| MTP | 0 | 1,870.13 | 62.70 | 15.95 |
| MTP | 4,096 | 1,787.34 | 51.44 | 19.44 |

## The shape of the problem

Flash-Next decode is not one bandwidth-bound sweep. It is a chain of small
kernels. The layer loop in
[`executor.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/executor.cpp)
(`ForwardBody` → `LinearAttention`/`Attention` → `Combine` → `HcMix` → `Moe`)
issues roughly **20 launches per layer, about 940 per token**. At 33.3 ms per
token the mean kernel occupies **~35 µs**.

That is the opposite of Qwen27B, whose decode is 64 layers of ~10 kernels at
~126 µs each and which already runs at the machine's measured streaming
ceiling (~217 GB/s). Short kernels cannot reach steady-state DRAM bandwidth,
and each graph node costs a measurable idle gap: the comment above
`HcMixQ8FusedKernel`
([`kernels.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp),
search `A decode batch's mixer in one launch`) records **~1.9 µs of idle gap
per replayed graph node**. Across ~940 nodes that is ~1.8 ms, or 5.4% of the
token, before any kernel does useful work.

Two invariants drive most of the tracks below:

1. **Every overlap is gated to prefill.** `MatrixRows`, `ExpertMatrixRows` and
   `WidePrefill` all test `prefill_phase` or `rows > 8`/`rows > 32`. So on the
   decode path the shared expert runs *serially* before the routed experts and
   the GDN Z rows run *serially* before the recurrence, even though prefill
   deliberately overlaps both on `side_stream_`.
2. **Four of the six GDN inner kernels are near-empty at width 1.**
   `GdnPrepKernel` launches 2 blocks and `GdnEpilogueKernel` 6 blocks, and
   `HistoryShiftKernel` + `CopyKernel` move 123 KB each just to roll a
   three-row convolution history.

## Measured components

Only two decode components have isolated measurements recorded in this
repository; the rest of the split is derived and must be confirmed by profile.

| Component | Per token | Share | Evidence |
| --- | ---: | ---: | --- |
| HC mixers (`HcMixQ8Fused`, 2 per layer) | 4.27 ms | 12.8% | 44.5 µs/mixer, [`../EXPERIMENTS.md`](../EXPERIMENTS.md) "One-launch decode HC mixer" |
| Router GEMV (F32, 512x2560) | 0.85 ms | 2.6% | 17.7 µs, [`../EXPERIMENTS.md`](../EXPERIMENTS.md) "Vector loads for F16/BF16 small projections" |
| GDN alpha/beta GEMV | 0.39 ms | 1.2% | 8.1 µs, same row |
| Graph-node idle gaps | ~1.8 ms | 5.4% | 1.9 µs x ~940 nodes, `kernels.hip.cpp` |
| Remainder (experts, GDN projections, attention, head, combines) | ~26 ms | 78% | derived, needs profile |

## The MTP cycle cost curve

[`mtp_costs.hpp`](../../../../src/models/qwen38_flash_next/mtp_costs.hpp)
carries the calibrated C1 cycle costs, which fit a clean line at depth 0:

```
T(w) = 38.5 + 8.33 * (w - 1)   ms,  w = 1..8 drafts
```

That split is the most useful single fact for planning:

- **Fixed part ~38.5 ms** — the ordinary token (33.3 ms) plus MTP catch-up.
  All of tracks 1-5 attack this part, and it is shared by every verification
  row, so a 10% cut here is a 10% cut at every width.
- **Marginal part ~8.33 ms per verified row** — the extra routed-expert
  weight reads that a wider verification adds (about 1.5 GB at ~177 GB/s).
  This is what track 6 targets.

Because the marginal row costs a quarter of a token, adding a draft row only
pays if it raises acceptance by more than ~25% of a token. The adaptive
controller in [`mtp_policy.hpp`](../../../../src/models/qwen38_flash_next/mtp_policy.hpp)
is therefore already near the optimum width; the lever is moving the curve,
not the width.

## Tracks

| # | Track | Est. gain | Effort | Risk | Status |
| --- | --- | ---: | --- | --- | --- |
| 1 | [Side-stream overlap on decode](01-side-stream-overlap.md) | **+1.2-1.3%** (measured) | medium | medium (graph capture) | **1a implemented and measured** |
| 2 | [Fuse the GDN inner block](02-gdn-inner-fusion.md) | +1.5-2.5% | medium | low | not started |
| 3 | [Fuse the small and same-input projections](03-gdn-input-projection-fusion.md) | +2-2.5% | low | low | 3b already done at load; 3a not started |
| 4 | [HC mixer bandwidth](04-hc-mixer-bandwidth.md) | +2-3% | high | medium | not started |
| 5 | [Fuse the router chain and shared expert](05-moe-router-fusion.md) | +1.5-2% | low | low | not started |
| 6 | [MTP cycle cost](06-mtp-cycle-cost.md) | +5-10% | medium | medium | not started |

Gains are not additive; tracks 1-3 and 5 remove the same idle gaps from
different places, so expect the combined result below the sum.

Track 6 is not a separate code change. It is the accounting that says where
the MTP gains come from: tracks 1-5 cut the fixed part of every cycle, and a
routed-expert bandwidth fix cuts the marginal row. Read it before deciding
what to do with the MTP numbers.

Ordering: 1a is measured and retained (+1.2-1.3%). Then take 3a and 5
(cheapest, lowest risk), then 2, then 6, and treat 4 as a research item.

The 1a result recalibrates the plan: the shared expert overlapped cleanly but
bought only 1.2%, so the decode MoE block is closer to bandwidth-bound than
latency-bound. Expect [track 5](05-moe-router-fusion.md)'s launch-latency
estimates to come in low and [track 4](04-hc-mixer-bandwidth.md)'s alignment
work to matter more.

1a also carries an always-on assertion that the shared-expert side stream
joined the decode graph capture, because a capture that silently dropped those
nodes would produce wrong logits instead of an error. That assertion is the
first thing to watch on the first decode request.

Two load-time facts that shrink the plan, both found while implementing 1a:
`DeviceModel` already stacks same-input Q8_0 projections, so `ssm_qkv` +
`ssm_gate` and `attn_q` + `attn_k` + `attn_v` are each one launch already
(that removes most of track 3 and all of track 1b), and the shared expert's
decode kernels allocate nothing from the per-stream scratch pool, which is why
1a is safe to overlap.

## Measure before building

The component split above is partly derived. Confirm it before writing code.

```sh
# AR decode profile. --gaps reports the idle gaps that tracks 1-5 target.
nix develop -c tools/prof/prof.py run --stages qwen-flash -o /tmp/fn-ar-d0 -- \
  result/bin/gufo bench -m $TARGET --speculative none \
  -p 2048 -n 128 -d 0 -r 2

nix develop -c tools/prof/prof.py show /tmp/fn-ar-d0/*.db \
  --stages qwen-flash --gaps 8 --top 40

# MTP cycle profile at two depths, to size track 6.
nix develop -c tools/prof/prof.py run --stages qwen-flash -o /tmp/fn-mtp -- \
  result/bin/gufo bench -m $TARGET --speculative mtp --mtp-model $MTP \
  -p 2048 -n 128 -d 0,4096 -r 2
```

What to look for:

- GPU-busy fraction and the largest idle gaps. If `--gaps` shows hundreds of
  gaps of a few microseconds each, tracks 1-5 are confirmed as the right
  target. A C4 mixed MTP trace was already 84.9% GPU-busy, so there is real
  headroom in the single-user case too.
- Whether `HcMixQ8Fused` really lands near 44.5 µs at width 1.
- The block counts of `GdnPrepKernel` and `GdnEpilogueKernel` (expect 2 and 6).

Ground truth for the cost curve:

```sh
nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_gpu_probe
nix develop -c ./build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model $TARGET --mtp-model $MTP --cost-audit 0
```

## Quality gates

Tracks 1-5 change dispatch, stream assignment and launch grouping only. They
do not change arithmetic, so the bar is byte-identical output, not a tolerance:

- `qwen38_flash_next` operator tests, in particular `hc_mix_ops`,
  `hc_route_ops`, `routed_wmma_ops`, `projection_ops`, `gdn_ops`,
  `moe_epilogue_ops`.
- Fixed-history logit dumps at verification widths 1-8
  (`gpu_probe --logit-rows`), compared against the F32 reference fixture in
  `tests/fixtures/qwen38_flash_next/f32_reference/`.
- Ragged row checks at 1/7/9/33/95/96 rows, including rows that straddle an
  allocation boundary.
- Greedy output hashes and draft/acceptance counts at d0/4096/32768 for
  C1-C8; sampled requests additionally need seeded replay.
- `tools/ci/check-format.py` and `check-kernel-resources.py --compare` for any
  kernel that is modified: no new spills, no lower occupancy.

Track 6 touches the controller, so it additionally needs the MTP
qualification described in [`../QUALITY.md`](../QUALITY.md): greedy output,
sampled p/q rejection and residual correction, seeded replay, EOS, multi-turn
continuation and snapshot restore.

## Reporting

One row per experiment in [`../EXPERIMENTS.md`](../EXPERIMENTS.md), with the
rejected variants and their numbers, in the style already used there. Refresh
[`../BENCHMARKS.md`](../BENCHMARKS.md) only when a gain survives unprofiled
HTTP serving, not just `gufo bench`.
