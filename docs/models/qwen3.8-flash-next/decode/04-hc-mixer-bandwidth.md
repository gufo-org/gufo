# Track 4: HC mixer bandwidth

**Est. gain:** +2-3% AR decode.
**Effort:** high. **Risk:** medium — the fused kernel's phase/counter design
constrains any change to the grid.

## Goal

Raise the hyper-connection mixer from ~156 GB/s toward the ~220 GB/s the
machine sustains on streaming weight reads.

## Current behaviour

`HcMixQ8FusedKernel`
([`kernels.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp),
search `HcMixQ8FusedKernel`) replaces five kernels with one launch whose
phases are work lists taken from atomic counters. It runs twice per layer
(`hc_attn`, `hc_ffn`), so 96 launches per token.

Geometry (`kFused*` constants just above the kernel):

| Constant | Value |
| --- | ---: |
| `kFusedHidden` | 2560 |
| `kFusedStreams` | 4 |
| `kFusedRank` | 320 |
| `kFusedDim` | 10240 |
| `kFusedDimBlocks` | 320 |
| `kFusedRankBlocks` | 10 |
| `kFusedWaves` | 8 |
| `kFusedDownTiles` | 40 |
| `kFusedUnits` | 80 |
| grid | `kFusedDownTiles + kFusedUnits` = **120 blocks** |
| block | 256 threads |

Bytes per mixer:

- down `10240 -> 320`, Q8_0: 320 rows x 10,880 B = **3.48 MB**
- up `320 -> 10240`, Q8_0: 10,240 rows x 340 B = **3.48 MB**
- combine and epilogue activations: tens of KB

Measured 44.5 µs for ~6.96 MB of weights = **156 GB/s**
([`../EXPERIMENTS.md`](../EXPERIMENTS.md) "One-launch decode HC mixer"). Two
per layer x 48 layers = 4.27 ms, **12.8% of the token**, the largest single
measured component.

## Two concrete inefficiencies

### 4a. The down phase uses a third of the machine

The grid is 120 blocks of 256 threads = 30,720 threads, against 81,920 thread
slots on 40 CUs. But the phases do not distribute evenly:

- **combine:** 4 items (`kRows = kTokens * kFusedStreams`)
- **down:** **40 items** (`kFusedDownTiles`), each covering 8 rows
- **up:** 80 items (`kFusedUnits`), each covering 128 rows

During the down phase only 40 blocks hold an item. The other 80 have already
moved on to take an up item and are parked in `FusedWait(&st->done[1], ...)`.
So the down phase — half the mixer's weight traffic — runs with **40 blocks x
256 threads = 10,240 threads, 12.5% of the thread slots**.

The up phase is better (80 of 120 blocks) but still leaves a third idle.

### 4b. Q8_0 blocks are 34 bytes and never 16-byte aligned

A Q8_0 block is `{half d; int8 qs[32]}` = 34 bytes. Every block load straddles
a sector boundary, and the per-lane access inside the down loop is 8 bytes of
codes plus a 2-byte scale.

This repository already exploited the aligned case elsewhere: "Coalesced routed
Q5_1 down weights ... a row's two Q5_1 blocks per stage are 48 contiguous,
16-byte aligned bytes, so the block loads them as whole 16-byte chunks"
([`../EXPERIMENTS.md`](../EXPERIMENTS.md)). Q8_0 cannot do that in place.

## Change

### 4a-fix: give the down phase more items

The constraint is that the grid must stay within residency, because a block
parked in `FusedWait` holds a unit item and a not-yet-scheduled block would
never run. 120 x 256 = 30,720 threads is comfortably inside 81,920, so there
is headroom to grow the grid — but not to 320 blocks x 256 = 81,920, which is
exactly the limit and leaves no slack for occupancy limits.

Options, in order of increasing risk:

1. **Split each down row across 2 blocks** and combine the two half-K partial
   sums through LDS in a fixed order. This doubles `kFusedDownTiles` to 80
   without changing the grid. The reduction order must be preserved exactly:
   the current loop accumulates `sum += product` over `first = lane/4` stepping
   `8 * kPrefetch`, so a split must keep each lane's operand sequence and only
   merge two disjoint ranges in a spelled-out order.
2. **Reduce `kFusedWaves` per row** so more rows are in flight: currently one
   wave per row and 8 rows per block. Four waves per row would give 2 rows per
   block and 160 tiles, but needs a 4-wave partial-sum merge.
3. **Grow the grid** to 160-200 blocks and raise `kFusedDownTiles`
   accordingly, staying well inside residency. Cheapest to try, but check that
   the `FusedWait` spin cannot starve an unclaimed item at the chosen size.

Option 1 or 2 is the real fix; option 3 is the probe that tells you whether
occupancy is the limiter at all. Do option 3 first as a diagnostic.

### 4b-fix: repack the HC Q8_0 weights at load

Split each Q8_0 block into planar arrays:

```
d[rows][blocks]        // 2 bytes per block, rows x 320 or rows x 10
qs[rows][blocks][32]   // 32 bytes per block, 16-byte aligned
```

Same total bytes, same values, same arithmetic — only the layout changes. The
down loop then reads 32 contiguous aligned bytes per block instead of a 34-byte
straddling record, and the scale array becomes a separate coalesced stream.

Cost and risk:

- The HC tensors are `hc_attn.down/up`, `hc_ffn.down/up` and `output_hc.down/up`
  — 6 matrices per layer plus the head, ~334 MB total. If nothing else reads
  them, the repack can **replace** the mapped region, so resident bytes are
  unchanged; only a load-time copy pass is added.
- `HcMixQ8FusedKernel` is the only consumer, so the blast radius is one kernel
  plus the loader.
- Precedent to weigh: "Pinned weights and aligned Q8 head layouts" was
  rejected for the output head because it retained another 644 MiB for 0.8%
  end to end. That rejection was about *retaining* a second copy. Here the
  target is 12.8% of the token and the repack replaces rather than duplicates,
  so the trade is different — but state it explicitly in the experiment row.

## Why not just prefetch deeper

The down loop already runs `kPrefetch = 2` iterations ahead, each iteration
covering `8 * kPrefetch = 16` blocks per lane. Deeper prefetch raises VGPR
pressure in a kernel that already carries a large state struct. The repository
has rejected deeper prefetch in the routed expert kernels for exactly this
reason. Try it as a cheap probe (one-line change, measure), but expect the
occupancy and alignment fixes to matter more.

## Validation

Correctness:

- `hc_mix_ops` test, eager, in place and graph-replayed — the existing test
  already compares against the five separate kernels bit for bit.
- `hc_route_ops`, which carries the mixer's inject partials into the residual.
- Fixed-history logit dumps at widths 1-8.
- Greedy output hashes at d0/4096/32768.
- `check-kernel-resources.py --compare`: the fused kernel must not gain
  scratch or spills and must not lose occupancy.

Timing:

- `qwen38_flash_next_hc_mix_bench` is the right instrument: it reports
  microseconds per mixer at T1/T2/T4/T8 graph-replayed. Target T1 specifically.
- Confirm with the `hyperconnections: mixing` stage in `prof.py`, then
  unprofiled `gufo bench`.

## Decision criteria

Keep if `hc_mix_bench` T1 drops and the unprofiled tg at d0 improves. Because
this is 12.8% of the token, even a 20% mixer gain is worth only ~2.5% overall —
so do not spend more than a bounded amount of time here before trying
[track 1](01-side-stream-overlap.md), which is cheaper.
