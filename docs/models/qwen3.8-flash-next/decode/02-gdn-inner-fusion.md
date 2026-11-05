# Track 2: fuse the GDN inner block for single-token decode

**Est. gain:** +1.5-2.5% AR decode.
**Effort:** medium. **Risk:** low — no arithmetic change if the fused kernel
reproduces the existing lane mappings.

## Goal

Collapse the six-kernel Gated DeltaNet inner block into fewer, larger launches
on the width-1 decode path.

## Current behaviour

`GatedDeltaNet`
([`kernels.hip.cpp`](../../../../src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp),
search `bool GatedDeltaNet(`) takes two routes. Row-split prefill
(`row_split && d == kGdnDim && state_snapshots.rows[0] == nullptr`) runs a
single fused `GdnPrefillKernel`. Everything else — including all decode — runs
the unfused chain:

| # | Kernel | Grid at `n_tokens == 1` | Work |
| --- | --- | --- | --- |
| 1 | `SsmConv4Kernel<true,false>` | `Blocks(10240)` | causal conv over 10240 channels |
| 2 | `HistoryShiftKernel` | `Blocks(3 * 10240)` | 3 history rows -> scratch (123 KB) |
| 3 | `CopyKernel` | `Blocks(3 * 10240)` | scratch -> history (123 KB) |
| 4 | `GdnPrepKernel<false>` | **2 blocks** | L2 norm of q and k, 16 key heads |
| 5 | `GdnKernel<false>` | `(48, 128/32)` = 192 blocks | the recurrence |
| 6 | `GdnEpilogueKernel<false>` | **6 blocks** | gated norm, 48 value heads |

Kernels 2 and 3 exist only because the shift cannot be done in place with the
current thread mapping. Kernels 4 and 6 are near-empty: on a 40-CU part,
2 blocks and 6 blocks use 2 and 6 CUs respectively and are pure launch
latency.

The same `HistoryShiftKernel` + `CopyKernel` pair also appears in the PLE
convolution path (`PleConvKernel` -> `HistoryShiftKernel` -> `CopyKernel`), so
any fix here applies twice.

Note the comment already in the code: "Device copies stay kernels: a copy
engine transfer is not reliably ordered behind the kernels on this stream."
Replacing the pair with `hipMemcpyAsync` is not an option; it has to be a
kernel.

## Change

### 2a. In-place history shift (removes one launch and 123 KB per layer)

`HistoryShiftKernel` maps `idx = blockIdx.x * blockDim.x + threadIdx.x`,
`j = idx / channels`, `c = idx % channels`, so a 256-thread block covers 256
consecutive channels of a single `j`. That mapping cannot write in place
because the thread for `j = 0` reads row 1 while the thread for `j = 1` writes
row 1.

Restructure to one block per 256-channel chunk, where each thread owns one
channel and all `hist` rows of it:

1. load `history[j][c]` for `j = 0 .. hist-1` and the new row into registers,
2. `__syncthreads()`,
3. write the shifted rows back.

Every thread then touches only its own channel, so a block-local barrier is
sufficient and the scratch buffer disappears. Saves one launch and 123 KB of
write+read per layer, and the scratch buffer can be reclaimed.

### 2b. Fold the shift into the convolution kernel

`SsmConv4Kernel` already reads the history and produces the convolved output,
so it is the natural owner. The catch is its mapping is per output element, not
per channel, so the same restructure as 2a is needed inside it. If 2a lands
cleanly and 2b turns out to need a mapping change that costs more than it
saves, keep 2a only.

### 2c. Fuse prep and epilogue into the recurrence

`GdnPrefillKernel` already does exactly this for row-split prefill: one block
per value head, 8 waves, q/k norms computed when the window commits, epilogue
run from LDS. Extending it to `n_tokens == 1` removes kernels 4 and 6 and the
`qn`/`kn`/`raw` round trips.

**This is the part to measure carefully.** `GdnKernel` runs 192 blocks of 128
threads (24,576 threads); a per-head fused kernel runs 48 blocks of 256 threads
(12,288 threads). The repository already records that this shape is sensitive
to grid size: "With 48 head blocks on 20 WGPs, 8 WGPs run 3 blocks (grid 40 ->
48 costs 1.28x)". The fused kernel may therefore be slower at the recurrence
itself even while removing two launches. At width 1 the `raw` and `qn`/`kn`
buffers are only ~24 KB each, so the traffic saved is negligible — the gain is
entirely launch latency.

If 2c regresses, keep 2a and stop there.

## Why the gain is modest

At `n_tokens == 1` the buffers the fusion would avoid materialising are tiny
(24 KB of `raw`, 8 KB of `qn`/`kn`), unlike prefill where the fused kernel
saves real traffic. So this track is worth roughly two launch latencies plus
one 123 KB round trip per layer:

```
36 linear layers x (2 launches x ~5 us + 1 x ~5 us) ~= 0.5 ms  ~= 1.5%
```

The upper end of the estimate requires the fused recurrence to also be faster
than the unfused one, which is not established.

## Validation

Correctness:

- `gdn_ops` operator test, including a 99-row prefill split into a prefix and
  a 1-40-row tail (the existing test already covers the width-1 tail).
- Every rollback prefix, since `GdnKernel` writes state snapshots inline and
  the fused kernel must reproduce them.
- Fixed-history logit dumps at widths 1-8.
- Greedy output hashes at d0/4096/32768.

The convolution history is the risky part: a wrong shift is silent for one
token and corrupts the next. Test a sequence of at least `kernel` tokens and
compare against the unfused path, and test the PLE conv path too since it uses
the same pair.

Timing:

- `qwen38_flash_next_gdn_ops_test` and the `gdn` stage in `prof.py` before and
  after. The stage total is the number that matters, not the individual
  kernels.
- Confirm the fused recurrence's grid does not lose more than it gains: compare
  the `GdnKernel`-equivalent portion in isolation with
  `tools/bench/build.sh` if the profile is ambiguous.

## Decision criteria

Keep 2a unconditionally if it is byte-identical — it strictly removes work.
Keep 2b and 2c only if the `gdn` stage total drops in the profile and the
unprofiled `gufo bench` tg at d0 improves.
