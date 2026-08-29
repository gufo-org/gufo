# HRX optimization ideas

Refreshed 2026-08-29 after a full panoramic over the kernels
(`tools/loom/*.loom`), the executor (`src/models/qwen/hrx/`) and the vendored
Loom compiler (`hrx-system/`, local clone now checked out at the same revision
the Nix derivation pins). Every card states the evidence it rests on, so a
later reader can tell a measured claim from a hypothesis. Everything that was
closed out is deleted rather than archived; the retained and rejected results
live in `benchmarks/qwen3.8-27b/README.md` under
"## HRX native backend (Loom) experiments".

## Where the time is

Measured with the executor's stage traces (`GUFO_HRX_TRACE_STAGES`,
`GUFO_HRX_TRACE_SSM`, `GUFO_HRX_TRACE_FFN`, `GUFO_HRX_TRACE_ATTENTION`), 2048
tokens, all retained fusions on. Total chunk 4564 ms; attention 532, SSM 1340,
FFN 2692.

| Work | ms of 4564 | Share |
| :--- | ---: | ---: |
| Blocked W8A8 projection, all shapes | ~3500 | 77% |
| DeltaNet recurrence | 322 | 7.1% |
| Attention kernel | 208 | 4.6% |
| SwiGLU + quantize (fused) | 95 | 2.1% |
| RMSNorm + quantize (fused) | 52 | 1.1% |
| SSM prepare + conv | 80 | 1.8% |
| Residual adds | 75 | 1.6% |
| Everything else | ~230 | 5.0% |

`pp2048` is 450 t/s against HIP's 557 when HIP's page cache is warm, so 81%. `tg16` is 7.90 against 7.78, so decode is already ahead and is not a
target. **Prefill parity requires the projection to improve; nothing else is
large enough.**

## 1. ~~VOPD `fmac` in the projection~~ SOLVED AND REJECTED

**Done, and the premise was wrong.** The compiler change works and is recorded
in `benchmarks/qwen3.8-27b/README.md` under "Solved and rejected: VOPD `fmac`
in the projection". Two pieces: the missing `_f32_fmac_rule`, and a *forward*
destination-reservation walk in the tie-coalescer (the backward walk can never
succeed -- the carried value it reaches is live by construction; the forward
walk works because the tied result is destined for those exact registers).

All 57 kernels compile, the projection emits 64 `v_dual_fmac_f32` against 64
`v_fma_f32`, and FMA issue slots in the loop drop 64 to 33. It is **4.0% slower
end to end** (`pp2048` 450.42 to 432.46), because the tie forces the
accumulator into fixed registers and the allocator pays 46 more `v_mov_b32` per
loop than the pairing saves. Logits bit-identical.

**This retires the largest card on the list and invalidates the arithmetic that
made it largest.** The projection's ~55-issue-slot gap to HIP is not closed by
recovering the 32 FMA slots; those are recoverable and recovering them loses.
The remaining difference must be in how HIP's accumulators avoid copies at all,
which is a register-assignment question, not an instruction-selection one.

The patch is not in tree. It is reconstructible from the README section, and
anyone revisiting should attack the copies rather than the pairing.

## 2. Short prompts stream weights at 2.2x lower bandwidth than HIP

**The largest measured gap, correctly diagnosed, with the fix identified and
blocked on register headroom.**

`pp32` and `pp64` are 44% and 43% of HIP, `pp128` 66%, 512 and above 82%.

Per-projection timings decompose it exactly (ms):

| projection | 32 tok | 128 tok | 512 tok |
| :--- | ---: | ---: | ---: |
| FFN gate/up | 3.081 | 3.268 | 6.232 |
| FFN down | 1.042 | 1.088 | 3.404 |
| SSM qkv | 1.354 | 1.389 | 3.627 |
| SSM alpha/beta | 0.211 | 0.219 | 0.212 |

32 and 128 tokens are near-identical because both are one token group, and
128 to 512 costs only 1.91x for 4x the tokens. So a projection call is **a fixed
cost plus a cheap marginal one**: the first token group streams the whole weight
matrix with no reuse (189 MiB in ~2.4 ms for gate/up, **79 GB/s**), and each
additional group is ~1.5 ms of compute riding cache at ~30 TOPS. HIP streams
that first pass at roughly **152 GB/s**.

Two tile changes were built and both lost, which rules out padding waste and
occupancy — see the README section "The sub-128 regime is weight-stream bound".

**The structural difference from HIP is `BK`.** HIP stages two K blocks per LDS
round, so threads `2r` and `2r+1` fetch *adjacent* 34-byte blocks of the same
row — 68 contiguous bytes per thread pair. HRX stages one, so every staging
thread sits on its own row, 5440 bytes from its neighbour, and a 128-byte line
is consumed across four separate loop iterations with 127 other rows competing
for cache in between. HIP's throughput config is literally
`W8A8BlockedWmmaGEMMKernel<128, 128, 2, 4, 2>` — the `2` is BK.

**Why this has not been done here, and what it needs.** BK=2 was tried twice in
this repo and rejected at 236 VGPRs and 0.4072 ms, both times measured on the
isolated harness — which re-reads one hot matrix and therefore cannot show a
weight-streaming benefit at all. The rejection was made in the one regime where
the change is invisible. HIP's own comment says BK=2 "is only reachable with the
register pressure the addressing rewrite and the one-token-tile-at-a-time
compute loop freed up", and notes that without them it "needed 500 bytes/lane of
scratch and ran 4.3x slower". Our kernel is at 184 of the 192 VGPRs an
eight-wave tier allows.

So the order is: **free registers first, then take BK=2, then measure end to end
at 32-128 tokens, not in the isolated harness.** Single-buffering the LDS stage
pays for BK=2's extra stage bytes exactly (2x128x32 ping-pong equals 1x2x128x32),
so the LDS budget is already there; it is the fragment and epilogue temporaries
that need to shrink.

**Acceptance:** `pp32`/`pp64`/`pp128` improve beyond noise, `pp2048` not worse,
logits bit-identical. Worth up to 2x on short prompts.

## 3. The alpha/beta projection is a single workgroup

`SsmAlphaBetaWidth` is 48, so the projection is 96 rows, and 96 rows on a
128-row macro tile is **one workgroup** — one CU busy out of twenty, at every
prompt length. It measures 0.211 / 0.219 / 0.212 ms at 32 / 128 / 512 tokens,
flat, because the token count never enters. Times 48 layers that is **10.1 ms
per chunk**: 0.22% at 2048 tokens but **2.5% at 32**.

At 160 K iterations and 8 waves on one CU it is 2 waves per SIMD, which cannot
hide latency; the measured 210 us against a ~49 us issue floor is 4.3x stalled.
The fix is K-parallelism — split the 160 blocks across 8 workgroups and reduce —
not a smaller row tile, which would still give only a handful of workgroups.

**Acceptance:** bit-identical (the reduction is over the same terms in a fixed
order); measurable at `pp32`, expect nothing at `pp2048`.

## 4. Chunkwise (matrix-form) DeltaNet recurrence

322 ms, 7.0% of the pass, and the largest non-projection block. The kernel is
sequential over the 2048 tokens of a chunk; its rows-per-workgroup was swept at
the current chunk size and 8 is the optimum, so the shape is not the lever. The
remaining lever is the algorithm: the chunkwise DeltaNet formulation replaces
the per-token scan with a matrix product over a token block, which is what
turns this into WMMA-shaped work.

**There is no reference to port.** `src/models/qwen/hip/batched_ssm.hip` is
also a sequential per-token scan, and the HIP README's own ranked headroom
records that its two obvious reformulations were tried and rejected. So this is
original derivation of the WY/UT chunkwise form, validated only end to end.

**Acceptance:** a math change, not a reassociation, so it needs an explicit
numerical decision up front. The parity gate is now route-aware and real, so it
will catch a mistake. Halving it is +3.4% of the pass. Do not start this
without deciding first what envelope the result is allowed to move.

## 5. A masked WMMA prefill attention kernel

208 ms, 4.5%. The current kernel is f32 online softmax at roughly 3.5 TFLOPS
against a 27 TFLOPS f32 VALU ceiling, so 13%. The cost is structural rather
than arithmetic: each `(position, head)` does a `vector.dotf` of 8 FMAs
followed by a `kernel.subgroup.reduce<addf>` whose DPP chain costs about as
much as the dot it reduces. HIP runs a masked WMMA kernel for the same work.

Arithmetic savings at this scale are *not* worth taking on their own: fusing
the accumulator rescale into an FMA cut the kernel 14.7 to 12.98 ms per layer
and was invisible end to end while costing precision (rejected this session).
Only the structural rewrite is worth doing.

A reference does exist -- `src/models/qwen/hip/kernels/attention_wmma.hip`,
575 lines, 15.9-17.4 TFLOPS or 29-32% of the WMMA ceiling -- but porting it is
not mechanical. It converts the query to FP16 (a precision change), depends on
a specific wave32 fragment layout, and needs V transposed into LDS to avoid a
32-way bank conflict. This project also measured that **f16 WMMA runs at half
the int8 WMMA rate on gfx1151**, so the ceiling a port would chase is lower
here than the HIP note implies.

**Acceptance:** prefill logits within a gated envelope; `pp2048` improves
beyond noise. Halving attention is +2.3% of the pass. Budget this as a port of
a tuned kernel, not an afternoon.

## 6. Fold the attention context quantize into the attention kernel

The SSM readout half of this card is **done** (+0.9%, bit-identical). One site
is left: the attention context, 0.47 ms x 16 layers, about 0.16% of the pass.

It is harder than the three folds already landed. In the attention kernel a
lane holds eight contiguous f32 of a head, so a 32-element Q8 block spans four
lanes and the amax needs a *clustered* subgroup reduce (`cluster_size = 4`,
which `kernel.subgroup.reduce` does support) rather than the plain wave-wide
reduce the readout could use.

**Acceptance:** bit-identical logits. Low priority at 0.16%.

## 7. Fuse the attention K and V projections

They have identical shapes (1024 rows, K=5120) and each re-reads the same
quantized activation tile: 1.14 + 1.09 ms per attention layer, 16 layers, about
0.7% of the pass.

It needs more than the `MergeAdjacent` trick `ssm_alpha_beta` uses. A merged
projection writes one token-major `[tokens][2048]` tile with K in rows 0..1023
and V in 1024..2047, while `batch_key` and `batch_value` are separate arena
buffers that `DispatchRoPEKVCacheBatch` reads at a 1024 stride. So the arena
layout and the RoPE/KV-cache kernel's strides have to change with it.

The upside is also smaller than it first looked. Per output row K and V cost
1.08 us against the gate projection's 0.73 us, so the loss is grid tail, not
activation traffic: 1024 rows is 8 row groups x 16 token groups = 128
workgroups against ~80 resident, so the second round is mostly empty. Merging
recovers roughly 0.72 ms per attention layer, **0.25% of the pass**, not 0.7%.
Not worth a layout change.

## 8. Ride upstream Loom, and know what blocks it

The derivation now pins `bce2ba37` (2026-08-27), ten days and ~220 Loom commits
newer than the previous pin. Performance-neutral and parity-identical, so this
is maintenance, not a win.

**Main is not usable yet.** `0cc34d04` ("[HAL/AMDGPU] Model ROCr AQL queue
execution modes", 2026-08-27) queries `HSA_AMD_AGENT_INFO_PM4_EMULATION`, which
the ROCr in `rocmPackages` 7.2.3 rejects with
`HSA_STATUS_ERROR_INVALID_ARGUMENT`; the AMDGPU accelerator then reports
unavailable and every HRX backend init fails. Re-try the bump when the ROCm pin
moves. Note also that upstream's locked-dependency helper now hard-fails at
configure time if any dependency declares patches and `git` is absent, which is
why `git` is in `nativeBuildInputs`.

Several AMDGPU codegen commits landed in this window that are worth re-reading
if the projection is revisited: "Narrow address arithmetic from facts", "Reuse
issue-consumed fragment addresses", "Consume VMEM VGPR sources at issue".

## A note on the remaining cards' size

Cards 4 and 5 are multi-session rewrites, not afternoon work, and should not be
started half-way. Card 4 has no reference implementation anywhere in this repo
and needs the WY/UT chunkwise DeltaNet derived and a numerical decision taken
up front. Card 5 is a port of a 575-line tuned HIP kernel that converts the
query to FP16 and depends on a transposed-V LDS layout, onto a target where f16
WMMA runs at half the int8 rate.

Cards 6 and 7 are **below this host's measurement resolution** (0.16% and 0.25%
against a ~0.5% noise floor on `pp2048`). They are correct and cheap, but
building them would produce changes that cannot be shown to help. Take them only
bundled with something measurable, or on a quieter host.

## Considered this round and not pursued

- **Fold the residual add into the following norm+quantize.** The same pattern
  that paid three times, worth about 0.5%, and bit-identical. Not taken because
  it defers a stage-N residual into stage N+1's norm, and HIP measured exactly
  this as `opt-c175-residual-defer` and rejected it as inside noise. 0.5% is at
  this host's resolution, so it would not be distinguishable either.
- **Fold the attention context quantize** (card 4's remaining half) at 0.16%,
  into the most intricate kernel in the set. Poor expected value.

## What is closed, and must not be retried without new information

All measured, all in the README with numbers:

- The blocked projection's tile shape. 128x128 with 8 waves per SIMD is a sharp
  local optimum; both dimensions are pinned, and every traffic argument loses
  to the occupancy cliff.
- Fragment loads taken straight from LDS into the WMMA operand bank: 208 VGPRs
  against 184, loses a tier.
- Unrolling the K loop for a static ping-pong parity: 256 VGPRs, 8 waves to 6.
- Carrying the accumulator bank as 64 scalars: unlocks VOPD, costs 56 moves.
- Fusing the attention accumulator rescale into an FMA: neutral end to end,
  costs precision.
- Per-token activation scales. This would remove the epilogue's activation
  multiply, worth about 5% of prefill, but it is a systematic precision
  reduction rather than a reassociation, and HIP explicitly declined the same
  trade. Beating HIP by quantizing more coarsely than HIP is not parity.
