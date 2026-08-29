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

## 1. Teach Loom to lower a vector FMA to tied `fmac`s on the aggregate

**The single largest item, and the only one that can close the projection gap.**
Worth roughly +9% on the pass on its own.

Per K block the projection issues 64 `v_fma_f32`. They are VOP3, so RDNA3's
VOPD packer cannot pair them; HIP's equivalent packs into 32
`v_dual_fmac_f32`. That is 32 of the ~55 issue slots between the two kernels.

What is already known, from five instrumented compiler builds this session:

- Loom ships the `fmac_f32` VOPD component and the `amdgpu.v_fmac_f32`
  descriptor with TIED/DESTRUCTIVE constraints, but **no lowering rule offers
  the tied VOP2 form** for `vector.fmaf`/`scalar.fmaf`. Adding one
  (`_f32_fmac_rule` beside `_f32_fma_rule` in
  `loom/py/loom/target/arch/amdgpu/contracts/arithmetic.py`, plus the
  descriptor key in the `amdgpu.arithmetic` set) is correct: 54 of 57
  production kernels compile with it untouched.
- **A scalar loop-carried accumulator ties fine** and emits `v_fmac_f32`. A
  carried `vector<8xf32>` does not, because it is scalarized into per-lane
  slices and concatenated back, and the blocking interval is the carried
  aggregate two hops away (concat, then edge) from the tied result.
- It is the active set, not storage leases: the conflict reproduces under all
  three `LOOM_LOW_ALLOCATION_STORAGE_RELEASE_*` policies.
- Two allocator fixes were tried and rejected (widening
  `storage_alias_relation` to the edge causes; unit-granular liveness in
  `collect_tied_storage_aliases`). The kernel-side workaround (carry 64
  scalars) *does* unlock pairing but costs 56 extra moves per K block and
  measures 510.75 us against 496.31.

**So the change belongs in `loom/src/loom/transforms/vector/to_scalar*.c`:**
lower a vector `fmaf` to eight tied `fmac`s addressing the aggregate's units
directly, instead of materializing per-lane slices and a concat. Then the tie
is one hop, no extract moves appear, and the existing VOPD planner does the
rest.

**The alias chain is now fully mapped and the blocker is definitively the
liveness model, not the alias walk.** An instrumented build dumps it:

    tied operand 93  <-CONCAT-  92  <-COPY-  37

`93` is the tied operand (1 unit at base 16), `92` the per-iteration aggregate,
`37` the loop-carried accumulator (8 units at base 16, live [10,80]) and the
only interval that actually occupies the location. Both `LOW_CONCAT` and
`LOW_COPY` are already whitelisted alias causes; the walk in
`collect_tied_storage_aliases` is simply **one hop**, so it finds `92` --
correctly ignorable, it is dead at the tie -- and never reaches `37`.

Making the walk transitive (depth-capped, visited set, unit-granular liveness
query) was built and **still fails**: `37`'s units are reported live at the
tied definition because a loop-carried value is modeled live across the whole
body through the back edge. So no amount of alias chasing legalizes the tie.

The remaining fix is therefore **live-range splitting around the back edge** --
teaching the allocator that a carried unit dies at its redefinition inside the
body and is reborn on the edge. That is a substantially larger allocator change
than anything attempted here, and it is the *only* thing standing between this
kernel and HIP's issue count.

The verification this card previously asked for came back positive, for what it
is worth: the compiler already lowers `vector.mulf` on `vector<8xf32>` lanewise
into 31 `v_dual_mul_f32` with no extract moves, so internal lanewise lowering is
free. It is only the accumulator that cannot tie.

**Acceptance:** the three blocked projections compile; `v_dual_fmac_f32`
appears in the K-block loop; moves do not rise; prefill logits stay
bit-identical (a pure encoding change); `pp2048` improves beyond noise.
**Risk:** miscompilation-class change in a vendored compiler with no upstream
test suite here. The parity gate is now real, so a regression has something to
trip.

## 2. Chunkwise (matrix-form) DeltaNet recurrence

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

## 3. A masked WMMA prefill attention kernel

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

## 4. Fold the attention context quantize into the attention kernel

The SSM readout half of this card is **done** (+0.9%, bit-identical). One site
is left: the attention context, 0.47 ms x 16 layers, about 0.16% of the pass.

It is harder than the three folds already landed. In the attention kernel a
lane holds eight contiguous f32 of a head, so a 32-element Q8 block spans four
lanes and the amax needs a *clustered* subgroup reduce (`cluster_size = 4`,
which `kernel.subgroup.reduce` does support) rather than the plain wave-wide
reduce the readout could use.

**Acceptance:** bit-identical logits. Low priority at 0.16%.

## 5. Fuse the attention K and V projections

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

## 6. Ride upstream Loom, and know what blocks it

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
