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
tokens, all retained fusions on. Total chunk 4580 ms; attention 532, SSM 1355,
FFN 2692.

| Work | ms of 4580 | Share |
| :--- | ---: | ---: |
| Blocked W8A8 projection, all shapes | ~3500 | 76% |
| DeltaNet recurrence | 322 | 7.0% |
| Attention kernel | 208 | 4.5% |
| SwiGLU + quantize (fused) | 95 | 2.1% |
| RMSNorm + quantize (fused) | 52 | 1.1% |
| SSM prepare + conv | 80 | 1.7% |
| Residual adds | 75 | 1.6% |
| Everything else | ~250 | 5.5% |

`pp2048` is 446-450 t/s against HIP's 557 when HIP's page cache is warm, so
80%. `tg16` is 7.90 against 7.78, so decode is already ahead and is not a
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

One more thing is known since: `sroa-vector-banks` already splits a carried
`vector<8x8xf32>` bank into eight `vector<8xf32>` slots, so what blocks the tie
is the *slot*, not the bank. Splitting a slot into lanes is exactly what the
rejected source-level variant did at a cost of 56 moves; doing it inside the
compiler avoids those moves only if the lane reads of the non-accumulator
operands stay subregister reads rather than materialized values. That is the
one unverified assumption, and it should be verified before the pass is
written.

**Acceptance:** the three blocked projections compile; `v_dual_fmac_f32`
appears in the K-block loop; moves do not rise; prefill logits stay
bit-identical (this is a pure encoding change); `pp2048` improves beyond noise.
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

**Acceptance:** this is a math change, not a reassociation, so it needs an
explicit numerical decision and a parity envelope that actually gates (card 6).
Halving it is +3.4% of the pass.

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

**Acceptance:** prefill logits within a gated envelope; `pp2048` improves
beyond noise. Halving attention is +2.3% of the pass.

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
quantized activation tile: 1.14 + 1.09 ms per attention layer, 16 layers.
Worth about 0.7% of the pass, and only if `attn_k` and `attn_v` turn out to be
adjacent in the checkpoint the way `ssm_alpha_beta` already is. Cheap to check
before committing to it.

## 7. Ride upstream Loom, and know what blocks it

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
