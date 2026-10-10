# 15 · Qwen 27B AR adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 (after 14 per D2) ·
**Size:** L (the inventory can land first as its own S PR) ·
**Affects:** nothing served until card 19 · **Status:** draft; timing qualification inconclusive

## Goal

Implement the adapter for Qwen 27B AR and prove, with model-local tests, that
capture and restore reproduce the exact state.

## Step 1: inventory (reviewable on its own)

Starting point is `QwenGpuSnapshot`:

| Component | Proposed kind | Notes |
| --- | --- | --- |
| Attention KV (f32 or f16 by storage mode) | append rows | attention layers, KV width, full-attention interval |
| DeltaNet state | private | per recurrent layer |
| Convolution state | private | per recurrent layer |
| Valid context | position | must match the target KV rows |
| Vision RoPE layout (image grids) | private metadata + input identity | changes positions after an image |

Mutation paths to audit, from `QwenGpuArena` and the executor: prefill,
decode `Advance`, `Reset`, `RestoreSnapshot`, `SaveState` and `RestoreState`,
SSM replay capture. List anything else found. DFlash is out of scope
(card 16).

## Step 2: adapter

- Descriptors, `CapturePrivate`, row copies through card 08's primitives,
  `Validate`.
- Guard calls in every audited mutation path.
- **No cost while inactive:** until card 19 attaches the new cache, the guard
  calls do nothing on the production path.
- The runner state holds the model-level session the adapter owns (card 02
  review point).

## Test first

Model-local tests:

- capture at a boundary, restore into another slot: the bytes are equal and
  next-token logits match an uninterrupted run;
- borrowed rows are preserved before an overwrite on each mutation path;
- an edit before a checkpoint restores an earlier boundary and never
  truncates recurrent state;
- an image changes positions after it, and a different image at the
  boundary is not a hit.

## Step baseline

At 32k and 100k tokens: private bytes per checkpoint, KV bytes per token,
capture time and restore time into another slot. RFC for comparison: full
copies of 235–265 ms at 149k; about 152 MiB fixed plus 64 KiB per token.

## Done when

- [x] Tests above pass on gfx1151.
- [ ] The model's standard speed benchmark against matched `main` shows no
  regression from the inactive guard calls (per phase, 5% / 3 ms).

## Review focus

- Is the inventory complete? Missing state is silent corruption.
- Are the f16/f32 KV storage modes two layouts or one descriptor with a
  parameter?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025): DeltaNet state semantics.
- [Hybrid Models Meet SGLang: More than Full Attention](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/): recurrent-state checkpoints in a prefix cache.
- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): context checkpoints for recurrent models.
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025): exact-boundary hits for hybrid models.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules) ·
[Correctness contract](../RFC.md#correctness-contract) ·
[Why recurrent state needs exact checkpoints](../RFC.md#why-recurrent-state-needs-exact-checkpoints)

## Review notes

## Inventory and implementation

[The model-local inventory](../../models/qwen3.8-27b/CONTINUATION-ADAPTER.md)
records the exact stable AR frontier and every guarded mutation. Separate
layout versions distinguish native FP16 KV and gathered FP32 KV; alternate
recurrent precision also has a separate version and compatibility identity.
The adapter owns the same `QwenGpuExecutor` type used by the serving runner.
The new cache remains unattached until card 19.

Portable geometry checks and model-local restore/guard/image checks are added.
Production capacity probes commit 8 GiB backing before prefill, pack the
checkpoint into 64 MiB slabs, restore into a different slot and verify every
piece and sixteen continuation logit rows. This isolates the adapter; it is
not qualification of the common checkpoint allocator's size classes/slack.

## Results

October 10, 2026, on gfx1151: the Q4 and Q8 model-local checks pass with FP16
and FP32 KV. They compare every component byte and full continuation logits,
cover earlier-boundary edits, split/out-of-order transfers, actual leased-row
preservation, failure recovery, completion lifetime and image input identity.
Both batched decode and verification reject a guard refusal before any write
and leave untouched sessions valid. Both quantizations have full FP16/FP32
restore checks and focused follow-ups for the reviewed image-suffix fix.
The focused image checks use production FP16 KV.
The hosted suite passes 56/56 and the shared formatting check passes.

Three independent reviews are complete. The first found that restoration
discarded future images from the attached request; this is fixed and tested
before the first image and between two images. The second and third found no
confirmed source defect. Focused scalar-overwrite, committed-replay and
successful verification-batch preservation checks address the third review's
coverage observations; their final Q4/Q8 runs pass, including byte-exact
committed recurrent replay against scalar continuation in both KV layouts.

Production FP16 KV step probes pass with two allocated slots and 8 GiB of
committed backing. Every checkpoint piece and sixteen continuation logit
rows/token decisions match exactly after restoring into the other slot.
Private state is 159,852,608 bytes and KV is 65,536 bytes per token.

| Target | Tokens | Private capture (ms) | Full capture (ms) | Restore (ms) | Assigned 64 MiB slabs (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 | 32,768 | 6.460 | 33.750 | 29.022 | 2,348,810,240 |
| Q4 | 100,000 | 5.998 | 93.168 | 82.284 | 6,777,995,264 |
| Q8 | 32,768 | 4.555 | 32.275 | 29.221 | 2,348,810,240 |
| Q8 | 100,000 | 4.193 | 90.496 | 82.034 | 6,777,995,264 |

These are single observations, not averages or allocator qualification. At
100K, the two native request states occupy 13,441,695,744 bytes and temporary
scratch 4,639,245,480 bytes; resident model claims are 17,559,178,144 bytes (Q4)
or 31,457,991,680 bytes (Q8), in addition to committed backing. The production
capacity probes use FP16 KV; FP32 KV has the model-local restore checks above.
Alternate BF16 recurrence has portable geometry coverage only.

Production qualification uses frozen `main` `3fac1bb5`, with
the same GCC 15.3.0 / ROCm 7.2.3 toolchain. Production source `2e4f0c54e05b`
includes the image-suffix fix. Earlier source `506a73ffe72f` evidence is
retained separately under `pre-review-image-fix/`; subsequent guard additions
change tests only. The serving candidate
binary SHA-256 `9484c9c38669340485e3194cc43703fb5be1c29477518696e43e0c50f88a12dc`
is pinned. Raw tests, binaries and measurements are retained outside Git in
`/home/mixer/gufo-qualification/cache-card15-2026-10-10`.

The bounded inactive-path benchmark covers pp2048/tg128 at d0/d32K for Q4/Q8
AR and mixed/repetitive DFlash2, plus C1/C2/C4/C6/C8 AR, two mixed corpus
cases and repetitive DFlash2. All 146 paired request observations have exact
completion hashes and token counts. Actual draft counts are zero for all
52 AR requests per build and positive for all 94 DFlash2 requests per build.
Nine of the initial 292 request/phase timings were flagged. Reverse-order
affected pairs and unchanged-main controls retain every observation; the
final comparison has 263 passing phases, 29 inconclusive phases and zero
confirmed regressions. Main's shallow Q4 prefill itself varied by roughly
10%; mixed draft traffic and per-peer decode times also vary between runs.
The unchanged 5% / 3 ms gate is applied per request/phase without averaging.
**Inactive-path speed remains unqualified**, and the speed checkbox above
remains unchecked.

All four HTTP profiles (Q4/Q8 × off/DFlash2) pass the selected long-context
and cache checks: 45 requests per profile per build, exact output and
prefill/cache work, with loaded mode and actual draft execution verified.
Initial timings flag 93 of 2,408 metrics. Reverse-order follow-ups and
unchanged-main controls retain every observation, with the same 5% / 3 ms gate.

| HTTP profile | Passing metrics | Inconclusive metrics | Confirmed regressions |
| --- | ---: | ---: | ---: |
| Q4 AR | 570 | 32 | 0 |
| Q4 DFlash2 | 522 | 80 | 0 |
| Q8 AR | 538 | 64 | 0 |
| Q8 DFlash2 | 414 | 188 | 0 |

Four Q8 DFlash2 snapshot/queue metrics remained flagged after the first
follow-up and main control. One additional matched pair replayed that profile's
complete preceding long-context and cache history. Three candidate timings
returned to the observed main range; the remaining thinking/tool snapshot
also stalled on unchanged main (128.696 ms versus 22.196 ms in that candidate
replay). The earlier candidate's 658.235 ms live checkpoint remains in the
raw logs. The final comparison includes all observations: 2,044 passing,
364 inconclusive and zero confirmed regressions. Outputs, cache work and
loaded mode/actual drafts match throughout. **HTTP speed remains unqualified**;
these controls do not turn noisy timings into a pass.

No geometry-only check or unexecuted test qualifies these measurements.
The restore checks establish execution consistency, not original-weight or
upstream parity.
