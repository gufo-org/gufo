# 15 · Qwen 27B AR adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 (after 14 per D2) ·
**Size:** L (the inventory can land first as its own S PR) ·
**Affects:** nothing served until card 19 · **Status:** proposed

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

- [ ] Tests above pass on gfx1151.
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

