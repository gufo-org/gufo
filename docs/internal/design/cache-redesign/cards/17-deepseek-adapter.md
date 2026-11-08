# 17 · DeepSeek V4 Flash adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 (after 16 per D2) ·
**Size:** L · **Affects:** DeepSeek V4 Flash AR and DSpark serving ·
**Status:** done

## Goal

Implement the adapter for DeepSeek V4 Flash in AR and DSpark modes, and prove
with model-local tests that capture and restore reproduce the exact state.

## Scope

- **Capacity.** The RFC research has no continuation-capacity campaign for
  DeepSeek. Measure and record its rows, with committed backing included,
  before the adapter.
- **Inventory**, starting from `Session` and `SessionSnapshot`:
  - compressed KV pools;
  - sliding windows;
  - indexers;
  - mutable compressor state;
  - DSpark draft state.

  Compressed pools may not have a plain "row per token" geometry. Decide per
  component whether it can be shared or must be private.
- **Mutation paths to audit:** session runtime and ROCm graph replay,
  including rollback.
- Guard calls on every audited mutation path.
- **No cost while inactive:** until card 19 attaches the new cache, the guard
  calls do nothing on the production path.

## Test first

Model-local round trips for each component, and next-step behavior matching
an uninterrupted run in both modes.

## Step baseline

At 32k and 100k tokens, both modes: private bytes per checkpoint, shareable
bytes per token, capture and restore time.

## Done when

- [x] Capacity rows recorded; tests above pass for both modes.
- [x] Functional timing against matched `main` shows no regression (see below).

## Review focus

- Chunk geometry for compressed pools (an RFC open question).

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [ds4](https://github.com/antirez/ds4): the upstream of Gufo's DeepSeek port and its session state; already listed in `THIRD_PARTY_NOTICES.md`.
- [vLLM hybrid KV cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/): state groups whose block geometry differs, as compressed pools may.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules)

## Review notes

Serving moved with this card rather than waiting for card 19: the runner opts in
through `UsesComponentCache()`, as Flash-Next and Qwen 27B do.

Chunk geometry needed no special case. Compressed rows are append-only and final
once emitted, so they share like Flash-Next's pooled keys, with component-specific
positions of `position / ratio`. Sliding windows and compressor frontiers change
every token and stay private, one component per layer. The FP16 attention mirror
is derived and rebuilt on restore. Rows are copied rather than borrowed because
compressed caches grow by reallocation.

Capacity, measured with `ds4_continuation_adapter_test --capacity` on gfx1151
using pageable test buffers and one transfer at a time, so serving can be faster:

| Mode | Depth | Private bytes | Row bytes | Row bytes per token | Capture | Restore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 32,768 | 23,995,584 | 451,018,752 | 13,764 | 157 ms | 398 ms |
| AR | 100,000 | 23,995,584 | 1,376,389,760 | 13,764 | 497 ms | 1,329 ms |
| DSpark | 32,768 | 24,782,016 | 451,018,752 | 13,764 | 166 ms | 80 ms |
| DSpark | 100,000 | 24,782,016 | 1,376,389,760 | 13,764 | 434 ms | 1,555 ms |

A checkpoint inherits completed row chunks from its parent, so only new rows and
the private state are copied.
