# 17 · DeepSeek V4 Flash adapter

**Milestone:** Adapters · **Depends on:** 02, 06, 08 (after 16 per D2) ·
**Size:** L · **Affects:** nothing served until card 19 ·
**Status:** proposed

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

- [ ] Capacity rows recorded; tests above pass for both modes.
- [ ] The model's standard speed benchmark against matched `main` shows no
  regression from the inactive guard calls (per phase, 5% / 3 ms).

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

