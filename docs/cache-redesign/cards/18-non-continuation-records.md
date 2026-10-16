# 18 · Capability records for non-continuation models

**Milestone:** Adapters · **Depends on:** 02 · **Size:** S ·
**Affects:** nothing user-visible · **Status:** agreed

## Goal

Every model family states its continuation capability explicitly, so "all
models are covered" is checkable rather than implied.

## Scope

- Qwen3 ASR, Qwen3 TTS, Qwen Image 2.1 and MiniMax H3 declare
  `continuation = false`.
- Check whether ASR or TTS has internal state that a later request could
  reuse (streaming sessions, for example). If so, record it as a separate
  future capability, not a token-prefix continuation.
- A test enumerates every registered model family and fails if one has no
  capability record.

## Not in this PR

Any reuse for these models. Diffusion trajectories are not forced into the
token-prefix interface.

## Test first

The enumeration test, failing until each family has its record.

## Step baseline

None: declarations only.

## Done when

- [ ] Every family has a record and the test passes.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules)

## Review notes

