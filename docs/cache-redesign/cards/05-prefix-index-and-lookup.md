# 05 · Prefix index and lookup

**Milestone:** Common package · **Depends on:** 04 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** agreed

## Goal

One prefix index over checkpoints and live frontiers, aware of storage tiers
from the start, that returns the deepest coherent compatible boundary and
explains its choice.

## Scope

- Token-prefix tree per compatibility identity.
- Availability per component: resident or durable. Only resident is used until
  card 13, but the field exists now so card 13 changes no data structures.
- Candidates returned: compatible live frontier, deepest coherent checkpoint,
  and shorter resident checkpoints, each with its estimated transfer cost.
- Default selection: an exact live continuation keeps its cheap path; otherwise
  the deepest usable checkpoint.
- Port the stable-prefix rule from `ContinuationCache::Acquire`: beyond a
  stable boundary, a later checkpoint is usable only with an eligible fallback
  at or before it.
- Port the input-identity prefix rule: a boundary after an image needs a
  matching image identity.
- Selection reason on every lookup, for the per-request reporting in card 19.
- Equivalents of `CachedPrefixTokens` and `CommonPrefixTokens`, which serving
  uses to learn branch points.

## Not in this PR

Disk candidates (card 13). Any cost-aware rule that picks a shorter boundary
(an RFC open question; it needs its own measured card).

## Test first

Table-driven tests:

- an exact live continuation beats restoring a checkpoint;
- the deepest coherent checkpoint is selected by default;
- a different image at the boundary is rejected;
- the stable-prefix rule matches today's behavior;
- a checkpoint with a missing component is not a hit;
- ties break deterministically.

## Step baseline

Lookup time for a 128k-token prompt with 128 retained checkpoints, and index
memory per checkpoint.

## Done when

- [ ] Tests above pass, including the lookup cases ported from the current
  cache's unit tests.

## Review focus

- Is the stable-prefix port faithful to the current cache?
- Lookup cost on a long prompt with many retained checkpoints.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (RadixAttention): radix-tree prefix matching.
- [Unified Radix Cache: one tree for hybrid model prefix caching](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache) (SGLang): one tree for all components, where depth is not the same as a reusable boundary.
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025): branch-point checkpoints and exact-match lookup for hybrid models.

## RFC

[Lookup, slot selection and restore](../RFC.md#lookup-slot-selection-and-restore)

## Review notes

