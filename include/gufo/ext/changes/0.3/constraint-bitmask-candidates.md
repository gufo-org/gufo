# Constrained compact candidate sampling and cached token bitmasks

Additive: `gufo/ext/sampling.hpp` (forwarding `src/core/json_constraint.hpp`
and `src/core/sampling.hpp`) adds cached packed token bitmasks and constrained
compact-candidate distribution helpers for GPU speculative verification.

- `sampling::TokenBitmask` holds a packed 32-bit word bitmask (`words`) and
  `restricts` flag.
- `sampling::TokenConstraint` adds `AllowedBitmask(state)` and
  `LookupCachedBitmask(state)`, backed by a 512-entry LRU mask cache alongside
  `Allowed(state)`.
- `sampling::SamplerState` adds `HasConstraint()`, `AllowsToken(token)`,
  `ConstraintBitmask()`, `CachedConstraintBitmask()`, and
  `TryDistributionFromCandidates(logits, unpenalized_count)`.

Migration: none.
