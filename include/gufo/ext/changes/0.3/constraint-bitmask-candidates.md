# Constrained compact candidate sampling and cached token bitmasks

Additive: `gufo/ext/sampling.hpp` (forwarding `src/core/json_constraint.hpp`
and `src/core/sampling.hpp`) adds cached packed token bitmasks and constrained
compact-candidate distributions for GPU speculative verification.

- `sampling::TokenBitmask` holds a packed 32-bit word bitmask (`words`) and
  `restricts` flag.
- `sampling::TokenConstraint` adds `LookupCachedBitmask(state)`; its mask
  cache keeps the 128 most recently used states.
- `sampling::SamplerState` adds `HasConstraint()`, `AllowsToken(token)`,
  `CachedConstraintBitmask()` and
  `TryDistributionFromCandidates(logits, unpenalized_count)`.

Migration: none.
