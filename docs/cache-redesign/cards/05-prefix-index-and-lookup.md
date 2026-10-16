# 05 · Prefix index and lookup

**Milestone:** Common package · **Depends on:** 04 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** done

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

### Implementation record

[`src/cache/prefix_index.hpp`](../../../src/cache/prefix_index.hpp) and
[`prefix_index.cpp`](../../../src/cache/prefix_index.cpp) add the common
package's compressed token-prefix tree, partitioned by compatibility identity.
Each identity registers the adapter's required component inventory. Immutable
checkpoint handles and live slot/generation descriptions share the tree;
lookup returns every eligible resident candidate, logical transfer bytes,
a selected boundary and an explicit selection reason. The deepest usable
boundary wins, with a live continuation preferred on equal boundaries.
Checkpoint ID, then slot/generation and entry ID resolve remaining ties.
This changes no serving behavior; slot leases and generation enforcement remain
with card 06, and durable-only candidates remain ineligible until card 13.

Per-component availability describes resident, durable, both or neither. Lookup
requires every registered component, the expected layout and a resident supply.
Live frontiers supply independent component positions; private state must match
the exact token boundary. Append-row counts remain component-specific. Transfer
estimates count logical row bytes plus private-state bytes, rather than backing
capacity; live continuation estimates zero transfer bytes.

The stable-prefix port preserves `ContinuationCache::Acquire`'s two fallback
forms: a compatible retained checkpoint at/before the requested stable boundary,
or a compatible checkpoint attesting a previously established stable boundary
at/before it. A live frontier alone cannot attest a fallback. Availability and
input identity gate both fallback forms. The supplemental identity is compared
at each saved boundary, so a pre-image checkpoint remains usable after an image
changes, while a boundary after that image requires its identity to match.

`CachedPrefixTokens` ignores the stable-prefix rule and reuse flag and considers
only usable resident boundaries. `CommonPrefixTokens` learns token agreement
with stored records even when no saved boundary exists there or a live frontier
is busy. It conservatively requires the stored record's complete input identity
at its full boundary; a record longer than the query compares against the
query's complete supplemental identity. It never invents a restorable checkpoint
at a divergence point.

Index operations are caller-serialized and invoke no adapter/device callbacks.
Returned checkpoint handles retain payload ownership after index eviction.
Metadata admission uses the resource ledger; splits preserve existing record
pointers and release obsolete token capacity. Traversal and normal teardown
are iterative. Erasure prunes empty branches and returns their metadata charges.
The index excludes separately charged checkpoint tokens/storage from its own
metadata figures. Allocator, shared-pointer and ledger bookkeeping and temporary
lookup result vectors are outside those figures.

[`tests/cache/prefix_index_test.cpp`](../../../tests/cache/prefix_index_test.cpp)
covers the selection table, faithful stable-boundary eligibility, appended,
changed and removed images, missing/durable-only components, incompatible
inventories/layouts, live private-state positions, deterministic ties,
compatibility isolation, eviction with result pins, failed admission, branch
learning and the 128k/128-checkpoint workload in both insertion orders. The test
is included in CPU presets and the hosted PR target. Size-only capture fixtures
exercise actual checkpoint and ledger operations without allocating payload
buffers or claiming numerical restore validation.

## Results

Step baseline on Linux 7.2.9 x86-64, AMD Ryzen AI MAX+ 395, based on main
`087c192d`, with flake.lock-pinned GCC 15.3.0, CMake 4.3.4 and Python 3.14.6:

| Measurement | Result |
| --- | ---: |
| Lookup, 131,072-token prompt, 128 resident checkpoints | 35.13 µs/lookup |
| Empty registered index metadata | 497 bytes |
| Retained index metadata, 128 checkpoints | 579,057 bytes |
| Incremental index metadata per checkpoint | 4,520 bytes |

The benchmark uses three components with independent row frontiers, boundaries
spaced by 1,024 tokens, 20 warmup lookups and 1,000 measured lookups. It includes
candidate construction, identity/coherence checks, cost estimates and sorting;
each iteration verifies the deepest boundary and all 128 candidates. The initial timing is
one CPU microbenchmark run in `RelWithDebInfo`, not an inference, GPU transfer or
contention measurement. This is a step baseline, with no directly comparable
RFC measurement or speedup claim. Metadata includes shared edge tokens and
record capacities; both ascending and descending insertion produce the same
retained metadata. Teardown returns the ledger to zero.

### Validation

CPU commands run in a `pkgs.mkShell` using the repository's flake.lock-pinned
nixpkgs with CMake, Ninja, pkg-config, Python, clang-tools, ICU, curl, libpng,
libjpeg, libwebp and OpenSSL. Release configuration uses the production
package's `inputsFrom` dependency shell.

```sh
cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
cmake --build --preset cpu-test --target cache_prefix_index_test cache_prefix_index_bench --parallel 4
ctest --preset cpu-test -R '^cache_(prefix_index|checkpoint|ledger|adapter|adapter_lifecycle|boundary)_test$' --output-on-failure
build/cpu-test/src/cache/cache_prefix_index_bench
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_prefix_index_test cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(prefix_index|checkpoint|ledger|adapter|adapter_lifecycle|boundary)_test$' --output-on-failure
cmake --preset release -DGUFO_BUILD_TOOLS=OFF
cmake --build --preset release --target gufo_cache --parallel 4
cmake --build --preset pr --parallel 4
nix shell --inputs-from . nixpkgs#clang-tools -c python3 tools/ci/check-format.py
```

All six focused cache tests passed in ordinary and ASan/UBSan builds; all 47
hosted PR tests passed. The production release cache library compiled, and
focused clang-tidy passed for the new index source/header. Shared formatting,
documentation and diff whitespace checks passed. The focused tidy line filter
excludes pre-existing header diagnostics outside the changed index files.
Build artifacts remain in ignored `build/` directories. No inference path is
connected, so model functional/timing qualification remains with switch-over.

### Review follow-up, round 1

A fresh-context review identified retained empty unary nodes after prefix
insert/erase churn and token-edge allocation before ledger admission. Both are
fixed. Erasure coalesces empty unary paths while preserving surviving record
pointers. If temporary replacement admission/allocation fails, erasure succeeds
with a coherent tree and a later erasure retries compaction. A regression retains
one 131,072-token checkpoint through 1,023 short live-prefix insert/erase cycles
and verifies unchanged metadata and exactly one candidate after every cycle.
Additional reserve/convert fault tests verify safe deferred compaction.

Index construction, registered inventories, new edges and split suffixes now
reserve predictable capacity before allocating, verify actual retained capacity
with the pinned standard library, and convert after successful construction.
Publication map nodes are preallocated before mutating the token tree. Rejection
tests observe actual heap allocations and confirm that denied new edges, split
suffixes and component inventories allocate no unadmitted large buffer.

The focused test passed with ASan/UBSan and ordinary assertions; focused tidy,
formatting, documentation and the production cache-library rebuild passed.
The benchmark rerun after these fixes recorded **36.16 µs/lookup** and unchanged
metadata (**579,057 bytes**, **4,520 incremental bytes/checkpoint**). The initial
35.13 µs result remains above as prior step evidence; this is not a regression
qualification or an inference timing comparison.

### Review follow-up, rounds 2 and 3

Round 2 confirmed the prior fixes and found no new actionable defect. Its
independent scratch oracle passed 100,000 lookup/common-prefix comparisons
across 20,000 mixed live insert/erase operations.

Round 3 found that checkpoint eligibility omitted the exact private-state
position check already applied to live frontiers. Lookup now rejects a
checkpoint whose private execution position differs from its advertised token
boundary, and excludes it from both retained and attested stable fallbacks.
The new regression failed before the guard and covers private state before and
after the boundary, both fallback forms, and a valid matching checkpoint.
Append-row frontiers remain independent of target token count.

All six ordinary focused cache tests, the expanded index test with ASan/UBSan,
focused tidy, shared formatting, documentation, whitespace checks and the release
cache-library rebuild passed. The post-fix benchmark recorded **36.50 µs/lookup**,
with metadata unchanged. Earlier measurements remain above as step evidence.

## Done when

- [x] Tests above pass, including the lookup cases ported from the current
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

