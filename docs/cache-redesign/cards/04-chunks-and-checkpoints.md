# 04 · Chunks, checkpoints and provenance

**Milestone:** Common package · **Depends on:** 03 · **Size:** M–L ·
**Affects:** nothing at runtime · **Status:** done

## Goal

Represent immutable shared chunks, checkpoint records and the lineage that
decides which checkpoints may share bytes.

## Scope

- **Chunk:** component, row range, lineage, and location. Location is either
  borrowed (slot plus generation) or committed backing; the disk location is
  added in card 10.
- **Reference counts:** checkpoint references, reader pins and persistence pins.
  A chunk is freed only when all three reach zero.
- **Checkpoint:** boundary, compatibility and input identity, per-component
  positions, private-state handle, chunk references, private tail references,
  purpose and rank.
- **Visibility:** a checkpoint is visible to lookup only after its private
  state and all component descriptions are complete.
- **Lineage rules:**
  - a cold prefill starts a new lineage;
  - restoring a checkpoint and continuing inherits its lineage;
  - partial tails are private to their checkpoint.
- Chunk geometry comes from `rows_per_chunk` in each descriptor, not from a
  fixed 2,048.

## Not in this PR

Lookup (card 05). Slot leases and mutation (card 06). Disk location (card 10).

## Test first

With the fake adapter:

- two checkpoints in one lineage share their common chunks;
- two cold prefills of identical tokens create separate lineages and share
  nothing;
- evicting a child keeps the parent's chunks;
- removing the last reference frees the chunk and the ledger returns to zero.

## Step baseline

- Metadata bytes per checkpoint and per chunk.
- Retained bytes for the RFC's eight-checkpoint 27B example replayed with real
  sizes in the fake adapter (the RFC estimate is 7.94 GiB).

### Implementation record

[`src/cache/checkpoint.hpp`](../../../src/cache/checkpoint.hpp) and
[`checkpoint.cpp`](../../../src/cache/checkpoint.cpp) add immutable checkpoint
records, full-chunk references, checkpoint-private state/tails and execution
history. `gufo_cache` remains independent of serving; this changes no runtime
behavior. Prefix lookup, slot protection, preservation/materialization and the
physical backing pool remain with cards 05, 06 and 08.

A cold execution gets a new lineage. A successfully restored checkpoint seeds
its new execution history with exactly that checkpoint's full chunks and inherits
its lineage. Each branch has its own weak chunk directory: a lineage ID alone
never authorizes sharing a later sibling's suffix. Partial tails are always new
private payloads, including repeated capture at the same boundary. Completing
a chunk later leaves the older checkpoint's tail intact. History checks token
and supplemental-input prefixes and independent component frontiers before
calling the capture provider. Geometry comes from each component descriptor.

Capture returns a `shared_ptr<const Checkpoint>` only after all component
callbacks and metadata admission/conversion succeed. Callbacks return completed
private copies or borrowed descriptions with already assigned backing; callers
settle adapter transfers and preserve a coherent execution boundary. Failed
capture unwinds private owners, new chunks and temporary references without
publishing or changing the history. Reporting peaks retain attempted admission;
current byte categories return to their pre-attempt values. The provider cannot
reenter capture. The ledger facade outlives histories.

Chunks distinguish checkpoint references, reader pins and persistence pins.
Pins retain both actual storage and accounting; persistence adds the ledger's
unique-byte overlay. History directories hold weak references and cannot keep
payload alive after the last checkpoint/reader/writer reference. `Prune` releases
expired weak bookkeeping; retained directory capacity stays charged until
replacement or teardown. Opaque payload owners release before their charges.
The ledger's new `Info` accessor validates capacity/category and distinguishes
the pool handle from the active payload assignment. Borrowed locations carry
slot ID and generation; enforcing that generation belongs to card 06.

[`tests/cache/checkpoint_test.cpp`](../../../tests/cache/checkpoint_test.cpp)
uses the existing fake adapter for captures, actual restore/continuation and
content comparisons. It covers sharing, identical independent cold prefills,
parent/child eviction, private tails, sibling forks, differently sized component
chunks and frontiers, borrowed backing, pin failures, metadata/transfer failure,
every reserve/convert failure point, identity changes and zero-byte teardown.
It is registered in both CPU presets and the hosted PR target.

## Results

Step baseline on Linux x86-64 (AMD Ryzen AI MAX+ 395), based on `c8d3ab55`,
with flake.lock-pinned GCC 15.3.0, CMake 4.3.4 and Python 3.14.6:

| Measurement | Recorded bytes |
| --- | ---: |
| Full-chunk record (`sizeof`, excluding allocator bookkeeping) | 176 |
| Checkpoint metadata at 86,016 tokens, including token/component/reference capacities | 348,397 |
| Checkpoint metadata at 88,064 tokens | 356,669 |
| Checkpoint metadata at 90,112 tokens | 364,941 |
| Checkpoint metadata at 92,160 tokens | 373,213 |
| Checkpoint metadata at 94,208 tokens | 381,485 |
| Checkpoint metadata at 96,256 tokens | 389,757 |
| Checkpoint metadata at 98,304 tokens | 398,029 |
| Checkpoint metadata at 100,352 tokens | 406,301 |
| Eight checkpoints' metadata combined | 3,018,792 |
| All retained metadata, including history, lineage and chunk records | 3,447,181 |
| Retained payload: 98 full chunks plus eight private-state copies | 8,526,268,672 (7.9407 GiB) |

The sized fake replay uses the RFC's eight boundaries, 65,536 combined row
bytes/token and 243,700,000 private bytes/checkpoint. Its two row components split
the combined row size equally. It exercises real ownership and ledger operations
with size-only opaque backing owners. It does not allocate 7.94 GiB of physical
buffers or qualify GPU numerics, transfers, inference speed or contention.
Metadata records object sizes and retained vector capacities; allocator,
shared-pointer and ledger-token bookkeeping and temporary test inputs are outside
these reported figures. The replay ran once on Linux 7.2.9 using the `cpu-test`
build. Teardown returned the ledger to zero.

### Validation

Validation commands (CPU dependency shell from pinned nixpkgs; production
package `inputsFrom` for the release configuration):

```sh
cmake --preset cpu-test
cmake --build --preset cpu-test --target cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-test -R '^cache_(checkpoint|adapter|adapter_lifecycle|ledger|boundary)_test$' --output-on-failure
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_checkpoint_test cache_ledger_test cache_adapter_test cache_adapter_lifecycle_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(checkpoint|adapter|adapter_lifecycle|ledger|boundary)_test$' --output-on-failure
cmake --preset release
cmake --build --preset release --target gufo_cache --parallel 4
cmake --build --preset pr --parallel 4
build/cpu-test/tests/cache/cache_checkpoint_test
```

All five focused tests passed in ordinary and ASan/UBSan builds; all 46 hosted
PR tests passed. The production release library, shared formatting check,
focused clang-tidy, documentation checks and diff whitespace checks passed.
Build artifacts remain in ignored `build/` directories. These results were
recorded from the implementation committed as `bc61da51`.

### Review follow-up

Independent review found two missing edge cases. Empty, short and private-only
checkpoints now skip entry-directory reservations when no full chunks exist,
both during capture and when seeding a restored history. Rejected committed or
borrowed descriptions first place the physical owner and accounting token in a
local `Payload`, so exception cleanup destroys the owner before releasing its
charge or allowing the committed pool block to be reassigned.

The new regression tests failed before these fixes. They verify empty/short
capture and restore, private-only histories, and deleters that observe a live
charge and cannot reassign borrowed backing while the old owner is being
destroyed. Restore checks now compare complete target rows, draft rows and
private bytes with independent uncached executions, both at the restored
boundary and after continued execution. The focused ordinary and ASan/UBSan
cache checks, hosted PR suite, release library and shared repository checks
were rerun for the fixes. The step-baseline figures above remain unchanged.

## Done when

- [x] Tests above pass.

## Review focus

- Lineage rules: are there restore paths that should inherit but would
  start a new lineage, or the reverse?
- Tail ownership when a later checkpoint fills the chunk the tail belongs to.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104) (RadixAttention): sharing prefix storage among requests through a tree.
- [vLLM automatic prefix caching, v0.22.1](https://docs.vllm.ai/en/v0.22.1/design/prefix_caching/): block identity chained on the parent block. Contrast: we share by inherited lineage, not by token hash.
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025): recurrent state allows only exact-boundary hits; why checkpoints, not truncation.

## RFC

[Sharing and provenance](../RFC.md#sharing-and-provenance) ·
[Capture and borrowing](../RFC.md#capture-and-borrowing) ·
[Expected resource savings](../RFC.md#expected-resource-savings-and-their-limits)

## Review notes

