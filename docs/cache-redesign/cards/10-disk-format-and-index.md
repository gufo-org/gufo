# 10 · Disk format, directory lock and startup index

**Milestone:** Disk store · **Depends on:** 04, D4 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** done

## Goal

Define the versioned on-disk layout, give one process exclusive ownership of
the directory, and build the startup index from metadata only.

## Scope

- **Layout:** `<dir>/v2/` with `manifests/`, `chunks/`, `private/`, `tmp/`,
  plus a `LOCK` file. Format and component layout versions are recorded
  separately from model display names.
- **Manifest:**
  - format version and compatibility identity (hashed);
  - exact tokens, needed for exact verification;
  - component positions;
  - chunk references with sizes and checksums;
  - private-state and tail files with sizes and checksums.
- **Exclusive lock:** a second process fails at startup with a clear message
  naming the directory.
- **Startup:** read manifests (each bounded in size) and stat their
  dependencies. Read no payload bytes. Mark entries "durable, unverified".
- **Legacy files:** recognize the current store's managed files (`*.kvc` with
  the `GUFO` magic, `.tmp-*`) and remove them within the disk budget. Never
  touch unknown files and never recurse into unrelated directories.

## Not in this PR

Writing and publication (card 11). Streaming payloads (card 12). Lookup
across tiers (card 13).

## Test first

CPU tests:

- a second store on the same directory fails with the ownership error;
- bytes read at startup grow with the number of manifests, not with payload
  size (assert through an I/O counter);
- unknown files and directories are untouched;
- legacy managed files are removed and the disk budget is respected
  throughout.

## Step baseline

Startup index time and bytes read for 100 and 1,000 manifests, and manifest
size at 100k tokens.

### Implementation record

[`src/cache/disk.hpp`](../../../src/cache/disk.hpp) and
[`disk.cpp`](../../../src/cache/disk.cpp) define a standalone CPU disk store.
Serving does not instantiate it until card 19. `LOCK` is at the configured
root so ownership covers legacy cleanup and `v2/` together; it is never
unlinked. Linux `flock` fails immediately with the directory in the ownership
error. All managed namespace directories and files reject symlinks; indexed
files also reject hard links and nonregular files.

Published manifests, full chunks and private payloads use 128-bit opaque file
IDs encoded as 32 lowercase hex digits plus `.bin`, within `manifests/`,
`chunks/` and `private/` respectively. `tmp/` reserves the same naming convention
for card 11. Names outside this convention are unknown. Startup counts the
logical sizes of every recognized regular v2 file, including invalid manifests,
orphan payloads and temporary files, once per directory entry. Filesystem block
allocation and directory/lock inode overhead are outside this payload budget.
No payload or orphan reclamation runs at startup. `DiskStore` receives the
common `ResourceLedger`: index capacity, decoded token/input/component/chunk
vectors, raw manifest read buffers and validation working capacity are admitted
as metadata before allocation. Insufficient RAM admission fails startup and
releases every charge and the directory lock. Reservations include conservative
working capacity; accounting excludes allocator/control-block bookkeeping.

The binary manifest has this ordered encoding. Every integer is unsigned
little-endian, without padding; component layout versions remain separate from
the disk format version and compatibility identity:

| Field | Encoding |
| --- | --- |
| Magic, format | Eight bytes `GUFOMNF2`, u32 version 2 |
| Checkpoint, lineage | Two u64 IDs |
| Compatibility | 32-byte SHA-256 digest of the adapter's canonical identity, supplied by caller |
| Purpose, rank | u8 enum, u32 |
| Supplemental input | u32 byte count, exact identity bytes |
| Tokens | u32 count, exact u32 token IDs |
| Components | u32 count, records sorted by component ID |
| Component record | u32 ID, u32 layout version, u8 kind, u64 row bytes, u64 rows/chunk, u64 state bytes, u64 valid rows |
| Full chunks | u32 count, ordered payload references covering complete chunks from row zero |
| Tail, private state | Each: u8 presence (0/1), then reference if present |
| Payload reference | 16-byte file ID, u64 logical size, u64 CRC-64 |
| Manifest checksum | u64 CRC-64 over all preceding bytes |

CRC-64/ECMA-182 uses polynomial `0x42f0e1eba9ea3693`, initial value zero,
no reflection and no final xor. `DiskChecksum` supports incremental payload
verification for card 12; it detects accidental corruption, not hostile
modification. The format stores tokens inline: at 100k tokens they occupy
400,000 bytes, avoiding another dependency/file for lineage tokens. Payload
bytes never enter manifest serialization. Limits are 16 MiB per manifest,
2,097,152 tokens, 65,536 input bytes, 256 components and 65,536 total references.
Decoding checks counts against remaining bytes before allocation, geometry,
private-state boundary, canonical ordering, enum values and arithmetic overflow.

Startup validates manifest checksums, then stats dependency sizes without
opening payload files. Complete entries are durable with unverified payloads.
Conflicting shared-file lineage/range/layout/checksum declarations, private-file
aliases and duplicate checkpoint IDs exclude every conflicting entry, regardless
of directory enumeration order. Card 12 must validate payload checksums during
restore before execution; card 13 supplies lookup across tiers.

Legacy cleanup examines only root regular files: `.tmp-*` and `.kvc` files
whose first four bytes are `GUFO`. Unknown files, symlinks, hard links and all
unrelated directories remain untouched. Cleanup only removes bytes and allocates
no disk payload; after cleanup, excess v2 usage fails startup before new writes
can occur. A preexisting over-budget legacy store may therefore shrink to fit;
the constructor cannot retroactively bound bytes written by the old store.
Card 11 adds publication reservations and reclamation.

### Review follow-up

Round 1 found three edge cases, now covered by regressions: startup metadata
must use RAM admission, legacy magic reads must retry `EINTR` and short reads,
and a root symlink with `/` or `/.` must not bypass `O_NOFOLLOW`. Root path
normalization strips terminal separators/dot components without resolving
symlinks. Exact RAM peak boundaries and every reserve/convert failure point
verify complete rollback, lock release and removal of rejected-entry charges.
Round 2 caught lexical normalization of `..` across a symlink redirecting
ownership and cleanup away from the configured directory. Terminal-only stripping
now preserves the kernel's `..` resolution; a regression checks both the lock
location and which directory's legacy file is removed.
Round 3 found no actionable startup issues. Its nonblocking format observation
also led to explicit rejection of whole-component byte overflow, matching
`ExecutionHistory` geometry checks. The new regression failed before the guard
and passed afterward; impossible component geometry now fails encode/decode,
before startup dependency checks.

### Validation and step baseline

The focused `cache_disk_test` covers two owners in one process and a forked
second process, reopen after release, format round trips, every truncation and
single-byte corruption, checksum-valid malformed metadata, missing/wrong-sized
and symlink dependencies, conflicting references, exact disk-budget boundaries,
legacy cleanup and preservation of unknown contents. Linker wrappers count
actual `read`/`pread` bytes independently of store statistics: increasing a
private payload to a sparse 1 GiB changes no manifest reads and reads zero
payload bytes. Both ordinary and ASan/UBSan builds cover this test and the
cache package boundary test. The new test is part of the hosted PR target.

Reproduce using the pinned CPU dependency shell (CMake, Ninja, pkg-config,
Python and ICU/curl/PNG/JPEG/WebP/OpenSSL from `flake.lock`):

```sh
cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
cmake --build --preset cpu-test --target cache_disk_test cache_disk_bench --parallel 4
ctest --preset cpu-test -R '^cache_(disk|boundary)_test$' --output-on-failure
build/cpu-test/tests/cache/cache_disk_bench
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_disk_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(disk|boundary)_test$' --output-on-failure
```

Step baseline on Linux 7.2.9, AMD Ryzen AI MAX+ 395, flake.lock-pinned GCC
15.3.0/CMake 4.3.4/Python 3.14.6, starting from main `525d9c4d`:

| Measurement | Result |
| --- | ---: |
| Manifest size at exactly 100,000 tokens | 401,779 bytes |
| Initial startup, 100 manifests | 85.257 ms; 40,177,900 manifest bytes read; 5,000 dependency stats |
| Initial startup, 1,000 manifests | 849.831 ms; 401,779,000 manifest bytes read; 50,000 dependency stats |

After round 1's metadata admission fixes, a retained repeat recorded 85.155 ms
for 100 manifests and 842.178 ms for 1,000; manifest bytes and dependency stats
were identical to the initial measurements. These are separate observations,
not an averaged timing or a claim of a performance improvement.

One warm-cache run per size for each revision, immediately after writing fixtures
on `/tmp`.
Every checkpoint has 100k tokens, one row component (4 bytes/row, 2,048
rows/chunk) and a 16-byte private component. Full chunks are shared in one
lineage; each checkpoint has its own tail and private state. Sparse payloads
intentionally have unverified checksums. These figures measure CPU startup
and metadata I/O, not cold-disk restore, publication, inference or GPU quality.
The benchmark asserts indexed count and exact bytes read at both sizes.

## Done when

- [x] Tests above pass.

## Review focus

- Manifest encoding: binary with a checksum, or JSON? Tokens make manifests
  large at 100k+ tokens. Consider a separate tokens file per lineage.
- File count per checkpoint: the research measured about 13% overhead for
  chunk files against one large file.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) (USENIX ATC 2024): a host and disk hierarchy for multi-turn KV state.
- [LMCache hybrid model support](https://docs.lmcache.ai/mp/hybrid_models.html): storage backends for hybrid-model state.
- [llama.cpp server slot save and restore](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/README.md): a simple per-slot file format and its limits.

## RFC

[Disk representation and bounded transfers](../RFC.md#disk-representation-and-bounded-transfers) ·
[Publication, crash recovery and format upgrades](../RFC.md#publication-crash-recovery-and-format-upgrades)

## Review notes

