# 12 · Bounded streaming writes and restores

**Milestone:** Disk store · **Depends on:** 08, 11 · **Size:** L (split
candidate: write path, restore path) · **Affects:** nothing at runtime until
card 19 · **Status:** done

## Goal

Move checkpoints between device and disk in bounded pieces in both
directions, so no checkpoint is too large to persist or restore, and verify
payloads during restore before the slot can execute.

## Scope

- **Write:** device → staging piece → file, on leased streams (card 08). No
  path builds a whole payload in one host buffer.
- **Restore:** file → staging piece → device. Checksum each dependency while
  streaming. The destination becomes executable only after every checksum and
  position check passes; any failure invalidates it before a fallback runs.
- **Verification cache:** a verified immutable file stays verified for its
  file identity (inode, size, modification time).
- **Persistence queue:**
  - bounded depth and pinned bytes;
  - redundant jobs are coalesced or skipped;
  - optional writes yield to model work.
- Check whether the unexplained 782 s write stall recorded in the RFC research
  can recur on this path: record allocator, stream, metadata-lock, filesystem
  and `fsync` waits separately.

## Not in this PR

Selecting disk candidates in lookup (card 13).

## Test first

With the fake adapter:

- a checkpoint ten times the staging budget round-trips;
- peak staging never exceeds the budget;
- one corrupt byte invalidates the destination and triggers a fallback;
- a cancelled restore releases pins and leaves no executable partial state.

## Step baseline

- Write and restore throughput by piece size on the target disk; the default
  staging size (D4) is chosen from this.
- Restore time for a Flash-Next-sized 100k-token checkpoint.
- Peak staging during both.

## Done when

- [x] Tests above pass.

## Review focus

- Interference of disk I/O with decoding peers: card 19 measures it, but the
  queue design decides it.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Cost-Efficient LLM Serving for Multi-turn Conversations with CachedAttention](https://www.usenix.org/conference/atc24/presentation/gao-bin-cost) (USENIX ATC 2024): layer-wise preloading and asynchronous saving that hide I/O behind computation.
- [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400) (NeurIPS 2025): background prefetch fully overlapped with generation.
- [SGLang HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design): storage I/O and prefetch policies.
- [LMCache hybrid model support](https://docs.lmcache.ai/mp/hybrid_models.html): streaming state to and from a storage backend.

## RFC

[Disk representation and bounded transfers](../RFC.md#disk-representation-and-bounded-transfers)

## Review notes

Four independent fresh-context review rounds completed. The final review of
`8b1e29cf` found no actionable issues and passed six focused CPU checks and
three sanitizer checks. Draft PR: [#515](https://github.com/gufo-org/gufo/pull/515).

Fresh reviews found and fixed cancellation and shutdown issues. Interrupted staging
waits now retain their elapsed timing. Streamed publication checks cancellation
while acquiring the disk I/O lock, so stopping a queue does not wait for another
writer that is indefinitely yielding. Regression tests hold the staging piece
and the publication lock separately, then cancel the competing operation.
Concurrent queue stops serialize their worker join without holding the queue
state mutex. A regression holds asynchronous completion settlement while two
callers stop the same queue, then confirms both return and release all pins.
The retained transfer samples below precede this cancellation-aware lock change;
they do not qualify concurrent queue shutdown or serving interference.


## Implementation

`StreamedStore` owns one startup-admitted, committed staging piece. The HIP
allocator uses coherent pinned storage and the card 08 stream pool supplies
precreated stream/event leases. Each completion settles before a piece is reused;
row pieces preserve component alignment and private state supports byte ranges.
Persistence sources retain immutable committed checkpoint owners, not live slot
readers. Borrowed rows must be preserved before constructing a source.

Publication streams through card 11's transaction and computes CRC without a
whole-payload buffer. Reads retain an owned metadata snapshot and dependency pin,
stream into a non-executable destination, check each CRC and stable file identity,
and call adapter validation only after all components settle. Any error drains
loads and invalidates before fallback; cancellation invalidates without fallback.
The caller supplies the SHA-256 compatibility digest for its current adapter;
layout checks and independent target/draft positions remain mandatory.

The verification cache includes device, inode, size, nanosecond mtime and ctime,
and expected CRC. It is bounded to 1,024 admitted metadata nodes; cache admission
failure skips memoization. A cache hit still reads every byte and checks the file
identity again after copying. Retirement is blocked by snapshot pins.

One optional persistence worker bounds active plus pending depth and retained
allocation capacities. Admission/coalescing precedes pinning; duplicate logical
checkpoints are skipped. Optional writes check model idleness between pieces,
without holding live-source readers, a stream, staging, or the disk metadata
lock. Staging is leased through each complete read/CRC/write piece, so a
foreground restore can finish while an optional writer waits for model idleness. Source storage and
callback metadata are caller-admitted before submission. Queue stop cancels,
drains and joins. Filesystem mutations remain serialized, while index snapshots,
pins and accounting queries use separate short locks.

Timing counters separate initialization allocation, request metadata allocation,
stream acquisition, completion waits, index-lock waits, staging/I/O serialization,
filesystem calls, CRC, directory/file `fsync`, and optional yielding. Partial
attempts retain observations even when a transfer throws.

## Validation and measurements

The fake adapter round-trips 336 bytes with a ten-byte staging allocation, with
unaligned row-sized pieces and split private state. It verifies target/draft
positions and continued recurrent hashes, corrupts an already memoized file and
checks cold invalidation before fallback, and cancels after partial loads without
leaking a disk pin. Queue tests prove active/pending depth and 672-byte pin bounds,
coalescing, skipping, yielding without blocking index/accounting queries, and
stop cleanup. A concurrent restore completes while an optional writer yields,
which would deadlock if the writer held staging through its transaction. Admission tests inject every staging reserve/convert and source pin
failure, and verification-cache admission/commit faults skip memoization; physical staging is freed while its charge is still live. Reused chunks
also publish with no new sources and no hidden verification buffer.

Normal and ASan/UBSan checks cover the changed common cache contract. The pinned
production CMake configuration (`BUILD_TESTING=OFF`, tools enabled) builds the
standalone HIP transfer benchmark with GCC 15.3.0, ROCm 7.2.3 and HIP Clang 22.0.0.
The target is Ryzen AI MAX+ 395, gfx1151, Linux 7.2.9 and encrypted Btrfs with
`compress=zstd:3`. The benchmark writes deterministic noncompressible device
bytes, hints file-cache eviction with `POSIX_FADV_DONTNEED`, restores to another
device allocation, and independently checks every restored byte after timing.
The hint does not prove physical cold reads. Source/destination commitment and
staging initialization are outside transfer timing. This is a synthetic transfer
fixture, not model qualification or serving throughput.

All samples are retained in [12-gfx1151.csv](../measurements/12-gfx1151.csv).
Two initial 256 MiB runs per piece size gave:

| Piece | Write ms | Advised restore ms | Peak staging |
| --- | --- | --- | --- |
| 256 KiB | 927.865 / 925.471 | 762.862 / 768.317 | 256 KiB |
| 1 MiB | 889.774 / 898.311 | 735.972 / 715.577 | 1 MiB |
| 4 MiB | 890.014 / 889.907 | 719.268 / 684.201 | 4 MiB |
| 16 MiB | 891.041 / 892.328 | 722.026 / 725.014 | 16 MiB |

The 1 MiB default starts the write-throughput plateau while keeping memory and
between-piece yield intervals smaller. These initial samples precede the final
short accounting lock and exception-observation plumbing; they are retained as
piece-selection evidence. A subsequent synthetic Flash-Next MTP 100k-sized
**2,865,327,944-byte** fixture with that plumbing took **9,255.995 ms** to publish
and **7,656.989 ms** to restore, at a **1 MiB** peak. This approximates the RFC's
113.8 MiB + 27.46 decimal KB/token payload, represented as one private file to
exercise a payload much larger than staging. It does not simulate the model's
component layout. Publication spent 4,478.579 ms on CRC, 4,473.257 ms on `fsync`,
and 151.289 ms waiting for D2H; restore spent 7,179.146 ms on CRC, 157.016 ms in
file reads/stats and 249.701 ms waiting for H2D. Later results with piece leases and build overlap are also retained: the 100k
write took 5,254.727 ms with 224.853 ms of `fsync`, and restore took 7,726.329 ms.
The large `fsync` variation is visibly inconclusive as a performance comparison;
no speedup is claimed. The subsequent isolated final 100k run took
10,789.963 ms to publish (6,008.299 ms `fsync`, 4,482.152 ms CRC) and
7,683.856 ms to restore (7,220.908 ms CRC, 152.175 ms filesystem,
243.297 ms H2D), again at 1 MiB peak staging. Its 256 MiB control took
894.190 / 702.942 ms for publish/restore. Publication timing remains
unqualified because of filesystem sync variance; the byte round trips pass.
No 782-second stall occurred in this bounded fixture; decode-peer interference remains card 19's measurement.
