# 12 · Bounded streaming writes and restores

**Milestone:** Disk store · **Depends on:** 08, 11 · **Size:** L (split
candidate: write path, restore path) · **Affects:** nothing at runtime until
card 19 · **Status:** agreed

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

- [ ] Tests above pass.

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

