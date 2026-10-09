# 04 · Chunks, checkpoints and provenance

**Milestone:** Common package · **Depends on:** 03 · **Size:** M–L ·
**Affects:** nothing at runtime · **Status:** agreed

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

## Done when

- [ ] Tests above pass.

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

