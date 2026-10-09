# 13 · RAM + disk lookup and reference eviction

**Milestone:** Disk store · **Depends on:** 05, 11 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** agreed

## Goal

Consider RAM and disk checkpoints together in one lookup, and evict across
tiers by references, so a short RAM hit never hides a deeper disk checkpoint.

## Scope

- Use the per-component availability added in card 05: resident, durable or
  both.
- Default selection stays "deepest usable". A deeper durable checkpoint beats
  a shorter resident one, and the reason is logged.
- **RAM eviction** of a checkpoint that is also durable keeps it as a disk
  candidate.
- **Disk eviction** under `--cache-disk-bytes` removes manifests, then files
  nothing references.
- **Corruption:** a failed dependency invalidates every checkpoint that
  references it (reverse dependencies). Lookup falls back to an earlier
  complete checkpoint or a cold prefill.

## Not in this PR

A cost-aware rule that prefers a shorter RAM hit for latency. It is an RFC
open question and needs its own measured card.

## Test first

With the fake adapter:

- W2 shape: a short RAM checkpoint beside a deeper disk one selects the disk
  one;
- a corrupt shared chunk invalidates all its dependents and the next lookup
  falls back correctly;
- disk eviction never removes a file another manifest still references.

## Step baseline

Lookup time with 1,000 durable checkpoints, and disk eviction time for one
checkpoint with shared and with unshared dependencies.

## Done when

- [ ] Tests above pass.

## Review focus

- Disk eviction order. Today's disk tier is global LRU; choose the order for
  the new store explicitly here, or keep LRU and leave changes to a separate
  measured policy card.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [SGLang HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design): one radix tree across device, host and storage tiers.
- [Unified Radix Cache: one tree for hybrid model prefix caching](https://www.lmsys.org/blog/2026-08-11-unified-radix-cache) (SGLang): component availability attached to tree nodes.
- [Stateful Large Language Model Serving with Pensieve](https://arxiv.org/abs/2312.05516) (EuroSys 2025): multi-tier conversation state and its eviction.

## RFC

[Lookup, slot selection and restore](../RFC.md#lookup-slot-selection-and-restore) ·
[Checkpoint selection and eviction](../RFC.md#checkpoint-selection-and-eviction)

## Review notes

