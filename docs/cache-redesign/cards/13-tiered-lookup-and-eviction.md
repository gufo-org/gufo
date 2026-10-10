# 13 · RAM + disk lookup and reference eviction

**Milestone:** Disk store · **Depends on:** 05, 11 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** done

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

- [x] Tests above pass.

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

Four independent fresh-context reviews completed. The final review of
`a592c0c3` against main `081e8a04` found no actionable issues and passed seven
focused normal checks plus three ASan/UBSan checks. After the final fix, six
affected checks also passed in both normal and sanitizer builds. PR:
[#517](https://github.com/gufo-org/gufo/pull/517).

The compressed per-compatibility token tree now admits resident, durable and
mixed component candidates. Persistent descriptions bind the caller's canonical
identity to its supplied SHA-256 digest, exact inventory/layout and independent
component positions. Deepest usable remains the first rule; equal-depth ties
prefer live, resident, mixed, then durable candidates. Distinct selection reasons
expose durable/mixed choices to the later serving logger. A description owns
only admitted manifest metadata. Lookup takes a publication-epoch pin, so an old
description cannot acquire a replacement checkpoint with the same ID. Lookup
candidate capacity and retained availability plans are admitted metadata too.

RAM retention removal drops its checkpoint owner while leaving a complete
durable description in the tree. The fake W2 test proves the former checkpoint
object expires, restores target 40 / draft 36 into another slot, checks every
component byte, and compares continued execution with the uninterrupted slot.
Mixed-source plans remain stable across availability changes. Payload validity
is checked for each selected RAM component; an invalid unused RAM component
can still be supplied from disk alongside healthy RAM components. Digest/layout
mismatches and all catalog reserve/convert fault points fail without leakage.

`DiskCatalog` keeps global LRU across all discovered disk identities, including
ones not loaded into lookup. New publications become most recent; callers touch
successful uses. Startup ties use ascending checkpoint ID. Eviction skips reader
pins and immediately refuses a busy writer. Manifest retirement and directory
sync still precede dependency deletion; shared files survive another manifest's
retirement. Only completed durable deletion contributes reusable disk budget.

A reverse dependency map quarantines every affected catalog/index record before
fallback, including its RAM lookup candidate. The disk layer also quarantines
untracked and provisional claims without waiting for the writer's I/O lock. A
writer rechecks quarantine before creating its manifest and before indexing;
late quarantine retains the inherited recovery gate until its unindexed manifest
has been durably reclaimed. Tests exercise invalidation during verification,
after dependency sync, and after the manifest barriers. Physical cleanup remains
caller-scheduled at an idle boundary after reader pins settle.

Reused-dependency verification during publication and restore quarantine at the
point of detection, raising a typed dependency error. An admitted atomic marker
shared by a publication's descriptions makes even its RAM-bound lookup record
unusable immediately, without worker mutation of the caller-serialized index.
`DiskCatalog::Reconcile` removes these records at a serialized caller boundary;
eviction and invalid-entry reclamation also reconcile automatically. Ordinary
retirement leaves coherent RAM copies usable. Conflicting caller claims are
rejected before verification and never quarantine a healthy publication.
The first independent review found the publication-verification and mixed RAM
validity gaps; regressions now cover synchronous and queued corruption plus
failed borrowed-target preservation with healthy draft/private RAM state.

These are common-cache and fake-adapter checks. Real model state and serving
integration remain the following cards; no model quality or HTTP speed claim is
made here.

The 13 common-cache checks pass in normal and ASan/UBSan builds. The shared
format check passes 554 C++ files. The production `BUILD_TESTING=OFF` benchmark
uses pinned GCC 15.3.0, ROCm 7.2.3 and the release preset on Ryzen AI MAX+ 395,
gfx1151, Linux 7.2.9, with encrypted Btrfs (`compress=zstd:3`). Run it with:

```sh
cmake --build --preset release --target cache_tiered_bench
build/release/src/cache/cache_tiered_bench /tmp
```

All samples are retained in [13-gfx1151.csv](../measurements/13-gfx1151.csv).
After ten warmups, 100 lookups through 1,000 matching durable boundaries,
including acquisition/release of every candidate's pin, had median **0.366598
ms**, range **0.331145–0.510994 ms** at `17703771`. Retiring one checkpoint
with a still-shared dependency took **0.694962 ms**; one with an unshared
dependency took **1.232860 ms**. Each crossed three directory sync barriers. The eviction catalog sizes
were 1,003 and 1,002, including the fixture's additional records; lookup retained
1,000 candidates. These are warmed metadata lookups and tiny synthetic 8/32-byte
payload files, not full-model disk-transfer timings. No speedup is claimed.

The second independent review found that replacing a quarantined publication
before catalog reconciliation could shed its marker from an old RAM record.
`Track` now honors both catalog and disk quarantine when replacing an epoch;
a direct retire/republish regression proves the old RAM record stays excluded.

The third completed fresh-context review found that durable insertion and
attachment allocated their internal availability lists before metadata
admission. Both now reserve the complete plan capacity before allocating the
object or vector, and convert only after construction succeeds. An allocation
observer regression fails on the reviewed source and proves denied admission
allocates no plan storage beyond the ledger's own bookkeeping; conversion
failure preserves ledger totals and prior lookup state in both paths.

The CSV also retains the original `aca631cd` and first-review-fix `59062b47`
baselines. These are separate step observations, without a matched performance
gain or regression claim. The `17703771` numbers above include immediate
quarantine-marker checks and caller-boundary reconciliation, and precede the
subsequent admission-order fix in durable insertion and attachment. That fix
does not change lookup or eviction code.
