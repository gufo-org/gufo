# 11 · Crash-safe publication and orphan recovery

**Milestone:** Disk store · **Depends on:** 10 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** agreed

## Goal

Publish checkpoints so that a crash at any point exposes only complete,
durable checkpoints, and reclaim whatever a crash leaves behind.

## Scope

- **Publication order:**
  1. write missing chunks and private files to `tmp/`;
  2. verify checksums, `fsync` the files, `fsync` their directories;
  3. write the manifest, `fsync` it, `rename` it into `manifests/`, `fsync`
     the directory;
  4. mark the checkpoint durable in the index and release writer
     reservations.
- **Retirement by references:** remove the manifest first, then files no
  remaining manifest references.
- **Orphan reclamation:** only after full manifest discovery, in the
  background, outside the startup path.
- **Pins:** writers and readers pin dependencies so eviction cannot remove a
  file in use.
- **Test-only crash hook** at each publication step.

## Not in this PR

Streaming payload transfer (card 12); this card can write from host buffers
in tests.

## Test first

For each publication step, crash (abort the child process) and restart:

- only complete checkpoints are visible;
- no manifest references a missing file;
- orphans are reclaimed in the background;
- concurrent eviction never removes a pinned file.

## Step baseline

Publication latency per checkpoint on the target disk: number of `fsync`
calls and their total time, for a small and a 100k-token checkpoint.

## Done when

- [ ] Crash tests pass for every step.

## Review focus

- The `fsync` ordering, especially directory `fsync` after `rename`.
- Cost: how many `fsync` calls per checkpoint, and can they be batched safely?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [All File Systems Are Not Created Equal: On the Complexity of Crafting Crash-Consistent Applications](https://research.cs.wisc.edu/adsl/Publications/alice-osdi14.html) (OSDI 2014): which `fsync`, `rename` and directory orderings actually survive a crash.
- [Ensuring data reaches disk](https://lwn.net/Articles/457667/) (LWN): every buffering layer between the application and stable storage.

## RFC

[Publication, crash recovery and format upgrades](../RFC.md#publication-crash-recovery-and-format-upgrades)

## Review notes

