# 11 · Crash-safe publication and orphan recovery

**Milestone:** Disk store · **Depends on:** 10 · **Size:** M ·
**Affects:** nothing at runtime until card 19 · **Status:** done

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

### Implementation record

The standalone [`DiskStore`](../../../src/cache/disk.hpp) now accepts a
manifest and caller-owned immutable host buffers. Serving does not instantiate
it until card 19. Publication reserves the complete additional logical disk
size before writing, and admits encoding, decoded index metadata, reference
validation, vector growth and verification buffers through `ResourceLedger`.
A store mutex serializes publication, retirement and recovery: an active writer
holds its dependencies until its manifest is durably indexed. Reader pin tokens
block retirement of their checkpoint and therefore protect every dependency.
`Entries()` is a quiescent view; callers must not hold its span across mutations.
Pin acquisition, retirement and statistics are thread-safe; recovery scheduling
and joining are caller-ordered.

Missing payloads are checksum-verified, written exclusively into `tmp/`, synced
individually and moved with Linux `renameat2(RENAME_NOREPLACE)` into their final
namespaces. Existing referenced chunks are independently checksum-verified
using a 64 KiB admitted buffer. Unknown destinations and unreferenced existing
files cannot be overwritten or adopted. After all payload moves, `tmp/`,
`chunks/` and `private/` are synced once each. Only then is the checksummed
manifest written, synced, moved into `manifests/`, and its destination directory
synced. A final `tmp/` sync durably removes the source entry before the index
commits. On failure, actual written bytes remain charged as orphans; an uncertain
manifest rename may be discovered as complete on restart even when the caller
received an error. After a manifest rename attempt fails or a later step fails,
further publication and retirement are blocked until background recovery
durably removes unindexed manifests. This protects their shared references and
prevents duplicate-checkpoint retries. Startup sets the same barrier when it
rejects managed manifests, so new payload files cannot revive their old claims
before recovery.

Layout creation syncs `v2/`, the root and its ancestors. Startup also syncs
`manifests/` after full validation so a surviving rename from an aborted process
becomes durable before its entry is exposed. Neither operation reads payloads
or reclaims orphans. Retirement unlinks and syncs the manifest directory before
removing dependencies no remaining entry references. If that sync fails, the
entry stops being exposed and dependencies remain for later recovery. Uncertain
manifest unlink/sync outcomes also block publication and retirement until
successful background recovery.

`ReclaimOrphans()` explicitly launches a worker after full discovery;
`WaitForReclamation()` delivers errors, and destruction joins before releasing
state or directory ownership. The worker removes rejected/unindexed manifests
and syncs their directory before deleting orphan payloads or temporary files.
Only managed, regular, singly linked files are removed; unknown files,
directories, symlinks and hard links are preserved. Failed reclamation can be
retried. Test hooks are compiled only with `BUILD_TESTING`; the production
library has no crash hook.

### Validation and step baseline

`cache_disk_publication_test` aborts forked children at all twelve publication
boundaries, including each occurrence of the three per-payload boundaries:
18 actual abort cases and 18 successful controls. Reopening verifies every
visible payload's length and checksum, then runs background reclamation and
checks exact byte accounting. Independent linker wrappers observe actual
`fsync` descriptors and manifest-first `unlinkat` ordering, inject every
publication sync failure plus retirement sync failure, and exercise `EINTR`,
short writes and partial `ENOSPC` writes. Tests also cover reader pins, writer
versus eviction concurrency, shared references, conflicting publication,
checksum rejection, disk budget admission, metadata reserve/convert failures,
and preservation of unknown/symlink/hard-link contents.

The focused disk, publication and package-boundary tests pass in ordinary and
ASan/UBSan builds with pinned GCC 15.3.0 and CMake 4.3.4. A separate
`BUILD_TESTING=OFF` CPU build of `gufo_cache` validates the production path.
These process-abort and syscall-order checks do not simulate a physical power
failure or filesystem-specific storage failures.

Reproduce in the pinned CPU dependency shell described by card 10:

```sh
cmake --preset cpu-test -DGUFO_BUILD_TOOLS=ON
cmake --build --preset cpu-test --target cache_disk_test cache_disk_publication_test cache_disk_publication_bench --parallel 4
ctest --preset cpu-test -R '^cache_(disk|disk_publication|boundary)_test$' --output-on-failure
build/cpu-test/tests/cache/cache_disk_publication_bench /tmp
cmake --preset cpu-sanitizer
cmake --build --preset cpu-sanitizer --target cache_disk_test cache_disk_publication_test --parallel 4
ctest --preset cpu-sanitizer -R '^cache_(disk|disk_publication|boundary)_test$' --output-on-failure
```

Step baseline on Linux 7.2.9, AMD Ryzen AI MAX+ 395, Btrfs on
`/dev/mapper/cryptroot`, using the pinned CPU toolchain above, from main
`9024101b`. Each observation publishes one fresh checkpoint into a fresh `/tmp`
directory with one append-row component (4 bytes/row, 2,048 rows/chunk), its
private tail and a 16-byte private-state component. Buffers and tokens are built
before timing; layout creation is excluded. These are host-buffer store
measurements, not model payload sizes, inference performance or cold restores.

| Checkpoint | Logical bytes | `fsync` calls | Initial publication / sync time | Retained repeat publication / sync time |
| --- | ---: | ---: | ---: | ---: |
| 128 tokens | 1,283 | 8 | 3.499 / 3.329 ms | 3.600 / 3.488 ms |
| 100,000 tokens | 801,795 | 56 | 36.120 / 32.219 ms | 36.041 / 31.871 ms |

Times are separate observations, never averaged. The repeat includes the
metadata admission and additional crash-boundary refinements. Successful
publication uses one sync per new payload, one manifest sync and five directory
syncs: `new_payload_count + 6`. Directory syncs are batched after payload moves;
file syncs remain individual so dependency contents are durable before the
manifest can reference them. Group commit across checkpoints is deferred.

## Done when

- [x] Crash tests pass for every step.

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

Round 1 found that a failed publication could leave a valid, unindexed manifest:
retirement of an indexed checkpoint then deleted chunks the failed manifest
shared, and retrying the same checkpoint with fresh file IDs created duplicate
manifests that startup rejected. Publication now sets a recovery barrier after
any failed manifest rename attempt or subsequent step; publication and retirement
remain blocked until successful background reclamation, including a durable
manifest-directory sync. Regressions cover both failures and a failed recovery
sync before retry. An additional accounting audit corrected index growth so the
old vector allocation is freed before releasing its capacity charge. CI also
caught a post-check comment wrap, now corrected and rechecked.


Round 2 reproduced a restart variant: startup rejected a checksum-valid manifest
with a missing chunk, but a subsequent publication filled that missing file and
created a second manifest for the same checkpoint. The next restart rejected
both. Startup now sets the same recovery barrier whenever it rejects a managed
manifest. The regression proves publication is blocked until background recovery,
then verifies a successful publication survives reopening with one entry.

Round 3 modeled an allowed power-loss state after retirement's manifest-directory
sync failed: a retry could publish a new manifest while the old unlink remained
nondurable, allowing both checkpoint identities to appear at restart. Retirement
now sets the recovery barrier on manifest unlink or sync errors. Regressions
cover both errors, a failed recovery sync, blocked publication until recovery,
and successful publication afterward. The review's replay probe models the old
unlink returning after a crash; it is not a physical power-loss simulation.

Round 4 identified an uncertain unlink that takes effect but returns an error:
the old entry remained indexed, so recovery treated its missing manifest as
retained and cleared the barrier. Retirement now removes the index claim before
attempting unlink. Dependencies stay protected until the manifest-directory sync
succeeds; failed operations leave them for background recovery. A linker wrapper
models both an error before unlink and an applied unlink followed by EIO, checking
that neither exposes the old entry, recovery reconciles every byte, and a new
publication succeeds. The applied-unlink case failed against the prior revision.

The final audit found a disk-budget variant after payload-directory sync failure:
released bytes could fund another partial publication before the old deletions
were durable, and a power-loss replay could resurrect orphans above budget.
Every retirement failure now sets the recovery barrier, and reclamation sets it
before any cleanup and clears it only after all directory syncs and accounting
reconciliation succeed. Regressions inject all three retirement sync failures
and all four reclamation sync failures, checking fresh-ID writes stay blocked
until successful recovery. The audit used a modeled crash namespace, not a
physical power-loss test.

The follow-up audit reproduced the same deletion-credit risk across a process
restart: startup counted the current payload namespace but synced only manifests.
Startup now syncs all four managed directories before returning, making surviving
renames and deletions durable before any freed capacity can be reused. The test
restarts after every retirement sync failure, independently observes all four
startup directory syncs, and rejects construction when any of those syncs fails.
The review probe modeled 2,048 surviving bytes at a 1,192-byte budget; it did not
claim physical power-loss testing.
