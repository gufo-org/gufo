# D4 · Options and budgets

**Type:** decision · **Blocks:** 08, 10, 19 · **Status:** done

## Question

The new cache needs memory the current one does not reserve (committed spill
backing), and the meaning of the staging budget changes. How do existing
options map, and do we add any?

## Current options

From `src/cli/serve/serve.cpp` and `docs/SERVER.md`:

| Option | Today |
| --- | --- |
| `--cache-ram-bytes N` | 0 (default) picks an automatic snapshot budget. |
| `--cache-disk DIR` | Enables restart-safe reuse. Several processes may share a directory. |
| `--cache-disk-bytes N` | Disk budget, 8 GiB by default. |
| `--cache-disk-staging-bytes N` | 0 picks an automatic size; limits the largest checkpoint that can be written or restored. |
| (internal) | 128 checkpoint records. |

## Proposal

1. **Spill backing comes out of `--cache-ram-bytes`.** One RAM budget; the
   ledger (card 03) splits it into committed backing, private state, tails and
   metadata. No new option. Backing is committed at startup. At the measured
   ~25 GB/s page-commit rate, 8 GiB adds roughly 0.34 s to startup (estimate).
2. **`--cache-disk-staging-bytes` keeps its name; its meaning narrows.** It
   bounds the transfer buffer only and no longer limits checkpoint size. Card 12
   selects a 1 MiB default from target-disk piece measurements; it is a
   transfer-piece budget, independent of checkpoint size. (Revised for #573:
   Flash-Next now ignores the option and always uses 1 MiB. Its size made no
   measured difference, and carried-over 4–6 GiB values starved hosts in #562.
   The legacy store of the other models still reads it.)
3. **`--cache-disk-bytes` keeps its meaning.** Referenced, temporary and orphan
   bytes all count against it.
4. **`--cache-disk DIR` becomes single-process.** A second process gets a clear
   ownership error at startup, not a silent fallback.
5. **The record limit stays an internal constant (128).** Private-state bytes
   already bound checkpoint density.

## Alternatives considered

- A separate `--cache-spill-bytes`: rejected, one more knob with no clear user
  decision behind it.
- Allocating spill backing lazily: rejected by the RFC because page commits
  block other HIP calls ([PR #445](https://github.com/gufo-org/gufo/pull/445)
  measured an ~8 ms decode stall).

## Also decide

- Is the automatic `--cache-ram-bytes` value still right when the memory is
  committed at startup instead of growing on demand? It becomes visible as
  resident memory immediately.
- Is the single-process directory change a breaking change for the release
  policy (`!` in the PR title)? Recommendation: yes, mark card 19 as breaking.

## Review notes

## Results (2026-10-12)

Done for Flash-Next, the only model the component cache serves on the card 19
branch; Qwen 27B and DeepSeek adopt it with cards 15–17 and 19. Qwen 27B and DeepSeek keep the legacy cache and its option
meanings until their adapters land (cards 15–17); `--help`, `docs/SERVER.md`,
`docs/KV-CACHE.md` and `docs/CLI.md` describe both.

1. **RAM.** Card 19 already commits the backing at startup: serving allocates
   the execution slots, observes the remaining host RAM, lowers the ledger
   limits, then commits the pool. The Flash-Next provider sizes 64 KiB pages
   plus 1 KiB of allocator metadata each from nine tenths of the budget after
   4 MiB, so about 89% of the budget is pinned at startup (14,350,483,456 of
   16,198,375,424 bytes in the smoke run below). The rest is ledger headroom
   for index, history, catalog and queue metadata, allocated as it grows. The
   startup line (`schema=component-cache-v1 event=configured`) now also
   reports the post-slot `automatic_bytes` and `max_bytes`, as the legacy line
   does, and `disk_capacity_bytes`.
2. **Staging.** The 1 MiB default came with card 12. Restores, and
   disk-only captures of rows observed in place, move whole rows per staging
   piece, so a budget below the largest append row (1,024 bytes on
   Flash-Next) could never restore a checkpoint: RAM checkpoints still
   published and used disk budget, but restores rejected them, and disk-only
   captures failed. Startup now rejects such a budget with the required
   size. The old advice to raise staging to the checkpoint size for 262K
   Flash-Next contexts is removed: the buffer is pinned host memory allocated
   at startup, so a large value only costs RAM.
3. **Disk bytes.** The store already counted referenced, temporary and orphan
   files, but refused to open a directory above the budget, while the legacy
   store evicts to fit. A lowered `--cache-disk-bytes` now opens: serving
   reclaims orphans and temporaries, then evicts the oldest checkpoints
   (by creation order) until the directory fits, and logs `event=disk_evicted
   reason=budget` (or `event=disk_over_budget` if it cannot fit). Startup
   evicts by checkpoint ID, assigned at creation, because the catalog does
   not persist use history; the legacy store orders by file modification time. Opening above budget
   made an unsigned underflow in the publication check reachable
   (`budget - managed`), which would have admitted any write; the check now
   refuses while the directory is above budget.
4. **Single process.** Card 10's `LOCK` file and `flock` apply. A second
   Flash-Next server fails at startup with `Error loading model '...': cache
   directory ownership unavailable (another process may own it): DIR` and
   exit code 1. The store now opens before the execution slots allocate and
   the backing commits, so a losing server no longer claims its RAM budget
   first. It still opens after the weights load. The lock does not wait, so a
   restarted server whose predecessor is still draining disk writes fails
   unless the predecessor finishes during the model load. Legacy-model
   servers do not take this lock. The ownership error is tested on CPU
   through `TextRunnerPool`; serving reports load errors this way, as the
   GPU staging check below shows. No GPU run started two Flash-Next
   servers: one uses about 98 GB of device memory, so a second likely cannot
   load its weights on the 128 GB target before reaching the lock.
5. **Records.** `RetentionPolicy::kRecordLimit` stays 128.

Decisions on the open questions:

- **Automatic RAM budget:** unchanged (half of the post-slot available RAM, at
  most 32 GiB). The old cache reached a similar footprint as checkpoints
  accumulated; committing at startup makes it visible earlier, which is safer
  for co-resident servers that size themselves at their own startup.
  `docs/KV-CACHE.md` says so next to the #556 advice to set an explicit
  budget. Co-resident image, ASR and TTS capacity stays an open resource gate
  in [todo.md](todo.md). The 0.34 s commit-time estimate above was not
  measured separately; the whole Flash-Next load took 21 s.
- **Breaking change:** yes. PR #543 merged into `fedeizzo/cache-staging`
  without `!`; the pull request that brings the branch into `main` must carry
  `!` (recorded in the todo's final row).

Checks: `component_text_cache_test` adds startup eviction under a lowered
budget, the one-row staging floor with a publish/restore round trip at
exactly one row, and directory ownership through `TextRunnerPool`, including
that the losing server commits no backing. `cache_disk_test` covers opening
above budget, refusing publication until retirement, and publishing again.
Each new assertion fails on the preceding code: startup throws `cache
directory exceeds disk budget`, and removing the staging check, the
over-budget refusal or the early store opening each fails its assertion. All
57 hosted contracts and the shared formatter pass. `cache_streaming_test`
checks that `MinimumStagingBytes` takes the widest of differently sized rows.
A further serving case plants an unreferenced chunk that puts the directory
over a budget the referenced checkpoints still fit: startup reclaims it and
both conversations still restore (removing the wait for reclamation fails
5 of 5 runs). The ownership test also checks that the losing server
allocates no slot. The `disk_over_budget` warning has no test: once
reclamation finishes, startup can always retire unpinned entries.

Production-build smoke runs (release `gufo serve llm`, one session, context
16384) check the startup paths on gfx1151. In AR, the first server published
two unrelated 4,741-token prompts as 251,663,008-byte checkpoints (503,415,538
managed bytes) and reported `capacity_bytes` and `automatic_bytes`
16,198,375,424 and `max_bytes` 28,101,783,552. A restart with
`--cache-disk-bytes 503415537` logged `event=disk_evicted reason=budget
entries=1 managed_bytes=251707769`; the newer prompt then restored 4,736 tokens
from disk and prefilled 5, and the evicted one prefilled all 4,741. A start
with `--cache-disk-staging-bytes 1` exited 1 with `--cache-disk-staging-bytes 1
is smaller than one 1024-byte cache row`. Model load took 21 s. These AR
runs used `8397d793`. On `fafd0ed5` (binary SHA-256 in the evidence directory), the same
MTP smoke run published 263,080,352-byte checkpoints, evicted one entry under a
budget one byte below the 526,253,734 managed bytes, restored 4,736 tokens and
prefilled 5 for the newer prompt, executed MTP draft rounds, and rejected
1-byte staging with the same message. The AR disk lifecycle runner
(`cache_lifecycle.py`, context 32768, one session, default 2 GiB staging)
passed restart, tiered, oversized (a 473,634,676-byte checkpoint through 1 MiB
staging), crash and side-traffic. The AR `cache` suite (`run.py
--record-baseline --suite cache`, two sessions, context 32768) passed all
eight cases, including the disk variants.
These are startup and correctness observations, not timing qualification;
the request path is unchanged.

Evidence: `/home/mixer/gufo-qualification/d4-options-2026-10-12` (`smoke.py`,
`run-gpu.sh`, server logs, `results.json`, candidate binary and SHA-256).

## Revision: pages on demand (2026-10-12)

Feedback on #562 showed the startup commit costing more than it saved. A
12 GiB budget left 5.2 GB of host RAM after load. Pinning it evicted the disk
checkpoints from the page cache, so the first restore after a restart read
them from disk again. The backing pool now commits its pages as captures need
them, up to the same budget, and keeps them. This reverses item 1's "committed
at startup" and the rejected alternative above, for these measured reasons:

- A segment commit on another thread did not stall a concurrent
  launch-and-synchronize loop at 64 MiB, 256 MiB or 1 GiB
  (`tools/bench/host_commit_stall.hip`). A 1 GiB commit took about 60 ms with
  free pages. In the server, captures run on the scheduler thread, so a step
  also pauses peers' decode for its length.
- When the kernel must reclaim page cache (the model files read at load), a
  step slows down. 256 MiB steps took 100–670 ms inside a busy request, and
  4.8 GB of growth cost a 157K-token turn about 3.5 s. Startup paid the same
  reclaim before: at a 17 GiB budget model load took 29–39 s, against 16.4 s
  now.
- Pinned registrations cannot be joined: a copy spanning two of them fails
  with `invalid argument`. A payload therefore stays inside one segment, and
  every segment edge strands space a ~140 MB private state cannot use. With
  256 MiB segments that changed retention at a full 8 GiB pool. A retry copy
  was refused that the single pool had kept, and two cases prefilled 7 more
  tokens. Segments are now equal and about 1 GiB, at least two private
  states: an 8 GiB budget grows in eight steps.
- Each capture that leaves less than one segment free starts the next
  segment on a background thread, so a capture only pays for growth when it
  outruns that thread. Before, steps 14 and 15 of a #573 run took 945 ms and
  1,752 ms on the capturing thread and delayed peers through
  `shared_prefix_wait_ms`.
- Pinned host pages count against the device heap: 8 GiB of `hipHostMalloc`
  lowered `hipMemGetInfo` free memory by 8.00 GiB, exactly as `hipMalloc`
  did. Growth therefore also stops while a segment would leave the heap less
  than #460's 4 GiB dispatch reserve (`reason=device_heap_reserve`), so it
  cannot push later session and scratch allocations onto managed memory on a
  host with the default half-RAM `ttm.pages_limit`.
- Growth stops early while the host would keep less than 4 GiB available
  (`event=backing_limited`). A capture that finds no room in a growing pool
  reads the ledger's free backing first, and skips a scan of every committed
  block that cannot succeed.

Disk restores also hint the next files from four threads, up to 32 MiB ahead.
Before, a single thread hinted one file at a time. A 157K-token checkpoint is
4.45 GB in 18,672 files. Restoring it took 8.7–9.9 s cold before; it now takes
1.3–1.4 s cold and 0.85 s from the page cache.

Results with both changes (1 session, context 262144, 157K-token
conversation, restarted between turns):

| | Before | After |
| --- | --- | --- |
| Host RAM available after load, 17 GiB budget | 6.1 GB | 21.9 GB |
| Model load, 17 GiB budget | 29–39 s | 16.4 s |
| First deep turn after restart, 17 GiB budget | 9.2–10.7 s | 1.7–5.5 s |
| First deep turn after restart, 2 GiB budget, files on disk | 9.1 s | 1.7 s |

The card 21 six-suite history in MTP at 8 GiB passes; the base is
`edd0d282`. One cache-work change remains, and it is an improvement: a
full-pool retry copy that the base refused now fits, so
`cache_edit_rewind_unchanged` prefills 0 tokens instead of 7. An
unchanged-base control run showed no cache-work changes. Per-suite totals
match the control except `cache-growth` capture time, +240 ms for the eight
growth steps. A first base run was faster across every suite (run-order drift).

Evidence: `/home/mixer/gufo-qualification/issue562-2026-10-12`
(`restore_probe.py`, `prefetch_bench.py`, `parallel_prefetch_bench.py`,
`lazy3_qual.sh`, `control_check.py`, reports, logs, binaries and SHA-256).
