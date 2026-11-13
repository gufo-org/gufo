# Windows UMA cache follow-up, November 13, 2026

This follow-up validates the device checkpoint change after merging main
`6858788b7cbd20519b5bb7ab16706f244f3fabbd`, including concurrent disk sync
(#1205) and request admission (#1207). It supersedes the November 12 timing
baseline for this revision. Earlier observations remain historical evidence.
The [evidence](artifacts/uma-cache-windows-20261113.json) retains binary/source
identities, request records, timing qualification and earlier-attempt hashes.

## Revisions and isolation

The final code revision is `f5e9bad112c014bf74919233c8afe705fefe43f3`, packaged
as `uma-main-13c`. Source hashes cover 1,148 files. Official controls and the
candidate use TheRock 10 / Clang 23, RelWithDebInfo, the same dependencies,
and the same pre-existing Windows Flash-Next compiler-flag overlay. That
overlay is excluded from the PR. HTTP comparison arms use separate runtime
packages, a private loopback port, and fresh cache directories on the secondary
KIOXIA NTFS NVMe drive. Durability readers deliberately reuse published caches.

The existing profiles are Qwen3.8-27B UD-Q8_K_XL with Q4_K_M DFlash2 and
Flash-Next UD-Q4_K_XL with shared Q8_0 MTP. Both retain the 262144-token context,
one session, and 10 GiB disk cache. Automatic RAM budgeting is tested separately
from explicit fixed-budget controls. The production binary, launcher settings,
cache, model files and rollback backup remain unchanged.

## Upstream disk sync result

The pure disk-fix control compares `eabda7acffe94a9c07b9285a61e25a700f9ee997`
with its child `70bd0fdaf313de517c641d6698e38fc32ef0651c`. The production-source
diff is only `src/cache/disk.cpp`. Each arm uses exactly 6,000,000,000 host
bytes and the same 98,375-token request. Order is after, before, before, after.
Output hashes, token counts and actual draft work match across all four runs.

| Observation | Before first replay (s) | After first replay (s) | Before publication wait (s) | After publication wait (s) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 26.674 | 20.908 | 22.626 | 16.764 |
| 2 | 27.115 | 20.257 | 22.908 | 16.209 |

First replay improves by 22–25% in these observations; publication wait falls
by 26–29%. Cold requests remain 281.6–283.2 seconds. Subsequent disk replays
range from 3.387–3.469 seconds before to 3.435–3.926 seconds after, so this
does not establish faster disk restore. The fix changes `fsync_ms` from a
sequential sum to batch wall time; the comparison uses end-to-end request and
publication-wait timings instead.

Latest main `6858788b` separately takes 21.965 seconds for the first replay
and 3.778 seconds for the next under the same fixed budget. An automatic-budget
control grants 7,685,793,792 host bytes but still waits: its first replay takes
22.471 seconds. A nominal budget above one payload's size does not guarantee
capture headroom for retained checkpoints and readers. These tests measure
request latency, not desktop responsiveness or a disk-bandwidth limit.

## Device backing on latest main

The final candidate and main `6858788b` use the same 6,000,000,000-byte host
budget and fresh caches. The request, seven output tokens, output hash and
actual DFlash2 work match: 14 proposed tokens, five accepted, two rounds.

| Request | Main wall (s) | Candidate wall (s) | Cached tokens, main / candidate |
| --- | ---: | ---: | ---: |
| Cold | 283.913 | 283.853 | 0 / 0 |
| First replay | 21.965 | 1.712 | 98368 / 98375 |
| Next replay | 3.778 | 0.787 | 98368 / 98375 |

The candidate commits 3,892,314,112 device bytes (3.625 GiB). Both replays
prefill zero tokens and avoid disk restore. Overwriting the live execution
slot still permits an exact resident replay in 0.771 seconds. The first
replay is 12.8 times faster in this observation, or 92.2% lower wall time;
this is not an estimate across devices or memory partitions.

After the selected capture IDs finish publication and their manifests exist,
the candidate restarts and restores the disk checkpoint with the same output.
An unchanged main `6858788b` binary then reads that same cache successfully.
Both disk readers reuse 98,368 tokens and prefill seven; the restarted candidate's
next resident replay reuses all 98,375. This verifies format compatibility;
restart observations are not matched speed comparisons.

Use the [98K request recipe](UMA-CACHE.md#reproduce-the-98k-request) with
`--cache-ram-bytes 6000000000` and a fresh cache directory per arm to reproduce
the fixed-budget comparison. Run each binary separately with the same weights,
dependencies, context, sampler and speculative mode.

## Short-history regression and fix

Validation found a real regression after the initial merge: constructing an
unused 8 GiB device pool immediately charged 16 MiB plus 5 KiB of metadata to
the host budget. Under the fixed 5 GiB Qwen control, a 3,530-token unchanged
history then retained only 3,525 tokens and prefilled the remaining five.
Main retained all 3,530. Matching output alone would have missed this change.

The provider now initializes device metadata only when a row needs device
backing. A host quota rejection falls back to host allocation; an actual
allocator failure still propagates. The regression test fails on the earlier
implementation and passes with the fix, including a quota that fits only the
host pool. The real-model replay retains all 3,530 tokens with zero prefill
in both final-candidate repeats. Long-context coverage then initializes the
device pool after short-history pressure and restores all 98,375 tokens with
zero prefill and no disk restore.

## Linux CI fixture follow-up

The checks for PR head `35055b5c` exposed two test-isolation failures.
[Follow-up evidence](artifacts/uma-cache-linux-ci-20261113.json) retains both
reproductions and the unsuccessful order-only experiment. These fixes change
tests and their runner; the measured production runtime remains `f5e9bad1`.

The CPU restart test withheld every background transfer stream to delay device
persistence. That also denied streams to frontier capture and idle preservation.
An isolated Linux reproduction failed the same device-copy/persistence-read
assertion. The fixture now gates only reads from preserved device backing,
leaving preservation streams available and keeping all original assertions.
The complete component test passes 20 consecutive Linux runs (120.99 s total)
and one Windows run (14.53 s).

The Linux Q4 disk-spacing check restored 3,320 of 6,576 tokens and prefilled
3,256, above its unchanged 2,560-token bound. A native Windows Q4 reproduction
also failed. Manifest inventories show its 6,278-token checkpoint survived
publication and restart, then unrelated restore fixtures evicted it from their
shared 8 GiB cache. Merely replaying spacing first displaced other fixtures.
The runner therefore gives spacing its own 8 GiB cache and startup/restart
pair, releasing the earlier cache first. Capacity, spacing, output and prefill
assertions are unchanged. A bounded-cache counterexample fails the old runner
and passes the isolated version.

The fixed runner passes all 14 cache subchecks on Q4/DFlash2 and on the existing
Q8/DFlash2 and Flash-Next/MTP model/generation presets (42 subchecks). Q4 restores
6,278 tokens with 298 prefill tokens and the exact output. Additional loader
time for the isolated pair is 22.891 s, 30.375 s and 53.641 s respectively on
this Windows host. Separate logs preserve request timing attribution and verify
the loaded mode in all four processes. Runner tests pass 90/90 on Linux and
Windows; a deterministic fake clock avoids zero-duration Windows mock startup.
These are test overhead observations, not a new engine-speed comparison.

Local Linux checks use an isolated Ubuntu 24.04/GCC 13.3 container with the
actual serving/cache test sources and assertions. Real-model replays use native
Windows gfx1151; official Linux model CI remains the Linux GPU validation gate.

## Final qualification

The final binary passes all 12 functional suites (332 named checks):
`conversation`, `cache-growth` and `long-context` for both existing speculative
profiles and both serial controls. Speculative functional runs use fixed
5 GiB host memory for Qwen and 6,900,056,064 bytes for Flash to match the
timing controls; serial runs retain automatic host budgeting.

Warmed 2K/8K inputs with 256 output tokens retain automatic budgeting in the
existing speculative profiles. All outputs and actual draft work match main.
Qwen's four prefill and four decode observations differ by less than 0.3%.
Flash's automatic-budget run flags three prefill observations; its GPU starts
7–11 degrees C warmer than the baseline, and automatic host grants differ.
These observations remain unqualified, without assigning a cause.

Two additional final-candidate Flash runs use the original fixed
6,900,056,064-byte budget and bounded cooldown protocol. They reuse the two
immutable `main-6858788b` controls with matching toolchain, harness, models,
settings and request/cache history. Across all observed ranges, five phase
groups are within the margin and three remain inconclusive: both 2K prefills
and the first 8K decode. There is no confirmed regression. The two numerical
smoke checks reproduce latest main's reported values exactly.

Fixed-budget conversation/cache-growth comparisons keep the upstream 5% and
10 ms limits for every request and phase. A third matched request-prefix
control retains all preceding requests and both unchanged-main controls.
Qwen has 921 measurements within the margin and 120 inconclusive; Flash-Next
has 773 within the margin and 268 inconclusive. Neither has a confirmed timing
regression or output/prefill-work mismatch. Inconclusive is not a performance
pass or evidence that a slowdown is impossible.

Qwen's initially flagged queue, snapshot and cancelled-request prefill ranges
overlap after the third control. Flash's
`cache_growth_messages_keep_thinking_turn_2` snapshot remains an advisory:
all three matched pairs are slower, but unchanged-main variation and the
observed ranges overlap. Both arms prefill 199 tokens and capture the same
413,763,356 bytes in three captures. The evidence does not establish a cause.

| Flash snapshot observation | Main (ms) | Candidate (ms) |
| --- | ---: | ---: |
| 1 | 40.496 | 68.215 |
| 2 | 36.909 | 54.849 |
| 3 | 57.574 | 74.367 |

Windows cache contracts passed 5/5 (16.29 s), Linux CPU cache contracts 9/9
(13.88 s), and the newly merged request-admission test 1/1 (8.83 s). The final
device-provider regression test passed in 0.50 s after failing on the earlier
implementation. It includes actual GPU readback, segmented ownership,
reader lifetime, device-to-host persistence and host fallback.

Changed C++ files pass formatting. The whole-tree formatter reports the same
eight pre-existing diagnostics as official main. The extension source API
remains 0.3.8; its additive record and surface checks pass with the required
`ext-api` label supplied to the local validator.

An earlier official Flash numerical process made no observed progress and was
stopped; its separate retry passed. Delayed model loading also occurred on
both main and candidate during this work. Those startup observations remain
in the evidence and are not attributed to this cache change or folded into
warmed request timing. The numerical checks are 128-token prefill smoke tests,
not independent full-model quality certification.

Device backing remains Qwen27-only. Flash-Next is compatibility coverage.
The change avoids publication waits when the complete checkpoint remains
resident; it does not reduce disk writes or accelerate fsync. Device segments
remain committed until their cache and last readers close, with an 8 GiB cap
and growth admission checks. Other GPUs, Linux GPU behavior and OS-driven
reclamation are outside this Windows validation.
