# Prompt checkpoint optimization: draft status

Status at draft publication, 2026-10-05. This is work in progress, not a
completed performance qualification. Base: `main` at
`96a464769b097e44d7df43e3bfcaf06ba3aa8e45`. Branch:
`perf/flash-next-prompt-checkpoints`.

The external investigation identified the stable-boundary prefill split and
whole-state snapshot transfers as serving-path costs. Its patch and machine
paths were unavailable here. This branch implements the optimization locally;
external throughput figures are not treated as measurements of this machine.

## Implementation

- Flash-Next RAM snapshots freeze the mutable recurrent, convolution, PLE,
  indexer-ring, kept-hidden and draft state. The private mutable device storage
  is approximately 111 MB in AR and 119 MB with MTP on this model.
- Append-only K/V and pooled rows remain borrowed from the live session. Normal
  conversation extension performs no whole-prefix K/V snapshot copy. Before an
  overwrite, only the affected rows are protected. Snapshots sharing rows share
  backing blocks, grouped by their actual consumers to avoid retaining a deep
  checkpoint's suffix through a short checkpoint.
- Reset preserves borrowed rows until the first overwrite. Destruction protects
  remaining borrowed data before releasing session buffers. Byte export and
  disk serialization materialize the complete version-16 payload. RAM admission
  still charges the full logical snapshot size.
- Same-session restore preserves the live K/V prefix without uploading it.
  Cross-session restore copies device data directly and uses snapshot lineage
  to reuse only prefixes whose identity is proven. Matching token text alone
  does not prove identical K/V after different prefill shapes.
- Snapshot allocation/free uses a private HIP pool and transfer stream. The
  latest revision shares this allocator across an executor's sessions. This
  last change remains experimental: elimination of timing outliers has not
  been demonstrated.
- Flash-Next captures a stable prompt boundary inside one prefill pass: GDN
  recurrent state in both kernel routes, convolution/PLE history, kept hidden
  rows, boundary logits and token/position metadata. A through-capacity of 2056
  absorbs a final short assistant suffix into a normal 2048-token prefill.
  Attention query partitioning retains the boundary's prefill grouping.
- The serving runner reserves cache capacity before capture, adopts the
  in-pass snapshot through the existing publication path, and avoids redundant
  capture where a usable identical checkpoint already exists. Other models,
  image paths, disk captures and intermediate/shared checkpoints retain their
  existing split path.
- Cache acquisition prefers an available snapshot's source slot. A busy source
  does not block a branch into another slot; other choices retain LRU behavior.
  Other models default to no source preference.
- Restore keeps recorded graphs but clears unrecorded warmup shapes, avoiding
  graph compilation on the first cached decode. Capture time is now forwarded
  to request metrics instead of being reported as zero.
- Added native snapshot lifetime, branch, overwrite, boundary, concurrent-read
  and retention checks; independent GDN checkpoint comparisons; runner/cache
  tests; and a functional `cache-depth` suite. Functional concurrency requests
  record cohort membership, stagger admission by 20 ms and match histories by
  cache work rather than timing. Timing margins remain 5% and 3 ms.

## Measured prefill and decode

Production builds use the same pinned Nix toolchain on this Strix Halo gfx1151
host, with Qwen3.8 Flash-Next UD-Q4_K_XL and the Q8_0 MTP sidecar. AR measurements
use `tools/bench/model-bench.py`, `single-ar`, pp2048/tg128. Candidate labels
below identify intermediate builds, not the draft head.

| Depth | Rebuilt original `f797b5b`, v32 pair | Candidate v32 PP | Candidate v32 TG | Later v34 PP | Later v34 TG |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1557.10 | 1458.47 | 25.96 | 1517.24 | 25.99 |
| 16384 | 1311.38 | 1296.96 | 25.90 | 1357.93 | 25.91 |
| 32768 | 1355.56 | 1321.93 | 25.75 | 1348.34 | 25.76 |
| 65536 | 1210.16 | 1183.41 | 25.54 | 1184.49 | 25.51 |
| 131072 | 1150.96 | 1172.67 | 24.89 | invalid | invalid |

The v34 131k run overlapped a build because of an orchestration error, was
interrupted and is excluded. Its intended original control was not run. The
four completed points are measurements, not a complete paired comparison.

A separate v32 d0 ABBA check retained all four samples: candidate
1499.55/1492.55 PP, original 1518.68/1508.16 PP. Candidate TG was 25.97 versus
original 25.90/25.91. The first sweep's shallow result was not stable enough
to establish a 6% regression. The draft-head v38 has no completed PP/MTP sweep.
Original/candidate completion hashes differ at 16k and 32k in the first sweep;
bit-identical prose against the old original is not claimed. Cache and token
counts matched. Published September benchmark tables remain unchanged.

## Profiling and investigations

- v32 long-context profiling identified 41 allocations of 8 MiB around a deep
  restore, taking roughly 14.3–14.5 ms. Grouping by consumers reduced allocation
  count, but v34 still spent 14.458 ms on one large allocation. Grouping alone
  did not solve restore latency.
- Source-slot preference reduced affected long-context restore measurements
  from 24–26 ms to 2.7/3.1/8.9 ms, versus fresh main controls around 8–9 ms.
- Source preference exposed first-cached-decode graph compilation. Profiling
  showed graph instantiation plus recording overhead. Clearing only unrecorded
  warmup shapes fixed the affected long-context timing flags; v37 and v38
  long-context qualification passed.
- A cache-growth trace captured a 674.79 ms `hipEventSynchronize` interval with
  unusually long GPU kernels across several stages. No other-thread HIP
  allocation/copy overlapped it. A later run showed a roughly 500 ms decode
  gap; v38 growth still showed a cold-prefill timing increase of roughly
  210 ms. The cause is unresolved. Unchanged-main evidence has not established
  that this stall is pre-existing noise.
- Some concurrent depth branches changed physical decode cohort width between
  control and candidate. This explains why their decode times are not directly
  comparable, but their raw flags remain inconclusive; they are not averaged
  away or counted as passes.
- A first retention test used global `hipMemGetInfo`, which includes allocator
  caching and could not isolate checkpoint retention. It was replaced by a
  snapshot-owned `DeviceBytes()` check and exact restore validation. The new
  guard is compiled but has not yet run on the latest build.

## Completed checks and their limits

- Latest v38: production Nix build and native snapshot/session targets build;
  bounded CPU/repository checks pass 37/37; shared formatting check passes
  491 C++ files. This is not a full CPU or GPU qualification claim.
- v32: all 13 selected functional correctness suites passed: MTP C4
  long-context/cache/edits/growth/depth/rotation/concurrency/shared-prefix,
  MTP C1 depth, and AR C4 cache/edits/growth/depth. Actual draft execution was
  observed. Only AR cache-edits passed timing; the other 12 were inconclusive.
- v32 native snapshot tests passed exact round trips, in-pass checkpoints,
  overwrite/branch/reset/destruction and concurrent-reader checks in AR/MTP.
- v33 session batch tests passed peer graph capture with snapshots,
  cancellation/image isolation and C2/C4/C6/C8 AR/MTP logits and RNG checks,
  with sampled accepted and rejected cycles.
- Earlier independent GDN checks compared against CPU references without
  widening numerical tolerances. These do not replace latest-head model tests.
- v36: seven affected functional suites passed correctness after source-slot
  changes; most timing results remained inconclusive. v37 long-context passed
  correctness and timing; growth/depth passed correctness with timing flags.
- At the v38 publication snapshot, all five selected jobs pass correctness:
  MTP growth, long-context, depth and cache, plus AR cache. MTP long-context
  also passes timing; the other four remain inconclusive with timing flags.
- A literal-protocol Pi agent replay passed actual write/read/bash assertions
  on an earlier candidate. The latest candidate still needs that replay.

## Missing before this draft is ready

1. Investigate remaining per-request functional timing flags. Establish or fix
   the growth stall with affected matched controls.
   Do not label unqualified measurements regression-free.
2. Run full native snapshot/session checks on the final source, including the
   corrected retention guard. Recheck affected single-session/rotation and
   concurrent histories after any further source change.
3. Run final matched AR and MTP depth sweeps at 0/16k/32k/65k/131k; standard
   mixed/repetitive MTP and fixed C1/C2/C4/C6/C8 throughput checks; confirm
   decode and real draft execution. Current performance claims concern earlier
   binaries only.
4. Replay the final literal-protocol Pi workflow and inspect retained tool
   results. Finish quality/evidence documentation and final formatting checks.
5. Decide whether the latest shared allocator change is justified by evidence.
   Extending in-pass checkpoints to Qwen27B/DeepSeek is outside this draft.
   Refresh published PP tables only after the required depth recovery is shown.

## Evidence location

The committed companion `artifacts/prompt-checkpoints-draft.json` records build
identities, source hashes, functional status and every available flagged
comparison for the retained qualification batches, plus raw AR benchmark JSON.
It is a draft snapshot, not a final qualification certificate. Full profiler
traces, request/session logs, unchanged-main controls and interrupted-run
records remain locally under `artifacts/flash-next-checkpoints/`; those large
files are not included in the commit. Selected native/build logs are included
in the companion to distinguish completed checks from pending reruns.
