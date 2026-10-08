# Cost model

Per-operation costs of today's cache and of the [hybrid design](hybrid-design.md),
built only from constants measured on this machine (Strix Halo, gfx1151, 128 GB
unified memory, NVMe under dm-crypt). Each constant has a plausibility check
against something the production server already does. The tables come from
[cost_model.py](scripts/cost_model.py).

## Measured constants

| Constant | Value | Source | Plausibility check |
| --- | --- | --- | --- |
| Checkpoint size, Flash-Next (MTP) | 113.8 MiB + 27.46 KB/token | E1 disk payloads, exact fit | #275's production logs give the same slope |
| Checkpoint size, 27B (DFlash2) | 232.4 MiB + 64 KiB/token | E1 disk payloads, exact fit | — |
| Device copy (device to device) | 104–110 GB/s | [copybench.hip](scripts/copybench.hip) | Real RAM restores: 27B 33k tokens (2.4 GB) in 25 ms, 96 GB/s |
| Host copy (pinned, either way) | ~85 GB/s; pageable 61–83 GB/s | copybench.hip | — |
| Capture of fixed state only | Flash-Next 2–5 ms at any depth | E2 W1 `live_checkpoint` | Flash-Next already captures only fixed state today (#445): 2.2–5.4 ms from 1.8k to 149k tokens |
| Capture, 27B full copy | 14 ms at 0.36 GB, 235 ms at 10 GB (≈ 67 ms + 17 ms/GB, r² 0.51) | E2 W1 `live_checkpoint`, one agent, no concurrency | Its fixed state is 0.24 GB, so a fixed-state-only 27B capture is ~10–15 ms, matching the 1.8k-token captures |
| RAM held by checkpoints | Flash-Next: physical well below accounted; 27B: physical ≈ accounted | E2 W1 logs: `retained_bytes` vs `host_available_mib` | Flash-Next accounted 7.1–8.9 GB while available memory fell only 4–6 GB including the live session; 27B accounted 16.5–22.3 GB for a 14–20 GB fall. Counting unique bytes matches what the hardware holds |
| Disk write + fsync, raw | 0.58–0.60 GB/s at every size | [diskbench.py](scripts/diskbench.py) | — |
| Disk write in gufo | 0.44 GB/s (serialize + checksum + write + fsync) | E2 `write_ms` against `file_bytes`, about 170 writes | Per-write medians 0.36 GB/s (27B) and 0.52 GB/s (Flash-Next). One unexplained outlier at concurrency 4: 1.67 GB in 782 s |
| Disk cold read | 1.1 GB/s; 1.4–1.5 GB/s when partly cached | diskbench.py; E2 disk restores | Real restores: 55.5k-token 27B file in 2.5 s |
| Assembling a session from chunks in memory | 106 GB/s for 4 MiB pieces, 88 GB/s for 1 MiB, vs 103 GB/s for one copy | [chunkcopy.hip](scripts/chunkcopy.hip), 8 GiB as many device copies | Per-layer pieces of a 2,048-token chunk are an estimated 1–4 MiB |
| Chunk files instead of one file | +13% write time, +13% cold-read time (72 × 56 MiB vs 4 GiB) | diskbench.py | Compaction removes it |
| Prefill N tokens on D cached | Flash-Next 0.36 s + 0.79 ms/token + small depth term; 27B 0.52 s + 2.66 ms/token + 16.5 ns × N × (D + N/2) | [fit_prefill.py](scripts/fit_prefill.py), 150 requests each; median error 9.8% and 3.9% | Simulated prefill time matches measured within 3–8% (E7) |

`cache_snapshot_ms` in `usage.gufo` is not used as a capture cost. It is
measured from capture start to when the capture is joined
(`src/cli/serve/text_model_runner.cpp:913`), so it includes overlapped wall
time. Its medians (0.3–0.5 s per request on Flash-Next, about 1 s on 27B) are
an upper bound, not time on the critical path.

## Per-operation costs

RAM and staging budgets are the automatic values chosen with two sessions:
9.2 GB and 2.3 GB for Flash-Next, 23 GB and 5.75 GB for 27B. "Persist" for the
hybrid means one 2,048-token step: fixed state plus 2,048 tokens of KV.

### Flash-Next (MTP)

| Operation | Depth | Today | Hybrid |
| --- | ---: | ---: | ---: |
| Checkpoint bytes | 32k | 1.00 GB | 119 MB + shared KV |
| Capture (on the request path) | 32k | 3 ms | ~3 ms |
| Persist one checkpoint | 32k | 2.3 s | 399 ms |
| Restore from RAM | 32k | 10 ms | 10 ms |
| Restore from disk (cold) | 32k | 907 ms | 1.0 s |
| Prefill instead (cold) | 32k | 25.8 s | 25.8 s |
| Checkpoints of this conversation in the RAM budget | 32k | 9 | 56 |
| Checkpoint bytes | 100k | 2.87 GB | 119 MB + shared KV |
| Capture (on the request path) | 100k | 3 ms | ~3 ms |
| Persist one checkpoint | 100k | 6.5 s (skipped: > staging) | 399 ms |
| Restore from RAM | 100k | 29 ms | 29 ms |
| Restore from disk (cold) | 100k | 2.6 s | 2.9 s |
| Prefill instead (cold) | 100k | 80.4 s | 80.4 s |
| Checkpoints of this conversation in the RAM budget | 100k | 3 | 43 |
| Checkpoint bytes | 149k | 4.21 GB | 119 MB + shared KV |
| Capture (on the request path) | 149k | 3 ms | ~3 ms |
| Persist one checkpoint | 149k | 9.6 s (skipped: > staging) | 399 ms |
| Restore from RAM | 149k | 42 ms | 42 ms |
| Restore from disk (cold) | 149k | 3.8 s | 4.3 s |
| Prefill instead (cold) | 149k | 2.0 min | 2.0 min |
| Checkpoints of this conversation in the RAM budget | 149k | 2 | 34 |

### 27B (DFlash2)

| Operation | Depth | Today | Hybrid |
| --- | ---: | ---: | ---: |
| Checkpoint bytes | 32k | 2.34 GB | 244 MB + shared KV |
| Capture (on the request path) | 32k | 106 ms | ~14 ms |
| Persist one checkpoint | 32k | 5.3 s | 859 ms |
| Restore from RAM | 32k | 23 ms | 23 ms |
| Restore from disk (cold) | 32k | 2.1 s | 2.4 s |
| Prefill instead (cold) | 32k | 94.0 s | 94.0 s |
| Checkpoints of this conversation in the RAM budget | 32k | 9 | 67 |
| Checkpoint bytes | 100k | 6.80 GB | 244 MB + shared KV |
| Capture (on the request path) | 100k | 181 ms | ~14 ms |
| Persist one checkpoint | 100k | 15.4 s (skipped: > staging) | 859 ms |
| Restore from RAM | 100k | 68 ms | 68 ms |
| Restore from disk (cold) | 100k | 6.2 s | 7.0 s |
| Prefill instead (cold) | 100k | 5.8 min | 5.8 min |
| Checkpoints of this conversation in the RAM budget | 100k | 3 | 52 |
| Checkpoint bytes | 149k | 10.01 GB | 244 MB + shared KV |
| Capture (on the request path) | 149k | 236 ms | ~14 ms |
| Persist one checkpoint | 149k | 22.7 s (skipped: > staging) | 859 ms |
| Restore from RAM | 149k | 100 ms | 100 ms |
| Restore from disk (cold) | 149k | 9.1 s | 10.3 s |
| Prefill instead (cold) | 149k | 9.7 min | 9.7 min |
| Checkpoints of this conversation in the RAM budget | 149k | 2 | 42 |

"Prefill instead (cold)" extrapolates the fitted model to a cold prompt. The
earlier measurements in [KV-CACHE.md](../KV-CACHE.md) were 280 s for 101,545
27B tokens (a different configuration) and 184 s for 203,047 Flash-Next tokens,
so these numbers are within about 25% and directionally right.

## Background costs

**Compaction** (hybrid only) rewrites a lineage's chunk files into one file.
It reads and writes the KV bytes once, at 1.1 GB/s and 0.59 GB/s:

| Conversation | KV bytes | Compaction I/O |
| --- | ---: | ---: |
| Flash-Next at 149k tokens | 4.1 GB | ~11 s, in the background |
| 27B at 149k tokens | 9.8 GB | ~25 s, in the background |

**Disk writes per long agent session** (the W1 runs to 149k tokens, from E8 revision 2; compaction not included):

| Model | Today | Phase 0 | Hybrid |
| --- | ---: | ---: | ---: |
| Flash-Next | 7.0 GB, with deep writes skipped | 24.1 GB | 5.7 GB |
| 27B | 16.5 GB, with deep writes skipped | 56.8 GB | 13.2 GB |

An illustration only, under an assumed endurance rating: at ten such 27B
sessions a day, Phase 0 writes about 570 GB/day and the hybrid about
130 GB/day. A drive rated for 600 TB written would last roughly 2.9 years
versus 12.6 years.

## Reading the tables

1. **Restore costs the same in both designs.** It is bound by bytes, and the
   hybrid reads the same bytes, 13% slower from chunk files on disk until
   compaction runs. Restoring from disk is 28–64× cheaper than prefilling.
2. **Capture matters only for 27B.** Flash-Next already borrows KV, so the
   hybrid changes nothing there. On 27B it saves about 90–220 ms per
   checkpoint at 32–149k tokens, on the request path.
3. **Persisting is where today's design breaks.** A deep full checkpoint takes
   6–23 s to write and does not fit the automatic staging budget, so it is
   skipped. The hybrid writes 0.4–0.9 s per 2,048-token step.
4. **RAM capacity:**
   - today the budget holds 2–3 checkpoints of one long conversation;
   - the hybrid holds 34–52, which is what makes dense checkpoints
     (message boundaries, a shared system prompt) affordable.
5. **Phase 0 can stream full checkpoints to disk** instead of staging them in
   RAM. That restores persistence, at 6–23 s of disk time and 3–10 GB per deep
   checkpoint, so a disk budget holds only a few of them.
