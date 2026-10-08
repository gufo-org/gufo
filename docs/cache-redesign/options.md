# Options

## Candidates

| Option | Summary |
| --- | --- |
| **A. Current design + #409** | Full snapshots. Covered intermediates evicted first on disk. Optionally count Flash-Next RAM checkpoints by unique bytes |
| **B. llama.cpp style** | Diffs only inside a live slot; full copies for everything cached |
| **C. Chunked KV** | KV in 2,048-token chunks shared by checkpoints; fixed state per checkpoint; existing prefix tree and eviction ranks; budgets count each chunk once. RAM, disk or both |
| **D. SGLang style** | One radix tree with per-component reuse rules (full KV, window, recurrent checkpoints), paged KV, host and storage tiers |
| **E. Hybrid** | Option C organised around live sessions: captures copy only fixed state, KV rows are spilled to a shared chunk pool only when a session overwrites them, one index covers RAM and disk, keyframes come from background compaction. See [Hybrid design](hybrid-design.md) |

B would barely change gufo. Flash-Next already keeps KV once in the live
session, and gufo's retention across conversations is already richer than
deleting contained prompts. D's extra machinery addresses scale gufo does not
have (see [below](#what-sglang-needs-that-gufo-does-not)). The real choice is
between A (with the Phase 0 fixes) and E, which is C made concrete.

## Diffs and keyframes

One proposal from the discussion: keep every diff, take a full snapshot
(keyframe) from time to time, keep X keyframes, and restore from the latest
keyframe plus the following diffs. It fits gufo with three refinements:

1. **A diff must carry the recurrent state at its end.** Recurrent state
   cannot be rebuilt from KV rows; only another prefill produces it (see
   [example 7](hybrid-design.md#example-7-why-kv-rows-cannot-rebuild-recurrent-state)).
   A KV-only diff saves nothing on Flash-Next or 27B.
2. **With that, applying diffs is a copy, not a computation.** Restoring at
   diff k reads the keyframe's KV, the KV rows of diffs 1..k, and diff k's
   fixed state: the same bytes as one full snapshot. Keyframes therefore do not
   speed up restores. They bound dependency chains and simplify eviction.
3. **Build keyframes on disk, not from the GPU.** A keyframe captured from the
   GPU is a full copy, the cost this design removes. Merging existing files in
   the background (compaction) produces the same file without touching the
   GPU.

Branches (edits, forks, subagents) turn the diff chain into a tree. Finding a
diff's parent by token prefix, with the existing prefix tree, handles that.
Chunks (option C) deduplicate a prefix across the conversations that restored
it; parent pointers deduplicate only through a shared checkpoint. Neither may
share KV between independent computations of the same tokens, because equal
tokens can produce different KV bytes (see
[chunk identity](hybrid-design.md#chunk-identity-and-provenance)).
Option E combines both: chunks for KV, checkpoints for state.

## Gain depends on context length

Recurrent and fixed state stay a full copy per checkpoint, so the saving comes
from KV only. Flash-Next figures, using the derived size model in
[current design](current-design.md):

| Context | Full copy | Chunked, 2,048 tokens past parent | Gain |
| ---: | ---: | ---: | ---: |
| 8k | ~330 MB | ~170 MB | ~2× |
| 145k | ~4 GB | ~170 MB | ~24× |

## Disk: current design vs chunked

| Aspect | Current: full copy per checkpoint | Chunked: shared KV + per-checkpoint state | Better |
| --- | --- | --- | --- |
| Bytes written per checkpoint | Whole context, ~4 GB at 145k | New KV + fixed state, ~170 MB | Chunked |
| Space per long conversation | One full copy per retained checkpoint | KV once + ~112 MiB per checkpoint | Chunked |
| Prefix shared by many conversations | Repeated in every file | Stored once per lineage (conversations that restored it, not independent cold prefills) | Chunked |
| Retention policy (#275, #409) | Needs rules to evict redundant copies | Redundancy mostly gone; shared chunks protected while referenced | Chunked |
| Restore | Read one file | Read a chain of chunks, same total bytes | Current, slightly |
| Corrupt or missing data | Loses one entry | Loses every checkpoint that uses that chunk | Current |
| Eviction | Any entry, independently | Chunks freed only when unreferenced | Current, simpler |
| Several processes sharing one directory | Safe: standalone files, atomic publish | Needs reference counts across processes | Current |
| Model coupling | One opaque payload | Each model separates append-only KV from mutable state | Current |
| Recurrent state | Inside the blob | Still a full copy per restore point | Equal |
| Lookup | Token prefix tree | Same tree | Equal |
| Effort and risk | Exists and tested | New store, new snapshot interface for each model, new format version | Current |

## RAM: current design vs chunked

| Aspect | Current | Chunked | Better |
| --- | --- | --- | --- |
| Capture cost | 27B: full copy. Flash-Next: mutable state only | Mutable state only, every model | Chunked (Flash-Next already there) |
| Physical memory | 27B: full copy per checkpoint. Flash-Next: KV shared | KV shared, every model | Chunked (Flash-Next already there) |
| Budget accounting | Full size per checkpoint, ~4 GB at 145k | Unique chunks; ~112 MiB fixed state per extra Flash-Next checkpoint | Chunked: many more checkpoints per budget |
| Budget safety | Full size covers the worst case, where borrowed rows must later be copied out | Must track chunks the live session still owns | Current |
| Restore | Device copy, or zero-copy reuse of the source slot | Same | Equal |
| Retention policy | Rank rules for redundancy | Most redundancy gone; evicting an intermediate frees only its unique bytes | Chunked |
| Persistence, corruption, multiple processes | Not applicable | Not applicable | Equal: the main disk downsides disappear |
| Effort | Exists | Flash-Next mostly there; 27B and DeepSeek V4 need it | Current |

## Downsides reported by other engines

1. **Recurrent state is large and reusable only at exact positions.** Reuse
   happens only where a checkpoint exists. vLLM restores only at block
   boundaries; finer reuse costs more checkpoints.
2. **Two eviction rules.** KV goes from the leaves inward; recurrent
   checkpoints can go from anywhere. Supporting every model type pushed SGLang
   into a rewrite.
3. **Chunk size is a tradeoff.** Small chunks mean more metadata and I/O;
   large chunks mean fewer partial hits.
4. **Restores fetch many pieces.** At scale that needs prefetch policies; with
   local NVMe and 2,048-token chunks it matters less.
5. **Not bit-exact for GDN hybrids.** Different prefill chunk shapes, as with
   gufo's partial hits today.
6. **Coupled to model and kernels.** Chunk size follows the state layout, and
   cached data cannot cross backends or kernel block sizes.
7. **Young for hybrid models.** vLLM's Mamba prefix caching is experimental;
   SGLang unified its tree in 2026-08.

## What SGLang needs that gufo does not

| SGLang piece | Why SGLang needs it | Needed for gufo? |
| --- | --- | --- |
| Per-token paged KV and radix nodes | Hundreds of requests packing scarce GPU memory | No: at most 8 sessions |
| Host RAM tier (L2) | Small GPU memory, large host RAM | No: unified memory |
| Distributed storage (L3) | Sharing across machines | No: one machine, one directory |
| Prefetch policies, page-first layouts | Hiding network and storage latency | No: local NVMe, few large reads |
| Elastic pools for KV vs recurrent state | Changing ratios across many requests | No |
| Session IDs for eviction | Thousands of sessions | No: #409's covered rule plays that role without IDs |
| Per-component tree framework | Many model families | No: three models, a hand-written split each |
| **KV shared across checkpoints, recurrent state per checkpoint** | Avoid storing the same prefix many times | **Yes** |

## Features to decide

These determine which option fits. They are discussed after the experiments.

- Survive restarts (disk tier) at all, and at what cost per write.
- Several server processes sharing one cache directory.
- Keep rewind and edit points inside a conversation, or only the latest
  frontier.
- Sharing a long prefix across conversations (subagents, same system prompt).
- Output equality after a restore (greedy only, or sampled too).
- Which models matter most: Flash-Next, 27B, DeepSeek V4.
- Target context length and concurrency.
- Image prompts in cached state.
