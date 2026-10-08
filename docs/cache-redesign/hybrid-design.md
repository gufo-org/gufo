# Hybrid design: shared KV chunks and small checkpoints

This is option E in [Options](options.md): llama.cpp's idea of checkpointing
only recurrent state, made efficient for 2–8 concurrent sessions and extended
to disk. Figures come from [E1](experiments.md#e1-snapshot-size-model-2026-10-07)
unless stated otherwise:

- Flash-Next with MTP: 114 MiB of fixed state + 27.46 KB per token.
- 27B: 152 MiB + 64 KiB per token, or 232 MiB fixed with DFlash2.

## Three kinds of state

| Thing | Holds | Lives in | Size |
| --- | --- | --- | --- |
| **Session** (at most 8) | A running conversation: KV for positions 0..p, contiguous, plus recurrent state at p | GPU memory, as today | Grows with the conversation |
| **Chunk** | KV rows for 2,048 tokens. Immutable and reference-counted. Identified by its token prefix *and* the computation that produced it (see [chunk identity](#chunk-identity-and-provenance)) | Shared pool: RAM, disk, or both | 2,048 × 27.46 KB ≈ 56 MB (Flash-Next), 128 MiB (27B) |
| **Checkpoint** | A restorable position p: recurrent and other fixed state at p, the chunks covering 0..p, and a tail of fewer than 2,048 rows that do not fill a chunk | Pool | Fixed state + tail |

Why the split:

- **KV rows never change once written,** so every checkpoint and conversation
  that descends from the same computation of a prefix can share them.
- **Recurrent state is different at every position** and cannot be rebuilt
  from KV rows (see [example 7](#example-7-why-kv-rows-cannot-rebuild-recurrent-state)).
  Each checkpoint therefore keeps its own full copy. This is what llama.cpp's
  checkpoints store.

## Operations

| Operation | What happens | Cost |
| --- | --- | --- |
| **Capture** | Copy the fixed state at p. KV rows stay borrowed from the live session | 114–232 MiB, a few ms |
| **Spill** | Before a session overwrites rows (rewind, reset, reuse by another conversation), copy rows that no pool chunk holds yet into new chunks, once, into space already reserved. Checkpoints that borrowed them now reference the chunks | Only rows not yet in the pool |
| **Restore** | Find the longest checkpoint whose tokens are a prefix of the prompt. Copy its chunks and tail into the session's KV, load its fixed state, prefill the rest | Same bytes as restoring a full snapshot today |
| **Evict** | Drop checkpoints by today's ranks (covered intermediates first, as in #409). Free a chunk when nothing references it | — |
| **Persist** | Write behind, streamed in bounded pieces: the fixed state, chunks not yet on disk, and the tail | ~170 MB per 2,048 new tokens (Flash-Next), ~280–360 MB (27B) |
| **Compact** | Merge one lineage's chunk files on disk into larger files (keyframes), without the GPU | Background I/O |

## Chunk identity and provenance

Equal tokens do not guarantee equal KV bytes. Prefill chunk shapes and batch
composition change floating-point results; gufo already notes that a partial
hit can change sampled output for this reason. A checkpoint's recurrent state
was computed together with specific KV bytes, and pairing it with KV produced
by a different computation of the same tokens would create a state that never
existed.

Rules:
- **A chunk records its lineage.** Two checkpoints share a chunk only if both
  descend from the same computation of those rows. That happens when one
  restored the chunk and prefilled onward from it (examples 2 and 3), or both
  were captured in the same session lifetime.
- **No sharing by token prefix alone.** Two independent cold prefills of the
  same system prompt produce two different chunk sets. Deduplicating them
  requires byte-identical payloads, for example by comparing a content hash of
  the bytes.
- **The learned shared-prefix boundary routes new conversations to restore
  instead of prefilling,** which keeps them in one lineage.

## Memory accounting for borrowed rows

Rows borrowed from a live session are cheap only until the session overwrites
them; then they must be spilled. If borrowed rows were counted as free, a
session reset could require a large allocation the budget never reserved.

- Each borrowed row that some checkpoint still needs counts as a **reserved
  spill** against the RAM budget, together with transfer buffers.
- Before a session overwrites borrowed rows, the cache either has the
  reservation or evicts checkpoints until it does.
- Today's Flash-Next accounting (full size for every checkpoint) is the safe
  upper bound. Accounting for unique bytes has to replace it with exact
  reservations, not with zero.

**One index for both tiers.** One token prefix tree holds every checkpoint, and
each chunk records whether it is in RAM, on disk, or both. A restore takes the
longest matching checkpoint wherever its pieces live; missing chunks are read
from disk.

**Budget.** RAM and disk budgets count each chunk once, plus each checkpoint's
fixed state and tail. Rows still owned by a live session count as reserved
spill (see [above](#memory-accounting-for-borrowed-rows)).

**Model interface.** A model describes its state as a list of components,
each with its own valid position and kind:

| Component (Flash-Next example) | Kind | Valid up to |
| --- | --- | --- |
| Target attention KV | Append-only rows | Trunk position |
| MTP draft KV | Append-only rows | Draft position (`mtp_position`), which differs from the trunk |
| Pooled indexer rows | Append-only rows | Indexer blocks pooled so far |
| Raw indexer ring | Ring history (mutable) | Last `index_capacity` rows |
| Recurrent (GDN) state, kept hidden row, wide residual, sampling state | Mutable | Exactly the checkpoint position |

(`src/models/qwen38_flash_next/kernels/rocm/executor.hpp:58–176`.) Append-only
components can be chunked and shared; everything else is copied whole into the
checkpoint. 27B has target KV, DeltaNet and convolution state, and DFlash2
state, and copies all of it today.

Flash-Next already borrows rows and preserves them once on overwrite (#445).

## Bounded transfers

Neither persisting nor restoring may stage a whole checkpoint in RAM:
- **Writes** stream borrowed rows to disk in bounded pieces. Today
  `SessionSnapshot::bytes()` materializes the full payload on the host.
- **Restores** stream chunks into the session. Today `ReadImage` loads the
  whole file and refuses files larger than staging
  (`src/cli/serve/continuation_disk_store.cpp:777`).
- **Queued snapshots** waiting for the writer count against the staging
  budget.

This also applies to Phase 0: streaming writes alone would persist deep
checkpoints that still could not be restored.

## Crash consistency and garbage collection

- **Publish order:** chunk files are written and fsynced first; the
  checkpoint manifest, which lists its chunks, is then published atomically
  with a rename. A crash leaves at worst orphan chunks, never a manifest
  pointing at missing data.
- **Readers pin chunks** during a restore, so a concurrent eviction cannot
  delete a chunk being read.
- **Orphan recovery:** at startup, chunks referenced by no manifest are
  deleted after a grace period, which protects writers in other processes.
- **One writer per directory in the first version,** enforced by a lock file.
  Several processes sharing one directory needs cross-process reference
  counts; postpone it until required.

## Checkpoint density

Cheap checkpoints are still bounded by the record limit (128 today) and the
budget. Priority order when they compete:
1. stable boundaries;
2. learned shared prefixes;
3. the end of the system prompt;
4. recent message boundaries (edit points);
5. grid points.

## Example 1: one conversation, one session

Conversation A starts on session 1 with a 10,000-token prompt (Flash-Next).

| Step | Today | Hybrid |
| --- | --- | --- |
| Prefill 10,000 tokens and checkpoint the prompt | Full snapshot: 114 MiB + 10,000 × 27.46 KB ≈ 395 MB counted against the RAM budget | Copy 114 MiB of fixed state; KV rows stay in session 1 |
| Turn 2 adds 1,500 tokens on session 1 | Nothing restored; another full snapshot, ≈ 436 MB | Nothing restored; another 114 MiB copy |

On 27B at 149k tokens the difference is larger: today's capture copies 10 GB
and took 235–265 ms per checkpoint in E2 W1; the hybrid copies 152 MiB.

## Example 2: a session switches conversation

A pauses at 12,000 tokens. Conversation B needs session 1. Later A comes back
and lands on session 2.

1. **Spill.** Before B overwrites session 1, A's rows go to the pool: 5 full
   chunks (rows 0–10,239) plus a tail of 1,760 rows, about 330 MB copied once.
   A's checkpoints now reference pool chunks instead of session 1.
2. **B runs** on session 1 as usual.
3. **A returns on session 2.** The newest checkpoint whose tokens are a prefix
   of A's prompt is at 12,000. Restore copies 5 chunks + tail (~330 MB) into
   session 2 and loads 114 MiB of fixed state. Only A's new message is
   prefilled.

llama.cpp copies A's whole state out of the slot into its host cache, and
copies it back when A returns. Today gufo keeps a full snapshot per retained
checkpoint.

## Example 3: subagents sharing a system prompt

Subagent S shares A's first 6,000 tokens (system prompt and tools).

1. **First subagent.** No checkpoint exists at exactly 6,000, and recurrent
   state needs one there. S restores the nearest earlier checkpoint (say the
   grid checkpoint at 4,096) and prefills only the 1,904-token gap. During that
   prefill it captures a checkpoint at 6,000:
   - fixed state at 6,000;
   - references to A's chunks 0–1 (rows 0–4,095). S restored them before
     prefilling, so its state descends from exactly those bytes;
   - its own tail of rows 4,096–5,999. A's chunk 2 also holds A-only tokens
     past 6,000, so S cannot share it. At most one chunk is duplicated per
     divergence.
2. **Every later subagent** restores the checkpoint at 6,000 directly. The
   shared 6,000 tokens of KV exist once, whichever session runs each subagent.
   If a second subagent had instead prefilled the prompt cold in parallel, its
   chunks would be a separate lineage and would not be merged with A's.

**Denser checkpoints shrink the gap.** A checkpoint costs only its fixed state,
so gufo can afford many more than today, within the bounds of
[checkpoint density](#checkpoint-density): the record limit and the budget
decide how many message boundaries and grid points are kept, in that priority
order. Where a boundary checkpoint survives, a new conversation prefills only
from it to its divergence point. Today each checkpoint is a full copy (2.9 GB
at 100k tokens on Flash-Next), so only a few are kept.

In E2 W4 a subagent created after a restart reused none of its 6,634 shared
tokens (27B: 21.6 s). No checkpoint at or below 6,634 had been persisted; disk
keeps only prompt boundaries. With cheap checkpoints, grid and message-boundary
checkpoints can go to disk too.

## Example 4: the parent returns after its forks

This is E2 W2. A parent agent reaches about 24,600 tokens. Two forks continue
from its history, then the parent takes another turn.

- **Today:** the RAM cache filled with the forks' checkpoints and refused
  others (`event=snapshot action=skipped reason=byte_capacity`, 22 times on
  Flash-Next). The parent's next turn found only the 4,661-token system prompt
  in RAM. Disk held the parent's 24,599-token checkpoint, but disk is consulted
  only when RAM has no hit at all (`src/cli/serve/text_model_runner.cpp:1877`).
  22,558 tokens were re-prefilled: 24 s on Flash-Next, about 60 s on 27B.
- **Hybrid:** the forks share the parent's chunks up to their divergence, so
  the parent's checkpoint costs only its fixed state and stays in RAM. Even if
  evicted, the one index finds it on disk.
- **Today's design with a one-line fix:** consulting disk whenever it holds a
  longer prefix recovers the same tokens in simulation (Phase 0).

## Example 5: a full RAM budget

Production-like Flash-Next (MTP, 2 sessions) got an automatic RAM budget of
9.23 GB.

- **Today:** a checkpoint at 100k tokens is 2.87 GB, so the budget holds three.
  An intermediate checkpoint may only evict entries of equal or lower rank
  (`MaxRemovalPriority`, `src/cli/serve/continuation_cache.cpp:32`), so once
  the budget is full of other conversations' last copies, new intermediates are
  refused.
- **Hybrid:** one 100k-token conversation's KV costs 2.75 GB once. Every further
  checkpoint of it costs 114 MiB plus a tail of at most 56 MB. The same
  9.23 GB holds that conversation with about 40–50 checkpoints, or three
  100k-token conversations with about two checkpoints each.

## Example 6: restart during a long 27B session

E2 W1 grew a real `pi` coding session on 27B to 149k tokens. With today's
defaults (automatic staging 5.74 GB), checkpoints past about 75k tokens never
reached disk: 85 writes were skipped with `reason=staging_capacity`, because a
27B checkpoint at 64k is already 4.35 GB and two queue at once.

Simulated graceful restarts on that trace (E8 revision 2, which reproduces
all 36 real requests):

| Restart at | Today restores | Hybrid restores | Extra prefill today |
| ---: | ---: | ---: | ---: |
| 60,622 tokens | 60,396 | 60,396 | none |
| 104,505 tokens | 75,026 | 104,386 | ~29k tokens, ~2 min |
| 148,523 tokens | 75,026 | 133,842 | ~59k tokens, ~4.3 min |

Prefill at that depth ran at 196–250 tokens/s. The hybrid writes about 152 MiB
of fixed state plus 128 MiB per 2,048 new tokens per checkpoint, so staging is
never the limit.

## Example 7: why KV rows cannot rebuild recurrent state

The model stacks layers; attention layers sit between GDN or DeltaNet recurrent
layers. Each recurrent layer updates its state token by token from that token's
hidden state at that layer, which is the output of every layer below,
including MoE and MLP blocks:

```text
token t -> GDN layer 1 (state updated from h1(t)) -> ... -> attention layer (stores K, V of t)
        -> ... -> GDN layer k (needs hk(t)) -> ...
```

Only the attention layers' K and V are stored. The hidden states the recurrent
layers consumed are not, so the recurrent state at position p can only be
obtained by running every layer again over the tokens since the last saved
state, which is a prefill. Stored KV rows save only the attention over earlier
positions. Storing the hidden states instead would cost an estimated 10–15×
more than KV (hidden size × layers × 2 bytes per token).

Consequences:

- **A diff must carry the recurrent state at its end position.** KV-only diffs
  save nothing on Flash-Next or 27B. llama.cpp's checkpoints are the other half
  of the same idea: recurrent state only, with KV kept live in the slot.
- **Applying diffs costs no compute.** Restoring at diff k means taking the
  KV rows of the keyframe and of diffs 1..k, plus diff k's fixed state. That
  reads the same bytes as one full snapshot.
- **Keyframes do not speed up restores.** They bound dependency chains, which
  limits the damage of a lost file and simplifies eviction. Built from the GPU
  they are full copies, exactly the cost this design avoids, so the hybrid
  builds them by background compaction on disk.
- **The leftover prefill is the gap to the nearest checkpoint.** It is zero
  only when a checkpoint sits exactly at the divergence point, which is why
  cheap, dense checkpoints matter (example 3).

## Phasing

| Phase | Scope | Addresses |
| --- | --- | --- |
| 0 | On today's design: consult disk when it holds a longer prefix than RAM; bounded, streamed writes **and** restores (both are limited by staging today); exact reservations so Flash-Next RAM checkpoints can be counted by unique bytes | Example 4; deep checkpoints reaching and coming back from disk; part of example 5 |
| 1 | Model component interface (example above); RAM chunk pool with lineage and reserved spills; small checkpoints; bounded density | Examples 1, 2, 3, 5; capture cost on 27B |
| 2 | Disk tier on the same chunks: manifests published after chunks, pinning, orphan recovery, one writer per directory; one index for RAM and disk. Compaction off at first, enabled only if measured restores need it | Example 6; write volume; restart value |
| 3 (optional) | Paged KV in the attention kernels: zero-copy restore, live sessions physically sharing prefix KV, no full-context preallocation per session | Memory per session at high concurrency |

## Open questions

- **Compaction policy and cost.** Each compaction rewrites a lineage's KV
  bytes once, adding to write volume and needing temporary disk space. The
  write volumes in [E7/E8](experiments.md) exclude it.

- **Chunk size.** 2,048 matches today's grid and disk spacing. Smaller chunks
  duplicate less at a divergence but need more metadata.
- **Checkpoint density policy:** message boundaries, a fixed grid, or both.
- **Several server processes sharing one disk directory:** reference counting
  across processes, versus giving each process its own directory.
- **Crash consistency** of chunk and checkpoint files (write order, fsync).
- **Output equality after a restore,** which has the same chunk-shape caveat as
  today's partial hits.
- **Image prompts:** chunk identity must include image identity.
