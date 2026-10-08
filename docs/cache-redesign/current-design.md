# Current design

How the continuation cache works on main `b39c530e`, limited to what matters
for a redesign. [KV-CACHE.md](../KV-CACHE.md) is the user-facing description.
Source references are `path:line` on that revision.

## Snapshot contents

A snapshot is opaque to the cache and owned by the model runner. It can only be
reused whole, and only when its tokens are an exact prefix of the new prompt.

| Model | State in a snapshot | Notes |
| --- | --- | --- |
| Qwen3.8 Flash-Next | Attention KV, GDN recurrent state, indexer, position, sampling and MTP state | Hybrid recurrent/QSA model |
| Qwen3.8 27B | Attention KV plus convolution and DeltaNet buffers (`src/models/qwen/hip/executor.hpp:124`) | The executor places attention every `full_attention_interval` layers. [KV-CACHE.md](../KV-CACHE.md) describes the 27B payload as attention KV; experiment 1 measures its fixed part |
| DeepSeek V4 Flash | Session state, not yet itemised here | About 16 KB per token **(measured 2026-09-13)**: an 11.6k-token prefix took 185 MB |

**Flash-Next size model (derived).** #275 logged two Flash-Next disk files of
the same conversation: 3,642,154,856 bytes at 128,294 tokens and
4,114,500,368 bytes at 145,488 tokens. That gives a marginal cost of
27.47 KB per token and a fixed part of about 112 MiB. A later comment on the
issue reported 113.8 MiB + 27.45 KB/token. Smaller-context runs in #409 do not
fit the same line exactly, so experiment 1 measures it per configuration.

## RAM tier

- **Records and budget.** At most 128 immutable checkpoint records
  (`src/cli/serve/text_model_runner.hpp:47`), bounded by a byte budget:
  automatic, or set with `--cache-ram-bytes`.
- **What is captured.**
  - The stable boundary: the prompt before the generation suffix.
  - The complete prompt.
  - Up to four intermediate checkpoints on a 2,048-token grid
    (`src/cli/serve/text_model_runner.cpp:741`).
  - Learned divergence points, once a common prefix adds at least 512 tokens
    over what a request could already restore
    (`src/cli/serve/text_model_runner.hpp:478`).
- **Eviction.** Ranks, lowest removed first: retry 0, history 1, redundant
  prefix of a longer same-identity continuation 2, branch point or last copy 3.
  LRU within a rank (`src/cli/serve/continuation_cache.cpp:32`, `:107`, `:130`).
- **Physical memory.**
  - 27B: each `QwenGpuSnapshot` owns its buffers, so each checkpoint is a full
    copy.
  - Flash-Next (#445): mutable state is copied at capture. Append-only KV rows
    are borrowed from the live session until a rewind, reset, fork or byte
    read needs them (`src/models/qwen38_flash_next/engine.hpp:211-218`). When
    the session is about to overwrite them, the rows are preserved once into
    blocks shared by the affected checkpoints
    (`src/models/qwen38_flash_next/kernels/rocm/executor.cpp:451`).
- **Accounting.** Admission counts the complete payload size, including
  borrowed rows (`src/cli/serve/inference_backend.cpp:1609` for DeepSeek V4,
  `:2378` for Flash-Next). The budget therefore admits fewer Flash-Next
  checkpoints than physically fit.

## Disk tier

- **Options.** `--cache-disk DIR`, `--cache-disk-bytes`,
  `--cache-disk-staging-bytes`.
- **Files.** One file per checkpoint, with the persistence identity, the exact
  tokens and the full payload. Each file is checksummed, published atomically
  and readable by the owner only. Several server processes may share a
  directory.
- **What is written.** Prompt and stable boundaries, plus learned
  shared-prefix boundaries. A checkpoint less than 2,048 tokens past a stored
  prefix is skipped (`min_checkpoint_step_tokens`,
  `src/cli/serve/text_model_runner.hpp:71`), except at shared-prefix
  boundaries (#348).
- **Lookup.** A token prefix tree per persistence identity
  (`src/cli/serve/continuation_disk_store.cpp:469`). A restore past a request's
  stable boundary needs some entry at or below that boundary
  (`RestoreLongestPrefix`, `:1295`).
- **Eviction.** Global LRU on main (`:991`). #409 removes covered intermediate
  checkpoints first.

## Known costs

- **Full copies at long context.** In #275's Flash-Next run, each file was
  about 4.1 GB at 145k tokens and took 2.3–4.6 s to write, two writes per turn
  before #348 **(measured)**.
- **Disk budget filled by one conversation.** #275; mitigated by #348 (fewer
  writes) and #409 (eviction order).
- **RAM byte pressure.** #451: under byte pressure the retry copy is refused,
  so an unchanged retry prefills 5–7 tokens.
- **Capture contention.** On 27B Q8 with DFlash2, a disk write overlapping the
  next turn's capture raised capture time from about 20 ms to 57–230 ms (#348)
  **(measured)**.
- **Disk-hit side effect.** A disk restore skips the intermediate RAM
  checkpoints below the restored position. A later full prefill creates them,
  costing about 40 ms in #409's run **(measured)**.

## Limits found by the experiments

Details in [Experiments](experiments.md#e3-missed-reuse):

1. **A RAM hit of any length hides a longer disk hit.** Disk is consulted only
   when RAM has no hit at all (`src/cli/serve/text_model_runner.cpp:1877`).
   Cost: 22,558 tokens re-prefilled in E2 W2.
2. **The RAM cache refuses checkpoints instead of evicting.** An incoming
   checkpoint can evict only entries of equal or lower rank
   (`MaxRemovalPriority`). A full budget therefore refuses new intermediates:
   19–42 refusals per run.
3. **Automatic budgets are small for long contexts.** With two sessions:
   - RAM 9.2 GB (Flash-Next) or 23 GB (27B): three checkpoints at 100k tokens;
   - staging 2.3 GB or 5.8 GB, so no checkpoint past about 75k tokens reaches
     disk. Restores have the same limit: `ReadImage` loads the whole file and
     refuses files larger than staging
     (`src/cli/serve/continuation_disk_store.cpp:777`).
4. **27B captures copy the whole state:** 235–265 ms per checkpoint at 149k
   tokens, up to 635 ms observed.
