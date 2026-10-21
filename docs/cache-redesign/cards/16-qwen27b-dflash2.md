# 16 · Qwen 27B DFlash2 draft state

**Milestone:** Adapters · **Depends on:** 15 · **Size:** M ·
**Affects:** nothing served until card 19 · **Status:** proposed

## Goal

Extend the Qwen adapter to DFlash2, with draft state restored exactly and
speculative rollback guarded.

Qwen 27B serving supports AR and DFlash2. This migration covers those modes;
MTP remains unsupported, as confirmed for the October 13 qualification.

## Scope

- **Inventory:**
  - `QwenDFlashGpuSnapshot` and `QwenDFlashDraftSnapshot`;
  - draft KV, whose position is independent of the target;
  - any draft recurrent or hidden state;
  - controller state: request-owned (reset by `BeginRequest`) or
    model-owned.
- Guard calls on the speculative rollback paths.
- **No cost while inactive:** until card 19 attaches the new cache, the guard
  calls do nothing on the production path.
- Confirm the RFC's DFlash2 capacity row with committed backing included; the
  draft changes memory claims.

## Test first

Model-local tests:

- round trip with draft state: target and draft positions restored exactly,
  and next-step drafts match an uninterrupted run;
- a rollback after a rejected draft preserves needed rows first.

## Step baseline

At 32k and 100k tokens: private bytes per checkpoint including draft state
(RFC: about 80 MiB extra), capture and restore time, and draft acceptance on
a fixed prompt set before and after a restore.

## Done when

- [ ] Tests above pass on gfx1151.
- [ ] The model's standard speed benchmark against matched `main` shows no
  regression from the inactive guard calls (per phase, 5% / 3 ms).

## Review focus

- Which draft state is per request and which belongs to the checkpoint.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [DFlash upstream](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a): the draft model and its state.
- [Hybrid Models Meet SGLang: More than Full Attention](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/): recurrent state across speculative draft tokens.

## RFC

[Component descriptors and model modules](../RFC.md#component-descriptors-and-model-modules)

## Review notes

## Qwen27B implementation update

AR and DFlash2 serving now bind component-owned executors. The adapter supplies
borrowed FP16 target rows, canonical FP32 transfers, bounded private serialization
and committed backing shared with Flash-Next. Draft components retain physical
ring KV and its independent frontier, pending target features and controller
metadata. Sampled frontier logits and up to nine pending output tokens accompany
the target checkpoint. The Qwen runner's legacy snapshot/persistence hooks are
removed; DeepSeek keeps its existing snapshot path.

At `c7053529`, Q4 and Q8 full cache-serving checks pass for AR and DFlash2:
generated frontiers, greedy/seeded forks, cancellation history, disk restart,
artifact compatibility misses, concurrency and cleanup. The shared verification,
component cache and runner checks pass, as do all 59 hosted checks and formatting.
The current DFlash component round-trip also passes. These checks do not qualify
the outstanding vision cold-prefill discrepancy, deep workloads or resource gates.
Per user instruction, performance measurements follow functional completion;
further FP16/FP32 long-capacity telemetry probes are skipped. Historical results
remain tied to their revisions.
