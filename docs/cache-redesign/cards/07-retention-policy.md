# 07 · Retention policy port

**Milestone:** Common package · **Depends on:** 05, 06 · **Size:** M ·
**Affects:** nothing at runtime · **Status:** agreed

## Goal

Port today's checkpoint selection and eviction ranks into the new package
unchanged, and log every admission and removal so retention can be replayed.

## Scope

- Capture candidates, as today:
  - stable boundary and complete prompt;
  - up to four 2,048-token grid points;
  - learned divergence points with at least 512 tokens of improvement;
  - skip grid points within 128 tokens of the prompt end.
- Removal ranks, as today: retry 0, history 1, covered continuation 2, branch
  point or last copy 3. LRU within a rank. Record limit 128.
- Adapter pass-plan query, so optional boundaries align with planned prefill
  passes. Exact required boundaries are never rounded.
- An event for every admission, refusal and removal, with rank, reason and
  unique bytes freed.

## One unavoidable difference

Admission charges unique bytes instead of each checkpoint's full payload. The
same budget therefore admits more checkpoints than today. This is intended, but
it changes behavior, so card 19 reports it separately from storage effects.

## Not in this PR

Any new policy: rank changes, a cost-aware shorter restore, density tuning.
Each needs its own card with measurements.

## Test first

Port the policy cases from the current cache's unit tests and replay them
against the new package with the fake adapter.

## Step baseline

Replay the RFC's archived W1–W4 request traces through the package with the
fake adapter using real component sizes. Record checkpoints admitted, refused
and evicted, and unique retained bytes over time.

## Done when

- [ ] Ported cases pass.
- [ ] A recorded event log is enough to replay the retention decisions.

## Review focus

- Parity with the current cache's policy: anything missed?
- Is the pass-plan query enough to avoid splitting prefill passes for
  optional checkpoints?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025): admitting recurrent states by reuse likelihood and FLOP-aware eviction. For later policy cards: this card ports today's policy unchanged.
- [KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400) (NeurIPS 2025): eviction beyond LRU using the expected next use of an agent's prefix.
- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560): misses after long idle gaps between user turns; step-type driven eviction.
- [llama.cpp server slots and context checkpoints](https://github.com/ggml-org/llama.cpp/blob/41abbfd59/tools/server/server-context.cpp): checkpoint spacing and bounded checkpoint lists.

## RFC

[Checkpoint selection and eviction](../RFC.md#checkpoint-selection-and-eviction)

## Review notes

