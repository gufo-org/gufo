# 01 · Functional coverage for real cache workloads

**Milestone:** Preparation · **Depends on:** — · **Size:** M (a series: one PR
per scenario family) · **Affects:** tests only · **Status:** proposed

## Goal

Grow functional coverage of the ways clients actually use the cache, so the
switch-over (card 19) can show the new cache handles them. Any scenario that
improves coverage is welcome, whether today's cache passes or fails it.

## Already covered

Keep these; extend them where a pattern below fits better than a new suite.

| Suite or tool | What it covers |
| --- | --- |
| `cache-growth` | Growing conversation, unchanged retry, Messages thinking replay, dropped reasoning |
| `cache-edits` | Edited history |
| `cache-depth` | Deep borrowed frontiers across branches and slot reuse |
| `cache-rotation` | Switching conversations keeps useful history; snapshot budget |
| `cache-shared-prefix` | New conversations reuse a shared system prompt |
| `cache-concurrency` | Concurrent requests sharing a prefix prefill it once |
| `cache-bridge` | Bridge rewriting the user message in history, with short unrelated requests in between |
| `cache`, `cache_disk_spacing.py`, `continuation.py --restore` | Restart persistence, cancellation and replay, an image appended after a cached turn |
| `system-injection` | Mid-conversation system messages (Qwen hoists them into the leading system turn) |
| `agent_long.py`, `pi_agent.py`, `opencode_agent.py` | Real agents past 100k tokens, with compaction and pruning **disabled** |

## Usage patterns to add

| # | Pattern | Seen in | The scenario checks |
| --- | --- | --- | --- |
| 1 | **Compaction near the context limit.** History is replaced by a summary; system prompt and tools stay. | Claude Code, Codex, OpenCode. Rare but large drops (> 64k tokens) in TraceLab. | Reuse covers system prompt and tools only. Under pressure, the abandoned history's checkpoints go before live conversations' ones. |
| 2 | **Tool-output pruning.** Old tool results are replaced by a placeholder deep in history. | OpenCode prune, Claude Code tool-output clearing. llama.cpp reports full re-processing on Gated DeltaNet models. | Restore the last coherent checkpoint before the first pruned result; prefill only from there. |
| 3 | **Micro-reductions at user turns.** A few hundred tokens removed near the end. | Codex (TraceLab). | Reuse up to a checkpoint before the change, not a cold prefill. |
| 4 | **Human-paced gaps.** A conversation idles while others run, then resumes. | TraceLab: most user-turn misses follow > 5 min idle. | A resumes from RAM or disk after other conversations filled slots and budget. Simulate with pressure, not wall time. |
| 5 | **One large tool result.** A single append of 10k+ tokens, then a retry or edit right after it. | TraceLab: rare 10k+ appends carry over 70% of prefill. | The retry or edit does not prefill the large append again. |
| 6 | **Volatile tokens near the start.** A per-request header or timestamp in the system prompt. | Claude Code's attribution header defeats llama.cpp reuse. | No false hit, and one-use entries do not push useful checkpoints out. |
| 7 | **UI side requests.** Title, tag and follow-up generation after every reply, with a prompt that embeds the chat. | Open WebUI background tasks. | The main conversation's next turn still reuses its history. |
| 8 | **Claude Code over `/v1/messages`** with tools and streaming. | gufo #469. | A long tool loop reuses history turn after turn. |
| 9 | **Subagent fan-out, then the parent resumes.** | Claude Code Task tool; RFC W2. | Children share the parent prefix; the parent's deep checkpoint survives in RAM or on disk. |
| 10 | **Model swap restart.** The server is stopped and started between turns, as llama-swap does; also an abrupt kill while a write is queued. | llama-swap setups; RFC W4. | The next turn restores from disk; after a kill, only published checkpoints hit. |
| 11 | **A short RAM hit beside a deeper disk checkpoint.** | RFC E3 (W2). | The deeper usable checkpoint is selected. |
| 12 | **A checkpoint larger than the staging budget.** | RFC. | It persists and restores after a restart. |

Several of these fail on today's cache (2, 11, 12 at least); others pass. Both
kinds are useful.

## Rules for expectations

- Express expected work through boundaries ("prefill ≤ gap to the nearest
  retained boundary"), not exact token counts tied to today's grid policy.
- Assert through what the server reports today (cached tokens, prompt
  progress). Card 19 adds richer per-request fields.
- Compare answers with uncached controls, as the existing suites do.
- Suites that need disk or restarts are selected explicitly, not part of `all`.

## Not in this PR

Changes to the current cache to make failing scenarios pass.

## Step baseline

For each new scenario, run it on current `main` when it lands and record per
request: prefilled and restored tokens, TTFT, and pass or fail. That run is
the comparison point for card 19.

## Done when

- [ ] Each added scenario runs on current `main`, with its result recorded.
- [ ] New suites are documented in `tests/functional/README.md`.

## Review focus

- Which patterns matter most for our users? Order the PR series by that.
- Patterns missing from the list.

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560)
  (Claude Code and Codex traces: idle gaps, append sizes, compaction).
- [Claude Code, llama.cpp, and the hidden prompt cache killer](https://mykolaaleksandrov.dev/posts/2026/06/claude-code-llamacpp-prompt-cache-fix/)
  (attribution header changing the start of the prompt).
- [llama.cpp #24714](https://github.com/ggml-org/llama.cpp/issues/24714) and
  [#21831](https://github.com/ggml-org/llama.cpp/issues/21831) (full
  re-processing on hybrid models).
- [Open WebUI #17123](https://github.com/open-webui/open-webui/issues/17123)
  (title and follow-up generation after every reply).
- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379)
  (branch-point checkpoints and admission for recurrent state).
- [TraceLab dataset and analysis code](https://github.com/uw-syfi/TraceLab): real Claude Code and Codex sessions to derive scenarios from.
- [Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297): turn structure and tool-use phases of agent workloads.

## RFC

[TDD and behavioral contracts first](../RFC.md#tdd-and-behavioral-contracts-first) ·
[Client history transformations](../RFC.md#client-history-transformations)

## Review notes

