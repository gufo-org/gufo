# 20 · End-to-end baseline and qualification

**Milestone:** After switch-over · **Depends on:** 19 · **Size:** evidence (no
code, except fixes for confirmed regressions) · **Affects:** release
readiness and follow-up work · **Status:** proposed

## Goal

Turn the step baselines and the switch-over runs into one end-to-end
baseline, qualify every admitted capacity row, then choose the end-to-end
numbers we want and open optimization cards for the steps that matter (D3).

## Scope

- **Collect step baselines** from cards 03–19 in one table: what each step
  costs in time, memory and disk.
- **Qualify** every admitted (model, mode, concurrency, depth) row. C = 1, 2,
  4 and 8 only where the row admits it; the long-agent row separate from the
  concurrent and mixed rows.
- **Client history shapes:** thinking-on clients that drop
  `reasoning_content` and clients that replay it (`cache-growth`), the
  rewriting bridge (`cache-bridge`), and the card 01 patterns, with rotation,
  pressure and restart.
- **Agent replays** with `tests/functional/pi_agent.py`; inspect actual tool
  results and drafts.
- **Choose targets:** for each workload, identify which steps dominate
  end-to-end time and memory, agree on target numbers, and open one
  optimization card per step worth improving.

## Output

A results document next to the RFC:

- the end-to-end baseline, per row and workload;
- residual limits and rows excluded as infeasible, with the reason;
- the agreed targets and the list of optimization cards.

The RFC's historical evidence stays as it is.

## Done when

- [ ] Every model/mode has a capability record, a capacity row and qualification
  evidence.
- [ ] Targets are agreed and optimization cards exist for the chosen steps.

## Review focus

- Are any admitted rows missing from the runs?
- Do the chosen targets follow the RFC priority: latency, then RAM, then disk
  space and writes?

## References

Material to consider during implementation. Before reusing code from a
project, check its license and add it to `THIRD_PARTY_NOTICES.md` when
required (see the README).

- [TraceLab: Characterizing Coding Agent Workloads for LLM Serving](https://arxiv.org/abs/2606.30560): workload shapes to compare against: idle gaps, append sizes, compaction.
- [TraceLab dataset and analysis code](https://github.com/uw-syfi/TraceLab): released Claude Code and Codex traces that could be replayed.
- [Agentic AI Workload Characteristics](https://arxiv.org/abs/2605.26297): cache-hit ratios and decode-dominated agent workloads.

## RFC

[Success criteria and evaluation](../RFC.md#success-criteria-and-evaluation) ·
[Client history transformations](../RFC.md#client-history-transformations)

## Review notes

