# 22 · Retention under small RAM budgets

**Milestone:** Switch-over review · **Depends on:** 07, 19, 21 · **Size:** M ·
**Affects:** Flash-Next AR/MTP serving · **Status:** in progress

## Problem

Every Flash-Next checkpoint holds about 132 MB (AR) or 133 MB (MTP) of private
state. A 2 GiB budget therefore holds only 12–13 checkpoints. The
`cache-bridge` then `cache-pressure` history at 2 GiB restored zero tokens in
four places in both AR and MTP on `def3763e`. The retention events show why:

- **Compacted history:** each 11K-token turn captured four grid boundaries, so
  one conversation filled more than half the budget. Rank-then-LRU removed the
  abandoned conversation's 2048-token history boundary (rank 1) before older
  150-token fill checkpoints (rank 3).
- **Idle BETA resume:** six 87-token side requests were newer than BETA. LRU
  among last copies removed BETA's only 11K-token checkpoint.
- **After volatile traffic:** the abandoned-branch probe's captures removed
  ALPHA's continuation, the oldest last copy, although ALPHA had just been
  continued.
- **After UI traffic:** three one-use 11.3K-token title/tag/follow-up requests
  removed the conversation they summarize.

## Policy change

Record-capacity removal, retry admission's rank limit, card 21's history
admission and per-turn edit-boundary protection are unchanged.

1. **Grid density.** A turn plans at most `min(4, RAM budget / (12 × private
   state bytes))` optional grid points, shallowest first. 2 GiB keeps one grid;
   8 GiB keeps the existing four.
2. **Byte-pressure order.** Removal takes, in order:
   1. retry copies;
   2. cheap checkpoints, whose private state exceeds eight times the rows they
      restore (under about 600 Flash-Next tokens), unless reused within one
      RAM-budget turnover: those go with the unreused conversations by
      recency, so a short prompt restored one request ago survives its retry
      (`state-edges`, #573), and unless they are a prefix of the incoming
      prompt, so a new conversation does not evict the system/tools prefix
      it shares with the previous one (`tool-agent` peers, #573);
   3. conversations not reused within one RAM-budget turnover, least recently
      used first, where recency includes retained descendants;
   4. reused conversations.

   Within an unreused conversation, covered near-duplicates go first, then the
   leaf, then shared boundaries: the shallowest history boundary, which other
   prompts may reach, and forks. Within a reused conversation, covered records
   go first, then leaves, then forks.

   An unchanged retry's full-prompt copy may replace the checkpoint it was
   restored from when at most 64 tokens (the assistant framing) separate
   them. No other conversation loses a checkpoint, so this keeps the #451
   rule that a retry copy never evicts another conversation, and restores
   main's zero-prefill replay (AR `state-edges`, #573). Records a disk job is
   writing are never victims: removing them frees nothing until the write
   ends.
3. **Reuse.** A turn that restored or continued a cached prefix marks the
   checkpoints it admits; a RAM restore marks its source. The mark lasts until
   admitted unique payload since that reuse exceeds the RAM budget, so a stale
   conversation ages out like any other.
4. **Forks.** A record that two retained continuations extend directly keeps
   that status. Subagents sharing a parent turn diverge a few tokens after it,
   so the parent is not an exact branch point.
5. A redundant source still advances only when every eligible candidate is a
   last copy, as in card 07's port.

The policy caches the descendant relation per record, so a removal decision
performs no token-prefix comparisons beyond the existing rank computation.

## Design evidence

A simulator replays the request structure with the retention ranks, the serving
capture order, chunk sharing and sizes calibrated from the `def3763e` logs. It
reproduces the four failures. Blind alternatives (GreedyDual-Size/Frequency,
segmented LRU, second-chance and class/LRU mixes) flipped outcomes with
single-checkpoint margins and are rejected. The retained design passes AR at
0.9–1.1× and MTP at 0.95–1.2× the calibrated capacity; removing any one of the
grid limit, cheap class, reuse protection or conversation order fails at least
one scenario at every scale. Its constants sit on a plateau: ratio 4–8 and
turnover 1–1.5 budgets pass equally.

## Results

Evidence: `/home/mixer/gufo-qualification/pr543-pressure-2026-10-12` (drivers,
binaries, patches, simulator, every run). Both revisions ran the same
`cache-bridge` then `cache-pressure` history with two sessions, context 32768
and `--cache-ram-bytes 2147483648`, fresh `def3763e` controls first.

| Request | Floor | `def3763e` AR / MTP reuse | Candidate AR / MTP reuse | TTFT AR, ms | TTFT MTP, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pressure_compacted` | 1 | 0 / 0 | 2048 / 2048 | 1601 → 370 | 1720 → 391 |
| `pressure_idle_resume_BETA` | 11223 | 0 / 0 | 11232 / 11232 | 7127 → 157 | 7613 → 161 |
| `pressure_after_volatile` | 11242 | 0 / 0 | 11251 / 11251 | 7096 → 156 | 7613 → 162 |
| `pressure_after_ui` | 11263 | 0 / 0 | 11272 / 11272 | 7165 → 166 | 7620 → 161 |

Every other reuse assertion and every cold-output comparison passes in both
modes; the abandoned-branch probe stays under its ceiling (2048 AR, 0 MTP).
`cache-bridge` passes, and its second turn now restores the remaining 2048
grid (AR TTFT 4568 → 3150 ms). MTP proposes 318 drafts and accepts 146 across
the pressure suite.

The card 21 six-suite history at 8 GiB still passes growth, depth, shared
prefix, rotation, concurrency and edits in AR and MTP. Last-user, older-tool and
rewind edits restore 2048 tokens and prefill 1199/1656/1199, as before; MTP
proposes 1506 drafts and accepts 871. Cache work changed on a few requests.
Unchanged retries in growth (both modes) and edits (MTP) now restore their
whole prompt instead of prefilling seven tokens, and an MTP concurrency turn
restores 4096 tokens instead of none (TTFT 4215 → 1580 ms). The MTP concurrency
warm probe lost its exact retry copy: it prefills seven tokens and its TTFT
rises from 7.3 to 94.4 ms. Rotation and depth turns move by up to nine tokens
in either direction.

All 57 hosted contracts and shared formatting pass. Each new retention test
fails when byte pressure uses the preceding order; the serving test fails when
the turn's reuse mark is removed.
Card 21's protection-lifetime test keeps its assertion that protection ends
with the turn; once it ends, the deeper covered boundary is now reclaimed before
the shallower one, which the preceding LRU order removed first.

**Timing is not qualified.** The policy itself is cheap: a host microbenchmark
of victim selection and admission matches the preceding order (about 0.1–0.2 ms
at 12 records and 3.7 ms at 64 Flash-sized records per removal), and a CPU
profile of the whole AR six-suite history attributes 0.30% of process samples to
retention admission and reclamation, versus 0.26% on `def3763e`. Both orders
take about 0.7 s per removal with 128 records along one 100K-token chain, an
existing limit of the rank computation.

Two alternations ran both histories in AR and MTP, control first and then
candidate first. Matched requests have identical prompt, restored and prefilled
tokens; flags use the strict 5% and 3 ms gate per request and phase.

- AR bridge turns 3–6 prefill 5.7–7.6% slower in the first alternation and
  5.0–6.3% faster in the second: run-order drift, not the change.
- Reassignment queue time moves between adjacent requests with the retained
  set. The control pays 15.0–17.3 ms on the UI tag request and 2.5–4.3 ms on
  the follow-up; the candidate pays 2.2–4.7 ms and 12.9–16.8 ms. A 38–39 ms wait
  moves from the first shared-prefix cold control to the next, and `pressure_live_BETA` waits
  3.1–3.4 ms longer in AR. These belong to the open reassignment-preservation
  blocker; they are reproduced flags, not passes.
- Fan-out `pressure_child_BETA` (AR) reaches a 267 ms token gap in both
  alternations, versus 80 ms. The candidate admits its prompt-end retry copy,
  which the control refused for lack of space; that synchronous post-token
  capture overlaps the other children's restores.
- `depth_unchanged` prefill is 7.3–7.6 ms slower in every candidate run in both
  modes, and one MTP growth cold control is 9–10% slower in both final-binary
  runs, each with identical cache events and pass chunking. The earlier binary
  with the same policy matched the control on the growth request. Neither cause
  is found; the profile excludes retention CPU time.
- Asynchronous capture time and a few decode/token-gap phases also flag in both
  directions; capture work differs where the retained set differs.

No timing improvement or pass is claimed beyond the changed-work requests.

## Done when

The correctness criteria pass; the timing gate below is still open
(2026-10-12).

- [x] Each new retention rule has a regression that fails on the preceding
  order; a serving regression fails without the turn's reuse mark.
- [x] `cache-bridge` then `cache-pressure` at 2 GiB pass every reuse assertion
  and cold-output comparison in AR and MTP on the final revision.
- [x] The six-suite card 21 history still passes in AR and MTP.
- [ ] Affected request timings meet the strict 5% / 3 ms gate against matched
  `def3763e` controls. Two alternations are retained; the reproduced flags
  above remain open.

## References

[Card 07](07-retention-policy.md) ports the original ranks; [card 21](21-edit-boundary-admission.md)
adds edit-boundary admission. GreedyDual-Size (Cao and Irani, 1997), segmented
LRU and CLOCK were evaluated and rejected. Marconi (Pan et al., 2024) motivates
cost-aware retention for hybrid recurrent models. No code is copied from these
works.
