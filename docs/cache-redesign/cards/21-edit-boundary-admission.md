# 21 · Edit-boundary admission under byte pressure

**Milestone:** Switch-over review · **Depends on:** 07, 19 · **Size:** S ·
**Affects:** Flash-Next AR/MTP serving · **Status:** in progress

## Problem

After several unrelated conversations fill the backing pool, optional history
captures can all refuse while older families retain rank-3 continuations. The
new complete-turn capture then displaces the turn's own covered history. An
older-message edit has no compatible boundary and prefills the entire prompt.
The production six-suite replay reproduces this in both AR and MTP; correcting
reverse-prefix coverage alone does not resolve it.

## Policy change

Under byte pressure, a history capture may evict an older family's last copy.
Computed ranks and LRU ordering within each rank remain unchanged. Retry
admission and the full-record optional-admission guard retain their rank limits.
An impossible payload still refuses before eviction.

Each execution lease protects the history checkpoints it successfully admits
from its later captures. The bounded ID array is charged as metadata. Protection
ends with that lease; it is not a permanent retention preference or a guarantee
that other conversations cannot evict a boundary.

This is an explicit change from card 07's policy port. It does not add a
prefill-cost score, density tuning, family quota or new public option.

## Results

The retention regression fails on the preceding policy, then verifies byte
pressure evicts the oldest eligible family. Separate coverage checks protected
boundaries, source protection and reclamation after turn protection ends.

The serving regression fills four private backing blocks with older families,
captures two history boundaries and a complete turn, then restores edits at
both boundaries. Restored recurrent state matches independently cold execution.
Removing only the turn's protection makes this regression fail at the first
reuse assertion. Restoring protection makes it pass. All 57 repository contracts
and shared formatting pass.

The production six-suite history passes in both AR and MTP: growth, depth,
shared prefix, rotation, concurrency and edits, in that order with two sessions,
context 32768 and 8 GiB RAM. Each last-user, older-tool and rewind edit restores
2048 tokens, then prefills 1199/1656/1199 respectively; all edited outputs match
their cold controls. The preceding policy failed edits in both modes under the
same history. MTP proposes 1506 drafts and accepts 871 across the new replay.
Constrained-RAM pressure and matched request timings remain under qualification;
this correctness result does not establish a latency pass.

Evidence: `/home/mixer/gufo-qualification/pr543-review-2026-10-11`,
`grid-red.log`, `grid-protection-red.log`, `grid-protection-green.log`,
`grid-pr-final.log`, `grid-format-after-control.log`, `grid-functional-summary.json`.
Production binary SHA-256:
`def206c3bb6a37af4cb6eaed2a914209252ca9781d1321cfb4c45f47062442f7`.

## Done when

The correctness criteria pass; the timing gate below is still open
(2026-10-12).

- [x] Actual backing-pressure regression fails without protection and passes
  with it, including independent cold-state checks.
- [x] Production AR/MTP edits and rotation pass after the preceding suites.
- [x] Constrained-RAM pressure passes; [card 22](22-small-budget-retention.md)
  records the separate retention change it needed.
- [ ] Affected request timings meet the strict 5% / 3 ms gate.

## References

[Card 07](07-retention-policy.md) records the original policy port and the #553
coverage correction. [Card 19](19-switch-over.md) retains broader qualification
and unresolved latency/retention findings.
