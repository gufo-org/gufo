# D3 · Step baselines instead of upfront targets

**Type:** decision · **Blocks:** every PR card · **Status:** agreed

## Decision

We do not set target numbers now. We do not know well enough what each step
will cost, and the RFC research already measured the current cache.

Instead:

1. **Every card records a step baseline when it lands.** Each card says what
   to measure (operation cost, bytes, latency) and records the result in its
   **Results** section and summarizes it in the PR. Important measurements
   belong in the card; do not add standalone measurement JSON files. A step
   baseline is a reference point, not a pass/fail target.
2. **Correctness and repository rules still gate every card.** Exact restored
   state, numerical quality, no partial-state execution, and at the switch-over
   (card 19) the repository's per-request 5% / 3 ms timing gates against
   matched `main` controls.
3. **After the switch-over, card 20 builds the end-to-end baseline** from the
   step baselines and the qualification runs. We then pick the end-to-end
   numbers we want and open optimization cards against specific steps.

## What a step baseline contains

- What was measured, on which machine and revision, with the command.
- The result, and how it compares with the RFC evidence when a comparable
  number exists. Comparisons are informative only.
- Anything surprising, as a note for card 20.

## Review notes

