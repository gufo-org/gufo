# D4 · Options and budgets

**Type:** decision · **Blocks:** 08, 10, 19 · **Status:** agreed

## Question

The new cache needs memory the current one does not reserve (committed spill
backing), and the meaning of the staging budget changes. How do existing
options map, and do we add any?

## Current options

From `src/cli/serve/serve.cpp` and `docs/SERVER.md`:

| Option | Today |
| --- | --- |
| `--cache-ram-bytes N` | 0 (default) picks an automatic snapshot budget. |
| `--cache-disk DIR` | Enables restart-safe reuse. Several processes may share a directory. |
| `--cache-disk-bytes N` | Disk budget, 8 GiB by default. |
| `--cache-disk-staging-bytes N` | 0 picks an automatic size; limits the largest checkpoint that can be written or restored. |
| (internal) | 128 checkpoint records. |

## Proposal

1. **Spill backing comes out of `--cache-ram-bytes`.** One RAM budget; the
   ledger (card 03) splits it into committed backing, private state, tails and
   metadata. No new option. Backing is committed at startup. At the measured
   ~25 GB/s page-commit rate, 8 GiB adds roughly 0.34 s to startup (estimate).
2. **`--cache-disk-staging-bytes` keeps its name; its meaning narrows.** It
   bounds the transfer buffer only and no longer limits checkpoint size. Card 12
   measures the default.
3. **`--cache-disk-bytes` keeps its meaning.** Referenced, temporary and orphan
   bytes all count against it.
4. **`--cache-disk DIR` becomes single-process.** A second process gets a clear
   ownership error at startup, not a silent fallback.
5. **The record limit stays an internal constant (128).** Private-state bytes
   already bound checkpoint density.

## Alternatives considered

- A separate `--cache-spill-bytes`: rejected, one more knob with no clear user
  decision behind it.
- Allocating spill backing lazily: rejected by the RFC because page commits
  block other HIP calls ([PR #445](https://github.com/gufo-org/gufo/pull/445)
  measured an ~8 ms decode stall).

## Also decide

- Is the automatic `--cache-ram-bytes` value still right when the memory is
  committed at startup instead of growing on demand? It becomes visible as
  resident memory immediately.
- Is the single-process directory change a breaking change for the release
  policy (`!` in the PR title)? Recommendation: yes, mark card 19 as breaking.

## Review notes

