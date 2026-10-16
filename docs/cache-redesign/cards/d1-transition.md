# D1 · One cache at a time, one switch-over

**Type:** decision · **Blocks:** 19 · **Status:** agreed

## Decision

Serving never runs the old and the new cache side by side, and the new cache
does not need to read anything the old one wrote. There is no routing rule and
no compatibility layer.

The switch happens once, at the end:

1. The common package, the disk store and every text adapter land on `main`
   first. None of them changes runtime behavior; fake-adapter and model-local
   tests prove them.
2. One switch-over card (19) wires the new cache into serving for every
   converted mode and deletes the legacy cache in the same change.

## Why

- Simpler: no seam that has to serve two implementations, no per-mode routing,
  no period with two caches to reason about.
- No regression window on `main`: every continuation mode, including
  `--cache-disk` restart reuse, keeps working until the switch, and works on
  the new cache right after it.

## Where PRs land

Cards 01–18 merge into `main` as ordinary PRs: they change no runtime
behavior, and `main`'s CI keeps their tests running while model code keeps
changing. Only card 19 is built on a short-lived branch, as a stack merged
together. A long-lived integration branch was considered and rejected: the
executor and serving code the adapters hook into changes constantly on
`main`, CI only runs on PRs to `main`, and the final merge would become one
unreviewable PR.

## Consequences

- Real-request evidence comes late. To get it earlier, run the switch-over
  branch locally for Flash-Next as soon as its adapter (card 14) and the
  package exist, before the other adapters are finished. That branch is not
  merged until card 19's checks pass.
- The switch-over PR is large. Card 19 lists how to keep it reviewable.
- Cache directories from the old format are rebuilt after the upgrade (D4).

## Options not taken

- Routing by adapter capability, with both caches behind a seam.
- Switching early for Flash-Next only (RAM first, disk later), accepting that
  other modes and `--cache-disk` lose reuse until later cards.

## Review notes

