# D2 · Adapter order

**Type:** decision · **Blocks:** 14–17 · **Status:** agreed

## Decision

1. Flash-Next, MTP and AR (card 14). Most users run it.
2. Qwen 27B AR (card 15).
3. Qwen 27B DFlash2 (card 16).
4. DeepSeek V4 Flash, AR and DSpark (card 17).

## Consequences

- The first adapter has the most complex inventory: target KV, MTP KV,
  pooled indexer, raw indexer ring, GDN state and kept hidden rows. Card 14 can
  be split into an inventory PR, then AR, then MTP.
- Flash-Next already borrows rows from live slots into shared backing blocks.
  Its RAM gain is smaller than 27B's, so its step baseline must separate
  capture-time gains from retained-memory gains.
- Under D1 the order only decides which adapter is built and proven first.
  All four are wired into serving together by card 19.

## Review notes

