# Experimental expert streaming store

Additive: `gufo/ext/experimental/expert_stream.hpp`
(`src/core/hip/expert_stream/store.hpp`) streams a mixture-of-experts model's
routed experts from its files into a bounded device cache, linked through
`gufo_ext_hip`.

- `Layout` describes the experts: per layer, up to `kMaxPieces` pieces, each a
  file descriptor, offset and per-expert size (expert `e` at
  `offset + e * bytes`); the expert count, top-k, the largest batch one
  `Prepare` routes, and an optional fixed table width.
- `Policy` sets the cache's sizing (limit, reserve, live floor) and eviction
  (use half-life, classes, admission of large batches), the stage sizes, an
  optional routing trace and the log tag.
- `Store::Create(layout, policy)`; `Prepare(layer, routes, n_tokens, tables,
  overlap, context, routes_ready)` returns one device table per piece mapping
  each expert to its weights, and may call `overlap` with the resident and read
  rows so kernels run on cached experts while misses load. `Seed`,
  `SetReserve`, `AttachTails`, `Prefetch`, `Predict`, `Expect`,
  `PredictFlush`, `TracePrediction`, `GetStats` and `Log` complete it.
- `AvailableMemory()` is the host memory the store sizes itself from.

Migration: none.
