# Experimental expert streaming store

Additive: `gufo/ext/experimental/expert_stream.hpp`
(`src/core/hip/expert_stream/store.hpp`) streams a mixture-of-experts model's
routed experts from its files into a bounded device cache, linked through
`gufo_ext_hip`. It carries no compatibility promise.

- `Layout` describes the experts: per layer, up to `kMaxPieces` pieces, each a
  `platform::PlatformFile`, offset and per-expert size (expert `e` at
  `offset + e * bytes`); the expert count, top-k, the largest batch one
  `Prepare` routes, an optional fixed table width, and `read_margin_bytes`,
  the readable bytes kernels may touch past the end of an expert's last
  piece. The store reopens each file for its reads, unbuffered where the
  platform allows it.
- `Policy` sets the cache's sizing (limit, live floor), the experts to cache
  first (`ranking`, a `Ranked` list of layer, expert and weight), eviction
  (use half-life, classes, admission of large batches), the stage sizes, an
  optional routing trace and the log tag.
- `Store::Create(layout, policy)`; `Prepare(layer, routes, n_tokens, tables,
  overlap, context, routes_ready, stream)` returns one device table per piece
  mapping each expert to its weights, and may call `overlap` with the resident
  and read rows so kernels run on cached experts while misses load. It reads
  the routes once the work queued on the caller's `stream` before the call has
  run, and queues the copies the tables depend on there; the null stream (the
  default) keeps default-stream ordering. `AttachTails`, `Prefetch`,
  `Predict`, `Expect`, `PredictFlush`, `TracePrediction`, `GetStats` and `Log`
  complete it.
- `Fit(grant)` sizes the cache to the host's grant, its prefetch stage and
  read margin included, and fills the room left from `Policy::ranking`, then
  with the experts it leaves out; until the first `Fit` nothing is cached.
  `Release(bytes)` frees the newest chunks, dropping their experts without
  reading them, and holds the cache there, `Restore()` lets it regrow to the
  grant, and `ReleasedBytes()` and `FloorBytes()` report what was released
  and the free memory below which it stops growing.
- The cache is mapped from 2 MiB device pages (`hipMemCreate`) where the
  device supports virtual memory management and its heap covers host memory
  (`mem::HeapCoversHost`), from managed memory elsewhere. When it maps pages
  where it released others, it settles `mem::RemapBarrier` before the tables
  point there, and on destruction it leaves its addresses clean. Its reader
  threads bracket their uploads with `mem::DeviceWorkScope`.
- `Policy::trace_path` writes a versioned binary routing trace (layout,
  policy, seeded cache, then each `Prepare`'s routes and hits).
  `tools/expert_stream/` reads it and replays eviction and prefetch rules
  offline.

Migration: none.
