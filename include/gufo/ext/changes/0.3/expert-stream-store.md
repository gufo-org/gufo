# Experimental expert streaming store

Additive: `gufo/ext/experimental/expert_stream.hpp`
(`src/core/hip/expert_stream/store.hpp`) streams a mixture-of-experts model's
routed experts from its files into a bounded device cache, linked through
`gufo_ext_hip`, and serving sizes that cache together with the RAM prompt
cache. The store carries no compatibility promise.

- `Layout` describes the experts: per layer, up to `kMaxPieces` pieces, each a
  `platform::PlatformFile`, offset and per-expert size (expert `e` at
  `offset + e * bytes`); the expert count, top-k, the largest batch one
  `Prepare` routes, an optional fixed table width whose extra entries are
  null, and `read_margin_bytes`, the readable bytes kernels may touch past
  the end of an expert's last piece. The store reopens each file for its
  reads, unbuffered where the platform allows it.
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
  default) keeps default-stream ordering. A failed `Prepare` caches none of
  the experts it was reading. `AttachTails`, `Prefetch`, `Predict`, `Expect`,
  `PredictFlush`, `TracePrediction`, `GetStats` and `Log` complete it.
- `Fit(grant)` sizes the cache to the host's grant, its prefetch stage and
  read margin included, and fills the room left from `Policy::ranking`, then
  with the experts it leaves out; until the first `Fit` nothing is cached.
  `Release(bytes)` drops any background prefetch, frees the newest chunks,
  each size class keeping the experts eviction ranks highest, and the rows
  misses borrowed when no slot could be lent, and holds the cache there, even
  when it freed nothing. `Restore()` lets it regrow to the grant, and
  `ReleasedBytes()` and `FloorBytes()` report what was released and the free
  memory below which it stops growing.
- The cache is mapped from 2 MiB device pages (`hipMemCreate`) where the
  device supports virtual memory management and its heap covers host memory
  (`mem::HeapCoversHost`), from managed memory elsewhere. When it maps pages
  where it released others, it settles `mem::RemapBarrier` before the tables
  point there, and on destruction it leaves its addresses clean. Its reader
  threads bracket their uploads with `mem::DeviceWorkScope`.
- `Policy::trace_path` writes a versioned binary routing trace (layout,
  policy, seeded cache, then each `Prepare`'s routes and hits, and a new
  snapshot of the cache after `Fit`, `Release` or `Restore` changed it).
  `tools/expert_stream/` reads it and replays eviction and prefetch rules
  offline.
- `TextRunnerResourceClaim::expert_cache` (`TextRunnerExpertCacheClaim`)
  names the runner's `expert_stream::Store`, the host memory to leave free
  after the expert cache (`headroom_bytes`), and an optional automatic
  prompt-cache budget per session while experts stream
  (`prompt_cache_share_per_session_bytes`). `AutomaticRetainedBytes(claim,
  sessions)` is the automatic prompt-cache budget the claim allows.
- `cache::kPreparedBackingBytes` names the backing `commit_backing`
  prepares before readiness, at most; `ComponentCacheResources::
  committed_backing_bytes` (optional, set by the HIP resources) reports what
  it committed.
- Once the sessions exist, serving commits the prompt cache's whole backing
  (beyond `kPreparedBackingBytes` through `prepare_capture`, where the
  resources provide it), then grants the store what it holds plus the host
  memory available, less the headroom and any budget still to commit, calls
  `Store::Fit(grant)`, and logs both budgets with the backing actually
  committed.
- Host pressure (#1275) is measured against the store's floor
  (`Store::FloorBytes()`, the larger of it and the free CMA pages) instead of
  the host reserve while experts stream: the expert cache fills the host down
  to that floor on purpose. Room under the process's cgroup counts too, held
  to the floor itself. Below it, the idle scheduler releases experts first
  (`Store::Release`, logged as `event=expert_cache_release`), then
  prompt-cache memory; once what it gave back is free again (the floor plus
  `Store::ReleasedBytes()` and 256 MiB available), `Store::Restore` lets the
  expert cache regrow to its grant (`event=expert_cache_restore`). New
  requests are refused below half the floor.
- `HostAvailableBytes()` (`gufo/ext/host_budget.hpp`) is the host memory
  available after cgroup limits and promised session state.

Migration: none.
