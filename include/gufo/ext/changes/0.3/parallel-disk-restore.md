# Parallel disk restore and warm-up

Additive, through the forwarded cache headers:

- `cache::kDiskStagingPieceBytes`, `cache::kDiskRestoreReaders` and
  `cache::kDiskReadsInFlight` describe disk staging. `kDefaultDiskStagingBytes`
  is now `kDiskRestoreReaders * kDiskStagingPieceBytes` (8 MiB instead of
  1 MiB): a restore reads and checks files into several pieces on reader
  threads while it copies others out; a file larger than a piece is read by
  one reader into successive pieces. Writes still move one piece at a time.
- `StreamedStore` takes an optional `piece_bytes` (default
  `kDiskStagingPieceBytes`). Staging smaller than two pieces, or rows wider
  than one, restores one piece at a time as before.
- A `DiskStore::ReadPayload` overload takes a per-piece buffer provider
  instead of one staging span.
- `DiskStore::WarmPayload` reads an owned dependency into the page cache
  without validating it, so a later `ReadPayload` finds it there.
- `platform::kAdviseReachesOs` is false where `platform::Advise` makes no OS
  call (Windows); restores then start no read-ahead hint helpers.

Restores still call adapters and wait for completions on the restoring thread
only; `Completion` gains no cross-thread requirement.

Migration: none. Pinned staging counts against the RAM cache ceiling as before,
so a disk cache keeps 7 MiB less RAM cache.
