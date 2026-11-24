# Restore copies beside decode

Additive: `cache::StreamedStore::Restore` (`src/cache/streaming.hpp`) takes an
optional last argument, a `cache::RestoreCopyRunner`. The runner receives the
restore's copy phase (file reads, checks, and the adapter's `CopyRowsIn` and
`LoadPrivatePiece` calls with their completions) and may run it on another
thread while its caller does other work, returning only once the copies have
returned and rethrowing what they threw. `BeginRestore`, `Validate` and
`Invalidate` stay on the calling thread. The serving scheduler uses it to keep
decoding other sessions while an admission restores a checkpoint, only when the
provider sets the new `cache::ComponentCacheResources::restore_beside_decode`
(default false): inside a `cache::BesideDecodeScope` (new, with
`cache::CopyingBesideDecode()`), which such a runner holds on the thread that
runs the copies, its `restore_stream` leases streams whose copies never wait
behind decode work. On streams that share a hardware queue with decode, the
copies wait behind it, slower than stopping it.

The HIP provider does that with the GPU's SDMA copy engine. New
`src/core/hip/copy_engine.hpp`: `hip::CopyEngine` (fences that count queued
copies; `Copy`, `Done`, `Wait`) and `hip::DeviceCopyEngine()`, null where the
runtime has no engine (Windows). `hip::TransferPool`'s constructor takes an
optional last `CopyEngine*` (default null), `TransferPool::TryAcquire` an
optional `on_engine` flag (default false), and `TransferPool::HasEngine()`
reports an engine. `hip::TransferStream` gains `OnEngine()` and
`EngineCopy(destination, source, bytes)`; an engine lease's `Complete` covers
its engine copies and its `Native` work, and records no stream event without
`Native` work. `hip::CopySnapshotToDevice` gains an overload taking the
`hip::TransferStream&` instead of a `hipStream_t`: an engine lease copies
device memory and registered host memory on the engine, anything else on the
lease's native stream. The HIP component-cache resources lease restores beside
decode on the engine and set `restore_beside_decode` when there is one.

Migration: none required. An adapter's restore loads reach the engine only
through the new `CopySnapshotToDevice` overload; the in-tree Flash-Next, Qwen
and DeepSeek adapters use it. Loads through `Native()` keep working on a
priority stream, but beside decode they wait behind it.
