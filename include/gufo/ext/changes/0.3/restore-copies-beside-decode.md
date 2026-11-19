# Restore copies beside decode

Additive: `cache::StreamedStore::Restore` (`src/cache/streaming.hpp`) takes an
optional last argument, a `cache::RestoreCopyRunner`. The runner receives the
restore's copy phase (file reads, checks, and the adapter's `CopyRowsIn` and
`LoadPrivatePiece` calls with their completions) and may run it on another
thread while its caller does other work, returning only once the copies have
returned and rethrowing what they threw. `BeginRestore`, `Validate` and
`Invalidate` stay on the calling thread. The serving scheduler uses it to keep
decoding other sessions while an admission restores a checkpoint, only when the
provider sets the new `cache::ComponentCacheResources::restore_own_queue`
(default false): inside a `cache::BesideDecodeScope` (new, with
`cache::CopyingBesideDecode()`), which such a runner holds on the thread that
runs the copies, its `restore_stream` leases streams with a hardware queue of
their own. Without one, the copies wait behind decode work, slower than
stopping it.

`hip::TransferPool`'s constructor takes an optional last `own_queues` flag
(default false): each native stream then gets a hardware queue of its own (a
CU-masked stream with every CU) while the process allowance lasts, falling back
to a stream of the given priority past it or where the runtime refuses one.
`TransferPool::OwnQueues()` reports whether every native stream got one.
`hip::SetOwnQueueAllowance` sets that allowance (one unless set) and
`hip::OwnQueueAllowance` reads what is left; the serving layer sets it from its
hardware queue plan. HIP creates a CU-masked stream
blocking, so a model that launches work on the legacy null stream must not ask
for one. `hip::ComponentCacheHooks` gains `own_restore_queue` (default false),
which adds one such stream for restores beside decode while the allowance has
one left (other restores keep the two priority streams), and sets
`restore_own_queue` when it got one. HIP
otherwise shares a pooled hardware queue between a restore stream and the
model's executor stream, so a copy waited behind whole decode graphs.

Migration: none. Adapters must already keep those calls off the executor stream;
the in-tree Flash-Next, Qwen and DeepSeek adapters copy on the transfer stream
they are given. Flash-Next and Qwen set `own_restore_queue`; DeepSeek, whose
kernels launch on the null stream, does not.
