# Restore copies beside decode

Additive: `cache::StreamedStore::Restore` (`src/cache/streaming.hpp`) takes an
optional last argument, a `cache::RestoreCopyRunner`. The runner receives the
restore's copy phase (file reads, checks, and the adapter's `CopyRowsIn` and
`LoadPrivatePiece` calls with their completions) and may run it on another
thread while its caller does other work, returning only once the copies have
returned and rethrowing what they threw. `BeginRestore`, `Validate` and
`Invalidate` stay on the calling thread. The serving scheduler uses it to keep
decoding other sessions while an admission restores a checkpoint.

`hip::TransferPool` keeps its declaration; a nonzero `priority` now gives each
native stream a hardware queue of its own (a CU-masked stream with every CU),
falling back to a stream of that priority where the runtime refuses one. HIP
otherwise shares a pooled hardware queue between such a stream and the model's
executor stream, so a copy waited behind whole decode graphs.

Migration: none. Adapters must already keep those calls off the executor stream;
the in-tree Flash-Next, Qwen and DeepSeek adapters copy on the transfer stream
they are given.
