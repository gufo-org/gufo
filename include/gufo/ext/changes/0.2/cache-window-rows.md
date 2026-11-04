# Window-row cache components

Continuation cache (`src/cache/types.hpp`, reached through `gufo/ext/cache.hpp`):
`ComponentKind::kWindowRows` describes append rows of which a continuation reads
only the last `ComponentDescriptor::window_rows` (sliding-window attention).
A checkpoint at row `n` keeps only the chunks from the one containing row
`n - window_rows`, shares them with neighbouring checkpoints, and lets older
chunks be released. `window_rows` is nonzero exactly for this kind. New helpers
`cache::HasRows`, `cache::WindowFirstRow` and `cache::FirstChunkRow` give the
row ranges; `kWindowRows` checkpoint components start at `FirstChunkRow`, not
row 0, and `Adapter::CopyRowsIn`/`BorrowRowsView` see only retained rows. The
disk manifest gains format version 3 (written only with window components);
version 2 manifests stay readable. Additive: adapters using `kAppendRows` and
`kPrivateState` are unchanged.
