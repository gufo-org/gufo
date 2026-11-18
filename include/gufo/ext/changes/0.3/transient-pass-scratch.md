# Transient pass scratch

Additive: `gufo/ext/experimental/state_memory.hpp` now also provides
`mem::ScratchArena` (`src/core/hip/scratch_arena.hpp`), pass scratch at fixed
addresses whose rows past a resident width take memory only while a wider
pass needs them (#952).

- `ScratchArena::Reserve(capacity, error)` reserves address space; it
  returns null where device memory is not demand-mapped
  (`mem::DemandMapped()` is false), and the runner then allocates its narrow
  scratch whole.
- `Take(bytes, resident, error)` carves a buffer at its widest size and backs
  its first `resident` bytes for the arena's life; `MapResident` backs further
  ranges of buffers carved together.
- `MapTransient()` backs every other carved page; `ReleaseTransient()` returns
  them to the driver, waiting for in-flight GPU work. The arena promises
  nothing to the state ledger: callers map transient pages only when the host
  can spare them (`mem::MemGetInfo`, `mem::HostAvailableBytes`).

Migration: none.
