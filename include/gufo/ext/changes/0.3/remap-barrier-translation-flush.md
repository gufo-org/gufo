# Remap barrier translation flush

Additive: `mem::RemapBarrier` (`src/mem/remap_barrier.hpp`) also flushes the
address translations the device cached for the process, which the idle time
alone does not drop while another process keeps the GPU busy, and reports
when it could not.

- `RemapBarrier::SetTranslationFlush(flush)` installs how the device backend
  drops them (false when it could not). `Settle` runs it after its idle
  time, up to `RemapBarrier::kFlushAttempts` times. Gufo's HIP backend
  installs it with the drain wherever the page pool or the expert stream
  store runs.
- `RemapBarrier::Settle` returns `[[nodiscard]] bool`: false when every
  flush attempt failed. Then no address stamped before the call is clean and
  no `NoteRemap` before it is answered, so the caller must not let the
  device use the remapped addresses; a later `Settle` tries again.
- `RemapBarrier::Counts::unflushed` counts the `Settle` calls that failed.
- `mem::SharedRows::SettleRemaps` (`src/mem/shared_rows.hpp`) is false as well
  when the barrier failed.

Compatibility: source compatible after rebuilding. A call that ignores the
result of `Settle` still compiles, with a warning: it must check it to stay
correct where a flush is installed. Code that takes `Settle`'s address needs
the new type, `bool (*)() noexcept`.
