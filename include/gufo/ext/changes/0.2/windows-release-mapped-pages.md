# Windows: release mapped pages

Platform (`src/core/platform/prefault.hpp`): `ReleaseMappedPages` now also
drops resident file-view pages on Windows.
- On Windows it calls `VirtualUnlock` on the page-aligned range. Pages that are
  not locked leave the process working set and stay in the standby list, so
  the view keeps its contents and a later read is a soft fault. Before, it was
  a no-op there.
- New `ReleaseMappedRange(view, view_bytes, data, bytes)` releases only the
  part of `[data, data + bytes)` that lies inside the view
  `[view, view + view_bytes)`; host memory outside the view is never trimmed.

Compatibility: additive. POSIX behavior is unchanged. Callers that already
release pages after their uploads (for example Qwen3-TTS and Qwen3-ASR through
`LoadResult::ReleaseHostPages`) now also release them on Windows.
