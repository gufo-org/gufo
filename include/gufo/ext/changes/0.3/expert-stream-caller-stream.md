# Expert store on the caller's stream

Additive, experimental: `expert_stream::Store` serves models that run on
their own streams and whose kernels read past an expert's end.

- `Store::Prepare` takes the caller's `hipStream_t` (last, default the null
  stream). It reads the routes once the work queued there before the call has
  run, and queues the table, split-route and cache-admission copies on it, so
  kernels queued on it afterwards see them. A `routes_ready` event is recorded
  on that stream. The default keeps the previous null-stream behaviour.
- `Layout::read_margin_bytes`: readable bytes kernels may touch past the end
  of an expert's last piece. Every cache chunk and stage buffer keeps this
  much beyond its last expert; the prefetch stage's margin counts against the
  grant `Store::Fit` takes.

Migration: none; existing callers keep the null stream and no margin.
