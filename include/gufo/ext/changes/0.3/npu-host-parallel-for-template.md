# NPU host ParallelFor template, row rotary and bf16 softmax

NPU host helpers (`src/core/npu/host_ops.hpp`, namespace `gufo::npu::host`,
reached through `include/gufo/ext/experimental/npu.hpp`):
- `ParallelFor(n, grain, fn)` is now a function template taking any callable
  `fn(begin, end)`. Calls that run on the calling thread (one chunk, or nested
  inside a pool task) invoke `fn` directly; pool runs go through the new
  `ParallelForPool`, which keeps the former `std::function` signature.
- `RunsInline(n, grain)`: whether `ParallelFor` with these arguments runs on
  the calling thread.
- `RopeNeoxRow(x, heads, head_dim, rotary, cos, sin)`: `RopeNeox` of one row
  from precomputed cos/sin of its position.
- `SoftmaxRowBf16(x, n, lo, hi, tmp)`: in-place softmax of the bf16 row range
  `[lo, hi)`, zeroing the rest of the row's `n` values.

Compatibility: additive at source level. Existing calls to `ParallelFor`
with a lambda or `std::function` compile unchanged; code that took the
address of `ParallelFor` must now name `ParallelForPool`.
