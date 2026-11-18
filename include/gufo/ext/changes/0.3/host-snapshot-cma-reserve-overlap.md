# Host snapshot ceiling: CMA pages and the reserve overlap

Additive: `gufo/ext/host_budget.hpp` now exposes the host sample the snapshot
budgets derive from (#1198).

- `HostMemorySample` holds MemAvailable, CmaFree, the cgroup v2 room and the
  state ledger's outstanding promises; `MeminfoSample(text)` parses the first
  two from /proc/meminfo text and `SampleHostMemory()` samples this host.
- `PinnableBytes(sample)`, `SnapshotBudgetBytes(sample)` and
  `SnapshotCeilingBytes(sample)` compute the budgets from a sample;
  `DescribeHostMemory(sample)` formats it for refusal logs.

Behavior: `HostSnapshotCeilingBytes()` now subtracts the larger of CmaFree and
`kHostSnapshotHeadroomBytes` from MemAvailable instead of their sum. Free CMA
pages (KHO scratch on Ubuntu 26.04's 7.0 kernel, up to 8.7 GiB after boot)
hold only movable pages, so the OS keeps its reserve there; pinned and GPU
memory still never count them. A cgroup limit keeps the full headroom below
it. Without free CMA pages every result is unchanged.
`HostSnapshotBudgetBytes()` and `MeminfoAvailableBytes()` are unchanged.

Migration: none.
