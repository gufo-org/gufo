# Shared native transfer streams

`TransferPool(capacity, native_streams)` preallocates independent event leases
across fewer native FIFO streams. Its one-argument constructor keeps exclusive
stream/event pairs. Shared leases settle at their own system-visible fence;
failed lanes are quarantined and exhausted degraded pools throw rather than
wait for capacity that cannot return. Buffers must outlive settled completion,
and shared-stream submissions must not introduce circular inter-lease waits.

`ComponentCacheHooks::transfer_leases` and `PlanComponentTransfers(sessions)`
size eager native resources independently of outstanding transfers.
`TextModelRunner::CreateComponentCacheResourcesForSessions` receives the actual
serving-slot count, with a default forwarding implementation for existing
providers. Flash-Next selects shared resources through this factory.
