# Component cache is the only continuation contract

Breaking: series 0.2 starts here, and an extension must request `>= 0.2.0`. The
legacy continuation cache is gone, so every text runner provides continuation
through the component cache.

Removed from `TextModelRunner`: `CreateState`, `UsesComponentCache`,
`SnapshotPayloadBytes`, `Snapshot`, `SnapshotForPersistence`, `RestoreOrFork`,
`PersistentSnapshotPayloadBytes`, `SerializePersistentSnapshot`,
`StreamPersistentSnapshot`, `RestorePersistentSnapshot`, `SnapshotSink` and the
single-boundary `PrefillThrough`.

`CreateComponentCacheResources`, `BindComponentState` and
`ReconcileComponentState` are now pure virtual. `InPassConsumer` takes only the
boundary index, because a capture always goes through the lease.

Also removed: `TextRunnerSnapshot`, `ContinuationSnapshot`,
`TextRunnerPersistenceDescriptor`, `TextRunnerDescriptor::persistence`,
`TextRunnerCapabilities::snapshot` and `fork`,
`TextRunnerRamCacheOptions::kMaxEntries`, and the
`DiskStartupStats::legacy_probe_bytes_read` and `removed_legacy_bytes` counters.
`prefix_reuse` alone now says whether a runner retains continuations.

`transfer_pool.hpp` and `snapshot_copy.hpp` move from `gufo/ext/experimental/`
to `gufo/ext/`: they are the supported helpers for a continuation adapter.
