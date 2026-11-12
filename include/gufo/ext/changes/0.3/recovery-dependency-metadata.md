# Distinct dependency observations during disk recovery

`DiskStartupStats` adds `dependency_references` and
`distinct_dependency_files`. Cold recovery now reuses one actual file metadata
observation for each distinct `(namespace, file ID)`; `dependency_stats` counts
those calls rather than every reference. Directory enumeration performs
separate metadata calls and is not included in these counters.

The on-disk format, managed-byte quota and restore validation contracts are
unchanged. The temporary observations are discarded after recovery. The
`GUFO_CACHE_TESTING` API also exposes a current-thread reference-validation
scope hook for deterministic allocation-failure tests; it is absent from
production builds.
