# Component cache transfer stream count

HIP component cache (`src/core/hip/component_cache_resources.hpp`):
`ComponentCacheHooks::transfer_streams` sets how many transfer streams
`CreateHipComponentCacheResources` creates for captures, restores and idle
spill. Each stream keeps HIP queue memory resident, which on a unified-memory
host comes out of the memory left for the model's own buffers. A capture or
restore batch leases up to 16 streams; with fewer free, concurrent batches run
narrower or wait. Zero is rejected.

Compatibility: additive. The field defaults to 64, the previous fixed count.
