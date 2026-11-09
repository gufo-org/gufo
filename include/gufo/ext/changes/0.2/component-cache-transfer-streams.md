# Component cache transfer stream count

HIP component cache (`src/core/hip/component_cache_resources.hpp`):
`ComponentCacheHooks::transfer_streams` sets how many transfer streams
`CreateHipComponentCacheResources` creates for captures, restores and idle
spill. Each stream keeps HIP queue memory resident, which on a unified-memory
host comes out of the memory left for the model's own buffers. Each serving
session holds one preservation stream, so providers must leave additional streams for
captures, restores and idle spill. Batches lease up to 16 streams and drain their
own work before retrying an exhausted pool; acquisition does not wait for other
users and can fail. Zero is rejected.

Compatibility: additive. The field defaults to 64, the previous fixed count.
