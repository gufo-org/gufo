#ifndef GUFO_CORE_HIP_SNAPSHOT_COPY_HPP_
#define GUFO_CORE_HIP_SNAPSHOT_COPY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::hip {

// Byte-exact, asynchronous snapshot download. Aligned registered host backing
// uses its device alias; pageable/unaligned buffers retain HIP's copy path.
// The caller retains both buffers until the stream's completion has settled.
[[nodiscard]] hipError_t CopySnapshotToHost(void* destination,
                                            const void* source,
                                            std::size_t bytes,
                                            hipStream_t stream);

// Registered host snapshots use the same byte-exact kernel for restore.
// Other sources retain HIP's inferred direction, including borrowed GPU rows.
[[nodiscard]] hipError_t CopySnapshotToDevice(void* destination,
                                              const void* source,
                                              std::size_t bytes,
                                              hipStream_t stream);

}  // namespace gufo::hip
#endif
