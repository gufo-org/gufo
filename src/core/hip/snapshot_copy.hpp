#ifndef GUFO_CORE_HIP_SNAPSHOT_COPY_HPP_
#define GUFO_CORE_HIP_SNAPSHOT_COPY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::hip {
class TransferStream;

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

// The same restore through a transfer lease. An OnEngine lease copies on its
// copy engine what it can read in place: device memory, and registered host
// memory whose device alias is its own address. Other sources, and any lease
// without an engine, take the stream path above on the lease's Native
// stream. Both settle with the lease's completion.
[[nodiscard]] hipError_t CopySnapshotToDevice(void* destination,
                                              const void* source,
                                              std::size_t bytes,
                                              TransferStream& transfer);

}  // namespace gufo::hip
#endif
