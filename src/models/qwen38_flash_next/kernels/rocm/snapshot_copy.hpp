#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SNAPSHOT_COPY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_SNAPSHOT_COPY_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>

namespace gufo::models::qwen38_flash_next::rocm {

// Byte-exact, asynchronous snapshot download. Aligned registered host backing
// uses its device alias; pageable/unaligned buffers retain HIP's copy path.
// The caller retains both buffers until the stream's completion has settled.
[[nodiscard]] hipError_t CopySnapshotToHost(void* destination,
                                            const void* source,
                                            std::size_t bytes,
                                            hipStream_t stream);

}  // namespace gufo::models::qwen38_flash_next::rocm
#endif
