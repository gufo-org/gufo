#include <algorithm>
#include <cstdint>

#include "src/models/qwen38_flash_next/kernels/rocm/snapshot_copy.hpp"

namespace gufo::models::qwen38_flash_next::rocm {
namespace {
__global__ void CopyAlignedSnapshot(unsigned char* destination,
                                    const unsigned char* source,
                                    std::size_t bytes) {
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  const auto stride = std::size_t{gridDim.x} * blockDim.x;
  for (auto i = thread; i < bytes / sizeof(uint4); i += stride)
    reinterpret_cast<uint4*>(destination)[i] =
        reinterpret_cast<const uint4*>(source)[i];
  for (auto i = bytes / sizeof(uint4) * sizeof(uint4) + thread; i < bytes;
       i += stride)
    destination[i] = source[i];
}
bool Aligned(const void* pointer) {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15) == 0;
}
hipError_t CopySnapshot(void* destination, const void* source,
                        std::size_t bytes, hipStream_t stream, bool download) {
  if (bytes >= 4096 && destination && source && Aligned(destination) &&
      Aligned(source)) {
    void* mapped{};
    const auto status = hipHostGetDevicePointer(
        &mapped, download ? destination : const_cast<void*>(source), 0);
    if (status == hipSuccess && mapped && Aligned(mapped)) {
      const auto blocks = static_cast<unsigned>(
          std::min<std::size_t>(160, (bytes - 1) / 4096 + 1));
      hipLaunchKernelGGL(
          CopyAlignedSnapshot, dim3(blocks), dim3(256), 0, stream,
          static_cast<unsigned char*>(download ? mapped : destination),
          static_cast<const unsigned char*>(download ? source : mapped), bytes);
      return hipGetLastError();
    }
    if (status != hipSuccess) {
      if (status != hipErrorInvalidValue)
        return status;
      // Ordinary host vectors are not registered; preserve their established
      // HIP behavior without leaving the rejected alias query sticky.
      (void)hipGetLastError();
    }
  }
  return hipMemcpyAsync(destination, source, bytes,
                        download ? hipMemcpyDeviceToHost : hipMemcpyDefault,
                        stream);
}
}  // namespace

hipError_t CopySnapshotToHost(void* destination, const void* source,
                              std::size_t bytes, hipStream_t stream) {
  return CopySnapshot(destination, source, bytes, stream, true);
}

hipError_t CopySnapshotToDevice(void* destination, const void* source,
                                std::size_t bytes, hipStream_t stream) {
  return CopySnapshot(destination, source, bytes, stream, false);
}

}  // namespace gufo::models::qwen38_flash_next::rocm
