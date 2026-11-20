#include <algorithm>
#include <cstdint>

#include "src/core/hip/snapshot_copy.hpp"
#include "src/core/hip/transfer_pool.hpp"

namespace gufo::hip {
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
// Device alias of `host + offset` equals `mapped + offset`.
bool AliasFollows(const unsigned char* host, const unsigned char* mapped,
                  std::size_t offset) {
  void* alias{};
  if (hipHostGetDevicePointer(&alias, const_cast<unsigned char*>(host) + offset,
                              0) != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  return alias == mapped + offset;
}
// Bytes from `host` whose device alias continues `mapped`. Adjacent host
// registrations need not have adjacent aliases (Windows maps each one on its
// own), so a host-contiguous range can leave its first mapping.
std::size_t ContiguousAlias(const unsigned char* host,
                            const unsigned char* mapped, std::size_t bytes) {
  if (AliasFollows(host, mapped, bytes - 1))
    return bytes;
  // Registrations are page granular: find the first page that does not follow.
  // The page holding the last byte does not; the one holding `host` does.
  constexpr std::size_t page = 4096;
  const auto base = reinterpret_cast<std::uintptr_t>(host);
  auto good = base / page + 1, bad = (base + bytes - 1) / page;
  while (good < bad) {
    const auto middle = good + (bad - good) / 2;
    if (AliasFollows(host, mapped, middle * page - base))
      good = middle + 1;
    else
      bad = middle;
  }
  return bad * page - base;
}
// Bytes from `host` inside the registration that holds it. hipMemcpyAsync
// rejects a host range spanning two registrations even when their aliases are
// adjacent, as on Linux, where each alias equals its host address.
std::size_t RegisteredBytes(const unsigned char* host, std::size_t bytes) {
  void* start{};
  std::size_t size{};
  auto* pointer = const_cast<unsigned char*>(host);
  if (hipPointerGetAttribute(&start, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                             pointer) != hipSuccess ||
      hipPointerGetAttribute(&size, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                             pointer) != hipSuccess) {
    (void)hipGetLastError();
    return bytes;
  }
  const auto* first = static_cast<const unsigned char*>(start);
  return host >= first && host < first + size
             ? std::min<std::size_t>(bytes, first + size - host)
             : bytes;
}
hipError_t CopySnapshot(void* destination, const void* source,
                        std::size_t bytes, hipStream_t stream, bool download) {
  if (bytes && destination && source) {
    auto* host =
        static_cast<const unsigned char*>(download ? destination : source);
    void* mapped{};
    const auto status =
        hipHostGetDevicePointer(&mapped, const_cast<unsigned char*>(host), 0);
    // Restore joins host-adjacent chunks, which may come from separate
    // registrations. Neither the alias kernel nor hipMemcpyAsync may cross one.
    if (status == hipSuccess && mapped) {
      const auto first =
          ContiguousAlias(host, static_cast<const unsigned char*>(mapped),
                          RegisteredBytes(host, bytes));
      if (first < bytes) {
        auto* d = static_cast<unsigned char*>(destination);
        auto* s = static_cast<const unsigned char*>(source);
        if (const auto error = CopySnapshot(d, s, first, stream, download);
            error != hipSuccess)
          return error;
        return CopySnapshot(d + first, s + first, bytes - first, stream,
                            download);
      }
    }
    if (status == hipSuccess && mapped && bytes >= 4096 &&
        Aligned(destination) && Aligned(source) && Aligned(mapped)) {
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
// Bytes from `source` the copy engine reaches at that address, all inside one
// HIP allocation or registration; zero when it cannot reach `source`.
std::size_t EngineReadable(const unsigned char* source, std::size_t bytes) {
  auto* pointer = const_cast<unsigned char*>(source);
  hipPointerAttribute_t attributes{};
  void* start{};
  std::size_t size{};
  if (hipPointerGetAttributes(&attributes, pointer) != hipSuccess ||
      hipPointerGetAttribute(&start, HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                             pointer) != hipSuccess ||
      hipPointerGetAttribute(&size, HIP_POINTER_ATTRIBUTE_RANGE_SIZE,
                             pointer) != hipSuccess) {
    (void)hipGetLastError();
    return 0;
  }
  const bool readable =
      attributes.type == hipMemoryTypeDevice ||
      (attributes.type == hipMemoryTypeHost &&
       attributes.devicePointer == static_cast<void*>(pointer));
  const auto* first = static_cast<const unsigned char*>(start);
  if (!readable || source < first || source >= first + size)
    return 0;
  return std::min<std::size_t>(bytes, first + size - source);
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

hipError_t CopySnapshotToDevice(void* destination, const void* source,
                                std::size_t bytes, TransferStream& transfer) {
  if (!transfer.OnEngine() || !bytes || !destination || !source)
    return CopySnapshotToDevice(destination, source, bytes, transfer.Native());
  auto* d = static_cast<unsigned char*>(destination);
  auto* s = static_cast<const unsigned char*>(source);
  while (bytes) {
    // The engine writes by address too.
    const auto run =
        std::min(EngineReadable(s, bytes), EngineReadable(d, bytes));
    if (!run)
      return CopySnapshotToDevice(d, s, bytes, transfer.Native());
    if (!transfer.EngineCopy(d, s, run))
      return hipErrorLaunchFailure;
    d += run;
    s += run;
    bytes -= run;
  }
  return hipSuccess;
}

}  // namespace gufo::hip
