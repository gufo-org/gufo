#if defined(ENGINE_ENABLE_HIP)
#include "src/core/hip/device.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <utility>

#include "src/core/hip/hip_utils.hpp"

namespace gufo::hip {
namespace {

/// Staging chunk for peer-less copies: large enough to amortize per-chunk
/// synchronization, small enough to keep the pinned footprint bounded.
constexpr std::size_t kStagingChunkBytes = 8ULL << 20;

[[nodiscard]] int PeerKey(int destination, int source) noexcept {
  return destination * 64 + source;
}

std::mutex& PeerMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_set<int>& EnabledPeerPairs() {
  static std::unordered_set<int> pairs;
  return pairs;
}

/// True when `destination` can directly address `source` allocations. Enables
/// peer access once per ordered pair and remembers the result.
[[nodiscard]] bool PeerAccessReady(int destination, int source) {
  int can = 0;
  if (hipDeviceCanAccessPeer(&can, destination, source) != hipSuccess ||
      can == 0) {
    return false;
  }
  const int key = PeerKey(destination, source);
  {
    std::lock_guard<std::mutex> lock(PeerMutex());
    if (EnabledPeerPairs().count(key) != 0)
      return true;
  }
  DeviceGuard guard(destination);
  const auto status = hipDeviceEnablePeerAccess(source, 0);
  if (status != hipSuccess && status != hipErrorPeerAccessAlreadyEnabled)
    return false;
  // hipDeviceEnablePeerAccess leaves a sticky error on the already-enabled
  // path; clear it so later checks see a clean thread state.
  (void)hipGetLastError();
  std::lock_guard<std::mutex> lock(PeerMutex());
  EnabledPeerPairs().insert(key);
  return true;
}

/// Thread-local pinned staging buffer, grown on demand.
[[nodiscard]] void* StagingBuffer(std::size_t bytes) {
  thread_local void* staging = nullptr;
  thread_local std::size_t capacity = 0;
  if (capacity < bytes) {
    if (staging != nullptr)
      (void)hipHostFree(staging);
    staging = nullptr;
    HIP_CHECK(hipHostMalloc(&staging, bytes));
    capacity = bytes;
  }
  return staging;
}

void DrainDevice(int device) {
  DeviceGuard guard(device);
  HIP_CHECK(hipDeviceSynchronize());
}

/// One staged chunk: source device -> pinned host -> destination device.
void StageChunk(void* destination, int destination_device, const void* source,
                int source_device, std::size_t bytes, void* staging) {
  {
    DeviceGuard guard(source_device);
    HIP_CHECK(hipMemcpy(staging, source, bytes, hipMemcpyDeviceToHost));
  }
  {
    DeviceGuard guard(destination_device);
    HIP_CHECK(hipMemcpy(destination, staging, bytes, hipMemcpyHostToDevice));
  }
}

}  // namespace

int GetDeviceCount() noexcept {
  int count = 0;
  if (hipGetDeviceCount(&count) != hipSuccess)
    return 0;
  return count;
}

int CurrentDevice() noexcept {
  int device = -1;
  if (hipGetDevice(&device) != hipSuccess)
    return -1;
  return device;
}

DeviceGuard::DeviceGuard(int device) {
  previous_ = CurrentDevice();
  if (previous_ != device)
    switched_ = hipSetDevice(device) == hipSuccess;
}

DeviceGuard::~DeviceGuard() {
  if (switched_ && previous_ >= 0)
    (void)hipSetDevice(previous_);
}

void CrossDeviceCopy(void* destination, int destination_device,
                     const void* source, int source_device,
                     std::size_t bytes) {
  if (bytes == 0)
    return;
  if (source_device == destination_device) {
    DeviceGuard guard(destination_device);
    HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyDeviceToDevice));
    return;
  }
  DrainDevice(source_device);
  if (PeerAccessReady(destination_device, source_device)) {
    DeviceGuard guard(destination_device);
    HIP_CHECK(hipMemcpy(destination, source, bytes, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    return;
  }
  void* staging = StagingBuffer(kStagingChunkBytes);
  for (std::size_t offset = 0; offset < bytes;) {
    const auto chunk = std::min(kStagingChunkBytes, bytes - offset);
    StageChunk(static_cast<std::uint8_t*>(destination) + offset,
               destination_device,
               static_cast<const std::uint8_t*>(source) + offset,
               source_device, chunk, staging);
    offset += chunk;
  }
}

void CrossDeviceCopy2D(void* destination, int destination_device,
                       std::size_t destination_pitch, const void* source,
                       int source_device, std::size_t source_pitch,
                       std::size_t width_bytes, std::size_t height) {
  if (width_bytes == 0 || height == 0)
    return;
  if (source_device == destination_device) {
    DeviceGuard guard(destination_device);
    HIP_CHECK(hipMemcpy2D(destination, destination_pitch, source, source_pitch,
                          width_bytes, height, hipMemcpyDeviceToDevice));
    return;
  }
  DrainDevice(source_device);
  if (PeerAccessReady(destination_device, source_device)) {
    DeviceGuard guard(destination_device);
    HIP_CHECK(hipMemcpy2D(destination, destination_pitch, source, source_pitch,
                          width_bytes, height, hipMemcpyDeviceToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    return;
  }
  void* staging = StagingBuffer(std::min(width_bytes, kStagingChunkBytes));
  for (std::size_t row = 0; row < height; ++row) {
    auto* row_destination = static_cast<std::uint8_t*>(destination) +
                            row * destination_pitch;
    const auto* row_source = static_cast<const std::uint8_t*>(source) +
                             row * source_pitch;
    for (std::size_t offset = 0; offset < width_bytes;) {
      const auto chunk = std::min(kStagingChunkBytes, width_bytes - offset);
      StageChunk(row_destination + offset, destination_device,
                 row_source + offset, source_device, chunk, staging);
      offset += chunk;
    }
  }
}

}  // namespace gufo::hip
#endif  // defined(ENGINE_ENABLE_HIP)
