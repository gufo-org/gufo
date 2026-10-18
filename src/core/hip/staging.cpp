#include "src/core/hip/staging.hpp"

#include <hip/hip_runtime.h>

#include <cstring>
#include <stdexcept>

#include "src/core/hip/host_allocations.hpp"

namespace gufo::hip {
namespace {
struct PinnedStorage {
  cache::ResourceCharge metadata;
  HostAllocations allocations;
  void* data{};
};
}  // namespace
cache::StagingAllocator PinnedStagingAllocator(cache::ResourceLedger& ledger) {
  return [&ledger](std::size_t bytes) {
    auto admission = ledger.Reserve(cache::ResourceCategory::kMetadata,
                                    sizeof(PinnedStorage) + 64);
    auto storage = std::make_shared<PinnedStorage>();
    storage->metadata = admission.Convert();
    const auto status = storage->allocations.Allocate(&storage->data, bytes,
                                                      hipHostMallocCoherent);
    if (status != hipSuccess)
      throw std::runtime_error(hipGetErrorString(status));
    // Commit pages once; no first-touch allocation in the transfer worker.
    std::memset(storage->data, 0, bytes);
    return cache::StagingAllocation{
        storage, {static_cast<std::uint8_t*>(storage->data), bytes}};
  };
}
}  // namespace gufo::hip
