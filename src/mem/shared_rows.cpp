#include "src/mem/shared_rows.hpp"

#include <atomic>

namespace gufo::mem {
namespace {
std::atomic<SharedRowsFactory> factory{nullptr};
}  // namespace

void SetSharedRowsFactory(SharedRowsFactory value) noexcept {
  factory.store(value, std::memory_order_release);
}

std::shared_ptr<SharedRows> CreateSharedRows() {
  const auto create = factory.load(std::memory_order_acquire);
  return create ? create() : nullptr;
}

}  // namespace gufo::mem
