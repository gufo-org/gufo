#include "src/mem/lendable.hpp"

#include <atomic>

namespace gufo::mem {
namespace {
std::atomic<LendableRangeFactory> factory{nullptr};
}  // namespace

void SetLendableRangeFactory(LendableRangeFactory value) noexcept {
  factory.store(value, std::memory_order_release);
}

std::shared_ptr<LendableRange> CreateLendableRange(std::size_t bytes) {
  const auto create = factory.load(std::memory_order_acquire);
  return create ? create(bytes) : nullptr;
}

}  // namespace gufo::mem
