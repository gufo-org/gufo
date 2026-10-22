#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"

#include <algorithm>
#include <cstring>
#include <new>

namespace gufo::hip {

void PromptHiddenBuffer::Reserve(std::size_t elements, std::size_t pinned_limit,
                                 cache::ResourceLedger* ledger) {
  if (ledger) {
    ledger_ = ledger;
  }
  pinned_limit_ = pinned_limit;
  if (use_pageable_ || elements > pinned_limit_) {
    MoveToPageable(elements);
    return;
  }
  if (elements > pinned_capacity_) {
    GrowPinned(elements);
  }
}

void PromptHiddenBuffer::Resize(std::size_t elements) {
  if (!use_pageable_ && elements > pinned_capacity_) {
    if (elements <= pinned_limit_) {
      GrowPinned(
          std::min(std::max(elements, 2 * pinned_capacity_), pinned_limit_));
    } else {
      MoveToPageable(elements);
    }
  }
  if (use_pageable_) {
    pageable_.resize(elements);
  }
  size_ = elements;
}

void PromptHiddenBuffer::Clear() noexcept {
  size_ = 0;
  pageable_.clear();
  use_pageable_ = pinned_failed_;
}

float* PromptHiddenBuffer::Data() noexcept {
  return use_pageable_ ? pageable_.data() : pinned_;
}

std::span<const float> PromptHiddenBuffer::View() const noexcept {
  return {use_pageable_ ? pageable_.data() : pinned_, size_};
}

void PromptHiddenBuffer::GrowPinned(std::size_t elements) {
  // Locked pages are host RAM the cache cannot reclaim. Admit them before
  // allocating, as streaming.cpp admits staging; a failed allocation drops
  // the admission.
  cache::ResourceReservation admission;
  if (ledger_) {
    try {
      admission = ledger_->Reserve(cache::ResourceCategory::kMetadata,
                                   elements * sizeof(float));
    } catch (const std::bad_alloc&) {
      StopPinning(elements);
      return;
    }
  }
  auto owner = std::make_unique<HostAllocations>();
  float* data = nullptr;
  if (owner->Allocate(&data, elements * sizeof(float)) != hipSuccess) {
    // Pinned memory is an optimization; keep capturing into pageable memory.
    (void)hipGetLastError();
    StopPinning(elements);
    return;
  }
  // Callers have synchronized the stream: no copy is still in flight to the
  // old pages, which this reads and then frees.
  if (size_ > 0) {
    std::memcpy(data, pinned_, size_ * sizeof(float));
  }
  auto charge = admission ? admission.Convert() : cache::ResourceCharge{};
  // Free the previous pages before releasing their charge.
  pinned_owner_ = std::move(owner);
  charge_ = std::move(charge);
  pinned_ = data;
  pinned_capacity_ = elements;
}

void PromptHiddenBuffer::StopPinning(std::size_t elements) {
  pinned_failed_ = true;
  MoveToPageable(elements);
  // The rows are copied and no capture pins again: free the pages, then
  // release their charge.
  pinned_owner_.reset();
  charge_ = {};
  pinned_ = nullptr;
  pinned_capacity_ = 0;
}

void PromptHiddenBuffer::MoveToPageable(std::size_t elements) {
  if (!use_pageable_) {
    pageable_.assign(pinned_, pinned_ + size_);
    use_pageable_ = true;
  }
  pageable_.reserve(elements);
}

}  // namespace gufo::hip
