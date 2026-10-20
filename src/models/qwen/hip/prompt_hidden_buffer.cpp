#include "src/models/qwen/hip/prompt_hidden_buffer.hpp"

#include <algorithm>
#include <cstring>

namespace gufo::hip {

void PromptHiddenBuffer::Reserve(std::size_t elements,
                                 std::size_t pinned_limit) {
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
  use_pageable_ = false;
}

float* PromptHiddenBuffer::Data() noexcept {
  return use_pageable_ ? pageable_.data() : pinned_;
}

std::span<const float> PromptHiddenBuffer::View() const noexcept {
  return {use_pageable_ ? pageable_.data() : pinned_, size_};
}

void PromptHiddenBuffer::GrowPinned(std::size_t elements) {
  auto owner = std::make_unique<HostAllocations>();
  float* data = nullptr;
  if (owner->Allocate(&data, elements * sizeof(float)) != hipSuccess) {
    // Pinned memory is an optimization; keep capturing into pageable memory.
    (void)hipGetLastError();
    MoveToPageable(elements);
    return;
  }
  if (size_ > 0) {
    std::memcpy(data, pinned_, size_ * sizeof(float));
  }
  pinned_owner_ = std::move(owner);
  pinned_ = data;
  pinned_capacity_ = elements;
}

void PromptHiddenBuffer::MoveToPageable(std::size_t elements) {
  if (!use_pageable_) {
    pageable_.assign(pinned_, pinned_ + size_);
    use_pageable_ = true;
  }
  pageable_.reserve(elements);
}

}  // namespace gufo::hip
