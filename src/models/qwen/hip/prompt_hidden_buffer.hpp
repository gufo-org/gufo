#ifndef GUFO_MODELS_QWEN_HIP_PROMPT_HIDDEN_BUFFER_HPP_
#define GUFO_MODELS_QWEN_HIP_PROMPT_HIDDEN_BUFFER_HPP_

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "src/cache/ledger.hpp"
#include "src/core/hip/host_allocations.hpp"

namespace gufo::hip {

/// Host destination of the DFlash2 prompt capture. Up to a bound (one prompt
/// batch of every captured layer) it is pinned, so the per-layer device-to-host
/// copies avoid HIP's staged pageable path; a larger request (a full prompt
/// from a non-serve caller) uses ordinary pageable memory instead, so the
/// pinned size stays bounded. Grow-only like the std::vector it replaces:
/// Clear() keeps the pinned capacity. Resize() does not zero new elements;
/// every capture fills the rows it adds. With a ledger, the pinned bytes are
/// charged as RAM before they are allocated, and a refused charge captures
/// into pageable memory instead.
class PromptHiddenBuffer {
public:
  /// Sets the pinned bound for the next prompt and makes room for `elements`.
  /// A ledger, once given, charges this and every later pinned allocation.
  void Reserve(std::size_t elements, std::size_t pinned_limit,
               cache::ResourceLedger* ledger = nullptr);
  void Resize(std::size_t elements);
  void Clear() noexcept;
  [[nodiscard]] float* Data() noexcept;
  [[nodiscard]] std::size_t Size() const noexcept { return size_; }
  [[nodiscard]] std::span<const float> View() const noexcept;

private:
  void GrowPinned(std::size_t elements);
  void MoveToPageable(std::size_t elements);

  cache::ResourceLedger* ledger_{};
  // Declared before its allocation, which is freed first.
  cache::ResourceCharge charge_;
  std::unique_ptr<HostAllocations> pinned_owner_;
  float* pinned_{};
  std::size_t pinned_capacity_{};
  std::size_t pinned_limit_{};
  std::vector<float> pageable_;
  bool use_pageable_{};
  std::size_t size_{};
};

}  // namespace gufo::hip
#endif
