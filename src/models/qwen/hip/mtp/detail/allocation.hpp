#ifndef GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/managed_alloc.hpp"

namespace gufo::hip::detail {

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = gufo::hip::AllocateDevice(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP allocation failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

}  // namespace gufo::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
