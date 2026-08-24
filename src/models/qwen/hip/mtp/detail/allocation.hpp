#ifndef STRIX_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
#define STRIX_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

namespace strix::hip::detail {

[[nodiscard]] inline void* AllocateDevice(std::size_t bytes) {
  void* pointer = nullptr;
  const auto error = hipMalloc(&pointer, bytes);
  if (error != hipSuccess) {
    throw std::runtime_error(std::string("MTP hipMalloc failed: ") +
                             hipGetErrorString(error));
  }
  return pointer;
}

}  // namespace strix::hip::detail
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_MODELS_QWEN_HIP_MTP_DETAIL_ALLOCATION_HPP_
