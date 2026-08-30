#ifndef GUFO_CORE_HRX_HRX_BUFFER_BINDING_HPP_
#define GUFO_CORE_HRX_HRX_BUFFER_BINDING_HPP_

#include <hrx/hrx_runtime.h>

#include <cstddef>
#include <limits>

namespace gufo::hrx {

/// A validated byte range within an HRX buffer.
///
/// The range may represent a complete allocation or a tensor payload imported
/// from the middle of a mapped GGUF shard. It does not encode tensor shape,
/// element type, or arbitrary stride metadata.
struct HrxBufferBinding {
  hrx_buffer_t buffer{nullptr};
  std::size_t offset{0};
  std::size_t length{0};

  [[nodiscard]] bool IsValid() const noexcept {
    return buffer != nullptr && length != 0 &&
           offset <= std::numeric_limits<std::size_t>::max() - length;
  }
};

/// Narrows a binding to the exact byte range required by one artifact operand.
/// The original imported-buffer offset is preserved.
[[nodiscard]] inline bool TryMakeBufferRef(
    const HrxBufferBinding& binding, std::size_t required_bytes,
    hrx_buffer_ref_t* buffer_ref) noexcept {
  if (buffer_ref == nullptr || required_bytes == 0 || !binding.IsValid() ||
      required_bytes > binding.length ||
      binding.offset >
          std::numeric_limits<std::size_t>::max() - required_bytes) {
    return false;
  }
  *buffer_ref = {binding.buffer, binding.offset, required_bytes};
  return true;
}

}  // namespace gufo::hrx

#endif  // GUFO_CORE_HRX_HRX_BUFFER_BINDING_HPP_
