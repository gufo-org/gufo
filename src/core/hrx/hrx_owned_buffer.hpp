#ifndef GUFO_CORE_HRX_HRX_OWNED_BUFFER_HPP_
#define GUFO_CORE_HRX_HRX_OWNED_BUFFER_HPP_

#include <hrx/hrx_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "src/core/hrx/hrx_buffer_binding.hpp"

namespace gufo::hrx {

/// Move-only owner for one HRX allocation.
class HrxOwnedBuffer {
public:
  HrxOwnedBuffer() = default;
  ~HrxOwnedBuffer();

  HrxOwnedBuffer(const HrxOwnedBuffer&) = delete;
  HrxOwnedBuffer& operator=(const HrxOwnedBuffer&) = delete;
  HrxOwnedBuffer(HrxOwnedBuffer&& other) noexcept;
  HrxOwnedBuffer& operator=(HrxOwnedBuffer&& other) noexcept;

  [[nodiscard]] static std::optional<HrxOwnedBuffer> Allocate(
      hrx_stream_t stream, std::size_t size, std::string* error_msg = nullptr);

  [[nodiscard]] hrx_buffer_t Get() const noexcept { return buffer_; }
  [[nodiscard]] std::size_t Size() const noexcept { return size_; }
  [[nodiscard]] bool IsValid() const noexcept {
    return buffer_ != nullptr && size_ != 0;
  }
  [[nodiscard]] HrxBufferBinding Binding() const noexcept {
    return {.buffer = buffer_, .offset = 0, .length = size_};
  }
  [[nodiscard]] std::optional<HrxBufferBinding> Slice(
      std::size_t offset, std::size_t length) const noexcept;

  void Reset() noexcept;

private:
  HrxOwnedBuffer(hrx_buffer_t buffer, std::size_t size) noexcept
      : buffer_(buffer), size_(size) {}

  hrx_buffer_t buffer_{nullptr};
  std::size_t size_{0};
};

/// Checked synchronous transfers. These are initialization/readback helpers;
/// model execution must not use them as a hidden device-to-device fallback.
[[nodiscard]] bool HrxCopyFromHost(hrx_device_t device, const void* source,
                                   const HrxBufferBinding& destination,
                                   std::size_t bytes,
                                   std::string* error_msg = nullptr);
[[nodiscard]] bool HrxCopyToHost(hrx_device_t device,
                                 const HrxBufferBinding& source,
                                 void* destination, std::size_t bytes,
                                 std::string* error_msg = nullptr);

/// Native fill implemented with the established HRX graph fill-node API.
[[nodiscard]] bool HrxFillBuffer(hrx_device_t device, hrx_stream_t stream,
                                 const HrxBufferBinding& destination,
                                 std::uint32_t pattern,
                                 std::string* error_msg = nullptr);

/// Device copies are supplied by the packaged qwen_copy_f32 artifact rather
/// than an assumed runtime API symbol.
inline constexpr bool kHrxNativeDeviceCopyAvailable = true;
inline constexpr bool kHrxNativeDeviceFillAvailable = true;

}  // namespace gufo::hrx

#endif  // GUFO_CORE_HRX_HRX_OWNED_BUFFER_HPP_
