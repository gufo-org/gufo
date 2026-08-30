#include "src/core/hrx/hrx_owned_buffer.hpp"

#include <utility>

#include "src/core/hrx/hrx_graph_executor.hpp"

namespace gufo::hrx {
namespace {

void ClearError(std::string* error_msg) {
  if (error_msg != nullptr) {
    error_msg->clear();
  }
}

bool Reject(const char* message, std::string* error_msg) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
  return false;
}

bool RejectStatus(hrx_status_t status, const char* operation,
                  std::string* error_msg) {
  char* message = nullptr;
  std::size_t message_length = 0;
  hrx_status_to_string(status, &message, &message_length);
  if (error_msg != nullptr) {
    *error_msg = operation;
    *error_msg += ": ";
    *error_msg += message != nullptr ? message : "unknown HRX error";
  }
  hrx_status_free_message(message);
  hrx_status_ignore(status);
  return false;
}

}  // namespace

HrxOwnedBuffer::~HrxOwnedBuffer() {
  Reset();
}

HrxOwnedBuffer::HrxOwnedBuffer(HrxOwnedBuffer&& other) noexcept
    : buffer_(std::exchange(other.buffer_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

HrxOwnedBuffer& HrxOwnedBuffer::operator=(HrxOwnedBuffer&& other) noexcept {
  if (this != &other) {
    Reset();
    buffer_ = std::exchange(other.buffer_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

std::optional<HrxOwnedBuffer> HrxOwnedBuffer::Allocate(hrx_stream_t stream,
                                                       std::size_t size,
                                                       std::string* error_msg) {
  if (stream == nullptr) {
    Reject("HRX allocation requires a stream", error_msg);
    return std::nullopt;
  }
  if (size == 0) {
    Reject("HRX allocation size must be non-zero", error_msg);
    return std::nullopt;
  }

  hrx_buffer_t buffer = nullptr;
  const auto status =
      hrx_buffer_allocate(stream, size, HRX_MEMORY_TYPE_DEVICE_LOCAL,
                          HRX_BUFFER_USAGE_DEFAULT, &buffer);
  if (!hrx_status_is_ok(status)) {
    RejectStatus(status, "hrx_buffer_allocate", error_msg);
    return std::nullopt;
  }
  if (buffer == nullptr) {
    Reject("hrx_buffer_allocate returned a null buffer", error_msg);
    return std::nullopt;
  }

  ClearError(error_msg);
  return HrxOwnedBuffer(buffer, size);
}

std::optional<HrxBufferBinding> HrxOwnedBuffer::Slice(
    std::size_t offset, std::size_t length) const noexcept {
  if (!IsValid() || length == 0 || offset > size_ || length > size_ - offset) {
    return std::nullopt;
  }
  return HrxBufferBinding{
      .buffer = buffer_, .offset = offset, .length = length};
}

void HrxOwnedBuffer::Reset() noexcept {
  if (buffer_ != nullptr) {
    hrx_buffer_release(buffer_);
    buffer_ = nullptr;
  }
  size_ = 0;
}

bool HrxCopyFromHost(hrx_device_t device, const void* source,
                     const HrxBufferBinding& destination, std::size_t bytes,
                     std::string* error_msg) {
  hrx_buffer_ref_t target{};
  if (device == nullptr || source == nullptr) {
    return Reject("HRX H2D copy requires a device and source", error_msg);
  }
  if (!TryMakeBufferRef(destination, bytes, &target)) {
    return Reject("HRX H2D destination range is invalid", error_msg);
  }
  const auto status = hrx_synchronous_h2d(device, source, target.buffer,
                                          target.offset, target.length);
  if (!hrx_status_is_ok(status)) {
    return RejectStatus(status, "hrx_synchronous_h2d", error_msg);
  }
  ClearError(error_msg);
  return true;
}

bool HrxCopyToHost(hrx_device_t device, const HrxBufferBinding& source,
                   void* destination, std::size_t bytes,
                   std::string* error_msg) {
  hrx_buffer_ref_t input{};
  if (device == nullptr || destination == nullptr) {
    return Reject("HRX D2H copy requires a device and destination", error_msg);
  }
  if (!TryMakeBufferRef(source, bytes, &input)) {
    return Reject("HRX D2H source range is invalid", error_msg);
  }
  const auto status = hrx_synchronous_d2h(device, input.buffer, input.offset,
                                          destination, input.length);
  if (!hrx_status_is_ok(status)) {
    return RejectStatus(status, "hrx_synchronous_d2h", error_msg);
  }
  ClearError(error_msg);
  return true;
}

bool HrxFillBuffer(hrx_device_t device, hrx_stream_t stream,
                   const HrxBufferBinding& destination, std::uint32_t pattern,
                   std::string* error_msg) {
  if (device == nullptr || stream == nullptr || !destination.IsValid() ||
      (destination.length % sizeof(pattern)) != 0) {
    return Reject(
        "HRX fill requires an aligned destination, device, and stream",
        error_msg);
  }

  // Fill is a mandatory memory-management primitive. It must remain usable
  // when optional model graph capture is disabled via GUFO_ENABLE_HRX_GRAPH.
  HrxGraphDecodeExecutor graph(/*ignore_graph_kill_switch=*/true);
  const HrxGraphCaptureKey key{static_cast<std::uint64_t>(destination.length),
                               pattern};
  if (!graph.InitializeGraph(device, key)) {
    return Reject("failed to create HRX fill graph", error_msg);
  }
  hrx_buffer_ref_t destination_ref{};
  if (!TryMakeBufferRef(destination, destination.length, &destination_ref) ||
      !graph.AddFillBufferNode(destination_ref, pattern, sizeof(pattern)) ||
      !graph.Instantiate() || !graph.Launch(stream)) {
    return Reject("failed to build or launch HRX fill graph", error_msg);
  }
  const auto status = hrx_stream_synchronize(stream);
  if (!hrx_status_is_ok(status)) {
    return RejectStatus(status, "hrx_stream_synchronize", error_msg);
  }
  ClearError(error_msg);
  return true;
}

}  // namespace gufo::hrx
