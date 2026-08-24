#ifndef STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_BUFFER_HPP_
#define STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_BUFFER_HPP_

#include <cstddef>
#include <cstdlib>
#include <span>
#include <utility>
#include <vector>

#if defined(ENGINE_ENABLE_HIP)
#include <hip/hip_runtime.h>

#include "src/core/hip/hip_utils.hpp"

namespace strix::test {

/// Small move-only owner for test-only HIP allocations.
template<typename T>
class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t size) : size_(size) {
    if (size_ != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_),
                          size_ * sizeof(T)));
    }
  }

  explicit DeviceBuffer(std::span<const T> values)
      : DeviceBuffer(values.size()) {
    CopyFrom(values);
  }

  explicit DeviceBuffer(const std::vector<T>& values)
      : DeviceBuffer(std::span<const T>(values)) {}

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)hipFree(data_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)) {}

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      if (data_ != nullptr) {
        (void)hipFree(data_);
      }
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
    }
    return *this;
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void CopyFrom(std::span<const T> values) {
    if (values.size() != size_) {
      std::abort();
    }
    if (!values.empty()) {
      HIP_CHECK(hipMemcpy(data_, values.data(), size_ * sizeof(T),
                          hipMemcpyHostToDevice));
    }
  }

  [[nodiscard]] std::vector<T> CopyToHost() const {
    std::vector<T> values(size_);
    if (!values.empty()) {
      HIP_CHECK(hipMemcpy(values.data(), data_, size_ * sizeof(T),
                          hipMemcpyDeviceToHost));
    }
    return values;
  }

private:
  T* data_{nullptr};
  std::size_t size_{0};
};

}  // namespace strix::test
#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // STRIX_TESTS_MODELS_QWEN_HIP_SUPPORT_DEVICE_BUFFER_HPP_
