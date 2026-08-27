#ifndef GUFO_MODELS_QWEN_HRX_QWEN_HRX_TENSOR_BINDING_HPP_
#define GUFO_MODELS_QWEN_HRX_QWEN_HRX_TENSOR_BINDING_HPP_

#include <cstddef>
#include <limits>
#include <optional>
#include <string>

#include "src/models/qwen/state.hpp"

namespace gufo::hrx {

/// Returns rows * columns, or nullopt for zero dimensions or size_t overflow.
[[nodiscard]] inline std::optional<std::size_t> HrxMatrixElementCount(
    std::size_t rows, std::size_t columns) noexcept {
  if (rows == 0 || columns == 0 ||
      rows > std::numeric_limits<std::size_t>::max() / columns) {
    return std::nullopt;
  }
  return rows * columns;
}

/// Validates the contiguous encoded payload represented by QwenTensorRef.
/// Exact type and element count establish the payload size. QwenTensorRef does
/// not carry arbitrary tensor stride metadata, so strided views are outside
/// this contract.
[[nodiscard]] inline bool ValidateHrxTensorPayload(
    const models::QwenTensorRef& tensor, core::GgmlType expected_type,
    std::size_t expected_elements, std::string* error_msg = nullptr) noexcept {
  const auto reject = [error_msg](const char* message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };
  if (expected_elements == 0 || tensor.empty()) {
    return reject("HRX tensor binding requires a non-empty tensor contract");
  }
  if (tensor.type != expected_type) {
    return reject("GGUF tensor type does not match the HRX artifact ABI");
  }
  if (tensor.num_elements != expected_elements) {
    return reject(
        "GGUF tensor element count does not match the HRX artifact ABI");
  }
  if (!tensor.FitsAvailableStorage()) {
    return reject(
        "GGUF tensor payload is truncated or has an invalid encoded size");
  }
  if (error_msg != nullptr) {
    error_msg->clear();
  }
  return true;
}

}  // namespace gufo::hrx

#endif  // GUFO_MODELS_QWEN_HRX_QWEN_HRX_TENSOR_BINDING_HPP_
