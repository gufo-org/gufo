#include "src/models/fish_drama3/tensor.hpp"

#include <limits>

namespace gufo::models::fish_drama3 {

DType ParseDType(const std::string& raw) {
  if (raw == "F32") {
    return DType::kF32;
  }
  if (raw == "BF16") {
    return DType::kBF16;
  }
  if (raw == "F16") {
    return DType::kF16;
  }
  if (raw == "F8_E4M3") {
    return DType::kF8_E4M3;
  }
  if (raw == "I64") {
    return DType::kI64;
  }
  if (raw == "I32") {
    return DType::kI32;
  }
  if (raw == "U8") {
    return DType::kU8;
  }
  return DType::kUnknown;
}

std::size_t DTypeElementSize(DType dtype) {
  switch (dtype) {
    case DType::kF32:
      return 4;
    case DType::kBF16:
    case DType::kF16:
      return 2;
    case DType::kF8_E4M3:
    case DType::kU8:
      return 1;
    case DType::kI64:
      return 8;
    case DType::kI32:
      return 4;
    case DType::kUnknown:
      return 0;
  }
  return 0;
}

std::size_t TensorView::NumElements() const {
  if (shape.empty()) {
    return 1;
  }
  std::size_t prod = 1;
  for (std::uint64_t dim : shape) {
    if (dim == 0) {
      return 0;
    }
    if (prod > std::numeric_limits<std::size_t>::max() /
                   static_cast<std::size_t>(dim)) {
      return 0;
    }
    prod *= static_cast<std::size_t>(dim);
  }
  return prod;
}

bool TensorView::IsContiguousValid() const {
  if (data == nullptr || dtype == DType::kUnknown) {
    return false;
  }
  const std::size_t elem_size = DTypeElementSize(dtype);
  if (elem_size == 0) {
    return false;
  }
  const std::size_t count = NumElements();
  if (count == 0 && !shape.empty()) {
    for (std::uint64_t dim : shape) {
      if (dim == 0) {
        return byte_size == 0;
      }
    }
    return false;
  }
  if (count > std::numeric_limits<std::size_t>::max() / elem_size) {
    return false;
  }
  return count * elem_size == byte_size;
}

}  // namespace gufo::models::fish_drama3
