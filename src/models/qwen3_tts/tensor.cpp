#include "src/models/qwen3_tts/tensor.hpp"

#include <algorithm>
#include <limits>
#include <optional>

namespace gufo::models::qwen3_tts {

std::size_t DTypeSize(DType dtype) {
  switch (dtype) {
    case DType::kF32:
    case DType::kI32:
      return 4;
    case DType::kBF16:
    case DType::kF16:
      return 2;
    case DType::kI64:
      return 8;
    case DType::kU8:
      return 1;
    case DType::kUnknown:
      return 0;
  }
  return 0;
}

std::optional<std::size_t> Tensor::NumElements() const {
  std::size_t elements = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension > std::numeric_limits<std::size_t>::max() ||
        (dimension != 0 &&
         elements > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(dimension))) {
      return std::nullopt;
    }
    elements *= static_cast<std::size_t>(dimension);
  }
  return elements;
}

const Tensor* TensorStore::Find(std::string_view name) const {
  for (const auto& tensor : tensors_) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

void TensorStore::Add(Tensor tensor) {
  tensors_.push_back(std::move(tensor));
}

}  // namespace gufo::models::qwen3_tts
