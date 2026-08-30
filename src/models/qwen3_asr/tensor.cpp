#include "src/models/qwen3_asr/tensor.hpp"

#include <limits>
#include <utility>

namespace gufo::models::qwen3_asr {

std::size_t DTypeSize(DType dtype) noexcept {
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

std::optional<std::size_t> Tensor::NumElements() const noexcept {
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

const Tensor* TensorStore::Find(std::string_view name) const noexcept {
  const auto found = tensors_.find(std::string(name));
  return found == tensors_.end() ? nullptr : &found->second;
}

bool TensorStore::Add(Tensor tensor) {
  const std::string name = tensor.name;
  return tensors_.emplace(name, std::move(tensor)).second;
}

}  // namespace gufo::models::qwen3_asr
