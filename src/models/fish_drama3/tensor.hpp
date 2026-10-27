#ifndef GUFO_MODELS_FISH_DRAMA3_TENSOR_HPP_
#define GUFO_MODELS_FISH_DRAMA3_TENSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gufo::models::fish_drama3 {

enum class DType {
  kF32,
  kBF16,
  kF16,
  kF8_E4M3,
  kI64,
  kI32,
  kU8,
  kUnknown,
};

struct TensorView {
  std::string name;
  DType dtype = DType::kUnknown;
  std::vector<std::uint64_t> shape;
  const std::uint8_t* data = nullptr;
  std::size_t byte_size = 0;

  [[nodiscard]] std::size_t NumElements() const;
  [[nodiscard]] bool IsContiguousValid() const;
};

[[nodiscard]] DType ParseDType(const std::string& raw);
[[nodiscard]] std::size_t DTypeElementSize(DType dtype);

}  // namespace gufo::models::fish_drama3

#endif  // GUFO_MODELS_FISH_DRAMA3_TENSOR_HPP_
