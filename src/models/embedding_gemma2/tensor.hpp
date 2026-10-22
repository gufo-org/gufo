#ifndef GUFO_MODELS_EMBEDDING_GEMMA2_TENSOR_HPP_
#define GUFO_MODELS_EMBEDDING_GEMMA2_TENSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::embedding_gemma2 {

enum class DType : std::uint8_t {
  kF32,
  kBF16,
  kF16,
  kUnknown,
};

[[nodiscard]] std::size_t DTypeSize(DType dtype) noexcept;

struct Tensor {
  std::string name;
  DType dtype{DType::kUnknown};
  std::vector<std::uint64_t> shape;
  const std::byte* data{nullptr};
  std::uint64_t byte_count{0};

  [[nodiscard]] std::optional<std::size_t> NumElements() const noexcept;
};

class TensorStore {
public:
  [[nodiscard]] const Tensor* Find(std::string_view name) const noexcept;
  [[nodiscard]] std::size_t size() const noexcept { return tensors_.size(); }
  [[nodiscard]] bool Add(Tensor tensor);

private:
  std::unordered_map<std::string, Tensor> tensors_;
};

}  // namespace gufo::models::embedding_gemma2

#endif  // GUFO_MODELS_EMBEDDING_GEMMA2_TENSOR_HPP_
