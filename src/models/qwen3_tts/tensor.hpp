#ifndef GUFO_MODELS_QWEN3_TTS_TENSOR_HPP_
#define GUFO_MODELS_QWEN3_TTS_TENSOR_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen3_tts {

enum class DType : std::uint8_t {
  kF32,
  kBF16,
  kF16,
  kI64,
  kI32,
  kU8,
  kUnknown,
};

[[nodiscard]] std::size_t DTypeSize(DType dtype);

/// A host-resident tensor reference (weights stay mapped from safetensors).
struct Tensor {
  std::string name;
  DType dtype = DType::kUnknown;
  std::vector<std::uint64_t> shape;
  const std::byte* data = nullptr;  // payload pointer (start of raw bytes)
  std::uint64_t byte_count = 0;

  [[nodiscard]] std::optional<std::size_t> NumElements() const;
};

/// Everything loaded from a single model file (shard).
class TensorStore {
public:
  [[nodiscard]] const Tensor* Find(std::string_view name) const;
  [[nodiscard]] std::size_t size() const { return tensors_.size(); }
  void Add(Tensor tensor);
  void SetData(const std::string& name, const void* data, std::uint64_t bytes);
  void Clear() { tensors_.clear(); }

private:
  friend struct LoadResult;
  std::vector<Tensor> tensors_;
};

}  // namespace gufo::models::qwen3_tts

#endif  // GUFO_MODELS_QWEN3_TTS_TENSOR_HPP_
