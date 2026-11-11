#ifndef GUFO_MODELS_QWEN_IMAGE_21_LORA_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_LORA_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <gufo/ext/experimental/lora_adapter.hpp>
#include <string_view>
#include <vector>

namespace gufo::models::qwen_image_21 {

using LoraDtype = lora::Dtype;
using LoraTensor = lora::Tensor;
using LoraPatch = lora::Patch;
using LoraTerm = lora::Term;

inline std::vector<float> LoraFloats(const LoraTensor& tensor,
                                     std::size_t first, std::size_t count) {
  return lora::Floats(tensor, first, count);
}

/// A validated Qwen-Image-2.1 transformer adapter file (LoRA, DoRA, diffs).
class LoraAdapter : public lora::Adapter {
public:
  /// Base tensor shape by Gufo name, or empty when the tensor does not exist.
  using BaseShape = std::function<std::vector<int>(std::string_view)>;
  LoraAdapter(const std::filesystem::path& path, const BaseShape& base);
};

}  // namespace gufo::models::qwen_image_21
#endif
