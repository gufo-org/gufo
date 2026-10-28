#ifndef GUFO_MODELS_QWEN_IMAGE_21_LORA_HPP_
#define GUFO_MODELS_QWEN_IMAGE_21_LORA_HPP_

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::models::qwen_image_21 {

enum class LoraDtype { kBf16, kF16, kF32 };

/// Row-major view into a mapped adapter file.
struct LoraTensor {
  const void* data{nullptr};
  LoraDtype dtype{LoraDtype::kF32};
  std::vector<int> shape;
};

/// Elements [first, first + count) as FP32 (exact for all three dtypes).
std::vector<float> LoraFloats(const LoraTensor& tensor, std::size_t first,
                              std::size_t count);

/// One update of one base tensor (ComfyUI semantics, see lora.cpp).
struct LoraPatch {
  // Base tensor, e.g. "transformer.transformer_blocks.0.attn.to_q.weight".
  std::string target;
  int rows{0}, cols{0};  // target shape; 1-D targets have cols == 1
  int rank{0};           // 0 for a diff
  int first_row{0};      // first row used of up/diff/magnitude (fused gate_up)
  float scale{1};        // alpha / rank; strength is applied by the caller
  LoraTensor down, up, diff, magnitude;  // absent tensors have data == nullptr
};

struct LoraTerm {
  const LoraPatch* patch{nullptr};
  float strength{1};
};

/// A validated Qwen-Image-2.1 transformer adapter file (LoRA, DoRA, diffs).
class LoraAdapter {
public:
  /// Base tensor shape by Gufo name, or empty when the tensor does not exist.
  using BaseShape = std::function<std::vector<int>(std::string_view)>;
  LoraAdapter(const std::filesystem::path& path, const BaseShape& base);
  const std::vector<LoraPatch>& patches() const { return patches_; }
  /// Layout name with features, e.g. "kohya+alpha+gate_up".
  const std::string& format() const { return format_; }
  /// Smallest and largest pair rank; 0 if the file only has diffs.
  int min_rank() const;
  int max_rank() const;

private:
  std::shared_ptr<void> mapping_;
  std::vector<LoraPatch> patches_;
  std::string format_;
};

}  // namespace gufo::models::qwen_image_21
#endif
